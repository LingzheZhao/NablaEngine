"""Screen every declared reference state against the complete physical terrain model.

Run on explicitly provisioned compute. Each frame is an independent mj_forward
query at its recorded qpos/qvel, zero warm start and current reference joint
position targets. Contact forces describe those imposed states at 50 Hz. They
are not forces from a policy rollout. All frames, failures and extrema remain
in the output. Selected packets are inputs for subsequent native/MuJoCo checks.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True
import numpy as np

PROTOCOL = dict(
    format='nablaengine-terrain-reference-screen-v1', expected_motions=10, reference_hz=50.,
    timestep_s=.005, substeps_per_control=4, solver_iterations=10,
    state_queries='Independent recorded qpos/qvel; current-frame reference joint targets; zero warm start and applied forces',
    force_support='One forward-dynamics solution per reference frame, 50 Hz reference sampling; no integration',
    penetration='max(0,-minimum MuJoCo contact distance), including inactive contact candidates',
    selection=['initial', 'middle', 'last', 'max_contacts', 'max_contact_rows', 'max_rows',
               'max_penetration', 'max_body_force', 'first_nonfinite'],
    tie_rule='Earliest frame at the exact observed maximum, retaining every selection reason',
    future_controls=[1, 4],
    body_force='mj_rnePostConstraint cfrc_ext, world rotation:torque then force, about the root subtree COM; applied forces zero',
)


def select_frames(ncon, contact_rows, rows, penetration, body_force, finite):
    """Select declared states without discarding nonfinite or terminal frames."""
    count = len(ncon)
    chosen = {}

    def add(frame, reason):
        chosen.setdefault(int(frame), []).append(reason)

    add(0, 'initial')
    add((count-1)//2, 'middle')
    add(count-1, 'last')
    for reason, values in (('max_contacts', ncon), ('max_contact_rows', contact_rows), ('max_rows', rows),
                           ('max_penetration', penetration), ('max_body_force', body_force)):
        available = np.flatnonzero(np.isfinite(values))
        if len(available):
            add(available[np.argmax(np.asarray(values)[available])], reason)
    bad = np.flatnonzero(~np.asarray(finite))
    if len(bad):
        add(bad[0], 'first_nonfinite')
    return [dict(frame=frame, reasons=chosen[frame], available_future_controls=count-1-frame,
                 requested_future_controls=PROTOCOL['future_controls']) for frame in sorted(chosen)]


def load_inputs(scene, reference):
    """Validate declarations and the saved physical model at the owning boundary."""
    import mujoco
    from nablaengine.model import capability
    contract = json.loads((scene/'contract.json').read_text())
    screening = json.loads((reference/'screening.json').read_text())
    index = json.loads((reference/'index.json').read_text())
    if set(index['clips']) != {f'motion_{i:02d}' for i in range(PROTOCOL['expected_motions'])}:
        raise ValueError('The screen requires all ten declared reference motions')
    if screening['failed_conditions'] or screening['completed_motion_count'] != PROTOCOL['expected_motions']:
        raise ValueError('The selected motion producer did not complete all ten conditions')
    binding = screening['binding']
    if screening['target_artifact_id'] != contract['artifact_id'] or binding['scene_artifact_id'] != contract['artifact_id']:
        raise ValueError('Motion and physical scene artifact bindings differ')
    if binding['terrain_translations_m'] != contract['frame']['terrain_translations_m']:
        raise ValueError('Motion and physical tile translations differ')
    if binding['base_translation_m'] != contract['frame']['coacd_input_to_tile_translation_m']:
        raise ValueError('Motion and physical input-frame bindings differ')
    if index['fps'] != PROTOCOL['reference_hz']:
        raise ValueError('Expected the declared 50 Hz references')
    model = mujoco.MjModel.from_binary_path(str(scene/'model.mjb'))
    order = [model.joint(model.actuator_trnid[a, 0]).name for a in range(model.nu)]
    if order != contract['actuator_joint_names'] or order != binding['target_joint_order']:
        raise ValueError('Motion, saved contract and constructed actuator bindings differ')
    if model.body(model.jnt_bodyid[0]).name != 'torso_link' or contract['free_root_body'] != 'torso_link':
        raise ValueError('Expected the exact-source torso-root G1')
    if model.npair != contract['counts']['pairs'] or model.ngeom != contract['counts']['geoms']:
        raise ValueError('Constructed model counts differ from the selected contract')
    if not np.isclose(model.opt.timestep, PROTOCOL['timestep_s'], rtol=0, atol=1e-15) or model.opt.iterations != PROTOCOL['solver_iterations']:
        raise ValueError('Physical clock or solver iterations differ from the declared source condition')
    if not np.allclose(model.dof_damping[6:], .5, rtol=0, atol=1e-12) or not np.allclose(model.dof_frictionloss[6:], .1, rtol=0, atol=1e-12):
        raise ValueError('Expected source passive damping .5 and dry friction .1 on all hinges')
    report = capability.check(model, np.asarray(contract['foot_pairs']), scope='engine')
    if report.errors():
        raise capability.UnsupportedModel(report, 'reference-state screen')
    terrains = {row['terrain_index']: row for row in contract['terrains']}
    motions = {row['name']: row for row in screening['motions']}
    sources = {}
    for name, clip in index['clips'].items():
        declared = motions[name]
        terrain = terrains[clip['terrain_id']]
        if declared['terrain_id'] != clip['terrain_id'] or declared['terrain_file'] != terrain['terrain_file']:
            raise ValueError(f'{name} terrain identity differs between independent declarations')
        if clip['world_translation_m'] != terrain['input_to_world_translation_m']:
            raise ValueError(f'{name} world translation differs from its physical terrain')
        if clip['motion_file'] != declared['motion_file']:
            raise ValueError(f'{name} source-motion binding differs')
        with np.load(reference/f"{clip['shard']}.npz", allow_pickle=False) as data:
            if data['names'].tolist() != [name] or data['starts'].tolist() != [0] or data['lengths'].tolist() != [clip['length']]:
                raise ValueError(f'{name} must use the current single-motion shard schema')
            qpos, qvel, body_pos = data['qpos'], data['qvel'], data['body_pos']
        if qpos.shape != (clip['length'], model.nq) or qvel.shape != (clip['length'], model.nv) or not len(qpos):
            raise ValueError(f'{name} physical state dimensions differ from the model')
        if body_pos.shape != (clip['length'], len(index['bodies']), 3):
            raise ValueError(f'{name} recorded body-position dimensions differ from the declared body order')
        sources[name] = dict(qpos=qpos, qvel=qvel, body_pos=body_pos,
                             body_ids=np.asarray([model.body(body).id for body in index['bodies']]),
                             clip=clip, terrain=terrain)
    return model, contract, screening, index, sources, report.to_dict()


def screen_motion(model, source, output):
    import mujoco
    qpos, qvel = source['qpos'], source['qvel']
    count = len(qpos)
    columns = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    metrics = dict(contacts=np.full(count, np.nan), active_contacts=np.full(count, np.nan),
                   rows=np.full(count, np.nan), contact_rows=np.full(count, np.nan),
                   penetration_m=np.full(count, np.nan), max_body_force_N=np.full(count, np.nan),
                   fk_body_position_max_error_m=np.full(count, np.nan),
                   finite=np.zeros(count, dtype=bool), body_contact_counts=np.zeros((count, model.nbody), dtype=np.int32),
                   body_external_wrench_world=np.full((count, model.nbody, 6), np.nan))
    failures = []
    initial_reading = None
    data = mujoco.MjData(model)
    contact_types = [int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS),
                     int(mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL),
                     int(mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC)]
    geom_to_terrain = {model.geom(name).id: row['terrain_index'] for row in source['all_terrains']
                       for name in row['hull_geom_names']}
    with (output/'frames.jsonl').open('w') as frames:
        for frame in range(count):
            row = dict(frame=frame, reference_time_s=frame/PROTOCOL['reference_hz'],
                       declared_terrain_id=source['clip']['terrain_id'], state='failed')
            try:
                if not np.isfinite(qpos[frame]).all() or not np.isfinite(qvel[frame]).all():
                    raise ValueError('Reference state contains nonfinite values')
                mujoco.mj_resetData(model, data)
                data.qpos[:], data.qvel[:], data.ctrl[:] = qpos[frame], qvel[frame], qpos[frame, columns]
                data.qacc_warmstart[:] = 0
                mujoco.mj_forward(model, data)
                mujoco.mj_rnePostConstraint(model, data)
                contacts = []
                for index, contact in enumerate(data.contact):
                    first, second = int(contact.geom1), int(contact.geom2)
                    bodies = [int(model.geom_bodyid[first]), int(model.geom_bodyid[second])]
                    metrics['body_contact_counts'][frame, bodies] += 1
                    force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, index, force)
                    contacts.append(dict(geoms=[first, second], bodies=bodies,
                                         terrain_ids=[geom_to_terrain[g] for g in (first, second) if g in geom_to_terrain],
                                         distance_m=float(contact.dist), efc_address=int(contact.efc_address),
                                         dimension=int(contact.dim), contact_frame_wrench=force.tolist()))
                finite = all(np.isfinite(value).all() for value in (data.qacc, data.qfrc_constraint, data.efc_force,
                              data.cfrc_ext, np.asarray([c['distance_m'] for c in contacts]),
                              np.asarray([c['contact_frame_wrench'] for c in contacts])))
                penetration = max([max(0., -item['distance_m']) for item in contacts] + [0.])
                metrics['contacts'][frame] = data.ncon
                metrics['active_contacts'][frame] = sum(item['efc_address'] >= 0 for item in contacts)
                metrics['rows'][frame] = data.nefc
                metrics['contact_rows'][frame] = np.isin(data.efc_type, contact_types).sum()
                metrics['penetration_m'][frame] = penetration
                metrics['body_external_wrench_world'][frame] = data.cfrc_ext
                metrics['max_body_force_N'][frame] = np.linalg.norm(data.cfrc_ext[1:, 3:], axis=1).max()
                fk_error = float(np.linalg.norm(data.xpos[source['body_ids']]-source['body_pos'][frame], axis=1).max())
                finite = finite and bool(np.isfinite(fk_error))
                metrics['fk_body_position_max_error_m'][frame] = fk_error
                metrics['finite'][frame] = finite
                row.update(state='measured', finite=bool(finite), contacts=contacts,
                           ncon=int(data.ncon), nefc=int(data.nefc), contact_rows=int(metrics['contact_rows'][frame]),
                           friction_rows=int(data.nf), limit_rows=int(data.nl), penetration_m=penetration,
                           fk_body_position_max_error_m=fk_error,
                           body_contact_counts=metrics['body_contact_counts'][frame].tolist(),
                           body_external_wrench_world=data.cfrc_ext.tolist(),
                           solver_iterations=data.solver_niter.tolist(), warnings=data.warning.number.tolist())
            except Exception as error:
                row.update(failure_type=type(error).__name__, failure=str(error))
                failures.append(dict(frame=frame, failure_type=type(error).__name__, failure=str(error)))
            if frame == 0:
                initial_reading = row
            frames.write(json.dumps(row)+'\n')
            frames.flush()
    np.savez_compressed(output/'frame_metrics.npz', **metrics)
    selected = select_frames(metrics['contacts'], metrics['contact_rows'], metrics['rows'], metrics['penetration_m'],
                             metrics['max_body_force_N'], metrics['finite'])
    selected_frames = np.asarray([row['frame'] for row in selected], dtype=np.int32)
    np.savez_compressed(output/'selected_states.npz', frames=selected_frames, qpos=qpos[selected_frames], qvel=qvel[selected_frames],
                        ctrl=qpos[selected_frames][:, columns], qacc_warmstart=np.zeros((len(selected), model.nv)),
                        friction=np.ones(len(selected)), com=np.zeros((len(selected), 3)))
    result = dict(frames=count, measured_frames=count-len(failures), failures=failures,
                  nonfinite_frames=np.flatnonzero(~metrics['finite']).tolist(), selected=selected,
                  source_motion=source['clip'], initial_state=initial_reading,
                  maximum_contacts=float(np.nanmax(metrics['contacts'])) if np.isfinite(metrics['contacts']).any() else None,
                  maximum_rows=float(np.nanmax(metrics['rows'])) if np.isfinite(metrics['rows']).any() else None,
                  maximum_contact_rows=float(np.nanmax(metrics['contact_rows'])) if np.isfinite(metrics['contact_rows']).any() else None,
                  maximum_penetration_m=float(np.nanmax(metrics['penetration_m'])) if np.isfinite(metrics['penetration_m']).any() else None)
    (output/'result.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--code-revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.scene, args.reference, args.output = args.scene.resolve(), args.reference.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.chdir(args.output)
    report = dict(state='started', protocol=PROTOCOL, code_revision=args.code_revision,
                  scene=str(args.scene), reference=str(args.reference), motions={})

    def write_report():
        (args.output/'report.json').write_text(json.dumps(report, indent=2))

    def fail(kind, value, tb):
        report.update(state='failed', failure_type=kind.__name__, failure=str(value),
                      traceback=''.join(traceback.format_exception(kind, value, tb)))
        write_report()
        sys.__excepthook__(kind, value, tb)

    write_report()
    sys.excepthook = fail
    model, contract, screening, index, sources, validation = load_inputs(args.scene, args.reference)
    report.update(scene_artifact_id=contract['artifact_id'], reference_artifact_id=screening['artifact_id'],
                  capability=validation, body_names=[model.body(b).name for b in range(model.nbody)],
                  declared_physics=dict(damping=model.dof_damping.tolist(), frictionloss=model.dof_frictionloss.tolist(),
                                        timestep=model.opt.timestep, iterations=model.opt.iterations))
    for name, source in sources.items():
        output = args.output/name
        output.mkdir()
        source['all_terrains'] = contract['terrains']
        report['motions'][name] = screen_motion(model, source, output)
        write_report()
    report.update(state='completed', requested_frames=sum(row['frames'] for row in report['motions'].values()),
                  measured_frames=sum(row['measured_frames'] for row in report['motions'].values()),
                  failed_frames=sum(len(row['failures']) for row in report['motions'].values()),
                  nonfinite_frames=sum(len(row['nonfinite_frames']) for row in report['motions'].values()))
    write_report()


if __name__ == '__main__':
    main()

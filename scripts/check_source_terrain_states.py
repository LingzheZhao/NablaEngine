"""Compare every selected source-reference state with native and MuJoCo dynamics.

Execute only on provisioned CUDA compute. Forces describe imposed reference-state
starts followed by constant controls. They are not evidence of policy flight or
policy skill retention. The --self-test uses generated arrays and no simulator.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

PROTOCOL = dict(
    timestep_s=.005, solver_iterations=10, substeps=4, control_horizons=[1, 4],
    contact_capacity=128, constraint_row_capacity=576,
    control='Selected ctrl held constant for every substep and control',
    selected_states='Every selected_states.npz row, including last, nonfinite and fallen states',
    directions_per_block=3, direction_seed=20261106, blocks=['qpos_tangent', 'qvel', 'ctrl'],
    fd_steps=[1e-5, 5e-6], fd_stability_atol=.001, fd_stability_rtol=.01,
    gradient_atol=.01, gradient_rtol=.03, qpos_atol=.0005, qvel_atol=.05,
    root_position_atol_m=.0005, body_position_atol_m=.0005,
    backward_normal=3, backward_clamp_kv=0,
    scalar_loss='Fixed unit random terminal qpos and qvel cotangents; same cotangents for native and MuJoCo',
    fd_rule='Central float64 MuJoCo differences; unstable FD stays a failed direction',
    force_sampling_hz=200., force_sample='Final solved forces at each substep input before integration',
    force_scope='Reference-state constant-control diagnostic, no policy-flight inference',
    native_force_convention='World force then moment about world origin, per body; terrain and all-contact arrays saved',
)


def close(actual, expected, atol, rtol=0.):
    return bool(np.isfinite(actual).all() and np.isfinite(expected).all()
                and np.all(np.abs(actual-expected) <= atol+rtol*np.abs(expected)))


def probes(nq, nv, nu):
    rng = np.random.default_rng(PROTOCOL['direction_seed'])
    cq, cv = rng.normal(size=nq), rng.normal(size=nv)
    cq /= np.linalg.norm(cq)
    cv /= np.linalg.norm(cv)
    directions = []
    for block, width in zip(PROTOCOL['blocks'], [nv, nv, nu]):
        for index in range(PROTOCOL['directions_per_block']):
            value = rng.normal(size=width)
            directions.append(dict(block=block, index=index, direction=value/np.linalg.norm(value)))
    return cq, cv, directions


def selected_packets(path, result, source, model):
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in
                  ('frames', 'qpos', 'qvel', 'ctrl', 'qacc_warmstart', 'friction', 'com')}
    frames = arrays['frames']
    count = len(frames)
    if frames.tolist() != [row['frame'] for row in result['selected']]:
        raise ValueError('Selected packet order differs from the recorded selection')
    shapes = dict(frames=(count,), qpos=(count, model.nq), qvel=(count, model.nv),
                  ctrl=(count, model.nu), qacc_warmstart=(count, model.nv), friction=(count,), com=(count, 3))
    if any(arrays[key].shape != shape for key, shape in shapes.items()):
        raise ValueError('Selected packet dimensions differ from the physical model')
    if not np.issubdtype(frames.dtype, np.integer) or len(set(frames.tolist())) != count:
        raise ValueError('Selected frame IDs must be distinct integers')
    if np.any(frames < 0) or np.any(frames >= len(source['qpos'])):
        raise ValueError('Selected frame is outside reference support')
    columns = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    for key, expected in [('qpos', source['qpos'][frames]), ('qvel', source['qvel'][frames]),
                          ('ctrl', source['qpos'][frames][:, columns])]:
        if not np.array_equal(arrays[key], expected, equal_nan=True):
            raise ValueError(f'Selected {key} differs from the independently loaded reference')
    if np.any(arrays['qacc_warmstart']) or np.any(arrays['com']) or not np.all(arrays['friction'] == 1):
        raise ValueError('Expected the screen protocol zero warm start/COM and unit friction')
    return arrays


def self_test():
    from types import SimpleNamespace
    import tempfile
    nq, nv, nu = 10, 9, 3
    q = np.arange(50, dtype=float).reshape(5, nq)
    v = np.zeros((5, nv))
    q[2, 0] = np.nan
    frames = np.array([0, 2, 4])
    model = SimpleNamespace(nq=nq, nv=nv, nu=nu, jnt_qposadr=np.array([7,8,9]),
                            actuator_trnid=np.array([[0,0],[1,0],[2,0]]))
    result = dict(selected=[dict(frame=int(f), reasons=['last' if f == 4 else 'generated']) for f in frames])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'selected_states.npz'
        np.savez(path, frames=frames, qpos=q[frames], qvel=v[frames], ctrl=q[frames,7:],
                 qacc_warmstart=v[frames], friction=np.ones(3), com=np.zeros((3,3)))
        packets = selected_packets(path, result, dict(qpos=q, qvel=v), model)
        assert packets['frames'][-1] == 4 and np.isnan(packets['qpos'][1,0])
    cq, cv, directions = probes(nq, nv, nu)
    assert len(directions) == 9 and all(np.isclose(np.linalg.norm(row['direction']), 1) for row in directions)
    assert np.isclose(np.linalg.norm(cq), 1) and np.isclose(np.linalg.norm(cv), 1)
    assert not close(np.array([1., 2.]), np.array(1.), .001, .01)
    print(json.dumps(dict(self_test='passed', selected_last_retained=True, directions=len(directions))))


def run(args, report, write_report):
    import mujoco
    import torch
    from nablaengine import torch_ffi as F
    from nablaengine.g1_engine import tape, step
    from screen_terrain_references import load_inputs
    from check_box_step_gradients import perturb

    model, contract, screening, index, sources, validation = load_inputs(args.scene, args.reference)
    screen = json.loads((args.screen/'report.json').read_text())
    if screen['state'] != 'completed' or screen['scene_artifact_id'] != contract['artifact_id'] or screen['reference_artifact_id'] != screening['artifact_id']:
        raise ValueError('Selection screen does not bind the selected scene and references')
    if set(screen['motions']) != set(sources):
        raise ValueError('Selection screen does not contain every declared motion')
    if (model.opt.timestep != PROTOCOL['timestep_s'] or model.opt.iterations != PROTOCOL['solver_iterations']
            or index['fps']*PROTOCOL['substeps']*model.opt.timestep != 1.):
        raise ValueError('Constructed physical clock or solver differs from predeclared protocol')
    if step.maxcon(model) != PROTOCOL['contact_capacity'] or step.NEFC != PROTOCOL['constraint_row_capacity']:
        raise ValueError('Native capacities differ from the predeclared 128 contacts and 576 rows')
    packets = {name: selected_packets(args.screen/name/'selected_states.npz', screen['motions'][name], source, model)
               for name, source in sources.items()}
    report.update(scene_artifact_id=contract['artifact_id'], reference_artifact_id=screening['artifact_id'],
                  capability=validation, selection_screen=str(args.screen),
                  body_names=[model.body(b).name for b in range(model.nbody)],
                  physical_timestep_s=float(model.opt.timestep), physical_iterations=int(model.opt.iterations))
    os.environ['DK_CACHE_DIR'] = str(args.output/'build')
    os.environ.pop('DK_TORCH_ARTIFACT', None)
    cc = torch.cuda.get_device_capability()
    artifact = F.build(f'sm_{cc[0]}{cc[1]}', model=model, foot_pairs=contract['foot_pairs'], set_current=False)
    F.init(artifact=artifact)
    report.update(artifact=str(artifact), environment=dict(torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(), mujoco=mujoco.__version__, numpy=np.__version__))
    cq, cv, directions = probes(model.nq, model.nv, model.nu)
    report.update(terminal_cotangent=dict(qpos=cq.tolist(), qvel=cv.tolist()),
                  directions=[dict(row, direction=row['direction'].tolist()) for row in directions])
    bad_mask = tape.FLAG_UNSUPPORTED | tape.FLAG_NONCONV | tape.FLAG_SINGULAR | tape.FLAG_VERIFY_DIFF
    def tensor(values):
        return torch.tensor(values, dtype=torch.float32, device='cuda').contiguous()

    def physical(initial, controls, measured=False):
        data = mujoco.MjData(model)
        data.qpos[:], data.qvel[:], data.ctrl[:] = initial['q'], initial['v'], initial['u']
        data.qacc_warmstart[:] = initial['warm']
        readings = []
        for substep in range(controls*PROTOCOL['substeps']):
            mujoco.mj_step(model, data)
            if measured:
                mujoco.mj_rnePostConstraint(model, data)
                contacts = []
                for contact_index, contact in enumerate(data.contact):
                    force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, contact_index, force)
                    contacts.append(dict(geoms=[int(contact.geom1), int(contact.geom2)],
                                         distance_m=float(contact.dist), efc_address=int(contact.efc_address),
                                         dimension=int(contact.dim), contact_frame_wrench=force.tolist()))
                readings.append(dict(substep=substep, time_s=substep*PROTOCOL['timestep_s'], ncon=int(data.ncon),
                                     nefc=int(data.nefc), friction_rows=int(data.nf), limit_rows=int(data.nl),
                                     contact_rows=int(np.isin(data.efc_type, [int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS),
                                         int(mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL),
                                         int(mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC)]).sum()),
                                     contacts=contacts, body_external_wrench=data.cfrc_ext.tolist(),
                                     body_wrench_convention='World torque:force about root subtree COM',
                                     solver_iterations=data.solver_niter.tolist(), warnings=data.warning.number.tolist()))
        result = np.r_[data.qpos.copy(), data.qvel.copy()]
        mujoco.mj_kinematics(model, data)
        return result, data.xpos.copy(), readings

    def native(initial, controls):
        qt, vt, warm = tensor(initial['q'][None]), tensor(initial['v'][None]), tensor(initial['warm'][None])
        ctrl, friction, com = tensor(initial['u'][None]), tensor([1.]), tensor([[0.,0.,0.]])
        F.read_diagnostics(reset=True)
        saved, measured, decoded, states = [], [], [], []
        for _ in range(controls):
            output = F.forward_measurements(qt, vt, warm, ctrl, friction, com,
                                           iterations=PROTOCOL['solver_iterations'], substeps=PROTOCOL['substeps'], tape=True)
            qt, vt, warm = output['qpos'], output['qvel'], output['warm']
            saved.append(output['tape'])
            measured.append({key: value.detach().cpu().numpy() for key, value in output.items()})
            decoded.append(tape.decode(measured[-1]['tape'].reshape(1, PROTOCOL['substeps'], tape.rec_words(model)), model))
            states.append(np.r_[qt.detach().cpu().numpy()[0], vt.detach().cpu().numpy()[0]])
        return dict(q=qt, v=vt, tapes=saved, ctrl=ctrl, friction=friction, com=com,
                    measured=measured, decoded=decoded, states=states, diagnostics=F.read_diagnostics(reset=True))

    for name, source in sources.items():
        packet = packets[name]
        for position, selected in enumerate(screen['motions'][name]['selected']):
            frame = int(packet['frames'][position])
            initial = dict(q=np.array(packet['qpos'][position], dtype=np.float64),
                           v=np.array(packet['qvel'][position], dtype=np.float64),
                           u=np.array(packet['ctrl'][position], dtype=np.float64),
                           warm=np.array(packet['qacc_warmstart'][position], dtype=np.float64))
            for controls in PROTOCOL['control_horizons']:
                row = dict(motion=name, terrain_id=source['clip']['terrain_id'], frame=frame,
                           selection=selected, requested_controls=controls, requested_seconds=controls*.02,
                           requested_gradient_directions=len(directions), pass_=False)
                report['rows'].append(row)
                directory = args.output/f'{name}_frame_{frame}_controls_{controls}'
                directory.mkdir()
                try:
                    nominal = native(initial, controls)
                    expected, expected_bodies, force_readings = physical(initial, controls, measured=True)
                    actual = np.r_[nominal['q'].cpu().numpy()[0], nominal['v'].cpu().numpy()[0]]
                    _, bodies, _, _, _ = F.kinematics(nominal['q'], nominal['v'], nominal['com'])
                    body_error = np.linalg.norm(bodies.cpu().numpy()[0, source['body_ids']]-expected_bodies[source['body_ids']], axis=-1)
                    root_error = float(np.linalg.norm(actual[:3]-expected[:3]))
                    row.update(executed_controls=len(nominal['states']), diagnostics=nominal['diagnostics'], root_position_error_m=root_error,
                               body_position_max_error_m=float(body_error.max()), body_position_mean_error_m=float(body_error.mean()),
                               qpos_max_error=float(abs(actual[:model.nq]-expected[:model.nq]).max()),
                               qvel_max_error=float(abs(actual[model.nq:]-expected[model.nq:]).max()),
                               native_substeps=[{key: decoded[key].tolist() for key in ('ncon','nefc','flags','pair','slot')} for decoded in nominal['decoded']],
                               mujoco_substep_forces=force_readings,
                               native_contact_rows=[(decoded['nefc']-decoded['nlim']-int((model.dof_frictionloss>0).sum())).tolist()
                                                    for decoded in nominal['decoded']],
                               native_force_summary=[dict(
                                   maximum_body_terrain_force_N=float(np.linalg.norm(step['terrain_wrench'][0,:,1:,:3],axis=-1).max()),
                                   maximum_body_all_contact_force_N=float(np.linalg.norm(step['all_contact_wrench'][0,:,1:,:3],axis=-1).max()),
                                   body_terrain_contact_count=step['terrain_contact_count'].tolist()) for step in nominal['measured']])
                    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all() and all(np.isfinite(step[key]).all() for step in nominal['measured']
                                  for key in ('terrain_wrench','all_contact_wrench','actuator_torque')))
                    overflow_ok = not any(nominal['diagnostics'][key] for key in
                        ('fwd_contact_overflow_worlds', 'fwd_row_overflow_worlds'))
                    flags_ok = overflow_ok and all(not np.any(decoded['flags'] & bad_mask) for decoded in nominal['decoded'])
                    unsupported_ok = all(not np.any(step['info'][:,3]) for step in nominal['measured'])
                    row.update(finite=finite, native_flags_ok=bool(flags_ok and unsupported_ok),
                               physical_reference_ok=bool(close(actual[:model.nq],expected[:model.nq],PROTOCOL['qpos_atol'])
                                 and close(actual[model.nq:],expected[model.nq:],PROTOCOL['qvel_atol'])
                                 and root_error<=PROTOCOL['root_position_atol_m']
                                 and body_error.max()<=PROTOCOL['body_position_atol_m']))
                    np.savez_compressed(directory/'execution.npz', **{f'input_{k}':v for k,v in initial.items()},
                                        mujoco_terminal_state=expected, native_states=np.array(nominal['states']),
                                        **{f'control_{i}_{key}':value for i,step in enumerate(nominal['measured']) for key,value in step.items()})
                    row['evidence']=str(directory/'execution.npz')
                    row['reference_endpoint_available'] = frame+controls < len(source['qpos'])
                    if row['reference_endpoint_available']:
                        target = source['body_pos'][frame+controls]
                        row['native_reference_body_position_mean_error_m'] = float(np.linalg.norm(bodies.cpu().numpy()[0,source['body_ids']]-target, axis=-1).mean())
                    if not finite or not flags_ok or not unsupported_ok:
                        row['failure']='Native nonfinite state, measurement or strict flags'
                        continue
                    gq, gv, gu = tensor(cq[None]), tensor(cv[None]), torch.zeros_like(nominal['ctrl'])
                    backward_flags=[]
                    for saved in reversed(nominal['tapes']):
                        gq,gv,control_gradient,flags = F.backward(saved,nominal['friction'],nominal['com'],gq.contiguous(),gv.contiguous(),
                            substeps=PROTOCOL['substeps'],clamp_kv=PROTOCOL['backward_clamp_kv'],normal=PROTOCOL['backward_normal'])
                        gu += control_gradient
                        backward_flags.extend(flags.cpu().numpy().reshape(-1).tolist())
                    row['backward_diagnostics'] = F.read_diagnostics(reset=True)
                    gradient=np.r_[gq.cpu().numpy()[0],gv.cpu().numpy()[0],gu.cpu().numpy()[0]]
                    row.update(backward_flags=backward_flags, backward_ok=bool(np.isfinite(gradient).all() and
                        all(not (flag & (bad_mask | (3<<12) | (15<<16))) for flag in backward_flags)), directions=[])
                    for direction in directions:
                        fds=[]
                        for epsilon in PROTOCOL['fd_steps']:
                            losses=[]
                            for sign in (-1,1):
                                q,v,u=perturb(model,initial,direction,sign*epsilon)
                                losses.append(physical(dict(initial,q=q,v=v,u=u),controls)[0]@np.r_[cq,cv])
                            fds.append((losses[1]-losses[0])/(2*epsilon))
                        if direction['block']=='qpos_tangent':
                            qp,_,_=perturb(model,initial,direction,1e-6)
                            qm,_,_=perturb(model,initial,direction,-1e-6)
                            analytic=float(gq.cpu().numpy()[0]@((qp-qm)/2e-6))
                        else:
                            block=gv if direction['block']=='qvel' else gu
                            analytic=float(block.cpu().numpy()[0]@direction['direction'])
                        stable=close(np.array(fds),np.array(fds[-1]),PROTOCOL['fd_stability_atol'],PROTOCOL['fd_stability_rtol'])
                        accurate=close(np.array(analytic),np.array(fds[-1]),PROTOCOL['gradient_atol'],PROTOCOL['gradient_rtol'])
                        row['directions'].append(dict(block=direction['block'],index=direction['index'],analytic=analytic,
                            mujoco_float64_fd=fds,fd_stable=stable,gradient_ok=accurate,pass_=stable and accurate))
                    np.savez_compressed(directory/'gradients.npz', gradient=gradient)
                    row['pass_']=bool(row['physical_reference_ok'] and row['backward_ok'] and all(d['pass_'] for d in row['directions']))
                except Exception as error:
                    row.update(failure_type=type(error).__name__, failure=str(error), traceback=traceback.format_exc())
                finally:
                    write_report()
    report.update(state='completed', requested_conditions=sum(len(p['frames']) for p in packets.values())*2,
                  passed_conditions=sum(bool(row['pass_']) for row in report['rows']),
                  failed_conditions=sum(not bool(row['pass_']) for row in report['rows']),
                  measured_gradient_directions=sum(len(row.get('directions', [])) for row in report['rows']),
                  unstable_fd_directions=sum(not d['fd_stable'] for row in report['rows'] for d in row.get('directions',[])))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('scene','reference','screen','output'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--code-revision')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test()
        return
    if any(getattr(args,name) is None for name in ('scene','reference','screen','output','code_revision')):
        parser.error('--scene, --reference, --screen, --output and --code-revision are required')
    args.scene=args.scene.resolve()
    args.reference=args.reference.resolve()
    args.screen=args.screen.resolve()
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(state='started',protocol=PROTOCOL,code_revision=args.code_revision,rows=[],
                scene=str(args.scene),reference=str(args.reference))
    def write_report():
        (args.output/'report.json').write_text(json.dumps(report,indent=2))
    write_report()
    try:
        run(args,report,write_report)
    except Exception as error:
        report.update(state='failed',failure_type=type(error).__name__,failure=str(error),traceback=traceback.format_exc())
        raise
    finally:
        write_report()


if __name__=='__main__':
    main()

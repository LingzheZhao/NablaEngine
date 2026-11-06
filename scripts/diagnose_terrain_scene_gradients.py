"""Separate native engine self-gradient readings from physical-oracle comparisons.

Runs the same selected reference states, targets, output cotangent and three input
directions as check_terrain_scene. Centered FD queries call native F.forward.
Analytic gradients call native F.backward from the nominal GPU tape. MuJoCo is
used for coordinate integration of qpos perturbations and model construction.

Record every perturbed state, actual rounded GPU input, contact/zone/clamp support
and strict flag. A strict-invalid batch stops at that control and retains its
prefix. Report FD stability before interpreting derivative agreement. Previous
MuJoCo rollout comparisons remain separate evidence at their original revisions.

Run only on provisioned compute. --out requires a fresh external directory.

"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

import check_terrain_scene as scene_check
import check_box_step_gradients as box_check

PROTOCOL = dict(scope='native F.backward versus centered native F.forward differences',
                horizons=[1, 4], substeps=4, iterations=50, clamp_kv=1, normal=0,
                fd_steps=[.002, .001, .0005], gradient_atol=.01, gradient_rtol=.03,
                fd_stability_atol=.003, fd_stability_rtol=.02,
                body_time_support='requested horizon or explicitly retained strict-invalid prefix')


def run(args, context, report, out):
    import mujoco
    import torch
    from nablaengine import torch_ffi as F
    from nablaengine.g1_engine import tape
    model, fp = context['model'], context['foot_pairs']
    qpos, qvel, frames = context['qpos'], context['qvel'], context['frames']
    model.opt.iterations = PROTOCOL['iterations']
    os.environ['DK_CACHE_DIR'] = str(out/'build')
    os.environ.pop('DK_TORCH_ARTIFACT', None)
    cc = torch.cuda.get_device_capability()
    arch = f'sm_{cc[0]}{cc[1]}'
    artifact = F.build(arch, model=model, foot_pairs=fp, set_current=False)
    F.init(artifact=artifact)
    report.update(artifact=str(artifact), self_fd_protocol=PROTOCOL,
                  environment=dict(torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__,
                                   mujoco=mujoco.__version__, gpu=torch.cuda.get_device_name()), readings=[])
    cq, cv, directions = scene_check.gradient_probes(model)
    report['terminal_cotangent'] = dict(qpos=cq.tolist(), qvel=cv.tolist())
    report['directions'] = [dict(d, direction=d['direction'].tolist()) for d in directions]
    joint_columns = [int(model.jnt_qposadr[int(j)]) for j in model.actuator_trnid[:, 0]]
    bad_mask = 31 | tape.FLAG_NONCONV | tape.FLAG_SINGULAR | tape.FLAG_VERIFY_DIFF
    def tensor(value):
        return torch.tensor(value, dtype=torch.float32, device='cuda').contiguous()
    def forward(frame, q, v, residual, controls):
        q, v = tensor(q), tensor(v)
        warm = torch.zeros_like(v)
        friction, com = torch.ones(len(q), device='cuda'), torch.zeros((len(q), 3), device='cuda')
        tapes, decoded, signature, actual_targets = [], [], [], []
        valid = True
        for control in range(controls):
            target = np.asarray(qpos[frame+control, joint_columns][None]+residual, np.float32)
            actual_targets.append(target)
            result = F.forward(q, v, warm, tensor(target), friction, com, iterations=50, substeps=4, tape=True)
            q, v, warm, info = result[:4]
            tapes.append(result[4])
            record = tape.decode(result[4].cpu().numpy().reshape(len(q), 4, tape.rec_words(model)), model)
            decoded.append(record)
            active_zones = record['zone'] & (np.arange(record['zone'].shape[-1]) < record['nefc'][..., None])
            signature.append(np.concatenate((record['ncon'][..., None], record['nefc'][..., None],
                                              record['nlim'][..., None], record['pair'], record['slot'],
                                              record['clamp_dof'], active_zones), axis=-1))
            valid = (all(bool(torch.isfinite(value).all()) for value in (q, v, warm))
                     and not np.any(info.cpu().numpy()[:, 3]) and not np.any(record['flags'] & bad_mask))
            if not valid:
                break
        return dict(q=q, v=v, warm=warm, friction=friction, com=com, tapes=tapes, decoded=decoded,
                    signature=signature, targets=np.stack(actual_targets), valid=valid and len(tapes) == controls)
    for frame in map(int, frames):
        for controls in PROTOCOL['horizons']:
            row = dict(frame=frame, controls=controls, requested_seconds=controls*4*float(model.opt.timestep),
                       completed=False, directions=[])
            report['readings'].append(row)
            if frame+controls >= len(qpos):
                row['failure'] = 'reference support does not cover horizon'
                continue
            initial = dict(q=qpos[frame].copy(), v=qvel[frame].copy(), u=np.zeros(model.nu))
            nominal = forward(frame, initial['q'][None], initial['v'][None], initial['u'][None], controls)
            row['nominal_executed_controls'] = len(nominal['tapes'])
            row['nominal_flags'] = [r['flags'].tolist() for r in nominal['decoded']]
            nominal_path = out/f'frame_{frame}_{controls}_nominal.npz'
            np.savez_compressed(nominal_path, tapes=np.stack([t.cpu().numpy() for t in nominal['tapes']]),
                                input_qpos=initial['q'], input_qvel=initial['v'], targets=nominal['targets'],
                                qpos=nominal['q'].cpu().numpy(), qvel=nominal['v'].cpu().numpy())
            row['nominal_evidence'] = str(nominal_path)
            if not nominal['valid']:
                row['failure'] = 'strict-invalid nominal trajectory'
                continue
            gq, gv = tensor(cq[None]), tensor(cv[None])
            control_gradients, backward_flags = [], []
            for saved in reversed(nominal['tapes']):
                gq, gv, gu, flags = F.backward(saved, nominal['friction'], nominal['com'], gq.contiguous(), gv.contiguous(),
                                               substeps=4, clamp_kv=1, normal=0)
                control_gradients.append(gu.cpu().numpy()[0])
                backward_flags.extend(flags.cpu().numpy().tolist())
            control_gradients = np.stack(control_gradients[::-1])
            gq, gv = gq.cpu().numpy()[0], gv.cpu().numpy()[0]
            row['backward_flags'] = backward_flags
            backward_ok = (np.isfinite(gq).all() and np.isfinite(gv).all() and np.isfinite(control_gradients).all()
                           and all(not (flag & (bad_mask | (3 << 12) | (15 << 16))) for flag in backward_flags))
            row['backward_finite_supported'] = bool(backward_ok)
            requests = [(i, step, sign) for i in range(len(directions)) for step in PROTOCOL['fd_steps'] for sign in (-1, 1)]
            perturbed = [box_check.perturb(model, initial, directions[i], step*sign) for i, step, sign in requests]
            input_q = np.stack([x[0] for x in perturbed]).astype(np.float32)
            input_v = np.stack([x[1] for x in perturbed]).astype(np.float32)
            residuals = np.stack([x[2] for x in perturbed])
            trial = forward(frame, input_q, input_v, residuals, controls)
            row.update(perturbed_executed_controls=len(trial['tapes']),
                       perturbed_flags=[r['flags'].tolist() for r in trial['decoded']])
            trial_path = out/f'frame_{frame}_{controls}_perturbed.npz'
            np.savez_compressed(trial_path, input_qpos=input_q, input_qvel=input_v, residuals=residuals,
                                targets=trial['targets'], tapes=np.stack([t.cpu().numpy() for t in trial['tapes']]),
                                qpos=trial['q'].cpu().numpy(), qvel=trial['v'].cpu().numpy(),
                                grad_qpos=gq, grad_qvel=gv, grad_controls=control_gradients)
            row['perturbed_evidence'] = str(trial_path)
            if not trial['valid']:
                row['failure'] = 'strict-invalid perturbed batch, full-horizon derivative unavailable'
                continue
            scalar = (trial['q'].cpu().numpy().astype(float)@cq+trial['v'].cpu().numpy().astype(float)@cv)
            for index, direction in enumerate(directions):
                indices = np.arange(index*6, (index+1)*6)
                values = scalar[indices]
                fd = np.array([(values[j+1]-values[j])/(2*step)
                               for j, step in zip((0, 2, 4), PROTOCOL['fd_steps'])])
                if direction['block'] == 'qpos_tangent':
                    qp, _, _ = box_check.perturb(model, initial, direction, 1e-6)
                    qm, _, _ = box_check.perturb(model, initial, direction, -1e-6)
                    intended = float(gq@((qp-qm)/(2e-6)))
                elif direction['block'] == 'qvel':
                    intended = float(gv@direction['direction'])
                else:
                    intended = float(control_gradients.sum(axis=0)@direction['direction'])
                realized = []
                for step_index, step in enumerate(PROTOCOL['fd_steps']):
                    lo, hi = indices[2*step_index:2*step_index+2]
                    dq = (input_q[hi].astype(float)-input_q[lo].astype(float))/(2*step)
                    dv = (input_v[hi].astype(float)-input_v[lo].astype(float))/(2*step)
                    du = (trial['targets'][:, hi].astype(float)-trial['targets'][:, lo].astype(float))/(2*step)
                    realized.append(float(gq@dq+gv@dv+np.sum(control_gradients*du)))
                topology_ok = all(np.array_equal(s[indices], np.broadcast_to(base, s[indices].shape))
                                  for s, base in zip(trial['signature'], nominal['signature']))
                stable = box_check.close(fd, fd[-1], PROTOCOL['fd_stability_atol'], PROTOCOL['fd_stability_rtol'])
                realized_ok = box_check.close(np.asarray(realized), fd, PROTOCOL['gradient_atol'], PROTOCOL['gradient_rtol'])
                reading = dict(block=direction['block'], analytic_intended_direction=intended,
                               analytic_realized_float32_secants=realized, engine_self_finite_differences=fd.tolist(),
                               recorded_topology_equal=topology_ok, fd_stable=stable,
                               analytic_vs_self_fd=realized_ok,
                               absolute_errors=np.abs(np.asarray(realized)-fd).tolist())
                row['directions'].append(reading)
            row['completed'] = True
    report['diagnostics'] = F.read_diagnostics(reset=True)
    report['diagnostic_completed'] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--motion', default='motion_07')
    parser.add_argument('--frames', default='0,190,235,320')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--code-revision', required=True)
    args = parser.parse_args()
    if args.frames == 'all':
        parser.error('self-FD requires explicit selected frames')
    out = args.out.resolve()
    if out.is_relative_to(scene_check.ROOT):
        parser.error('--out must be outside checkout')
    out.mkdir(parents=True, exist_ok=False)
    report = dict(code_revision=args.code_revision, diagnostic_protocol=PROTOCOL,
                  diagnostic_completed=False)
    try:
        context = scene_check.screen(args, report, out)
        run(args, context, report, out)
        return 0
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    raise SystemExit(main())

"""Check opt-in native solver measurements on provisioned CUDA compute.

Generated G1 starts include foot contact, no terrain contact, and saturated
actuators. Forces come from Nabla's solved rows. MuJoCo supplies independent
mass/bias/Jacobian algebra at the recorded inputs, with no reference contact
solver or integration used for the force balance check. All files go into a
fresh external output root.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from check_box_step_gradients import states

PROTOCOL = dict(substeps=4, iterations=50, contact_penetration_m=.002,
                force_balance_atol_N=.02, force_balance_rtol=.0002,
                wrench_balance_atol=.0001, actuator_atol_Nm=.0003, actuator_rtol=.00002,
                sampling='one final-solver force at each substep input t_k',
                actuator_order='generalized actuated dofs 6..nv-1',
                reference='MuJoCo mass, bias and wrench-Jacobian algebra only')


def close(a, b, atol, rtol=0):
    return bool(np.isfinite(a).all() and np.isfinite(b).all()
                and np.all(np.abs(a-b) <= atol+rtol*np.abs(b)))


def run(out, report):
    import mujoco
    import torch
    from nablaengine import fixtures, torch_ffi as F
    from nablaengine.g1_engine import tape, source
    m, metadata = fixtures.load_model(check='engine')
    fp = metadata['foot_pairs']
    state = next(s for s in states(m, fp) if s['name'] == 'qpos0_asymmetric_contact')
    q = np.repeat(state['q'][None], 3, axis=0).astype(np.float32)
    v = np.zeros((3, m.nv), np.float32)
    u = np.repeat(state['u'][None], 3, axis=0).astype(np.float32)
    q[1, 2] += 2.
    u[2] += np.where(np.arange(m.nu) % 2, -100., 100.)
    os.environ['DK_CACHE_DIR'] = str(out/'build')
    os.environ.pop('DK_TORCH_ARTIFACT', None)
    cc = torch.cuda.get_device_capability()
    artifact = F.build(f'sm_{cc[0]}{cc[1]}', model=m, foot_pairs=fp, set_current=False)
    info = F.init(artifact=artifact)
    def tensor(x):
        return torch.tensor(x, dtype=torch.float32, device='cuda').contiguous()
    inputs = [tensor(q), tensor(v), tensor(v), tensor(u), torch.ones(3, device='cuda'),
              torch.zeros((3, 3), device='cuda')]
    options = dict(iterations=PROTOCOL['iterations'], substeps=PROTOCOL['substeps'], tape=True)
    ordinary = F.forward(*inputs, **options)
    measured = F.forward_measurements(*inputs, **options)
    report.update(artifact=str(artifact), environment=dict(torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(), mujoco=mujoco.__version__, numpy=np.__version__),
                  timestep_s=float(m.opt.timestep), sampling_hz=1./float(m.opt.timestep),
                  body_names=[m.body(i).name for i in range(m.nbody)],
                  dof_joint_names=[m.joint(m.dof_jntid[i]).name for i in range(6, m.nv)])
    # Tape contains integer fields transported as float32, including NaN bit
    # patterns for absent contacts. Compare storage bits, including signed zero.
    equal = {key: torch.equal(ordinary[i].view(torch.int32), measured[key].view(torch.int32))
             for i, key in enumerate(('qpos', 'qvel', 'warm', 'info', 'tape'))}
    different_words = {key: int((ordinary[i].view(torch.int32) != measured[key].view(torch.int32)).sum())
                       for i, key in enumerate(('qpos', 'qvel', 'warm', 'info', 'tape'))}
    arrays = {key: value.detach().cpu().numpy() for key, value in measured.items()}
    np.savez_compressed(out/'measurements.npz', input_qpos=q, input_qvel=v, input_ctrl=u,
                        ordinary_tape=ordinary[4].detach().cpu().numpy(), **arrays)
    decoded = tape.decode(arrays['tape'].reshape(3, 4, tape.rec_words(m)), m)
    wrench = arrays['terrain_wrench'].astype(float)
    count = arrays['terrain_contact_count']
    actuator = arrays['actuator_torque'].astype(float)
    data = mujoco.MjData(m)
    matrix = np.empty((m.nv, m.nv))
    bias = np.empty(m.nv)
    rows = []
    constants, _ = source.constants(m)
    # source.constants is transport storage. Read the same packed scalar bindings
    # using its declared layout, with no second model adaptation.
    offset = 0
    packed = {}
    for symbol, kind, count_key, width in source.CONST_LAYOUT:
        count_value = {'nb': m.nbody, 'nv': m.nv, 'one': 1}.get(count_key, count_key)
        size = int(count_value)*width
        packed[symbol] = constants[offset:offset+size]
        offset += size
    for world in range(3):
        for substep in range(4):
            data.qpos[:] = decoded['qpos'][world, substep]
            data.qvel[:] = decoded['qvel'][world, substep]
            mujoco.mj_kinematics(m, data)
            mujoco.mj_comPos(m, data)
            mujoco.mj_crb(m, data)
            mujoco.mj_comVel(m, data)
            mujoco.mj_rne(m, data, 0, bias)
            mujoco.mj_fullM(m, data, matrix)
            generalized = np.zeros(m.nv)
            for body in range(1, m.nbody):
                force = np.ascontiguousarray(wrench[world, substep, body, :3])
                moment = np.ascontiguousarray(wrench[world, substep, body, 3:])
                mujoco.mj_applyFT(m, data, force, moment, np.zeros(3), body, generalized)
            # Root equations have no joint-limit/frictionloss contribution.
            expected = matrix@decoded['qacc'][world, substep]+bias+m.dof_damping*data.qvel
            balance_ok = close(generalized[:6], expected[:6], PROTOCOL['force_balance_atol_N'],
                               PROTOCOL['force_balance_rtol'])
            kp, kv, limit = (packed[key][6:].astype(float) for key in ('c_kp', 'c_kv', 'c_frclim'))
            expected_actuator = np.clip(kp*(u[world]-data.qpos[7:])-kv*data.qvel[6:], -limit, limit)
            torque_ok = close(actuator[world, substep], expected_actuator,
                              PROTOCOL['actuator_atol_Nm'], PROTOCOL['actuator_rtol'])
            rows.append(dict(world=world, substep=substep, sample_time_s=substep*float(m.opt.timestep),
                             terrain_contact_count=int(count[world, substep, 1:].sum()),
                             force_world_N=wrench[world, substep, 1:, :3].sum(axis=0).tolist(),
                             root_generalized_force=generalized[:6].tolist(),
                             root_dynamics_force=expected[:6].tolist(),
                             root_balance_max_error=float(np.max(abs(generalized[:6]-expected[:6]))),
                             root_balance_ok=balance_ok, actuator_ok=torque_ok))
    reaction_ok = close(wrench.sum(axis=2), np.zeros((3, 4, 6)), PROTOCOL['wrench_balance_atol'])
    airborne_zero = bool(np.count_nonzero(wrench[1]) == 0 and np.count_nonzero(count[1]) == 0)
    contact_ok = bool(np.all(count[0, :, 1:].sum(axis=-1) > 0))
    bad_mask = 31 | tape.FLAG_NONCONV | tape.FLAG_SINGULAR | tape.FLAG_VERIFY_DIFF
    flags_ok = not np.any(decoded['flags'] & bad_mask)
    all_finite = all(np.isfinite(arrays[key]).all() for key in ('qpos', 'qvel', 'warm', 'terrain_wrench', 'actuator_torque'))
    saturated = bool(np.any(np.isclose(abs(actuator[2, 0]), packed['c_frclim'][6:], atol=.0003, rtol=0)))
    report.update(bitwise_equal=equal, different_storage_words=different_words, rows=rows, saturated_actuator_present=saturated, terrain_reaction_balances=reaction_ok,
                  no_terrain_contact_zero_force=airborne_zero, foot_contact_present=contact_ok,
                  strict_flags_ok=bool(flags_ok), finite=all_finite,
                  diagnostics=F.read_diagnostics(reset=True))
    report['pass'] = (all(equal.values()) and reaction_ok and airborne_zero and contact_ok and flags_ok and all_finite
                      and saturated and all(row['root_balance_ok'] and row['actuator_ok'] for row in rows))
    print(json.dumps({key: report[key] for key in ('bitwise_equal', 'terrain_reaction_balances',
                                                  'no_terrain_contact_zero_force', 'strict_flags_ok', 'pass')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--code-revision', required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    root = Path(__file__).resolve().parents[1]
    if out.is_relative_to(root):
        parser.error('--out must be outside checkout')
    out.mkdir(parents=True, exist_ok=False)
    report = dict(protocol=PROTOCOL, code_revision=args.code_revision, **{'pass': False})
    try:
        run(out, report)
        return 0 if report['pass'] else 1
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    raise SystemExit(main())

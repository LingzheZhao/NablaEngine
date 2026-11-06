"""Compare recorded first-substep convex contacts with MuJoCo collision geometry.

Run in provisioned compute. Inputs are a model and an existing scene-check run.
No dynamics advance. Every queried terrain pair, slot and oracle contact is saved.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import mujoco
    from nablaengine.g1_engine import step
    from nablaengine.model.convex import pack_static_convex_hulls
    import check_static_convex_cuda_gradients as checker

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--frames', type=int, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    model = mujoco.MjModel.from_binary_path(str(args.model))
    geoms = step.collision_geoms(model)
    local_ids = {geom: index for index, geom in enumerate(geoms)}
    packed = pack_static_convex_hulls(model, geoms)
    # Reuse the declared primitive compiler and its exact packed representation.
    checker.CUDA_RUNNER = checker.CUDA_RUNNER.replace('static_convex_contact(0,', 'static_convex_contact(int(x[16]),')
    checker.CUDA_RUNNER = checker.CUDA_RUNNER.replace('static_convex_vjp(0,', 'static_convex_vjp(int(x[16]),')
    checker.CUDA_RUNNER = checker.CUDA_RUNNER.replace('16*index', '17*index').replace('count*16', 'count*17')
    checker.CUDA_RUNNER = checker.CUDA_RUNNER.replace('basis < 13', 'basis < 0')
    checker.build(packed, args.out)
    query = checker.load_library(args.out/'primitive.so', 'run')
    pairs = []
    for pair in range(model.npair):
        a, b = int(model.pair_geom1[pair]), int(model.pair_geom2[pair])
        if model.geom_type[a] == mujoco.mjtGeom.mjGEOM_MESH:
            a, b = b, a
        if model.geom_type[b] == mujoco.mjtGeom.mjGEOM_MESH:
            pairs.append((pair, a, b))
    rows = []
    for frame in args.frames:
        captured = np.load(args.run/f'frame_{frame}_1_controls.npz', allow_pickle=False)
        data = mujoco.MjData(model)
        data.qpos[:] = captured['input_qpos']
        data.qvel[:] = captured['input_qvel']
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_collision(model, data)
        inputs, identities = [], []
        for pair, robot, hull in pairs:
            axis = data.geom_xmat[robot].reshape(3, 3)[:, 2]
            half_length = float(model.geom_size[robot, 1]) if model.geom_type[robot] == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.
            for slot in range(2):
                # MjData exposes geom rotation matrices, not geom quaternions.
                quaternion = np.empty(4)
                mujoco.mju_mat2Quat(quaternion, data.geom_xmat[hull])
                inputs.append([*data.geom_xpos[robot], *axis, model.geom_size[robot, 0], half_length,
                              *data.geom_xpos[hull], *quaternion, slot, local_ids[hull]])
                identities.append((pair, robot, hull, slot))
        x = np.asarray(inputs, np.float32)
        values = np.full((len(x), 13), np.nan, np.float32)
        jacobian = np.empty((len(x), 13, 6), np.float32)
        status = np.empty(len(x), np.int32)
        error = query(len(x), x.ctypes.data, values.ctypes.data, jacobian.ctypes.data, status.ctypes.data)
        if error:
            raise RuntimeError(f'CUDA query failed with {error}')
        contacts = [dict(geoms=[int(c.geom1), int(c.geom2)], distance=float(c.dist),
                         position=c.pos.tolist(), frame=c.frame.reshape(3, 3).tolist()) for c in data.contact]
        rows.append(dict(frame=frame, mujoco_contacts=contacts,
                         primitive_contacts=[dict(pair=p, geoms=[a,b], slot=s, status=int(status[i]),
                            input=x[i].tolist(), output=values[i].tolist())
                            for i,(p,a,b,s) in enumerate(identities) if status[i] < 0 or (status[i] == 1 and values[i,0] < 0)]))
        np.savez_compressed(args.out/f'frame_{frame}.npz', inputs=x, values=values, status=status,
                            identities=np.asarray(identities))
    (args.out/'report.json').write_text(json.dumps(dict(model=str(args.model), run=str(args.run), frames=rows), indent=2))


if __name__ == '__main__':
    main()

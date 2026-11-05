"""Compare native source-camera depth rays to independent MuJoCo rays.

Run the real-scene panel on provisioned compute. Images are 27x48 using the
source camera installation, full torso pose, FOV 87x58, groups 0 and 2, robot
self occlusion and near-surface skip-and-continue. No physics step is executed.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

PROTOCOL = dict(raw_shape=[27, 48], fov_degrees=[87, 58], geom_groups=[0, 2],
                range_atol_m=2e-4, optical_depth_atol_m=2e-4,
                max_distance_m=1e6, min_distance_m=.05, near_hit_hops=24,
                surface_epsilon_m=1e-5, body_exclude=-1,
                camera_position=[.04764571478+.0039635-.0042*math.cos(math.radians(48)),
                                 .015, .46268178553-.044+.0042*math.sin(math.radians(48))+.016],
                camera_quaternion_wxyz=[math.cos(math.radians(.5)/2)*math.cos(math.radians(48)/2),
                                        math.sin(math.radians(.5)/2), math.sin(math.radians(48)/2), 0.],
                geometry='all visible compiled geom surfaces including robot self occlusion',
                image_gradient='stop gradient')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--states', type=Path, required=True)
    parser.add_argument('--frames', type=int, nargs='+', default=[0, 190, 235, 320])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--code-revision', required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'protocol.json').write_text(json.dumps(dict(PROTOCOL, code_revision=args.code_revision,
        model=str(args.model), states=str(args.states), frames=args.frames, device=args.device), indent=2)+'\n')
    import mujoco
    import torch
    from nablaengine.depth_geometry import DepthGeometry, _rotate

    model = mujoco.MjModel.from_binary_path(str(args.model))
    data = mujoco.MjData(model)
    geometry = DepthGeometry.from_model(model, device=args.device)
    states = np.load(args.states, allow_pickle=False)
    frame_ids = states['frames'] if 'frames' in states else np.arange(len(states['qpos']))
    positions, quaternions = [], []
    for frame in args.frames:
        matches = np.flatnonzero(frame_ids == frame)
        if len(matches) != 1:
            raise ValueError(f'Frame {frame} absent or duplicated in selected state data')
        data.qpos[:] = states['qpos'][matches[0]]
        mujoco.mj_kinematics(model, data)
        positions.append(data.xpos.copy())
        quaternions.append(data.xquat.copy())
    tensor = lambda value: torch.tensor(np.asarray(value), device=args.device, dtype=torch.float32)
    body_positions, body_quaternions = tensor(positions), tensor(quaternions)
    torso = model.body('torso_link').id
    q = body_quaternions[:, torso]
    offset = tensor(PROTOCOL['camera_position'])
    origin = body_positions[:, torso]+_rotate(q, offset.expand(len(q), 3))
    camera_offset = tensor(PROTOCOL['camera_quaternion_wxyz'])
    camera_offset /= camera_offset.norm()
    # Quaternion product with WXYZ order.
    w = q[:, :1]*camera_offset[:1]-(q[:, 1:]*camera_offset[1:]).sum(-1, keepdim=True)
    vector = q[:, :1]*camera_offset[1:]+camera_offset[:1]*q[:, 1:]+torch.linalg.cross(q[:, 1:], camera_offset[1:].expand(len(q), 3))
    camera_quaternion = torch.cat([w, vector], -1)
    row, column = torch.meshgrid(torch.arange(27, device=args.device)+.5,
                                 torch.arange(48, device=args.device)+.5, indexing='ij')
    fx, fy = 48/(2*math.tan(math.radians(87)/2)), 27/(2*math.tan(math.radians(58)/2))
    local = torch.stack([torch.ones_like(row), -(column-24)/fx, -(row-13.5)/fy], -1).reshape(-1, 3)
    local /= local.norm(dim=-1, keepdim=True)
    directions = _rotate(camera_quaternion[:, None].expand(-1, len(local), -1), local[None].expand(len(q), -1, -1))
    origins = origin[:, None].expand_as(directions)
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    import time
    begin = time.perf_counter()
    ranges, ids = geometry.raycast(body_positions, body_quaternions, origins, directions)
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    seconds = time.perf_counter()-begin
    actual, actual_ids = ranges.cpu().numpy(), ids.cpu().numpy()
    origins, directions = origins.cpu().numpy().astype(float), directions.cpu().numpy().astype(float)
    # Float32 quaternion rotation introduces tiny norm drift. Match the unit
    # direction contract for both backends before comparing metric ray ranges.
    groups = np.zeros(6, np.uint8)
    groups[[0, 2]] = 1
    reference = np.full(actual.shape, 1e6)
    reference_ids = np.full(actual.shape, -1, np.int32)
    for index, frame in enumerate(args.frames):
        data.qpos[:] = states['qpos'][np.flatnonzero(frame_ids == frame)[0]]
        mujoco.mj_kinematics(model, data)
        for pixel, (point, direction) in enumerate(zip(origins[index], directions[index])):
            traveled = 0.
            current = point.copy()
            geom_id = np.array([-1], np.int32)
            for _ in range(PROTOCOL['near_hit_hops']+1):
                hit = mujoco.mj_ray(model, data, current, direction, groups, True, -1, geom_id)
                if hit < 0:
                    break
                if traveled+hit >= PROTOCOL['min_distance_m']:
                    reference[index, pixel] = traveled+hit
                    reference_ids[index, pixel] = geom_id[0]
                    break
                advance = hit+PROTOCOL['surface_epsilon_m']
                current += direction*advance
                traveled += advance
    np.savez_compressed(args.out/'rays.npz', frames=args.frames, actual_range=actual,
        reference_range=reference, actual_geom=actual_ids, reference_geom=reference_ids,
        origins=origins, directions=directions, local_rays=local.cpu().numpy())
    rows = []
    optical_cos = local[:, 0].cpu().numpy()
    for index, frame in enumerate(args.frames):
        both_hit = (actual_ids[index] >= 0) & (reference_ids[index] >= 0)
        hit_mismatch = (actual_ids[index] >= 0) != (reference_ids[index] >= 0)
        error = np.abs(actual[index]-reference[index])
        depth_error = np.abs(np.clip(actual[index]*optical_cos, 0, 2)-np.clip(reference[index]*optical_cos, 0, 2))
        self_hits = reference_ids[index] >= 0
        self_hits[self_hits] = model.geom_bodyid[reference_ids[index, self_hits]] != 0
        rows.append(dict(frame=frame, raw_pixels=1296, hits=int(both_hit.sum()),
                         self_occlusion_pixels=int(self_hits.sum()), hit_mismatches=int(hit_mismatch.sum()),
                         geom_id_mismatches=int(np.sum(actual_ids[index] != reference_ids[index])),
                         max_range_error_m=float(error[both_hit].max(initial=0)),
                         max_clipped_optical_depth_error_m=float(depth_error.max()),
                         passed=bool(not hit_mismatch.any() and error[both_hit].max(initial=0) <= PROTOCOL['range_atol_m']
                                     and depth_error.max() <= PROTOCOL['optical_depth_atol_m'])))
    result = dict(passed=all(row['passed'] for row in rows), frames=rows,
                  render_seconds=seconds, rendered_frames=len(args.frames),
                  torch=torch.__version__, mujoco=mujoco.__version__,
                  visible_geoms=len(geometry.geoms), code_revision=args.code_revision)
    (args.out/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()

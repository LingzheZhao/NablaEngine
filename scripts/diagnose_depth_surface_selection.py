"""Capture the first unrecovered depth surface in an eager training window.

Pass the original NablaMimic torch_train arguments after --, with a fresh training
output. The wrapper appends --execution eager and preserves seed/model/policy and
physical arguments. It raises before the production device assertion after saving
the exact ray packet. Primary traversal instrumentation independently records the
accepted hop's triangle; its ranges are compared with the unchanged legacy kernel.
A reproduced failure is diagnostic evidence, never a successful training result.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def plane_intersections(triangles, origin, direction):
    edge1, edge2 = triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
    cross = np.cross(direction, edge2)
    determinant = np.sum(edge1*cross, axis=-1)
    safe = np.where(determinant != 0, determinant, 1.)
    offset = origin-triangles[:, 0]
    u = np.sum(offset*cross, axis=-1)/safe
    q = np.cross(offset, edge1)
    v = np.sum(direction*q, axis=-1)/safe
    distance = np.sum(edge2*q, axis=-1)/safe
    threshold = np.finfo(np.float32).eps*np.linalg.norm(edge1, axis=-1)*np.linalg.norm(edge2, axis=-1)
    valid = (np.abs(determinant) > threshold) & (u >= 0) & (v >= 0) & (u+v <= 1) & (distance >= 0)
    return distance, u, v, valid


def inverse_rotate(q, vector):
    axis = -q[1:]
    twice = 2*np.cross(axis, vector)
    return vector+q[0]*twice+np.cross(axis, twice)


def save_packet(out, geometry, inputs, options, legacy, recovered, primary, call):
    import torch

    native = geometry.cuda_backend
    arrays = {name: value.detach().cpu().numpy() for name, value in zip(
        ('body_positions', 'body_quaternions', 'origins', 'directions'), inputs)}
    arrays.update({name: value.detach().cpu().numpy() for name, value in zip(
        ('legacy_ranges', 'legacy_geom_ids', 'recovered_packed_ids', 'recovered_triangle_ids',
         'primary_ranges', 'primary_geom_ids', 'primary_packed_ids', 'primary_triangle_ids', 'primary_hop_trace'),
        (*legacy, *recovered, *primary))})
    arrays.update({name: getattr(native, name).detach().cpu().numpy() for name in
        ('geometry', 'poses', 'sizes', 'bounds', 'nodes', 'triangles')})
    info = arrays['geometry'][arrays['recovered_packed_ids'].clip(min=0)]
    invalid = (arrays['legacy_geom_ids'] >= 0) & ((arrays['recovered_packed_ids'] < 0)
        | ((info[..., 1] != 0) & (arrays['recovered_triangle_ids'] < 0)))
    indices = np.argwhere(invalid)
    np.savez_compressed(out/'packet.npz', **arrays, invalid_indices=indices)
    world, pixel = map(int, indices[0])
    packed = int(arrays['primary_packed_ids'][world, pixel])
    triangle_id = int(arrays['primary_triangle_ids'][world, pixel])
    record = dict(status='reproduced_depth_surface_failure', training_passed=False,
        depth_call=call, invalid_rays=len(indices), batch=int(invalid.shape[0]), pixels=int(invalid.shape[1]),
        first_world=world, first_pixel=pixel,
        legacy_range=float(arrays['legacy_ranges'][world, pixel]),
        legacy_geom_id=int(arrays['legacy_geom_ids'][world, pixel]),
        recovered_triangle_id=int(arrays['recovered_triangle_ids'][world, pixel]),
        primary_packed_id=packed, primary_triangle_id=triangle_id,
        primary_hop_trace=arrays['primary_hop_trace'][world, pixel].tolist(),
        primary_ranges_equal=bool(torch.equal(legacy[0], primary[0])),
        primary_geom_ids_equal=bool(torch.equal(legacy[1], primary[1])), options=options,
        units='world/geom coordinates and ranges in metres; trace is world origin XYZ, travelled range, accepted hop index')
    if packed >= 0 and triangle_id >= 0:
        body = int(arrays['geometry'][packed, 0])
        pose = arrays['poses'][packed].astype(float)
        bp = arrays['body_positions'][world, body].astype(float)
        bq = arrays['body_quaternions'][world, body].astype(float)
        direction = inverse_rotate(pose[3:], inverse_rotate(bq, arrays['directions'][world, pixel].astype(float)))
        selected = arrays['triangles'][triangle_id:triangle_id+1].astype(float)
        for label, origin in [('original', arrays['origins'][world, pixel]),
                              ('accepted_hop', arrays['primary_hop_trace'][world, pixel, :3])]:
            local = inverse_rotate(pose[3:], inverse_rotate(bq, origin.astype(float)-bp)-pose[:3])
            distance, u, v, valid = plane_intersections(selected, local, direction)
            record[label+'_selected_face_float64'] = dict(range=float(distance[0]), u=float(u[0]),
                v=float(v[0]), within_triangle=bool(valid[0]), geom_local_origin=local.tolist())
        nodes = arrays['nodes'][slice(*map(int, arrays['geometry'][packed, 2:4]))]
        leaves = nodes[nodes[:, 1] > 0]
        ids = np.concatenate([np.arange(start, start+count) for start, count in leaves[:, :2]])
        local = np.asarray(record['original_selected_face_float64']['geom_local_origin'])
        distance, u, v, valid = plane_intersections(arrays['triangles'][ids].astype(float), local, direction)
        errors = np.abs(distance-record['legacy_range'])
        order = np.argsort(np.where(valid, errors, np.inf))[:3]
        record['original_ray_nearest_valid_faces_float64'] = [dict(triangle_id=int(ids[i]),
            range=float(distance[i]), error=float(errors[i]), u=float(u[i]), v=float(v[i]), valid=bool(valid[i])) for i in order]
        record['hit_body'] = body
        record['hit_body_world_position'] = bp.tolist()
    (out/'report.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--code-revision', required=True)
    parser.add_argument('--max-calls', type=int, default=128)
    parser.add_argument('training_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    training_args = args.training_args[1:] if args.training_args[:1] == ['--'] else args.training_args
    if not training_args or '--out' not in training_args:
        parser.error('Pass the original training arguments and a fresh --out after --')
    args.out.mkdir(parents=True, exist_ok=False)
    training_args += ['--execution', 'eager']
    (args.out/'protocol.json').write_text(json.dumps(dict(code_revision=args.code_revision,
        training_args=training_args, max_depth_calls=args.max_calls,
        operation='Capture first failed recovery before the production assertion; compare instrumented primary traversal',
        execution_change='eager only', result='Failure packet or explicit not-reproduced condition'), indent=2)+'\n')
    import torch
    from nablaengine.depth_geometry import DepthGeometry
    from nablamimic import torch_train

    original = DepthGeometry.raycast_with_pose_gradients
    calls = 0
    class Captured(RuntimeError):
        pass
    class NotReproduced(RuntimeError):
        pass
    def checked(self, *inputs, **options):
        nonlocal calls
        calls += 1
        native = self.cuda_backend
        legacy = self.raycast(*inputs, **options)
        recovered = native.pose_hits(*inputs, *legacy)
        packed, triangles = recovered
        info = native.geometry[packed.clamp_min(0)]
        invalid = (legacy[1] >= 0) & ((packed < 0) | ((info[..., 1] != 0) & (triangles < 0)))
        if bool(invalid.any()):
            primary = native.trace_primary_hits(*inputs, **options)
            save_packet(args.out, self, inputs, options, legacy, recovered, primary, calls)
            raise Captured(f'First failed depth surface saved to {args.out}')
        if calls >= args.max_calls:
            raise NotReproduced('No failed depth recovery in the declared call limit')
        return original(self, *inputs, **options)
    DepthGeometry.raycast_with_pose_gradients = checked
    try:
        torch_train.main(training_args)
    except Captured:
        return
    except NotReproduced as error:
        (args.out/'report.json').write_text(json.dumps(dict(status='not_reproduced', depth_calls=calls,
            training_passed=False, condition=str(error)), indent=2)+'\n')
        raise SystemExit(1)
    finally:
        DepthGeometry.raycast_with_pose_gradients = original
    (args.out/'report.json').write_text(json.dumps(dict(status='not_reproduced', depth_calls=calls,
        training_passed=False, condition='Training returned without a failed recovery'), indent=2)+'\n')
    raise SystemExit(1)


if __name__ == '__main__':
    main()

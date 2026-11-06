"""Check fixed-hit depth pose derivatives with generated math and source rays.

CPU mode evaluates the runtime Torch plane formula against independent generated
geometry. CUDA mode additionally checks the native selection, unchanged forward
bits, real camera finite differences and captured forward/backward execution.
Run CUDA mode on provisioned compute. No simulation step is executed.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from check_depth_pose_math import analytic, generated_state, matrix, ray, triangle_hit

PROTOCOL = dict(dtype='float32 CUDA, float64 CPU mathematics',
    fixed_hit_derivative='selected compiled geom-local triangle plane against the original ray',
    cpu_atol=2e-11, gpu_atol=3e-5, gpu_rtol=3e-4,
    camera_fd_steps=[.002, .001], camera_fd_atol=.008, camera_fd_rtol=.02,
    boundary_policy='selection changes recorded separately, no smooth derivative assertion',
    source_samples='one triangle-interior terrain hit and self hit per frame, maximum barycentric margin',
    misses='constant maximum range, zero derivative',
    grazing='zero derivative at the forward determinant rejection threshold',
    near_hits='skip near surfaces, differentiate the accepted plane against the original ray')


def quaternion_product(a, b):
    import torch
    return torch.cat([a[..., :1]*b[..., :1]-(a[..., 1:]*b[..., 1:]).sum(-1, keepdim=True),
        a[..., :1]*b[..., 1:]+b[..., :1]*a[..., 1:]+torch.linalg.cross(a[..., 1:], b[..., 1:])], -1)


def generated_math(device):
    import torch
    from nablaengine.depth_geometry import _selected_plane_range

    dtype = torch.float64 if device == 'cpu' else torch.float32
    tensor = lambda value: torch.tensor(np.asarray(value), dtype=dtype, device=device)
    state = generated_state()
    expected = analytic(state)
    bp, bq, origin, direction = [tensor(v).requires_grad_() for v in
        (state['body_position'], state['body_quaternion'], *ray(state))]
    triangles = tensor(state['triangle'])
    poses = tensor([0., 0., 0., 1., 0., 0., 0.])
    distance, valid = _selected_plane_range(bp, bq, origin, direction, poses, triangles, tensor(False).bool())
    gbp, gbq, go, gd = torch.autograd.grad(distance, (bp, bq, origin, direction))
    rotation = []
    for axis in np.eye(3):
        tangent = .5*quaternion_product(tensor(np.r_[0., axis]), bq)
        rotation.append((gbq*tangent).sum())
    rows = []
    for name, actual, reference in [('body_translation', gbp, expected['body_translation']),
            ('body_rotation', torch.stack(rotation), expected['body_rotation']),
            ('origin', go, expected['origin']), ('direction', gd, expected['direction'])]:
        values = actual.detach().cpu().numpy()
        atol, rtol = (PROTOCOL['cpu_atol'], 0.) if device == 'cpu' else (PROTOCOL['gpu_atol'], PROTOCOL['gpu_rtol'])
        rows.append(dict(case=name, actual=values.tolist(), expected=np.asarray(reference).tolist(),
            max_error=float(np.max(np.abs(values-reference))), passed=bool(np.allclose(values, reference, atol=atol, rtol=rtol))))
    # Re-express the same plane in a nonidentity compiled geom frame.
    gq = np.array([.9, .1, -.2, .3]); gq /= np.linalg.norm(gq)
    gp = np.array([.3, -.1, .2])
    local = (state['triangle']-gp)@matrix(gq)
    moved, _ = _selected_plane_range(bp, bq, origin, direction, tensor(np.r_[gp, gq]), tensor(local), tensor(False).bool())
    tolerance = PROTOCOL['cpu_atol'] if device == 'cpu' else PROTOCOL['gpu_atol']
    rows.append(dict(case='compiled_geom_frame', error=float((moved-distance).abs().detach()),
                     passed=bool((moved-distance).abs() <= tolerance)))
    # Rigid world co-motion includes rotating both body and camera positions.
    omega = tensor([.3, -.2, .4])
    common = (gbp*torch.linalg.cross(omega, bp)).sum()+(torch.stack(rotation)*omega).sum()
    common += (go*torch.linalg.cross(omega, origin)).sum()+(gd*torch.linalg.cross(omega, direction)).sum()
    rows.append(dict(case='common_world_rotation', derivative=float(common.detach()), passed=bool(common.abs() <= tolerance)))
    rows.append(dict(case='common_translation', max_error=float((gbp+go).abs().max()),
                     passed=bool((gbp+go).abs().max() <= tolerance)))
    # Constant outcomes and exactly grazing planes retain finite zero VJPs.
    grazing_direction = tensor([1., 0., 0.]).requires_grad_()
    grazing_origin = tensor([0., 0., 0.]).requires_grad_()
    identity = tensor([1., 0., 0., 0.])
    plane_value, plane_valid = _selected_plane_range(tensor([0., 0., 0.]), identity,
        grazing_origin, grazing_direction, poses, triangles, tensor(False).bool())
    grazing = torch.where(plane_valid, plane_value-plane_value.detach(), 0.)
    grazing_gradients = torch.autograd.grad(grazing, (grazing_origin, grazing_direction))
    rows.append(dict(case='grazing_zero_derivative', passed=not bool(plane_valid) and
        all(bool(torch.count_nonzero(g) == 0 and torch.isfinite(g).all()) for g in grazing_gradients)))
    clipped_value, _ = _selected_plane_range(bp, bq, origin, direction, poses, triangles, tensor(False).bool())
    image = (clipped_value/1.5).clamp(0, 1)
    image_gradient = torch.autograd.grad(image, origin)[0]
    rows.append(dict(case='clipped_image_zero_derivative', passed=bool(torch.count_nonzero(image_gradient) == 0)))
    assert bool(valid)
    return rows


def compare_graphs(geometry, inputs):
    import torch

    def render(*values):
        return geometry.raycast_with_pose_gradients(*values)

    buffers = [value.detach().clone().requires_grad_() for value in inputs]
    weights = torch.arange(inputs[2].shape[1], device=inputs[2].device).remainder(7).float()[None]/7
    compiled = torch.compile(render, fullgraph=True)
    def evaluate(function):
        result, ids = function(*buffers)
        gradients = torch.autograd.grad((result*weights).sum(), buffers)
        return result, ids, gradients
    for _ in range(3):
        evaluate(render)
        evaluate(compiled)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    compiled_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = evaluate(render)
    with torch.cuda.graph(compiled_graph):
        captured_compiled = evaluate(compiled)
    rows = []
    for variant in range(2):
        with torch.no_grad():
            for buffer, original in zip(buffers, inputs):
                buffer.copy_(original.roll(variant+1, 0))
            buffers[2][..., 2] += .001*(variant+1)
        expected = evaluate(render)
        graph.replay()
        compiled_graph.replay()
        for name, actual in [('cuda_graph', captured), ('fullgraph', evaluate(compiled)),
                             ('compiled_cuda_graph', captured_compiled)]:
            error = max(float((a-b).abs().max()) for a, b in zip(actual[2], expected[2]))
            rows.append(dict(mode=name, variant=variant,
                ranges_equal=bool(torch.equal(actual[0], expected[0])),
                geom_ids_equal=bool(torch.equal(actual[1], expected[1])), max_gradient_error=error,
                gradients_close=all(bool(torch.allclose(a, b, atol=3e-5, rtol=3e-4))
                                    for a, b in zip(actual[2], expected[2]))))
    return dict(passed=all(r['ranges_equal'] and r['geom_ids_equal'] and r['gradients_close'] for r in rows), comparisons=rows)


def check_generated_cuda(device):
    import torch
    from nablaengine.depth_cuda import CudaDepthGeometry
    from nablaengine.depth_geometry import DepthGeometry, VisibleGeom

    tensor = lambda x: torch.tensor(np.asarray(x), device=device, dtype=torch.float32)
    # Two parallel surfaces on the same geom exercise near-hit skipping and
    # recovery of an exit/later face from the original ray.
    triangles = np.array([[[-2, -2, .02], [2, -2, .02], [0, 2, .02]],
                          [[-2, -2, .7], [2, -2, .7], [0, 2, .7]]], np.float32)
    geom = VisibleGeom(0, 1, 7, tensor([0, 0, 0]).cpu(), tensor([1, 0, 0, 0]).cpu(),
                       tensor([1, 1, 1]).cpu(), tensor(triangles).cpu())
    geometry = DepthGeometry((geom,), 2, CudaDepthGeometry((geom,), 2, device))
    positions = tensor([[[0, 0, 0], [0, 0, 0]]]).requires_grad_()
    quaternions = tensor([[[1, 0, 0, 0], [1, 0, 0, 0]]]).requires_grad_()
    origins = tensor([[[0, 0, 0], [4, 0, 0]]]).requires_grad_()
    directions = tensor([[[0, 0, 1], [0, 0, 1]]]).requires_grad_()
    ranges, ids = geometry.raycast_with_pose_gradients(positions, quaternions, origins, directions)
    plain = geometry.raycast(positions, quaternions, origins, directions)
    gradients = torch.autograd.grad(ranges.sum(), (positions, quaternions, origins, directions))
    near = dict(case='accepted_plane_after_near_skip', range=float(ranges[0, 0]),
        origin_gradient=gradients[2][0, 0].tolist(), body_gradient=gradients[0][0, 1].tolist(),
        passed=bool(abs(float(ranges[0, 0])-.7) < 1e-6 and
            torch.allclose(gradients[2][0, 0], tensor([0, 0, -1])) and
            torch.allclose(gradients[0][0, 1], tensor([0, 0, 1]))))
    miss = dict(case='miss_zero_derivative', passed=bool(ids[0, 1] == -1 and
        torch.equal(gradients[2][0, 1], tensor([0, 0, 0])) and
        torch.equal(gradients[3][0, 1], tensor([0, 0, 0]))))
    bits = dict(case='forward_bits', passed=bool(torch.equal(ranges, plain[0]) and torch.equal(ids, plain[1])))
    clipped, _ = geometry.raycast_with_pose_gradients(positions, quaternions, origins, directions, max_distance=.5)
    clipped_grad = torch.autograd.grad(clipped.sum(), (origins, directions))
    far = dict(case='far_range_zero_derivative', passed=all(bool(torch.count_nonzero(g) == 0) for g in clipped_grad))
    state = generated_state()
    tilted_geom = VisibleGeom(0, 1, 7, tensor([0, 0, 0]).cpu(), tensor([1, 0, 0, 0]).cpu(),
        tensor([1, 1, 1]).cpu(), tensor(state['triangle'][None]).cpu())
    tilted = DepthGeometry((tilted_geom,), 2, CudaDepthGeometry((tilted_geom,), 2, device))
    pose_inputs = [tensor(value).requires_grad_() for value in
        ([[[0, 0, 0], state['body_position']]], [[[1, 0, 0, 0], state['body_quaternion']]],
         [[ray(state)[0]]], [[ray(state)[1]]])]
    value, _ = tilted.raycast_with_pose_gradients(*pose_inputs)
    pose_gradients = torch.autograd.grad(value.sum(), pose_inputs)
    expected = analytic(state)
    rotation = torch.stack([(pose_gradients[1][0, 1]*.5*quaternion_product(
        tensor(np.r_[0., axis]), pose_inputs[1][0, 1])).sum() for axis in np.eye(3)])
    pose_rows = []
    for name, actual, reference in [('body_translation', pose_gradients[0][0, 1], expected['body_translation']),
            ('body_rotation', rotation, expected['body_rotation']),
            ('origin', pose_gradients[2][0, 0], expected['origin']),
            ('direction', pose_gradients[3][0, 0], expected['direction'])]:
        error = float((actual-tensor(reference)).abs().max().detach())
        pose_rows.append(dict(case='native_tilted_'+name, max_error=error,
            passed=bool(torch.allclose(actual, tensor(reference), atol=PROTOCOL['gpu_atol'], rtol=PROTOCOL['gpu_rtol']))))
    return [near, miss, bits, far]+pose_rows, compare_graphs(geometry, (positions, quaternions, origins, directions))


def check_source(args, model, geometry):
    import mujoco
    import torch
    data = mujoco.MjData(model)
    states, rays = np.load(args.states, allow_pickle=False), np.load(args.rays, allow_pickle=False)
    qpos, qvel = states['qpos'], states['qvel']
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or qvel.shape != (len(qpos), model.nv):
        raise ValueError('Reference qpos/qvel dimensions must match the loaded model')
    if (states['names'].shape != (1,) or states['starts'].tolist() != [0]
            or states['lengths'].tolist() != [len(qpos)] or not len(qpos)):
        raise ValueError('Reference shard must contain one complete motion starting at frame zero')
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise ValueError('Reference qpos/qvel must be finite')
    frames = rays['frames']
    if (frames.ndim != 1 or not len(frames) or frames.dtype.kind not in 'iu'
            or np.any(frames < 0) or np.any(frames >= len(qpos))):
        raise ValueError('Ray frames must be integer indices within the reference motion')
    positions, quaternions = [], []
    for frame in frames:
        data.qpos[:] = qpos[frame]
        mujoco.mj_kinematics(model, data)
        positions.append(data.xpos.copy())
        quaternions.append(data.xquat.copy())
    tensor = lambda value: torch.tensor(np.asarray(value), device=args.device, dtype=torch.float32)
    inputs = [tensor(value).requires_grad_() for value in
              (positions, quaternions, rays['origins'], rays['directions'])]
    plain, plain_ids = geometry.raycast(*inputs)
    actual, ids = geometry.raycast_with_pose_gradients(*inputs)
    _, _, packed, triangle_ids = geometry.cuda_backend.raycast_with_hits(*inputs)
    # Compute barycentric margin on the selected triangle to choose samples
    # before finite differences. Edge/surface switches remain separately visible.
    owner = geometry.cuda_backend
    g = packed.clamp_min(0)
    body = owner.geometry[g, 0].long()
    batch = torch.arange(len(inputs[0]), device=args.device)[:, None]
    from nablaengine.depth_geometry import _rotate
    conjugate = tensor([1, -1, -1, -1])
    local = _rotate(inputs[1][batch, body]*conjugate,
        inputs[2]+actual[..., None]*inputs[3]-inputs[0][batch, body])
    local = _rotate(owner.poses[g, 3:]*conjugate, local-owner.poses[g, :3])
    triangles = owner.triangles[triangle_ids.clamp_min(0)]
    a, b = triangles[..., 1, :]-triangles[..., 0, :], triangles[..., 2, :]-triangles[..., 0, :]
    normal = torch.linalg.cross(a, b)
    squared = normal.square().sum(-1).clamp_min(torch.finfo(torch.float32).tiny)
    u = (torch.linalg.cross(local-triangles[..., 0, :], b)*normal).sum(-1)/squared
    v = (torch.linalg.cross(a, local-triangles[..., 0, :])*normal).sum(-1)/squared
    margins = torch.minimum(torch.minimum(u, v), 1-u-v).detach()
    rows = []
    groups = np.array([1, 0, 1, 0, 0, 0], np.uint8)
    for f, frame in enumerate(frames):
        data.qpos[:] = qpos[frame]
        mujoco.mj_kinematics(model, data)
        for category, mask in [('terrain', body[f] == 0), ('self', body[f] != 0)]:
            candidate = (ids[f] >= 0) & (triangle_ids[f] >= 0) & mask
            if not bool(candidate.any()):
                rows.append(dict(frame=int(frame), category=category, condition='absent'))
                continue
            pixel = int(torch.where(candidate, margins[f], -torch.inf).argmax())
            origin_gradient = torch.autograd.grad(actual[f, pixel], inputs[2], retain_graph=True)[0][f, pixel]
            triangle = triangles[f, pixel].detach().cpu().numpy().astype(float)
            model_geom = int(ids[f, pixel])
            world_triangle = triangle@data.geom_xmat[model_geom].reshape(3, 3).T+data.geom_xpos[model_geom]
            for axis in range(3):
                fd, selected, face_errors = [], [], []
                origin = inputs[2][f, pixel].detach().cpu().numpy().astype(float)
                direction = inputs[3][f, pixel].detach().cpu().numpy().astype(float)
                for step in PROTOCOL['camera_fd_steps']:
                    distances, geoms = [], []
                    for sign in (-1, 1):
                        point = origin.copy(); point[axis] += sign*step
                        selected_plane_hit = triangle_hit(world_triangle, point, direction)
                        traveled = 0.; gid = np.array([-1], np.int32)
                        distance = 1e6
                        for hop in range(25):
                            hit = mujoco.mj_ray(model, data, point, direction, groups, True, -1, gid)
                            if hit < 0:
                                gid[0] = -1
                                break
                            if hit+traveled >= .05:
                                distance = hit+traveled
                                break
                            if hop == 24:
                                gid[0] = -1
                                break
                            point += direction*(hit+1e-5); traveled += hit+1e-5
                        distances.append(distance); geoms.append(int(gid[0]))
                        face_errors.append(abs(distance-selected_plane_hit))
                    fd.append((distances[1]-distances[0])/(2*step)); selected.extend(geoms)
                derivative = float(origin_gradient[axis])
                smooth = all(k == int(ids[f, pixel]) for k in selected) and max(face_errors) <= 2e-5
                passed = bool(np.allclose(fd, derivative, atol=PROTOCOL['camera_fd_atol'], rtol=PROTOCOL['camera_fd_rtol'])) if smooth else None
                rows.append(dict(frame=int(frame), category=category, pixel=pixel, axis=axis,
                    barycentric_margin=float(margins[f, pixel]), derivative=derivative, independent_fd=fd,
                    selected_geoms=selected, max_selected_face_error=max(face_errors),
                    condition='same_face' if smooth else 'surface_switch', passed=passed))
    baseline_error = np.abs(plain.detach().cpu().numpy()-rays['reference_range'])
    hit = rays['reference_geom'] >= 0
    return dict(forward_ranges_bitwise=bool(torch.equal(plain, actual)),
        forward_geom_ids_bitwise=bool(torch.equal(plain_ids, ids)),
        independent_geom_ids_equal=bool(np.array_equal(ids.cpu().numpy(), rays['reference_geom'])),
        independent_max_range_error=float(baseline_error[hit].max(initial=0.)),
        finite_gradients=all(bool(torch.isfinite(g).all()) for g in
            torch.autograd.grad(torch.where(ids >= 0, actual, 0.).sum(), inputs)),
        rows=rows, visible_geoms=len(geometry.geoms),
        smooth_terrain_samples=sum(r.get('condition') == 'same_face' and r['category'] == 'terrain' for r in rows),
        smooth_self_samples=sum(r.get('condition') == 'same_face' and r['category'] == 'self' for r in rows))



def check_failure_packet(args, geometry):
    """Replay the recorded failed secondary query with primary face selection."""
    import torch
    from diagnose_depth_surface_selection import inverse_rotate

    packet = np.load(args.packet, allow_pickle=False)
    native = geometry.cuda_backend
    for name in ('geometry', 'poses', 'sizes', 'bounds', 'nodes', 'triangles'):
        if not np.array_equal(getattr(native, name).cpu().numpy(), packet[name]):
            raise ValueError(f'Failure packet {name} differs from the loaded model packing')
    inputs = [torch.tensor(packet[name], device=args.device).requires_grad_() for name in
              ('body_positions', 'body_quaternions', 'origins', 'directions')]
    ranges, ids = geometry.raycast_with_pose_gradients(*inputs)
    primary = native.raycast_with_hits(*inputs)
    hit = ids >= 0
    gradients = torch.autograd.grad(torch.where(hit, ranges, 0.).sum(), inputs)
    rows = []
    for world, pixel in packet['invalid_indices']:
        g = int(primary[2][world, pixel]); face = int(primary[3][world, pixel])
        body = int(packet['geometry'][g, 0]); pose = packet['poses'][g].astype(float)
        q = packet['body_quaternions'][world, body].astype(float)
        bp = packet['body_positions'][world, body].astype(float)
        origin = packet['origins'][world, pixel].astype(float)
        direction = packet['directions'][world, pixel].astype(float)
        triangle = packet['triangles'][face].astype(float)
        normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
        local_origin = inverse_rotate(pose[3:], inverse_rotate(q, origin-bp)-pose[:3])
        local_direction = inverse_rotate(pose[3:], inverse_rotate(q, direction))
        denominator = normal@local_direction
        distance = (normal@(triangle[0]-local_origin))/denominator
        world_normal = matrix(q)@matrix(pose[3:])@normal
        expected_origin, expected_direction = -world_normal/denominator, -distance*world_normal/denominator
        origin_gradient = gradients[2][world, pixel].cpu().numpy()
        direction_gradient = gradients[3][world, pixel].cpu().numpy()
        close = all(np.allclose(a, b, atol=PROTOCOL['gpu_atol'], rtol=PROTOCOL['gpu_rtol'])
                    for a, b in ((origin_gradient, expected_origin), (direction_gradient, expected_direction)))
        rows.append(dict(world=int(world), pixel=int(pixel), primary_triangle_id=face,
            max_origin_gradient_error=float(np.max(np.abs(origin_gradient-expected_origin))),
            max_direction_gradient_error=float(np.max(np.abs(direction_gradient-expected_direction))),
            passed=bool(close)))
    return dict(previously_unrecovered_rays=len(rows),
        ranges_bitwise=bool(np.array_equal(ranges.detach().cpu().numpy().view(np.uint32), packet['legacy_ranges'].view(np.uint32))),
        geom_ids_equal=bool(np.array_equal(ids.cpu().numpy(), packet['legacy_geom_ids'])),
        primary_triangle_ids_equal=bool(np.array_equal(primary[3].cpu().numpy(), packet['primary_triangle_ids'])),
        finite_gradients=all(bool(torch.isfinite(g).all()) for g in gradients), comparisons=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-math-only', action='store_true')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--states', type=Path, help='current single-motion reference shard')
    parser.add_argument('--rays', type=Path, help='rays.npz produced by check_depth_geometry.py for this model/states')
    parser.add_argument('--packet', type=Path, help='packet.npz from diagnose_depth_surface_selection.py')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--code-revision', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not args.cpu_math_only and not all((args.model, args.states, args.rays)):
        parser.error('CUDA checks require --model, --states and --rays')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'protocol.json').write_text(json.dumps(dict(PROTOCOL, code_revision=args.code_revision,
        model=str(args.model), states=str(args.states), rays=str(args.rays), packet=str(args.packet)), indent=2)+'\n')
    result = dict(math=generated_math('cpu' if args.cpu_math_only else args.device))
    if not args.cpu_math_only:
        import torch
        result['generated_cuda'], result['graphs'] = check_generated_cuda(args.device)
        import mujoco
        from nablaengine.depth_geometry import DepthGeometry
        model = mujoco.MjModel.from_binary_path(str(args.model))
        geometry = DepthGeometry.from_model(model, device=args.device)
        result['source'] = check_source(args, model, geometry)
        if args.packet is not None:
            result['failure_packet'] = check_failure_packet(args, geometry)
        result['gpu'] = torch.cuda.get_device_name(args.device)
    passed = all(row['passed'] for row in result['math'])
    if not args.cpu_math_only:
        source = result['source']
        passed = (passed and all(row['passed'] for row in result['generated_cuda']) and result['graphs']['passed']
            and source['forward_ranges_bitwise'] and source['forward_geom_ids_bitwise'] and source['finite_gradients']
            and source['smooth_terrain_samples'] > 0 and source['smooth_self_samples'] > 0
            and source['independent_geom_ids_equal'] and source['independent_max_range_error'] <= 2e-4
            and all(row.get('passed') is not False for row in source['rows']))
    if 'failure_packet' in result:
        packet = result['failure_packet']
        passed = (passed and packet['ranges_bitwise'] and packet['geom_ids_equal']
            and packet['primary_triangle_ids_equal'] and packet['finite_gradients']
            and all(row['passed'] for row in packet['comparisons']))
    result['passed'] = bool(passed)
    (args.out/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()

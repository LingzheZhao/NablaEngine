"""Screen reference resets against the compiled physical convex terrain.

All geometry comes from the model's checked packed hull representation. The
reference is a current prepare_instinct_motion output directory. World transforms,
terrain identities and explicit foot-pair bindings are checked before screening.
Reference poses and joints retain their source values. Static geometry screening records every
selected frame, including centerline intersections and absent foot proximity.

Run full-reference screening on provisioned compute. Output requires a fresh
external directory. The small --self-test uses generated cube geometry only.
Add --execute, --sensor DIRECTORY and explicit --frames to run whole-step GPU
VJPs, physical reference comparisons, continuous chains and a native SGD update.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = dict(geometry_tolerance_m=2e-5, support_distance_m=0.,
                near_support_distance_m=.03, chunk_frames=16,
                screening="all selected reference frames, unchanged positions, joints and velocities",
                quantities="static centerline intersections and capsule-surface distances",
                physics_evidence="geometry screening")


def point_edge_squared(points, start, edge):
    displacement = points[:, None, :]-start[None]
    denominator = np.einsum('ij,ij->i', edge, edge)
    parameter = np.clip(np.einsum('nij,ij->ni', displacement, edge)/denominator, 0, 1)
    residual = displacement-parameter[..., None]*edge[None]
    return np.einsum('nij,nij->ni', residual, residual)


def segment_hull(starts, ends, hull):
    """Exact segment-surface distance and tolerance-expanded solid intersection.

    Intersecting centerlines return an unavailable distance and an explicit flag.
    Outside a convex solid, the nearest pair is an endpoint/face, endpoint/edge,
    vertex/segment or interior segment/edge pair. Segment/face intersections are
    caught by clipping against every outward halfspace.
    """
    vertices, triangles, normals = hull['vertices'], hull['triangles'], hull['normals']
    delta = ends-starts
    offset = np.einsum('nfd,fd->nf', starts[:, None]-triangles[None, :, 0], normals)
    slope = delta@normals.T
    tolerance = PROTOCOL['geometry_tolerance_m']
    parallel = np.abs(slope) <= tolerance
    crossing = (tolerance-offset)/np.where(parallel, 1., slope)
    lower = np.maximum(0., np.max(np.where(slope < -tolerance, crossing, -np.inf), axis=1))
    upper = np.minimum(1., np.min(np.where(slope > tolerance, crossing, np.inf), axis=1))
    blocked = np.any(parallel & (offset > tolerance), axis=1)
    intersects = (~blocked) & (lower <= upper)
    edge_start, edge_end = hull['edges'][:, 0], hull['edges'][:, 1]
    edges = edge_end-edge_start
    squared = np.minimum(point_edge_squared(starts, edge_start, edges).min(axis=1),
                         point_edge_squared(ends, edge_start, edges).min(axis=1))
    # Endpoints projected into triangle interiors.
    e0, e1 = triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
    d00 = np.einsum('ij,ij->i', e0, e0)
    d01 = np.einsum('ij,ij->i', e0, e1)
    d11 = np.einsum('ij,ij->i', e1, e1)
    determinant = d00*d11-d01*d01
    for points in (starts, ends):
        difference = points[:, None]-triangles[None, :, 0]
        height = np.einsum('nfd,fd->nf', difference, normals)
        projected = difference-height[..., None]*normals[None]
        d20 = np.einsum('nfd,fd->nf', projected, e0)
        d21 = np.einsum('nfd,fd->nf', projected, e1)
        u = (d11*d20-d01*d21)/determinant
        v = (d00*d21-d01*d20)/determinant
        interior = (u >= 0) & (v >= 0) & (u+v <= 1)
        squared = np.minimum(squared, np.where(interior, height*height, np.inf).min(axis=1))
    # Hull vertices projected onto the capsule centerline.
    length_squared = np.einsum('ij,ij->i', delta, delta)
    difference = vertices[None]-starts[:, None]
    parameter = np.clip(np.einsum('nvd,nd->nv', difference, delta)
                        /np.where(length_squared > 0, length_squared, 1.)[:, None], 0, 1)
    residual = difference-parameter[..., None]*delta[:, None]
    squared = np.minimum(squared, np.einsum('nvd,nvd->nv', residual, residual).min(axis=1))
    # The unconstrained interior minimizer of two nonparallel segments.
    edge_squared = np.einsum('ij,ij->i', edges, edges)
    uv = delta@edges.T
    det = length_squared[:, None]*edge_squared[None]-uv*uv
    origin_delta = starts[:, None]-edge_start[None]
    u_origin = np.einsum('ned,nd->ne', origin_delta, delta)
    v_origin = np.einsum('ned,ed->ne', origin_delta, edges)
    safe_det = np.where(det > 1e-20, det, 1.)
    s = (uv*v_origin-edge_squared[None]*u_origin)/safe_det
    t = (length_squared[:, None]*v_origin-uv*u_origin)/safe_det
    residual = origin_delta+s[..., None]*delta[:, None]-t[..., None]*edges[None]
    interior = (det > 1e-20) & (s > 0) & (s < 1) & (t > 0) & (t < 1)
    squared = np.minimum(squared, np.where(interior, np.einsum('ned,ned->ne', residual, residual), np.inf).min(axis=1))
    intersects |= squared <= tolerance*tolerance
    distance = np.where(intersects, np.nan, np.sqrt(np.maximum(squared, 0)))
    return distance, intersects


def hull_geometry(vertices, faces, normals=None):
    triangles = vertices[faces]
    if normals is None:
        normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        sign = np.einsum('ij,ij->i', normals, triangles[:, 0]-vertices.mean(axis=0))
        normals *= np.where(sign >= 0, 1., -1.)[:, None]
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    return dict(vertices=vertices, triangles=triangles, normals=normals, edges=vertices[edges])


def self_test():
    vertices = np.array([[1 if index & (1 << axis) else -1 for axis in range(3)] for index in range(8)], float)
    quads = [[0, 2, 6, 4], [1, 3, 7, 5], [0, 1, 5, 4], [2, 3, 7, 6], [0, 1, 3, 2], [4, 5, 7, 6]]
    faces = np.array([triangle for a, b, c, d in quads for triangle in ([a, b, c], [a, c, d])])
    hull = hull_geometry(vertices, faces)
    starts = np.array([[1.1, 0, 0], [-.5, 0, 1.1], [1.2, 0, 1.4], [-2, 0, 0], [0, 0, 0]])
    ends = np.array([[1.1, 0, 0], [.5, 0, 1.1], [1.8, 0, 1.4], [2, 0, 0], [0, 0, 0]])
    distance, inside = segment_hull(starts, ends, hull)
    assert np.allclose(distance[:3], [.1, .1, np.sqrt(.2**2+.4**2)], rtol=0, atol=1e-12)
    assert inside.tolist() == [False, False, False, True, True]
    return dict(generated_cases=5, passed=True)


def screen(args, report, out):
    import mujoco
    from nablaengine.g1_engine import step
    from nablaengine.model import capability
    from nablaengine.model.convex import pack_static_convex_hulls
    contract = json.loads((args.scene/'contract.json').read_text())
    motion_contract = json.loads((args.reference/'screening.json').read_text())
    selected = [m for m in motion_contract['motions'] if m['name'] == args.motion]
    if len(selected) != 1 or selected[0]['terrain_id'] != contract['terrain_index']:
        raise ValueError('Reference motion and physical scene terrain identities differ')
    if motion_contract['binding']['world_translation_m'] != contract['frame']['input_to_world_translation_m']:
        raise ValueError('Reference and physical scene world frame translations differ')
    if motion_contract['binding']['body_linear_velocity'] != 'World velocity of xpos body origin, mj_objectVelocity(mjOBJ_XBODY)':
        raise ValueError('Current body-origin velocity reference binding required')
    model = mujoco.MjModel.from_binary_path(str(args.scene/'model.mjb'))
    foot_pairs = list(map(int, contract['foot_pairs']))
    actual_order = [model.joint(model.actuator_trnid[i, 0]).name for i in range(model.nu)]
    if actual_order != contract['actuator_joint_names']:
        raise ValueError('Scene actuator contract mismatch')
    heuristic = [i for i in range(model.npair) if '_foot' in model.pair(i).name and model.pair(i).name.endswith('_floor')]
    if heuristic != foot_pairs:
        raise ValueError('Training environment foot selection differs from scene contract')
    validation = capability.check(model, foot_pairs, scope='engine')
    if not validation.ok():
        raise capability.UnsupportedModel(validation, 'terrain scene screen')
    geoms = step.collision_geoms(model)
    packed = pack_static_convex_hulls(model, geoms)
    arrays = {name: values for name, _, values in packed.arrays}
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    hulls = {}
    for local, geom in enumerate(geoms):
        index = int(arrays['c_ghull'][local])
        if index < 0:
            continue
        if model.geom_bodyid[geom] != 0:
            raise ValueError('A terrain hull is attached to a moving body')
        start, count = (int(arrays[key][index]) for key in ('c_hull_vertexadr', 'c_hull_vertexnum'))
        first, faces = (int(arrays[key][index]) for key in ('c_hull_faceadr', 'c_hull_facenum'))
        rotation = data.geom_xmat[geom].reshape(3, 3)
        vertices = (arrays['c_hull_vertices'][start:start+count].astype(float)
                    +arrays['c_hull_origin'][index])@rotation.T+data.geom_xpos[geom]
        normals = arrays['c_hull_normals'][first:first+faces].astype(float)@rotation.T
        hulls[geom] = hull_geometry(vertices, arrays['c_hull_faces'][first:first+faces], normals)
    robots = sorted({int(g) for p in range(model.npair) for g in (model.pair_geom1[p], model.pair_geom2[p])
                     if model.geom_bodyid[g] != 0})
    pairs = []
    for p, (a, b) in enumerate(zip(model.pair_geom1, model.pair_geom2)):
        if a in hulls and b in robots:
            pairs.append((p, int(b), int(a)))
        elif b in hulls and a in robots:
            pairs.append((p, int(a), int(b)))
    robot_index = {g: i for i, g in enumerate(robots)}
    with np.load(args.reference/f'{args.motion}.npz', allow_pickle=False) as source:
        qpos, qvel = source['qpos'].copy(), source['qvel'].copy()
    if (qpos.ndim != 2 or qpos.shape[1] != model.nq or qvel.shape != (len(qpos), model.nv)
            or not np.isfinite(qpos).all() or not np.isfinite(qvel).all()
            or not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1., rtol=0, atol=1e-5)):
        raise ValueError('Reference rows require finite model-bound states and normalized WXYZ quaternions')
    frames = np.arange(len(qpos)) if args.frames == 'all' else np.array([int(v) for v in args.frames.split(',')])
    if len(set(frames.tolist())) != len(frames) or np.any(frames < 0) or np.any(frames >= len(qpos)):
        raise ValueError('Selected frames must be distinct supported reference rows')
    distance = np.empty((len(frames), len(pairs)))
    intersects = np.empty(distance.shape, bool)
    radii = np.array([model.geom_size[g, 0] for g in robots])
    half = np.array([model.geom_size[g, 1] if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE else 0. for g in robots])
    for offset in range(0, len(frames), PROTOCOL['chunk_frames']):
        chunk = frames[offset:offset+PROTOCOL['chunk_frames']]
        centers, axes = [], []
        for frame in chunk:
            data.qpos[:] = qpos[frame]
            mujoco.mj_kinematics(model, data)
            centers.append(data.geom_xpos[robots].copy())
            axes.append(data.geom_xmat[robots].reshape(-1, 3, 3)[:, :, 2].copy())
        centers, axes = np.asarray(centers), np.asarray(axes)
        start = (centers-axes*half[None, :, None]).reshape(-1, 3)
        end = (centers+axes*half[None, :, None]).reshape(-1, 3)
        for hull_geom, hull in hulls.items():
            separation, invalid = segment_hull(start, end, hull)
            separation = separation.reshape(len(chunk), len(robots))-radii[None]
            invalid = invalid.reshape(len(chunk), len(robots))
            for pair_index, (_, robot, terrain) in enumerate(pairs):
                if terrain == hull_geom:
                    distance[offset:offset+len(chunk), pair_index] = separation[:, robot_index[robot]]
                    intersects[offset:offset+len(chunk), pair_index] = invalid[:, robot_index[robot]]
    foot = np.isin([p[0] for p in pairs], foot_pairs)
    contact = (distance <= PROTOCOL['support_distance_m']) & ~intersects
    near = (distance <= PROTOCOL['near_support_distance_m']) & ~intersects
    report.update(scene_contract=contract, reference_artifact_id=motion_contract['artifact_id'],
                  reference_path=str(args.reference), motion=args.motion, frames=frames.tolist(),
                  capability=validation.to_dict(), selected_frame_count=len(frames),
                  centerline_invalid_frames=int(intersects.any(axis=1).sum()),
                  centerline_invalid_fraction=float(intersects.any(axis=1).mean()),
                  terrain_pair_bindings=[dict(pair=p, robot_geom=model.geom(r).name, hull_geom=model.geom(h).name)
                                         for p, r, h in pairs],
                  foot_contact_candidate_frames=int(contact[:, foot].any(axis=1).sum()),
                  foot_near_support_frames=int(near[:, foot].any(axis=1).sum()), rows=[])
    for index, frame in enumerate(frames):
        report['rows'].append(dict(frame=int(frame), unsupported_pair_ids=[pairs[p][0] for p in np.flatnonzero(intersects[index])],
                                   capsule_contact_candidate_pair_ids=[pairs[p][0] for p in np.flatnonzero(contact[index])],
                                   closest_foot_surface_distance_m=(float(np.nanmin(distance[index, foot]))
                                       if np.isfinite(distance[index, foot]).any() else None),
                                   minimum_surface_distance_m=(float(np.nanmin(distance[index]))
                                       if np.isfinite(distance[index]).any() else None)))
    np.savez_compressed(out/'geometry.npz', frames=frames, pair_ids=np.array([p[0] for p in pairs]),
                        distance=distance, centerline_intersects=intersects, foot_pair_mask=foot,
                        qpos=qpos[frames], qvel=qvel[frames])
    report['completed'] = True
    return dict(model=model, foot_pairs=foot_pairs, qpos=qpos, qvel=qvel, frames=frames,
                contract=contract, packed_hulls=packed, hull_geoms=list(hulls),
                fps=json.loads((args.reference/'index.json').read_text())['fps'])


EXECUTION_PROTOCOL = dict(horizons=[1, 4], substeps=4, iterations=50, friction=1., com=[0., 0., 0.],
                          fd_steps=[1e-5, 5e-6], gradient_atol=.01, gradient_rtol=.03,
                          fd_stability_atol=.001, fd_stability_rtol=.01,
                          qpos_atol=.0005, qvel_atol=.05, sensor_ray_atol_m=.0005, continuous_controls=32,
                          term_height=.25, term_gravity_z=.8, update_learning_rate=1e-4,
                          update="29-parameter residual action, native Torch F.step and SGD",
                          control="reference actuator joint targets plus one constant residual vector")


def execute(args, context, report, out):
    import os
    import mujoco
    import torch
    from nablaengine import torch_ffi as F
    from nablaengine.g1_engine import tape
    from nablamimic.terrain_observation import ConvexTerrain, height_scan
    from check_pair_capacity import backward_chain
    import check_box_step_gradients as box_check
    model, foot_pairs = context['model'], context['foot_pairs']
    qpos, qvel, frames = context['qpos'], context['qvel'], context['frames']
    sensor_contract = json.loads((args.sensor/'contract.json').read_text())
    if sensor_contract['format'] != 'nablamimic-convex-terrain-v1':
        raise ValueError('Current convex-terrain sensor format required')
    if sensor_contract['artifact_id'] != context['contract']['terrain_artifact_id']:
        raise ValueError('Physical terrain and height sensor artifact identities differ')
    sensor_data = np.load(args.sensor/'terrain.npz', allow_pickle=False)
    device = 'cuda'
    def tensor(values):
        return torch.tensor(values, dtype=torch.float32, device=device).contiguous()
    terrain = ConvexTerrain(tensor(sensor_data['normals']), tensor(sensor_data['offsets']),
                            torch.tensor(sensor_data['valid_faces'], dtype=torch.bool, device=device))
    scan_offsets = torch.cartesian_prod(torch.linspace(-.8, 1.2, 11, device=device),
                                       torch.linspace(-.5, .5, 7, device=device))
    if (terrain.normals.ndim != 3 or terrain.normals.shape[-1] != 3
            or terrain.offsets.shape != terrain.normals.shape[:2]
            or terrain.valid_faces.shape != terrain.offsets.shape
            or not bool(torch.isfinite(terrain.normals).all() & torch.isfinite(terrain.offsets).all())
            or not bool(terrain.valid_faces.any(dim=-1).all())):
        raise ValueError('Malformed sensor halfspaces')
    normal_lengths = torch.linalg.vector_norm(terrain.normals[terrain.valid_faces], dim=-1)
    if not bool(torch.allclose(normal_lengths, torch.ones_like(normal_lengths), atol=1e-5, rtol=0)):
        raise ValueError('Sensor face normals require unit length')
    report['sensor_rays'] = []
    ray_data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, ray_data)
    offsets_numpy = scan_offsets.cpu().numpy().astype(float)
    for frame in frames:
        q = qpos[int(frame)]
        w, x, y, z = q[3:7]
        yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        c, s = np.cos(yaw), np.sin(yaw)
        origins = np.repeat(q[None, :3], len(offsets_numpy), axis=0)
        origins[:, 0] += c*offsets_numpy[:, 0]-s*offsets_numpy[:, 1]
        origins[:, 1] += s*offsets_numpy[:, 0]+c*offsets_numpy[:, 1]
        origins[:, 2] += 1.
        reference_distances = []
        for origin in origins:
            hits = [mujoco.mj_rayMesh(model, ray_data, geom, origin, np.array([0., 0., -1.]))
                    for geom in context['hull_geoms']]
            reference_distances.append(min([hit for hit in hits if hit >= 0]+[3.]))
        measured = height_scan(terrain, tensor(q[None, :3]), tensor(q[None, 3:7]), scan_offsets).cpu().numpy()[0]
        error = float(np.max(np.abs(measured-reference_distances)))
        report['sensor_rays'].append(dict(frame=int(frame), measured_m=measured.tolist(),
                                          mujoco_mesh_ray_reference_m=reference_distances,
                                          max_error_m=error, pass_=error <= EXECUTION_PROTOCOL['sensor_ray_atol_m']))
    model.opt.iterations = EXECUTION_PROTOCOL['iterations']
    if not np.isclose(model.opt.timestep*4, 1./context['fps'], atol=1e-12, rtol=0):
        raise ValueError('Reference control rate differs from four physics substeps')
    os.environ['DK_CACHE_DIR'] = str(out/'build')
    os.environ.pop('DK_TORCH_ARTIFACT', None)
    cc = torch.cuda.get_device_capability()
    artifact = F.build(f'sm_{cc[0]}{cc[1]}', model=model, foot_pairs=foot_pairs, set_current=False)
    F.init(artifact=artifact)
    report['execution_protocol'] = EXECUTION_PROTOCOL
    report['environment'] = dict(torch=torch.__version__, cuda=torch.version.cuda,
                                 gpu=torch.cuda.get_device_name(), mujoco=mujoco.__version__, numpy=np.__version__)
    report['artifact'] = str(artifact)
    report['executions'] = []
    joint_columns = [int(model.jnt_qposadr[int(j)]) for j in model.actuator_trnid[:, 0]]
    bad_mask = 31 | tape.FLAG_NONCONV | tape.FLAG_SINGULAR | tape.FLAG_VERIFY_DIFF
    def forward(frame, q, v, residual, controls, initial_warm=None):
        qt, vt = tensor(q[None]), tensor(v[None])
        warm = torch.zeros_like(vt) if initial_warm is None else tensor(initial_warm[None])
        friction, com = torch.ones(1, device=device), torch.zeros((1, 3), device=device)
        saved, decoded, trace = [], [], []
        valid = True
        for control in range(controls):
            target = qpos[frame+control, joint_columns]+residual
            result = F.forward(qt, vt, warm, tensor(target[None]), friction, com,
                               iterations=50, substeps=4, tape=True)
            qt, vt, warm, info = result[:4]
            saved.append(result[4])
            decoded_record = tape.decode(result[4].cpu().numpy().reshape(1, 4, tape.rec_words(model)), model)
            decoded.append(decoded_record)
            valid = (all(bool(torch.isfinite(value).all()) for value in (qt, vt, warm))
                     and not np.any(info.cpu().numpy()[:, 3]) and not np.any(decoded_record['flags'] & bad_mask))
            trace.append(dict(control=control, qpos=qt.cpu().numpy()[0].tolist(), qvel=vt.cpu().numpy()[0].tolist(),
                              info=info.cpu().numpy().tolist(),
                              tape={key: decoded_record[key].tolist() for key in ('ncon', 'nefc', 'flags', 'pair', 'slot')}))
            if not valid:
                break
        return dict(q=qt, v=vt, warm=warm, ctrl=tensor(residual[None]), friction=friction, com=com,
                    tapes=saved, decoded=decoded, valid=valid and len(saved) == controls, trace=trace)
    def physical(frame, q, v, residual, controls):
        data = mujoco.MjData(model)
        data.qpos[:], data.qvel[:] = q, v
        data.qacc_warmstart[:] = 0
        for control in range(controls):
            data.ctrl[:] = qpos[frame+control, joint_columns]+residual
            for _ in range(4):
                mujoco.mj_step(model, data)
        return np.r_[data.qpos.copy(), data.qvel.copy()]
    rng = np.random.default_rng(20261105)
    cq, cv = rng.normal(size=model.nq), rng.normal(size=model.nv)
    cq /= np.linalg.norm(cq)
    cv /= np.linalg.norm(cv)
    directions = []
    for block, width in (('qpos_tangent', model.nv), ('qvel', model.nv), ('ctrl', model.nu)):
        direction = rng.normal(size=width)
        direction /= np.linalg.norm(direction)
        directions.append(dict(block=block, index=0, direction=direction))
    report['execution_directions'] = [dict(d, direction=d['direction'].tolist()) for d in directions]
    report['terminal_cotangent'] = dict(qpos=cq.tolist(), qvel=cv.tolist())
    for frame in frames:
        frame = int(frame)
        for controls in EXECUTION_PROTOCOL['horizons']:
            row = dict(frame=frame, requested_controls=controls, requested_seconds=controls*4*float(model.opt.timestep),
                       pass_=False)
            report['executions'].append(row)
            if frame+controls >= len(qpos):
                row['failure'] = 'reference support does not cover requested controls'
                continue
            initial = dict(q=qpos[frame].copy(), v=qvel[frame].copy(), u=np.zeros(model.nu))
            nominal = forward(frame, initial['q'], initial['v'], initial['u'], controls)
            row.update(executed_controls=len(nominal['tapes']), trace=nominal['trace'], strict_valid=nominal['valid'])
            if not nominal['valid']:
                row['failure'] = 'strict engine flags or nonfinite trajectory'
                continue
            actual = np.r_[nominal['q'].cpu().numpy()[0], nominal['v'].cpu().numpy()[0]]
            reference = physical(frame, initial['q'], initial['v'], initial['u'], controls)
            row['qpos_reference_max_error'] = float(abs(actual[:model.nq]-reference[:model.nq]).max())
            row['qvel_reference_max_error'] = float(abs(actual[model.nq:]-reference[model.nq:]).max())
            row['physical_reference_ok'] = (box_check.close(actual[:model.nq], reference[:model.nq], EXECUTION_PROTOCOL['qpos_atol'], 0)
                                            and box_check.close(actual[model.nq:], reference[model.nq:], EXECUTION_PROTOCOL['qvel_atol'], 0))
            gq, gv, gu, flags = backward_chain(F, nominal, tensor(cq[None]), tensor(cv[None]), 4)
            gradient = np.r_[gq.cpu().numpy()[0], gv.cpu().numpy()[0], gu.cpu().numpy()[0]]
            row['backward_flags'] = flags
            row['backward_ok'] = bool(np.isfinite(gradient).all() and all(not (flag & (bad_mask | (3 << 12) | (15 << 16))) for flag in flags))
            row['directions'] = []
            for direction in directions:
                fds = []
                for step in EXECUTION_PROTOCOL['fd_steps']:
                    values = []
                    for sign in (-1, 1):
                        q, v, u = box_check.perturb(model, initial, direction, sign*step)
                        values.append(physical(frame, q, v, u, controls)@np.r_[cq, cv])
                    fds.append((values[1]-values[0])/(2*step))
                if direction['block'] == 'qpos_tangent':
                    qp, _, _ = box_check.perturb(model, initial, direction, 1e-6)
                    qm, _, _ = box_check.perturb(model, initial, direction, -1e-6)
                    analytic = float(gq.cpu().numpy()[0]@((qp-qm)/(2e-6)))
                else:
                    gradient_block = gv if direction['block'] == 'qvel' else gu
                    analytic = float(gradient_block.cpu().numpy()[0]@direction['direction'])
                stable = box_check.close(np.array(fds), fds[-1], EXECUTION_PROTOCOL['fd_stability_atol'], EXECUTION_PROTOCOL['fd_stability_rtol'])
                accurate = box_check.close(np.array(analytic), fds[-1], EXECUTION_PROTOCOL['gradient_atol'], EXECUTION_PROTOCOL['gradient_rtol'])
                row['directions'].append(dict(block=direction['block'], analytic=analytic, mujoco_double_fd=fds,
                                               fd_stable=stable, gradient_ok=accurate))
            row['pass_'] = row['physical_reference_ok'] and row['backward_ok'] and all(d['fd_stable'] and d['gradient_ok'] for d in row['directions'])
            np.savez_compressed(out/f'frame_{frame}_{controls}_controls.npz',
                                gradients=gradient, tapes=np.stack([t.cpu().numpy() for t in nominal['tapes']]))
        # One continuous physical state chain with an explicit termination reading.
        if frame+EXECUTION_PROTOCOL['continuous_controls'] < len(qpos):
            chain = []
            state_q, state_v = qpos[frame].copy(), qvel[frame].copy()
            state_warm = None
            for control in range(EXECUTION_PROTOCOL['continuous_controls']):
                result = forward(frame+control, state_q, state_v, np.zeros(model.nu), 1, state_warm)
                state_warm = result['warm'].cpu().numpy()[0]
                state_q, state_v = result['q'].cpu().numpy()[0], result['v'].cpu().numpy()[0]
                target = qpos[frame+control+1]
                gravity_z = 1-2*(state_q[4]**2+state_q[5]**2)
                target_gravity_z = 1-2*(target[4]**2+target[5]**2)
                terminated = bool(abs(state_q[2]-target[2]) > EXECUTION_PROTOCOL['term_height']
                                  or abs(gravity_z-target_gravity_z) > EXECUTION_PROTOCOL['term_gravity_z'])
                chain.append(dict(control=control, qpos=state_q.tolist(), qvel=state_v.tolist(), strict_valid=result['valid'],
                                  terminated=terminated, root_position_error_m=float(np.linalg.norm(state_q[:3]-target[:3])),
                                  contacts=result['trace'][0]['tape']))
                if not result['valid'] or terminated:
                    break
            report.setdefault('continuous_chains', []).append(dict(frame=frame, requested_controls=EXECUTION_PROTOCOL['continuous_controls'],
                                                                   executed_controls=len(chain), trajectory=chain))
        else:
            report.setdefault('continuous_chains', []).append(dict(frame=frame, requested_controls=EXECUTION_PROTOCOL['continuous_controls'],
                                                                   executed_controls=0, trajectory=[],
                                                                   failure='reference support does not cover chain'))
        # Native autograd and optimizer, reached only on a valid one-control prefix.
        update = dict(frame=frame, completed=False)
        report.setdefault('native_updates', []).append(update)
        if frame+1 >= len(qpos):
            update['failure'] = 'reference support does not cover one control'
            continue
        prefix = forward(frame, qpos[frame], qvel[frame], np.zeros(model.nu), 1)
        if not prefix['valid']:
            update['failure'] = 'strict-invalid one-control prefix'
            continue
        parameter = torch.nn.Parameter(torch.zeros((1, model.nu), device=device))
        optimizer = torch.optim.SGD([parameter], lr=EXECUTION_PROTOCOL['update_learning_rate'])
        target_q, target_v = tensor(qpos[frame+1][None]), tensor(qvel[frame+1][None])
        def loss():
            q, v, _ = F.step(tensor(qpos[frame][None]), tensor(qvel[frame][None]),
                             torch.zeros((1, model.nv), device=device),
                             tensor(qpos[frame, joint_columns][None])+parameter,
                             torch.ones(1, device=device), torch.zeros((1, 3), device=device),
                             iterations=50, substeps=4, clamp_kv=1, normal=0)
            scan = height_scan(terrain, q[:, :3], q[:, 3:7], scan_offsets)
            return ((q-target_q)**2).mean()+.01*((v-target_v)**2).mean(), scan
        F.read_diagnostics(reset=True)
        before, scan = loss()
        before.backward()
        gradient = parameter.grad.detach().clone()
        optimizer.step()
        after, after_scan = loss()
        diagnostics = F.read_diagnostics(reset=True)
        strict = all(diagnostics[key] == 0 for key in ('fwd_unsupported_worlds', 'nonfinite_rows', 'replay_mismatch_worlds',
                                                       'fwd_contact_overflow_worlds', 'fwd_row_overflow_worlds', 'nonconv_records', 'singular_records'))
        update.update(loss_before=float(before.detach()), loss_after=float(after.detach()),
                      gradient_norm=float(gradient.norm()), parameter_change_norm=float(parameter.detach().norm()),
                      scan_min_m=float(scan.min()), scan_max_m=float(scan.max()), diagnostics=diagnostics,
                      completed=bool(strict and torch.isfinite(gradient).all() and torch.isfinite(after)
                                     and torch.isfinite(scan).all() and torch.isfinite(after_scan).all()
                                     and parameter.detach().norm() > 0))
    report['continuous_chain_strict_valid'] = all(
        'failure' not in chain and all(step['strict_valid'] for step in chain['trajectory'])
        for chain in report.get('continuous_chains', []))
    report['continuous_chain_task_completion'] = [dict(frame=chain['frame'], complete=
        chain['executed_controls'] == chain['requested_controls']
        and not any(step['terminated'] for step in chain['trajectory']))
        for chain in report.get('continuous_chains', [])]
    report['execution_pass'] = (report['continuous_chain_strict_valid']
                                and all(ray['pass_'] for ray in report['sensor_rays'])
                                and all(row['pass_'] for row in report['executions'])
                                and all(update['completed'] for update in report['native_updates']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--motion', default='motion_07')
    parser.add_argument('--frames', default='all')
    parser.add_argument('--code-revision')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--execute', action='store_true', help='run GPU whole-step, chain and native optimizer on provisioned compute')
    parser.add_argument('--sensor', type=Path)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
        return 0
    if any(getattr(args, key) is None for key in ('scene', 'reference', 'code_revision', 'out')):
        parser.error('--scene, --reference, --code-revision and --out are required')
    if args.execute and (args.sensor is None or args.frames == 'all'):
        parser.error('--execute requires --sensor and explicit comma-separated --frames')
    out = args.out.resolve()
    if out.is_relative_to(ROOT):
        parser.error('--out must be outside checkout')
    out.mkdir(parents=True, exist_ok=False)
    report = dict(protocol=PROTOCOL, code_revision=args.code_revision, completed=False)
    try:
        context = screen(args, report, out)
        if args.execute:
            execute(args, context, report, out)
            return 0 if report['execution_pass'] else 1
        return 0
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    raise SystemExit(main())

"""Screen reference resets against the compiled physical convex terrain.

All geometry comes from the model's checked packed hull representation. The
reference is a current prepare_instinct_motion output directory. World transforms,
terrain identities and explicit foot-pair bindings are checked before screening.
No reference pose or joint is modified. Static geometry screening records every
selected frame, including centerline intersections and absent foot proximity.

Run full-reference screening on provisioned compute. Output requires a fresh
external directory. The small --self-test uses generated cube geometry only.
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
    return np.sqrt(np.maximum(squared, 0)), intersects


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
                  capability=validation.to_dict(), centerline_invalid_frames=int(intersects.any(axis=1).sum()),
                  foot_contact_candidate_frames=int(contact[:, foot].any(axis=1).sum()),
                  foot_near_support_frames=int(near[:, foot].any(axis=1).sum()), rows=[])
    for index, frame in enumerate(frames):
        report['rows'].append(dict(frame=int(frame), unsupported_pair_ids=[pairs[p][0] for p in np.flatnonzero(intersects[index])],
                                   capsule_contact_candidate_pair_ids=[pairs[p][0] for p in np.flatnonzero(contact[index])],
                                   closest_foot_surface_distance_m=float(distance[index, foot].min()),
                                   minimum_surface_distance_m=float(distance[index].min())))
    np.savez_compressed(out/'geometry.npz', frames=frames, pair_ids=np.array([p[0] for p in pairs]),
                        distance=distance, centerline_intersects=intersects, foot_pair_mask=foot,
                        qpos=qpos[frames], qvel=qvel[frames])
    report['completed'] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--motion', default='motion_07')
    parser.add_argument('--frames', default='all')
    parser.add_argument('--code-revision')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
        return 0
    if any(getattr(args, key) is None for key in ('scene', 'reference', 'code_revision', 'out')):
        parser.error('--scene, --reference, --code-revision and --out are required')
    out = args.out.resolve()
    if out.is_relative_to(ROOT):
        parser.error('--out must be outside checkout')
    out.mkdir(parents=True, exist_ok=False)
    report = dict(protocol=PROTOCOL, code_revision=args.code_revision, completed=False)
    try:
        screen(args, report, out)
        return 0
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    raise SystemExit(main())

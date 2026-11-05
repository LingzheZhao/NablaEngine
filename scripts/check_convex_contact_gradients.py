"""Static convex primitive qualification; run only on provisioned compute.

No engine dispatch, solver, mj_step, or rollout claim. Outputs (including errors)
are append-only records in a fresh external output root. Supply the original
CoACD hull_0.ply explicitly; it is never copied into the repository.
"""
import argparse
import json
import struct
import traceback
from pathlib import Path

import numpy as np

# Declared before launch; no thresholds are fitted to results.
PROTOCOL = {
    'scope': 'external centerline static convex primitive, eight outputs by six pose directions',
    'outputs': ['distance', 'position_xyz', 'normal_xyz', 'segment_parameter'],
    'coverage': ['generated_box', 'terrain_hull_0', 'sphere_face', 'sphere_edge',
                 'sphere_corner', 'sphere_radius_penetration', 'capsule_endpoint',
                 'capsule_interior', 'inside_rejected', 'intersection_rejected', 'touch_rejected', 'parallel_face_rejected'],
    'float64': {'geometry_tolerance': 2e-7, 'distance_atol': 2e-5,
                'fd_steps': [2e-5, 1e-5], 'jacobian_atol': 3e-4, 'jacobian_rtol': 3e-3},
    'float32': {'geometry_tolerance': 2e-5, 'distance_atol': 5e-5,
                'fd_steps': [2e-3, 1e-3], 'jacobian_atol': 1e-2, 'jacobian_rtol': 2e-2},
    'compiled_mesh_world_vertex_atol': 2e-5,
    'fd_stability_rule': 'same Jacobian margins as AD versus FD; unstable cases fail smooth coverage',
    'independent_oracle': 'MuJoCo mj_geomDistance on generated compiled mesh; distance only',
    'gpu_fd': 'reuse same-dtype CPU finite differences; require GPU/CPU value and Jacobian agreement',
    'time_support': 'single configuration; no integration',
    'unsupported': 'inside, intersecting or touching centerline, including sphere center on hull',
}


def load_ply(path):
    """Read the selected CoACD triangle PLY layout, not historical result formats."""
    with open(path, 'rb') as f:
        header = []
        while True:
            line = f.readline().decode('ascii').strip(); header.append(line)
            if line == 'end_header': break
            if not line: raise ValueError('incomplete PLY header')
        nv = int(next(x.split()[-1] for x in header if x.startswith('element vertex ')))
        nf = int(next(x.split()[-1] for x in header if x.startswith('element face ')))
        if 'format binary_little_endian 1.0' not in header:
            raise ValueError('expected selected binary little endian CoACD PLY')
        if [x for x in header if x.startswith('property ')] != [
            'property float x', 'property float y', 'property float z',
            'property list uchar int vertex_indices']:
            raise ValueError('unexpected CoACD PLY properties')
        v = np.frombuffer(f.read(12*nv), dtype='<f4').reshape(nv, 3).astype(float)
        faces = []
        for _ in range(nf):
            if f.read(1) != b'\x03': raise ValueError('nontriangle face')
            faces.append(struct.unpack('<3i', f.read(12)))
        if f.read(1): raise ValueError('unexpected PLY trailing bytes')
    return v, np.asarray(faces, dtype=np.int64)


def rotation(w):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(w).as_matrix()


def hulls(path):
    from scipy.spatial import ConvexHull
    box = np.array([[x, y, z] for x in [-.6, .6] for y in [-.4, .4] for z in [-.3, .3]])
    yield 'generated_box', box, ConvexHull(box).simplices
    v, f = load_ply(path)
    if (len(v), len(f)) != (46, 88): raise ValueError('expected terrain hull_0 with 46 vertices / 88 faces')
    yield 'terrain_hull_0', v, f


def cases(v, f):
    tri = v[f]; n = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    n /= np.linalg.norm(n, axis=1)[:, None]
    n *= np.where(np.sum((tri[:, 0]-v.mean(0))*n, axis=1) >= 0, 1., -1.)[:, None]
    # Largest face triangle provides a robust face-interior perturbation margin.
    area = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)
    fi = int(np.argmax(area)); qface = tri[fi].mean(0); nf = n[fi]
    adjacency = {}
    for i, face in enumerate(f):
        for j in range(3): adjacency.setdefault(tuple(sorted((int(face[j]), int(face[(j+1)%3])))), []).append(i)
    edges = [(edge, ids) for edge, ids in adjacency.items()
             if len(ids) == 2 and np.dot(n[ids[0]], n[ids[1]]) < .98]
    edge, ids = max(edges, key=lambda item: np.linalg.norm(v[item[0][0]]-v[item[0][1]]))
    ne = n[ids].sum(0); ne /= np.linalg.norm(ne); qe = v[list(edge)].mean(0)
    vi = int(np.argmax(np.linalg.norm(v-v.mean(0), axis=1)))
    nc = n[np.any(f == vi, axis=1)].sum(0); nc /= np.linalg.norm(nc)
    # Fixed metric gaps; original terrain scale/translation is retained.
    result = []
    def add(name, q, normal, hl=0., axis=None, radius=.04, reject=False, smooth=True):
        result.append(dict(name=name, center=np.asarray(q), axis=np.asarray(normal if axis is None else axis),
                           half_length=hl, radius=radius, reject=reject, smooth=smooth))
    for name, q, normal in [('face', qface, nf), ('edge', qe, ne), ('corner', v[vi], nc)]:
        add('sphere_'+name, q+normal*.08, normal)
    add('sphere_radius_penetration', qface+nf*.03, nf, radius=.05)
    # Closest endpoint has a strict gap and projects into the large face.
    add('capsule_endpoint', qface+nf*.13, nf, hl=.04)
    # A segment crossing the outward edge-normal bisector has a unique interior witness.
    tangent = np.cross(ne, v[edge[1]]-v[edge[0]]); tangent /= np.linalg.norm(tangent)
    add('capsule_interior', qe+ne*.08, ne, hl=.025, axis=tangent)
    add('inside_rejected', v.mean(0), nf, reject=True, smooth=False)
    add('intersection_rejected', v.mean(0), nf, hl=float(np.ptp(v, axis=0).max()*2), reject=True, smooth=False)
    add('touch_rejected', qface, nf, reject=True, smooth=False)
    face_tangent = tri[fi, 1]-tri[fi, 0]; face_tangent /= np.linalg.norm(face_tangent)
    add('parallel_face_rejected', qface+nf*.08, nf, hl=.025, axis=face_tangent, reject=True, smooth=False)
    return result


class MujocoOracle:
    def __init__(self, vertices, faces, case, *, explicit_pair=False):
        import mujoco
        from scipy.spatial.transform import Rotation
        self.mj = mujoco
        R = rotation(np.array([.23, -.31, .42])); p = np.array([.4, -.3, .2])
        self.R, self.p = R, p
        quat = Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]]
        fmt = lambda a: ' '.join(format(float(x), '.17g') for x in np.asarray(a).ravel())
        kind = 'sphere' if case['half_length'] == 0 else 'capsule'
        size = [case['radius']] if kind == 'sphere' else [case['radius'], case['half_length']]
        xml = ('<mujoco><option><flag multiccd="enable"/></option><asset>'
               f'<mesh name="h" vertex="{fmt(vertices)}" face="{fmt(faces)}"/></asset><worldbody>'
               f'<geom name="terrain" type="mesh" mesh="h" pos="{fmt(p)}" quat="{fmt(quat)}"/>'
               f'<body mocap="true"><geom name="robot" type="{kind}" size="{fmt(size)}"/></body>'
               '</worldbody>'
               + ('<contact><pair geom1="robot" geom2="terrain"/></contact>' if explicit_pair else '')
               + '</mujoco>')
        self.model = mujoco.MjModel.from_xml_string(xml); self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.h = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'terrain')
        self.robot = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'robot')
        mid = self.model.geom_dataid[self.h]
        a = self.model.mesh_vertadr[mid]; count = self.model.mesh_vertnum[mid]
        compiled = self.data.geom_xpos[self.h] + self.model.mesh_vert[a:a+count] @ self.data.geom_xmat[self.h].reshape(3, 3).T
        expected = p + vertices @ R.T
        err = float(np.linalg.norm(compiled[:, None]-expected[None, :], axis=2).min(1).max())
        err = max(err, float(np.linalg.norm(compiled[:, None]-expected[None, :], axis=2).min(0).max()))
        self.transform = dict(mesh_pos=self.model.mesh_pos[mid].tolist(), mesh_quat=self.model.mesh_quat[mid].tolist(),
                              geom_xpos=self.data.geom_xpos[self.h].tolist(), geom_xmat=self.data.geom_xmat[self.h].tolist(),
                              world_vertex_max_error=err)
        if err > PROTOCOL['compiled_mesh_world_vertex_atol']: raise ValueError(f'compiled mesh frame mismatch {err}')

    def distance(self, case):
        from scipy.spatial.transform import Rotation
        axis = self.R @ case['axis']; z = np.array([0., 0., 1.])
        cross = np.cross(z, axis); dot = np.dot(z, axis)
        if dot < -1+1e-12: rot = rotation(np.array([np.pi, 0., 0.]))
        else:
            K = np.array([[0., -cross[2], cross[1]], [cross[2], 0., -cross[0]], [-cross[1], cross[0], 0.]])
            rot = np.eye(3)+K+K@K/(1+dot)
        self.data.mocap_pos[0] = self.p+self.R@case['center']
        self.data.mocap_quat[0] = Rotation.from_matrix(rot).as_quat()[[3, 0, 1, 2]]
        self.mj.mj_forward(self.model, self.data)
        witness = np.zeros(6)
        d = self.mj.mj_geomDistance(self.model, self.data, self.robot, self.h, 1000., witness)
        return float(d), witness.tolist()


def evaluate(v, f, case, device, dtype, out, cpu_reference=None):
    import torch
    from nablaengine.convex_contact import capsule_contact, UnsupportedContact
    spec = PROTOCOL[dtype]
    tensor = lambda x: torch.as_tensor(x, device=device, dtype=getattr(torch, dtype))
    vt, ft = tensor(v), torch.as_tensor(f.copy(), device=device, dtype=torch.long)
    cp, ax = tensor(case['center']), tensor(case['axis'])
    def fn(pose):
        w = pose[3:]; zero = w[0]*0
        K = torch.stack([zero, -w[2], w[1], w[2], zero, -w[0], -w[1], w[0], zero]).reshape(3, 3)
        axis = torch.matrix_exp(K) @ ax
        c = capsule_contact(cp+pose[:3], axis, case['half_length'], case['radius'], vt, ft,
                            tolerance=spec['geometry_tolerance'])
        return torch.cat([c.distance.reshape(1), c.position, c.normal, c.segment_parameter.reshape(1)])
    pose = tensor(np.zeros(6)).requires_grad_()
    row = dict(device=device, dtype=dtype, case=case['name'], input={k: (x.tolist() if isinstance(x, np.ndarray) else x) for k, x in case.items()})
    try:
        value = fn(pose)
    except UnsupportedContact as exc:
        row.update(rejected=True, passed=case['reject'], reason=str(exc)); return row
    row.update(rejected=False, value=value.detach().cpu().double().tolist())
    if case['reject']:
        row.update(passed=False, reason='unsupported centerline accepted'); return row
    oracle = MujocoOracle(v, f, case)
    distance, witness = oracle.distance(case)
    row.update(oracle_distance=distance, oracle_witness=witness, compiled_mesh_transform=oracle.transform)
    row['distance_error'] = abs(row['value'][0]-distance)
    row['distance_passed'] = row['distance_error'] <= spec['distance_atol']
    jac = torch.autograd.functional.jacobian(fn, pose).detach().cpu().double().numpy()
    fds = []
    if cpu_reference is not None:
        fds = [np.asarray(x) for x in cpu_reference['finite_differences']]
        row['cpu_value_max_error'] = float(np.abs(np.asarray(row['value'])-cpu_reference['value']).max())
        row['cpu_jacobian_max_error'] = float(np.abs(jac-np.asarray(cpu_reference['jacobian'])).max())
        row['cpu_agreement'] = bool(np.allclose(row['value'], cpu_reference['value'], atol=spec['distance_atol'], rtol=spec['jacobian_rtol']) and np.allclose(jac, cpu_reference['jacobian'], atol=spec['jacobian_atol'], rtol=spec['jacobian_rtol']))
    with torch.no_grad():
        for step in ([] if cpu_reference is not None else spec['fd_steps']):
            cols = []
            for i in range(6):
                delta = tensor(np.eye(6)[i]*step)
                cols.append(((fn(delta)-fn(-delta))/(2*step)).cpu().double().numpy())
            fds.append(np.stack(cols, axis=1))
    close = lambda a, b: bool(np.allclose(a, b, atol=spec['jacobian_atol'], rtol=spec['jacobian_rtol']))
    row.update(jacobian=jac.tolist(), finite_differences=[x.tolist() for x in fds],
               fd_stable=close(*fds), gradient_passed=all(close(jac, fd) for fd in fds),
               jacobian_max_error=max(float(np.abs(jac-fd).max()) for fd in fds))
    row['passed'] = row['distance_passed'] and row['fd_stable'] and row['gradient_passed'] and row.get('cpu_agreement', True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hull', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--code-revision', required=True)
    parser.add_argument('--devices', nargs='+', choices=['cpu', 'cuda'], default=['cpu', 'cuda'])
    parser.add_argument('--dtypes', nargs='+', choices=['float32', 'float64'], default=['float32', 'float64'])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'protocol.json').write_text(json.dumps(dict(PROTOCOL, code_revision=args.code_revision,
        hull=str(args.hull), devices=args.devices, dtypes=args.dtypes), indent=2)+'\n')
    import torch, mujoco
    (args.out/'environment.json').write_text(json.dumps(dict(torch=torch.__version__, mujoco=mujoco.__version__,
        cuda_available=torch.cuda.is_available()), indent=2)+'\n')
    rows = []; cpu_cache = {}
    with (args.out/'conditions.jsonl').open('x') as stream:
        try:
            for name, v, f in hulls(args.hull):
                for device in sorted(args.devices, key=lambda x: x != 'cpu'):
                    for dtype in args.dtypes:
                        for case in cases(v, f):
                            try:
                                reference = cpu_cache.get((name, dtype, case['name']))
                                if device == 'cuda' and not case['reject'] and reference is None:
                                    reference = evaluate(v, f, case, 'cpu', dtype, args.out)
                                    stream.write(json.dumps(dict(reference, hull=name, supporting_cpu_reference=True))+'\n'); stream.flush()
                                row = evaluate(v, f, case, device, dtype, args.out, reference if device == 'cuda' and not case['reject'] else None)
                                if device == 'cpu' and 'finite_differences' in row: cpu_cache[name, dtype, case['name']] = row
                            except Exception:
                                row = dict(device=device, dtype=dtype, case=case['name'], passed=False,
                                           error=traceback.format_exc())
                            row['hull'] = name; rows.append(row)
                            stream.write(json.dumps(row)+'\n'); stream.flush()
                            print(name, device, dtype, case['name'], row['passed'], flush=True)
        except Exception:
            stream.write(json.dumps(dict(passed=False, setup_error=traceback.format_exc()))+'\n'); stream.flush()
            raise
    result = dict(passed=bool(rows) and all(x['passed'] for x in rows), conditions=len(rows),
                  failed=sum(not x['passed'] for x in rows), scope=PROTOCOL['scope'])
    (args.out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__': main()

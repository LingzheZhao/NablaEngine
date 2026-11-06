"""Check positive-volume thin facets with generated geometry and optional selected model.

CPU packing and float32 host arithmetic only. No stepping, collision queries or
rendering. CUDA primal and derivative qualification remain separate checks.
"""
import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from nablaengine.model.convex import pack_static_convex_hulls


def generated_model():
    epsilon = np.finfo(np.float32).eps
    # A cube with one exposed edge midpoint moved outwards and downwards by
    # one float32 ULP. The convex solid remains well sized and gains a thin face.
    vertices = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)]
                        + [[1.+epsilon, 0., 1.-epsilon]], dtype=np.float32)
    faces = ConvexHull(vertices).simplices
    spec = mujoco.MjSpec()
    spec.add_mesh(name='generated', uservert=vertices.ravel().tolist(), userface=faces.ravel().tolist())
    spec.worldbody.add_geom(name='generated', type=mujoco.mjtGeom.mjGEOM_MESH,
                            meshname='generated', contype=0, conaffinity=0, mass=0)
    return spec.compile()


def check(model, geom):
    packed = {name: value for name, _, value in pack_static_convex_hulls(model, [geom]).arrays}
    points, faces = packed['c_hull_vertices'], packed['c_hull_faces']
    normals = packed['c_hull_normals']
    normals = normals / np.linalg.norm(normals, axis=1)[:, None]
    thin = np.flatnonzero(packed['c_hull_face_thin'])
    if not len(thin):
        raise ValueError('Selected math check requires at least one thin facet')
    probes = []
    for face in thin:
        triangle = points[faces[face]]
        normal = normals[face]
        center = triangle.mean(0)
        tangent = triangle[1]-triangle[0]
        tangent /= np.linalg.norm(tangent)
        for label, expected, point in (
                ('face_interior', True, center+normal*np.float32(.07)),
                ('outside_body', False, center+np.float32(10)*tangent+normal*np.float32(.07))):
            projection = point-normal*np.dot(normal, point-triangle[0])
            excess = np.sum(normals*(projection-points[faces[:, 0]]), axis=1)
            actual = bool(excess.max() <= 2e-5)
            if actual != expected:
                raise AssertionError(f'Thin face {face} {label} classification differs')
            probes.append(dict(face=int(face), case=label, inside=actual, maximum_halfspace_excess_m=float(excess.max())))
    mesh = int(model.geom_dataid[geom])
    start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
    original = np.asarray(model.mesh_vert[start:start+count], dtype=float)
    restored = points.astype(float)+packed['c_hull_origin'][0].astype(float)
    before, after = ConvexHull(original), ConvexHull(restored)
    if len(points) != len(original) or not np.array_equal(faces, before.simplices):
        raise AssertionError('Packing changed the compiled vertices or canonical triangle indices')
    directions = np.random.default_rng(20261106).normal(size=(256, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    support_error = float(abs((original@directions.T).max(0)-(restored@directions.T).max(0)).max())
    if support_error > 2e-5:
        raise AssertionError('Packed support values exceed the existing geometry tolerance')
    return dict(vertices=len(points), faces=len(faces), thin_faces=len(thin),
                original_volume_m3=float(before.volume), packed_volume_m3=float(after.volume),
                support_error_m=support_error, probes=probes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, help='Optional explicitly selected model for additional diagnostics')
    parser.add_argument('--geom', help='Named mesh geom in the optional model')
    parser.add_argument('--code-revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if bool(args.model) != bool(args.geom):
        parser.error('--model and --geom must be supplied together')
    if args.model:
        args.model = args.model.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.chdir(args.output)
    report = dict(code_revision=args.code_revision, scope=__doc__, generated=check(generated_model(), 0))
    if args.model:
        model = mujoco.MjModel.from_binary_path(str(args.model))
        report['selected_model'] = dict(path=str(args.model), geom=args.geom,
                                        **check(model, model.geom(args.geom).id))
    (args.output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()

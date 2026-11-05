"""Measure every selected compiled hull without changing its triangles or vertices.

Run on provisioned CPU compute. Consumes the scene producer's construction.json
and constructed_model.mjb, including a construction whose numerical gate failed.
Reports volume, supporting planes, duplicate vertices, thin triangle sensitivity,
packed-coordinate rounding and original-CoACD geometry agreement. No mj_forward,
mj_collision, stepping, face deletion or tolerance relaxation occurs.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True
import mujoco
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
import trimesh

PROTOCOL = dict(seed=20261106, support_directions=256, geometry_tolerance_m=2e-5,
                thin_face_rule='cross_length <= float32_epsilon * longest_edge_squared',
                scope='unchanged source PLY, compiled mesh, and existing packed-coordinate representation')


def readings(points):
    hull = ConvexHull(points)
    triangles = points[hull.simplices]
    edges = triangles[:, [1, 2, 0]] - triangles
    crosses = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    lengths = np.linalg.norm(crosses, axis=1)
    scale_squared = np.max(np.sum(edges**2, axis=2), axis=1)
    thin = lengths <= np.finfo(np.float32).eps*scale_squared
    normals = crosses / np.where(lengths > 0, lengths, 1)[:, None]
    normals *= np.where(np.sum(normals*hull.equations[:, :3], axis=1) >= 0, 1., -1.)[:, None]
    supporting = points @ hull.equations[:, :3].T + hull.equations[:, 3]
    summary = dict(vertices=len(points), unique_vertices=len(np.unique(points, axis=0)),
                   hull_vertices=len(hull.vertices), triangles=len(hull.simplices),
                   volume_m3=float(hull.volume), area_m2=float(hull.area), axis_spans_m=np.ptp(points, axis=0).tolist(),
                   min_cross_length_m2=float(lengths.min()), min_relative_cross_length=float((lengths/scale_squared).min()),
                   thin_triangles=int(thin.sum()), zero_area_triangles=int((lengths == 0).sum()),
                   support_plane_max_excess_m=float(supporting.max()),
                   cross_normal_vs_qhull_normal_max_error=float(np.linalg.norm(normals-hull.equations[:, :3], axis=1).max()))
    return summary, hull, thin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--mesh', type=int, action='append', help='Explicit selected mesh IDs, default all terrain hulls')
    parser.add_argument('--code-revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.scene, args.output = args.scene.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.chdir(args.output)
    report = dict(state='started', code_revision=args.code_revision, scene=str(args.scene),
                  protocol=PROTOCOL, meshes=[])

    def write_report():
        (args.output / 'report.json').write_text(json.dumps(report, indent=2))

    def fail(kind, value, tb):
        report.update(state='failed', failure_type=kind.__name__, failure=str(value),
                      traceback=''.join(traceback.format_exception(kind, value, tb)))
        write_report()
        sys.__excepthook__(kind, value, tb)

    write_report()
    sys.excepthook = fail
    construction = json.loads((args.scene / 'construction.json').read_text())
    if construction['format'] != 'nablaengine-terrain-construction-v1':
        raise ValueError('Expected the scene producer construction record')
    model = mujoco.MjModel.from_binary_path(str(args.scene / 'constructed_model.mjb'))
    report.update(artifact_id=construction['artifact_id'], model_counts=construction['counts'])
    directions = np.random.default_rng(PROTOCOL['seed']).normal(size=(PROTOCOL['support_directions'], 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    selected = set(args.mesh) if args.mesh is not None else None
    seen = set()
    for terrain in construction['terrains']:
        for index, name in enumerate(terrain['hull_geom_names']):
            geom = model.geom(name).id
            mesh = int(model.geom_dataid[geom])
            if selected is not None and mesh not in selected:
                continue
            seen.add(mesh)
            row = dict(mesh=mesh, geom=geom, geom_name=name, terrain_index=terrain['terrain_index'], hull_index=index,
                       source_ply=str(Path(terrain['coacd']) / f'hull_{index}.ply'))
            report['meshes'].append(row)
            try:
                start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
                points = np.asarray(model.mesh_vert[start:start+count], dtype=float)
                compiled, hull, thin = readings(points)
                row['compiled'] = compiled
                origin = points.mean(0)
                packed = (points-origin).astype(np.float32)
                packed_reading, packed_hull, packed_thin = readings(packed.astype(float))
                row['packed'] = packed_reading
                reconstructed = packed.astype(float) + origin.astype(np.float32).astype(float)
                row['packed_vertex_roundtrip_max_error_m'] = float(np.linalg.norm(reconstructed-points, axis=1).max())
                raw_mesh = trimesh.load(row['source_ply'], process=False)
                raw = np.asarray(raw_mesh.vertices, dtype=float)
                raw_reading, raw_hull, raw_thin = readings(raw)
                row['source'] = raw_reading
                rotation = np.empty(9)
                mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
                world = points @ rotation.reshape(3, 3).T + model.geom_pos[geom]
                source_world = raw + terrain['translation_m']
                row['source_world_vertex_max_error_m'] = float(max(
                    cKDTree(world).query(source_world)[0].max(), cKDTree(source_world).query(world)[0].max()))
                row['source_world_support_max_error_m'] = float(abs((world @ directions.T).max(0) - (source_world @ directions.T).max(0)).max())
                row['compiled_volume_minus_source_m3'] = compiled['volume_m3'] - raw_reading['volume_m3']
                row['packed_volume_minus_compiled_m3'] = packed_reading['volume_m3'] - compiled['volume_m3']
                faces = hull.simplices
                triangle32 = packed[faces]
                crosses32 = np.cross(triangle32[:, 1]-triangle32[:, 0], triangle32[:, 2]-triangle32[:, 0])
                row['existing_face_float32_zero_crosses'] = int((np.linalg.norm(crosses32.astype(float), axis=1) == 0).sum())
                row['thin_face_indices'] = np.flatnonzero(thin).tolist()
                if thin.any() or packed_thin.any():
                    evidence = args.output / f'mesh_{mesh}_triangles.npz'
                    np.savez_compressed(evidence, compiled_vertices=points, source_vertices=raw,
                                        faces=faces, qhull_equations=hull.equations,
                                        packed_vertices=packed, packed_faces=packed_hull.simplices,
                                        packed_qhull_equations=packed_hull.equations,
                                        thin_faces=np.flatnonzero(thin), packed_thin_faces=np.flatnonzero(packed_thin),
                                        float32_crosses=crosses32)
                    row['triangle_evidence'] = str(evidence)
                row['state'] = 'measured'
            except Exception as error:
                row.update(state='failed', failure_type=type(error).__name__, failure=str(error))
            write_report()
    if selected is not None and seen != selected:
        raise ValueError(f'Selected mesh IDs absent from terrain construction: {sorted(selected-seen)}')
    report.update(state='completed', mesh_count=len(report['meshes']),
                  failed_meshes=[row['mesh'] for row in report['meshes'] if row['state'] == 'failed'],
                  thin_meshes=[row['mesh'] for row in report['meshes'] if row.get('compiled', {}).get('thin_triangles', 0)])
    write_report()
    if report['failed_meshes']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

"""Pack compiled static convex meshes for the engine's collision primitives.

Vertices are in MuJoCo's compiled mesh frame. The compiled geom pose already
contains the mesh alignment and mass-centering transform. Applying mesh_pos or
mesh_quat here a second time would move the physical terrain.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PackedConvexHulls:
    arrays: tuple

    def constants(self):
        """Values in the existing host loader's float32 transport format."""
        return np.concatenate([values.ravel() for _, _, values in self.arrays]).astype(np.float32)

    def layout(self):
        """Existing host loader layout entries, with concrete element counts."""
        return [(name, kind, values.size, 1) for name, kind, values in self.arrays]

    def header(self):
        declarations = []
        for name, kind, values in self.arrays:
            scalar = 'int' if kind == 'i' else 'float'
            declarations.append(f'DKC {scalar} {name}[{values.size}];')
        return '\n'.join(declarations) + '\n'


def pack_static_convex_hulls(model, collision_geoms):
    """Validate and pack only mesh assets selected by explicit collision pairs.

    The input geom order is the engine collision-geom table order. Every selected
    mesh must be fixed to world body zero and bound a closed convex triangle
    surface. Triangulation is rebuilt from every compiled vertex because mesh
    centering/rotation can change the triangulation of nearly coplanar faces.
    Indices fit exactly in the float32 host transport representation.
    """
    import mujoco
    from scipy.spatial import ConvexHull

    geom_hulls = np.full(len(collision_geoms), -1, dtype=np.int32)
    vertex_addresses, vertex_counts, face_addresses, face_counts = [], [], [], []
    vertices, faces, normals, origins = [], [], [], []
    mesh_indices = {}
    vertex_count, face_count = 0, 0
    for local_geom, geom in enumerate(collision_geoms):
        if int(model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        if int(model.geom_bodyid[geom]) != 0:
            raise ValueError(f'mesh geom {geom} must be fixed to world body zero')
        mesh = int(model.geom_dataid[geom])
        if not 0 <= mesh < model.nmesh:
            raise ValueError(f'mesh geom {geom} has invalid mesh index {mesh}')
        if mesh in mesh_indices:
            geom_hulls[local_geom] = mesh_indices[mesh]
            continue
        vertex_address = int(model.mesh_vertadr[mesh])
        count = int(model.mesh_vertnum[mesh])
        points = np.asarray(model.mesh_vert[vertex_address:vertex_address+count], dtype=np.float64)
        if count < 4 or not np.isfinite(points).all():
            raise ValueError(f'mesh {mesh} requires finite closed solid geometry')
        # MuJoCo's convex mesh support map uses all compiled vertices. Rebuild
        # that same convex body after compilation, keeping all input vertices.
        # Reusing a pre-compilation near-coplanar triangulation can produce
        # nonconvex thin faces after the compiler's float32 frame transform.
        indices = np.asarray(ConvexHull(points).simplices, dtype=np.int32)
        triangles = len(indices)
        if indices.shape != (triangles, 3) or indices.min() < 0 or indices.max() >= count:
            raise ValueError(f'mesh {mesh} has invalid triangle indices')
        edge_counts = {}
        for triangle in indices:
            for i in range(3):
                edge = tuple(sorted((int(triangle[i]), int(triangle[(i+1) % 3]))))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
        if any(value != 2 for value in edge_counts.values()):
            raise ValueError(f'mesh {mesh} must have closed two-manifold triangle topology')
        origin = points.mean(axis=0)
        points = points-origin
        triangle_points = points[indices]
        cross = np.cross(triangle_points[:, 1]-triangle_points[:, 0], triangle_points[:, 2]-triangle_points[:, 0])
        lengths = np.linalg.norm(cross, axis=1)
        edge_scale = np.max(np.stack([np.linalg.norm(triangle_points[:, (i+1) % 3]-triangle_points[:, i], axis=1)
                                     for i in range(3)]), axis=0)
        if np.any(lengths <= np.finfo(np.float32).eps*edge_scale**2):
            raise ValueError(f'mesh {mesh} has unresolved degenerate triangles')
        face_normals = cross/lengths[:, None]
        face_normals *= np.where(np.sum(triangle_points[:, 0]*face_normals, axis=1) >= 0, 1., -1.)[:, None]
        offsets = np.sum((points[None]-triangle_points[:, 0, None])*face_normals[:, None], axis=2)
        if np.max(offsets) > 2e-5:
            raise ValueError(f'mesh {mesh} is not convex within the declared 2e-5 meter geometry tolerance')
        hull = len(vertex_counts)
        mesh_indices[mesh] = hull
        geom_hulls[local_geom] = hull
        vertex_addresses.append(vertex_count)
        vertex_counts.append(count)
        face_addresses.append(face_count)
        face_counts.append(triangles)
        vertices.append(points.astype(np.float32))
        faces.append(indices)
        normals.append(face_normals.astype(np.float32))
        origins.append(origin.astype(np.float32))
        vertex_count += count
        face_count += triangles
    if max(vertex_count, face_count, len(vertex_counts)) >= 2**24:
        raise ValueError('convex mesh indices exceed exact float32 host transport range')
    # A no-mesh model still gets valid nonzero C array declarations. Its geom map
    # contains only -1, so none of these inert entries can be consumed.
    packed = (
        ('c_ghull', 'i', geom_hulls if len(geom_hulls) else np.array([-1], np.int32)),
        ('c_hull_vertexadr', 'i', np.asarray(vertex_addresses or [0], np.int32)),
        ('c_hull_vertexnum', 'i', np.asarray(vertex_counts or [0], np.int32)),
        ('c_hull_faceadr', 'i', np.asarray(face_addresses or [0], np.int32)),
        ('c_hull_facenum', 'i', np.asarray(face_counts or [0], np.int32)),
        ('c_hull_vertices', 'f', np.concatenate(vertices) if vertices else np.zeros((1, 3), np.float32)),
        ('c_hull_faces', 'i', np.concatenate(faces) if faces else np.zeros((1, 3), np.int32)),
        ('c_hull_normals', 'f', np.concatenate(normals) if normals else np.zeros((1, 3), np.float32)),
        ('c_hull_origin', 'f', np.asarray(origins, np.float32) if origins else np.zeros((1, 3), np.float32)),
    )
    return PackedConvexHulls(packed)

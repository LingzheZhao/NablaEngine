"""First-hit depth rays against a fixed model's visible geometry.

Mesh triangles retain their compiled representation and geom frame. Dynamic
robot geoms use the supplied body poses, including self occlusion. Images are
stop-gradient observations. Camera rays, image-plane conversion and image
processing belong to the camera consumer.
"""
from dataclasses import dataclass

import numpy as np
import torch


def _rotate(quaternion, vector):
    qvector = quaternion[..., 1:]
    twice_cross = 2*torch.linalg.cross(qvector, vector)
    return vector+quaternion[..., :1]*twice_cross+torch.linalg.cross(qvector, twice_cross)


def _sphere(origins, directions, radius):
    projection = (origins*directions).sum(-1)
    discriminant = projection.square()-(origins.square().sum(-1)-radius*radius)
    root = discriminant.clamp_min(0).sqrt()
    near, far = -projection-root, -projection+root
    distance = torch.where(near >= 0, near, far)
    return torch.where((discriminant >= 0) & (distance >= 0), distance, torch.inf)


def _capsule(origins, directions, radius, half_length):
    radial_speed = directions[..., :2].square().sum(-1)
    projection = (origins[..., :2]*directions[..., :2]).sum(-1)
    constant = origins[..., :2].square().sum(-1)-radius*radius
    discriminant = projection.square()-radial_speed*constant
    root = discriminant.clamp_min(0).sqrt()
    denominator = radial_speed.clamp_min(torch.finfo(origins.dtype).tiny)
    result = torch.full_like(projection, torch.inf)
    for distance in ((-projection-root)/denominator, (-projection+root)/denominator):
        height = origins[..., 2]+distance*directions[..., 2]
        valid = (radial_speed > 0) & (discriminant >= 0) & (distance >= 0) & (height.abs() <= half_length)
        result = torch.minimum(result, torch.where(valid, distance, torch.inf))
    for sign in (-1, 1):
        shifted = origins.clone()
        shifted[..., 2] -= sign*half_length
        projection = (shifted*directions).sum(-1)
        discriminant = projection.square()-(shifted.square().sum(-1)-radius*radius)
        root = discriminant.clamp_min(0).sqrt()
        for distance in (-projection-root, -projection+root):
            height = origins[..., 2]+distance*directions[..., 2]
            valid = (discriminant >= 0) & (distance >= 0) & (sign*height >= half_length)
            result = torch.minimum(result, torch.where(valid, distance, torch.inf))
    return result


def _triangles(origins, directions, triangles, triangle_chunk):
    result = torch.full(origins.shape[:-1], torch.inf, device=origins.device, dtype=origins.dtype)
    for start in range(0, len(triangles), triangle_chunk):
        tri = triangles[start:start+triangle_chunk]
        edge1, edge2 = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
        cross = torch.linalg.cross(directions[..., None, :], edge2[None, None])
        determinant = (edge1*cross).sum(-1)
        valid_det = determinant.abs() > torch.finfo(origins.dtype).eps*edge1.norm(dim=-1)*edge2.norm(dim=-1)
        inverse = torch.where(valid_det, determinant.reciprocal(), 0.)
        offset = origins[..., None, :]-tri[:, 0]
        u = (offset*cross).sum(-1)*inverse
        q = torch.linalg.cross(offset, edge1[None, None])
        v = (directions[..., None, :]*q).sum(-1)*inverse
        distance = (edge2*q).sum(-1)*inverse
        valid = valid_det & (u >= 0) & (v >= 0) & (u+v <= 1) & (distance >= 0)
        result = torch.minimum(result, torch.where(valid, distance, torch.inf).amin(-1))
    return result


@dataclass(frozen=True)
class VisibleGeom:
    geom_id: int
    body_id: int
    kind: int
    position: torch.Tensor
    quaternion: torch.Tensor
    size: torch.Tensor
    triangles: torch.Tensor | None


@dataclass(frozen=True)
class DepthGeometry:
    """Model-owned visible geometry packed once on the observation device."""
    geoms: tuple[VisibleGeom, ...]
    body_count: int

    @classmethod
    def from_model(cls, model, *, device, dtype=torch.float32, groups=(0, 2)):
        import mujoco

        selected = set(groups)
        supported = {int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                     int(mujoco.mjtGeom.mjGEOM_MESH), int(mujoco.mjtGeom.mjGEOM_BOX),
                     int(mujoco.mjtGeom.mjGEOM_PLANE)}
        geoms = []
        def tensor(value):
            return torch.tensor(np.array(value, copy=True), device=device, dtype=dtype)
        for geom in range(model.ngeom):
            if int(model.geom_group[geom]) not in selected or model.geom_rgba[geom, 3] == 0:
                continue
            kind = int(model.geom_type[geom])
            if kind not in supported:
                raise ValueError(f'Visible geom {geom} has unsupported ray geometry type {kind}')
            triangles = None
            if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
                mesh = int(model.geom_dataid[geom])
                va, vn = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
                fa, fn = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
                vertices = np.asarray(model.mesh_vert[va:va+vn])
                faces = np.asarray(model.mesh_face[fa:fa+fn])
                triangles = tensor(vertices[faces])
            geoms.append(VisibleGeom(geom, int(model.geom_bodyid[geom]), kind,
                                     tensor(model.geom_pos[geom]), tensor(model.geom_quat[geom]),
                                     tensor(model.geom_size[geom]), triangles))
        return cls(tuple(geoms), int(model.nbody))

    @torch.no_grad()
    def raycast(self, body_positions, body_quaternions, ray_origins, ray_directions,
                *, max_distance=1e6, min_distance=.05, max_hops=24,
                surface_epsilon=1e-5, ray_chunk=256, triangle_chunk=128):
        """Return metric unit-ray range and model geom ID, with -1 for misses.

        Inputs have shapes [B,NB,3], [B,NB,4] (WXYZ), [B,P,3], [B,P,3].
        Hits closer than min_distance are skipped by advancing the ray beyond
        that surface and continuing, retaining source camera near-hit behavior.
        """
        if (body_positions.shape[1:] != (self.body_count, 3)
                or body_quaternions.shape != (*body_positions.shape[:2], 4)
                or ray_origins.shape != ray_directions.shape or ray_origins.ndim != 3
                or ray_origins.shape[-1] != 3 or ray_origins.shape[0] != body_positions.shape[0]):
            raise ValueError('Body poses and rays require matching batch and model dimensions')
        ranges, geom_ids = [], []
        for first in range(0, ray_origins.shape[1], ray_chunk):
            origin = ray_origins[:, first:first+ray_chunk]
            direction = ray_directions[:, first:first+ray_chunk]
            distance, geom_id = self._first_hit(body_positions, body_quaternions, origin, direction,
                                                  triangle_chunk=triangle_chunk)
            traveled = torch.zeros_like(distance)
            for _ in range(max_hops):
                skip = (distance+traveled < min_distance) & torch.isfinite(distance)
                if not bool(skip.any()):
                    break
                advance = torch.where(skip, distance+surface_epsilon, 0.)
                traveled += advance
                origin = origin+direction*advance[..., None]
                next_distance, next_geom = self._first_hit(body_positions, body_quaternions, origin, direction,
                                                            triangle_chunk=triangle_chunk)
                distance = torch.where(skip, next_distance, distance)
                geom_id = torch.where(skip, next_geom, geom_id)
            total = distance+traveled
            valid = (total >= min_distance) & (total <= max_distance) & torch.isfinite(total)
            ranges.append(torch.where(valid, total, max_distance))
            geom_ids.append(torch.where(valid, geom_id, -1))
        return torch.cat(ranges, dim=1), torch.cat(geom_ids, dim=1)

    def _first_hit(self, body_positions, body_quaternions, origins, directions, *, triangle_chunk=128):
        nearest = torch.full(origins.shape[:-1], torch.inf, device=origins.device, dtype=origins.dtype)
        ids = torch.full(origins.shape[:-1], -1, device=origins.device, dtype=torch.long)
        for geom in self.geoms:
            bp = body_positions[:, geom.body_id, None]
            bq = body_quaternions[:, geom.body_id, None]
            # Compose ray transformations in two stages to avoid duplicating
            # model poses and to retain MuJoCo's compiled mesh geom frame.
            inverse_body = torch.cat([bq[..., :1], -bq[..., 1:]], dim=-1)
            inverse_body = inverse_body.expand(*origins.shape[:-1], 4)
            local_origin = _rotate(inverse_body, origins-bp)
            local_direction = _rotate(inverse_body, directions)
            inverse_geom = torch.cat([geom.quaternion[:1], -geom.quaternion[1:]])
            inverse_geom = inverse_geom.expand(*origins.shape[:-1], 4)
            local_origin = _rotate(inverse_geom, local_origin-geom.position)
            local_direction = _rotate(inverse_geom, local_direction)
            if geom.kind == 2:
                distance = _sphere(local_origin, local_direction, geom.size[0])
            elif geom.kind == 3:
                distance = _capsule(local_origin, local_direction, geom.size[0], geom.size[1])
            elif geom.kind == 7:
                distance = _triangles(local_origin, local_direction, geom.triangles, triangle_chunk)
            elif geom.kind == 6:
                parallel = local_direction == 0
                reciprocal = torch.where(parallel, torch.inf, local_direction.reciprocal())
                a, b = (-geom.size-local_origin)*reciprocal, (geom.size-local_origin)*reciprocal
                lower, upper = torch.minimum(a, b).amax(-1), torch.maximum(a, b).amin(-1)
                distance = torch.where(lower >= 0, lower, upper)
                valid = (lower <= upper) & (distance >= 0) & ~(parallel & (local_origin.abs() > geom.size)).any(-1)
                distance = torch.where(valid, distance, torch.inf)
            else:
                distance = -local_origin[..., 2]/local_direction[..., 2]
                distance = torch.where((distance >= 0) & (local_direction[..., 2] != 0), distance, torch.inf)
            better = distance < nearest
            nearest = torch.where(better, distance, nearest)
            ids = torch.where(better, geom.geom_id, ids)
        return nearest, ids

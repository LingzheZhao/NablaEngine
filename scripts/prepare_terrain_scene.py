"""Build a pelvis-root terrain model from this engine's fixture and completed CoACD output.

CPU construction and source-generation checks only. Physical contact and gradient
qualification must run separately on explicitly provisioned compute.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np
import trimesh
from scipy.spatial import ConvexHull

from capability_check import spec_from_model
from nablaengine.g1_engine import kinematics
from nablaengine.model import capability

TRACKED_BODIES = (
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)


def verify_robot(fixture, model):
    """Assert compiled body, joint, dof and actuator dynamics retain fixture values."""
    if (fixture.nbody, fixture.njnt, fixture.nq, fixture.nv, fixture.nu) != (
            model.nbody, model.njnt, model.nq, model.nv, model.nu):
        raise ValueError("Terrain construction changed robot sizes")
    fields = (
        "body_parentid", "body_pos", "body_quat", "body_mass", "body_ipos",
        "body_iquat", "body_inertia", "jnt_type", "jnt_bodyid", "jnt_qposadr",
        "jnt_dofadr", "jnt_pos", "jnt_axis", "jnt_range", "jnt_limited",
        "jnt_actfrclimited", "jnt_actfrcrange", "dof_armature", "dof_frictionloss",
        "dof_damping", "actuator_trnid", "actuator_trntype", "actuator_gaintype",
        "actuator_biastype", "actuator_gainprm", "actuator_biasprm", "actuator_gear",
        "actuator_ctrlrange", "actuator_ctrllimited", "actuator_forcerange",
        "actuator_forcelimited", "qpos0",
    )
    errors = {}
    for name in fields:
        before, after = np.asarray(getattr(fixture, name)), np.asarray(getattr(model, name))
        error = float(np.max(np.abs(before.astype(float) - after.astype(float)), initial=0))
        if error > 1e-12:
            raise ValueError(f"Terrain construction changed {name}: {error}")
        errors[name] = error
    for kind, count in ((mujoco.mjtObj.mjOBJ_BODY, model.nbody),
                        (mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
                        (mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)):
        if [mujoco.mj_id2name(fixture, kind, i) for i in range(count)] != [
                mujoco.mj_id2name(model, kind, i) for i in range(count)]:
            raise ValueError("Terrain construction changed robot names")
    for index in range(fixture.ngeom):
        if fixture.geom_bodyid[index] == 0:
            continue
        target = model.geom(fixture.geom(index).name).id
        for field in ("geom_type", "geom_bodyid", "geom_size", "geom_pos", "geom_quat",
                      "geom_contype", "geom_conaffinity", "geom_condim", "geom_friction",
                      "geom_solref", "geom_solimp", "geom_margin", "geom_gap"):
            if not np.allclose(getattr(fixture, field)[index], getattr(model, field)[target], rtol=0, atol=1e-12):
                raise ValueError(f"Terrain construction changed {fixture.geom(index).name} {field}")
    for index in range(fixture.npair):
        if any(fixture.geom_bodyid[g] == 0 for g in (fixture.pair_geom1[index], fixture.pair_geom2[index])):
            continue
        target = model.pair(fixture.pair(index).name).id
        for field in ("pair_dim", "pair_friction", "pair_solref", "pair_solimp", "pair_margin", "pair_gap"):
            if not np.allclose(getattr(fixture, field)[index], getattr(model, field)[target], rtol=0, atol=1e-12):
                raise ValueError(f"Terrain construction changed self pair {fixture.pair(index).name} {field}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--coacd", type=Path, required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--terrain-artifact-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    execution = json.loads((args.coacd / "execution.json").read_text())
    result = json.loads((args.coacd / "result.json").read_text())
    if execution["state"] != "completed" or execution.get("synthetic_hulls_only"):
        raise ValueError("Scene requires a completed, real CoACD decomposition")
    if execution["artifact_id"] != args.terrain_artifact_id or result["artifact_id"] != args.terrain_artifact_id:
        raise ValueError("Selected CoACD artifact identity mismatch")
    if execution["frame"] != result["frame"]:
        raise ValueError("CoACD frame records disagree")
    if result["frame"]["input_to_world_translation_m"] != [15.0, 8.0, 0.0]:
        raise ValueError("Expected declared upstream 30 by 16 metre input frame")
    fixture = mujoco.MjModel.from_binary_path(str(args.fixture))
    if fixture.body(fixture.jnt_bodyid[0]).name != "pelvis":
        raise ValueError("Fixture must have a free pelvis root")
    spec = spec_from_model(fixture)
    # motion_matched_terrain imports its STL floor and creates no additional plane.
    # Retain all 469 fixture self pairs. Replace its 33 plane pairs with one
    # explicit pair per robot collision geom per hull, retaining pair parameters.
    planes = [i for i in range(fixture.ngeom) if fixture.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE]
    if len(planes) != 1 or fixture.geom_bodyid[planes[0]] != 0:
        raise ValueError("Expected exactly one fixture world plane")
    floor = planes[0]
    floor_pairs = {}
    for pair in list(spec.pairs):
        if fixture.geom(floor).name in (pair.geomname1, pair.geomname2):
            robot = pair.geomname2 if pair.geomname1 == fixture.geom(floor).name else pair.geomname1
            floor_pairs[robot] = pair
    robots = [fixture.geom(i).name for i in range(fixture.ngeom) if fixture.geom_bodyid[i] != 0]
    if len(robots) != 33 or set(floor_pairs) != set(robots):
        raise ValueError("Expected all 33 robot collision geoms bound to fixture plane pairs")
    # Copy values before deleting owned MjSpec objects.
    parameters = {name: dict(condim=int(p.condim), friction=np.array(p.friction).copy(),
                            solref=np.array(p.solref).copy(), solimp=np.array(p.solimp).copy(),
                            margin=float(p.margin), gap=float(p.gap)) for name, p in floor_pairs.items()}
    for pair in floor_pairs.values():
        spec.delete(pair)
    spec.delete(spec.geom(fixture.geom(floor).name))
    self_pairs = len(list(spec.pairs))
    hull_names = []
    expected_vertices = []
    hull_face_counts = []
    for index, record in enumerate(result["hulls"]):
        hull = trimesh.load(args.coacd / f"hull_{index}.ply", process=False)
        vertices, faces = np.asarray(hull.vertices), np.asarray(hull.faces)
        if vertices.shape != (record["vertices"], 3) or faces.shape != (record["faces"], 3):
            raise ValueError(f"Hull {index} differs from its completed decomposition record")
        if not np.isfinite(vertices).all() or not hull.is_watertight:
            raise ValueError(f"Hull {index} is not a finite closed mesh")
        # MuJoCo collides against the convex hull. Re-triangulate the stored
        # float32 vertices so near-coplanar CoACD faces cannot encode tiny dents.
        convex = ConvexHull(vertices)
        faces = convex.simplices.copy()
        triangles = vertices[faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        inward = np.sum(normals * (triangles[:, 0] - vertices.mean(0)), axis=1) < 0
        faces[inward] = faces[inward][:, [0, 2, 1]]
        hull_face_counts.append(dict(coacd_faces=record["faces"], convex_faces=len(faces)))
        name = f"terrain_hull_{index}"
        hull_names.append(name)
        expected_vertices.append(vertices)
        spec.add_mesh(name=name, uservert=vertices.ravel().tolist(), userface=faces.ravel().tolist())
        spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH,
                                meshname=name, contype=0, conaffinity=0, mass=0)
        for robot in robots:
            # Foot names preserve the environment's established foot-pair selector.
            suffix = "floor" if "_foot" in robot else "terrain"
            pair = spec.add_pair(name=f"{robot}__hull_{index}_{suffix}",
                                 geomname1=robot, geomname2=name)
            for field, value in parameters[robot].items():
                setattr(pair, field, value)
    if not hull_names:
        raise ValueError("Empty terrain decomposition")
    model = spec.compile()
    robot_error = verify_robot(fixture, model)
    foot_pairs = [i for i in range(model.npair) if "_foot" in model.pair(i).name and model.pair(i).name.endswith("_floor")]
    if model.npair != self_pairs + 33 * len(hull_names):
        raise ValueError("Compiled terrain pair count mismatch")
    for name in TRACKED_BODIES:
        model.body(name)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    frame_errors = []
    for name, expected in zip(hull_names, expected_vertices):
        geom = model.geom(name).id
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        actual = data.geom_xpos[geom] + model.mesh_vert[start:start + count] @ data.geom_xmat[geom].reshape(3, 3).T
        error = float(max(np.linalg.norm(actual[:, None] - expected[None, :], axis=2).min(0).max(),
                          np.linalg.norm(actual[:, None] - expected[None, :], axis=2).min(1).max()))
        if error > 2e-5:
            raise ValueError(f"Compiled hull world vertices changed: {name}, {error}")
        frame_errors.append(error)
    reports = {scope: capability.check(model, np.asarray(foot_pairs), scope=scope)
               for scope in ("forward", "engine", "kinematics")}
    findings = {scope: [dict(code=f.code, status=f.status, what=f.what, value=f.value)
                        for f in report.findings if f.status != "ok"] for scope, report in reports.items()}
    (args.output / "capability.json").write_text(json.dumps(findings, indent=2))
    if any(report.errors() for report in reports.values()):
        raise ValueError(f"Terrain unsupported, {model.npair} explicit pairs. See capability.json")
    kinematics.check_model(model)
    from nablaengine.g1_engine import step, adjoint
    from nablaengine.model.convex import pack_static_convex_hulls
    packed_hulls = pack_static_convex_hulls(model, step.collision_geoms(model))
    forward_source = step.step_header(model, convex_hulls=packed_hulls)
    adjoint_source = adjoint.cuda_source(model, convex_hulls=packed_hulls)
    # The actual generated source path is checked by the owning engine API.
    adjoint.engine_constants(model, np.asarray(foot_pairs), convex_hulls=packed_hulls)
    contract = dict(
        artifact_id=args.artifact_id, terrain_artifact_id=args.terrain_artifact_id,
        fixture=str(args.fixture), coacd=str(args.coacd), terrain_file=result["terrain_file"],
        terrain_index=result["terrain_index"], actuator_joint_names=[model.joint(model.actuator_trnid[i, 0]).name for i in range(model.nu)],
        tracked_body_names=list(TRACKED_BODIES), foot_pairs=foot_pairs,
        versions={name: importlib.metadata.version(name) for name in ("mujoco", "numpy", "trimesh", "scipy")},
        engine_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip(),
        frame={**result["frame"], "coacd_auto_align_z_m": result["auto_align_z"],
               "hull_files_already_aligned": True},
        counts=dict(robot_geoms=len(robots), hull_geoms=len(hull_names), geoms=model.ngeom,
                    self_pairs=self_pairs, terrain_pairs=33 * len(hull_names), pairs=model.npair),
        hull_surface="ConvexHull triangulation of stored CoACD vertices, with outward winding. All vertices retained.",
        hull_face_counts=hull_face_counts,
        floor="Fixture plane and its pairs removed. Upstream motion_matched_terrain uses STL floor through CoACD hulls. Robot self pairs retained.",
        verification=dict(robot_field_max_errors=robot_error, hull_world_vertex_max_errors_m=frame_errors,
                          capability_scopes=list(reports), generated_forward_header_bytes=len(forward_source),
                          generated_adjoint_source_bytes=len(adjoint_source), adjoint_constants_created=True, physics_qualified=False),
    )
    mujoco.mj_saveModel(model, str(args.output / "model.mjb"))
    (args.output / "contract.json").write_text(json.dumps(contract, indent=2))
    print(json.dumps(dict(output=str(args.output), counts=contract["counts"])))


if __name__ == "__main__":
    main()

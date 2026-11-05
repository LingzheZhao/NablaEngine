"""Compile the source torso-root G1 with equivalent combined position/zero-velocity PD.

The source geometry, joints, passive damping, frictionloss and visual triangles
remain in their original frames. Source filter contacts become explicit pairs.
A declared placeholder floor provides templates for the terrain-scene producer,
which removes that floor and creates each robot/terrain pair. No stepping occurs.
"""
import argparse
import ast
import json
import operator
import os
from pathlib import Path
import re
import sys
import traceback

sys.dont_write_bytecode = True
import mujoco
import numpy as np
import yaml

from nablaengine.g1_engine import adjoint, kinematics
from nablaengine.model import capability


def source_pd_groups(path):
    """Evaluate only the numeric assignments selected by the source PD tuple."""
    assignments = {}
    for row in ast.parse(path.read_text()).body:
        if isinstance(row, ast.Assign) and len(row.targets) == 1 and isinstance(row.targets[0], ast.Name):
            assignments[row.targets[0].id] = row.value
        if isinstance(row, ast.AnnAssign) and isinstance(row.target, ast.Name):
            assignments[row.target.id] = row.value
    values = {}
    binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.Pow: operator.pow}

    def evaluate(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(evaluate(item) for item in node.elts)
        if isinstance(node, ast.Name):
            if node.id not in values:
                values[node.id] = evaluate(assignments[node.id])
            return values[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            return binary[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -evaluate(node.operand)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'BuiltinPdActuatorCfg' and not node.args:
            return {item.arg: evaluate(item.value) for item in node.keywords}
        raise ValueError(f'Unsupported expression in selected source PD configuration: {ast.dump(node)}')

    return evaluate(assignments['beyondmimic_g1_29dof_actuator_cfgs'])


def source_action_parameters(path, joint_names, actuator_parameters):
    assignments = {}
    for row in ast.parse(path.read_text()).body:
        if isinstance(row, ast.Assign) and len(row.targets) == 1 and isinstance(row.targets[0], ast.Name):
            assignments[row.targets[0].id] = row.value
        if isinstance(row, ast.AnnAssign) and isinstance(row.target, ast.Name):
            assignments[row.target.id] = row.value
    initial = assignments['_POPSICLE_INIT_STATE']
    positions = ast.literal_eval(next(item.value for item in initial.keywords if item.arg == 'joint_pos'))
    entity = assignments['G1_29DOF_TORSOBASE_POPSICLE_CFG']
    articulation = next(item.value for item in entity.keywords if item.arg == 'articulation')
    soft_limit = ast.literal_eval(next(item.value for item in articulation.keywords if item.arg == 'soft_joint_pos_limit_factor'))
    defaults = []
    for joint in joint_names:
        matched = [value for expression, value in positions.items() if re.fullmatch(expression, joint)]
        if len(matched) > 1:
            raise ValueError(f'Source initial joint position binds {joint} more than once')
        defaults.append(matched[0] if matched else 0.)
    return dict(default_joint_position=defaults, action_scale=[row['action_scale'] for row in actuator_parameters],
                soft_joint_pos_limit_factor=soft_limit,
                joint_names=joint_names, default_velocity_target=0.)


def pair_parameters(model, first, second):
    """The selected source's homogeneous contact material, checked explicitly."""
    for field in ('geom_condim', 'geom_friction', 'geom_solref', 'geom_solimp', 'geom_margin', 'geom_gap'):
        if not np.array_equal(getattr(model, field)[first], getattr(model, field)[second]):
            raise ValueError(f'Selected source contact materials differ at {field}; declare their mixing before use')
    if model.geom_margin[first] or model.geom_gap[first]:
        raise ValueError('Selected source contact materials require nonzero margin or gap')
    friction = model.geom_friction[first]
    return dict(condim=int(model.geom_condim[first]),
                friction=[friction[0], friction[0], friction[1], friction[2], friction[2]],
                solref=model.geom_solref[first], solimp=model.geom_solimp[first], margin=0., gap=0.)


def build_source_model(upstream, config):
    """Own construction of the source model and its equivalent explicit contacts."""
    assets = upstream / 'src/instinct_mj/assets'
    source_path = assets / 'resources/unitree_g1/xml/g1_29dof_torsobase_popsicle.xml'
    groups = source_pd_groups(assets / 'unitree_g1.py')
    spec = mujoco.MjSpec.from_file(str(source_path))
    spec.compiler.fusestatic = True
    spec.compiler.meshdir = str((source_path.parent / '../meshes').resolve())
    for index, geom in enumerate(spec.geoms):
        geom.name = f'source_geom_{index}'
    for name in ('timestep', 'iterations', 'ls_iterations', 'ccd_iterations'):
        setattr(spec.option, name, config[name])
    spec.option.integrator = getattr(mujoco.mjtIntegrator, 'mjINT_' + config['integrator'].upper())
    spec.option.solver = getattr(mujoco.mjtSolver, 'mjSOL_' + config['solver'].upper())
    spec.option.cone = getattr(mujoco.mjtCone, 'mjCONE_' + config['cone'].upper())
    spec.option.jacobian = getattr(mujoco.mjtJacobian, 'mjJAC_' + config['jacobian'].upper())
    actuators = []
    for joint in spec.joints:
        if joint.type == mujoco.mjtJoint.mjJNT_FREE:
            continue
        matches = [group for group in groups if any(re.fullmatch(expr, joint.name) for expr in group['target_names_expr'])]
        if len(matches) != 1:
            raise ValueError(f'Source PD configuration must bind {joint.name} once')
        group = matches[0]
        joint.armature = group['armature']
        joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
        joint.actfrcrange = [-group['effort_limit'], group['effort_limit']]
        actuator = spec.add_actuator(name=f'{joint.name}_pd', target=joint.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        actuator.gainprm = [group['stiffness']] + [0.]*9
        actuator.biasprm = [0., -group['stiffness'], -group['damping']] + [0.]*7
        actuator.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
        actuator.forcelimited = mujoco.mjtLimited.mjLIMITED_FALSE
        actuators.append(dict(joint=joint.name, kp=group['stiffness'], kd=group['damping'],
                              armature=group['armature'], effort_limit=group['effort_limit'],
                              action_scale=.25*group['effort_limit']/group['stiffness']))
    source = spec.compile()
    if (source.nbody, source.nq, source.nv, source.nu) != (31, 36, 35, 29) or source.body(1).name != 'torso_link':
        raise ValueError('Compiled source must have a torso free root and 29 ordered hinge servos after fixed-body fusion')
    eligible_pairs = capability.filter_pairs(source)
    collision = [i for i in range(source.ngeom) if source.geom_contype[i] or source.geom_conaffinity[i]]
    if not collision:
        raise ValueError('Selected source has no physical collision geometry')
    for first, second in eligible_pairs:
        parameters = pair_parameters(source, first, second)
        pair = spec.add_pair(name=f'source_self_{first}_{second}',
                             geomname1=source.geom(first).name, geomname2=source.geom(second).name)
        for name, value in parameters.items():
            setattr(pair, name, value)
    spec.worldbody.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0., 0., .1],
                            contype=0, conaffinity=0, group=0, mass=0)
    floor_parameters = pair_parameters(source, collision[0], collision[0])
    foot_names = set()
    for geom in collision:
        body = source.body(int(source.geom_bodyid[geom])).name
        is_foot = body in ('left_ankle_roll_link', 'right_ankle_roll_link')
        name = f'{body}_foot_{geom}_floor' if is_foot else f'source_geom_{geom}_floor'
        pair = spec.add_pair(name=name, geomname1=source.geom(geom).name, geomname2='floor')
        parameters = pair_parameters(source, geom, collision[0])
        for field, value in parameters.items():
            setattr(pair, field, value)
        if is_foot:
            foot_names.add(name)
    for geom in spec.geoms:
        geom.contype = 0
        geom.conaffinity = 0
    model = spec.compile()
    body_fields = ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'body_pos', 'body_quat',
                   'body_parentid', 'jnt_type', 'jnt_bodyid', 'jnt_pos', 'jnt_axis', 'jnt_range',
                   'dof_damping', 'dof_frictionloss', 'dof_armature', 'actuator_gainprm', 'actuator_biasprm')
    for name in body_fields:
        if not np.array_equal(getattr(source, name), getattr(model, name)):
            raise ValueError(f'Explicit contact construction changed source {name}')
    for geom in range(source.ngeom):
        target = model.geom(source.geom(geom).name).id
        for field in ('geom_bodyid', 'geom_type', 'geom_size', 'geom_pos', 'geom_quat', 'geom_group'):
            if not np.array_equal(getattr(source, field)[geom], getattr(model, field)[target]):
                raise ValueError(f'Explicit contact construction changed source {field}')
    return spec, source, model, actuators, collision, eligible_pairs, foot_names, floor_parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--upstream-revision', required=True)
    parser.add_argument('--code-revision', required=True)
    parser.add_argument('--artifact-id', required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/instinct_g1_model.yaml')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.upstream, args.config, args.output = args.upstream.resolve(), args.config.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.chdir(args.output)
    status = dict(state='started', artifact_id=args.artifact_id, code_revision=args.code_revision,
                  upstream_revision=args.upstream_revision)

    def write_status():
        (args.output / 'execution.json').write_text(json.dumps(status, indent=2))

    def fail(kind, value, tb):
        status.update(state='failed', failure_type=kind.__name__, failure=str(value),
                      traceback=''.join(traceback.format_exception(kind, value, tb)))
        write_status()
        sys.__excepthook__(kind, value, tb)

    write_status()
    sys.excepthook = fail
    config = yaml.safe_load(args.config.read_text())
    assets = args.upstream / 'src/instinct_mj/assets'
    source_path = assets / 'resources/unitree_g1/xml/g1_29dof_torsobase_popsicle.xml'
    spec, source, model, actuators, collision, eligible_pairs, foot_names, floor_parameters = build_source_model(args.upstream, config)
    mujoco.mj_saveModel(source, str(args.output / 'native_model.mjb'))
    feet = np.asarray(sorted(model.pair(name).id for name in foot_names), dtype=np.int32)
    reports = {scope: capability.check(model, feet, scope=scope).to_dict() for scope in ('forward', 'engine', 'kinematics')}
    (args.output / 'capability.json').write_text(json.dumps(reports, indent=2))
    if any(report['errors'] for report in reports.values()):
        raise ValueError('Source model failed native capability checks; see capability.json')
    kinematics.check_model(model)
    adjoint.cuda_source(model)
    adjoint.engine_constants(model, feet)
    joint_names = [model.joint(model.actuator_trnid[a, 0]).name for a in range(model.nu)]
    source_action = source_action_parameters(assets / 'unitree_g1.py', joint_names, actuators)
    joints = [dict(name=model.joint(j).name, qpos_address=int(model.jnt_qposadr[j]),
                   dof_address=int(model.jnt_dofadr[j]), damping=float(model.dof_damping[model.jnt_dofadr[j]]),
                   frictionloss=float(model.dof_frictionloss[model.jnt_dofadr[j]]),
                   armature=float(model.dof_armature[model.jnt_dofadr[j]])) for j in range(1, model.njnt)]
    contract = dict(format='instinct-g1-model-v1', upstream=str(args.upstream), artifact_id=args.artifact_id, code_revision=args.code_revision, upstream_revision=args.upstream_revision,
                    source_model=str(source_path), resolved_configuration=config, root_body='torso_link', free_root_body='torso_link',
                    source_action=source_action,
                    bodies=[dict(name=model.body(b).name, parent=int(model.body_parentid[b]), mass=float(model.body_mass[b]),
                                 position=model.body_pos[b].tolist(), quaternion_wxyz=model.body_quat[b].tolist(),
                                 inertia_position=model.body_ipos[b].tolist(), inertia_quaternion_wxyz=model.body_iquat[b].tolist(),
                                 diagonal_inertia=model.body_inertia[b].tolist()) for b in range(1, model.nbody)],
                    collision_geoms=[dict(name=source.geom(g).name, body=source.body(source.geom_bodyid[g]).name,
                                          type=mujoco.mjtGeom(int(source.geom_type[g])).name, group=int(source.geom_group[g]),
                                          size=source.geom_size[g].tolist(), position=source.geom_pos[g].tolist(),
                                          quaternion_wxyz=source.geom_quat[g].tolist()) for g in collision],
                    actuator_joint_names=[model.joint(model.actuator_trnid[a, 0]).name for a in range(model.nu)],
                    actuators=actuators, joints=joints, foot_pairs=feet.tolist(),
                    collision_geom_names=[source.geom(i).name for i in collision],
                    source_self_pairs=[[source.geom(a).name, source.geom(b).name] for a, b in eligible_pairs],
                    counts=dict(bodies=model.nbody, nq=model.nq, nv=model.nv, actuators=model.nu,
                                robot_collision_geoms=len(collision), robot_visual_geoms=int((source.geom_group == 2).sum()),
                                self_pairs=len(eligible_pairs), floor_pairs=len(collision), pairs=model.npair, geoms=model.ngeom),
                    solver=dict(integrator='implicitfast', solver='Newton', cone='pyramidal', jacobian='sparse'),
                    contact_material=dict(condim=floor_parameters['condim'], friction=floor_parameters['friction'],
                                          solref=floor_parameters['solref'].tolist(), solimp=floor_parameters['solimp'].tolist()),
                    geometry='Original source triangle meshes and capsule/sphere collision surfaces; fixed bodies fused without changing geometry',
                    combined_pd='tau = kp * (position_target - q) - kd * qvel; velocity target zero; summed effort clipped by source joint actuator force range',
                    placeholder_floor='Plane and robot-plane pairs provide terrain contact templates. Terrain construction removes the plane and replaces all its pairs.',
                    contact_selection='capability.filter_pairs on the constructed source, preserving contype/conaffinity, same-weld/parent filtering and explicit source body excludes before conversion',
                    physics_qualified=False)
    mujoco.mj_saveModel(model, str(args.output / 'model.mjb'))
    (args.output / 'contract.json').write_text(json.dumps(contract, indent=2))
    status.update(state='completed', counts=contract['counts'])
    write_status()
    print(json.dumps(status))


if __name__ == '__main__':
    main()

"""Generated CPU mathematics for fixed-hit triangle depth pose derivatives.

No simulator or GPU is used. Independent Moller-Trumbore triangle intersections
supply finite differences. Analytic derivatives use the selected world plane.
Quaternion perturbations follow normalized WXYZ unit-sphere tangents. Output is
written to a fresh external directory. Face-selection boundaries are recorded
with their one-sided derivatives.
"""
import argparse
import json
from pathlib import Path

import numpy as np

PROTOCOL = dict(fd_steps=[1e-5, 5e-6], absolute_tolerance=2e-8, relative_tolerance=2e-7,
                quaternion='WXYZ, normalized left increments from world angular velocity',
                smooth_cases='triangle-interior hits with unchanged face selection',
                boundary_case='two adjacent noncoplanar faces, one-sided derivatives retained',
                maximum_range_m=3., seed=20261106)


def unit(vector):
    return vector/np.linalg.norm(vector)


def multiply(a, b):
    aw, av, bw, bv = a[0], a[1:], b[0], b[1:]
    return np.r_[aw*bw-av@bv, aw*bv+bw*av+np.cross(av, bv)]


def quaternion(vector):
    angle = np.linalg.norm(vector)
    if angle == 0:
        return np.array([1., 0., 0., 0.])
    return np.r_[np.cos(angle/2), np.sin(angle/2)*vector/angle]


def matrix(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def ray(state):
    rotation = matrix(state['camera_quaternion'])
    return state['camera_position']+rotation@state['camera_offset'], rotation@state['camera_ray']


def triangle_world(state):
    return state['triangle']@matrix(state['body_quaternion']).T+state['body_position']


def triangle_hit(vertices, origin, direction):
    a, b, c = vertices
    first, second = b-a, c-a
    cross = np.cross(direction, second)
    denominator = first@cross
    if abs(denominator) < 1e-12:
        return np.inf
    offset = origin-a
    u = (offset@cross)/denominator
    normal_offset = np.cross(offset, first)
    v = (direction@normal_offset)/denominator
    distance = (second@normal_offset)/denominator
    if u < 0 or v < 0 or u+v > 1 or distance < 0:
        return np.inf
    return float(distance)


def range_and_id(state):
    origin, direction = ray(state)
    hit = triangle_hit(triangle_world(state), origin, direction)
    if not np.isfinite(hit) or hit > PROTOCOL['maximum_range_m']:
        return PROTOCOL['maximum_range_m'], -1
    return hit, 0


def analytic(state):
    origin, direction = ray(state)
    vertices = triangle_world(state)
    normal = unit(np.cross(vertices[1]-vertices[0], vertices[2]-vertices[0]))
    denominator = normal@direction
    distance = (normal@(vertices[0]-origin))/denominator
    hit = origin+distance*direction
    origin_gradient = -normal/denominator
    direction_gradient = -distance*normal/denominator
    return dict(distance=distance, origin=origin_gradient, direction=direction_gradient,
                camera_translation=origin_gradient,
                camera_rotation=-np.cross(hit-state['camera_position'], normal)/denominator,
                body_translation=normal/denominator,
                body_rotation=np.cross(hit-state['body_position'], normal)/denominator)


def perturb(state, owner, translation=None, rotation=None):
    result = {key: value.copy() for key, value in state.items()}
    if translation is not None:
        result[owner+'_position'] += translation
    if rotation is not None:
        result[owner+'_quaternion'] = unit(multiply(quaternion(rotation), result[owner+'_quaternion']))
    return result


def generated_state():
    body_q = quaternion(np.array([.2, -.35, .4]))
    camera_q = quaternion(np.array([.3, .15, -.2]))
    body_position = np.array([.4, -.2, 1.3])
    offset = np.array([.12, -.05, .07])
    direction = unit(np.array([.05, -.04, 1.]))
    hit = matrix(body_q)@np.array([.1, -.2, .6])+body_position
    origin = hit-2*matrix(camera_q)@direction
    return dict(body_position=body_position, body_quaternion=body_q,
                camera_position=origin-matrix(camera_q)@offset, camera_quaternion=camera_q,
                camera_offset=offset, camera_ray=direction,
                triangle=np.array([[-3., -3., .6], [3., -3., .6], [0., 3., .6]]))


def close(a, b):
    return bool(np.isfinite(a).all() and np.isfinite(b).all()
                and np.all(np.abs(a-b) <= PROTOCOL['absolute_tolerance']+PROTOCOL['relative_tolerance']*np.abs(b)))


def run():
    state = generated_state()
    derivative = analytic(state)
    assert close(np.array(range_and_id(state)[0]), np.array(2.))
    rows = []
    for owner in ('camera', 'body'):
        for operation in ('translation', 'rotation'):
            for axis in range(3):
                basis = np.eye(3)[axis]
                values, selections = [], []
                for step in PROTOCOL['fd_steps']:
                    pair = [range_and_id(perturb(state, owner, **{operation: sign*step*basis})) for sign in (-1, 1)]
                    values.append((pair[1][0]-pair[0][0])/(2*step))
                    selections.extend([pair[0][1], pair[1][1]])
                expected = derivative[owner+'_'+operation][axis]
                rows.append(dict(case=f'{owner}_{operation}_{axis}', analytic=float(expected), fd=values,
                                 selected_faces=selections, pass_=bool(close(np.array(values), expected) and all(i == 0 for i in selections))))
    # Check raw origin and the two physical unit-direction tangent directions.
    origin, direction = ray(state)
    vertices = triangle_world(state)
    tangent1 = unit(np.cross(direction, np.array([1., 0., 0.])))
    tangent2 = np.cross(direction, tangent1)
    for name, vectors in (('origin', np.eye(3)), ('direction_tangent', [tangent1, tangent2])):
        for index, vector in enumerate(vectors):
            expected = derivative['origin' if name == 'origin' else 'direction']@vector
            values = []
            for step in PROTOCOL['fd_steps']:
                pair = []
                for sign in (-1, 1):
                    moved_origin = origin+sign*step*vector if name == 'origin' else origin
                    moved_direction = unit(direction+sign*step*vector) if name != 'origin' else direction
                    pair.append(triangle_hit(vertices, moved_origin, moved_direction))
                values.append((pair[1]-pair[0])/(2*step))
            rows.append(dict(case=f'{name}_{index}', analytic=float(expected), fd=values,
                             pass_=close(np.array(values), expected)))
    # A common transform preserves distances, including offset camera origins.
    rng = np.random.default_rng(PROTOCOL['seed'])
    common_translation, common_rotation = rng.normal(size=(2, 3))
    for name, velocity, angular in (('common_translation', common_translation, np.zeros(3)),
                                    ('common_rotation', np.zeros(3), common_rotation)):
        expected = (derivative['camera_translation']@(velocity+np.cross(angular, state['camera_position']))
                    +derivative['camera_rotation']@angular
                    +derivative['body_translation']@(velocity+np.cross(angular, state['body_position']))
                    +derivative['body_rotation']@angular)
        values = []
        for step in PROTOCOL['fd_steps']:
            pair = []
            for sign in (-1, 1):
                moved = {key: value.copy() for key, value in state.items()}
                delta = quaternion(sign*step*angular)
                rotation = matrix(delta)
                for owner in ('camera', 'body'):
                    moved[owner+'_position'] = rotation@state[owner+'_position']+sign*step*velocity
                    moved[owner+'_quaternion'] = unit(multiply(delta, state[owner+'_quaternion']))
                pair.append(range_and_id(moved)[0])
            values.append((pair[1]-pair[0])/(2*step))
        rows.append(dict(case=name, analytic=float(expected), fd=values,
                         pass_=bool(close(np.array(values), 0.) and close(np.array(expected), 0.))))
    # Quaternion direct-coordinate tangent confirms the 0.5 left-increment factor.
    angular = unit(rng.normal(size=3))
    for owner in ('camera', 'body'):
        q = state[owner+'_quaternion']
        tangent = .5*multiply(np.r_[0., angular], q)
        assert abs(q@tangent) < 1e-15
        values = []
        for step in PROTOCOL['fd_steps']:
            pair = []
            for sign in (-1, 1):
                moved = {key: value.copy() for key, value in state.items()}
                moved[owner+'_quaternion'] = unit(q+sign*step*tangent)
                pair.append(range_and_id(moved)[0])
            values.append((pair[1]-pair[0])/(2*step))
        expected = derivative[owner+'_rotation']@angular
        rows.append(dict(case=f'{owner}_unit_quaternion_tangent', analytic=float(expected), fd=values,
                         pass_=close(np.array(values), expected)))
    for name in ('triangle_miss', 'maximum_range_clamp'):
        changed = {key: value.copy() for key, value in state.items()}
        changed['camera_position'] += (np.array([20., 0., 0.]) if name == 'triangle_miss' else -2.*direction)
        selections, values = [], []
        for step in PROTOCOL['fd_steps']:
            pair = [range_and_id(perturb(changed, 'camera', translation=sign*step*np.array([.2, -.1, .3])))
                    for sign in (-1, 1)]
            selections.extend([p[1] for p in pair])
            values.append((pair[1][0]-pair[0][0])/(2*step))
        rows.append(dict(case=name, analytic=0., fd=values, selected_faces=selections,
                         pass_=bool(close(np.array(values), 0.) and all(i == -1 for i in selections))))
    # Two triangles meet at x=0 with slopes +0.3 and -0.2. The camera
    # translates through the shared edge, making distinct one-sided derivatives.
    faces = [np.array([[-2., -2., 1.4], [0., -2., 2.], [0., 2., 2.]]),
             np.array([[0., -2., 2.], [2., -2., 1.6], [0., 2., 2.]])]
    def boundary_range(x):
        hits = [triangle_hit(face, np.array([x, -.5, 0.]), np.array([0., 0., 1.])) for face in faces]
        index = int(np.argmin(hits))
        return hits[index], index
    boundary = []
    for step in PROTOCOL['fd_steps']:
        negative, zero, positive = boundary_range(-step), boundary_range(0.), boundary_range(step)
        boundary.append(dict(step=step, negative_face=negative[1], positive_face=positive[1],
                             left_derivative=(zero[0]-negative[0])/step,
                             right_derivative=(positive[0]-zero[0])/step,
                             centered_difference=(positive[0]-negative[0])/(2*step)))
    assert all(row['negative_face'] != row['positive_face'] for row in boundary)
    return dict(protocol=PROTOCOL, smooth_cases=rows, face_switch_boundary=boundary,
                boundary_interpretation='one-sided derivatives differ, centered FD is not a smooth derivative',
                pass_=all(row['pass_'] for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--code-revision', required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.is_relative_to(Path(__file__).resolve().parents[1]):
        parser.error('--out must be outside checkout')
    out.mkdir(parents=True, exist_ok=False)
    report = run()
    report['code_revision'] = args.code_revision
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(pass_=report['pass_'], smooth_cases=len(report['smooth_cases']),
                         boundaries=len(report['face_switch_boundary']))))
    return 0 if report['pass_'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

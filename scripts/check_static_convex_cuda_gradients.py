"""Qualify the packed-mesh CUDA primitive on provisioned compute.

Declare the protocol before compiling or querying. All source, build logs,
conditions and failures stay in a fresh external output root. The widened C++
replay is a numerical reference sharing the CUDA formulas. MuJoCo supplies the
independent compiled-frame and physics-distance checks. No integration is run.
"""
import argparse
import ctypes
import json
import re
import subprocess
import traceback
from pathlib import Path

import numpy as np

from check_convex_contact_gradients import MujocoOracle, cases, load_ply, rotation

PROTOCOL = {
    'scope': 'float32 CUDA packed static convex hull sphere/capsule forward and six-pose VJP',
    'contacts': 'unique witness in slot0, near-minimum-distance min/max segment parameters in slots0/1',
    'geometry_tolerance_m': 2e-5,
    'manifold_distance_tolerance_m': 2e-6,
    'manifold_relative_tolerance': '8*float32_epsilon*max(1,centerline_distance)',
    'distinct_witness_tolerance': 'max(2e-5m,32*float32_epsilon*max(1,centerline_distance))',
    'distance_atol': 5e-5,
    'forward_atol': 5e-5,
    'forward_rtol': 2e-4,
    'jacobian_atol': 1e-2,
    'jacobian_rtol': 2e-2,
    'fd_steps': [2e-5, 1e-5],
    'fd_stability_atol': 3e-4,
    'fd_stability_rtol': 3e-3,
    'independent_oracle': 'MuJoCo mj_geomDistance distance only, compiled mesh world-frame check',
    'numerical_reference': 'same rounded primal formulas widened to float64 for forward and FD',
    'gradient_coverage': '13 basis output cotangents by six robot world translation/rotation directions, two FD steps',
    'manifold_gradient_gate': 'both endpoint slots gated on full 13 by 6 FD inside the near-tie band',
    'inside': 'minimum support facet of hull plus segment, signed distance, continuously reconstructed witnesses',
    'side_facet_support_tolerance': '8*float32_epsilon*max(1,max_abs_local_vertex)',
    'facet_depth_ties': 'original faces first, side facet must improve depth by 8*float32_epsilon*max(1,max_abs_local_vertex)',
    'near_zero': 'centerline separation at most 2e-5m uses supporting facet signed distance and normal',
    'unsupported': 'invalid geometry or failed witness reconstruction returns negative status, host must stop batch',
    'nonsmooth_gate': 'medial-axis cases require independent distance and finite outputs/AD; forward/FD differences recorded',
    'side_input_selection': 'first deterministic seed11205 inside query whose extruded-hull nearest facet is not an original face',
    'coverage': ['generated_box', 'every supplied real hull', 'sphere_face', 'sphere_edge', 'sphere_corner',
                 'sphere_radius_penetration', 'capsule_endpoint', 'capsule_interior', 'parallel_face_two_contacts',
                 'capsule_nearparallel_two_contacts', 'capsule_nearparallel_unique',
                 'sphere_inside_medial_axis', 'capsule_intersection_medial_axis', 'touch_supported',
                 'sphere_inside_face', 'capsule_inside_face_endpoint', 'capsule_inside_parallel_face',
                 'capsule_inside_side_facet'],
}

CUDA_RUNNER = r'''
__device__ int query(const float* x, float* y, const float* cot, float* gradient) {
  int status = static_convex_contact(0,int(x[15]),x,x+3,x[6],x[7],x+8,x+11,y,y+1,y+4);
  if (status == 1 && cot)
    return static_convex_vjp(0,int(x[15]),x,x+3,x[6],x[7],x+8,x+11,cot,gradient);
  return status;
}
__global__ void check(int count,const float* inputs,float* values,float* jacobians,int* statuses) {
  int index = blockIdx.x*blockDim.x+threadIdx.x;
  if (index >= count) return;
  const float* x = inputs+16*index;
  float* y = values+13*index;
  statuses[index] = query(x,y,nullptr,nullptr);
  if (statuses[index] != 1) return;
  for (int basis = 0; basis < 13; ++basis) {
    float cot[13] = {}, gradient[6]; cot[basis] = 1.f;
    int status = query(x,y,cot,gradient);
    if (status != 1) { statuses[index] = status; return; }
    for (int j = 0; j < 6; ++j) jacobians[(13*index+basis)*6+j] = gradient[j];
  }
}
extern "C" int run(int count,const float* x,float* y,float* jac,int* status) {
  float *dx=nullptr,*dy=nullptr,*dj=nullptr;int* ds=nullptr;int error=0;
#define CHECK(OP) if(!error) error=int(OP)
  CHECK(cudaMalloc(&dx,count*16*sizeof(float)));
  CHECK(cudaMalloc(&dy,count*13*sizeof(float)));
  CHECK(cudaMalloc(&dj,count*78*sizeof(float)));
  CHECK(cudaMalloc(&ds,count*sizeof(int)));
  CHECK(cudaMemcpy(dx,x,count*16*sizeof(float),cudaMemcpyHostToDevice));
  if(!error){check<<<(count+31)/32,32>>>(count,dx,dy,dj,ds);error=int(cudaGetLastError());}
  CHECK(cudaMemcpy(y,dy,count*13*sizeof(float),cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(jac,dj,count*78*sizeof(float),cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(status,ds,count*sizeof(int),cudaMemcpyDeviceToHost));
  cudaFree(dx);cudaFree(dy);cudaFree(dj);cudaFree(ds);return error;
#undef CHECK
}
'''


def build(packed, directory):
    from nablaengine.g1_engine.convex_contact import device_source
    declarations = []
    for name, kind, values in packed.arrays:
        scalar = 'int' if kind == 'i' else 'float'
        literals = [str(int(x)) if kind == 'i' else f'{float(x):.9e}f' for x in values.ravel()]
        declarations.append(f'DKC {scalar} {name}[{values.size}]={{{",".join(literals)}}};')
    common = '\n'.join(declarations)+device_source()
    gpu = '#include <cuda_runtime.h>\n#define DKC __device__\n'+common+CUDA_RUNNER
    (directory/'primitive.cu').write_text(gpu)
    cpu = '''#include <cmath>
using std::isfinite;
#define DKC
#define __device__
inline float __fadd_rn(float a,float b){return a+b;}
inline float __fsub_rn(float a,float b){return a-b;}
inline float __fmul_rn(float a,float b){return a*b;}
'''+common+CUDA_RUNNER[:CUDA_RUNNER.index('__global__')]
    cpu += '\nextern "C" int reference(const float*x,float*y){return query(x,y,nullptr,nullptr);}\n'
    cpu = re.sub(r'\bfloat\b', 'double', cpu)
    cpu = re.sub(r'(?<=\d)f\b', '', cpu)
    for name in ['sqrt', 'fmin', 'fmax', 'fabs']:
        cpu = cpu.replace(name+'f(', name+'(')
    (directory/'reference.cpp').write_text(cpu)
    commands = [
        ['g++', '-std=c++17', '-O2', '-ffp-contract=off', '-shared', '-fPIC',
         str(directory/'reference.cpp'), '-o', str(directory/'reference.so')],
        ['nvcc', '-std=c++17', '-O2', '-shared', '-Xcompiler=-fPIC',
         str(directory/'primitive.cu'), '-o', str(directory/'primitive.so')],
    ]
    for index, command in enumerate(commands):
        result = subprocess.run(command, capture_output=True, text=True)
        (directory/f'build-{index}.log').write_text(result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError(f'compile failed, see build-{index}.log')


def load_library(path, name):
    library = ctypes.CDLL(str(path))
    function = getattr(library, name)
    function.restype = ctypes.c_int
    function.argtypes = [ctypes.c_void_p]*2 if name == 'reference' else [ctypes.c_int]+[ctypes.c_void_p]*4
    return function


def pose_input(case, oracle, slot):
    from scipy.spatial.transform import Rotation
    # Read the actual compiled geom pose, including mesh centering/alignment.
    hull_position = oracle.data.geom_xpos[oracle.h]
    hull_quaternion = Rotation.from_matrix(oracle.data.geom_xmat[oracle.h].reshape(3, 3)).as_quat()[[3, 0, 1, 2]]
    return np.r_[oracle.p+oracle.R@case['center'], oracle.R@case['axis'],
                 case['radius'], case['half_length'], hull_position, hull_quaternion, slot].astype(np.float32)


def finite_differences(reference, x):
    def value(query):
        output = np.full(13, np.nan)
        status = reference(query.ctypes.data, output.ctypes.data)
        if status != 1:
            raise ValueError(f'FD perturbation returned status {status}')
        return output
    x = x.astype(np.float64)
    baseline = value(x)
    matrices = []
    for step in PROTOCOL['fd_steps']:
        columns = []
        for direction in range(6):
            plus, minus = x.copy(), x.copy()
            if direction < 3:
                plus[direction] += step
                minus[direction] -= step
            else:
                vector = np.eye(3)[direction-3]*step
                plus[3:6] = rotation(vector)@x[3:6]
                minus[3:6] = rotation(-vector)@x[3:6]
            columns.append((value(plus)-value(minus))/(2*step))
        matrices.append(np.stack(columns, axis=1))
    return baseline, matrices


def qualification_cases(vertices, faces):
    from scipy.spatial import ConvexHull
    matrix = cases(vertices, faces)
    for case in matrix:
        case['expected_contacts'] = 1
        if case['name'] == 'inside_rejected':
            case.update(name='sphere_inside_medial_axis', reject=False, smooth=False)
        elif case['name'] == 'intersection_rejected':
            case.update(name='capsule_intersection_medial_axis', reject=False, smooth=False, expected_contacts=None)
        elif case['name'] == 'touch_rejected':
            case.update(name='touch_supported', reject=False, smooth=True)
    parallel = next(case for case in matrix if case['name'] == 'parallel_face_rejected')
    parallel.update(name='parallel_face_two_contacts', reject=False, smooth=True, expected_contacts=2)
    outward = next(case['axis'] for case in matrix if case['name'] == 'sphere_face')
    for name, angle, contacts in [('capsule_nearparallel_two_contacts', 1e-5, 2),
                                   ('capsule_nearparallel_unique', 1e-4, 1)]:
        near_parallel = dict(parallel, name=name, expected_contacts=contacts)
        near_parallel['axis'] = parallel['axis']*np.cos(angle)+outward*np.sin(angle)
        matrix.append(near_parallel)
    face = next(case for case in matrix if case['name'] == 'sphere_face')
    face_point = face['center']-outward*.08
    inside = .9*face_point+.1*vertices.mean(axis=0)
    face_depth = float(np.dot(face_point-inside, outward))
    base = dict(face, center=inside, reject=False, smooth=True, expected_contacts=1)
    matrix.append(dict(base, name='sphere_inside_face'))
    matrix.append(dict(base, name='capsule_inside_face_endpoint', half_length=.4*face_depth))
    matrix.append(dict(parallel, name='capsule_inside_parallel_face', center=inside))
    # This is geometric generation from the declared source vertices, with an
    # independent explicit Minkowski convex hull selecting a side-facet input.
    original_normals = ConvexHull(vertices).equations[:, :3]
    rng = np.random.default_rng(11205)
    half_length = .2*float(np.ptp(vertices, axis=0).max())
    for _ in range(100):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        center = .4*vertices[rng.integers(len(vertices))]+.6*vertices.mean(axis=0)
        extruded = ConvexHull(np.concatenate([vertices-axis*half_length, vertices+axis*half_length]))
        depths = -extruded.equations[:, :3]@center-extruded.equations[:, 3]
        normal = extruded.equations[int(np.argmin(depths)), :3]
        if np.max(np.abs(original_normals@normal)) < 1-1e-6:
            matrix.append(dict(base, name='capsule_inside_side_facet', center=center,
                               axis=axis, half_length=half_length))
            break
    else:
        raise ValueError('declared generated side-facet coverage could not be constructed')
    return matrix


def run_hull(vertices, faces, directory, stream):
    from nablaengine.model.convex import pack_static_convex_hulls
    matrix = qualification_cases(vertices, faces)
    # One compiled mesh owns the packed representation. Robot sphere/capsule
    # models for the independent oracle all use the same compiled terrain pose.
    packed_oracle = MujocoOracle(vertices, faces, matrix[0])
    packed = pack_static_convex_hulls(packed_oracle.model, [packed_oracle.h])
    build(packed, directory)
    reference = load_library(directory/'reference.so', 'reference')
    gpu = load_library(directory/'primitive.so', 'run')
    inputs = []
    for case in matrix:
        inputs.extend(pose_input(case, packed_oracle, slot) for slot in range(2))
    inputs = np.stack(inputs)
    values = np.full((len(inputs), 13), np.nan, np.float32)
    jacobians = np.full((len(inputs), 13, 6), np.nan, np.float32)
    statuses = np.zeros(len(inputs), np.int32)
    error = gpu(len(inputs), inputs.ctypes.data, values.ctypes.data, jacobians.ctypes.data, statuses.ctypes.data)
    if error:
        raise RuntimeError(f'CUDA runtime error {error}')
    records = []
    for index, x in enumerate(inputs):
        case, slot = matrix[index//2], index % 2
        row = dict(hull=directory.name, case=case['name'], slot=slot, status=int(statuses[index]),
                   input=x.tolist(), smooth=case['smooth'])
        try:
            if row['status'] == 0:
                row['passed'] = slot == 1 and case['expected_contacts'] in (1, None)
            elif row['status'] != 1:
                row['passed'] = False
            else:
                oracle = MujocoOracle(vertices, faces, case)
                distance, witness = oracle.distance(case)
                row.update(value=values[index].tolist(), jacobian=jacobians[index].tolist(),
                           oracle_distance=distance, oracle_witness=witness,
                           compiled_mesh_transform=oracle.transform)
                row['distance_error'] = abs(float(values[index, 0])-distance)
                row['finite_output_and_jacobian'] = bool(np.isfinite(values[index]).all() and np.isfinite(jacobians[index]).all())
                reference_value = np.full(13, np.nan)
                reference_input = x.astype(np.float64)
                reference_status = reference(reference_input.ctypes.data, reference_value.ctypes.data)
                row['reference_status'] = reference_status
                row['forward_max_error'] = float(np.abs(values[index]-reference_value).max())
                row['forward_passed'] = bool(reference_status == 1 and np.allclose(values[index], reference_value,
                    atol=PROTOCOL['forward_atol'], rtol=PROTOCOL['forward_rtol']))
                row['passed'] = bool(row['distance_error'] <= PROTOCOL['distance_atol'] and row['finite_output_and_jacobian'])
                try:
                    baseline, fds = finite_differences(reference, x)
                    row['finite_differences'] = [fd.tolist() for fd in fds]
                    row['fd_stable'] = bool(np.allclose(*fds, atol=PROTOCOL['fd_stability_atol'],
                                                       rtol=PROTOCOL['fd_stability_rtol']))
                    row['gradient_passed'] = all(np.allclose(jacobians[index], fd,
                        atol=PROTOCOL['jacobian_atol'], rtol=PROTOCOL['jacobian_rtol']) for fd in fds)
                    row['jacobian_max_error'] = max(float(np.abs(jacobians[index]-fd).max()) for fd in fds)
                except Exception:
                    row.update(fd_stable=False, gradient_passed=False, fd_error=traceback.format_exc())
                if case['smooth']:
                    row['passed'] = bool(row['passed'] and row['forward_passed'] and row['fd_stable'] and row['gradient_passed'])
                else:
                    row['gradient_gate_unavailable'] = 'declared medial-axis condition has no unique smooth contact derivative'
                if slot == 1 and case['expected_contacts'] == 1:
                    row.update(passed=False, unexpected_second_contact=True)
        except Exception:
            row.update(passed=False, error=traceback.format_exc())
        stream.write(json.dumps(row)+'\n'); stream.flush()
        print(directory.name, case['name'], slot, row['status'], row['passed'], flush=True)
        records.append(row)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hulls', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--code-revision', required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'protocol.json').write_text(json.dumps(dict(PROTOCOL, code_revision=args.code_revision,
        hulls=[str(path) for path in args.hulls]), indent=2)+'\n')
    from scipy.spatial import ConvexHull
    import mujoco
    (args.out/'environment.json').write_text(json.dumps(dict(mujoco=mujoco.__version__), indent=2)+'\n')
    box = np.array([[x, y, z] for x in [-.6, .6] for y in [-.4, .4] for z in [-.3, .3]])
    shapes = [('generated_box', box, ConvexHull(box).simplices)]
    shapes.extend((path.stem, *load_ply(path)) for path in args.hulls)
    rows = []
    with (args.out/'conditions.jsonl').open('x') as stream:
        for name, vertices, faces in shapes:
            directory = args.out/name
            try:
                directory.mkdir(exist_ok=False)
                rows.extend(run_hull(vertices, faces, directory, stream))
            except Exception:
                failure = dict(hull=name, passed=False, setup_error=traceback.format_exc())
                stream.write(json.dumps(failure)+'\n'); stream.flush(); rows.append(failure)
    result = dict(passed=bool(rows) and all(row['passed'] for row in rows), records=len(rows),
                  failed=sum(not row['passed'] for row in rows), scope=PROTOCOL['scope'],
                  smooth_contacts=sum(row.get('status') == 1 and row.get('smooth', False) for row in rows),
                  smooth_gradient_failures=sum(row.get('status') == 1 and row.get('smooth', False)
                                               and not row.get('gradient_passed', False) for row in rows),
                  nonsmooth_contacts=sum('gradient_gate_unavailable' in row for row in rows))
    (args.out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__': main()

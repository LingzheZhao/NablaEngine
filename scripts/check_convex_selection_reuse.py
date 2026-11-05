"""Check convex selection reuse on generated geometry and fixed source packets.

The math command compiles CPU numerical evaluations of the declared baseline
source and current source. It compares both slots and all 13 basis VJPs without
physics integration. The compare command reads separately provisioned GPU runs
of check_convex_aabb.py on the same 56-state source selection with AABB enabled.
"""
import argparse
import ctypes
import json
from pathlib import Path
import runpy
import subprocess

import numpy as np

from check_convex_aabb import bit_comparison


def compiled_math(source, packed, directory, reuse):
    declarations = []
    for symbol, kind, values in packed.arrays:
        scalar = 'int' if kind == 'i' else 'float'
        data = ','.join(str(int(value)) if kind == 'i' else f'{float(value):.9e}f' for value in values.ravel())
        declarations.append(f'{scalar} {symbol}[{values.size}]={{{data}}};')
    common = '''#include <cmath>
using std::isfinite;
#define __device__
inline float __fadd_rn(float a,float b){return a+b;}
inline float __fsub_rn(float a,float b){return a-b;}
inline float __fmul_rn(float a,float b){return a*b;}
'''+ '\n'.join(declarations)+source
    prepare = '''static_convex::PreparedContact prepared;
int prepared_status=static_convex_prepare(0,x,x+3,x[6],x[7],x+8,x+11,prepared);'''
    contact = 'static_convex_contact(0,slot,x,x+3,x[6],x[7],x+8,x+11,y+13*slot,y+13*slot+1,y+13*slot+4)'
    vjp = ('static_convex_vjp_selected(prepared,slot,x,x+3,x[6],x[7],x+8,x+11,cot,gradient)'
           if reuse else 'static_convex_vjp(0,slot,x,x+3,x[6],x[7],x+8,x+11,cot,gradient)')
    wrapper = '''
extern "C" void evaluate(const float*x,float*y,float*jac,int*status){
@PREPARE@
for(int slot=0;slot<2;++slot){
status[slot]=@CONTACT@;
if(status[slot]!=1)continue;
for(int basis=0;basis<13;++basis){
float cot[13]={},gradient[6];cot[basis]=1.f;
int result=@VJP@;if(result!=1){status[slot]=result;break;}
for(int j=0;j<6;++j)jac[(slot*13+basis)*6+j]=gradient[j];
}}
}
'''.replace('@PREPARE@', prepare if reuse else '').replace('@CONTACT@', contact).replace('@VJP@', vjp)
    if reuse:
        wrapper = wrapper.replace('status[slot]='+contact+';',
                                  'status[slot]=prepared_status==1?'+contact+':prepared_status;')
        # Compile the actual adjoint dispatcher as well, with generated model
        # bindings. This exercises the primal-to-VJP cache handoff used by the
        # engine, separately from sharing a prepared selection across slots.
        from nablaengine.g1_engine import adjoint_contact
        dispatcher = adjoint_contact.DEVICE
        begin = dispatcher.index('__device__ __forceinline__ int adj_pair_geom(')
        end = dispatcher.index("// a geom's pose cotangent", begin)
        common += """
#define __forceinline__ inline
#define HAS_CONVEX 1
#define PG1(pair) 1
#define PG2(pair) 0
int c_gtype[2]={4,2}; float c_gsize[6]={}; float query_axis[3];
void zaxis_rn(const float*,float*axis){for(int k=0;k<3;++k)axis[k]=query_axis[k];}
int adj_contact_geom(...){return -1;}
float __int_as_float(int value){float result;__builtin_memcpy(&result,&value,sizeof(result));return result;}
"""+dispatcher[begin:end]
        wrapper += """
extern "C" void evaluate_adjoint(const float*x,float*y,float*jac,int*status){
float gx[2][3],gq[2][4];
for(int k=0;k<3;++k){gx[0][k]=x[8+k];gx[1][k]=x[k];query_axis[k]=x[3+k];}
for(int k=0;k<4;++k){gq[0][k]=x[11+k];gq[1][k]=k==0?1.f:0.f;}
c_gsize[3]=x[6];c_gsize[4]=x[7];
for(int slot=0;slot<2;++slot){
static_convex::PreparedContact prepared;float nu;
int branch=adj_pair_geom(0,slot,gx,gq,y+13*slot,y+13*slot+1,y+13*slot+4,&nu,nullptr,nullptr,&prepared);
status[slot]=branch<0?0:1;if(branch<0)continue;
for(int basis=0;basis<13;++basis){float cot[13]={},gb[12];cot[basis]=1.f;
branch=adj_pair_geom(0,slot,gx,gq,y+13*slot,y+13*slot+1,y+13*slot+4,&nu,cot,gb,&prepared);
if(branch<0){status[slot]=-1;break;}
for(int j=0;j<6;++j)jac[(slot*13+basis)*6+j]=gb[j];
}}
}
"""

    directory.mkdir(exist_ok=False)
    cpp = directory/'primitive.cpp'
    cpp.write_text(common+wrapper)
    result = subprocess.run(['g++', '-std=c++17', '-O2', '-ffp-contract=off', '-shared', '-fPIC',
                             str(cpp), '-o', str(directory/'primitive.so')], capture_output=True, text=True)
    (directory/'build.log').write_text(result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError(f'CPU numerical build failed, see {directory / "build.log"}')
    library = ctypes.CDLL(str(directory/'primitive.so'))
    evaluate = library.evaluate
    evaluate.argtypes = [ctypes.c_void_p]*4
    evaluate.restype = None
    functions = [evaluate]
    if reuse:
        adjoint = library.evaluate_adjoint
        adjoint.argtypes = [ctypes.c_void_p]*4
        adjoint.restype = None
        functions.append(adjoint)
    return functions


def math_check(args):
    from types import SimpleNamespace
    from scipy.spatial import ConvexHull
    from nablaengine.g1_engine.convex_contact import device_source
    from nablaengine.model.convex import pack_static_convex_hulls
    from check_static_convex_cuda_gradients import qualification_cases

    args.out.mkdir(parents=True, exist_ok=False)
    baseline = runpy.run_path(str(args.baseline_source))['device_source']()
    vertices = np.array([[x, y, z] for x in [-.6, .6] for y in [-.4, .4] for z in [-.3, .3]])
    faces = ConvexHull(vertices).simplices
    model = SimpleNamespace(geom_type=np.array([7]), geom_bodyid=np.array([0]), geom_dataid=np.array([0]),
        geom_pos=np.zeros((1, 3)), geom_quat=np.array([[1., 0., 0., 0.]]), nmesh=1,
        mesh_vertadr=np.array([0]), mesh_vertnum=np.array([8]), mesh_vert=vertices)
    packed = pack_static_convex_hulls(model, [0])
    functions = (compiled_math(baseline, packed, args.out/'baseline', False)
                 +compiled_math(device_source(), packed, args.out/'candidate', True))
    records = []
    for case in qualification_cases(vertices, faces):
        query = np.r_[case['center'], case['axis'], case['radius'], case['half_length'],
                      [0., 0., 0.], [1., 0., 0., 0.]].astype(np.float32)
        values, gradients, statuses = [], [], []
        for evaluate in functions:
            value, jac, status = np.zeros((2, 13), np.float32), np.zeros((2, 13, 6), np.float32), np.zeros(2, np.int32)
            evaluate(query.ctypes.data, value.ctypes.data, jac.ctypes.data, status.ctypes.data)
            values.append(value); gradients.append(jac); statuses.append(status)
        row = dict(case=case['name'], comparisons={})
        for index, name in [(1, 'standalone_forward_selected_vjp'), (2, 'adjoint_primal_vjp')]:
            row['comparisons'][name] = dict(status=bit_comparison(statuses[0], statuses[index]),
                values=bit_comparison(values[0], values[index]),
                jacobians=bit_comparison(gradients[0], gradients[index]))
        row['passed'] = all(value['equal'] for comparison in row['comparisons'].values()
                            for value in comparison.values())
        records.append(row)
    result = dict(passed=all(row['passed'] for row in records), generated_cases=len(records),
                  baseline_source=str(args.baseline_source), comparisons=records,
                  scope='CPU numerical source equivalence, two slots and 13 by 6 basis pose VJPs')
    (args.out/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


def compare(args):
    args.out.mkdir(parents=True, exist_ok=False)
    roots = (args.baseline, args.candidate)
    protocols = [json.loads((root/'protocol.json').read_text()) for root in roots]
    comparable = [{key: value for key, value in row.items() if key != 'code_revision'} for row in protocols]
    if comparable[0] != comparable[1] or any(row['resolved_macro']['value'] != 1 for row in protocols):
        raise ValueError('Selection reuse comparisons require identical source protocols with AABB enabled')
    selection = [json.loads((root/'selection.json').read_text()) for root in roots]
    comparisons = {}
    for filename in ('inputs.npz', 'arrays.npz', 'valid_tape.npz'):
        with np.load(roots[0]/filename, allow_pickle=False) as left, np.load(roots[1]/filename, allow_pickle=False) as right:
            if set(left.files) != set(right.files):
                raise ValueError(f'{filename} array names differ')
            for key in left.files:
                if filename == 'arrays.npz' and key == 'tape':
                    continue
                comparisons[f'{filename}/{key}'] = bit_comparison(left[key], right[key])
    reports = [json.loads((root/'report.json').read_text()) for root in roots]
    before, after = [row['timings']['median_forward_backward_wall_ms'] for row in reports]
    result = dict(same_selection=selection[0] == selection[1], comparisons=comparisons,
                  bitwise_equal=all(row['equal'] for row in comparisons.values()), conditions=reports,
                  forward_backward_wall_speed_ratio=before/after)
    result['passed'] = result['same_selection'] and result['bitwise_equal']
    (args.out/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    math = commands.add_parser('math')
    math.add_argument('--baseline-source', type=Path, required=True)
    math.add_argument('--out', type=Path, required=True)
    comparison = commands.add_parser('compare')
    comparison.add_argument('--baseline', type=Path, required=True)
    comparison.add_argument('--candidate', type=Path, required=True)
    comparison.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    math_check(args) if args.operation == 'math' else compare(args)


if __name__ == '__main__':
    main()

"""GPU checks for native self-contact measurements and state autograd.

The generated G1 retains its dynamics. Two added spheres, attached to pelvis and
left wrist, form its sole explicit contact pair. At qpos0 their centers are 0.1 m
apart, radii 0.06 m. The root is lifted 2 m, and all terrain pairs are removed.
This isolates the actual solved self-contact action/reaction at both bodies.

Compare measured/unmeasured state and gradient storage bits, count native calls,
and compare four substeps against a continuous three-plus-one split. Observational
measurement tensors must have no gradient. A separate generated two-contact row
example checks net vector cancellation through the production GPU measurement
helper. All outputs and builds stay in a new external output directory.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from capability_check import spec_from_model

PROTOCOL = dict(substeps=4, iterations=50, sphere_radius_m=.06, center_distance_m=.1,
                split_gradient_atol=1e-5, split_gradient_rtol=2e-5,
                reaction_atol=1e-5, cancellation_atol=0.,
                check='native measured state VJP plus retained self-contact force')


def make_model():
    import mujoco
    from nablaengine import fixtures
    from nablaengine.model import capability
    base, _ = fixtures.load_model(check=None)
    data = mujoco.MjData(base)
    mujoco.mj_kinematics(base, data)
    spec = spec_from_model(base)
    for pair in list(spec.pairs):
        spec.delete(pair)
    bodies = ['pelvis', 'left_wrist_yaw_link']
    center = data.xpos[base.body('pelvis').id].copy()
    for i, body in enumerate(bodies):
        body_id = base.body(body).id
        world = center+np.array([(-.5 if i == 0 else .5)*PROTOCOL['center_distance_m'], 0., 0.])
        local = data.xmat[body_id].reshape(3, 3).T@(world-data.xpos[body_id])
        spec.body(body).add_geom(name=f'measured_sphere_{i}', type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                 pos=local, size=[PROTOCOL['sphere_radius_m'], 0., 0.],
                                 mass=0., contype=0, conaffinity=0)
    spec.add_pair(name='measured_self_contact', geomname1='measured_sphere_0', geomname2='measured_sphere_1',
                  condim=3, friction=[1., 1., .005, .0001, .0001],
                  solref=base.pair_solref[0], solimp=base.pair_solimp[0])
    model = spec.compile()
    capability.require(model, [], capability.COM_BODY, 'engine', 'generated self-contact measurement input')
    return model, [model.body(body).id for body in bodies]


def cancellation_kernel(out):
    from nablaengine.g1_engine import measurements, step
    from check_box_contact_gradients import function
    header = r'''
#include <cuda_runtime.h>
#define NB 3
#define MAXCON 2
#define NFLOSS 0
#define PG1(pair) 0
#define PG2(pair) ((pair)+1)
__device__ int c_gbody[3]={1,2,0};
__device__ float c_floss[35]={};
__device__ void cross(const float*a,const float*b,float*c){
 c[0]=a[1]*b[2]-a[2]*b[1];c[1]=a[2]*b[0]-a[0]*b[2];c[2]=a[0]*b[1]-a[1]*b[0];
}
struct State {
 int ncon,con_pair[2],con_nrow[2],con_row[2],con_dim[2],efc_a[2];
 float con_mu[2],efc_D[2],efc_jar[2],con_frame[2][9],con_pos[2][3];
};
'''
    production = ''.join(function(step.STEP_DEVICE, name) for name in ('row_piece', 'row_cost', 'row_floss', 'row_eval'))
    runner = r'''
__global__ void check(float* output,int* counts){
 State s={};s.ncon=2;
 for(int c=0;c<2;c++){
  s.con_pair[c]=c;s.con_row[c]=c;s.con_nrow[c]=1;s.con_dim[c]=1;
  s.efc_D[c]=1.f;s.efc_jar[c]=-3.f;
  s.con_frame[c][0]=c==0?1.f:-1.f;
 }
 // Body 1 is geom1 of two contacts. Their actual forces are -3x and +3x.
 measure_terrain_wrenches(s,s,threadIdx.x,output,counts,output+18,counts+3);
}
extern "C" int run(float*out,int*count){
 float* d;int* c;cudaError_t error=cudaMalloc(&d,36*sizeof(float));if(error)return int(error);
 error=cudaMalloc(&c,6*sizeof(int));if(!error){check<<<1,32>>>(d,c);error=cudaGetLastError();}
 if(!error)error=cudaMemcpy(out,d,36*sizeof(float),cudaMemcpyDeviceToHost);
 if(!error)error=cudaMemcpy(count,c,6*sizeof(int),cudaMemcpyDeviceToHost);
 cudaFree(d);cudaFree(c);return int(error);
}
'''
    import ctypes
    source = out/'netforce.cu'
    library = out/'netforce.so'
    source.write_text(header+production+measurements.DEVICE+runner)
    result = subprocess.run(['nvcc', '-std=c++17', '-O2', '-shared', '-Xcompiler=-fPIC', str(source), '-o', str(library)],
                            text=True, capture_output=True)
    (out/'netforce-build.log').write_text(result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError('GPU netforce helper build failed')
    function = ctypes.CDLL(str(library)).run
    function.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    values = np.zeros(36, np.float32)
    counts = np.zeros(6, np.int32)
    error = function(values.ctypes.data, counts.ctypes.data)
    if error:
        raise RuntimeError(f'GPU netforce helper failed {error}')
    all_wrench = values[18:].reshape(3, 6)
    return dict(wrench=all_wrench.tolist(), counts=counts.tolist(),
                middle_body_cancels=bool(np.count_nonzero(all_wrench[1]) == 0),
                two_contacts_retained=bool(counts[4] == 2),
                reaction_balances=bool(np.count_nonzero(all_wrench.sum(axis=0)) == 0))


def run(out, report):
    import torch
    from nablaengine import torch_ffi as F
    model, bodies = make_model()
    os.environ['DK_CACHE_DIR'] = str(out/'build')
    os.environ.pop('DK_TORCH_ARTIFACT', None)
    cc = torch.cuda.get_device_capability()
    artifact = F.build(f'sm_{cc[0]}{cc[1]}', model=model, foot_pairs=[], set_current=False)
    F.init(artifact=artifact)
    report['artifact'] = str(artifact)
    q = model.qpos0.astype(np.float32)
    q[2] += 2.
    u = np.array([q[int(model.jnt_qposadr[int(j)])] for j in model.actuator_trnid[:, 0]])
    def tensor(value, grad=False):
        return torch.tensor(value, dtype=torch.float32, device='cuda', requires_grad=grad).contiguous()
    inputs = [tensor(q[None]), torch.zeros((1, model.nv), device='cuda'), tensor(u[None])]
    def evaluate(measured, split=False, kin=False):
        q0, v0, ctrl = [value.detach().clone().requires_grad_() for value in inputs]
        warm = torch.zeros_like(v0)
        friction, com = torch.ones(1, device='cuda'), torch.zeros((1, 3), device='cuda')
        F.read_diagnostics(reset=True)
        if split:
            prefix = F.step(q0, v0, warm, ctrl, friction, com, iterations=50, substeps=3,
                            clamp_kv=1, normal=0, measurements=measured)
            output = F.step(prefix[0], prefix[1], prefix[2], ctrl, friction, com, iterations=50, substeps=1,
                            clamp_kv=1, normal=0, kinematics=kin, measurements=measured)
        else:
            output = F.step(q0, v0, warm, ctrl, friction, com, iterations=50, substeps=4,
                            clamp_kv=1, normal=0, kinematics=kin, measurements=measured)
        weights_q = torch.linspace(-.4, .6, model.nq, device='cuda')
        weights_v = torch.linspace(.3, -.2, model.nv, device='cuda')
        loss = (output[0]*weights_q).sum()+(output[1]*weights_v).sum()
        if kin:
            loss = loss+.01*sum(x.sum() for x in output[3:8])
        loss.backward()
        gradients = [value.grad.detach() for value in (q0, v0, ctrl)]
        diagnostics = F.read_diagnostics(reset=True)
        result = dict(states=[x.detach() for x in output[:8 if kin else 3]], gradients=gradients, diagnostics=diagnostics)
        if measured:
            result['measurements'] = {key: (torch.cat((prefix[-1][key], output[-1][key]), dim=1) if split else output[-1][key])
                                      for key in F.MEASUREMENT_FIELDS}
            result['measurements_nondifferentiable'] = all(not x.requires_grad and x.grad_fn is None for x in output[-1].values())
        return result
    def bits(a, b):
        return bool(torch.equal(a.view(torch.int32), b.view(torch.int32)))
    report['comparisons'] = []
    for kin in (False, True):
        ordinary, measured, split = evaluate(False, kin=kin), evaluate(True, kin=kin), evaluate(True, split=True, kin=kin)
        parity = dict(kinematics=kin,
                      state_bits_equal=all(bits(a, b) for a, b in zip(ordinary['states'], measured['states'])),
                      gradient_bits_equal=all(bits(a, b) for a, b in zip(ordinary['gradients'], measured['gradients'])),
                      measurements_nondifferentiable=measured['measurements_nondifferentiable'],
                      one_forward_call=measured['diagnostics']['fwd_calls'] == 1,
                      split_state_bits_equal=all(bits(a, b) for a, b in zip(split['states'], measured['states'])),
                      split_gradient_close=all(bool(torch.allclose(a, b, atol=PROTOCOL['split_gradient_atol'], rtol=PROTOCOL['split_gradient_rtol']))
                                               for a, b in zip(split['gradients'], measured['gradients'])),
                      split_measurement_bits_equal=all(bits(split['measurements'][key], measured['measurements'][key])
                                                       for key in F.MEASUREMENT_FIELDS if key != 'substep_info'),
                      measured_diagnostics=measured['diagnostics'], split_diagnostics=split['diagnostics'])
        observed = measured['measurements']
        wrench = observed['all_contact_wrench'].detach().cpu().numpy()
        count = observed['all_contact_count'].detach().cpu().numpy()
        parity.update(terrain_zero=bool(torch.count_nonzero(observed['terrain_wrench']) == 0),
                      self_contact_retained=bool(np.all(count[0, :, bodies] > 0)),
                      strict_diagnostics=all(measured['diagnostics'][key] == 0 and split['diagnostics'][key] == 0
                          for key in ('fwd_unsupported_worlds', 'nonfinite_rows', 'replay_mismatch_worlds',
                                      'fwd_contact_overflow_worlds', 'fwd_row_overflow_worlds', 'nonconv_records')),
                      self_action_reaction=bool(np.allclose(wrench[0, :, bodies[0]]+wrench[0, :, bodies[1]], 0,
                                                           atol=PROTOCOL['reaction_atol'], rtol=0)))
        report['comparisons'].append(parity)
        np.savez_compressed(out/f'kinematics_{kin}.npz', **{key: value.cpu().numpy() for key, value in observed.items()})
    report['netforce_example'] = cancellation_kernel(out)
    checks = ['state_bits_equal', 'gradient_bits_equal', 'measurements_nondifferentiable', 'one_forward_call',
              'split_state_bits_equal', 'split_gradient_close', 'split_measurement_bits_equal', 'terrain_zero',
              'self_contact_retained', 'self_action_reaction', 'strict_diagnostics']
    report['pass'] = (all(row[key] for row in report['comparisons'] for key in checks)
                      and all(report['netforce_example'][key] for key in
                              ('middle_body_cancels', 'two_contacts_retained', 'reaction_balances')))
    print(json.dumps(dict(pass_=report['pass'], comparisons=report['comparisons'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--code-revision', required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.is_relative_to(Path(__file__).resolve().parents[1]):
        parser.error('--out must be outside checkout')
    out.mkdir(parents=True, exist_ok=False)
    report = dict(protocol=PROTOCOL, code_revision=args.code_revision, **{'pass': False})
    try:
        run(out, report)
        return 0 if report['pass'] else 1
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    raise SystemExit(main())

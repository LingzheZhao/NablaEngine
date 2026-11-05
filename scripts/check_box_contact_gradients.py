"""Qualify static-box primitives on generated inputs (no model, corpus, or rollout).

Run on provisioned compute with CUDA 12 nvcc, a supported NVIDIA GPU, Python,
NumPy and MuJoCo installed; PYTHONPATH=src python scripts/check_box_contact_gradients.py
--code-revision COMMITTED_REVISION --out /absolute/new/output. Builds and all machine-readable results stay there.

Forward float32 CUDA replay is compared to float64 C++ evaluation of the SAME
forward formulas. This is a numerical oracle, not an independent collision model.
Sphere face/edge/corner/inside cases additionally have a closed-form NumPy oracle.
Every one of 13 output basis cotangents is checked in six robot pose directions
(translation and world-axis rotation); box pose and shape remain fixed. Capsule
witness motion is included. Smooth-case gates are declared in PROTOCOL below;
feature/frame boundaries are retained and reported separately, without VJP gates.
This is primitive evidence only, not a rollout or trainer qualification.
"""
import argparse
import ctypes
import json
import platform
import re
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = dict(
    forward_atol=3e-6, forward_rtol=3e-6,
    gradient_atol=3e-4, gradient_rtol=3e-3,
    fd_steps=[1e-5, 5e-6], fd_stability_atol=2e-6, fd_stability_rtol=2e-5,
    negative_control_offset=0.05,
    description="All 13 basis output cotangents, all six robot pose directions; no percentile exclusion",
)


def function(source, name):
    """Extract a current named device function; fail if its interface disappears."""
    pat = r"(?:template <class S>\n)?__device__[^\n]*\b" + name + r"\("
    m = re.search(pat, source)
    if m is None:
        raise ValueError(f"missing current primitive {name}")
    begin, brace = m.start(), source.index("{", m.end())
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end] + "\n"


def forward_source():
    from nablaengine.g1_engine import source, step
    math = "\n".join(line for line in source.DEVICE.splitlines()
                     if line.startswith("#define R") and "(" in line)
    math += "\n" + "".join(function(source.DEVICE, n) for n in ("cross", "dot3", "cross_rn", "dot3_rn"))
    code = math + "".join(function(step.STEP_DEVICE, n) for n in
                         ("normalize3_rn", "make_frame", "mv3", "mtv3", "safe_div", "sphere_box",
                          "capsule_box_witness", "capsule_box_contact"))
    # Capture the original emitter independently of the slot replay helper.
    capture = r'''
struct Capture { bool active[2]; float y[26]; };
__device__ bool add_contact(Capture& s,int pair,int slot,float d,const float*p,const float*f){
  s.active[slot]=true;s.y[13*slot]=d;
  for(int k=0;k<3;k++)s.y[13*slot+1+k]=p[k];
  for(int k=0;k<9;k++)s.y[13*slot+4+k]=f[k];return true;
}
'''
    return code + capture + function(step.STEP_DEVICE, "capsule_box")


EVALUATE = r'''
__device__ int evaluate(const float*x,float*y,float*emitted){
  int kind=int(x[0]),slot=int(x[1]); const float *p=x+2,*a=x+5,*bp=x+10,*R=x+13,*bs=x+22;
  bool valid=true;
  if(kind==1){float n[3];sphere_box(p,x[8],bp,R,bs,y,y+1,n);make_frame(n,y+4);}
  else valid=capsule_box_contact(slot,p,a,x[8],x[9],bp,R,bs,y,y+1,y+4);
  if(!valid)return 0;
  if(kind==1){for(int k=0;k<13;k++)emitted[k]=y[k];return y[0]<0?3:1;}
  Capture s={};capsule_box(s,0,p,a,x[8],x[9],bp,R,bs);
  if(s.active[slot]){for(int k=0;k<13;k++)emitted[k]=s.y[13*slot+k];return 3;}
  return 1;
}
'''

SPHERE_VJP = r'''
__device__ bool static_box_primitive_vjp(int kind,int slot,const float*cp,const float*axis,float radius,float hl,
 const float*bp,const float*R,const float*bs,const float*cot,float*gb){
 if(kind==2)return static_box_capsule_vjp(slot,cp,axis,radius,hl,bp,R,bs,cot,gb);
 BoxDual p[3],b[3],rot[9],size[3],dist,pos[3],n[3],fr[9];
 for(int k=0;k<3;k++){p[k]=cp[k];p[k].d[k]=1.f;b[k]=bp[k];size[k]=bs[k];}
 for(int k=0;k<9;k++)rot[k]=R[k];
 dual_sphere_box(p,BoxDual(radius),b,rot,size,&dist,pos,n);make_frame(n,fr);
 for(int j=0;j<6;j++){gb[j]=cot[0]*dist.d[j];for(int k=0;k<3;k++)gb[j]+=cot[1+k]*pos[k].d[j];for(int k=0;k<9;k++)gb[j]+=cot[4+k]*fr[k].d[j];}
 return true;
}
'''

GPU = r'''
__global__ void check_kernel(int n,const float*x,const float*cot,float*y,float*emitted,float*g,int*valid){
  int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;
  x+=25*i;cot+=13*i;y+=13*i;emitted+=13*i;g+=6*i;
  valid[i]=evaluate(x,y,emitted);
  if(!valid[i])return;
  bool ok=static_box_primitive_vjp(int(x[0]),int(x[1]),x+2,x+5,x[8],x[9],x+10,x+13,x+22,cot,g);
  if(!ok)valid[i]=-1;
}
extern "C" int run(int n,const float*x,const float*c,float*y,float*e,float*g,int*v){
  float *dx=0,*dc=0,*dy=0,*de=0,*dg=0;int*dv=0;int err=0;
#define CK(OP) if(!err)err=int(OP)
  CK(cudaMalloc(&dx,n*25*sizeof(float))); CK(cudaMalloc(&dc,n*13*sizeof(float)));
  CK(cudaMalloc(&dy,n*13*sizeof(float))); CK(cudaMalloc(&de,n*13*sizeof(float)));
  CK(cudaMalloc(&dg,n*6*sizeof(float))); CK(cudaMalloc(&dv,n*sizeof(int)));
  CK(cudaMemcpy(dx,x,n*25*sizeof(float),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dc,c,n*13*sizeof(float),cudaMemcpyHostToDevice));
  if(!err){check_kernel<<<(n+63)/64,64>>>(n,dx,dc,dy,de,dg,dv);err=int(cudaGetLastError());}
  CK(cudaMemcpy(y,dy,n*13*sizeof(float),cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(e,de,n*13*sizeof(float),cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(g,dg,n*6*sizeof(float),cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(v,dv,n*sizeof(int),cudaMemcpyDeviceToHost));
  cudaFree(dx);cudaFree(dc);cudaFree(dy);cudaFree(de);cudaFree(dg);cudaFree(dv);return err;
#undef CK
}
'''


def build(out, prepare_only=False):
    from nablaengine.g1_engine import adjoint_contact
    base = forward_source()
    # Preserve the forward expressions but widen all arithmetic and constants for FD.
    cpu = base + EVALUATE + '''
extern "C" int reference(const float*x,float*y,float*e){return evaluate(x,y,e);}
extern "C" bool witness(const float*x,float*segment,float*offset){
 return capsule_box_witness(x+2,x+5,x[9],x+10,x+13,x+22,segment,offset);
}
'''
    cpu = re.sub(r"\bfloat\b", "double", cpu)
    cpu = re.sub(r"(?<=\d)f\b", "", cpu)
    for op in ("sqrt", "fabs", "fmin", "fmax"):
        cpu = cpu.replace(op + "f(", op + "(")
    cpu = cpu.replace("__device__", "").replace("__forceinline__", "inline")
    cpu = re.sub(r"#define R\w+\([^\n]+", "", cpu)
    cpu = '''#include <cmath>
#define RMUL(a,b) ((a)*(b))
#define RADD(a,b) ((a)+(b))
#define RSUB(a,b) ((a)-(b))
#define RFMA(a,b,c) std::fma(a,b,c)
#define RFMS(a,b,c) std::fma(-(a),b,c)
''' + cpu
    (out / "reference.cpp").write_text(cpu)
    gpu = "#include <cuda_runtime.h>\n" + base + adjoint_contact.static_box_dual_source() + SPHERE_VJP + EVALUATE + GPU
    (out / "primitive.cu").write_text(gpu)
    if prepare_only:
        return
    commands = [["g++", "-O2", "-std=c++17", "-shared", "-fPIC", str(out / "reference.cpp"),
                 "-o", str(out / "reference.so")],
                ["nvcc", "-O2", "-std=c++17", "-shared", "-Xcompiler=-fPIC", str(out / "primitive.cu"),
                 "-o", str(out / "primitive.so")]]
    for i, cmd in enumerate(commands):
        result = subprocess.run(cmd, capture_output=True, text=True)
        (out / f"build-{i}.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"build failed: see {out / f'build-{i}.log'}")


def rotation(w):
    angle = np.linalg.norm(w)
    if angle == 0:
        return np.eye(3)
    a = w / angle
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)


def cases():
    result = []
    def add(name, kind, p, a=(0, 0, 1), hl=0.4, slot=0, boundary=False, rotated=False):
        R = rotation(np.array([0.23, -0.31, 0.42])) if rotated else np.eye(3)
        bp = np.array([0.4, -0.3, 0.2])
        axis = np.asarray(a, float); axis /= np.linalg.norm(axis)
        x = np.r_[kind, slot, bp + R@p, R@axis, 0.3, hl, bp, R.ravel(), [1, 0.8, 0.6]]
        result.append(dict(name=name, boundary=boundary, x=x))
    for rotated in (False, True):
        suffix = "_rotated" if rotated else ""
        for name, p in (("face", [1.15, .12, .17]), ("edge", [1.1, .91, .13]),
                        ("corner", [1.1, .91, .71]), ("inside", [.81, .14, .12])):
            add("sphere_"+name+suffix, 1, p, rotated=rotated)
        add("capsule_endpoint"+suffix, 2, [1.2, .2, .9], a=[.2, .1, 1], rotated=rotated)
        add("capsule_interior"+suffix, 2, [1.12, .92, .11], a=[.1, -.2, 1], hl=.45, rotated=rotated)
        for slot in (0, 1):
            add(f"capsule_two_contacts_slot{slot}"+suffix, 2, [.2, .1, .75], a=[1, .17, .04],
                hl=.4, slot=slot, rotated=rotated)
    add("sphere_clamp_boundary", 1, [1, .95, .11], boundary=True)
    add("sphere_inside_face_tie", 1, [.5, .3, .1], boundary=True)
    # Normal's y component -0.5: make_frame changes its tangent basis here.
    add("sphere_frame_basis_boundary", 1, [1 + np.sqrt(3)*.1, .9, .1], boundary=True)
    add("capsule_parallel_boundary", 2, [.1, .1, .75], a=[1, 0, 0], boundary=True)
    return result


def sphere_reference(x):
    R, p, bp, bs, radius = x[13:22].reshape(3, 3), x[2:5], x[10:13], x[22:25], x[8]
    c = R.T@(p-bp); clamp = np.clip(c, -bs, bs); direction = clamp-c
    distance = np.linalg.norm(direction)
    if distance > 1e-15:
        n = direction/distance; pos = .5*(clamp+c+n*radius); d = distance-radius
    else:
        faces = np.abs(np.stack([-bs-c, bs-c], axis=1)).ravel()
        idx = np.argmin(faces); n = np.zeros(3); n[idx//2] = 1 if idx % 2 == 0 else -1
        pos = c+n*(radius-faces[idx])/2; d = -faces[idx]-radius
    n = R@n
    b = np.array([0., 1., 0.]) if -.5 < n[1] < .5 else np.array([0., 0., 1.])
    b -= n*(n@b); b /= np.linalg.norm(b)
    return np.r_[d, bp+R@pos, n, b, np.cross(n, b)]


def load_reference(out):
    lib = ctypes.CDLL(str(out / "reference.so"))
    fn = lib.reference; fn.argtypes = [ctypes.c_void_p]*3; fn.restype = ctypes.c_int
    def reference(x):
        x = np.ascontiguousarray(x, np.float64); y = np.full(13, np.nan); e = np.full(13, np.nan)
        valid = fn(x.ctypes.data, y.ctypes.data, e.ctypes.data)
        return valid, y, e
    witness_fn = lib.witness; witness_fn.argtypes = [ctypes.c_void_p]*3; witness_fn.restype = ctypes.c_bool
    def witness(x):
        x = np.ascontiguousarray(x, np.float64); segment = ctypes.c_double(); offset = ctypes.c_double()
        ok = witness_fn(x.ctypes.data, ctypes.byref(segment), ctypes.byref(offset))
        return dict(valid=bool(ok), segment=segment.value, offset=offset.value)
    reference.witness = witness
    return reference


def finite_difference(reference, x, h):
    jac = np.empty((13, 6)); validities = []
    for j in range(6):
        values = []
        for sign in (-1, 1):
            p = x.copy()
            if j < 3:
                p[2+j] += sign*h
            else:
                w = np.zeros(3); w[j-3] = sign*h; p[5:8] = rotation(w)@x[5:8]
            v, y, _ = reference(p); values.append(y); validities.append(v)
        jac[:, j] = (values[1]-values[0])/(2*h)
    return jac, validities


def close(actual, expected, atol, rtol):
    return bool(np.isfinite(actual).all() and np.isfinite(expected).all()
                and np.all(np.abs(actual-expected) <= atol+rtol*np.abs(expected)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--code-revision", required=True, help="declared committed revision of the staged source snapshot")
    ap.add_argument("--prepare-only", action="store_true", help="write generated sources; do not compile or run")
    args = ap.parse_args(); out = args.out.resolve()
    if out.is_relative_to(ROOT):
        ap.error("--out must be outside the checkout")
    out.mkdir(parents=True, exist_ok=False)
    report = dict(protocol=PROTOCOL, code_revision=args.code_revision,
                  inputs=[dict(c, x=c["x"].tolist()) for c in cases()], cases=[], pass_=False)
    try:
        import mujoco
        report["environment"] = dict(python=platform.python_version(), numpy=np.__version__,
                                     mujoco=mujoco.__version__, platform=platform.platform())
        for label, command in (("nvcc", ["nvcc", "--version"]), ("g++", ["g++", "--version"]),
                               ("gpu", ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])):
            try:
                info = subprocess.run(command, text=True, capture_output=True)
                report["environment"][label] = dict(returncode=info.returncode,
                                                     stdout=info.stdout.strip(), stderr=info.stderr.strip())
            except OSError as exc:
                report["environment"][label] = dict(error=str(exc))
        build(out, args.prepare_only)
        if args.prepare_only:
            report["status"] = "sources_prepared_without_execution"
            return 0
        reference = load_reference(out)
        lib = ctypes.CDLL(str(out / "primitive.so")); fn = lib.run
        fn.argtypes = [ctypes.c_int] + [ctypes.c_void_p]*6; fn.restype = ctypes.c_int
        tests = cases(); xs = np.repeat(np.stack([c["x"] for c in tests]).astype(np.float32), 13, axis=0)
        cots = np.tile(np.eye(13, dtype=np.float32), (len(tests), 1)); n = len(xs)
        ys = np.full((n, 13), np.nan, np.float32); emitted = ys.copy(); gs = np.full((n, 6), np.nan, np.float32)
        valid = np.zeros(n, np.int32)
        err = fn(n, *[a.ctypes.data for a in (xs, cots, ys, emitted, gs, valid)])
        if err:
            raise RuntimeError(f"CUDA error {err}")
        for i, c in enumerate(tests):
            sl = slice(13*i, 13*(i+1)); x = xs[13*i].astype(float)
            # Use exactly the GPU's rounded input in the high precision oracle.
            vr, yr, er = reference(x); jac, vf = finite_difference(reference, x, PROTOCOL["fd_steps"][0])
            jac2, vf2 = finite_difference(reference, x, PROTOCOL["fd_steps"][1])
            forward_ok = bool(np.all(valid[sl] == vr)) and close(ys[sl], yr, PROTOCOL["forward_atol"], PROTOCOL["forward_rtol"])
            emitter_ok = vr == 3 and close(emitted[sl], ys[sl], PROTOCOL["forward_atol"], PROTOCOL["forward_rtol"])
            if int(x[0]) == 1:
                forward_ok &= close(yr, sphere_reference(x), PROTOCOL["forward_atol"], PROTOCOL["forward_rtol"])
            fd_stable = close(jac, jac2, PROTOCOL["fd_stability_atol"], PROTOCOL["fd_stability_rtol"])
            topology_stable = all(v == vr for v in vf+vf2)
            gradient_ok = close(gs[sl], jac2, PROTOCOL["gradient_atol"], PROTOCOL["gradient_rtol"])
            negative_control = not close(gs[sl]+PROTOCOL["negative_control_offset"], jac2,
                                         PROTOCOL["gradient_atol"], PROTOCOL["gradient_rtol"])
            witness = reference.witness(x) if int(x[0]) == 2 else None
            coverage_ok = True
            if witness is not None and not c["boundary"]:
                segment = witness["segment"]
                coverage_ok = witness["valid"]
                if "endpoint" in c["name"]:
                    coverage_ok &= abs(segment) == 1
                if "interior" in c["name"]:
                    coverage_ok &= abs(segment) < .95
                if "two_contacts" in c["name"]:
                    coverage_ok &= witness["offset"] > -3 and abs(witness["offset"]) > .1
            passed = (forward_ok and emitter_ok and fd_stable and topology_stable and gradient_ok
                      and negative_control and coverage_ok)
            row = dict(name=c["name"], boundary=c["boundary"], valid=int(vr), forward_ok=forward_ok,
                       witness=witness, coverage_ok=bool(coverage_ok),
                       original_emitter_ok=emitter_ok, fd_stable=fd_stable, topology_stable=topology_stable,
                       gradient_ok=gradient_ok, negative_control_rejected=negative_control,
                       max_gradient_abs_error=float(np.max(np.abs(gs[sl]-jac2))),
                       resolved_input=x.tolist(), jacobian=gs[sl].tolist(), reference_jacobian=jac2.tolist(),
                       forward=ys[13*i].tolist(),
                       pass_=None if c["boundary"] else bool(passed))
            report["cases"].append(row)
            print(c["name"], "reported_boundary" if c["boundary"] else "PASS" if passed else "FAIL", flush=True)
        report["pass_"] = all(c["pass_"] for c in report["cases"] if not c["boundary"])
        return 0 if report["pass_"] else 1
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=True)+"\n")


if __name__ == "__main__":
    raise SystemExit(main())

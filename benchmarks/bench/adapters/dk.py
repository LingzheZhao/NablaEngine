"""Adapters of our engine (engine ids dk, dk_iface, dk_jac): the G1 engine through its public JAX API,
nablaengine.jax_ffi, under jax.jit, float32, Track A only (the engine is built for the mjlab G1; it is not a Track B
engine).

  python -m benchmarks.bench run --engines dk,dk_jac --quick --out runs/bench/test_dk/
  python -m benchmarks.bench run --engines dk_iface --nworld 4096 --out runs/bench/test_dk/
  python -m benchmarks.bench run --engines dk --variant dk=tol6 ...          # the matched-tolerance row

Engine ids (one class each; the gradient mode is fixed per id, so it is part of every cell id):
  dk        the default build per architecture, exact gradient mode: clamp_kv=1 (no -kv term on clamped dofs: the
            derivative of the clamped actuator force), normal=0 (exact self-contact normal; the near-parallel rule is
            off with it). The headline row for time and accuracy.
  dk_iface  the gradient mode of jax_ffi's interface defaults, a labelled row: clamp_kv=0 (keeps -kv on clamped dofs,
            the semantics of mjw_pr1535_fix), normal=3 (the self contacts' normal frozen below 1e-3 m; near-parallel
            rule on).
  dk_jac    the Jacobian-equivalent of dk (same exact mode): one forward with the tape and the nq + nv = 71 VJPs of
            the unit output cotangents, one per row of the (nq + nv) x (2 nv + nu) = 71 x 99 Jacobian, so MuJoCo C's
            FD Jacobian, MJX's forward-mode Jacobian and ours are compared on the same object.

Ops (every call starts from the loaded input states, all inputs resident on the GPU; AOT-compiled jax.jit functions,
XLA defaults; the FFI library is built by nvcc once per architecture and model and cached, jax_ffi.init):
  fwd        jax_ffi.step outside differentiation: the forward kernel without the tape; returns qpos', qvel', warm'
  fwd_train  jax_ffi.forward(..., tape=True): the forward a backward needs (outputs, info and the tape)
  bwd        jax_ffi.backward(tape, friction, com, cot_qpos, cot_qvel) of the stored cotangent 0, on the tape of one
             fwd_train made in prepare("bwd")
  vjp        jax.vjp(jax_ffi.step) and its pullback of the stored cotangent 0 in one jit (dk_step_fwd + dk_step_bwd)
  jac        (dk_jac only) forward with the tape, then the 71 VJPs tiled into the batch: rows_per_call =
             min(71, max(1, jac_worlds // nworld)) unit cotangents per backward call on a tape tiled that many times
             (one call per row when nworld >= jac_worlds); the qpos-form columns are mapped to the tangent form on
             the GPU (base.qpos_tangent's formula at the input qpos). Output (nworld, 71, 99) float32: rows qpos'
             (qpos form, 36) then qvel' (35); columns qpos tangent (35), qvel (35), ctrl (29), as MuJoCo C's chained
             FD (mjc).
The other ops of an id are not_supported with a pointer to the id that has them.

Options (run --opt <id>.<key>=<value>; the defaults are the id's):
  clamp_kv, normal  the adjoint variant (jax_ffi.VARIANTS); a value other than the id's default is recorded in
                    provenance (config_labels) and labels the cell non-default (diag extra build_label)
  tolerance         None (default: the problem's, 1e-8) or the Newton tolerance: a model constant of the engine, so
                    a library of its own (built on first use). Variant "tol6" of dk: 1e-6, MuJoCo Warp's internal value
                    (the matched-tolerance sensitivity row)
  jac_worlds        worlds per backward call of the jac op (default None: JAC_WORLDS_PER_SM = 8 x the GPU's SM count,
                    640 on the V100, where 512-1024 measured fastest at 64-1024 worlds: tiling a small batch fills the
                    GPU, tiling a large one only copies the tape); the results are bitwise independent of it

Build record (provenance()['build']): jax_ffi.init()'s library key, architecture, architecture flags per unit,
forward-unit -D options, PTX front ends (sm_120), cache hit, nablaengine version, commit and dirty flag, plus the
expected default build of the architecture (nablaengine.cuda's REG_CAPS / N7_VIRTUAL tables with every build switch
unset) and the build switches that are set. A build that differs from the default, a dirty source tree or a DK_ARCH
that is not the GPU's (provenance build.labels, build.default False), and options that differ from the id's defaults
(gradient mode, tolerance: provenance config_labels) are reported on stderr and in provenance label and every cell's
diag extra build_label ("default" or "non-default: ..."); only a library that cannot be built fails (status error).

Diagnostics (of one forward with the tape and one backward of cotangent 0 from the loaded states, after the timing):
contacts and constraint rows per world and substep and Newton iterations per world and substep from the tape records'
headers (niter_hist counts worlds by their largest substep iteration count), overflow (worlds whose contacts exceeded
the 96-contact capacity or whose rows exceeded 416: unsupported bits 4 / 16 of the forward's info), unsupported (bits
1 / 2 / 8: a geometry case outside the engine's scope), and jax_ffi.read_diagnostics()'s counters of those two calls
(iteration-limit worlds, refactored / non-converged / singular tape records, non-finite gradient rows, replay
mismatches).
"""

import os
import sys
import time

import numpy as np

from benchmarks.bench import prov
from benchmarks.bench import problem as P
from benchmarks.bench.adapters import base

EXACT = dict(clamp_kv=1, normal=0)
IFACE = dict(clamp_kv=0, normal=3)  # jax_ffi's interface defaults
JAC_WORLDS_PER_SM = 8  # default jac_worlds = this x the GPU's SM count (option jac_worlds)
JAC_WORLDS_FALLBACK = 1024  # where JAX does not report the SM count
# unsupported bits of jax_ffi.forward's info[:, 3] (g1_engine/step.py, accumulated over the substeps)
UNSUP_OTHER, UNSUP_POINTS, UNSUP_CONTACTS, UNSUP_MULTICCD, UNSUP_ROWS = 1, 2, 4, 8, 16
OVERFLOW_BITS = UNSUP_CONTACTS | UNSUP_ROWS
UNSUPPORTED_BITS = UNSUP_OTHER | UNSUP_POINTS | UNSUP_MULTICCD
# environment variables that select the FFI library's build or the engine's capacity (docs/configuration.md); all
# are recorded when set. Their effect on the build is judged from the flags init() reports (arch flags, forward
# defines, front ends, the GPU's architecture); LABELLED ones label the row whenever they are set.
BUILD_VARS = ("DK_ARCH", "DK_SM120_FRONTEND", "DK_SM120_FORWARD_FRONTEND", "DK_SM90_REG_CAPS", "DK_SM120_PRE_CAP",
              "DK_TEST_FORWARD_VIRTUAL_ARCH", "DK_TEST_VIRTUAL_ARCH", "DK_MAXCON")
LABELLED_VARS = ("DK_TEST_FORWARD_VIRTUAL_ARCH", "DK_TEST_VIRTUAL_ARCH", "DK_MAXCON")
OPS = ("fwd", "fwd_train", "bwd", "vjp")


def _jax():
    import jax
    import jax.numpy as jnp
    return jax, jnp


def qpos_tangent_jnp(g, q):
    """base.qpos_tangent on device: qpos-form gradients g (..., nq) at states q (..., nq) (broadcast) -> (..., nv)."""
    _, jnp = _jax()
    w, x, y, z = q[..., 3], q[..., 4], q[..., 5], q[..., 6]
    a, b, c, d = g[..., 3], g[..., 4], g[..., 5], g[..., 6]
    s = 0.5 / (w * w + x * x + y * y + z * z)
    v = jnp.stack([w * b - x * a - y * d + z * c, w * c + x * d - y * a - z * b, w * d - x * c + y * b - z * a], -1)
    return jnp.concatenate([g[..., :3], v * s[..., None], g[..., 7:]], -1)


def expected_build(arch):
    """The default build of an architecture with every build switch unset: (arch flags of the forward and adjoint
    units, the forward unit's -D options), from nablaengine.cuda's tables (cuda.py's module docstring)."""
    from nablaengine import cuda
    virt = cuda.N7_VIRTUAL.get(arch)
    flags = [f"-arch={virt}", f"-code={arch}"] if virt else [f"-arch={arch}"]
    defines = [f"-D{c}" for c in cuda.REG_CAPS[arch][1]] if arch in cuda.REG_CAPS else []
    return dict(arch_flags=dict(forward=flags, adjoint=flags), forward_defines=defines)


def build_record(info, device_arch=None):
    """provenance()['build'] from jax_ffi.init()'s info: the library and how it differs from the default build
    (device_arch: the GPU's architecture, default nablaengine.cuda.device_arch())."""
    from nablaengine import cuda
    switches = {k: os.environ[k] for k in BUILD_VARS if os.environ.get(k)}
    exp = expected_build(info["arch"])
    labels = []
    if info["arch_flags"] != exp["arch_flags"]:
        labels.append(f"arch flags {info['arch_flags']} (default {exp['arch_flags']})")
    if list(info["forward_defines"]) != exp["forward_defines"]:
        labels.append(f"forward defines {info['forward_defines']} (default {exp['forward_defines']})")
    fe = info.get("front_end")
    if info["arch"] in cuda.N7_VIRTUAL and not all(str(v or "").startswith("NVVM 7.") for v in (fe or {}).values()):
        labels.append(f"front ends {fe} (default NVVM 7 on {info['arch']})")
    dev = device_arch
    if dev is None:
        try:
            dev = cuda.device_arch()
        except Exception:  # noqa: BLE001
            dev = None
    if dev is not None and dev != info["arch"]:
        labels.append(f"built for {info['arch']} on a {dev} GPU")
    labels += [f"{k}={v}" for k, v in switches.items() if k in LABELLED_VARS]
    if info.get("dirty"):
        labels.append("dirty source tree")
    if info.get("commit") is None:
        labels.append("no source commit (not a git checkout)")
    return dict(default=not labels, labels=labels, key=info["key"], arch=info["arch"], device_arch=dev,
                library=prov.rel(info["path"]), cache_hit=not info["built"], version=info.get("version"),
                commit=info.get("commit"), dirty=info.get("dirty"), arch_flags=info["arch_flags"],
                forward_defines=list(info["forward_defines"]), front_end=fe, forward_unit=info.get("forward_unit"),
                expected=exp, switches=switches, maxcon=info.get("maxcon"), nefc=info.get("nefc"),
                rec_words=info.get("rec_words"), consts_sha256=info.get("consts_sha256"))


class DkAdapter(base.Adapter):
    id = "dk"
    label = "NablaEngine (jax_ffi, exact gradient mode)"
    native = "ours"
    device = "gpu"
    tracks = ("A",)
    variants = {"default": {}, "tol6": {"tolerance": 1e-6}}
    mode = EXACT
    timed_ops = OPS

    # ------------------------------------------------------------------ setup
    def setup(self, problem, states, nworld, iterations, clamp_kv=None, normal=None, tolerance=None,
              jac_worlds=None):
        jax, _ = _jax()
        from nablaengine import jax_ffi as F
        self.F = F
        if iterations is None:
            raise ValueError("dk: iterations must be set (Track A: 10 or 50)")
        if problem.track != "A" or not problem.dr:
            raise base.NotSupported(f"dk runs Track A only (problem {problem.id})")
        kv = self.mode["clamp_kv"] if clamp_kv is None else int(clamp_kv)
        nrm = self.mode["normal"] if normal is None else int(normal)
        if (kv, nrm) not in F.VARIANTS:
            raise ValueError(f"dk: no adjoint variant clamp_kv={kv} normal={nrm} (jax_ffi.VARIANTS)")
        self.kv, self.nrm = kv, nrm
        self.it, self.sub = int(iterations), int(problem.substeps)
        self.jac_worlds = None if jac_worlds is None else int(jac_worlds)
        if self.jac_worlds is not None and self.jac_worlds < 1:
            raise ValueError("dk: jac_worlds must be >= 1")
        self.nworld = int(nworld)
        self.problem = problem
        t0 = time.perf_counter()
        m = P.load_model(problem, iterations)
        if tolerance is not None:
            m.opt.tolerance = float(tolerance)
        self.m = m
        states.check(m.nq, m.nv, m.nu, dr=True)
        base.check_layout(m.nq, m.nv)
        self.nq, self.nv, self.nu = m.nq, m.nv, m.nu
        model_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.info = F.init(model=m, foot_pairs=problem.foot_pairs)  # builds (nvcc) or loads the cached library
        build_s = time.perf_counter() - t0
        import mujoco
        torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, problem.torso_body)
        if torso != self.info["torso_id"]:
            raise ValueError(f"dk: the problem's torso body {problem.torso_body!r} (id {torso}) is not the engine's "
                             f"(id {self.info['torso_id']})")
        self.build = build_record(self.info)  # build.default / build.labels: the library's build only
        self.config_labels = []  # options of this row that differ from the engine id's defaults
        if (kv, nrm) != (self.mode["clamp_kv"], self.mode["normal"]):
            self.config_labels.append(f"gradient mode clamp_kv={kv} normal={nrm} (the {self.id} default is "
                                      f"clamp_kv={self.mode['clamp_kv']} normal={self.mode['normal']})")
        if tolerance is not None:
            self.config_labels.append(f"tolerance {float(tolerance):g} (problem: {problem.tolerance:g})")
        labels = self.build["labels"] + self.config_labels
        self.build_label = "default" if not labels else "non-default: " + "; ".join(labels)
        if labels:
            print(f"[dk] {self.build_label}", file=sys.stderr, flush=True)
        t0 = time.perf_counter()
        dev = jax.devices("gpu")[0]
        x = states.inputs(np.float32)
        put = lambda a: jax.device_put(np.ascontiguousarray(a, np.float32), dev)  # noqa: E731
        self.args = tuple(put(x[k]) for k in ("qpos", "qvel", "qacc_warmstart", "ctrl", "friction", "com"))
        self.qpos64 = states.qpos.astype(np.float64)
        self.cot = None
        if states.cot_qpos is not None:
            self.cot = tuple(put(c) for c in states.cotangent(0, np.float32))
        jax.block_until_ready(self.args)
        load_s = time.perf_counter() - t0
        self.dev, self.fns, self.compile_s, self.tape, self.last = dev, {}, {}, None, None
        if self.jac_worlds is None:
            cores = getattr(dev, "core_count", None)
            self.jac_worlds = JAC_WORLDS_PER_SM * int(cores) if cores else JAC_WORLDS_FALLBACK
        return dict(build_s=build_s, load_s=load_s, model_s=model_s)

    # ------------------------------------------------------------------ the jitted functions
    def _kw(self):
        return dict(iterations=self.it, substeps=self.sub)

    def _make(self, name):
        """(the jax.jit function, its arguments) of one computation."""
        jax, jnp = _jax()
        F, kw, kv, nrm = self.F, self._kw(), self.kv, self.nrm
        a = self.args
        if name == "fwd":
            return jax.jit(lambda *x: F.step(*x, clamp_kv=kv, normal=nrm, **kw)), a
        if name == "fwd_train":
            return jax.jit(lambda *x: F.forward(*x, tape=True, **kw)), a
        if name == "fwd_info":  # diagnostics: the forward with the tape and its info
            return jax.jit(lambda *x: F.forward(*x, tape=True, **kw)), a
        if name == "bwd":
            def bwd(tp, fr, com, cq, cv):
                return F.backward(tp, fr, com, cq, cv, substeps=self.sub, clamp_kv=kv, normal=nrm)
            return jax.jit(bwd), None  # arguments given at the call (the tape)
        if name == "vjp":
            def vjp(*x):
                q, v, w, u, fr, com, cq, cv = x
                outs, pull = jax.vjp(lambda *y: F.step(*y, clamp_kv=kv, normal=nrm, **kw), q, v, w, u, fr, com)
                g = pull((cq, cv, jnp.zeros_like(outs[2])))
                return g[0], g[1], g[3]
            return jax.jit(vjp), None
        if name == "jac":
            return jax.jit(self._jac_fn()), a
        raise KeyError(name)

    def _jac_fn(self):
        """The Jacobian-equivalent (module docstring): forward with the tape, then the R = nq + nv unit-cotangent VJPs
        in chunks of rows_per_call rows tiled into the batch."""
        jax, jnp = _jax()
        F, kw, kv, nrm = self.F, self._kw(), self.kv, self.nrm
        B, nq, nv = self.nworld, self.nq, self.nv
        R = nq + nv
        r = max(1, min(R, self.jac_worlds // B))
        self.jac_chunks = [(i, min(R, i + r)) for i in range(0, R, r)]
        E = np.eye(R, dtype=np.float32)

        def jac(qpos, qvel, warm, ctrl, fr, com):
            tape = F.forward(qpos, qvel, warm, ctrl, fr, com, tape=True, **kw)[4]
            gq, gv, gu = [], [], []
            for a, b in self.jac_chunks:
                k = b - a
                e = jnp.asarray(E[a:b])
                cq, cv = jnp.repeat(e[:, :nq], B, axis=0), jnp.repeat(e[:, nq:], B, axis=0)  # row a + j: j*B + w
                if k > 1:
                    tp, f, c = jnp.tile(tape, (k, 1)), jnp.tile(fr, k), jnp.tile(com, (k, 1))
                else:
                    tp, f, c = tape, fr, com
                dq, dv, du, _ = F.backward(tp, f, c, cq, cv, substeps=self.sub, clamp_kv=kv, normal=nrm)
                gq.append(dq.reshape(k, B, -1))
                gv.append(dv.reshape(k, B, -1))
                gu.append(du.reshape(k, B, -1))
            Gq, Gv, Gu = (jnp.swapaxes(jnp.concatenate(g, 0), 0, 1) for g in (gq, gv, gu))  # (B, R, .)
            return jnp.concatenate([qpos_tangent_jnp(Gq, qpos[:, None, :]), Gv, Gu], -1)
        return jac

    def _fn(self, name, args=None):
        """The AOT-compiled function of a computation (compiled on first use; compile_s recorded)."""
        if name not in self.fns:
            f, a = self._make(name)
            a = a if a is not None else args
            t0 = time.perf_counter()
            c = f.lower(*a).compile()
            self.compile_s[name] = time.perf_counter() - t0
            self.fns[name] = c
        return self.fns[name]

    def _bwd_args(self):
        return (self.tape, self.args[4], self.args[5]) + self.cot

    def _vjp_args(self, cot=None):
        return self.args + (self.cot if cot is None else cot)

    # ------------------------------------------------------------------ ops
    def ops(self):
        out = {}
        if "fwd" in self.timed_ops:
            out["fwd"] = self._op_fwd
        if "fwd_train" in self.timed_ops:
            out["fwd_train"] = self._op_fwd_train
        if self.cot is not None:
            if "bwd" in self.timed_ops:
                out["bwd"] = self._op_bwd
            if "vjp" in self.timed_ops:
                out["vjp"] = self._op_vjp
        if "jac" in self.timed_ops:
            out["jac"] = self._op_jac
        return out

    def not_supported(self):
        out = {}
        for op in base.OPS:
            if op in self.timed_ops:
                if op in ("bwd", "vjp") and self.cot is None:
                    out[op] = f"{self.id}: the corpus has no stored cotangents"
            elif op == "jac":
                out[op] = (f"{self.id}: the Jacobian-equivalent (71 VJPs) is engine dk_jac's row (the dk_iface "
                           "gradient mode has none)" if self.id != "dk" else
                           "dk: the Jacobian-equivalent (71 VJPs of the exact mode) is engine dk_jac's row")
            else:
                out[op] = f"{self.id}: only the Jacobian-equivalent is timed here; {op} is engine dk's row"
        return out

    def _op_fwd(self):
        self.last = self.fns["fwd"](*self.args)
        return self.last

    def _op_fwd_train(self):
        self.last = self.fns["fwd_train"](*self.args)
        return self.last

    def _op_bwd(self):
        self.last = self.fns["bwd"](*self._bwd_args())
        return self.last

    def _op_vjp(self):
        self.last = self.fns["vjp"](*self._vjp_args())
        return self.last

    def _op_jac(self):
        self.last = self.fns["jac"](*self.args)
        return self.last

    def prepare(self, op):
        jax, _ = _jax()
        out = {}
        if op == "bwd":
            self._fn("fwd_train")
            t0 = time.perf_counter()
            self.tape = jax.block_until_ready(self.fns["fwd_train"](*self.args)[4])  # the residuals bwd pulls back
            out["tape_s"] = time.perf_counter() - t0
            self._fn("bwd", self._bwd_args())
        elif op == "vjp":
            self._fn("vjp", self._vjp_args())
        else:
            self._fn(op)
        out["compile_s"] = self.compile_s[op]
        return out

    def sync(self):
        jax, _ = _jax()
        if self.last is not None:
            jax.block_until_ready(self.last)

    # ------------------------------------------------------------------ outputs, gradients, diagnostics
    def outputs(self):
        jax, _ = _jax()
        q, v, w = jax.device_get(self._fn("fwd")(*self.args))
        return dict(qpos=np.asarray(q, np.float64), qvel=np.asarray(v, np.float64),
                    qacc_warmstart=np.asarray(w, np.float64))

    def gradient(self, cot_qpos, cot_qvel):
        jax, _ = _jax()
        cot = tuple(jax.device_put(np.ascontiguousarray(c, np.float32), self.dev) for c in (cot_qpos, cot_qvel))
        gq, gv, gu = jax.device_get(self._fn("vjp", self._vjp_args(cot))(*self._vjp_args(cot)))
        return dict(qpos=base.qpos_tangent(np.asarray(gq, np.float64), self.qpos64), qvel=np.asarray(gv, np.float64),
                    ctrl=np.asarray(gu, np.float64))

    def jacobian(self):
        jax, _ = _jax()
        return np.asarray(jax.device_get(self._fn("jac")(*self.args)), np.float64)

    def diagnostics(self):
        jax, _ = _jax()
        F = self.F
        F.read_diagnostics(reset=True)  # drop what the timed calls counted
        q, v, w, info, tape = jax.device_get(self._fn("fwd_info")(*self.args))
        if self.cot is not None:
            jax.block_until_ready(self._fn("vjp", self._vjp_args())(*self._vjp_args()))
        counters = F.read_diagnostics(reset=True)
        info = np.asarray(info)
        rec = self._records(np.asarray(tape))
        niter_w = rec["niter"].max(1)
        unsup = info[:, 3]
        return dict(ncon_mean=float(rec["ncon"].mean()), ncon_max=int(rec["ncon"].max()),
                    nefc_mean=float(rec["nefc"].mean()), nefc_max=int(rec["nefc"].max()),
                    niter_mean=float(rec["niter"].mean()), niter_max=int(rec["niter"].max()),
                    niter_hist=np.bincount(niter_w, minlength=int(niter_w.max()) + 1).tolist(),
                    overflow=int(((unsup & OVERFLOW_BITS) != 0).sum()),
                    unsupported=int(((unsup & UNSUPPORTED_BITS) != 0).sum()),
                    extra=dict(build_label=self.build_label, gradient_mode=dict(clamp_kv=self.kv, normal=self.nrm),
                               scope="ncon / nefc / niter per world and substep from the tape records; niter_hist: "
                                     "worlds by their largest substep iteration count",
                               overflow_contacts=int(((unsup & UNSUP_CONTACTS) != 0).sum()),
                               overflow_rows=int(((unsup & UNSUP_ROWS) != 0).sum()),
                               iteration_limit_records=int((rec["niter"] >= self.it).sum()),
                               nonconv_worlds=int(((rec["flags"] & self._flag("FLAG_NONCONV")) != 0).any(1).sum()),
                               counters=counters, counters_scope="one forward with the tape + one backward of "
                                                                 "cotangent 0 (jax_ffi.read_diagnostics)"
                                                                 if self.cot is not None else
                                                                 "one forward with the tape"))

    @staticmethod
    def _flag(name):
        from nablaengine.g1_engine import tape as g1_tape
        return getattr(g1_tape, name)

    def _records(self, tape):
        """{ncon, nefc, niter, flags}: (nworld, substeps) int64 from the tape records' header and flags words."""
        from nablaengine.g1_engine import tape as g1_tape
        lay = g1_tape.layout(self.m)
        u = np.ascontiguousarray(tape, np.float32).view(np.uint32).reshape(self.nworld, self.sub, -1)
        hdr = u[..., lay["hdr"][0]].astype(np.int64)
        flags = u[..., lay["flags"][0]].astype(np.int64)
        return dict(ncon=hdr & 127, nefc=(hdr >> 7) & 511, niter=hdr >> 24, flags=flags)

    def memory(self):
        out = dict(tape_bytes=int(self.F.tape_bytes(self.nworld, self.sub)))
        try:
            st = self.dev.memory_stats() or {}
            out.update(alloc_peak_bytes=st.get("peak_bytes_in_use"), alloc_bytes_in_use=st.get("bytes_in_use"),
                       alloc_peak_scope="the process: every op of the unit and the diagnostics (XLA's allocator)")
        except Exception:  # noqa: BLE001
            pass
        return out

    def provenance(self):
        import jax
        out = dict(engine="nablaengine.jax_ffi", build=self.build, config_labels=self.config_labels,
                   label=self.build_label, compile_s=dict(self.compile_s),
                   options=dict(iterations=self.it, substeps=self.sub, clamp_kv=self.kv, normal=self.nrm,
                                near_parallel_rule=self.nrm != 0, tolerance=float(self.m.opt.tolerance),
                                ls="the engine's exact line search (MuJoCo's ls_iterations does not apply)",
                                contacts_per_world=self.info.get("maxcon"), rows_per_world=self.info.get("nefc")),
                   problem=P.options(self.m), jax=jax.__version__, device=str(self.dev.device_kind),
                   launch=self.F.launch_info())
        if "jac" in self.timed_ops and getattr(self, "jac_chunks", None):
            out["jac"] = dict(jac_worlds=self.jac_worlds, rows_per_call=self.jac_chunks[0][1] - self.jac_chunks[0][0],
                              backward_calls=len(self.jac_chunks))
        return out

    def semantics(self):
        grad = ("analytic adjoint (tape replay) of the converged forward, discrete decisions frozen (contact set, "
                "active rows, clamped dofs); ")
        if (self.kv, self.nrm) == (1, 0):
            grad += "exact mode clamp_kv=1, normal=0 (near-parallel rule off)"
        elif (self.kv, self.nrm) == (0, 3):
            grad += ("interface-default mode clamp_kv=0 (-kv kept on clamped dofs), normal=3 (self-contact normal "
                     "frozen below 1e-3 m, near-parallel rule on)")
        else:
            grad += f"clamp_kv={self.kv}, normal={self.nrm}"
        out = dict(gradient=grad, solver_exit="per world (tolerance or the iteration cap)", precision="float32",
                   capacity=f"{self.info.get('maxcon')} contacts / {self.info.get('nefc')} rows per world; overflow "
                            "counted (diag overflow), not an error")
        if "jac" in self.timed_ops:
            out["jacobian"] = ("Jacobian-equivalent: 71 VJPs of unit output cotangents (rows qpos' in qpos form, then "
                               "qvel'), tiled into the batch; columns qpos tangent, qvel, ctrl")
        return out

    def close(self):
        self.fns, self.tape, self.last, self.args = {}, None, None, ()

    @classmethod
    def check_env(cls):
        import jax
        gpus = jax.devices("gpu")
        from nablaengine import cuda
        return dict(jax=jax.__version__, gpu=gpus[0].device_kind, arch=cuda.device_arch(), nvcc=prov.nvcc())


class DkIfaceAdapter(DkAdapter):
    id = "dk_iface"
    label = "NablaEngine (jax_ffi, interface-default gradient mode clamp_kv=0, normal=3)"
    native = "labelled"
    variants = {"default": {}}
    mode = IFACE


class DkJacAdapter(DkAdapter):
    id = "dk_jac"
    label = "NablaEngine, Jacobian-equivalent (71 VJPs, exact gradient mode)"
    native = "ours"
    variants = {"default": {}}
    mode = EXACT
    timed_ops = ("jac",)


ADAPTERS = {"dk": DkAdapter, "dk_iface": DkIfaceAdapter, "dk_jac": DkJacAdapter}

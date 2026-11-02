"""Shared code of the MuJoCo Warp adapters: mjw.py (PyPI mujoco-warp 3.17.0) and mjw_pr.py (etaoxing/mujoco_warp @
357a75d, the head of the unmerged google-deepmind/mujoco_warp#1535, and 357a75d + etaoxing/mujoco_warp#1); other
MuJoCo Warp trees subclass MjwBase / MjwAdjointBase the same way. Not an adapter itself (no ADAPTERS dict, so the
registry skips it);
standard library + numpy at import, warp / mujoco_warp only inside the classes (imported when setup runs, so a
subclass can set import-time switches first).

What the MuJoCo Warp family shares (MjwBase):
- model: problem.load_model (iterations, ls_iterations, tolerance, mjDSBL_SENSOR as the problem sets them), then
  mjw.put_model(mjm, batch_sizes={"pair_friction": nworld, "body_ipos": nworld}) with the per-world DR written once into
  the batched arrays (problem.DR.batched in float32: the values of MuJoCo Warp's own float32 DR, as
  nablaengine.reference.warp_physics._set_dr computes them); Track B has no DR: put_model(mjm);
- options: opt.warn_overflow without ITERATIONS and LS_ITERATIONS (warn="bench", as upstream's own
  benchmarks/unitree_g1 overrides it; "all" = put_model's default, "none" = 0), opt.graph_conditional as put_model sets
  it (True: the solver loop is a conditional graph node while any world is unconverged); MuJoCo Warp raises the
  tolerance to max(tol, 1e-6) internally, the effective value is read back from the device model and recorded;
- capacity: capacity() below, recorded with every unit;
- the states uploaded once as float32 arrays; the forward is the CUDA graph (wp.ScopedCapture, replayed by
  wp.capture_launch) of [copy the inputs into Data; substeps x mjw.step], so every call starts from the loaded states;
- outputs() runs one eager forward from the loaded states substep by substep and reads, after every substep, the
  contacts per world (contact.worldid of the first nacon contacts), the constraint rows (nefc), the solver iterations
  (solver_niter) and the overflow bits (d.overflow, cleared first); diagnostics() reports them per world as the maximum
  over the substeps (as the corpora's MuJoCo C labels), mean and maximum over the worlds;
- provenance: mujoco / mujoco_warp / warp versions and wheel tags, the mujoco_warp package's source hash (the
  definition of nablaengine.provenance.tree_sha256: sha256 over 'sha256  mujoco_warp/<path>' of its .py files) and the
  git commit of its tree when it is a checkout, the device model's effective options, the capacity and its source.

Capacity. MuJoCo Warp pools the contacts of all worlds (naconmax) and caps the constraint rows per
world (njmax). The same pool of naconmax entries also holds the broadphase's candidate pairs (d.ncollision; more than
naconmax sets the BROADPHASE overflow bit in every world and drops pairs), and for this model there are about twice as
many candidates as contacts, so "contacts" in the procedure means max(candidate pairs, contacts). Modes (adapter
option capacity):
  procedure  (default for mjw, mjw_pr1535, mjw_pr1535_fix) from the measurement of mjw_capacity.py, stored in
             benchmarks/data/mjw_capacity.json (adapter option capacity_file): with mu the largest mean number of
             pool entries per world (max of d.ncollision and d.nacon, divided by nworld) over every corpus, iteration
             count and substep, c_max the most contacts of one world and r_max the most constraint rows of one world
             (mjw_capacity.py also verifies the result on every state subset the harness runs: no overflow),
               naconmax = ceil(nworld * mu * (1 + MARGIN)) + c_max   (MARGIN = 0.25)
               nconmax  = ceil(mu * (1 + MARGIN))                     (per world; MuJoCo Warp's sparse njmax_nnz
                                                                       estimate and CCD capacity read it)
               njmax    = the smallest of MuJoCo Warp's own valid sizes (16, 24, 32, 48, 64, 96, 128, 192, ...) >= r_max
  matched    64 contacts and 288 rows per world (naconmax = 64 nworld, njmax 288): NablaEngine's fixed
             capacity, which the matched capacity follows. 288 is not one of MuJoCo Warp's valid sizes (... 192,
             256, 384 ...): whether put_data accepts, rounds or rejects it has not been checked, so check it in an
             environment with mujoco_warp before reporting a matched benchmark
  explicit   the options nconmax (per world) / naconmax (total) and njmax
Overflow (capacity bits of d.overflow, i.e. every bit except ITERATIONS and LS_ITERATIONS) is counted per world in
diagnostics()["overflow"]; it is never hidden.
"""

import importlib.metadata
import importlib.util
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np

from benchmarks.bench import prov
from benchmarks.bench import problem as P
from benchmarks.bench.adapters import base

REPO = Path(__file__).resolve().parents[3]
CAPACITY_FILE = "benchmarks/data/mjw_capacity.json"
CAPACITY_MODES = ("procedure", "matched", "explicit")
# the matched capacity follows NablaEngine's (64 contacts, 288 rows per world); 288 is not among MuJoCo Warp's valid
# sizes (VALID_SIZES: 192, 256, 384), and whether put_data accepts, rounds or rejects it is unchecked: check it in an
# environment with mujoco_warp before a matched benchmark is reported
MATCHED = dict(nconmax=64, njmax=288)
MARGIN = 0.25
# MuJoCo Warp's valid capacity sizes (io._default_nconmax / _default_njmax): 16, 24, 32, 48, ... 8192
VALID_SIZES = tuple(int(x) for x in (2 + (np.arange(19) % 2)) * (2 ** (np.arange(19) // 2 + 3)))
WARN_MODES = ("bench", "all", "none")
ITER_BITS = ("ITERATIONS", "LS_ITERATIONS")  # solver limits, not capacity: never an overflow
NO_BACKWARD = ("no backward in any MuJoCo Warp release: the mujoco-warp 3.17.0 README says 'Differentiability via "
               "Warp is not yet available. See #500 for progress' (https://github.com/google-deepmind/mujoco_warp/"
               "issues/500, open), the documentation FAQ answers 'Is MJWarp differentiable?' with 'No', and its "
               "modules are built with enable_backward=False (under wp.Tape, 3.17.0 runs without an error and its step "
               "contributes exactly zero to every derivative: benchmarks/bench/adapters/mjw.py tape_evidence)")


def valid_size(n):
    """The smallest of MuJoCo Warp's valid capacity sizes >= n (n itself beyond the list)."""
    n = int(math.ceil(n))
    return next((s for s in VALID_SIZES if s >= n), n)


def _abs(p):
    p = Path(p).expanduser()
    return p if p.is_absolute() else REPO / p


def load_capacity_file(path=None):
    p = _abs(path or CAPACITY_FILE)
    if not p.exists():
        raise FileNotFoundError(f"MuJoCo Warp capacity file {prov.rel(p)} not found: measure it with "
                                f"python -m benchmarks.bench.adapters.mjw_capacity (in the mjw-native environment), or "
                                f"run with --opt <engine>.capacity=\"matched\"")
    return p, json.loads(p.read_text())


def capacity(prob, nworld, corpus=None, mode="procedure", nconmax=None, naconmax=None, njmax=None,
             capacity_file=None):
    """{mode, nconmax (per world), naconmax (all worlds), njmax (per world), ...} for a unit (module docstring)."""
    if mode not in CAPACITY_MODES:
        raise ValueError(f"capacity must be one of {CAPACITY_MODES}, got {mode!r}")
    if mode == "matched":
        return dict(mode=mode, nconmax=MATCHED["nconmax"], naconmax=MATCHED["nconmax"] * nworld,
                    njmax=MATCHED["njmax"], rule="64 contacts and 288 rows per world (NablaEngine's)")
    if mode == "explicit":
        if njmax is None or (nconmax is None and naconmax is None):
            raise ValueError("capacity 'explicit' needs njmax and nconmax (per world) or naconmax (all worlds)")
        nc = int(nconmax) if nconmax is not None else int(math.ceil(naconmax / nworld))
        return dict(mode=mode, nconmax=nc, naconmax=int(naconmax) if naconmax is not None else nc * nworld,
                    njmax=int(njmax), rule="given")
    if nconmax is not None or naconmax is not None or njmax is not None:
        raise ValueError("nconmax / naconmax / njmax are for capacity 'explicit'")
    path, doc = load_capacity_file(capacity_file)
    entry = (doc.get("problems") or {}).get(prob.id)
    if entry is None:
        raise KeyError(f"{prov.rel(path)} has no entry for problem {prob.id!r} (measured: "
                       f"{sorted(doc.get('problems') or {})}); rerun benchmarks/bench/adapters/mjw_capacity.py")
    mu, cmax, rmax = float(entry["pool_mean_max"]), int(entry["contacts_world_max"]), int(entry["rows_max"])
    margin = float(entry.get("margin", MARGIN))
    out = dict(mode=mode, nconmax=int(math.ceil(mu * (1 + margin))),
               naconmax=int(math.ceil(nworld * mu * (1 + margin))) + cmax, njmax=valid_size(rmax),
               rule="naconmax = ceil(nworld mu (1 + margin)) + c_max, nconmax = ceil(mu (1 + margin)), njmax = the "
                    "smallest valid size >= r_max",
               mu=mu, c_max=cmax, r_max=rmax, margin=margin, file=prov.rel(path), file_sha256=prov.sha256(path),
               verified=entry.get("verified"))
    if corpus is not None:
        measured = entry.get("corpora") or {}
        out["corpus_measured"] = any(c in measured for c in corpus.split("+"))
    return out


def tree_sha256(pkg_dir):
    """sha256 over 'sha256  <package>/<path>' lines of the package's .py files in byte order (the definition of
    nablaengine.provenance.tree_sha256)."""
    import hashlib
    pkg_dir = Path(pkg_dir)
    files = sorted(f"{pkg_dir.name}/{p.relative_to(pkg_dir).as_posix()}" for p in pkg_dir.rglob("*.py") if p.is_file())
    lines = "".join(f"{hashlib.sha256((pkg_dir.parent / f).read_bytes()).hexdigest()}  {f}\n" for f in files)
    return hashlib.sha256(lines.encode()).hexdigest()


def _git(tree, *args):
    try:
        r = subprocess.run(["git", "-C", str(tree), *args], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def mujoco_warp_source():
    """Where `import mujoco_warp` resolves here, without importing it: {path (relative), tree_sha256, version (the
    distribution's, None for a tree on PYTHONPATH), installed (a distribution owns these files), git: {commit, dirty}
    of the tree when it is a checkout}."""
    spec = importlib.util.find_spec("mujoco_warp")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("mujoco_warp is not importable here")
    pkg = Path(list(spec.submodule_search_locations)[0]).resolve()
    out = dict(path=prov.rel(pkg), tree_sha256=tree_sha256(pkg), version=None, installed=False, git=None)
    try:
        dist = importlib.metadata.distribution("mujoco-warp")
        loc = Path(dist.locate_file("mujoco_warp")).resolve()
        out["version"] = dist.version
        out["installed"] = loc == pkg
    except importlib.metadata.PackageNotFoundError:
        pass
    top = _git(pkg.parent, "rev-parse", "--show-toplevel")
    commit = _git(pkg.parent, "rev-parse", "HEAD") if top and Path(top).resolve() == pkg.parent else None
    if commit is not None:  # the tree is a checkout of its own (not a directory inside another repository)
        st = _git(pkg.parent, "status", "--porcelain", "--untracked-files=no")
        out["git"] = dict(commit=commit, dirty=None if st is None else bool(st))
    return out


def quiet_warp(wp):
    """No Warp start-up banner or per-module load lines (config.log_level in Warp >= 1.12, config.quiet before)."""
    wp.config.quiet = True
    if hasattr(wp, "LOG_WARNING") and hasattr(wp.config, "log_level"):
        wp.config.log_level = wp.LOG_WARNING


def _versions():
    import mujoco
    out = dict(mujoco=mujoco.__version__)
    for name in ("mujoco-warp", "warp-lang", "numpy"):
        out[name] = prov.package(name)
    return out


class MjwBase(base.Adapter):
    """The MuJoCo Warp family's common setup, forward graph, outputs, diagnostics and provenance (module docstring).

    Subclass hooks: OPTIONS (setup options and defaults; unknown ones raise TypeError), enable_grad (call
    mjw.enable_grad() right after the import, before any kernel is built), _pre_import(opts) (e.g. import-time
    switches), _check_source(src) (raise when mujoco_warp is not the expected tree), _build(opts) (the model / data /
    extra buffers; the default is the native construction), _ops_extra(), _prepare_extra(op)."""

    device = "gpu"
    tracks = ("A",)
    enable_grad = False
    OPTIONS = dict(capacity="procedure", capacity_file=None, nconmax=None, naconmax=None, njmax=None, warn="bench")

    # ------------------------------------------------------------------ setup
    def _options(self, opts):
        unknown = sorted(set(opts) - set(self.OPTIONS))
        if unknown:
            raise TypeError(f"{self.id}: unknown option(s) {unknown} (known: {sorted(self.OPTIONS)})")
        o = dict(self.OPTIONS, **opts)
        if o["warn"] not in WARN_MODES:
            raise ValueError(f"{self.id}: warn must be one of {WARN_MODES}, got {o['warn']!r}")
        return o

    def _pre_import(self, o):
        pass

    def _check_source(self, src):
        pass

    def _import(self):
        import warp as wp
        quiet_warp(wp)
        import mujoco_warp as mjw
        self.wp, self.mjw = wp, mjw
        if not wp.is_cuda_available():
            raise RuntimeError("Warp sees no CUDA device (a CPU-only fallback, e.g. the CUDA 13 warp-lang wheel on a "
                               "CUDA 12 driver or on Volta): install the +cu12 wheel (benchmarks/envs/setup.sh)")
        self.dev = wp.get_device("cuda:0")
        if self.enable_grad:
            mjw.enable_grad()

    def setup(self, problem, states, nworld, iterations, **opts):
        o = self._options(opts)
        self.o, self.problem, self.states, self.nworld = o, problem, states, int(nworld)
        self.iterations, self.substeps = iterations, int(problem.substeps)
        self.graphs, self.stats = {}, None
        t = {}
        t0 = time.perf_counter()
        self._pre_import(o)
        self.source = mujoco_warp_source()
        self._check_source(self.source)
        self._import()
        wp = self.wp
        self.mjm = P.load_model(problem, iterations)
        states.check(self.mjm.nq, self.mjm.nv, self.mjm.nu, problem.dr)
        base.check_layout(self.mjm.nq, self.mjm.nv)
        self.cap = self._capacity(o)
        with wp.ScopedDevice(self.dev):
            self._build(o)
            wp.synchronize_device(self.dev)
            t["build_s"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            self._upload()
            wp.synchronize_device(self.dev)
            t["load_s"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            self._fwd_body()  # builds / loads the forward kernels
            wp.synchronize_device(self.dev)
            t["compile_s"] = time.perf_counter() - t0
        return t

    def _capacity(self, o):
        return capacity(self.problem, self.nworld, self.states.corpus, o["capacity"], o["nconmax"], o["naconmax"],
                        o["njmax"], o["capacity_file"])

    def _warn_mask(self, m, mode):
        """opt.warn_overflow of a device model for warn mode bench / all / none."""
        mjw = self.mjw
        ot = getattr(mjw, "OverflowType", None)
        if ot is None:
            from mujoco_warp._src.types import OverflowType as ot
        self.overflow_type = ot
        if mode == "none":
            m.opt.warn_overflow = 0
        elif mode == "bench":
            m.opt.warn_overflow = int(m.opt.warn_overflow) & ~int(ot.ITERATIONS | ot.LS_ITERATIONS)

    def _model(self, mjm, nworld):
        """put_model with the per-world DR of the loaded states (module docstring)."""
        mjw = self.mjw
        if self.problem.dr:
            m = mjw.put_model(mjm, batch_sizes={"pair_friction": nworld, "body_ipos": nworld})
            pf, bi = P.DR(mjm, self.problem).batched(self.states.friction[:nworld], self.states.com[:nworld],
                                                     np.float32)
            m.pair_friction.assign(pf)
            m.body_ipos.assign(bi)
        else:
            m = mjw.put_model(mjm)
        return m

    def _init_data(self):
        """The MjData put_data copies into every world (overwritten by the loaded states before any step): mj_forward
        at qpos0 (consistent derived quantities), unless qpos0's contacts / rows exceed the capacity (put_data would
        refuse), then an MjData without mj_forward (no contacts); recorded as self.init_data."""
        import mujoco
        mjm, c = self.mjm, self.cap
        mjd = mujoco.MjData(mjm)
        mujoco.mj_forward(mjm, mjd)
        if mjd.ncon * self.nworld <= c["naconmax"] and mjd.ncon <= c["nconmax"] and mjd.nefc <= c["njmax"]:
            self.init_data = "mj_forward at qpos0"
            return mjd
        self.init_data = (f"MjData at qpos0 without mj_forward (mj_forward gives {mjd.ncon} contacts / {mjd.nefc} "
                          f"rows, more than the capacity)")
        return mujoco.MjData(mjm)

    def _build(self, o):
        mjm = self.mjm
        self.mjd = self._init_data()
        self.m = self._model(mjm, self.nworld)
        self._warn_mask(self.m, o["warn"])
        self.d = self._data()

    def _data(self):
        c = self.cap
        return self.mjw.put_data(self.mjm, self.mjd, nworld=self.nworld, nconmax=c["nconmax"], njmax=c["njmax"],
                                 naconmax=c["naconmax"])

    def _upload(self):
        wp, s = self.wp, self.states
        f32 = lambda x: wp.array(np.ascontiguousarray(x, np.float32), dtype=float)  # noqa: E731
        self.iq, self.iv, self.iw, self.iu = (f32(x) for x in (s.qpos, s.qvel, s.qacc_warmstart, s.ctrl))

    def _load(self, d):
        wp = self.wp
        wp.copy(d.qpos, self.iq)
        wp.copy(d.qvel, self.iv)
        wp.copy(d.qacc_warmstart, self.iw)
        wp.copy(d.ctrl, self.iu)

    def _fwd_body(self):
        """One control step from the loaded states into self.d (what the fwd graph holds)."""
        self._load(self.d)
        for _ in range(self.substeps):
            self.mjw.step(self.m, self.d)

    # ------------------------------------------------------------------ graphs and ops
    def _capture(self, name, body, first=None):
        """Capture body() as the CUDA graph `name` after one eager run (so kernels are built and lazily allocated
        buffers exist outside the capture); returns {capture_s} and, with first, {first: seconds of the eager run}
        (e.g. compile_s for a backward whose kernels that run builds)."""
        wp = self.wp
        if name in self.graphs:
            return {}
        with wp.ScopedDevice(self.dev):
            t0 = time.perf_counter()
            body()
            wp.synchronize_device(self.dev)
            t_first = time.perf_counter() - t0
            t0 = time.perf_counter()
            with wp.ScopedCapture() as cap:
                body()
            self.graphs[name] = cap.graph
            wp.synchronize_device(self.dev)
        out = dict(capture_s=time.perf_counter() - t0)
        if first:
            out[first] = t_first
        return out

    def _launch(self, name):
        self.wp.capture_launch(self.graphs[name])

    def _fwd(self):
        self._launch("fwd")

    def ops(self):
        return dict(fwd=self._fwd, **self._ops_extra())

    def _ops_extra(self):
        return {}

    def prepare(self, op):
        if op == "fwd":
            return self._capture("fwd", self._fwd_body)
        return self._prepare_extra(op)

    def _prepare_extra(self, op):
        return {}

    def sync(self):
        self.wp.synchronize_device(self.dev)

    # ------------------------------------------------------------------ outputs and diagnostics
    def _stats_forward(self, m, d, load=None):
        """One eager forward from the loaded states on (m, d), substep by substep, collecting the per-substep
        statistics into self.stats; d holds the outputs afterwards."""
        wp = self.wp
        n, ns = self.nworld, self.substeps
        with wp.ScopedDevice(self.dev):
            d.overflow.zero_()
            (load or self._load)(d)
            ncon = np.zeros((ns, n), np.int64)
            nefc = np.zeros((ns, n), np.int64)
            niter = np.zeros((ns, n), np.int64)
            nacon, ncoll = [], []
            for k in range(ns):
                self._step_eager(m, d)
                wp.synchronize_device(self.dev)
                na = int(d.nacon.numpy()[0])
                nacon.append(na)
                ncoll.append(int(d.ncollision.numpy()[0]) if hasattr(d, "ncollision") else None)
                wid = d.contact.worldid.numpy()[:min(na, d.contact.worldid.shape[0])]
                ncon[k] = np.bincount(wid[(wid >= 0) & (wid < n)], minlength=n)[:n]
                nefc[k] = d.nefc.numpy().reshape(-1)[:n]
                niter[k] = d.solver_niter.numpy().reshape(-1)[:n]
            ov = d.overflow.numpy().reshape(-1)[:n].astype(np.int64)
        self.stats = dict(ncon=ncon, nefc=nefc, niter=niter, nacon=nacon, ncollision=ncoll, overflow=ov,
                          naconmax=int(d.contact.worldid.shape[0]), njmax=int(d.efc.type.shape[1]))

    def _step_eager(self, m, d):
        self.mjw.step(m, d)

    def _outputs_forward(self):
        self._stats_forward(self.m, self.d)
        return self.d

    def outputs(self):
        d = self._outputs_forward()
        return dict(qpos=d.qpos.numpy().astype(np.float64), qvel=d.qvel.numpy().astype(np.float64),
                    qacc_warmstart=d.qacc_warmstart.numpy().astype(np.float64))

    def _overflow_summary(self, ov):
        ot = self.overflow_type
        names = [f.name for f in ot if f.value and not (f.value & (f.value - 1))]
        cap_mask = 0
        for nm in names:
            if nm not in ITER_BITS:
                cap_mask |= int(ot[nm])
        bits = int(np.bitwise_or.reduce(ov)) if ov.size else 0
        return dict(capacity_worlds=int(((ov & cap_mask) != 0).sum()),
                    bits=[nm for nm in names if bits & int(ot[nm])],
                    iterations_limit_worlds=int(((ov & int(ot.ITERATIONS)) != 0).sum()),
                    ls_iterations_limit_worlds=int(((ov & int(ot.LS_ITERATIONS)) != 0).sum()))

    def diagnostics(self):
        if self.stats is None:
            self._outputs_forward()
        s = self.stats
        ncon_w, nefc_w, niter_w = s["ncon"].max(0), s["nefc"].max(0), s["niter"].max(0)
        ov = self._overflow_summary(s["overflow"])
        out = dict(ncon_mean=float(ncon_w.mean()), ncon_max=int(ncon_w.max()), nefc_mean=float(nefc_w.mean()),
                   nefc_max=int(nefc_w.max()), niter_mean=float(niter_w.mean()), niter_max=int(niter_w.max()),
                   niter_hist=np.bincount(niter_w).tolist(), overflow=ov["capacity_worlds"], unsupported=None)
        out["extra"] = dict(
            per_world="maximum over the substeps (contacts, constraint rows, solver iterations)",
            contacts_mean_per_substep=[float(x) for x in s["ncon"].mean(1)],
            rows_mean_per_substep=[float(x) for x in s["nefc"].mean(1)],
            niter_mean_per_substep=[float(x) for x in s["niter"].mean(1)],
            nacon_per_substep=s["nacon"], naconmax=s["naconmax"], njmax=s["njmax"],
            ncollision_per_substep=s["ncollision"],
            pool_peak_share=max(max(s["nacon"]), max(x or 0 for x in s["ncollision"])) / max(1, s["naconmax"]),
            overflow_bits=ov["bits"],
            iterations_limit_worlds=ov["iterations_limit_worlds"],
            ls_iterations_limit_worlds=ov["ls_iterations_limit_worlds"],
            solver_exit="per batch: the Newton loop runs while any world is unconverged (graph_conditional); "
                        "solver_niter counts each world's own iterations")
        out["extra"].update(self._diag_extra())
        return out

    def _diag_extra(self):
        return {}

    def memory(self):
        wp = self.wp
        out = {}
        try:
            out["alloc_peak_bytes"] = int(wp.get_mempool_used_mem_high(self.dev))
            out["alloc_current_bytes"] = int(wp.get_mempool_used_mem_current(self.dev))
            out["allocator"] = "Warp mempool high-water mark"
        except Exception:  # noqa: BLE001  (mempool disabled or unsupported)
            out["allocator"] = None
        return out

    # ------------------------------------------------------------------ provenance
    def _device_options(self, m):
        def get(name):
            v = getattr(m.opt, name, None)
            if v is None:
                return None
            if hasattr(v, "numpy"):
                a = v.numpy().reshape(-1)
                return [float(x) for x in a[:4]] if a.size > 1 else float(a[0])
            if isinstance(v, (bool, np.bool_)):
                return bool(v)
            try:
                return int(v)
            except (TypeError, ValueError):
                return str(v)
        out = {k: get(k) for k in ("iterations", "ls_iterations", "tolerance", "ls_tolerance", "solver", "cone",
                                   "integrator", "graph_conditional", "warn_overflow")}
        out["is_sparse"] = bool(getattr(m, "is_sparse", getattr(m.opt, "is_sparse", None)))
        ot = getattr(self, "overflow_type", None)
        if ot is not None and out.get("warn_overflow") is not None:
            out["warn_overflow_masked"] = [f.name for f in ot if f.value and not (f.value & (f.value - 1))
                                           and not (out["warn_overflow"] & int(f.value))]
        return out

    def provenance(self):
        out = dict(engine=self.id, label=self.label, versions=_versions(), mujoco_warp=self.source,
                   device=self.dev.name, arch=f"sm_{self.dev.arch}" if hasattr(self.dev, "arch") else None,
                   problem_options=P.options(self.mjm), device_options=self._device_options(self.m),
                   capacity=self.cap, init_data=getattr(self, "init_data", None), warn=self.o["warn"],
                   substeps=self.substeps, iterations=self.iterations,
                   dr=("put_model batch_sizes {pair_friction, body_ipos}: per-world DR written once (float32)"
                       if self.problem.dr else None),
                   timed="CUDA graph (wp.ScopedCapture) replayed with wp.capture_launch; inputs copied in the graph")
        out.update(self._provenance_extra())
        return out

    def _provenance_extra(self):
        return {}

    def close(self):
        self.graphs = {}

    @classmethod
    def check_env(cls):
        src = mujoco_warp_source()
        cls()._check_source(src)  # the expected tree / release, as setup checks it
        import warp as wp
        quiet_warp(wp)
        if not wp.is_cuda_available():
            raise RuntimeError("Warp sees no CUDA device (CPU-only fallback): install the +cu12 warp-lang wheel on "
                               "CUDA 12 drivers and on Volta (benchmarks/envs/setup.sh)")
        return dict(mujoco_warp=src, warp=wp.config.version, warp_lang=prov.package("warp-lang"),
                    device=wp.get_device("cuda:0").name)


class MjwAdjointBase(MjwBase):
    """The MuJoCo Warp + PR #1535 adjoint (mjw_pr.py): enable_grad, cotangent and gradient buffers, the
    gradient in the harness convention. Subclasses define _bwd_body() (the backward from the loaded states and the
    cotangents in self.cq / self.cv into self.gq / self.gv / self.gu) and _fwd_train_body() if it differs from fwd."""

    enable_grad = True

    def _upload(self):
        super()._upload()
        wp, s = self.wp, self.states
        if s.cot_qpos is not None:
            cq, cv = s.cotangent(0, np.float32)
        else:  # no stored cotangents: ones (timing only; gradient() gets its cotangents as arguments)
            cq, cv = np.ones_like(s.qpos, np.float32), np.ones_like(s.qvel, np.float32)
        self.cq = wp.array(cq, dtype=float)
        self.cv = wp.array(cv, dtype=float)
        self.gq, self.gv, self.gu = wp.zeros_like(self.iq), wp.zeros_like(self.iv), wp.zeros_like(self.iu)

    def gradient(self, cot_qpos, cot_qvel):
        """The VJP of one control step for the given output cotangents, through the timed bwd graph: qpos in the
        35-dimensional tangent form (base.qpos_tangent of MuJoCo Warp's qpos-form gradient at the input qpos), qvel,
        ctrl summed over the substeps (ctrl is held)."""
        wp = self.wp
        self.prepare("bwd")
        self.cq.assign(np.ascontiguousarray(cot_qpos, np.float32))
        self.cv.assign(np.ascontiguousarray(cot_qvel, np.float32))
        self._gradient_launch()
        wp.synchronize_device(self.dev)
        g = dict(qpos=base.qpos_tangent(self.gq.numpy().astype(np.float64), self.states.qpos.astype(np.float64)),
                 qvel=self.gv.numpy().astype(np.float64), ctrl=self.gu.numpy().astype(np.float64))
        if self.states.cot_qpos is not None:  # restore the timed cotangent 0
            cq, cv = self.states.cotangent(0, np.float32)
            self.cq.assign(cq)
            self.cv.assign(cv)
        return g

    def _gradient_launch(self):
        self._launch("bwd")

    def not_supported(self):
        return dict(jac="MuJoCo Warp + PR #1535 has no Jacobian API, only the VJP (tape.backward); the full Jacobian "
                        "would be 71 VJPs built by the benchmark, not a native call")

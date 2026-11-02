"""The native MuJoCo Warp adapter 'mjw': PyPI mujoco-warp 3.17.0 with mujoco 3.17.0 and warp-lang 1.18.0 (the default
wheel where the driver supports CUDA 13, NVIDIA's +cu12 wheel otherwise; the installed tag is recorded), run through
its public API only (mjw.put_model / put_data / step, wp.ScopedCapture). Forward only: MuJoCo Warp has no backward in
any release (mjw_common.NO_BACKWARD), so bwd, vjp, fwd_train and jac are not_supported with that citation, and
tape_evidence() shows what 3.17.0 does under wp.Tape. Environment 'mjw-native' (benchmarks/envs/envs.toml).

Setup and timing are mjw_common.MjwBase's: per-world DR through put_model batch_sizes, opt.warn_overflow without
ITERATIONS / LS_ITERATIONS as upstream's benchmarks/unitree_g1, graph_conditional on, sensors as the problem sets them
(off on Track A), and op fwd = the CUDA graph of [copy the corpus inputs into Data; substeps x mjw.step].

Variants (cell ids <engine>~<variant>; any option also with run --opt mjw.<key>=<value>):
  default    capacity "procedure" (mjw_common: naconmax from the measured mean contacts + margin, njmax the smallest
             valid size >= the most rows over all corpora; benchmarks/data/mjw_capacity.json)
  matched    capacity 64 contacts / 288 rows per world (NablaEngine's)
  warn_all   put_model's default opt.warn_overflow (every warning on: the iteration-limit warnings print)
Options: capacity (procedure | matched | explicit), capacity_file, nconmax, naconmax, njmax (explicit), warn (bench |
all | none), tape_evidence (default true: diagnostics() also records tape_evidence() on the first 8 worlds).

  python -m benchmarks.bench run --engines mjw --quick --out runs/bench/test_mjw/
  # the wp.Tape evidence alone (in the mjw-native environment, repository root on PYTHONPATH):
  python -m benchmarks.bench.adapters.mjw --tape-evidence --corpus mf_s0 --out runs/bench/test_mjw/mjw_tape.json
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from benchmarks.bench.adapters import base, mjw_common

TAPE_WORLDS = 8


class MjwAdapter(mjw_common.MjwBase):
    id = "mjw"
    label = "MuJoCo Warp 3.17.0 (PyPI)"
    native = "native"
    tracks = ("A", "B")
    variants = {"default": {}, "matched": {"capacity": "matched"}, "warn_all": {"warn": "all"}}
    OPTIONS = dict(mjw_common.MjwBase.OPTIONS, tape_evidence=True)
    VERSION = "3.17.0"

    def _check_source(self, src):
        if not src["installed"] or src["version"] != self.VERSION:
            raise RuntimeError(f"mjw: `import mujoco_warp` resolves to {src['path']} (distribution version "
                               f"{src['version']}, installed {src['installed']}), not the PyPI mujoco-warp "
                               f"{self.VERSION} wheel: use the mjw-native environment (benchmarks/envs/setup.sh)")

    def not_supported(self):
        msg = mjw_common.NO_BACKWARD
        return dict(fwd_train=f"no forward that keeps residuals for a backward: {msg}", bwd=msg, vjp=msg, jac=msg)

    def gradient(self, cot_qpos, cot_qvel):
        raise base.NotSupported(mjw_common.NO_BACKWARD)

    def jacobian(self):
        raise base.NotSupported(mjw_common.NO_BACKWARD)

    def _diag_extra(self):
        if not self.o.get("tape_evidence"):
            return {}
        try:
            return dict(tape_evidence=tape_evidence(self.mjw, self.wp, self.mjm, self.problem, self.states,
                                                    min(TAPE_WORLDS, self.nworld), self.substeps))
        except Exception as e:  # noqa: BLE001  (evidence only; never hides the timed results)
            return dict(tape_evidence=dict(error=f"{type(e).__name__}: {e}"[:500]))

    def semantics(self):
        return dict(gradient="none: " + mjw_common.NO_BACKWARD,
                    solver_exit="per batch (graph_conditional): the Newton loop runs while any world is unconverged",
                    tolerance="MuJoCo Warp raises opt.tolerance to max(tolerance, 1e-6) internally (recorded in "
                              "device_options)")


def tape_evidence(mjw, wp, mjm, prob, states, nworld, substeps):
    """What MuJoCo Warp 3.17.0 returns under wp.Tape for one control step (separate model and data of nworld worlds of
    the loaded states, so the timed buffers are untouched): the inputs are copied into Data on the tape (wp.copy is
    differentiable), the substeps run, the stored cotangent 0 (ones without one) is seeded on the outputs and
    tape.backward() runs. Every module of 3.17.0 is built with enable_backward=False, so no adjoint kernel exists:
    the result is d/dctrl = 0 exactly and d/dqpos, d/dqvel = the seeded cotangent itself (the copies' identity; the
    step adds nothing), with no error raised; Warp's Tape.backward warns for every recorded launch whose kernel is
    built with enable_backward=False (counted here with Warp's own condition, the log silenced). Capacity 64 contacts /
    288 rows per world (any capacity that holds the worlds gives the same). Returns the tape's launches, the
    gradients' largest entries and these checks."""
    import mujoco
    n = int(nworld)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    if prob.dr:
        from benchmarks.bench import problem as P
        m = mjw.put_model(mjm, batch_sizes={"pair_friction": n, "body_ipos": n})
        pf, bi = P.DR(mjm, prob).batched(states.friction[:n], states.com[:n], np.float32)
        m.pair_friction.assign(pf)
        m.body_ipos.assign(bi)
    else:
        m = mjw.put_model(mjm)
    m.opt.warn_overflow = 0
    d = mjw.put_data(mjm, mjd, nworld=n, nconmax=mjw_common.MATCHED["nconmax"], njmax=mjw_common.MATCHED["njmax"])
    f32 = lambda x: np.ascontiguousarray(np.asarray(x)[:n], np.float32)  # noqa: E731
    ins = dict(qpos=wp.array(f32(states.qpos), dtype=float, requires_grad=True),
               qvel=wp.array(f32(states.qvel), dtype=float, requires_grad=True),
               qacc_warmstart=wp.array(f32(states.qacc_warmstart), dtype=float, requires_grad=False),
               ctrl=wp.array(f32(states.ctrl), dtype=float, requires_grad=True))
    for k in ("qpos", "qvel", "ctrl"):
        getattr(d, k).requires_grad = True
    if states.cot_qpos is not None:
        cq, cv = (f32(x) for x in states.cotangent(0))
    else:
        cq, cv = np.ones((n, mjm.nq), np.float32), np.ones((n, mjm.nv), np.float32)
    out = dict(mujoco_warp=mjw_common.mujoco_warp_source(), nworld=n, substeps=substeps,
               cotangent="stored cotangent 0" if states.cot_qpos is not None else "ones")
    t0 = time.perf_counter()
    tape = wp.Tape()
    with tape:
        for k, a in ins.items():
            wp.copy(getattr(d, k), a)
        for _ in range(substeps):
            mjw.step(m, d)
    d.qpos.grad.assign(cq)
    d.qvel.grad.assign(cv)
    launches = [x for x in tape.launches if not callable(x)]
    no_bwd = []
    for x in launches:  # the condition under which Warp's Tape.backward warns (warp/_src/tape.py)
        eb = x[0].options.get("enable_backward")
        if eb is None:
            eb = x[0].module.options.get("enable_backward")
        if eb is False:
            no_bwd.append(x[0].key)
    level = getattr(wp.config, "log_level", None)
    if level is not None and hasattr(wp, "LOG_ERROR"):  # Warp logs one warning per such launch: counted above instead
        wp.config.log_level = wp.LOG_ERROR
    try:
        tape.backward()
        wp.synchronize()
    finally:
        if level is not None:
            wp.config.log_level = level
    g = {k: ins[k].grad.numpy().astype(np.float64) for k in ("qpos", "qvel", "ctrl")}
    ov = d.overflow.numpy()
    out.update(
        error=None, overflow_bits_or=int(np.bitwise_or.reduce(ov)) if ov.size else 0,
        seconds=round(time.perf_counter() - t0, 3), tape_launches=len(tape.launches),
        grad_abs_max={k: float(np.abs(v).max()) for k, v in g.items()},
        grad_ctrl_exact_zero=bool(np.all(g["ctrl"] == 0.0)),
        grad_qpos_equals_seed=bool(np.array_equal(g["qpos"], cq.astype(np.float64))),
        grad_qvel_equals_seed=bool(np.array_equal(g["qvel"], cv.astype(np.float64))),
        outputs_finite=bool(np.isfinite(d.qvel.numpy()).all()),
        kernel_launches=len(launches), launches_without_backward=len(no_bwd),
        kernels_without_backward=len(set(no_bwd)),
        kernels_without_backward_sample=sorted({re.sub(r"__locals__.*", "", k) for k in no_bwd})[:8],
        note="no error raised; the step's own contribution to every derivative is exactly zero: d/dctrl = 0, "
             "d/dqpos and d/dqvel equal the seeded output cotangent (the input copies' identity)")
    return out


def _tape_cli(a):
    """--tape-evidence: tape_evidence() on the first --nworld states of a corpus (mjlab problem)."""
    import mujoco_warp as mjw
    import warp as wp

    from benchmarks.bench import problem as P
    from benchmarks.bench import prov
    mjw_common.quiet_warp(wp)
    if not wp.is_cuda_available():
        raise RuntimeError("Warp sees no CUDA device")
    prob = P.get(a.problem)
    states = P.select(a.corpus, a.nworld, 0)
    mjm = P.load_model(prob, a.iterations if prob.explicit_iterations else None)
    with wp.ScopedDevice("cuda:0"):
        res = tape_evidence(mjw, wp, mjm, prob, states, a.nworld, prob.substeps)
    res.update(problem=prob.id, corpus=a.corpus, iterations=int(mjm.opt.iterations), device=wp.get_device().name,
               software=prov.software(), command=" ".join(["python -m benchmarks.bench.adapters.mjw"] + sys.argv[1:]))
    text = json.dumps(res, indent=1)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")
    ok = res["grad_ctrl_exact_zero"] and res["grad_qpos_equals_seed"] and res["grad_qvel_equals_seed"]
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="MuJoCo Warp 3.17.0 under wp.Tape: the evidence that it has no backward")
    ap.add_argument("--tape-evidence", action="store_true", required=True)
    ap.add_argument("--problem", default="mjlab")
    ap.add_argument("--corpus", default="mf_s0")
    ap.add_argument("--nworld", type=int, default=TAPE_WORLDS)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--out", default=None, help="JSON file (e.g. runs/bench/test_mjw/mjw_tape.json)")
    return _tape_cli(ap.parse_args(argv))


ADAPTERS = {"mjw": MjwAdapter}

if __name__ == "__main__":
    sys.exit(main())

"""Native MuJoCo Warp forward labels for the truth's model check: PyPI mujoco-warp 3.17.0 (mujoco 3.17.0, warp-lang
1.18.0), its public API only, no patches, no gradients. Runs in the mjw-native environment (benchmarks/envs/envs.toml,
engine mjw); the truth tools (truth/cli.py) start it as a subprocess and cache its output. Imports numpy, mujoco,
warp and mujoco_warp; nothing from nablaengine or any other MuJoCo Warp tree.

For every state of a corpus one control step (problem.substeps x mjw.step, ctrl held, qfrc_applied 0) from the
corpus inputs (float32), with the per-world DR as the mjw adapter sets it (put_model(batch_sizes={"pair_friction",
"body_ipos"}) filled with problem.DR.batched in float32), at --iterations (default 50, the reference iteration count;
the tolerance is MuJoCo Warp's own max(tol, 1e-6), recorded), opt.warn_overflow without ITERATIONS and LS_ITERATIONS
(as upstream's own benchmarks/unitree_g1 sets it), capacity 64 contacts and 192 rows per world (naconmax = 64 nworld;
the capacity at which the committed labels were computed, NablaEngine release 0.1.0; any capacity overflow is an
error), opt.deterministic as put_model sets it (NONE). MuJoCo Warp's atomics make a rerun
differ in the last bits, so the model check of a few borderline states per corpus (0 to 2 of 4096 seen) can flip
between runs; DeterminismType.ALL did not make 3.17.0's outputs bitwise reproducible on the V100 either. The truth
tools cache the labels, and the truth file records the run it used. After every substep (its contacts, rows and
kinematics belong to the substep's input state) it records per world, in truth/fdlib.py's layout
(sorted by contact code):
  con_code (S, nsub, 64) int64  pair * 4 + plane-capsule end (0: the contact lies on the +axis side of the capsule
                                centre), -1 = no contact
  con_t (S, nsub, 64, 6) float32  the tangent rows t1, t2 of the contact frame
  con_dim, con_zone (S, nsub, 64) int32  condim; bit e = pyramid edge e (or the single row of condim 1) active,
                                active = efc.force > 0 (con_zone_state: efc.state == QUADRATIC instead)
  lim_rows, lim_act (S, nsub) int64  sum of 1 << joint id over the joint-limit rows / the active ones
  ncon, nefc, niter (S, nsub) int32; out_qpos, out_qvel, out_warm (S, .) float64 (of float32)

  python -m benchmarks.bench.truth.mjw_labels --corpus mf_s0 --out runs/bench/truth/mjw_mf_s0.npz
"""

import argparse
import importlib.metadata
import json
import sys
import time

import numpy as np

from benchmarks.bench import problem as P
from benchmarks.bench.truth import common

ITERATIONS = 50
NCONMAX, NJMAX = 64, 192  # the capacity at which the committed labels were computed (release 0.1.0, 192 rows)
LIMIT_JOINT = 3  # mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT (MuJoCo Warp's ConstraintType.LIMIT_JOINT)
QUADRATIC = 1  # ConstraintState.QUADRATIC
PLANE, CAPSULE = 0, 3  # mujoco.mjtGeom


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _substep(mjm, d, n, pair_of, plane, capsule):
    """The per-world contact signature of the substep just stepped (module docstring), numpy."""
    na = min(int(d.nacon.numpy()[0]), d.contact.worldid.shape[0])
    geom = d.contact.geom.numpy()[:na].astype(np.int64)
    wid = d.contact.worldid.numpy()[:na].astype(np.int64)
    pos = d.contact.pos.numpy()[:na].astype(np.float64)
    frame = d.contact.frame.numpy()[:na].astype(np.float64).reshape(na, 3, 3)
    dim = d.contact.dim.numpy()[:na].astype(np.int64)
    adr = d.contact.efc_address.numpy()[:na].astype(np.int64)
    gx = d.geom_xpos.numpy().astype(np.float64)
    gm = d.geom_xmat.numpy().astype(np.float64).reshape(n, mjm.ngeom, 3, 3)
    nefc = d.nefc.numpy().reshape(-1)[:n].astype(np.int64)
    etype, eid = d.efc.type.numpy()[:n], d.efc.id.numpy()[:n].astype(np.int64)
    force, state = d.efc.force.numpy()[:n], d.efc.state.numpy()[:n]
    g1, g2 = geom[:, 0], geom[:, 1]
    cap = np.where(plane[g1], g2, g1)
    pc = (plane[g1] | plane[g2]) & capsule[cap]
    ax = gm[wid, cap][:, :, 2]
    end = np.where(np.sum((pos - gx[wid, cap]) * ax, 1) > 0, 0, 1)
    pair = pair_of[g1, g2]
    if np.any(pair < 0):
        raise RuntimeError("a contact that is not one of the model's explicit pairs (the code needs a pair index)")
    code = pair * 4 + np.where(pc, end, 0)
    nrow = np.where(dim > 1, 2 * (dim - 1), 1)
    zone = np.zeros(na, np.int64)
    zone_st = np.zeros(na, np.int64)
    for e in range(adr.shape[1] if na else 0):
        use = e < nrow
        a = np.where(use, adr[:, e], 0)
        zone |= np.where(use & (force[wid, a] > 0), 1 << e, 0)
        zone_st |= np.where(use & (state[wid, a] == QUADRATIC), 1 << e, 0)
    order = np.lexsort((code, wid))  # stable: equal codes keep MuJoCo Warp's order
    out = dict(con_code=np.full((n, common.MAXC), -1, np.int64), con_t=np.zeros((n, common.MAXC, 6), np.float32),
               con_dim=np.zeros((n, common.MAXC), np.int32), con_zone=np.zeros((n, common.MAXC), np.int32),
               con_zone_state=np.zeros((n, common.MAXC), np.int32))
    ws = wid[order]
    first = np.searchsorted(ws, ws, side="left")
    rank = np.arange(len(ws)) - first
    keep = rank < common.MAXC
    o, w, r = order[keep], ws[keep], rank[keep]
    out["con_code"][w, r] = code[o]
    out["con_t"][w, r] = frame[o, 1:3].reshape(-1, 6)
    out["con_dim"][w, r] = dim[o]
    out["con_zone"][w, r] = zone[o]
    out["con_zone_state"][w, r] = zone_st[o]
    rows = np.arange(etype.shape[1])[None, :] < nefc[:, None]
    lim = rows & (etype == LIMIT_JOINT)
    bit = np.where(lim, np.left_shift(np.int64(1), np.where(lim, eid, 0)), 0)
    out["lim_rows"] = bit.sum(1)
    out["lim_act"] = np.where(force > 0, bit, 0).sum(1)
    out["lim_act_state"] = np.where(state[:, :etype.shape[1]] == QUADRATIC, bit, 0).sum(1)
    out["ncon"] = np.bincount(wid, minlength=n)[:n].astype(np.int32)
    out["nefc"] = nefc.astype(np.int32)
    out["niter"] = d.solver_niter.numpy().reshape(-1)[:n].astype(np.int32)
    return out


def run(prob, arrays, iterations=ITERATIONS, nconmax=NCONMAX, njmax=NJMAX):
    """(arrays (module docstring), meta) of one native MuJoCo Warp control step of every state in `arrays`."""
    import mujoco
    import warp as wp
    wp.config.quiet = True
    if hasattr(wp, "LOG_WARNING"):  # Warp >= 1.18 logs every module load at the default level
        wp.config.log_level = wp.LOG_WARNING
    import mujoco_warp as mjw
    if not wp.is_cuda_available():
        raise RuntimeError("Warp sees no CUDA device (a CPU-only fallback, e.g. the CUDA 13 warp-lang wheel on a "
                           "CUDA 12 driver or on Volta): use the +cu12 wheel (benchmarks/envs/setup.sh)")
    dev = wp.get_device("cuda:0")
    t0 = time.perf_counter()
    mjm = P.load_model(prob, iterations=iterations)
    n = len(arrays["qpos"])
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    pair_of = np.full((mjm.ngeom, mjm.ngeom), -1, np.int64)
    for i in range(mjm.npair):
        pair_of[mjm.pair_geom1[i], mjm.pair_geom2[i]] = pair_of[mjm.pair_geom2[i], mjm.pair_geom1[i]] = i
    plane, capsule = mjm.geom_type == PLANE, mjm.geom_type == CAPSULE
    with wp.ScopedDevice(dev):
        if prob.dr:
            m = mjw.put_model(mjm, batch_sizes={"pair_friction": n, "body_ipos": n})
            pf, bi = P.DR(mjm, prob).batched(arrays["friction"], arrays["com"], np.float32)
            m.pair_friction.assign(pf)
            m.body_ipos.assign(bi)
        else:
            m = mjw.put_model(mjm)
        ot = getattr(mjw, "OverflowType", None)
        if ot is None:
            from mujoco_warp._src.types import OverflowType as ot
        m.opt.warn_overflow = int(m.opt.warn_overflow) & ~int(ot.ITERATIONS | ot.LS_ITERATIONS)
        d = mjw.put_data(mjm, mjd, nworld=n, nconmax=nconmax, njmax=njmax, naconmax=nconmax * n)
        f32 = lambda x: wp.array(np.ascontiguousarray(x, np.float32), dtype=float)  # noqa: E731
        for dst, k in ((d.qpos, "qpos"), (d.qvel, "qvel"), (d.qacc_warmstart, "qacc_warmstart"), (d.ctrl, "ctrl")):
            wp.copy(dst, f32(arrays[k]))
        d.overflow.zero_()
        subs = []
        for _ in range(prob.substeps):
            mjw.step(m, d)
            wp.synchronize_device(dev)
            subs.append(_substep(mjm, d, n, pair_of, plane, capsule))
        ov = d.overflow.numpy().reshape(-1)[:n].astype(np.int64)
        out = {k: np.stack([s[k] for s in subs], 1) for k in subs[0]}
        out.update(out_qpos=d.qpos.numpy().astype(np.float64), out_qvel=d.qvel.numpy().astype(np.float64),
                   out_warm=d.qacc_warmstart.numpy().astype(np.float64))
        tol = float(np.asarray(m.opt.tolerance.numpy()).reshape(-1)[0]) if hasattr(m.opt.tolerance, "numpy") \
            else float(m.opt.tolerance)
    cap_mask = 0
    for f in ot:
        if f.value and not (f.value & (f.value - 1)) and f.name not in ("ITERATIONS", "LS_ITERATIONS"):
            cap_mask |= int(f.value)
    out["overflow"] = ov
    meta = dict(engine="mujoco_warp (native)", versions={k: _version(k) for k in ("mujoco", "mujoco-warp", "warp-lang",
                                                                                   "numpy")},
                device=dev.name, problem=prob.id, substeps=int(prob.substeps), iterations=int(iterations),
                ls_iterations=int(mjm.opt.ls_iterations), tolerance_model=float(mjm.opt.tolerance),
                tolerance_effective=tol, capacity=dict(nconmax=nconmax, njmax=njmax, naconmax=nconmax * n),
                warn_overflow="default without ITERATIONS and LS_ITERATIONS",
                deterministic=str(int(m.opt.deterministic)),
                capacity_overflow_worlds=int(((ov & cap_mask) != 0).sum()),
                iteration_limit_worlds=int(((ov & int(ot.ITERATIONS)) != 0).sum()),
                zone="efc.force > 0 (con_zone, lim_act); efc.state == QUADRATIC (con_zone_state, lim_act_state)",
                seconds=round(time.perf_counter() - t0, 2), inputs=common.inputs_digest(arrays), states=n)
    return out, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", required=True, help="corpus name (problem.load_corpus) or .npz path")
    ap.add_argument("--model-path", help="model file of the corpus's problem (default: the problem's)")
    ap.add_argument("--iterations", type=int, default=ITERATIONS)
    ap.add_argument("--out", required=True, help="npz to write (meta as a JSON string under 'meta')")
    ap.add_argument("--allow-overflow", action="store_true", help="write the labels even if a capacity overflowed")
    a = ap.parse_args(argv)
    corpus = P.load_corpus(a.corpus)
    prob = common.problem_of(corpus, a.model_path)
    common.check_model(corpus, prob)
    out, meta = run(prob, corpus.arrays, a.iterations)
    meta["corpus"] = corpus.name
    if meta["capacity_overflow_worlds"] and not a.allow_overflow:
        print(json.dumps(meta), file=sys.stderr)
        print(f"mjw_labels: {meta['capacity_overflow_worlds']} worlds overflowed the capacity", file=sys.stderr)
        return 1
    np.savez_compressed(a.out, meta=json.dumps(meta, sort_keys=True), **out)
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())

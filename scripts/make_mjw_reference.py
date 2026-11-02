"""MuJoCo Warp reference of a benchmark corpus for the forward checks: native PyPI mujoco-warp 3.17.0 (mujoco 3.17.0,
warp-lang 1.18.0), its public API only. Runs in an environment with those packages (benchmarks/envs: the mjw-native
environment); it imports numpy, mujoco, warp, mujoco_warp and benchmarks.bench, nothing from nablaengine.

Per state of the corpus (benchmarks/data/<corpus>.npz):
  - one control step at --iterations (default 50) with benchmarks.bench.truth.mjw_labels.run, the settings of the
    benchmark truth's model check: 4 x mjw.step with ctrl held, the per-world foot friction and torso COM offset
    through put_model(batch_sizes={"pair_friction", "body_ipos"}) filled in float32, 64 contacts and 192 rows per
    world (the capacity at which the committed reference was computed, NablaEngine release 0.1.0), opt.warn_overflow
    without ITERATIONS and LS_ITERATIONS (a capacity overflow is an error);
  - the contacts of MuJoCo Warp's collision at the input state (mjw.forward on the same model and inputs).

Output, a compressed npz written with fixed timestamps (default fixtures/mjlab/mjw317_<corpus>.npz, which
nablaengine.fixtures.load_corpus merges into the corpus as out_qpos / out_qvel / ncon / nefc / solver_niter and
load_contacts reads):
  out_qpos (S, nq), out_qvel (S, nv) float32   the state after the control step
  ncon, nefc, niter (S, 4) int16               contacts, constraint rows and solver iterations per substep
  con (S, K, 8) float32                        contacts at the input state, rows (pair, dist, pos[3], normal[3])
                                               sorted by pair and dist, padded with pair -1 (K: the largest count)
  con_count (S,) int16                         contacts per world at the input state
  meta                                         0-d JSON string: versions, device, options, corpus digests

MuJoCo Warp's atomics make a rerun differ in the last bits, so a regenerated file is not byte-identical; the forward
checks compare with tolerances (scripts/engine_step_check.py --gate).

  PYTHONPATH=. <mjw-native python> scripts/make_mjw_reference.py --corpora mf_s0,mf_s1
"""

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.bench import problem as P  # noqa: E402
from benchmarks.bench.truth import common, mjw_labels  # noqa: E402

OUT_DIR = ROOT / "fixtures" / "mjlab"
NAME = "mjw317_{corpus}.npz"
_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def write_npz(path, arrays):
    """A compressed npz with fixed member timestamps, written atomically."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for k in sorted(arrays):
            buf = io.BytesIO()
            v = np.asarray(arrays[k])  # (np.ascontiguousarray would make the 0-d meta 1-d)
            np.lib.format.write_array(buf, v if v.flags.c_contiguous else v.copy(order="C"), allow_pickle=False)
            info = zipfile.ZipInfo(f"{k}.npy", date_time=_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, buf.getvalue())
    tmp.replace(path)


def input_contacts(prob, arrays, nconmax=mjw_labels.NCONMAX, njmax=mjw_labels.NJMAX):
    """(con (S, K, 8), count (S,)): MuJoCo Warp's contacts at the input states (module docstring)."""
    import mujoco
    import warp as wp
    wp.config.quiet = True
    import mujoco_warp as mjw
    mjm = P.load_model(prob, iterations=mjw_labels.ITERATIONS)
    n = len(arrays["qpos"])
    pair_of = np.full((mjm.ngeom, mjm.ngeom), -1, np.int64)
    for i in range(mjm.npair):
        pair_of[mjm.pair_geom1[i], mjm.pair_geom2[i]] = pair_of[mjm.pair_geom2[i], mjm.pair_geom1[i]] = i
    with wp.ScopedDevice(wp.get_device("cuda:0")):
        m = mjw.put_model(mjm, batch_sizes={"pair_friction": n, "body_ipos": n})
        pf, bi = P.DR(mjm, prob).batched(arrays["friction"], arrays["com"], np.float32)
        m.pair_friction.assign(pf)
        m.body_ipos.assign(bi)
        m.opt.warn_overflow = 0  # mjw.forward also runs the solver; only the collision is kept
        mjd = mujoco.MjData(mjm)
        d = mjw.put_data(mjm, mjd, nworld=n, nconmax=nconmax, njmax=njmax, naconmax=nconmax * n)
        f32 = lambda x: wp.array(np.ascontiguousarray(x, np.float32), dtype=float)  # noqa: E731
        for dst, k in ((d.qpos, "qpos"), (d.qvel, "qvel"), (d.qacc_warmstart, "qacc_warmstart"), (d.ctrl, "ctrl")):
            wp.copy(dst, f32(arrays[k]))
        mjw.forward(m, d)
        wp.synchronize()
        na = min(int(d.nacon.numpy()[0]), d.contact.worldid.shape[0])
        geom = d.contact.geom.numpy()[:na].astype(np.int64)
        wid = d.contact.worldid.numpy()[:na].astype(np.int64)
        dist = d.contact.dist.numpy()[:na].astype(np.float32)
        pos = d.contact.pos.numpy()[:na].astype(np.float32)
        frame = d.contact.frame.numpy()[:na].astype(np.float32).reshape(na, 3, 3)
    per = [[] for _ in range(n)]
    for k in range(na):
        if dist[k] >= 0:
            continue
        pr = pair_of[geom[k, 0], geom[k, 1]]
        if pr < 0:
            raise RuntimeError("a contact that is not one of the model's explicit pairs")
        per[wid[k]].append((float(pr), float(dist[k]), *pos[k].tolist(), *frame[k, 0].tolist()))
    count = np.array([len(p) for p in per], np.int16)
    K = max(int(count.max()), 1)
    con = np.zeros((n, K, 8), np.float32)
    con[:, :, 0] = -1
    for w in range(n):
        for c, row in enumerate(sorted(per[w])):
            con[w, c] = row
    return con, count


def make(name, iterations, out_dir):
    corpus = P.load_corpus(name)
    prob = common.problem_of(corpus)
    desc = common.check_model(corpus, prob)
    out, meta = mjw_labels.run(prob, corpus.arrays, iterations)
    if meta["capacity_overflow_worlds"]:
        raise RuntimeError(f"{name}: {meta['capacity_overflow_worlds']} worlds overflowed the capacity")
    con, count = input_contacts(prob, corpus.arrays)
    arrays = dict(out_qpos=out["out_qpos"].astype(np.float32), out_qvel=out["out_qvel"].astype(np.float32),
                  ncon=out["ncon"].astype(np.int16), nefc=out["nefc"].astype(np.int16),
                  niter=out["niter"].astype(np.int16), con=con, con_count=count)
    for k in ("out_qpos", "out_qvel"):  # MuJoCo Warp's state is float32: the cast is exact
        assert np.array_equal(arrays[k].astype(np.float64), out[k]), k
    meta = dict(format="dk-mjw-reference/1", corpus=dict(name=name, files=corpus.files,
                                                         inputs=common.inputs_digest(corpus.arrays)),
                model=dict(path=desc.get("path"), sha256=desc.get("sha256")),
                contacts="mjw.forward at the input state, dist < 0, rows (pair, dist, pos, normal)", **{
                    k: meta[k] for k in ("engine", "versions", "device", "problem", "substeps", "iterations",
                                         "ls_iterations", "tolerance_model", "tolerance_effective", "capacity",
                                         "warn_overflow", "deterministic", "iteration_limit_worlds", "states")})
    arrays["meta"] = np.array(json.dumps(meta, sort_keys=True))
    path = Path(out_dir) / NAME.format(corpus=name)
    write_npz(path, arrays)
    print(json.dumps(dict(corpus=name, path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                          bytes=path.stat().st_size, contacts_max=int(count.max()), contacts_mean=float(count.mean()),
                          niter_max=int(arrays["niter"].max()), versions=meta["versions"], device=meta["device"])),
          flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpora", default="mf_s0,mf_s1", help="benchmark corpora (benchmarks/data/<name>.npz)")
    ap.add_argument("--iterations", type=int, default=mjw_labels.ITERATIONS)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    a = ap.parse_args(argv)
    for name in a.corpora.split(","):
        make(name.strip(), a.iterations, a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

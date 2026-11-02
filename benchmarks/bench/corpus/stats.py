"""Distribution statistics of benchmark corpora with MuJoCo C (benchmarks/data/README.md), optionally against
reference corpora.

Every state of every corpus runs one control step in MuJoCo C float64 (corpus/labels.py: the corpus's model,
substeps and per-state DR) at 10 and at 50 solver iterations; per corpus it reports contacts and constraint rows per
world (the most in any substep, and at the input state), solver iterations, |qvel| of the inputs, the pre-fall share
(the generator's label), and the shares of states with a self contact, a clamped actuator, a joint-limit row, the
iteration limit reached, and more than 64 contacts or 192 rows (the capacity at which the committed statistics were
computed: NablaEngine release 0.1.0, 192 rows), with histograms.
The same code labels every corpus, so the numbers compare like with like (they are MuJoCo C's). With --ref, for each
Track A benchmark corpus it lists the compared statistics (COMPARED) outside the range the reference corpora span,
and outside that range widened by 10% of a bound (5 percentage points for shares); the lists inform, they do not fail
the run.

  python -m benchmarks.bench.corpus.stats --threads 16 --corpora mf_s0,mf_s1,mf_s2,mf_s3,menagerie_s0 \
      --out benchmarks/data/corpus_stats.json --table

--corpora are benchmark corpora (<data>/<name>.npz, corpus/format.py; model and options from their metadata);
--ref (optional, default none) are reference corpora of the mjlab G1 in the engine's fixture format
(<ref-dir>/<name>_fwd.npz next to g1_train.mjb and g1_meta.json; --ref-dir default $DK_BENCH_FIXTURE_DIR). --out
refuses to overwrite an existing file without --force. Exit status 1 on any failure.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from benchmarks.bench.corpus import format as corpus_format
from benchmarks.bench.corpus.labels import Labeller

REPO = Path(__file__).resolve().parents[3]
CAPACITY = (64, 192)  # the capacity at which the committed statistics were computed (release 0.1.0, 192 rows)
ITERATIONS = (10, 50)
QUANTILES = (0.5, 0.9, 0.99)
# statistics compared against the range of the reference corpora (path in the corpus entry)
COMPARED = ("it10.ncon_max.mean", "it10.ncon_max.p99", "it10.nefc_max.mean", "it10.nefc_max.p99",
            "it10.niter_max.mean", "it50.niter_max.mean", "it50.niter_max.p99", "qvel_norm.p50", "qvel_norm.p99",
            "shares.prefall", "shares.self_contact", "shares.clamped", "shares.limit", "it10.nonconv_share")
TOL_REL, TOL_SHARE = 0.10, 0.05  # the widened range of acceptance(): 10% of a bound, 5 percentage points of a share


def rel(p):
    p = Path(p).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return f".../{p.name}"


def summary(x):
    x = np.asarray(x, np.float64)
    out = dict(mean=float(x.mean()), max=float(x.max()))
    out.update({f"p{int(q * 100)}": float(np.quantile(x, q)) for q in QUANTILES})
    return out


def hist(x, width=1):
    x = np.asarray(x, np.int64)
    edges = np.arange(0, (x.max() // width + 2) * width, width)
    h, _ = np.histogram(x, edges)
    return dict(bin_width=width, counts=h.tolist())


def load_bench(path):
    """A benchmark corpus: (states, model, substeps, dr, prefall, info)."""
    z = corpus_format.read(path)
    meta = z["metadata"]
    mrec = meta["model"]
    if "repo" in mrec:  # a model fetched at a pinned commit (models/fetch.py)
        from benchmarks.bench.models import fetch
        name = next((k for k, v in fetch.pins().items() if v["repo"] == mrec["repo"] and v["entry"] == mrec["file"]
                     and v["commit"] == mrec["commit"]), None)
        if name is None:
            raise RuntimeError(f"{rel(path)}: no pin for {mrec['repo']}@{mrec['commit']} {mrec['file']}")
        mpath = fetch.scene_path(name)
    else:
        mpath = REPO / mrec["path"]
    if corpus_format.sha256(mpath) != mrec["sha256"]:
        raise RuntimeError(f"{rel(path)}: model {rel(mpath)} does not match the corpus's sha256")
    m = mujoco.MjModel.from_binary_path(str(mpath)) if mpath.suffix == ".mjb" else \
        mujoco.MjModel.from_xml_path(str(mpath))
    ph = meta["physics"]
    m.opt.ls_iterations, m.opt.tolerance = ph["ls_iterations"], ph["tolerance"]
    dr = None
    if meta.get("dr"):
        torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, meta["dr"]["torso_body"])
        dr = (np.asarray(meta["dr"]["foot_pairs"], int), torso)
    info = dict(kind="benchmark", file=rel(path), sha256=corpus_format.sha256(path), track=meta["track"],
                model=mrec.get("path") or f"{mrec['repo']}@{mrec['commit'][:12]}:{mrec['file']}",
                substeps=ph["substeps"], generator=meta["generator"], seed=meta["seed"])
    return z, m, ph["substeps"], dr, z["label_prefall"], info


def load_ref(fixtures, name):
    """A reference corpus of the mjlab G1 in the fixture format (<fixtures>/<name>_fwd.npz)."""
    d = Path(fixtures)
    mt = json.loads((d / "g1_meta.json").read_text())
    m = mujoco.MjModel.from_binary_path(str(d / "g1_train.mjb"))
    m.opt.ls_iterations, m.opt.tolerance = 8, 1e-8
    path = d / f"{name}_fwd.npz"
    with np.load(path) as f:
        z = {k: f[k] for k in f.files}
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, mt["torso_body"])
    info = dict(kind="reference", file=rel(path), sha256=corpus_format.sha256(path), track="A",
                model=rel(d / "g1_train.mjb"), substeps=int(mt["substeps_per_control_step"]))
    return z, m, int(mt["substeps_per_control_step"]), (np.asarray(mt["foot_pairs"], int), torso), z["prefall"], info


def corpus_stats(z, m, substeps, dr, prefall, threads):
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
    lab = Labeller(m, substeps, dr=dr, threads=threads)
    out = dict(n=int(len(z["qpos"])))
    qv = np.linalg.norm(np.asarray(z["qvel"], np.float64), axis=1)
    out["qvel_norm"] = summary(qv)
    free = np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free):
        out["base_height"] = summary(np.asarray(z["qpos"][:, m.jnt_qposadr[free[0]] + 2], np.float64))
    u = np.asarray(z["ctrl"], np.float64)
    lim = m.actuator_ctrllimited.astype(bool)
    outside = (u < m.actuator_ctrlrange[:, 0]) | (u > m.actuator_ctrlrange[:, 1])
    out["ctrl_outside_ctrlrange"] = int(outside[:, lim].sum())
    shares = dict(prefall=float(np.mean(prefall)))
    for it in ITERATIONS:
        L = lab(z, it)
        over_c, over_r = L["ncon_max"] > CAPACITY[0], L["nefc_max"] > CAPACITY[1]
        out[f"it{it}"] = dict(
            ncon_max=summary(L["ncon_max"]), ncon0=summary(L["ncon0"]), nefc_max=summary(L["nefc_max"]),
            nefc0=summary(L["nefc0"]), niter_max=summary(L["niter_max"]),
            nonconv_share=float(L["nonconv"].mean()), self_contact=float(L["self_contact"].mean()),
            clamped=float(L["clamped"].mean()), limit=float(L["limit"].mean()),
            over_contacts=float(over_c.mean()), over_rows=float(over_r.mean()),
            over_capacity=float((over_c | over_r).mean()), bad=int(L["bad"].sum()),
            hist=dict(ncon_max=hist(L["ncon_max"]), nefc_max=hist(L["nefc_max"], 8), niter_max=hist(L["niter_max"])))
        if it == ITERATIONS[0]:
            shares.update(self_contact=out[f"it{it}"]["self_contact"], clamped=out[f"it{it}"]["clamped"],
                          limit=out[f"it{it}"]["limit"], over_capacity=out[f"it{it}"]["over_capacity"])
    out["shares"] = shares
    return out


def get(d, path):
    for k in path.split("."):
        d = d[k]
    return d


def acceptance(res, dev):
    """Per benchmark corpus of Track A: the COMPARED statistics outside [min, max] of the reference corpora dev
    (`outside`), and outside that range widened by TOL_SHARE (shares, absolute) or TOL_REL of the nearer bound
    (other statistics) (`outside_tolerance`). Informational: a few reference corpora span a narrow range."""
    out = {}
    if not dev:
        return out
    for name, r in res.items():
        if r["info"]["kind"] != "benchmark" or r["info"]["track"] != "A":
            continue
        rows = {}
        for p in COMPARED:
            lo, hi = min(get(res[d], p) for d in dev), max(get(res[d], p) for d in dev)
            v = get(r, p)
            share = p.startswith("shares.") or p.endswith("_share")
            tol_lo, tol_hi = (TOL_SHARE, TOL_SHARE) if share else (TOL_REL * abs(lo), TOL_REL * abs(hi))
            rows[p] = dict(value=v, ref_min=lo, ref_max=hi, inside=bool(lo <= v <= hi),
                           inside_tolerance=bool(lo - tol_lo <= v <= hi + tol_hi))
        out[name] = dict(outside=[p for p, x in rows.items() if not x["inside"]],
                         outside_tolerance=[p for p, x in rows.items() if not x["inside_tolerance"]], rows=rows)
    return out


def table(res):
    """Markdown table of the main statistics, one column per corpus."""
    names = list(res)
    f = lambda v, d=2: f"{v:.{d}f}"  # noqa: E731
    pct = lambda v: f"{100 * v:.1f}%"  # noqa: E731
    rows = [
        ("kind", lambda r: r["info"]["kind"] + (", Track B" if r["info"]["track"] == "B" else "")),
        ("states", lambda r: str(r["n"])),
        ("contacts / world, mean (it10)", lambda r: f(r["it10"]["ncon_max"]["mean"])),
        ("contacts / world, p99 / max", lambda r: f"{r['it10']['ncon_max']['p99']:.0f} / "
                                                  f"{r['it10']['ncon_max']['max']:.0f}"),
        ("contacts at the input state, mean", lambda r: f(r["it10"]["ncon0"]["mean"])),
        ("rows / world, mean", lambda r: f(r["it10"]["nefc_max"]["mean"], 1)),
        ("rows / world, p99 / max", lambda r: f"{r['it10']['nefc_max']['p99']:.0f} / "
                                              f"{r['it10']['nefc_max']['max']:.0f}"),
        ("solver iterations at 10, mean / p90", lambda r: f"{f(r['it10']['niter_max']['mean'])} / "
                                                          f"{r['it10']['niter_max']['p90']:.0f}"),
        ("at the limit of 10", lambda r: pct(r["it10"]["nonconv_share"])),
        ("solver iterations at 50, mean / p99 / max", lambda r: f"{f(r['it50']['niter_max']['mean'])} / "
                                                                f"{r['it50']['niter_max']['p99']:.0f} / "
                                                                f"{r['it50']['niter_max']['max']:.0f}"),
        ("at the limit of 50", lambda r: pct(r["it50"]["nonconv_share"])),
        ("\\|qvel\\|, p50 / p99 / max", lambda r: f"{f(r['qvel_norm']['p50'])} / {f(r['qvel_norm']['p99'])} / "
                                                  f"{f(r['qvel_norm']['max'])}"),
        ("pre-fall", lambda r: pct(r["shares"]["prefall"])),
        ("self contact", lambda r: pct(r["shares"]["self_contact"])),
        ("clamped actuator", lambda r: pct(r["shares"]["clamped"])),
        ("joint-limit row", lambda r: pct(r["shares"]["limit"])),
        ("> 64 contacts or > 192 rows", lambda r: pct(r["shares"]["over_capacity"])),
        ("ctrl entries outside the ctrlrange", lambda r: str(r["ctrl_outside_ctrlrange"])),
    ]
    lines = ["| | " + " | ".join(f"`{n}`" for n in names) + " |", "|---" * (len(names) + 1) + "|"]
    for label, fn in rows:
        lines.append(f"| {label} | " + " | ".join(fn(res[n]) for n in names) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpora", default="mf_s0,mf_s1,mf_s2,mf_s3,menagerie_s0", help="benchmark corpora")
    ap.add_argument("--ref", default="", help="reference corpora in the fixture format (default: none)")
    ap.add_argument("--data", default=os.environ.get("DK_BENCH_DATA") or str(REPO / "benchmarks" / "data"),
                    help="benchmark corpus directory (default $DK_BENCH_DATA, else benchmarks/data)")
    ap.add_argument("--ref-dir", default=os.environ.get("DK_BENCH_FIXTURE_DIR"),
                    help="directory of the reference corpora (default $DK_BENCH_FIXTURE_DIR)")
    ap.add_argument("--threads", type=int, default=None, help="threads (default: os.sched_getaffinity)")
    ap.add_argument("--out", default=None, help="statistics JSON")
    ap.add_argument("--force", action="store_true", help="overwrite --out")
    ap.add_argument("--table", action="store_true", help="print the markdown table")
    args = ap.parse_args(argv)
    if args.out and Path(args.out).exists() and not args.force:
        raise SystemExit(f"{args.out} exists (use --force to overwrite)")
    threads = args.threads or len(os.sched_getaffinity(0))
    t0 = time.time()
    res = {}
    dev = [n for n in args.ref.split(",") if n]
    if dev and not args.ref_dir:
        raise SystemExit("--ref needs --ref-dir (or $DK_BENCH_FIXTURE_DIR)")
    for n in dev:
        z, m, sub, dr, pre, info = load_ref(args.ref_dir, n)
        res[n] = dict(info=info, **corpus_stats(z, m, sub, dr, pre, threads))
        print(f"{n}: {time.time() - t0:.0f} s", flush=True)
    for n in [n for n in args.corpora.split(",") if n]:
        z, m, sub, dr, pre, info = load_bench(Path(args.data) / f"{n}.npz")
        res[n] = dict(info=info, **corpus_stats(z, m, sub, dr, pre, threads))
        print(f"{n}: {time.time() - t0:.0f} s", flush=True)
    acc = acceptance(res, dev)
    for n, a in acc.items():
        print(f"{n}: outside the reference range: {a['outside'] or 'none'}; outside it widened "
              f"({TOL_REL:.0%} / {TOL_SHARE:.0%} points): {a['outside_tolerance'] or 'none'}")
    if args.table:
        print(table(res))
    if args.out:
        prov = corpus_format.data_provenance()
        doc = dict(schema="dk-bench-corpus-stats/1", capacity=dict(contacts=CAPACITY[0], rows=CAPACITY[1]),
                   tolerance=dict(relative=TOL_REL, share=TOL_SHARE),
                   iterations=list(ITERATIONS), simulator="MuJoCo C float64 (benchmarks/bench/corpus/labels.py)",
                   command="python -m benchmarks.bench.corpus.stats "
                   + " ".join(sys.argv[1:] if argv is None else argv),
                   corpora=res, **(dict(acceptance=acc) if dev else {}), seconds=round(time.time() - t0, 1),
                   provenance=prov)
        tmp = Path(args.out).with_name(Path(args.out).name + ".tmp")
        tmp.write_text(json.dumps(doc, indent=1) + "\n")
        os.replace(tmp, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

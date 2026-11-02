"""Mocap-free benchmark corpus generator: MuJoCo C float64 rollouts of a PD-controlled humanoid driven by random
joint setpoints and base pushes, with falls and restarts, sampled into a corpus of independent control-step inputs
(benchmarks/data/README.md; file format: corpus/format.py).

Model requirements, and nothing else robot-specific: one free joint (the floating base), position actuators on joint
transmissions (ctrl is a joint setpoint), and for per-state domain randomization the foot-floor pair indices and the
torso body (g1_meta.json's foot_pairs and torso_body for the G1). Self contacts are contacts between two geoms of
bodies not welded to the world; falls are a height / tilt termination of the free joint (below).

Rollout: --worlds worlds x --steps control steps of --substeps mj_step calls with ctrl held, at --iterations solver
iterations, by mujoco.batch.Batch on --threads threads (bit-identical at any thread count):
  reset      a keyframe (--keyframes, each equally likely; qpos0 if the model has none) plus reset noise: base
             position U(+-(0.05, 0.05, 0.01)) m, orientation U(+-(0.1, 0.1, 0.2)) rad (a rotation vector), base
             velocity U(+-(0.5, 0.5, 0.2, 0.52, 0.52, 0.78)), joint positions U(+-0.1) rad clipped to 90% of their
             range, qacc_warmstart 0; with DR a per-episode domain randomization: foot-floor friction U[0.3, 1.2],
             torso COM offset U(+-(0.025, 0.05, 0.05)) m
  setpoints  the keyframe's joint pose plus an Ornstein-Uhlenbeck process per joint (stationary standard deviation
             --ou-sigma x action_scale, action_scale = 0.25 effort / kp per joint; correlation time
             --ou-tau s); at --jump-rate jumps per second, a jump moves each joint with probability --jump-share to a
             setpoint uniform in 90% of its range, from where the process relaxes back; setpoints are clipped to 90%
             of the joint range and to the ctrlrange
  pushes     every U[--kick-min, --kick-max] s the base velocity gets U(+-(0.5, 0.5, 0.2, 0.52, 0.52, 0.78)) x
             --kick-scale (m/s, rad/s)
  falls      a termination (TERM_HEIGHT, TERM_TILT) against the episode's start pose (the keyframe): the base
             height off by more than 0.25 m or the z component of gravity in the base frame off by more than 0.8;
             also a MuJoCo bad-state warning or a non-finite state. The world restarts with a new reset at the next
             control step.
Sampling of --states states from the (world, step) pairs: up to half from the last
--pre-fall control steps of an episode up to its fall, then states with a self contact until at least --self-share
of the corpus has one, the rest uniform; no capacity filter (states beyond 64 contacts or 192 rows stay); states whose
own control step raised a MuJoCo bad-state warning or went non-finite are excluded. The corpus is saved in a fixed
seeded permutation with inputs rounded to float32, --cotangents N(0, 1) float32 output cotangents per state and the
MuJoCo C labels of corpus/labels.py at --iterations, computed from the stored (rounded) inputs.

Track presets (--track): A = the mjlab G1 (fixtures/mjlab/g1_train.mjb, g1_meta.json: 4 substeps, DR on, keyframes
home and knees_bent), B = MuJoCo Menagerie's unitree_g1/scene_mjx.xml at the commit of models/pins.json (fetched by
models/fetch.py; 5 substeps of its 4 ms timestep, no DR, keyframe home). Explicit options override a preset.

  python -m benchmarks.bench.corpus.gen_mf --track A --seed 0 --threads 24   # benchmarks/data/mf_s0.npz
  python -m benchmarks.bench.corpus.gen_mf --track B --seed 0 --threads 24   # .../menagerie_s0.npz

It writes <out>/<name>.npz (default directory: $DK_BENCH_DATA when set and not empty, else benchmarks/data; an
existing file is kept unless --force) and records the file's sha256, content digest, command and provenance in
<out>/corpora.json. Track B reads the model from $DK_BENCH_MODELS (models/fetch.py). Neither variable changes the
corpus bits. Exit status 1 on any failure.
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
GENERATOR = "benchmarks/bench/corpus/gen_mf.py"
VERSION = 1  # bump when a change alters the corpora a seed produces

# reset noise, domain randomization and fall termination of the rollouts (module docstring)
RESET_POS, RESET_RPY = np.array([0.05, 0.05, 0.01]), np.array([0.1, 0.1, 0.2])
RESET_VEL, RESET_JOINT = np.array([0.5, 0.5, 0.2, 0.52, 0.52, 0.78]), 0.1
COM_LIM = np.array([0.025, 0.05, 0.05])
FRICTION = (0.3, 1.2)
TERM_HEIGHT, TERM_TILT = 0.25, 0.8
SOFT = 0.9  # setpoints and joint reset noise stay within this share of the joint range
# contacts / constraint rows per world (reported, not filtered): the capacity at which the committed corpora and
# their statistics were computed (release 0.1.0, 192 rows)
CAPACITY = (64, 192)
BAD_WARNINGS = (int(mujoco.mjtWarning.mjWARN_BADQPOS), int(mujoco.mjtWarning.mjWARN_BADQVEL),
                int(mujoco.mjtWarning.mjWARN_BADQACC))
PICKS = ("prefall", "self_contact", "uniform")

TRACKS = dict(
    A=dict(model="fixtures/mjlab/g1_train.mjb", meta="fixtures/mjlab/g1_meta.json", substeps=None, dr=True,
           keyframes=None, name="mf_s{seed}"),
    B=dict(model=None, meta=None, substeps=5, dr=False, keyframes="home", name="menagerie_s{seed}"),
)


def rel(p):
    """Repository-relative path when inside the checkout, else the name only (no site paths in results)."""
    p = Path(p).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return f".../{p.name}"


def file_sha256(p):
    return corpus_format.sha256(p)


def quat_mul(a, b):
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def gravity_z(q):
    """z component of the world gravity direction (0, 0, -1) in the frame of quaternion q (wxyz)."""
    w, x, y, z = np.moveaxis(q, -1, 0)
    return -(1 - 2 * (x * x + y * y))


class Robot:
    """What the generator needs from a model: the free joint, the joints and the position actuators."""

    def __init__(self, m, keyframes):
        jt = m.jnt_type
        free = np.flatnonzero(jt == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) != 1:
            raise ValueError(f"need exactly one free joint (the floating base), the model has {len(free)}")
        self.qa, self.va = int(m.jnt_qposadr[free[0]]), int(m.jnt_dofadr[free[0]])
        if np.any((jt == mujoco.mjtJoint.mjJNT_BALL)):
            raise ValueError("ball joints are not supported by the reset noise")
        self.joints = np.flatnonzero((jt == mujoco.mjtJoint.mjJNT_HINGE) | (jt == mujoco.mjtJoint.mjJNT_SLIDE))
        lim = m.jnt_limited[self.joints].astype(bool)
        rng = m.jnt_range[self.joints]
        mid, half = rng.mean(1), 0.5 * (rng[:, 1] - rng[:, 0]) * SOFT
        self.j_q = m.jnt_qposadr[self.joints]
        self.j_lo = np.where(lim, mid - half, -np.inf)
        self.j_hi = np.where(lim, mid + half, np.inf)
        ok = ((m.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT) & (m.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED)
              & (m.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE) & (m.actuator_dyntype == mujoco.mjtDyn.mjDYN_NONE)
              & np.isclose(m.actuator_biasprm[:, 1], -m.actuator_gainprm[:, 0]) & (m.actuator_gainprm[:, 0] > 0))
        if not ok.all():
            raise ValueError(f"actuators {np.flatnonzero(~ok).tolist()} are not position servos on a joint "
                             "(ctrl must be a joint setpoint)")
        jid = m.actuator_trnid[:, 0]
        if not np.isin(jid, self.joints).all():
            raise ValueError("an actuator drives a joint that is not a hinge or slide")
        self.a_q = m.jnt_qposadr[jid]
        jrow = np.searchsorted(self.joints, jid)
        self.a_lo, self.a_hi = self.j_lo[jrow], self.j_hi[jrow]
        clim = m.actuator_ctrllimited.astype(bool)
        self.c_lo = np.where(clim, m.actuator_ctrlrange[:, 0], -np.inf)
        self.c_hi = np.where(clim, m.actuator_ctrlrange[:, 1], np.inf)
        kp = m.actuator_gainprm[:, 0]
        effort = np.where(m.jnt_actfrclimited[jid].astype(bool), m.jnt_actfrcrange[jid, 1],
                          np.where(m.actuator_forcelimited.astype(bool), m.actuator_forcerange[:, 1], np.nan))
        half_a = 0.5 * (self.a_hi - self.a_lo)
        self.scale = np.where(np.isfinite(effort), 0.25 * effort / kp, 0.25 * half_a)  # action_scale
        if not np.isfinite(self.scale).all():
            raise ValueError("an actuator without force limits drives an unlimited joint: no setpoint scale")
        names = [m.key(i).name for i in range(m.nkey)]
        if keyframes is None:
            keys = list(range(m.nkey))
        else:
            unknown = [k for k in keyframes if k not in names]
            if unknown:
                raise ValueError(f"keyframes {unknown} not in the model ({names})")
            keys = [names.index(k) for k in keyframes]
        if keys:
            self.key_names = [names[k] for k in keys]
            self.key_qpos, self.key_qvel = m.key_qpos[keys].copy(), m.key_qvel[keys].copy()
        else:
            self.key_names = ["qpos0"]
            self.key_qpos, self.key_qvel = m.qpos0[None].copy(), np.zeros((1, m.nv))
        self.key_set = self.key_qpos[:, self.a_q]  # setpoint centres per keyframe
        self.key_z = self.key_qpos[:, self.qa + 2]
        self.key_gz = gravity_z(self.key_qpos[:, self.qa + 3:self.qa + 7])


def load_model(path):
    p = Path(path)
    return mujoco.MjModel.from_binary_path(str(p)) if p.suffix == ".mjb" else mujoco.MjModel.from_xml_path(str(p))


def resolve(args):
    """Fill the options a track preset gives; returns (model path, meta dict or None, model record)."""
    pre = TRACKS[args.track]
    if args.substeps is None:
        args.substeps = pre["substeps"]
    if args.dr is None:
        args.dr = pre["dr"]
    if args.keyframes is None and pre["keyframes"]:
        args.keyframes = pre["keyframes"]
    if args.name is None:
        args.name = pre["name"].format(seed=args.seed)
    record = {}
    if args.model is None:
        if args.track == "B":
            from benchmarks.bench.models import fetch
            entry = fetch.pin("menagerie_g1")
            path = fetch.scene_path()
            # where the files are fetched to does not enter the corpus: the pinned repository file names the model
            record.update(repo=entry["repo"], commit=entry["commit"], file=entry["entry"],
                          files_sha256=fetch.files_digest(entry), license=entry["license"]["url"])
            args.model = str(path)
        else:
            args.model = str(REPO / pre["model"])
    meta = None
    meta_path = args.meta or (str(REPO / pre["meta"]) if pre["meta"] else None)
    if meta_path:
        meta = json.loads(Path(meta_path).read_text())
        record["meta"] = rel(meta_path)
        record["meta_sha256"] = file_sha256(meta_path)
    if args.substeps is None:
        if not meta or "substeps_per_control_step" not in meta:
            raise ValueError("--substeps is needed (the model metadata has no substeps_per_control_step)")
        args.substeps = int(meta["substeps_per_control_step"])
    if args.dr and not (meta and "foot_pairs" in meta and "torso_body" in meta):
        raise ValueError("DR needs --meta with foot_pairs and torso_body")
    return meta, record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", choices=sorted(TRACKS), default="A", help="model preset (default A)")
    ap.add_argument("--model", default=None, help="model file (.mjb / .xml); default: the track's")
    ap.add_argument("--meta", default=None, help="model metadata JSON (foot_pairs, torso_body, "
                                                 "substeps_per_control_step); default: the track's")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None, help="corpus name (default mf_s<seed> / menagerie_s<seed>)")
    ap.add_argument("--out", default=os.environ.get("DK_BENCH_DATA") or str(REPO / "benchmarks" / "data"),
                    help="output directory (default $DK_BENCH_DATA, else benchmarks/data)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing corpus file")
    ap.add_argument("--threads", type=int, default=None,
                    help="MuJoCo C threads (default: the CPUs this process may use, os.sched_getaffinity)")
    ap.add_argument("--worlds", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=150, help="control steps of the rollout")
    ap.add_argument("--substeps", type=int, default=None, help="mj_step calls per control step (default: the track's)")
    ap.add_argument("--iterations", type=int, default=10, help="solver iterations of the rollout and the labels")
    ap.add_argument("--ls-iterations", type=int, default=8)
    ap.add_argument("--tolerance", type=float, default=1e-8)
    ap.add_argument("--dr", action=argparse.BooleanOptionalAction, default=None,
                    help="per-state friction / COM randomization (default: the track's)")
    ap.add_argument("--keyframes", default=None, help="comma-separated keyframe names to start from (default: the "
                                                      "track's; A: all of the model's)")
    ap.add_argument("--ou-sigma", type=float, default=0.75, help="setpoint OU stationary std in action units")
    ap.add_argument("--ou-tau", type=float, default=0.4, help="setpoint OU correlation time (s)")
    ap.add_argument("--jump-rate", type=float, default=1.0, help="setpoint jumps per second")
    ap.add_argument("--jump-share", type=float, default=0.2, help="share of the joints a jump moves (each joint "
                                                                   "independently)")
    ap.add_argument("--kick-min", type=float, default=0.25, help="shortest time between base pushes (s)")
    ap.add_argument("--kick-max", type=float, default=1.0, help="longest time between base pushes (s)")
    ap.add_argument("--kick-scale", type=float, default=1.5, help="push size in units of the reset velocity noise")
    ap.add_argument("--states", type=int, default=4096)
    ap.add_argument("--pre-fall", type=int, default=25, help="control steps up to a fall that count as pre-fall")
    ap.add_argument("--self-share", type=float, default=0.15, help="least share of states with a self contact")
    ap.add_argument("--cotangents", type=int, default=3)
    ap.add_argument("--label-chunk", type=int, default=4096, help="pool states labelled per round of the "
                                                                  "self-contact search")
    args = ap.parse_args(argv)
    t0 = time.time()
    meta, model_rec = resolve(args)
    keyframes = None if args.keyframes is None else [k for k in args.keyframes.split(",") if k]
    threads = args.threads or len(os.sched_getaffinity(0))
    out_dir = Path(args.out)
    out_path = out_dir / f"{args.name}.npz"
    if out_path.exists() and not args.force:
        raise SystemExit(f"{rel(out_path)} exists (use --force to overwrite)")

    m = load_model(args.model)
    m.opt.iterations, m.opt.ls_iterations, m.opt.tolerance = args.iterations, args.ls_iterations, args.tolerance
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
    rob = Robot(m, keyframes)
    dr = None
    if args.dr:
        torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, meta["torso_body"])
        if torso < 0:
            raise ValueError(f"body {meta['torso_body']!r} not in the model")
        dr = (np.asarray(meta["foot_pairs"], int), torso)
    if "repo" not in model_rec:
        model_rec["path"] = rel(args.model)
    model_rec.update(sha256=file_sha256(args.model), nq=m.nq, nv=m.nv, nu=m.nu)

    W, T, nsub = args.worlds, args.steps, args.substeps
    dt = m.opt.timestep * nsub
    ss = np.random.SeedSequence(args.seed)
    r_roll, r_pick, r_perm, r_cot = (np.random.default_rng(s) for s in ss.spawn(4))
    from mujoco import batch as mjbatch
    b = mjbatch.Batch(m, nsim=W, nthread=threads)
    if dr is not None:
        pf, bi = b.expand("pair_friction"), b.expand("body_ipos")
        pf[:], bi[:] = m.pair_friction, m.body_ipos
    nq, nv, nu = m.nq, m.nv, m.nu
    qpos_b, qvel_b, ctrl_b, warm_b = b.bind("qpos"), b.bind("qvel"), b.bind("ctrl"), b.bind("qacc_warmstart")
    ncon_b, nefc_b = b.bind("ncon"), b.bind("nefc")

    # per-world episode state
    key = np.zeros(W, int)
    ep = np.zeros(W, int)
    x = np.zeros((W, nu))  # OU deviation of the setpoints from the keyframe pose
    kick_cd = np.zeros(W, int)
    fric, com = np.full(W, np.nan), np.zeros((W, 3))
    ou_a = np.exp(-dt / args.ou_tau)
    ou_b = np.sqrt(1 - ou_a ** 2) * args.ou_sigma * rob.scale
    p_jump = 1 - np.exp(-args.jump_rate * dt)
    kick_vel = RESET_VEL * args.kick_scale
    kmin, kmax = max(1, int(round(args.kick_min / dt))), max(1, int(round(args.kick_max / dt)))

    def reset(idx):
        k = len(idx)
        if not k:
            return
        r = r_roll
        kf = r.integers(len(rob.key_names), size=k)
        q, v = rob.key_qpos[kf].copy(), rob.key_qvel[kf].copy()
        a, va = rob.qa, rob.va
        q[:, a:a + 3] += r.uniform(-RESET_POS, RESET_POS, (k, 3))
        rpy = r.uniform(-RESET_RPY, RESET_RPY, (k, 3))
        ang = np.linalg.norm(rpy, axis=1, keepdims=True)
        dq = np.concatenate([np.cos(ang / 2), np.sin(ang / 2) * rpy / np.maximum(ang, 1e-9)], 1)
        q[:, a + 3:a + 7] = quat_mul(dq, q[:, a + 3:a + 7])
        v[:, va:va + 6] += r.uniform(-RESET_VEL, RESET_VEL, (k, 6))
        qj = q[:, rob.j_q] + r.uniform(-RESET_JOINT, RESET_JOINT, (k, len(rob.j_q)))
        q[:, rob.j_q] = np.clip(qj, rob.j_lo, rob.j_hi)
        qpos_b[idx], qvel_b[idx], warm_b[idx] = q, v, 0.0
        b.time[idx] = 0.0
        key[idx], ep[idx], x[idx] = kf, 0, 0.0
        kick_cd[idx] = r.integers(kmin, kmax + 1, size=k)
        if dr is not None:
            fric[idx] = r.uniform(*FRICTION, k)
            com[idx] = r.uniform(-COM_LIM, COM_LIM, (k, 3))
            pairs, torso = dr
            for i, f, c in zip(idx, fric[idx], com[idx]):
                pf[i, pairs, 0] = pf[i, pairs, 1] = f
                bi[i, torso] = m.body_ipos[torso] + c

    rec = dict(qpos=np.zeros((T, W, nq), np.float32), qvel=np.zeros((T, W, nv), np.float32),
               qacc_warmstart=np.zeros((T, W, nv), np.float32), ctrl=np.zeros((T, W, nu), np.float32))
    if dr is not None:
        rec.update(friction=np.zeros((T, W), np.float32), com=np.zeros((T, W, 3), np.float32))
    ep_step, key_at = np.zeros((T, W), np.int32), np.zeros((T, W), np.int32)
    fell_at, bad_at = np.zeros((T, W), bool), np.zeros((T, W), bool)
    roll_ncon, roll_nefc = np.zeros((T, W), np.int32), np.zeros((T, W), np.int32)
    reset(np.arange(W))
    for t in range(T):
        r = r_roll
        # setpoints: OU step, then jumps to a uniform pose
        x = ou_a * x + ou_b * r.standard_normal((W, nu))
        jump = (r.random(W) < p_jump)[:, None] & (r.random((W, nu)) < args.jump_share)
        pose = r.uniform(rob.a_lo, rob.a_hi, (W, nu)) if np.isfinite(rob.a_lo).all() else None
        if pose is not None:
            x = np.where(jump, pose - rob.key_set[key], x)
        u = np.clip(np.clip(rob.key_set[key] + x, rob.a_lo, rob.a_hi), rob.c_lo, rob.c_hi)
        # pushes
        kick_cd -= 1
        kick = kick_cd <= 0
        dv = r.uniform(-kick_vel, kick_vel, (W, 6))
        if kick.any():
            qvel_b[kick, rob.va:rob.va + 6] += dv[kick]
            kick_cd[kick] = r.integers(kmin, kmax + 1, size=int(kick.sum()))
        ctrl_b[:] = u
        b.qfrc_applied[:] = 0.0
        for k in ("qpos", "qvel", "qacc_warmstart", "ctrl"):
            rec[k][t] = b.bind(k)
        if dr is not None:
            rec["friction"][t], rec["com"][t] = fric, com
        ep_step[t], key_at[t] = ep, key
        warn0 = b.warning[:, BAD_WARNINGS, 1].copy()
        bad = np.zeros(W, bool)
        for _ in range(nsub):
            try:
                b.step()
            except mjbatch.SimulationError as e:
                bad[e.ids] = True
            roll_ncon[t] = np.maximum(roll_ncon[t], ncon_b)
            roll_nefc[t] = np.maximum(roll_nefc[t], nefc_b)
        bad |= (b.warning[:, BAD_WARNINGS, 1] > warn0).any(1)
        q, v = qpos_b, qvel_b
        finite = np.isfinite(q).all(1) & np.isfinite(v).all(1)
        bad |= ~finite
        with np.errstate(invalid="ignore"):
            fell = ((np.abs(q[:, rob.qa + 2] - rob.key_z[key]) > TERM_HEIGHT)
                    | (np.abs(gravity_z(q[:, rob.qa + 3:rob.qa + 7]) - rob.key_gz[key]) > TERM_TILT))
        fell |= bad
        fell_at[t], bad_at[t] = fell, bad
        ep += 1
        reset(np.flatnonzero(fell))
    t_roll = time.time() - t0
    nfall = int(fell_at.sum())
    print(f"rollout {W} worlds x {T} steps ({nsub} substeps, {args.iterations} iterations, {threads} threads): "
          f"{nfall} falls, mean episode length at a fall {ep_step[fell_at].mean() + 1:.1f} steps, "
          f"{int(bad_at.sum())} bad steps; contacts mean {roll_ncon.mean():.2f} max {roll_ncon.max()}, rows mean "
          f"{roll_nefc.mean():.1f} max {roll_nefc.max()} ({t_roll:.0f} s)", flush=True)

    # pre-fall: the --pre-fall states of an episode up to and including the step into its fall
    pre = np.zeros((T, W), bool)
    for t, i in zip(*np.nonzero(fell_at)):
        k = min(args.pre_fall, ep_step[t, i] + 1)
        pre[t - k + 1:t + 1, i] = True
    pool = ~bad_at
    taken = np.zeros((T, W), bool)
    S = args.states
    lab = Labeller(m, nsub, dr=dr, threads=threads)

    def states_at(flat):
        tt, ii = np.unravel_index(flat, (T, W))
        return {k: v[tt, ii] for k, v in rec.items()}

    def pick(mask, k):
        cand = np.flatnonzero((mask & pool & ~taken).ravel())
        sel = np.sort(r_pick.choice(cand, min(k, len(cand)), replace=False))
        taken.ravel()[sel] = True
        return sel

    sel_pre = pick(pre, S // 2)
    lab_pre = lab(states_at(sel_pre), args.iterations)
    need = int(np.ceil(args.self_share * S)) - int(lab_pre["self_contact"].sum())
    sel_self, labelled = [], 0
    if need > 0:
        cand = r_pick.permutation(np.flatnonzero((pool & ~taken).ravel()))
        for lo in range(0, len(cand), args.label_chunk):
            chunk = cand[lo:lo + args.label_chunk]
            lc = lab(states_at(chunk), args.iterations)
            hit = chunk[lc["self_contact"] & ~lc["bad"]]
            labelled += len(chunk)
            sel_self.append(hit[:need])
            need -= len(sel_self[-1])
            if need <= 0:
                break
    sel_self = np.sort(np.concatenate(sel_self)) if sel_self else np.zeros(0, int)
    taken.ravel()[sel_self] = True
    if need > 0:
        print(f"warning: only {len(sel_self)} more states with a self contact in the pool; the self-contact share "
              f"stays below {args.self_share}", flush=True)
    sel_uni = pick(np.ones((T, W), bool), S - len(sel_pre) - len(sel_self))
    sel = np.concatenate([sel_pre, sel_self, sel_uni])
    how = np.concatenate([np.full(len(sel_pre), 0), np.full(len(sel_self), 1), np.full(len(sel_uni), 2)])
    if len(sel) < S:
        raise SystemExit(f"the pool has only {len(sel)} usable states, {S} wanted")

    # the corpus: a fixed permutation, inputs as stored (float32), labels from those inputs
    perm = r_perm.permutation(S)
    sel, how = sel[perm], how[perm]
    tt, ii = np.unravel_index(sel, (T, W))
    arrays = {k: np.ascontiguousarray(v[tt, ii]) for k, v in rec.items()}
    L = lab(arrays, args.iterations)
    if L["bad"].any():
        raise SystemExit(f"{int(L['bad'].sum())} corpus states give a bad MuJoCo C step from their float32 inputs")
    arrays["cot_qpos"] = r_cot.standard_normal((S, args.cotangents, nq), dtype=np.float32)
    arrays["cot_qvel"] = r_cot.standard_normal((S, args.cotangents, nv), dtype=np.float32)
    over = (L["ncon_max"] > CAPACITY[0]) | (L["nefc_max"] > CAPACITY[1])
    labels = dict(
        prefall=(pre[tt, ii], f"within the last {args.pre_fall} control steps of an episode up to its fall (rollout)"),
        pick=(how.astype(np.int32), "how the state was sampled: 0 pre-fall, 1 self contact, 2 uniform"),
        rollout_step=(ep_step[tt, ii], "control steps since the episode's reset (rollout)"),
        rollout_index=((tt * W + ii).astype(np.int32), "step * worlds + world of the rollout"),
        keyframe=(key_at[tt, ii], "index into metadata.reset.keyframes of the episode's start pose"),
        ncon_max=(L["ncon_max"], "MuJoCo C: most contacts in any substep"),
        nefc_max=(L["nefc_max"], "MuJoCo C: most constraint rows in any substep"),
        niter_max=(L["niter_max"], "MuJoCo C: most solver iterations in any substep at the corpus iterations"),
        self_contact=(L["self_contact"], "MuJoCo C: a contact between two robot geoms in some substep"),
        clamped=(L["clamped"], "MuJoCo C: an actuated joint's force outside its actuator force range in some "
                               "substep"),
        limit=(L["limit"], "MuJoCo C: a limit row (joint or tendon) in some substep"),
        nonconv=(L["nonconv"], "MuJoCo C: solver iterations at the limit in some substep"),
        over_capacity=(over, f"MuJoCo C: more than {CAPACITY[0]} contacts or {CAPACITY[1]} rows in some substep"),
    )
    for k, (v, _) in labels.items():
        arrays[f"label_{k}"] = np.ascontiguousarray(v, bool if v.dtype == bool else np.int32)
    counts = dict(states=S, picks=dict(zip(PICKS, (len(sel_pre), len(sel_self), len(sel_uni)))),
                  pool=int(pool.sum()), rollout_pairs=T * W, falls=nfall, bad_steps=int(bad_at.sum()),
                  labelled_for_self_contact=labelled,
                  shares={k: float(arrays[f"label_{k}"].mean()) for k in
                          ("prefall", "self_contact", "clamped", "limit", "nonconv", "over_capacity")})
    # the options that define the corpus (not where it goes, how fast it is made, or where the model lives)
    opts = {k: v for k, v in vars(args).items()
            if k not in ("out", "force", "threads", "model", "meta", "label_chunk")}
    metadata = dict(
        format=corpus_format.FORMAT, name=args.name, track=args.track,
        generator=dict(script=GENERATOR, version=VERSION), seed=args.seed, options=opts, model=model_rec,
        physics=dict(timestep=m.opt.timestep, substeps=nsub, control_dt=dt, iterations=args.iterations,
                     ls_iterations=args.ls_iterations, tolerance=args.tolerance, solver=int(m.opt.solver),
                     cone=int(m.opt.cone), integrator=int(m.opt.integrator), sensors=False,
                     simulator="MuJoCo C float64, mujoco.batch.Batch"),
        dr=None if dr is None else dict(foot_pairs=dr[0].tolist(), torso_body=meta["torso_body"],
                                        friction=list(FRICTION), com_limit=COM_LIM.tolist()),
        reset=dict(keyframes=rob.key_names, pos=RESET_POS.tolist(), rpy=RESET_RPY.tolist(), vel=RESET_VEL.tolist(),
                   joint=RESET_JOINT, soft_range=SOFT),
        termination=dict(height=TERM_HEIGHT, tilt=TERM_TILT, reference="the episode's start keyframe"),
        cotangents=dict(n=args.cotangents, distribution="N(0, 1)", dtype="float32"),
        capacity=dict(contacts=CAPACITY[0], rows=CAPACITY[1], filtered=False),
        labels={f"label_{k}": d for k, (_, d) in labels.items()}, counts=counts,
        packages=dict(mujoco=mujoco.__version__, numpy=np.__version__))
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = corpus_format.write(out_path, arrays, metadata)
    digest = corpus_format.content_digest(out_path)
    size = out_path.stat().st_size
    summary = dict(name=args.name, sha256=sha, content_sha256=digest, bytes=size, seconds=round(time.time() - t0, 1),
                   **counts)
    print(json.dumps(summary), flush=True)
    write_manifest(out_dir, args, argv, sha, digest, size, metadata, threads, time.time() - t0)
    return 0


def write_manifest(out_dir, args, argv, sha, digest, size, metadata, threads, seconds):
    """Record the corpus in <out>/corpora.json (one entry per corpus name, others kept)."""
    path = out_dir / "corpora.json"
    man = json.loads(path.read_text()) if path.exists() else {"corpora": {}}
    prov = corpus_format.data_provenance()
    try:
        cpu = next(line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name"))
    except (OSError, StopIteration):
        cpu = None
    cmd = "python -m benchmarks.bench.corpus.gen_mf " + " ".join(sys.argv[1:] if argv is None else argv)
    man["corpora"][args.name] = dict(
        file=rel(out_dir / f"{args.name}.npz"), sha256=sha, content_sha256=digest, bytes=size, track=args.track,
        seed=args.seed, generator=metadata["generator"], model=metadata["model"], counts=metadata["counts"],
        command=cmd, threads=threads, seconds=round(seconds, 1), cpu=cpu, provenance=prov)
    man["corpora"] = dict(sorted(man["corpora"].items()))
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(man, indent=1) + "\n")
    os.replace(tmp, path)


if __name__ == "__main__":
    sys.exit(main())

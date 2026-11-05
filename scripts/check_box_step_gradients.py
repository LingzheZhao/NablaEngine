"""Short-window engine check on a generated static-box floor and the fixture plane.

Run only on provisioned CUDA compute:
 PYTHONPATH=src python scripts/check_box_step_gradients.py --code-revision REV --out /new/external/root

The fixture model is loaded fresh, never rewritten. The box condition replaces
its single world plane in memory with a wide box whose top is z=0. All original
floor pairs must have a sphere or capsule on the other side. This is a generated
flat-box diagnostic, not released InstinctMJ terrain. No unsupported-model override
is permitted. We generate standing states from qpos0, place both feet in declared
2 mm penetration, and require executed floor contacts. Freeze all model parameters.

One and four controls use four substeps each. Three terminal cotangents and four
generated directions per qpos-tangent/qvel/constant-action input block compare the
analytic VJP with centered engine finite differences at three declared steps.
Every trajectory keeps its own contacts, zones, solver diagnostics and time support.
Non-smooth or numerically unstable differences remain failed/unqualified conditions;
no directions disappear from the report. No training or skill claim follows.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = dict(seed=20261105, horizons=[1, 4], substeps=4, iterations=50,
                clamp_kv=1, normal=0, directions_per_block=4,
                fd_steps=[.002, .001, .0005], gradient_atol=.003, gradient_rtol=.03,
                fd_stability_atol=.002, fd_stability_rtol=.02,
                mujoco_qpos_atol=.0002, mujoco_qvel_atol=.02,
                floor_half_size=[10., 10., .1], floor_position=[0., 0., -.1],
                initial_foot_penetration=.002, initial_asymmetry=.002,
                friction=1., com=[0., 0., 0.], negative_control_offset=.1, negative_control_relative=.2)


def close(a, b, atol, rtol):
    return bool(np.isfinite(a).all() and np.isfinite(b).all()
                and np.all(np.abs(a-b) <= atol+rtol*np.abs(b)))


def generated_model(condition):
    import mujoco
    from nablaengine import fixtures
    from nablaengine.model import capability
    if os.environ.get("DK_ALLOW_UNSUPPORTED"):
        raise ValueError("DK_ALLOW_UNSUPPORTED must be unset")
    m, metadata = fixtures.load_model(check=None)
    floors = np.flatnonzero(m.geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE))
    if len(floors) != 1 or m.geom_bodyid[floors[0]] != 0:
        raise ValueError("expected exactly one world plane")
    floor = int(floors[0]); pairs = []
    for p, (a, b) in enumerate(zip(m.pair_geom1, m.pair_geom2)):
        if floor in (a, b):
            other = int(b if a == floor else a)
            if int(m.geom_type[other]) not in (int(mujoco.mjtGeom.mjGEOM_SPHERE),
                                              int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
                raise ValueError(f"floor pair {p} has unsupported other geom {other}")
            pairs.append(dict(pair=p, robot_geom=other, mujoco_type=int(m.geom_type[other])))
    if not pairs:
        raise ValueError("no floor pairs")
    if condition == "box":
        m.geom_type[floor] = int(mujoco.mjtGeom.mjGEOM_BOX)
        m.geom_size[floor] = PROTOCOL["floor_half_size"]
        m.geom_pos[floor] = PROTOCOL["floor_position"]
        m.geom_quat[floor] = [1, 0, 0, 0]
        m.geom_rbound[floor] = np.linalg.norm(m.geom_size[floor])
        m.geom_aabb[floor, :3] = 0
        m.geom_aabb[floor, 3:] = m.geom_size[floor]
        # Validate hand-updated bounds against MuJoCo's compiler, rather than
        # assuming the old plane's derived fields remain valid for a box.
        probe = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="box" size="10 10 .1" '
                                                'pos="0 0 -.1"/></worldbody></mujoco>')
        # The plane was at its body's origin. Its compiled sameframe flag
        # would otherwise make MuJoCo ignore the new -0.1 m geom translation.
        m.geom_sameframe[floor] = probe.geom_sameframe[0]
        if not (np.array_equal(m.geom_aabb[floor], probe.geom_aabb[0])
                and np.isclose(m.geom_rbound[floor], probe.geom_rbound[0], rtol=0, atol=1e-12)):
            raise ValueError("generated floor bounds disagree with compiled box probe")
        kinematics = mujoco.MjData(m)
        mujoco.mj_kinematics(m, kinematics)
        if not (np.array_equal(kinematics.geom_xpos[floor], probe.geom_pos[0])
                and np.array_equal(kinematics.geom_xmat[floor], np.eye(3).ravel())):
            raise ValueError("generated floor world pose disagrees with compiled box probe")
        # Explicit pair order is preserved by MuJoCo's compiler. The body's
        # signature remains valid because the geom stays on the world body.
        for p in pairs:
            if int(m.pair_signature[p["pair"]]) != int(m.geom_bodyid[p["robot_geom"]]):
                raise ValueError("generated floor pair body signature changed")
    m.opt.iterations = PROTOCOL["iterations"]
    fp = [int(p) for p in metadata["foot_pairs"]]
    if not set(fp) <= {p["pair"] for p in pairs}:
        raise ValueError("fixture foot_pairs not a subset of checked floor pairs")
    m.pair_friction[fp, :2] = PROTOCOL["friction"]
    report = capability.check(m, fp, scope="engine")
    # Check explicitly before require so an environment override cannot hide failure.
    if not report.ok():
        raise capability.UnsupportedModel(report, "generated floor diagnostic")
    capability.require(m, fp, capability.COM_BODY, "engine", "generated floor diagnostic")
    return m, fp, pairs, report.to_dict()


def rejection_checks():
    import mujoco
    from nablaengine.model import capability
    results = []
    for condition in ("dynamic_box", "plane_box", "box_box"):
        m, fp, pairs, _ = generated_model("plane" if condition == "plane_box" else "box")
        p = pairs[0]
        a, b = int(m.pair_geom1[p["pair"]]), int(m.pair_geom2[p["pair"]])
        floor = a if m.geom_bodyid[a] == 0 else b
        if condition == "dynamic_box":
            m.geom_bodyid[floor] = m.geom_bodyid[p["robot_geom"]]
        else:
            m.geom_type[p["robot_geom"]] = int(mujoco.mjtGeom.mjGEOM_BOX)
        report = capability.check(m, fp, scope="engine")
        rejected = "H14b" in report.codes("error")
        results.append(dict(condition=condition, rejected=rejected, errors=report.codes("error")))
        if not rejected:
            raise ValueError(f"unsupported {condition} was not rejected by the contact adjoint gate")
    return results


def states(m, fp):
    import mujoco
    result = []
    rng = np.random.default_rng(PROTOCOL["seed"])
    asymmetry = rng.normal(size=m.nv)*PROTOCOL["initial_asymmetry"]
    asymmetry[:6] = 0
    for name, delta in (("qpos0_contact", np.zeros(m.nv)), ("qpos0_asymmetric_contact", asymmetry)):
        q = m.qpos0.copy()
        mujoco.mj_integratePos(m, q, delta, 1.)
        d = mujoco.MjData(m); d.qpos[:] = q; mujoco.mj_kinematics(m, d)
        sides = {-1: [], 1: []}
        side_pairs = {-1: [], 1: []}
        for p in fp:
            a, b = int(m.pair_geom1[p]), int(m.pair_geom2[p])
            g = b if m.geom_bodyid[a] == 0 else a
            center = d.geom_xpos[g]; axis = d.geom_xmat[g].reshape(3, 3)[:, 2]
            bottom = center[2]-m.geom_size[g, 0]
            if m.geom_type[g] == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                bottom -= m.geom_size[g, 1]*abs(axis[2])
            side = 1 if center[1] >= 0 else -1
            sides[side].append(float(bottom))
            side_pairs[side].append(int(p))
        if not all(sides.values()):
            raise ValueError("foot pairs must cover both lateral sides")
        minima = [min(sides[k]) for k in (-1, 1)]
        shift = -max(minima)-PROTOCOL["initial_foot_penetration"]
        q[2] += shift
        u = np.array([q[int(m.jnt_qposadr[int(j)])] for j in m.actuator_trnid[:, 0]])
        result.append(dict(name=name, q=q, v=np.zeros(m.nv), u=u,
                           root_height_shift=float(shift), foot_minima_before_shift=minima,
                           foot_pairs_by_side={str(k): v for k, v in side_pairs.items()}))
    return result


def directions(m):
    rng = np.random.default_rng(PROTOCOL["seed"]+1)
    result = []
    for block, n in (("qpos_tangent", m.nv), ("qvel", m.nv), ("ctrl", m.nu)):
        for k in range(PROTOCOL["directions_per_block"]):
            d = rng.normal(size=n); d /= np.linalg.norm(d)
            result.append(dict(block=block, index=k, direction=d))
    return result


def cotangents(m):
    rng = np.random.default_rng(PROTOCOL["seed"]+2)
    q, v = rng.normal(size=m.nq), rng.normal(size=m.nv)
    q /= np.linalg.norm(q); v /= np.linalg.norm(v)
    return [("qpos", q, v*0), ("qvel", q*0, v), ("mixed", q/np.sqrt(2), v/np.sqrt(2))]


def perturb(m, state, direction, h):
    import mujoco
    q, v, u = (state[k].copy() for k in ("q", "v", "u"))
    block, d = direction["block"], direction["direction"]
    if block == "qpos_tangent":
        mujoco.mj_integratePos(m, q, d, h)
    elif block == "qvel":
        v += h*d
    else:
        u += h*d
    return q, v, u


def run_condition(condition, out, report, *, diagnostic=None, artifact=None, artifact_validator=None):
    import mujoco
    import torch
    from nablaengine import torch_ffi as F
    from nablaengine.g1_engine import tape
    report["rejection_checks"] = rejection_checks()
    m, fp, floor_pairs, capability = generated_model(condition)
    report.update(capability=capability, floor_pairs=floor_pairs,
                  dimensions=dict(nq=m.nq, nv=m.nv, nu=m.nu, npair=m.npair), timestep=float(m.opt.timestep))
    os.environ["DK_CACHE_DIR"] = str(out / "build")
    os.environ.pop("DK_TORCH_ARTIFACT", None)
    cc = torch.cuda.get_device_capability(); arch = f"sm_{cc[0]}{cc[1]}"
    report["environment"] = dict(python=platform.python_version(), numpy=np.__version__, mujoco=mujoco.__version__,
                                 torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name())
    print(f"building {condition} {arch}", flush=True)
    if artifact is None:
        artifact = F.build(arch, model=m, foot_pairs=fp, set_current=False)
    else:
        if artifact_validator is None:
            raise ValueError("reusing an artifact requires a source and model binding validator")
        report["artifact_binding"] = artifact_validator(m, fp, arch)
    F.init(artifact=artifact)
    report["artifact"] = str(artifact)
    def tensor(a):
        return torch.tensor(a, dtype=torch.float32, device="cuda").contiguous()
    def forward(q, v, u, horizon, record=True):
        q, v, u = tensor(q), tensor(v), tensor(u)
        B = len(q); warm = torch.zeros_like(v); friction = torch.full((B,), PROTOCOL["friction"], device="cuda")
        com = torch.zeros((B, 3), device="cuda"); saved = []; summaries = []; signatures = []; finite = True
        for _ in range(horizon):
            r = F.forward(q, v, warm, u, friction, com, iterations=PROTOCOL["iterations"],
                          substeps=PROTOCOL["substeps"], tape=record)
            q, v, warm, info = r[:4]
            finite &= bool(torch.isfinite(q).all() and torch.isfinite(v).all() and torch.isfinite(warm).all())
            if record:
                saved.append(r[4])
                raw = r[4].cpu().numpy().reshape(B, PROTOCOL["substeps"], tape.rec_words(m))
                d = tape.decode(raw, m)
                summaries.append({k: d[k] for k in ("ncon", "nefc", "flags", "pair", "slot")})
                active_zone = d["zone"] & (np.arange(d["zone"].shape[-1]) < d["nefc"][..., None])
                signatures.append(np.concatenate([d["ncon"][..., None], d["nefc"][..., None], d["nlim"][..., None],
                                                  d["pair"], d["slot"], d["clamp_dof"], active_zone], axis=-1))
                for key in ("qpos", "qvel", "qacc", "qacc_adv", "L_H", "L_Q"):
                    finite &= bool(np.isfinite(d[key]).all())
            if np.any(info.cpu().numpy()[:, 3] != 0):
                finite = False
        return dict(q=q, v=v, warm=warm, saved=saved, summary=summaries, signatures=signatures, finite=finite,
                    friction=friction, com=com)
    def strict_ok(trajectory):
        bad_mask = 31 | tape.FLAG_NONCONV | tape.FLAG_SINGULAR | tape.FLAG_VERIFY_DIFF
        return trajectory["finite"] and all(not np.any(s["flags"] & bad_mask) for s in trajectory["summary"])
    def physical_reference(state, horizon):
        d = mujoco.MjData(m); d.qpos[:] = state["q"]; d.qvel[:] = state["v"]; d.ctrl[:] = state["u"]
        d.qacc_warmstart[:] = 0
        support = []
        floor_geom_pairs = {frozenset((int(m.pair_geom1[p["pair"]]), int(m.pair_geom2[p["pair"]])))
                            for p in floor_pairs}
        foot_geom_pairs = {side: {frozenset((int(m.pair_geom1[p]), int(m.pair_geom2[p]))) for p in ps}
                           for side, ps in state["foot_pairs_by_side"].items()}
        for _ in range(horizon*PROTOCOL["substeps"]):
            mujoco.mj_step(m, d)
            active = [frozenset((int(c.geom1), int(c.geom2))) for c in d.contact]
            support.append(dict(ncon=int(d.ncon), floor_contacts=sum(p in floor_geom_pairs for p in active),
                                foot_contacts_by_side={side: sum(p in ps for p in active)
                                                       for side, ps in foot_geom_pairs.items()}))
        return d.qpos.copy(), d.qvel.copy(), support
    steps = PROTOCOL["fd_steps"] if diagnostic is None else diagnostic["fd_steps"]
    all_directions = directions(m)
    if diagnostic is not None:
        all_directions = [d for d in all_directions if d["block"] == diagnostic["block"]
                          and d["index"] == diagnostic["direction_index"]]
        report["diagnostic_protocol"] = diagnostic
    cots = cotangents(m)
    report["directions"] = [dict(d, direction=d["direction"].tolist()) for d in all_directions]
    report["cotangents"] = [dict(name=n, qpos=q.tolist(), qvel=v.tolist()) for n, q, v in cots]
    generated_states = states(m, fp)
    report["initial_states"] = [{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in s.items()}
                                 for s in generated_states]
    report["conditions"] = []
    for state in generated_states:
        for horizon in PROTOCOL["horizons"]:
            base = forward(state["q"][None], state["v"][None], state["u"][None], horizon)
            bare = forward(state["q"][None], state["v"][None], state["u"][None], horizon, False)
            identical = all(torch.equal(base[k], bare[k]) for k in ("q", "v", "warm"))
            qr, vr, mujoco_support = physical_reference(state, horizon)
            qn, vn = base["q"].cpu().numpy()[0], base["v"].cpu().numpy()[0]
            physical_ok = (close(qn, qr, PROTOCOL["mujoco_qpos_atol"], 0)
                           and close(vn, vr, PROTOCOL["mujoco_qvel_atol"], 0)
                           and all(n > 0 for support in mujoco_support for n in support["foot_contacts_by_side"].values()))
            contact_counts = [int(np.isin(s["pair"], [p["pair"] for p in floor_pairs]).sum()) for s in base["summary"]]
            foot_counts = {side: [int(np.isin(summary["pair"], ps).sum()) for summary in base["summary"]]
                           for side, ps in state["foot_pairs_by_side"].items()}
            contact_ok = all(n > 0 for counts in foot_counts.values() for n in counts)
            row = dict(state=state["name"], controls=horizon, seconds=horizon*4*float(m.opt.timestep),
                       tape_matches_no_tape=identical, strict_diagnostics_ok=bool(strict_ok(base)),
                       physical_reference_ok=physical_ok, mujoco_contact_support=mujoco_support, mujoco_qpos_max_error=float(abs(qn-qr).max()),
                       mujoco_qvel_max_error=float(abs(vn-vr).max()), floor_contacts_per_control=contact_counts,
                       foot_contacts_by_side=foot_counts, final_qpos=qn.tolist(), final_qvel=vn.tolist(), directions=[],
                       tape=[{k: v.tolist() for k, v in s.items()} for s in base["summary"]])
            # Finite difference batches keep all requested worlds, including failures.
            requests = [(di, h, sign) for di in range(len(all_directions)) for h in steps for sign in (-1, 1)]
            inputs = [perturb(m, state, all_directions[di], h*sign) for di, h, sign in requests]
            fdtraj = forward(*[np.stack([x[k] for x in inputs]) for k in range(3)], horizon)
            fd_outputs = np.c_[fdtraj["q"].cpu().numpy(), fdtraj["v"].cpu().numpy()].astype(float)
            evidence_path = out / f"{state['name']}_{horizon}_controls.npz"
            np.savez_compressed(evidence_path,
                                input_qpos=np.stack([x[0] for x in inputs]).astype(np.float32),
                                input_qvel=np.stack([x[1] for x in inputs]).astype(np.float32),
                                input_ctrl=np.stack([x[2] for x in inputs]).astype(np.float32),
                                fd_final_qvel=fdtraj["v"].cpu().numpy(), fd_final_qpos=fdtraj["q"].cpu().numpy(),
                                fd_tape=np.stack([t.cpu().numpy() for t in fdtraj["saved"]], axis=1),
                                base_tape=np.stack([t.cpu().numpy() for t in base["saved"]], axis=1))
            row["finite_difference_evidence"] = str(evidence_path)
            row["fd_trials"] = [dict(direction_index=di, step=h, sign=sign,
                                      ncon=[s["ncon"][j].tolist() for s in fdtraj["summary"]],
                                      nefc=[s["nefc"][j].tolist() for s in fdtraj["summary"]],
                                      flags=[s["flags"][j].tolist() for s in fdtraj["summary"]])
                                for j, (di, h, sign) in enumerate(requests)]
            physical_fd_outputs = None
            if diagnostic is not None:
                physical_fd_outputs = []
                physical_fd_support = []
                for qin, vin, uin in inputs:
                    perturbed_state = dict(state, q=qin, v=vin, u=uin)
                    qp, vp, support = physical_reference(perturbed_state, horizon)
                    physical_fd_outputs.append(np.r_[qp, vp])
                    physical_fd_support.append(support)
                physical_fd_outputs = np.stack(physical_fd_outputs)
                row["mujoco_fd_contact_support"] = physical_fd_support
                np.savez_compressed(out / f"{state['name']}_{horizon}_controls_mujoco_fd.npz",
                                    input_qpos=np.stack([x[0] for x in inputs]),
                                    input_qvel=np.stack([x[1] for x in inputs]),
                                    input_ctrl=np.stack([x[2] for x in inputs]), output=physical_fd_outputs)
            g_results = []
            for cotname, cq, cv in cots:
                gq, gv = tensor(cq[None]), tensor(cv[None]); gu = torch.zeros((1, m.nu), device="cuda"); backward_flags = []
                for saved in reversed(base["saved"]):
                    gq, gv, gstep, flags = F.backward(saved, base["friction"], base["com"], gq.contiguous(), gv.contiguous(),
                                                    substeps=4, clamp_kv=1, normal=0)
                    gu = gu+gstep; backward_flags.extend(flags.cpu().numpy().tolist())
                finite_g = all(bool(torch.isfinite(t).all()) for t in (gq, gv, gu))
                bad_bwd = 31 | tape.FLAG_NONCONV | tape.FLAG_SINGULAR | tape.FLAG_VERIFY_DIFF | (3 << 12) | (15 << 16)
                bwd_ok = finite_g and all(not (f & bad_bwd) for f in backward_flags)
                g_results.append((cotname, cq, cv, gq.cpu().numpy()[0], gv.cpu().numpy()[0], gu.cpu().numpy()[0], bwd_ok, backward_flags))
            for di, direction in enumerate(all_directions):
                ix = np.arange(di*2*len(steps), (di+1)*2*len(steps))
                topology_ok = all(np.array_equal(sig[ix], np.broadcast_to(basesig, sig[ix].shape))
                                  for sig, basesig in zip(fdtraj["signatures"], base["signatures"]))
                dr = dict(block=direction["block"], index=direction["index"], topology_stable=topology_ok, cotangents=[])
                for cotname, cq, cv, gq, gv, gu, bwd_ok, flags in g_results:
                    scalar = fd_outputs[ix]@np.r_[cq, cv]
                    fds = np.array([(scalar[j+1]-scalar[j])/(2*h) for j, h in zip(range(0, 2*len(steps), 2), steps)])
                    if direction["block"] == "qpos_tangent":
                        qp, _, _ = perturb(m, state, direction, 1e-6); qm, _, _ = perturb(m, state, direction, -1e-6)
                        analytic = float(gq@((qp-qm)/(2e-6)))
                    else:
                        analytic = float((gv if direction["block"] == "qvel" else gu)@direction["direction"])
                    stable = close(fds, fds[-1], PROTOCOL["fd_stability_atol"], PROTOCOL["fd_stability_rtol"])
                    correct = close(np.asarray(analytic), fds[-1], PROTOCOL["gradient_atol"], PROTOCOL["gradient_rtol"])
                    negative = not close(np.asarray(analytic+PROTOCOL["negative_control_offset"]+PROTOCOL["negative_control_relative"]*abs(analytic)), fds[-1],
                                         PROTOCOL["gradient_atol"], PROTOCOL["gradient_rtol"])
                    mujoco_fds = None
                    if physical_fd_outputs is not None:
                        physical_scalar = physical_fd_outputs[ix]@np.r_[cq, cv]
                        mujoco_fds = [(physical_scalar[j+1]-physical_scalar[j])/(2*h)
                                      for j, h in zip(range(0, 2*len(steps), 2), steps)]
                    dr["cotangents"].append(dict(name=cotname, analytic=analytic, finite_differences=fds.tolist(),
                                                   mujoco_double_finite_differences=mujoco_fds,
                                                   fd_stable=stable, gradient_ok=correct, backward_ok=bwd_ok,
                                                   backward_flags=flags, negative_control_rejected=negative,
                                                   pass_=bool(stable and correct and negative and bwd_ok and topology_ok)))
                row["directions"].append(dr)
            row["fd_strict_diagnostics_ok"] = bool(strict_ok(fdtraj))
            row["pass_"] = bool(identical and strict_ok(base) and strict_ok(fdtraj) and physical_ok and contact_ok
                                 and all(c["pass_"] for d in row["directions"] for c in d["cotangents"]))
            report["conditions"].append(row)
            if diagnostic is None:
                print(condition, state["name"], horizon, "PASS" if row["pass_"] else "FAIL", flush=True)
            else:
                print(condition, state["name"], horizon, "DIAGNOSTIC recorded; qualification unchanged", flush=True)
    report["diagnostics"] = F.read_diagnostics()
    report["pass_"] = all(c["pass_"] for c in report["conditions"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--code-revision", required=True)
    ap.add_argument("--condition", choices=("plane", "box", "both"), default="both")
    args = ap.parse_args(); out = args.out.resolve()
    if out.is_relative_to(ROOT):
        ap.error("--out must be outside checkout")
    out.mkdir(parents=True, exist_ok=False)
    report = dict(protocol=PROTOCOL, code_revision=args.code_revision, condition=args.condition, pass_=False)
    try:
        if args.condition == "both":
            report["children"] = []
            for condition in ("plane", "box"):
                command = [sys.executable, __file__, "--condition", condition, "--code-revision", args.code_revision,
                           "--out", str(out / condition)]
                with (out / f"{condition}.log").open("w") as stream:
                    proc = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
                report["children"].append(dict(condition=condition, returncode=proc.returncode,
                                                report=str(out / condition / "report.json")))
            report["pass_"] = all(c["returncode"] == 0 for c in report["children"])
        else:
            run_condition(args.condition, out, report)
        return 0 if report["pass_"] else 1
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (out / "report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    raise SystemExit(main())

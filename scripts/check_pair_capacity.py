"""GPU qualification of contact pair IDs above 511 and 1023 on generated G1 models.

Run on provisioned CUDA compute with --code-revision REV --out /new/external/root.
The default runs three independent processes because Torch FFI loads one model
per process. The original 502-pair fixture retains its full collision selection.
Two compiled MjSpec models share 181 geoms and the G1 dynamics. Both have 33 old
floor pairs with the plane lowered to z=-10, plus 14 capsule/box support pairs.
The 1063-pair model adds 1016 unique, distant box/left-foot capsule pairs. MuJoCo
compiles pair signature ordering. Real support pairs consequently exceed ID 1023.
The 47-pair control omits those distant candidates. Every geom pair is unique.

Checks execute forward, tape decoding and backward on the GPU at one and four
controls. MuJoCo double-precision directional finite differences and the physically
equivalent 47-pair control check the high-ID backward. Masking actual GPU contact
IDs to nine or ten bits provides a deliberate corruption control. Positive runs
require finite results, supported/converged records, active bilateral support,
and zero contact or row overflow. Timing and memory are measured without a speed
threshold. Contact-capacity exhaustion is outside this sparse-contact protocol.
"""
import argparse
import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import check_box_step_gradients as box_check
from capability_check import spec_from_model

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = dict(seed=20261105, conditions=["fixture502", "control47", "capacity1063"],
                horizons=[1, 4], substeps=4, iterations=50, max_contacts=96,
                padding_pairs=1016, support_insertion_after_padding_pairs=500, distant_boxes=146, distant_box_start=[100., 20., 0.],
                distant_box_stride=1., distant_box_half_size=[.1, .1, .1],
                active_box_half_size=[10., 10., .1], active_box_position=[0., 0., -.1],
                old_plane_position=[0., 0., -10.], initial_state="qpos0_asymmetric_contact",
                friction=1., com=[0., 0., 0.], fd_steps=[1e-5, 5e-6],
                gradient_atol=.003, gradient_rtol=.03, fd_stability_atol=.0005, fd_stability_rtol=.005,
                forward_qpos_atol=.0002, forward_qvel_atol=.02,
                invariant_qpos_atol=2e-6, invariant_qvel_atol=.0002,
                invariant_gradient_atol=.0003, invariant_gradient_rtol=.003,
                timing_worlds=256, timing_warmups=3, timing_repeats=10,
                timing_gate=None, contact_overflow_allowed=0, row_overflow_allowed=0)


def name(m, kind, index):
    import mujoco
    return mujoco.mj_id2name(m, kind, int(index))


def pair_parameters(pair, m, index):
    pair.condim = int(m.pair_dim[index])
    pair.friction = m.pair_friction[index]
    pair.solref = m.pair_solref[index]
    pair.solimp = m.pair_solimp[index]
    pair.margin = float(m.pair_margin[index])
    pair.gap = float(m.pair_gap[index])


def generated_model(condition):
    import mujoco
    from nablaengine import fixtures
    from nablaengine.model import capability
    if os.environ.get("DK_ALLOW_UNSUPPORTED"):
        raise ValueError("DK_ALLOW_UNSUPPORTED must be unset")
    m, metadata = fixtures.load_model(check=None)
    fp = [int(p) for p in metadata["foot_pairs"]]
    selection = dict(original_pairs=int(m.npair), original_foot_pairs=fp)
    if condition != "fixture502":
        floor_ids = np.flatnonzero(m.geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE))
        if len(floor_ids) != 1:
            raise ValueError("expected one fixture world plane")
        floor = int(floor_ids[0])
        floor_name = name(m, mujoco.mjtObj.mjOBJ_GEOM, floor)
        floor_pairs = [p for p in range(m.npair) if floor in (m.pair_geom1[p], m.pair_geom2[p])]
        if len(floor_pairs) != 33:
            raise ValueError("expected 33 fixture floor pairs")
        floor_pair_names = {name(m, mujoco.mjtObj.mjOBJ_PAIR, p) for p in floor_pairs}
        foot_geoms = []
        for p in fp:
            a, b = int(m.pair_geom1[p]), int(m.pair_geom2[p])
            g = b if a == floor else a
            if m.geom_type[g] != int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                raise ValueError("all selected foot geoms must be capsules")
            foot_geoms.append((g, p))
        low_body = min(int(m.geom_bodyid[g]) for g, _ in foot_geoms)
        left = [(g, p) for g, p in foot_geoms if int(m.geom_bodyid[g]) == low_body]
        if len(left) != 7 or len(foot_geoms) != 14:
            raise ValueError("expected two seven-capsule feet")
        spec = spec_from_model(m)
        for pair in list(spec.pairs):
            if pair.name not in floor_pair_names:
                spec.delete(pair)
        spec.geom(floor_name).pos = PROTOCOL["old_plane_position"]
        for j in range(PROTOCOL["distant_boxes"]):
            center = np.array(PROTOCOL["distant_box_start"], float)
            center[0] += j*PROTOCOL["distant_box_stride"]
            spec.worldbody.add_geom(name=f"capacity_distant_box_{j}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                    pos=center, size=PROTOCOL["distant_box_half_size"],
                                    contype=0, conaffinity=0, mass=0.)
        spec.worldbody.add_geom(name="capacity_support_box", type=mujoco.mjtGeom.mjGEOM_BOX,
                                pos=PROTOCOL["active_box_position"], size=PROTOCOL["active_box_half_size"],
                                contype=0, conaffinity=0, mass=0.)
        active_names = {}
        def add_support(index):
            g, template = foot_geoms[index]
            pair_name = f"capacity_support_pair_{index}"
            pair = spec.add_pair(name=pair_name, geomname1="capacity_support_box",
                                 geomname2=name(m, mujoco.mjtObj.mjOBJ_GEOM, g))
            pair_parameters(pair, m, template)
            active_names[index] = pair_name
        if condition == "capacity1063":
            for k in range(PROTOCOL["padding_pairs"]):
                g, template = left[k % 7]
                pair = spec.add_pair(name=f"capacity_distant_pair_{k}",
                                     geomname1=f"capacity_distant_box_{k // 7}",
                                     geomname2=name(m, mujoco.mjtObj.mjOBJ_GEOM, g))
                pair_parameters(pair, m, template)
                if k+1 == PROTOCOL["support_insertion_after_padding_pairs"]:
                    for index, (g, _) in enumerate(foot_geoms):
                        if int(m.geom_bodyid[g]) == low_body:
                            add_support(index)
        for index in range(len(foot_geoms)):
            if index not in active_names:
                add_support(index)
        active_names = [active_names[k] for k in range(len(foot_geoms))]
        compiled = spec.compile()
        for field in ("body_mass", "body_inertia", "body_pos", "body_quat", "body_ipos", "body_iquat",
                      "qpos0", "dof_armature", "actuator_gainprm", "actuator_biasprm"):
            if not np.allclose(getattr(m, field), getattr(compiled, field), rtol=0, atol=1e-12):
                raise ValueError(f"reconstructed G1 physical field differs: {field}")
        m = compiled
        fp = [int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_PAIR, p)) for p in active_names]
        selection.update(retained_fixture_pairs=33, removed_fixture_self_pairs=469,
                         distant_pairs=1016 if condition == "capacity1063" else 0,
                         active_support_pairs=fp, selected_pair_names=active_names)
    expected = dict(fixture502=502, control47=47, capacity1063=1063)[condition]
    if m.npair != expected or m.ngeom > 255:
        raise ValueError(f"model has unexpected pair/geom counts {m.npair}/{m.ngeom}")
    unique = {frozenset((int(a), int(b))) for a, b in zip(m.pair_geom1, m.pair_geom2)}
    if len(unique) != m.npair:
        raise ValueError("duplicate geom pairs in generated input")
    if np.any(np.diff(m.pair_signature.astype(np.int64)) < 0):
        raise ValueError("compiled pair signatures are unsorted")
    if condition == "capacity1063" and not (any(511 < p <= 1023 for p in fp) and any(p > 1023 for p in fp)):
        raise ValueError("generated support must exercise both the ninth and tenth pair ID bits")
    m.opt.iterations = PROTOCOL["iterations"]
    m.pair_friction[fp, :2] = PROTOCOL["friction"]
    report = capability.check(m, fp, scope="engine")
    if not report.ok():
        raise capability.UnsupportedModel(report, "pair capacity input")
    capability.require(m, fp, capability.COM_BODY, "engine", "pair capacity input")
    from nablaengine.g1_engine import step
    selection.update(npair=int(m.npair), ngeom=int(m.ngeom), collision_geoms=len(step.collision_geoms(m)),
                     unique_geom_pairs=len(unique),
                     foot_pairs=fp, pair_signature_sorted=True)
    return m, fp, selection, report.to_dict()


def backward_chain(F, forward, cq, cv, substeps):
    import torch
    gq, gv = cq.contiguous(), cv.contiguous()
    gu = torch.zeros_like(forward["ctrl"])
    flags = []
    for saved in reversed(forward["tapes"]):
        gq, gv, step_gradient, info = F.backward(saved, forward["friction"], forward["com"],
                                                gq.contiguous(), gv.contiguous(), substeps=substeps,
                                                clamp_kv=1, normal=0)
        gu = gu + step_gradient
        flags.extend(info.cpu().numpy().tolist())
    return gq, gv, gu, flags


def run_condition(condition, out, report):
    import mujoco
    import torch
    from nablaengine import torch_ffi as F
    from nablaengine.g1_engine import tape
    os.environ["DK_MAXCON"] = str(PROTOCOL["max_contacts"])
    os.environ["DK_CACHE_DIR"] = str(out / "build")
    os.environ.pop("DK_TORCH_ARTIFACT", None)
    m, fp, selection, capability = generated_model(condition)
    report.update(selection=selection, capability=capability,
                  environment=dict(python=sys.version, numpy=np.__version__, mujoco=mujoco.__version__,
                                   torch=torch.__version__, cuda=torch.version.cuda,
                                   gpu=torch.cuda.get_device_name()),
                  tape_encoding=dict(pair_bits=tape.PAIR_BITS, format_marker=tape.FORMAT_MARKER,
                                     record_words=tape.rec_words(m)))
    cc = torch.cuda.get_device_capability()
    arch = f"sm_{cc[0]}{cc[1]}"
    # Rejection uses a generated obsolete metadata input, before any library loads.
    obsolete = out / "obsolete_artifact.json"
    obsolete.write_text(json.dumps(dict(artifact_version=F.ARTIFACT_VERSION-1)))
    try:
        F.init(artifact=obsolete)
    except RuntimeError as error:
        report["obsolete_artifact_rejection"] = str(error)
    else:
        raise AssertionError("obsolete artifact accepted")
    artifact = F.build(arch, model=m, foot_pairs=fp, set_current=False)
    info = F.init(artifact=artifact)
    report["artifact"] = str(artifact)
    library = ctypes.CDLL(info["path"])
    sizes = np.zeros(9, np.int64)
    sizes_fn = getattr(library, "dk_kv1_n0_sizes")
    sizes_fn.argtypes = [ctypes.c_void_p]
    sizes_fn(sizes.ctypes.data)
    report["compiled_memory"] = dict(sim_bytes=int(sizes[0]), adjoint_bytes=int(sizes[1]),
                                     tape_record_bytes=int(sizes[2])*4)
    state = next(s for s in box_check.states(m, fp) if s["name"] == PROTOCOL["initial_state"])
    report["initial_state"] = {key: value.tolist() if isinstance(value, np.ndarray) else value
                               for key, value in state.items()}
    def tensor(value):
        return torch.tensor(value, dtype=torch.float32, device="cuda").contiguous()
    def forward(q, v, u, horizon, record=True):
        q, v, u = tensor(q), tensor(v), tensor(u)
        warm = torch.zeros_like(v)
        friction = torch.full((len(q),), PROTOCOL["friction"], dtype=torch.float32, device="cuda")
        com = torch.zeros((len(q), 3), dtype=torch.float32, device="cuda")
        saved, decoded, infos = [], [], []
        finite = True
        for _ in range(horizon):
            result = F.forward(q, v, warm, u, friction, com, iterations=PROTOCOL["iterations"],
                               substeps=PROTOCOL["substeps"], tape=record)
            q, v, warm = result[:3]
            finite &= all(bool(torch.isfinite(x).all()) for x in (q, v, warm))
            infos.append(result[3].cpu().numpy())
            if record:
                saved.append(result[4])
                raw = result[4].cpu().numpy().reshape(len(q), PROTOCOL["substeps"], tape.rec_words(m))
                decoded.append(tape.decode(raw, m))
        return dict(q=q, v=v, warm=warm, ctrl=u, friction=friction, com=com, tapes=saved,
                    decoded=decoded, infos=infos, finite=finite)
    def physical(q, v, u, horizon):
        data = mujoco.MjData(m)
        data.qpos[:] = q
        data.qvel[:] = v
        data.ctrl[:] = u
        data.qacc_warmstart[:] = 0
        for _ in range(horizon*PROTOCOL["substeps"]):
            mujoco.mj_step(m, data)
        return np.r_[data.qpos.copy(), data.qvel.copy()]
    rng = np.random.default_rng(PROTOCOL["seed"]+3)
    test_directions = []
    for block, dimension in (("qpos_tangent", m.nv), ("qvel", m.nv), ("ctrl", m.nu)):
        vector = rng.normal(size=dimension)
        vector /= np.linalg.norm(vector)
        test_directions.append(dict(block=block, index=0, direction=vector))
    cotangents = box_check.cotangents(m)
    report["directions"] = [dict(d, direction=d["direction"].tolist()) for d in test_directions]
    report["conditions"] = []
    F.read_diagnostics(reset=True)
    for horizon in PROTOCOL["horizons"]:
        nominal = forward(state["q"][None], state["v"][None], state["u"][None], horizon)
        plain = forward(state["q"][None], state["v"][None], state["u"][None], horizon, False)
        qv = np.r_[nominal["q"].cpu().numpy()[0], nominal["v"].cpu().numpy()[0]]
        reference = physical(state["q"], state["v"], state["u"], horizon)
        active_pairs = sorted({int(p) for record in nominal["decoded"] for p in record["pair"].ravel() if p >= 0})
        high_ok = condition != "capacity1063" or (any(511 < p <= 1023 for p in active_pairs)
                                                   and any(p > 1023 for p in active_pairs))
        support_counts = {side: [np.isin(record["pair"], pairs).sum(axis=-1).tolist()
                                 for record in nominal["decoded"]]
                          for side, pairs in state["foot_pairs_by_side"].items()}
        feet_ok = all(np.all(np.asarray(counts) > 0) for counts in support_counts.values())
        bad_mask = 31 | tape.FLAG_NONCONV | tape.FLAG_SINGULAR | tape.FLAG_VERIFY_DIFF
        strict = nominal["finite"] and all(not np.any(d["flags"] & bad_mask) for d in nominal["decoded"])
        strict &= all(not np.any(i[:, 3]) for i in nominal["infos"])
        identical = all(torch.equal(nominal[k], plain[k]) for k in ("q", "v", "warm"))
        forward_ok = (box_check.close(qv[:m.nq], reference[:m.nq], PROTOCOL["forward_qpos_atol"], 0)
                      and box_check.close(qv[m.nq:], reference[m.nq:], PROTOCOL["forward_qvel_atol"], 0))
        row = dict(controls=horizon, active_pair_ids=active_pairs, high_pair_active=high_ok,
                   supported_finite_converged=bool(strict), tape_matches_plain=identical,
                   physical_reference_ok=forward_ok, support_active=bool(feet_ok),
                   foot_contacts_by_side=support_counts, readings=[],
                   qpos=qv[:m.nq].tolist(), qvel=qv[m.nq:].tolist(),
                   qpos_reference_max_error=float(np.max(abs(qv[:m.nq]-reference[:m.nq]))),
                   qvel_reference_max_error=float(np.max(abs(qv[m.nq:]-reference[m.nq:]))),
                   tape=[{key: d[key].tolist() for key in ("ncon", "nefc", "flags", "pair", "slot")}
                         for d in nominal["decoded"]])
        gradients = []
        for cot_name, cq, cv in cotangents:
            gq, gv, gu, flags = backward_chain(F, nominal, tensor(cq[None]), tensor(cv[None]), PROTOCOL["substeps"])
            gradient = np.r_[gq.cpu().numpy()[0], gv.cpu().numpy()[0], gu.cpu().numpy()[0]]
            gradients.append(gradient)
            backward_ok = np.isfinite(gradient).all() and all(not (f & (bad_mask | (3 << 12) | (15 << 16))) for f in flags)
            for direction in test_directions:
                fds = []
                for step in PROTOCOL["fd_steps"]:
                    values = []
                    for sign in (-1, 1):
                        q, v, u = box_check.perturb(m, state, direction, sign*step)
                        values.append(physical(q, v, u, horizon) @ np.r_[cq, cv])
                    fds.append((values[1]-values[0])/(2*step))
                if direction["block"] == "qpos_tangent":
                    qp, _, _ = box_check.perturb(m, state, direction, 1e-6)
                    qm, _, _ = box_check.perturb(m, state, direction, -1e-6)
                    analytic = float(gq.cpu().numpy()[0] @ ((qp-qm)/(2e-6)))
                else:
                    block_gradient = gv if direction["block"] == "qvel" else gu
                    analytic = float(block_gradient.cpu().numpy()[0] @ direction["direction"])
                stable = box_check.close(np.array(fds), fds[-1], PROTOCOL["fd_stability_atol"], PROTOCOL["fd_stability_rtol"])
                accurate = box_check.close(np.array(analytic), fds[-1], PROTOCOL["gradient_atol"], PROTOCOL["gradient_rtol"])
                row["readings"].append(dict(cotangent=cot_name, block=direction["block"], analytic=analytic,
                                             mujoco_double_fd=fds, fd_stable=stable, gradient_ok=accurate,
                                             backward_flags=flags, backward_ok=bool(backward_ok),
                                             pass_=bool(stable and accurate and backward_ok)))
        row["pass_"] = bool(strict and identical and high_ok and feet_ok and forward_ok
                             and all(r["pass_"] for r in row["readings"]))
        np.savez_compressed(out / f"{horizon}_controls.npz", qpos=qv[:m.nq], qvel=qv[m.nq:],
                            gradients=np.stack(gradients), tapes=np.stack([t.cpu().numpy() for t in nominal["tapes"]]))
        report["conditions"].append(row)
        print(condition, horizon, "PASS" if row["pass_"] else "FAIL", flush=True)
        if condition == "capacity1063":
            row["corruption_controls"] = []
            for mask in (511, 1023):
                corrupted = []
                changed = 0
                for saved in nominal["tapes"]:
                    raw = saved.cpu().numpy().copy().reshape(1, PROTOCOL["substeps"], tape.rec_words(m))
                    start, length = tape.layout(m)["con"]
                    contacts = np.ascontiguousarray(raw[..., start:start+length]).view(np.uint16)
                    active = contacts != tape.CON_NONE
                    original = contacts.copy()
                    contacts[active] = ((contacts[active] & tape.PAIR_MASK & mask)
                                        | ((contacts[active] >> tape.PAIR_BITS) << tape.PAIR_BITS))
                    changed += int(np.count_nonzero(contacts != original))
                    raw[..., start:start+length] = contacts.view(np.float32)
                    corrupted.append(tensor(raw.reshape(saved.shape)))
                corrupt_run = dict(nominal, tapes=corrupted)
                cq, cv = cotangents[-1][1:]
                gq, gv, gu, flags = backward_chain(F, corrupt_run, tensor(cq[None]), tensor(cv[None]), PROTOCOL["substeps"])
                altered = np.r_[gq.cpu().numpy()[0], gv.cpu().numpy()[0], gu.cpu().numpy()[0]]
                detected = changed > 0 and (any(f != 0 for f in flags)
                                             or not box_check.close(altered, gradients[-1], .0003, .003))
                row["corruption_controls"].append(dict(mask=mask, changed_contact_entries=changed,
                                                         backward_flags=flags, rejected_or_gradient_changed=bool(detected)))
                row["pass_"] &= bool(detected)
    report["positive_and_corruption_diagnostics"] = F.read_diagnostics(reset=True)
    # Timing uses supported nominal states and excludes host transfers from measured intervals.
    count = PROTOCOL["timing_worlds"]
    q = tensor(np.repeat(state["q"][None], count, axis=0))
    v = tensor(np.repeat(state["v"][None], count, axis=0))
    u = tensor(np.repeat(state["u"][None], count, axis=0))
    warm = torch.zeros_like(v)
    friction = torch.ones(count, device="cuda")
    com = torch.zeros((count, 3), device="cuda")
    cq, cv = tensor(np.repeat(cotangents[-1][1][None], count, axis=0)), tensor(np.repeat(cotangents[-1][2][None], count, axis=0))
    torch.cuda.reset_peak_memory_stats()
    times = []
    for iteration in range(PROTOCOL["timing_warmups"] + PROTOCOL["timing_repeats"]):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        result = F.forward(q, v, warm, u, friction, com, iterations=50, substeps=4, tape=True)
        F.backward(result[4], friction, com, cq, cv, substeps=4, clamp_kv=1, normal=0)
        end.record()
        end.synchronize()
        if iteration >= PROTOCOL["timing_warmups"]:
            times.append(begin.elapsed_time(end))
    diagnostics = F.read_diagnostics(reset=True)
    report["timing"] = dict(worlds=count, forward_plus_backward_ms=times, median_ms=float(np.median(times)),
                             peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                             peak_reserved_bytes=torch.cuda.max_memory_reserved(), diagnostics=diagnostics)
    zero_overflow = diagnostics["fwd_contact_overflow_worlds"] == 0 and diagnostics["fwd_row_overflow_worlds"] == 0
    timing_strict = all(diagnostics[key] == 0 for key in
                       ("fwd_unsupported_worlds", "nonfinite_rows", "replay_mismatch_worlds", "nonconv_records"))
    report["pass_"] = all(c["pass_"] for c in report["conditions"]) and zero_overflow and timing_strict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--condition", choices=(*PROTOCOL["conditions"], "all"), default="all")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.is_relative_to(ROOT):
        parser.error("--out must be outside the checkout")
    out.mkdir(parents=True, exist_ok=False)
    report = dict(protocol=PROTOCOL, code_revision=args.code_revision, condition=args.condition, pass_=False)
    try:
        if args.condition == "all":
            report["children"] = []
            for condition in PROTOCOL["conditions"]:
                command = [sys.executable, __file__, "--condition", condition, "--code-revision", args.code_revision,
                           "--out", str(out / condition)]
                with (out / f"{condition}.log").open("w") as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                report["children"].append(dict(condition=condition, returncode=result.returncode))
            comparisons = []
            for horizon in PROTOCOL["horizons"]:
                paths = [out / condition / f"{horizon}_controls.npz" for condition in ("control47", "capacity1063")]
                if not all(path.exists() for path in paths):
                    comparisons.append(dict(controls=horizon, pass_=False, error="child evidence unavailable"))
                    continue
                control, large = (np.load(path) for path in paths)
                checks = dict(qpos=box_check.close(large["qpos"], control["qpos"], PROTOCOL["invariant_qpos_atol"], 0),
                              qvel=box_check.close(large["qvel"], control["qvel"], PROTOCOL["invariant_qvel_atol"], 0),
                              gradients=box_check.close(large["gradients"], control["gradients"],
                                                        PROTOCOL["invariant_gradient_atol"], PROTOCOL["invariant_gradient_rtol"]))
                comparisons.append(dict(controls=horizon, checks=checks, pass_=all(checks.values())))
            report["distant_pair_invariance"] = comparisons
            report["pass_"] = all(c["returncode"] == 0 for c in report["children"]) and all(c["pass_"] for c in comparisons)
        else:
            run_condition(args.condition, out, report)
        return 0 if report["pass_"] else 1
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (out / "report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    raise SystemExit(main())

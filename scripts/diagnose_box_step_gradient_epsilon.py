"""Diagnose the existing qpos direction-2 finite differences; never changes qualification.

Run on provisioned CUDA compute, separately for --condition plane and box.
The predeclared sweep uses h=1e-3,5e-4,2.5e-4,1.25e-4,6.25e-5 at both
qpos0-derived starts and one/four controls. Engine float32 FD, analytic VJP,
and MuJoCo float64 FD are reported on the same direction and output cotangents.
MuJoCo evaluates the intended double-precision perturbations; engine inputs are
rounded to float32, and both input representations are retained beside output.
This helps distinguish truncation trends and numerical discrepancies without making
a smaller-epsilon reading replace the original qualification's failed conditions.
The comparison does not assume identical derivatives of the two implementations.

Without --artifact, build a fresh library. Reuse requires --artifact-source-root
(the retained repository snapshot used to build it) and --artifact-code-revision.
Package sources are compared directly, byte for byte, without computing hashes.
Model constants and exposed compilation settings must match the generated model.
A mismatch is an error; request a fresh build rather than accepting stale artifacts.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import check_box_step_gradients as checker

DIAGNOSTIC = dict(purpose="epsilon_diagnostic_not_original_qualification", block="qpos_tangent", direction_index=2,
                  fd_steps=[.001, .0005, .00025, .000125, .0000625],
                  reference="MuJoCo float64 on intended unrounded perturbations", qualification_pass=None)


def bind_artifact(artifact, source_root, declared_revision, model, foot_pairs, arch):
    import nablaengine
    from nablaengine import jax_ffi, torch_ffi
    current = Path(nablaengine.__file__).resolve().parent
    preserved = source_root / "src" / "nablaengine"
    checked = []
    for pattern in torch_ffi.SOURCES:
        actual = {p.relative_to(current) for p in current.glob(pattern)}
        original = {p.relative_to(preserved) for p in preserved.glob(pattern)}
        if not actual or actual != original:
            raise ValueError(f"artifact source inventory mismatch for {pattern}; fresh build required")
        for relative in sorted(actual):
            if (current / relative).read_bytes() != (preserved / relative).read_bytes():
                raise ValueError(f"artifact source differs: {relative}; fresh build required")
            checked.append(str(relative))
    metadata = json.loads(artifact.read_text())
    plan = jax_ffi.build_plan(model, foot_pairs, arch)
    expected = np.ascontiguousarray(plan["consts"], np.float32)
    recorded = np.load(artifact.parent / metadata["consts"], allow_pickle=False)
    if not np.array_equal(recorded, expected):
        raise ValueError("artifact model constants differ from generated diagnostic model; fresh build required")
    dimensions = dict(nq=model.nq, nv=model.nv, nu=model.nu, nbody=model.nbody)
    facts = dict(dimensions, arch=arch, foot_pairs=foot_pairs,
                 arch_flags=plan["arch_flags"], forward_defines=plan["forward_defines"])
    if any(metadata.get(k) != value for k, value in facts.items()):
        raise ValueError("artifact dimensions, physical bindings or compilation settings differ; fresh build required")
    return dict(artifact=str(artifact), artifact_code_revision=declared_revision,
                artifact_source_root=str(source_root), source_files_compared=checked,
                source_bytes_equal=True, model_constants_equal=True, model_facts_equal=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--code-revision", required=True)
    ap.add_argument("--condition", required=True, choices=("plane", "box"))
    ap.add_argument("--artifact", type=Path)
    ap.add_argument("--artifact-source-root", type=Path)
    ap.add_argument("--artifact-code-revision")
    args = ap.parse_args()
    if args.artifact is None and (args.artifact_source_root is not None or args.artifact_code_revision is not None):
        ap.error("artifact binding arguments require --artifact")
    if args.artifact is not None and (args.artifact_source_root is None or not args.artifact_code_revision):
        ap.error("--artifact requires --artifact-source-root and --artifact-code-revision")
    out = args.out.resolve()
    if out.is_relative_to(checker.ROOT):
        ap.error("--out must be outside checkout")
    out.mkdir(parents=True, exist_ok=False)
    report = dict(protocol=checker.PROTOCOL, diagnostic_protocol=DIAGNOSTIC,
                  code_revision=args.code_revision, condition=args.condition, qualification_pass=None,
                  diagnostic_completed=False)
    try:
        validator = None
        artifact = args.artifact.resolve() if args.artifact else None
        if artifact is not None:
            def validator(model, foot_pairs, arch):
                return bind_artifact(artifact, args.artifact_source_root.resolve(), args.artifact_code_revision,
                                     model, foot_pairs, arch)
        checker.run_condition(args.condition, out, report, diagnostic=DIAGNOSTIC,
                              artifact=artifact, artifact_validator=validator)
        # Keep every numerical failure, but do not give this diagnostic a
        # qualification pass label or replace the original protocol's result.
        report["all_original_threshold_checks_satisfied"] = report.pop("pass_")
        for condition in report["conditions"]:
            condition["original_threshold_checks_satisfied"] = condition.pop("pass_")
            for direction in condition["directions"]:
                for cot in direction["cotangents"]:
                    cot["original_threshold_checks_satisfied"] = cot.pop("pass_")
        report["diagnostic_completed"] = True
        return 0
    except Exception as exc:
        report.pop("pass_", None)
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (out / "report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    raise SystemExit(main())

# Changelog

## Unreleased

- Contact capacity raised from 64 to 96 and constraint-row capacity from 288 to 416 per world. Contact masks use
  32-bit tiles; the Hessian retains the original order (all world contacts, then all self contacts). DOF masks,
  collision geometry, solver equations and overflow flags are unchanged. The 7/9/8/8-bit tape header is unchanged;
  the mjlab G1 record grows to 1504 float32 words (6016 bytes), including 13 zone words. Libraries must be rebuilt;
  tapes from the previous capacity are not accepted by the current layout. The extension requires matched
  forward, adjoint and overflow validation before accepting a new experiment.

- Constraint-row capacity raised from 192 to 288 rows per world, so 64 pyramidal contacts (four rows each) and the
  29 hinge limits of the G1 fit without dropping rows; the tape header packs the contact count in 7 bits and the row
  count in 9 bits, the zone field grows from 6 to 9 words, and the readers in the JAX and PyTorch interfaces, the
  adjoint and the benchmark adapter decode the new layout. Libraries built before this change must be rebuilt.

## 0.1.0

First release.

- The fused CUDA engine for the Unitree G1 (mjlab model): one control step for thousands of worlds per call, with
  MuJoCo's physics and an analytic adjoint of the step.
- JAX interface (`nablaengine.jax_ffi`, `docs/api.md`, extra `nablaengine[jax]`) and PyTorch interface
  (`nablaengine.torch_ffi`, `docs/api_torch.md`, extra `nablaengine[torch]`), with bitwise the same results; the
  PyTorch one needs no JAX and runs a library built once per GPU architecture.
- Diagnostics on the device, including worlds beyond the contact (64) or constraint-row (192) capacity, whose extra
  contacts / rows are dropped: counted and flagged, with optional warnings (`set_overflow_warnings`).
- Verification against MuJoCo Warp, float64 MuJoCo and MJX (`scripts/verify.sh`, `docs/verification.md`) and the
  benchmark harness (`benchmarks/`).

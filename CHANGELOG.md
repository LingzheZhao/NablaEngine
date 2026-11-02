# Changelog

## Unreleased

- Constraint-row capacity raised from 192 to 288 rows per world, so 64 pyramidal contacts (four rows each) and the
  29 hinge limits of the G1 fit without dropping rows; the tape header packs the contact count in 7 bits and the row
  count in 9 bits, the zone field grows from 6 to 9 words, and the readers in the JAX and PyTorch interfaces, the
  adjoint and the benchmark adapter decode the new layout. Libraries built before this change must be rebuilt.
  The record length of the mjlab model stays 1472 words. The PyTorch artifact metadata records the capacity as
  `nefc`, and the matched MuJoCo Warp benchmark capacity follows the engine (`njmax` 288). 288 is not among
  MuJoCo Warp's listed sizes (192, 256, 384), so whether MuJoCo Warp accepts, rounds or rejects it must be checked in
  an environment with `mujoco_warp` before a matched benchmark is reported.

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

# Per-engine environments

Every engine's benchmark worker runs in the Python environment that engine needs; `envs.toml` maps engines to
environments and environments to interpreters. The driver (`python -m benchmarks.bench run`) runs in the `dk`
environment.

| Environment | Contents | Engines |
|---|---|---|
| `dk` | the NablaEngine environment of `scripts/setup_env.sh`: nablaengine, mujoco 3.17.0, mujoco-mjx 3.17.0, JAX, warp-lang 1.18.0 | `dk`, `dk_iface`, `dk_jac`, `mjx`, `mjc` |
| `mjw-native` | `requirements-mjw-native.txt`: PyPI mujoco-warp 3.17.0, mujoco 3.17.0, warp-lang 1.18.0 | `mjw` |
| `mjw-pr` | `requirements-mjw-pr.txt`: mujoco 3.17.0, warp-lang 1.18.0, no installed `mujoco_warp`; the source tree is selected per engine through `PYTHONPATH` | `mjw_pr1535`, `mjw_pr1535_fix` |
| `genesis` | PyTorch 2.14.1 from PyTorch's index (cu126 on CUDA 12 drivers and Volta, else cu128), then `requirements-genesis.txt`: genesis-world 1.4.5, quadrants 1.3.3, mujoco 3.17.0; not in `setup.sh`'s default set (`--envs genesis`) | `genesis` |
| `self` | the driver's own interpreter | `null` (harness self-test) |

The `mjw-pr` source trees, under `<prefix>/mjw-pr/src/`, both unmodified:

| Tree | What | Engine |
|---|---|---|
| `pr1535` | [etaoxing/mujoco_warp](https://github.com/etaoxing/mujoco_warp) at 357a75d, as published: the head of the open upstream pull request [google-deepmind/mujoco_warp#1535](https://github.com/google-deepmind/mujoco_warp/pull/1535) ("hybrid analytic differentiability") | `mjw_pr1535` |
| `pr1535_fix` | 357a75d + [etaoxing/mujoco_warp#1](https://github.com/etaoxing/mujoco_warp/pull/1) (4af093f), as published | `mjw_pr1535_fix` |

## Setup

```
benchmarks/envs/setup.sh [--prefix DIR] [--envs dk,mjw-native,mjw-pr,genesis] [--python PY] [--warp auto|default|cu12] [--dry-run]
```

- `--prefix` defaults to `$DK_BENCH_ENV_PREFIX`, else `runs/bench/envs` (ignored by git). Use the same value for
  `DK_BENCH_ENV_PREFIX` when running the driver.
- `dk`: an existing NablaEngine environment (`$DK_ENV`, set by its `activate.sh`) is used as it is; otherwise
  `scripts/setup_env.sh --env <prefix>/dk` creates one.
- `mjw-pr`: one clone of etaoxing/mujoco_warp, checked out at 357a75d, and a git worktree of it at 4af093f
  (fetched as `pull/1/head`).
- warp-lang: `auto` picks the default 1.18.0 wheel (CUDA 13) when the driver supports CUDA 13 (nvidia-smi's "CUDA
  Version" >= 13.0, drivers 580 and later), else NVIDIA's `1.18.0+cu12` wheel of the same release (pinned by sha256).
  The default wheel falls back to the CPU without an error on CUDA 12 drivers and on Volta; the choice is written to
  `<prefix>/<env>/warp_wheel.txt`, and every result records the installed version tag.
- Existing venvs and trees are only checked, never modified. `--dry-run` prints the commands.

## How the driver finds an interpreter

For each `[env.<name>]`, the first existing candidate of `python` wins; `$DK_BENCH_PY_<NAME>` (upper case, `-` as
`_`) is tried first. Placeholders: `{driver}` (the driver's interpreter), `{prefix}` (`$DK_BENCH_ENV_PREFIX`, default
`runs/bench/envs`), `{repo}` (the checkout), `{env:VAR}` (an environment variable; the candidate is skipped when it
is unset). Relative paths are relative to the checkout. The worker gets `PYTHONPATH` = the checkout + the
environment's and the engine's `pythonpath` entries (never the driver's `PYTHONPATH`), `PYTHONNOUSERSITE=1`, and the
`vars` of both.

Other `*.toml` files in this directory are merged over `envs.toml` in name order (the entries of an additional
adapter can live next to it), then machine-specific settings (another interpreter, another source tree) from an
untracked TOML with the same layout, `$DK_BENCH_ENVS` or by default `runs/bench/envs.local.toml`:

```toml
[env.mjw-native]
python = ["/path/to/venv/bin/python"]
```

`python -m benchmarks.bench env-check` shows which interpreter each engine resolves to and whether its adapter
imports there.

## The MuJoCo Warp engines

Each MuJoCo Warp adapter refuses to run on a tree other than its own (checked in setup and by `env-check`):

| Engine | Check | Adapter |
|---|---|---|
| `mjw` | `import mujoco_warp` is the installed PyPI distribution, version 3.17.0 | `bench/adapters/mjw.py` |
| `mjw_pr1535`, `mjw_pr1535_fix` | the package source hash (sha256 over the `mujoco_warp/**/*.py` files, as `nablaengine.provenance.tree_sha256` computes it) of 357a75d / 4af093f | `bench/adapters/mjw_pr.py` |

- Contact / constraint capacity: the three engines size it from `benchmarks/data/mjw_capacity.json` (the documented
  procedure in `bench/adapters/mjw_common.py`; remeasure with `python -m benchmarks.bench.adapters.mjw_capacity` in
  the `mjw-native` environment after a corpus changes), or with the `matched` variant at 64 contacts / 288 rows per
  world (NablaEngine's capacity; whether MuJoCo Warp accepts `njmax` 288, which is not among its listed sizes, is
  to be checked in this environment before a matched benchmark is reported).
- The evidence that MuJoCo Warp 3.17.0 has no backward (`wp.Tape` runs, the step contributes exactly zero):
  `PYTHONPATH=. runs/bench/envs/mjw-native/venv/bin/python -m benchmarks.bench.adapters.mjw --tape-evidence --out
  runs/bench/<dir>/mjw_tape.json`; every `mjw` unit also records it in its diagnostics (`extra.tape_evidence`).

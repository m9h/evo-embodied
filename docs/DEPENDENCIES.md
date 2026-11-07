# Dependency compatibility and upgrade evidence

Reviewed 2026-11-06 (America/Los_Angeles). Runtime checks ran on macOS arm64,
Python 3.12.13, with CPU JAX unless explicitly marked MPS. This is a dependency
upgrade with bounded integration checks, not competitive research validation.

## Adopted versions

`uv.lock` records the full resolved dependency graph, including CUDA packages.
The main environment intentionally does not select every latest release.

| Package | Previous lock | New lock | Decision |
|---|---|---|---|
| JAX / jaxlib | 0.9.2 | **0.10.2** | Newer 0.11.1 and 0.12.0.1 fail a float32 batching gate |
| MuJoCo / mujoco-mjx | 3.6.0 | **3.6.0** | 4.0.0 adds a Bittle replay failure |
| NumPy | 2.4.3 | 2.5.4 | Updated |
| Matplotlib | 3.10.8 | 3.11.2 | Updated |
| MediaPy | 1.2.6 | 1.2.7 | Updated |
| JupyterLab | 4.5.6 | 4.6.6 | Updated |
| ipykernel | 7.2.0 | 7.4.0 | Updated |
| ipywidgets | 8.1.8 | 8.1.9 | Updated |
| evosax | 0.2.0 | 0.3.2 | CMA-ES ask/evaluate/tell smoke passes |
| Brax | 0.14.2 | 0.14.2 | Already current; Ant MJX reset/step passes |
| dm-control | 1.0.38 | **1.0.38** | Latest 1.0.51 requires MuJoCo >=4.0.0 |
| Playground | 0.2.0 | **0.2.0** | Preserve the JAX/MJX reference suite |
| ONNX Runtime | 1.24.4 | 1.31.1 | CPU provider available; no policy inference tested |
| OpenAI SDK | 2.29.0 | 3.46.0 | Client construction/chat interface checked; no requests sent |

The resolver also updates transitive packages. Flax resolves to 0.12.8 under
the JAX <0.11 constraint. All 193 installed packages in the full macOS research
environment pass `uv pip check`.

## Install and verify

Run from the repository root:

```bash
uv sync --locked                   # core CPU environment
uv sync --locked --extra full      # includes strategies, RL, demos and LLM client
uv pip check
uv run --locked --extra full python experiments/validate_stack.py \
  --research --output /tmp/evo-stack.json

JAX_PLATFORMS=cpu uv run --locked python experiments/validate_physics.py \
  --model models/quadruped.xml --model models/bittle/bittle.xml \
  --steps 128 --output /tmp/evo-physics.json
```

Choose new output filenames for each run. The scripts refuse to overwrite
existing evidence. `uv run` synchronizes the requested extras; retain
`--extra full` on commands when you want all research packages kept installed.

`.python-version` selects Python 3.12 because `labmaze==1.0.6`, required by
dm-control, has a macOS arm64 wheel for that Python version. A trial with Python
3.13 attempted a source build and failed without Bazel. This is an optional RL
dependency installation constraint, not a claim that core Python 3.13 fails.
[Published labmaze files](https://pypi.org/project/labmaze/1.0.6/#files).

The `gpu` extra installs CUDA dependencies only on Linux. CUDA packages resolve
in the lockfile, but CUDA execution was not tested on this Mac. Use
`uv sync --locked --extra gpu` on the NVIDIA host and rerun the backend/physics
checks there before measuring training throughput.

## Why newer physics/JAX releases are held

**MuJoCo 4.0.0:** the 128-step Bittle zero-input replay newly exceeds the existing
CPU/MJX tolerances: maximum qvel error **6.58e-5** and sensor error **1.23e-4**.
The sine-input case still passes. This occurs with both JAX 0.11.1 and 0.12.0.1.
Keeping MuJoCo/MJX 3.6.0 restores the Bittle pass with updated JAX. See the
[4.0 candidate report](validation/physics-mujoco4-candidate-128.json).
This is a local acceptance-gate failure, not evidence that MuJoCo 4.0 is generally
incorrect. Its packaging and physics also change in the
[4.0 release notes](https://mujoco.readthedocs.io/en/4.0.0/changelog.html).

**JAX 0.11/0.12:** example 03's float32 `vmap` evaluation differs from evaluating
the same two controllers individually after 16 steps. With JAX 0.12.0.1 the
maximum absolute fitness difference is **1.68e-5**, exceeding the predeclared
`atol=1e-6, rtol=1e-5` smoke gate. 0.11.1 reproduces the same discrepancy;
0.9.2 and 0.10.2 pass with the same saved controller arrays. Float64 passes on
0.12.0.1, but switching the test's precision would not validate the examples'
default float32 behavior. See the [0.12 candidate checks](validation/stack-jax012-candidate.json).
The existing quadruped reset penetration may amplify numerical differences;
the mechanism and longer-horizon impact remain unproven. Keep the version bound
until valid-reset and batch/individual checks pass on representative workloads.

The final [32-step](validation/physics-updated-32-steps.json) and
[128-step](validation/physics-updated-128-steps.json) physics reports retain the
original outcomes: both Bittle inputs and the quadruped zero input pass; the
quadruped sine input fails. The overall physics command still correctly exits
**1**. No tolerance, model geometry or training objective was changed to obtain
these results. See [the original physics findings](VALIDATION_STATUS.md).

The intermediate [JAX 0.12/MuJoCo 3.6 replay](validation/physics-jax012-mj36-128-steps.json)
passes Bittle but does not satisfy the separate float32 batching gate. Candidate
reports record installed versions independently of the checkout lock hash:
those trials intentionally used isolated environments with different versions.
Final adopted reports match the current lockfile hash.

## Playground reference compatibility

[Playground 1.0](https://pypi.org/project/playground/1.0.0/) changes the runtime to
PyTorch, MuJoCo Warp and reaf-sim. The original JAX/MJX suite is retained upstream
under [rss2025](https://github.com/google-deepmind/mujoco_playground/tree/rss2025).
Treat 1.0 as a separate comparison platform, with an independent environment and
task/protocol mapping; do not silently replace the B1 reference suite.

The existing demos now use 0.2.0's `registry.ALL_ENVS` and explicitly request
`config_overrides={"impl": "jax"}`. Some upstream defaults select Warp/CUDA,
which cannot run on this CPU host. The smoke check resets and steps Cartpole
and exercises the G1 registry listing without downloading Menagerie assets.
The Go1/G1 commands inspect environments; they do not load pretrained policies.

## Separate experimental Apple GPU environment

The prior local environment had jax-mps 0.10.0 with incompatible JAX/jaxlib 0.9.2.
The main environment is now managed solely by its lockfile. The separate
[MPS project](../environments/mps/pyproject.toml) pins **jax-mps, JAX and jaxlib
0.11.1**, plus MuJoCo/MJX 3.6.0, and has its own lockfile:

```bash
uv sync --locked --project environments/mps
environments/mps/.venv/bin/python experiments/validate_stack.py \
  --backend mps --lockfile environments/mps/uv.lock \
  --output /tmp/evo-mps.json
```

Metal device access is required; a sandboxed run reported no Metal device, while
the same test with GPU access passed. [Recorded MPS checks](validation/stack-mps.json)
cover float32 JIT matrix multiplication, a simple gradient, and two batched
16-step Bittle MJX-JAX rollouts with finite, repeatable results. They explicitly
select `impl="jax"` for model and data construction. Automatic MJX device
selection otherwise rejects the MPS device in this configuration.

This profile is experimental: it does not pass the main stack's separate
float32 quadruped batching gate and is not approved for comparable research
scores. No CPU/MPS trajectory parity, physics gradients, throughput advantage or
long training stability is established. Metal lacks float64, so the existing
float64 physics probe is a CPU reference test, not an MPS test. Follow
[jax-mps version pinning](https://pypi.org/project/jax-mps/0.11.1/) rather than
installing its latest release into the main environment.

## Acceptance and next update

The adopted [dependency smoke report](validation/stack-updated.json) passes all
nine checks. It covers compiled numerical operations, the batched controller
evaluation, image output/notebook imports, CMA-ES, Brax, dm-control, Playground,
and client interfaces. It does not launch a notebook kernel, render a physics
viewer, run ONNX inference, train PPO or call an LLM service.

For the next upgrade, resolve in a fresh environment, run these checks plus both
physics horizons, compare with the retained baseline, and record versions and
failures before changing the pins. Release the MuJoCo/JAX holds only after the
specific new discrepancies are understood or resolved. The local pre-upgrade
environment is preserved in the ignored `.venv-before-dependency-upgrade/`;
restore the old manifest/lock and that directory's original `.venv` path for an
exact local rollback. Virtualenv command entrypoints embed their creation paths.

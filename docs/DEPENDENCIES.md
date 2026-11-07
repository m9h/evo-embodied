# Dependency compatibility and upgrade evidence

Reviewed 2026-11-06 (America/Los_Angeles). Runtime checks ran on macOS arm64,
Python 3.12.13, with CPU JAX unless explicitly marked MPS. This is a dependency
upgrade with bounded integration checks, not competitive research validation.

## Adopted versions

`uv.lock` records the full resolved dependency graph, including CUDA packages.
The main environment intentionally does not select every latest release.

| Package | Previous lock | New lock | Decision |
|---|---|---|---|
| JAX / jaxlib | 0.9.2 | **0.12.0.1** | Valid reset resolves the short float32 batching regression |
| MuJoCo / mujoco-mjx | 3.6.0 | **4.0.0** | Tightened solver convergence resolves Bittle replay regression |
| NumPy | 2.4.3 | 2.5.4 | Updated |
| Matplotlib | 3.10.8 | 3.11.2 | Updated |
| MediaPy | 1.2.6 | 1.2.7 | Updated |
| JupyterLab | 4.5.6 | 4.6.6 | Updated |
| ipykernel | 7.2.0 | 7.4.0 | Updated |
| ipywidgets | 8.1.8 | 8.1.9 | Updated |
| evosax | 0.2.0 | 0.3.2 | CMA-ES ask/evaluate/tell smoke passes |
| Brax | 0.14.2 | 0.14.2 | Already current; Ant MJX reset/step passes |
| dm-control | 1.0.38 | **1.0.51** | MuJoCo 4.0-compatible; Cartpole reset/step passes |
| Playground | 0.2.0 | **0.2.0** | Preserve the JAX/MJX reference suite |
| ONNX Runtime | 1.24.4 | 1.31.1 | CPU provider available; no policy inference tested |
| OpenAI SDK | 2.29.0 | 3.46.0 | Client construction/chat interface checked; no requests sent |

The resolver also updates transitive packages. Flax remains at 0.12.8. All 196
installed packages in the full macOS research environment pass `uv pip check`.
The main environment was rebuilt cleanly for MuJoCo 4.0's split core/studio
packaging. The former JAX 0.10.2 environment is preserved locally in the ignored
`.venv-before-modern-physics/` directory.

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

## How the upgrade regressions were resolved

The version holds at revision `f8125e6` were temporary acceptance failures, not
fundamental incompatibilities. The fixes change model setup and solver accuracy;
the comparison tolerances and float32 batching precision are unchanged.

### MuJoCo 4.0: converge Bittle's contact solver

The original Bittle solver tolerance was the default **1e-8**. At 512 steps,
zero-input native/MJX replay reaches maximum qvel error **6.58e-5** and sensor
error **1.23e-4**, reproducing the failed upgrade gate. With **1e-12** in the XML,
these fall to **6.14e-7** and **2.71e-6**. Both zero and sinusoidal inputs pass at
128 and 512 steps, including ground contact, with the original qpos/qvel/sensor
comparison tolerances (1e-6/1e-5/1e-4, rtol 1e-5).

This is a solver-convergence fix. A diagnostic sweep also passed at 1e-10 and
1e-14, with decreasing worst errors. MuJoCo changed solver stopping criteria
and scaling after 3.6; the [changelog](https://mujoco.readthedocs.io/en/stable/changelog.html)
and [solver options](https://mujoco.readthedocs.io/en/stable/XMLreference.html#option)
explain why a shared default tolerance need not imply identical accuracy across
versions. This does not identify a specific upstream commit as the sole cause.
Tighter convergence can cost more computation; these checks do not establish
training throughput.

Compare the [unchanged-tolerance diagnostic](validation/physics-bittle-loose-512.json)
with the [fixed 512-step replay](validation/physics-modern-fixed-512.json).
To reproduce the control without editing the XML:

```bash
JAX_PLATFORMS=cpu uv run --locked python experiments/validate_physics.py \
  --model models/bittle/bittle.xml --steps 512 --solver-tolerance 1e-8 \
  --output /tmp/bittle-loose-solver.json
```

`--solver-tolerance` controls physics convergence, never the acceptance tolerance.
The control is expected to exit 1; omit the override to use the fixed model.

### JAX 0.11/0.12: remove the penetrating quadruped reset

The old root height of **0.50 m** placed the 0.63 m legs **13 cm inside the floor**.
It produced a large artificial initial contact impulse and amplified differences
between compiled float32 batch and individual execution. Raising the root to
**0.64 m** gives **1 cm initial clearance**, with no change to geometry, mass,
actuators, reward or comparison tolerances. This is a physically admissible
starting pose, not a settled equilibrium.

The original two-controller, 16-step error was **1.68e-5** on JAX 0.12. The corrected
model passes the expanded gate: **12 cases** across seeds 0/1/42, populations 2/4,
and horizons 16/128, with exact repeatability. This also passes on
[JAX 0.11.1 under CPU](validation/batch-jax011-cpu-fixed.json); that report uses the
separate MPS environment's packages but does not test the Metal backend.
[JAX documents](https://docs.jax.dev/en/latest/faq.html#jit-changes-the-exact-numerics-of-outputs)
that compiled operation rearrangements can change floating-point outputs.
The reset intervention resolves the observed short regression; it does not prove
that every rounding difference came from a particular compiler change.

The expanded gate runs inside `validate_stack.py`. For detailed records:

```bash
JAX_PLATFORMS=cpu uv run --locked python experiments/validate_batch.py \
  --output /tmp/evo-batch.json
```

### Remaining research gates

- **500-step float32 batching:** some cases still fail the same strict tolerance
  on [JAX 0.12/MuJoCo 4.0](validation/batch-jax012-fixed.json) and on the
  [JAX 0.10.2/MuJoCo 3.6 control](validation/batch-jax010-reset-500.json), both with
  the corrected reset. These reports intentionally retain failures. Run
  `validate_batch.py --steps 500 --output /tmp/evo-batch-500.json` to reproduce.
  Do not mix batch and individual fitness in research selection without a
  validated evaluation contract. Quantify score/ranking sensitivity over longer
  horizons and population sizes before comparable training claims.
- **Quadruped native/MJX replay:** the driven case still fails. With the valid
  reset, step 45 has matching contact locations/normals and smooth forces but
  different tangential friction frames. The default pyramidal friction cone is
  orientation dependent. A diagnostic elliptic cone reduced this first
  discrepancy to numerical noise, but later capsule/torso contact generation
  differed: at step 112 one native replay had two identical contact points where
  MJX had one. Merely tightening the solver or disabling `multiccd` did not fix
  the full trajectory. Those experimental model changes were not adopted.
  [MuJoCo's contact model](https://mujoco.readthedocs.io/en/stable/computation.html)
  describes the contact frame and friction cone distinction. The next targeted
  check is matched capsule/box contact generation with identical input state.
- **New longer-horizon regression:** the corrected quadruped's zero-input replay
  passes at 128 steps, but at 512 steps the modern stack reaches qpos error
  **4.36e-5** and qvel error **8.18e-4**. The
  [old-stack control with the same models](validation/physics-old-stack-new-models-512.json)
  passes this case. Tightening the quadruped solver to 1e-12 did not resolve it.
  A native-only diagnostic showed that a 1e-12-radian initial hip perturbation
  grows to 4.28e-5 maximum qvel difference by 512 steps, indicating a sensitive
  straight-leg reset; this does not by itself prove the cause of the version
  regression. A stable, settled reset and an explicit perturbation-sensitivity
  test remain necessary. The main dependency upgrade therefore resolves the
  original short gates, **not all newly expanded physics checks**.
- **Benchmark comparability:** changing the reset changes the task's initial
  condition. Reevaluate saved controllers on the same model hash; do not combine
  scores from the old penetrating model and the corrected one. Historical XML
  is available at `f8125e6:models/quadruped.xml`. Settling, randomized
  parent/candidate fairness, CUDA/Metal parity and paper reproduction remain
  separate work in [the validation plan](VALIDATION_PLAN.md).

The [128-step](validation/physics-modern-fixed-128.json) and
[512-step](validation/physics-modern-fixed-512.json) reports record the remaining
physics failures and correctly exit **1** overall. Physics checks now also reject
penetrating resets and report the first failing timestep per field. Passing the
nine dependency smoke checks is not a claim that the full physics gate passes.

Historical failures remain available: [4.0 candidate](validation/physics-mujoco4-candidate-128.json),
[JAX 0.12 candidate](validation/stack-jax012-candidate.json), and the
[previous adopted 128-step replay](validation/physics-updated-128-steps.json).
Isolated comparison reports record installed versions independently of the
checkout lock hash; the main environment's final reports use the current lock.

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

This profile is experimental. Its JAX 0.11.1 packages now pass the corrected
quadruped batching gate on CPU; that does not establish the corresponding MPS
result. The retained Metal smoke report predates the model fixes. No CPU/MPS
trajectory parity, physics gradients, throughput advantage or
long training stability is established. Metal lacks float64, so the existing
float64 physics probe is a CPU reference test, not an MPS test. Follow
[jax-mps version pinning](https://pypi.org/project/jax-mps/0.11.1/) rather than
installing its latest release into the main environment.

## Acceptance and next update

The [modern dependency smoke report](validation/stack-modern-fixed.json) passes
all nine checks, including the expanded 12-case batching gate. It covers compiled
numerical operations, image output/notebook imports, CMA-ES, Brax, dm-control,
Playground and client interfaces. It does not launch a notebook kernel, render a
physics viewer, run ONNX inference, train PPO or call an LLM service.

For the next upgrade, resolve in a fresh environment, run dependency checks plus
both physics horizons and the longer batching diagnostic, and compare against
these retained results. Keep known failures visible rather than redefining them
as successes. The original pre-upgrade environment is also preserved locally in
`.venv-before-dependency-upgrade/`. Restoring either backup requires the matching
manifest/lock and the original `.venv` path because virtualenv command
entrypoints embed their creation paths.

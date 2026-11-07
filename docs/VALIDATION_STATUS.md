# Observed validation status

Original physics review: 2026-11-06. Base revision:
`fde9fc82d41b4aff541df86d023a80d1e43805e7`.

Dependency follow-up: [the upgrade report](DEPENDENCIES.md) records the adopted
MuJoCo/MJX 4.0 and JAX 0.12 stack. Correcting the quadruped's penetrating reset
resolves the short float32 batching regression, including JAX 0.11 CPU checks.
Tightening Bittle's solver tolerance resolves its 4.0 replay regression through
512 steps. The older driven-quadruped native/MJX mismatch remains, as does strict
500-step float32 batch agreement on both old and new stacks. The expanded
512-step zero-input quadruped check also exposes a new-stack regression that
solver tolerance alone does not fix; see the upgrade report's remaining gates.
The original versions, model settings and raw reports below remain historical evidence.

Ecosystem integration refresh, **2026-11-07**:

- Added [project/backend decisions](PROJECT_CHOICES.md) and source-pinned roles
  for QDax, Revolve2, EvoRL and TensorNEAT. Only QDax was newly installed; the
  other three remain planned adapters with explicit acceptance criteria.
- Added QDax 0.5.0 as optional `qd`/`full` dependencies. All 214 installed packages
  pass `uv pip check`; the prior MuJoCo/JAX versions and model XMLs are unchanged.
- The [ten-check integration report](validation/stack-ecosystem.json) passes,
  including QD archive replay/repeatability and Brax PPO policy construction and
  inference. PPO optimization, viewers and physical robot deployment were not run.
- The [three-seed arm report](validation/qdax-arm-integration.json) records
  192 candidate evaluations per method/seed, plus separately counted fixtures
  and elite replay. It contains occupied genomes/descriptors and histories.

| Seed | MAP-Elites coverage / QD score | Random-search coverage / QD score |
|---|---|---|
| 0 | 43.75% / 8.8933 | 50.00% / 10.3010 |
| 1 | 56.25% / 12.8023 | 46.875% / 9.7667 |
| 2 | 67.1875% / 15.0389 | 48.4375% / 10.1353 |

This is the upstream analytic planar arm with an 8×8 grid and fitness offset
0.5. The small integration run is not a superiority study; random search wins
seed 0. It does not exercise MuJoCo contacts, solve the open physics gates or
validate the planned dynamic-robot QD adapter. This integration run does not
retest or clear the existing physics failures; their November 6 model and
runtime versions remain recorded below and in the dependency report.

**Current conclusion:** the project is a promising experiment substrate with
simulation and training examples. Competitive research performance and paper reproduction have
not been established. The physics probes retain an actionable
failure on the quadruped and passes the tested Bittle cases. The current
quadruped reset has 1 cm ground clearance; settling and the remaining evaluator
issues still need work. These checks are part of B0, not completed platform
validation. Old/new model scores must be reevaluated under the same reset.

## What ran

```bash
env JAX_PLATFORMS=cpu .venv/bin/python experiments/validate_physics.py \
  --model models/quadruped.xml --model models/bittle/bittle.xml \
  --output /tmp/evo-physics-32.json

env JAX_PLATFORMS=cpu .venv/bin/python experiments/validate_physics.py \
  --model models/quadruped.xml --model models/bittle/bittle.xml \
  --steps 128 --output /tmp/evo-physics-128.json
```

The exact captured outputs are [32-step JSON](validation/physics-32-steps.json)
and [128-step JSON](validation/physics-128-steps.json). Output path names above
are examples for rerunning; the probe refuses to overwrite an existing file.
Both runs exited **1**, correctly indicating a failed case.

Environment: macOS arm64, Python 3.13.13, JAX/jaxlib 0.9.2, MuJoCo and
mujoco-mjx 3.6.0, NumPy 2.4.3, CPU backend, float64. The local environment emitted
a jax-mps/jaxlib compatibility warning and missing optional Warp imports. CPU was
selected explicitly; no claim is made about MPS, CUDA or MJX-Warp. Installed
versions are recorded independently of the lock hash because this existing
environment was not rebuilt for the check.

The probe uses zero commands and low-amplitude phase-shifted sinusoidal commands,
replaying the exact same input arrays through native MuJoCo and MJX-JAX. It checks
qpos, qvel and sensor traces, finite values, CPU warnings and exact repeatability
of the compiled MJX rollout. This is open-loop model replay, not a trained policy.

## Results

| Model / input | 32 steps (0.064 s) | 128 steps (0.256 s) | Maximum absolute qpos / qvel error at 128 steps |
|---|---|---|---|
| Quadruped / zero | Pass | Pass | 1.34e-8 / 1.19e-7 |
| Quadruped / sine | Fail | Fail | 3.28e-3 / 6.05e-2 |
| Bittle / zero | Pass, before contact | Pass, with contact | 1.20e-11 / 1.08e-8 |
| Bittle / sine | Pass, before contact | Pass, with contact | 2.64e-15 / 3.64e-13 |

These maxima range over mixed state coordinates (positions and angles, and their
velocities); they are not single-unit displacement measurements. Sensor error on
the driven quadruped reaches 3.28e-3. All recorded traces were finite, CPU warning
counts were zero, and MJX repeated exactly in these local runs.

The 128-step checks include at most four active contacts on the quadruped and
eight on Bittle. The quadruped starts with minimum contact distance **-0.13 m**;
Bittle starts without contacts. The short pre-contact Bittle success therefore
was not treated as sufficient, and the horizon was extended to include contact.

The quadruped discrepancy exceeds the predeclared probe tolerances (qpos 1e-6,
qvel 1e-5, sensors 1e-4, rtol 1e-5). The tolerances were not widened to obtain a
pass. Initial penetration is a separate observed problem and a candidate contributor;
it has not been established as the cause of CPU/MJX divergence. Next isolate the
first differing step, solver/constraint settings and reset behavior. Contact
engines need not be bitwise identical, but the behavioral impact must be measured
before treating CPU-rendered behavior as evidence for an MJX-trained controller.

The JSON includes synchronized warm timings and compilation time for diagnosis.
These tiny single-instance runs are not population-throughput benchmarks and do
not establish a CPU/GPU speedup.

## What changed in this review

- Added a source/version comparison catalog and B0-B9 experiment plan with
  baselines, budgets, failure criteria and a staged implementation backlog.
- Added the reproducible physics probe and retained both successful and failed
  raw results with hashes and environment metadata.
- Removed the estimated PyBullet speedup printed by example 03 and unsupported
  numeric speedup promises from the README; real throughput claims remain pending.
- Corrected the fitness explanation: fixed-horizon mean velocity is proportional
  to displacement; command penalties are proxies; phase-shifted clocks at one
  frequency are redundant inputs to a fully connected controller.
- Corrected the planned EvoGym control study to identify B from input histories
  and validate bounded feedback in the nonlinear simulator. Separated EvoGym
  from the differentiable SoftZoo track and historical from modern baselines.

No training campaign, paper reproduction, reference installation, hardware trial,
or LLM evolution was performed. Existing models and training algorithms remain
unchanged, including the randomized-selection issue documented in the plan.
The pre-existing untracked `experiments/mps_smoke.py` was left untouched.

## Next evidence to collect

1. Resolve PR-A in [the plan](VALIDATION_PLAN.md#implementation-sequence-and-deliverables):
   valid resets, CPU/MJX discrepancy, comparable randomized parent/candidate scores,
   and reliable evaluator semantics.
2. Build the common runner and independent-seed reporting, then measure B1 and
   export B2/B4 upstream traces.
3. Run the fixed-body EvoGym pilot before committing to the billion-step co-design
   reproduction budget. Publish failures and compute costs alongside scores.

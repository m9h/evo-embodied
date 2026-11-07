# Validation plan for a competitive research platform

Status: proposed experiments, reviewed 2026-11-06. Only the small engine replay
probe has been implemented and run; see [observed results](VALIDATION_STATUS.md).
[RESEARCH_COMPARISONS.md](RESEARCH_COMPARISONS.md) identifies papers, code, and
version boundaries. A demonstration video is not a validation result.

## Claims and evidence gates

| Claim | Required evidence | Current state |
|---|---|---|
| Correct research substrate | Matched CPU/MJX replay, reset/sensor/action tests, reproducible seed handling and saved provenance | B0 started; contact/reset issues remain |
| Reproduces developmental-neurosimulation demonstrations | B2-B4 reference runs, port checks, documented deviations, independent repeats | Not implemented |
| Competitive evolutionary-control platform | B1 and B5 matched-budget comparisons, uncertainty intervals, measured time-to-solution | Not measured |
| Competitive morphology/control co-design platform | B6 uses the same upstream tasks and inner controller budgets as its comparators | Documented direction only |
| A useful new research contribution | B7 or B9 improves held-out behavior over matched ablations | Hypotheses only |

Passing one claim does not establish the others. Report the tasks and budgets on
which a claim holds; do not infer general superiority over EvoGym or biological
validity from a few successful agents.

## P0: audit findings to resolve before training comparisons

| Finding at the inspected revision | Consequence | Required work / acceptance |
|---|---|---|
| `models/quadruped.xml` begins with penetrating ground contacts; the replay probe measures a minimum distance of -0.13 m | Initial contact impulse can contaminate movement fitness | Define a settled, physically admissible reset. Archive the legacy reset as its own condition; compare old/new resets and charge settling consistently |
| Driven quadruped replay differs between native MuJoCo and MJX-JAX | Rendering a policy on the CPU may not reproduce its training trajectory | B0: isolate first divergent step, constraint/contact/solver settings and precision; do not loosen tolerances after seeing the result |
| `examples/11_domain_randomization.py:evolve` keeps parent fitness while changing model/noise | Selection compares different objectives | Reevaluate parents and candidates on the same sampled model/noise panel every generation; regression case: an unchanged candidate must tie its parent. Select the final policy on a fixed validation panel |
| `experiments/sweep.py` still uses the older distance/height objective; example 09 uses height/uprightness-gated velocity | Scores and speed measurements across examples are not directly comparable | One versioned task/evaluator contract with component metrics. Keep the older objective as an explicit ablation |
| Many examples fix seed 42; no common multi-seed evaluator | Best-run reporting can hide instability | CLI/config seeds for initialization, world generation, mutation and evaluation; independent train/validation/test streams |
| `examples/03_mjx_parallel_evolution.py` estimates PyBullet runtime instead of measuring it | Its printed speedup is not benchmark evidence | Replace that estimate in the research runner with measured native CPU, serial MJX and batched MJX timings on identical workloads |
| `mean(v_x)` over a fixed horizon equals displacement/time (up to sampling); `ctrl^2` and command differences are proxies | Reward explanations can imply improvements the metrics do not establish | Use health/contact diagnostics and time-resolved progress; distinguish actuator effort, physical work, command smoothness and mechanical jerk |
| Phase-shifted sine/cosine inputs at one frequency are linearly redundant | More clock channels do not demonstrate a richer periodic basis | Compare no clock, one sine/cosine pair, redundant phase channels, and genuinely different frequencies at matched controller capacity |
| `models/single_link.xml` deliberately omits a touch-sensor site for a student exercise | Loading every XML is not a valid research readiness test | Mark it as an incomplete teaching fixture; use valid dedicated analytic fixtures in B0 |
| No committed shared benchmark runner, CI suite, environment locks for references, or statistical report | Examples alone cannot support published comparisons | Deliver the run contract below before a full campaign |

Hardware deployment, paid LLM evolution, long GPU jobs, and reference-platform
installation are outside this planning run. No experiment below has been silently
scheduled. Small local physics probes are recorded separately.

## Shared experiment contract

Before implementing individual adapters, add a `benchmarks/` package with task,
controller, search, evaluator and recorder interfaces. Preserve the teaching
examples as examples; route research runs through one evaluator. The existing
`experiments/sweep.py` is a starting point, not this interface already implemented.

Every resolved run configuration must include:

- Experiment ID and protocol version; source/repo commits and dirty patch hash;
  package lock, OS, device model, backend, precision and compiler flags.
- Task/model/body hashes, reset distribution, units, observation order and
  normalization, sensor timing, action range, actuator interpretation, control
  period, physics timestep, episode horizon, termination and fitness version.
- Optimizer/controller hyperparameters, population, morphology decoder/repair,
  training/validation/test seed lists, tuning budget and stopping rule.
- Charged costs: initial population, parent reevaluations, candidate evaluations,
  validation rollouts, physics steps, policy updates, invalid designs, retries,
  compilation, wall time, accelerator hours and LLM calls/tokens when relevant.

Proposed output layout (not a claim that these files exist today):

```text
results/<experiment>/<run-id>/
  config.json             # resolved settings, seed streams, task hashes
  provenance.json         # commits, patch, packages, hardware, backend
  metrics.jsonl           # generation/episode records and failures
  checkpoints/            # population, optimizer state, PRNG state, best policy
  evaluation.npz          # held-out episode metrics and optional trajectories
  summary.json            # machine-readable status and costs
  figures/                # generated by a separate report command
```

Resume must reproduce an uninterrupted run, including the next mutation and
evaluation seeds. Failed/NaN/invalid runs stay in the denominator and receive a
predeclared failure score; never drop them to improve the aggregate. Do not reuse
a run directory or select the final policy using test episodes.

### Statistics and budgets

For new experiments, use three pilot training seeds (0-2), then at least ten
independent confirmation seeds (100-109). A pilot diagnoses plumbing and estimates
variance; it cannot justify a competitive-performance claim. Size the final study
from pilot variance and a practically meaningful effect before opening test data.
For each trained policy use 20 fixed held-out initial-condition/perturbation seeds
(1000-1019), shared across methods. Average those episodes within training seed;
they are not 20 independent trained agents. Freeze normalization and tuning using
training/validation data only. Deterministic reference worlds need separately
specified perturbations for a robustness extension.

Report all seed-level scores, mean, median, interquartile mean, failure rate and
95% bootstrap intervals (10,000 resamples). Pair comparisons by shared scenario
blocks where possible; bootstrap training seeds and tasks rather than individual
timesteps. Treat exploratory multi-task tests separately from preregistered primary
comparisons. Preserve the original seed count for a historical reproduction;
the confirmation campaign is an additional analysis, not a retroactive change.

Proposed competitiveness rule: on each preregistered primary task, the lower 95%
confidence bound for candidate-minus-reference test return must exceed `-delta`.
Set `delta = 0.10 * abs(reference_pilot_mean - random_pilot_mean)` using pilot data,
then freeze it before confirmation. If that gap is negligible, choose and justify
an absolute task-unit margin before confirmation. Report both raw and normalized
scores. An efficiency claim additionally requires lower confidence-bound speedup
above 1 at matched quality/budget, with compilation included in end-to-end time.
A superiority claim requires the return-difference interval to be above zero.
These are proposed decision rules, not established literature thresholds.

Measure cold startup/compilation, warm simulation, optimization, evaluation and
end-to-end time separately. Synchronize JAX before stopping timers; use at least
five warm timing repeats. Compare both equal physics-step/evaluation budgets and
equal wall-clock budgets. Record CPU core/process counts, GPU memory, and batching;
single-agent latency and population throughput are different results.

## B0: engine, reset and evaluator correctness (first implementation)

**Available now:** `experiments/validate_physics.py`, a headless open-loop replay
probe. It uses native MuJoCo and MJX-JAX with identical XML/actions and float64,
checks complete qpos/qvel/sensor traces, repeated execution, and CPU warnings.
It records model/runner hashes and actual installed versions. Default engineering
tolerances are qpos 1e-6, qvel 1e-5, sensors 1e-4, with rtol 1e-5. It exits nonzero
on a failed case; these provisional tolerances are not universal contact guarantees.

```bash
env JAX_PLATFORMS=cpu .venv/bin/python experiments/validate_physics.py \
  --model models/quadruped.xml --model models/bittle/bittle.xml \
  --steps 128 --output /tmp/evo-physics-new-run.json
```

Use a new output path. The existing environment is sufficient; this command does
not update dependencies. It explicitly chooses CPU to avoid silently benchmarking
the local experimental MPS plugin. It does not render, train, or deploy a robot.

**Remaining:** add analytic free-fall/pendulum fixtures, settled-contact tests,
random policy replay, single-step sensor timing, warmstart/solver audits, and
CPU/JAX serial/batch agreement. Then test float32 on each intended GPU. Validate
at 1, 32, 128 and 2,000 steps; use short-horizon numerical checks and longer-horizon
task metrics because contact trajectories can diverge. Add a gradient directional
finite-difference check for any smooth experiment claiming autodiff; report contact
events separately. Save first divergence and terminal-condition disagreement.

**Exit:** valid resets, finite trajectories, frozen tolerances justified by
independent fixtures, deterministic CPU repeats, documented backend differences,
and evaluator agreement before interpreting optimizer comparisons.

## B1: locomotion and population throughput

**Question:** does batching improve time-to-quality without rewarding a launch,
fall, or body slide? Start with the repository quadruped and Bittle after B0.

Compare zero control, a fixed periodic controller, random search, parallel hill
climbing, evosax CMA-ES/OpenES, and PPO through a matched environment adapter.
First compare search methods on the same feedforward policy class. PPO is a
separate algorithm comparison with its architecture/normalization budget recorded.
Add a pinned MuJoCo Playground locomotion task only after reproducing its upstream
PPO baseline; do not compare its native reward with this repo's custom score.

Pilot: 3 seeds x 64 candidates x (20 generations + initialization) = 4,032
training rollouts per search method/task, each 100 control steps x 20 physics
steps. Confirmation starts at 10 seeds x 64 x 201 = 128,640 rollouts per method/task;
revise the declared budget using pilot costs before launching. Charge random
search the same total evaluations, including initialization and reevaluations.
Use populations 1/16/64/256 for timing on the same saved candidate pool.

Measure held-out forward displacement/speed, progress in each episode quarter,
survival time, foot versus torso contact, airborne fraction, uprightness, lateral
drift, actuator work `integral(abs(qfrc_actuator * qvel)) dt`, effort proxy, and
command variation. Include initial/reset energy and settling time. The current
height window is not a direct foot-contact measurement. Freeze a walking criterion
on pilot reference rollouts; do not equate an optimizer's scalar fitness with gait.

**Exit:** beat zero/random controls, satisfy the frozen physical behavior criteria,
pass the stated non-inferiority rule against the strongest tuned reference, and
publish measured time-to-quality. A static image or one best seed cannot pass.

## B2: BraGenBrain body/connectome experiment

**Target:** paper Fig. 3, 1,200 frames, scores sampled at frames 200/400/600/800/1,000/1,200.
Factorial: 3 body ratios (1:1, 3:2, 5:4) x 2 sensor-effector spacings x 2 network
sizes (5 and 10) = 12 conditions. Recover absolute dimensions and the exact meaning
of network size from pinned source/configuration before running. Paper settings:
mutation .05, crossover .05, selection .1 and neutral selection .05.

Implement a pure planar reference adapter preserving inverse-distance sensing,
integration order, DAG ordering, weight normalization/8-bit quantization,
mutation/crossover and movement-score calculation. Resolve the histogram binning,
entropy definition/log base and population-update schedule from code. Keep an
equation/code discrepancy log; never silently fix an ambiguous paper formula.

Export matched initial worlds, genomes and random operator draws from the Kotlin
reference. Test sensing, network outputs and one-step motion first, followed by
short traces and full distributions. Pilot 3 worlds per condition; confirmation
30 independent worlds per condition = 360 x 1,200 = 432,000 **world frames**
(multiply by actual population sizes for agent-step costs).

Metrics: original movement-score histograms/H curves, occupancy/coverage and
network statistics. Compare unselected random networks, original GA, weight-only
evosax, then topology evolution as a separate extension. Match all selection
budgets. The paper reports mixed effects; reproducing it does not require every
10-node network to win. Compare distributions/effect sizes; audit the printed
p-value inequalities instead of treating them as exact acceptance thresholds.

**Exit:** trace agreement with the exported reference and comparable population
distributions under a pilot-calibrated, frozen tolerance. Report behavioral
reproduction separately if exact original assets or random draws are unavailable.

## B3: MultiLearn associative learning and forgetting

**Target:** paper Figs. 4-6 and pp. 16-17. Reproduce olfactory Li-Hopfield dynamics,
gustatory boundary, generalized Hebbian learning with depression, learning-rate
schedule, memory and movement rules. STDP in bl1 is a subsequent mechanism test.

First compare stationary input sequences and association-matrix updates against
the Cython/C reference. Then use 20 sources for the illustrated field, with
matched random worlds. Pilot 3 worlds; confirmation 30 worlds, 30,000 steps each.
Replace the source field after steps 10,000 and 20,000 to test the reported loss
of recall. Record boundary/zero-denominator handling explicitly.

Baselines: learning disabled, shuffled odor/taste pairings, depression disabled,
and the original learner. Keep stimuli/motor rules fixed. Measure held-out
odor-to-taste error, actual versus ideal preference, unpleasant-source visits,
adaptation after each replacement, forgetting and weight norms. Do not change a
test learner using evaluation examples. A bl1 alternative needs the same sensory
information and explicitly budgeted neural updates/encoding/decoding.

**Exit:** stationary dynamics/update agreement plus paired evidence that learning
improves association/avoidance over disabled and shuffled controls. Publish
instability and capacity failures rather than tuning them away as a reproduction.

## B4: shared-stimulus vehicle collectives

**Target:** paper Table 2 and Fig. 9: 2b vortex, 3a crossing, 2a density/jamming,
and the multiple-stimulus mixture. Use the Processing implementation as reference.
Export coordinates and identify its timestep, activation, wheel kinematics,
initial density and boundary rules before attempting a port.

Match one agent's traces first, then a fixed stimulus, a moving stimulus and
multiple stimuli. No explicit agent communication/alignment or added contact force.
Proposed scalability extension: 10/100/1,000 agents x 3 stimulus conditions x
10 independent worlds x 2,000 frames = 180,000 world frames in total
(20,000 per population/stimulus combination); distinguish this sweep from original demo settings.

Metrics: radial occupancy, angular circulation around a stimulus, heading order,
density-cluster sizes, low-speed fraction and nearest-neighbor distributions.
Predefine cluster/low-speed thresholds from reference pilot trajectories, then
freeze. Compare native Processing, NumPy/Mesa and JAX under identical rules;
include stimulus-off and wiring-permutation controls. A dense cluster alone does
not establish mechanical jamming or communication.

**Exit:** matched individual dynamics and reference-distribution/motif agreement;
any throughput claim follows only after behavioral parity. JaxMARL is deferred
until studying the effect of introducing learning/communication.

## B5/B6: EvoGym control, then morphology/control co-design

**B5 fixed body:** reuse official reference bodies on `Walker-v0` and
`BridgeWalker-v0`. Keep upstream physics, reset, observations, actions, reward and
episode limits unchanged. Compare upstream PPO, random/periodic controls and
evo-embodied hill climbing/evosax controllers through the isolated worker. Match
total environment steps, controller capacity where possible, and tuning budgets.
This tests the algorithm adapter without confounding it with body search.

**B6 co-design:** compare upstream GA+PPO, BO+PPO and CPPN-NEAT+PPO with an
evo-embodied outer search using the **same PPO inner learner**. Only then cross
outer search with evolved versus PPO controllers. Include random-valid-body+PPO
and fixed-body+PPO controls. Define categorical voxel encoding, disconnected-body
rejection/repair, duplicates, caching and invalid-design costs before comparison.
Report proposal counts as well as unique trained bodies.

Two distinct campaigns:

1. **Historical reproduction:** pin release 1.0.0. Recover task budgets and all
   Appendix D parameters, reproduce the Fig. 3 Walker/BridgeWalker subset and
   the Fig. 4 Bridge Walker morphology snapshots. Fig. 3 uses six random seeds;
   Appendix D gives population 25, usually 5x5 bodies, and 512,000 RL steps/body.
   Verify the task-specific 250/500/750-body limits in the release before freezing.
   [Paper](https://arxiv.org/abs/2201.09863),
   [original release](https://github.com/EvolutionGym/evogym/releases/tag/1.0.0).
2. **Modern comparison:** pin the modern snapshot and retrain every comparator
   in that environment. Upstream's current examples are changed and their quick
   defaults are smoke runs. Do not overlay modern scores on old paper curves as
   though they used the same implementation. [Upstream instructions](https://github.com/EvolutionGym/evogym/blob/f63c649c688adda1accad5f86731aad96606388a/examples/README.md).

Proposed modern pilot: 10 unique bodies x 100,000 steps/body x 3 seeds x 2 tasks x
3 methods = **18 million environment steps**, plus evaluation and failed proposals.
A historical-sized subset at 250 bodies x 512,000 steps x 6 seeds x 2 tasks x
3 methods is **4.608 billion environment steps**. These are budget estimates,
not approved launches or wall-time predictions. Estimate wall time from B5 first.

Metrics: canonical test return versus unique bodies, total environment steps and
wall time; morphology diversity, validity/failure rate, and budget to a fixed
quality target. Save all bodies, policies and top-of-generation snapshots.
Generalization requires a separately specified held-out perturbation/terrain
suite; tuning on Walker and reporting only Walker is an in-task result.

**Exit:** reproduce the reference pipeline, then pass the frozen non-inferiority
criterion at matched resources. Later extend to `CaveCrawler-v0` and a manipulation
task to test whether gains extend beyond easy locomotion; report those separately.

## B7: controllability-guided EvoGym extension

Start with frozen bodies/policies from B5/B6. Collect persistently exciting input
perturbations around a nominal trajectory with bounded actions; record `(x,u,x_next)`.
Fit `x_next = A x + B u + c` using DMD with control, checking against
[PyDMD DMDc](https://pydmd.github.io/PyDMD/dmdc.html). Exact autonomous DMD on `x`
alone cannot supply B. Split by whole trajectories, fit normalization on training
data, and choose rank/regularization on validation trajectories.

Use lifted/centered coordinates and a finite-horizon reachability Gramian unless
stability permits an infinite-horizon solver. Check rank/conditioning and
stabilizability before Riccati solving. A single LTI fit around an entire gait
may fail; phase-conditioned/local models are a separate preregistered extension.
Apply bounded feedback around a nominal trajectory in the nonlinear EvoGym task,
with an explicit feedforward term and phase/state estimator. Do not assert that
unconstrained regulation to zero will track a walking limit cycle.

Compare nominal controller, zero correction, fitted LQR correction, a tuned
simple PD correction where its state mapping is meaningful, and a wrong-B/null
model. For morphology selection, compare return-only selection with
return+controllability under an equal budget and normalized state/input units.
Pilot 3 bodies; confirm on 10 unseen bodies x 20 perturbation episodes, clustering
statistics by body/training run rather than counting frames as independent trials.

Metrics: held-out multi-step prediction error versus persistence, perturbation
recovery, return/survival, effort, action saturation and prediction-failure rate.
**Exit:** improved nonlinear held-out recovery over the nominal controller,
with a valid identified input model; a larger Gramian alone is not success.

## B8/B9: research extensions after the reference gates

**B8 LLM morphology proposals:** pin the official RoboMorph release/paper version;
first match its grammar, terrains and controller training. For a local adaptation,
compare random grammar mutations, LLM proposals without performance feedback, and
LLM proposals with feedback under the same evaluator. Pilot 8 proposals per arm
and 3 runs; confirm at least 10 independent runs with a frozen proposal/training
budget. Cache prompts/responses, exact model identifier/date, retries and costs.
Include invalid designs in the denominator. Example 07's direct MJCF proposals
and hill-climbing inner loop cannot be labeled an official reproduction.

**B9 developmental/plastic controllers:** connect devograph/hgx or bl1 to the
already validated planar tasks. Compare fixed controllers, equal-budget weight
evolution, frozen developmental programs, growth-disabled programs, and
plasticity-disabled controls. Charge developmental/neural update compute and
match maximum nodes/parameters as well as sensory access. Evaluate adaptation to
unseen worlds, forgetting, topology changes and behavior together. Ten training
seeds x 20 held-out worlds is the initial confirmation design, subject to pilot
power analysis. This addresses the paper's future direction; it is a new study.

## Implementation sequence and deliverables

| Work item | Depends on | Concrete completion evidence |
|---|---|---|
| PR-A: reset/evaluator correctness | None | Resolve audit blockers; analytic/contact fixtures; parent-candidate same-panel regression; CPU/MJX trace report |
| PR-B: experiment runner and reporting | PR-A | Versioned configs, full seed control, cost accounting, checkpoint-resume equivalence, held-out evaluator, JSONL/NPZ report command |
| PR-C: B2/B4 planar references | PR-B | Exported upstream fixtures, equation/code discrepancy log, original-versus-port plots, seed-level results |
| PR-D: B3 learning reference | PR-C infrastructure | Stationary-update tests, acquisition/replacement experiments and forgetting curves |
| PR-E: B1 and B5 baselines | PR-B | Matched search/PPO contracts, timing curves, pinned EvoGym worker, fixed-body baseline report |
| PR-F: B6 co-design | PR-E | Historical reproduction report and separate modern comparison; body/policy archive and complete costs |
| PR-G: B7 controlled identification | PR-F | DMDc oracle check, held-out prediction and nonlinear disturbance-recovery ablations |
| PR-H: B8/B9 extensions | Relevant reference gate | Preregistered ablations and evidence for a scoped new research claim |

The first useful release should contain PR-A through PR-C and PR-E, with a public
failure ledger. PR-D/F/G then establish the developmental-learning and co-design
claims. Estimate person-time and hardware-hours from the pilot measurements, not
from assumed JAX speedups. Pin new references and resolve all unspecified protocol
details in a versioned config before confirmatory runs.

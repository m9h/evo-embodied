# Demo goals and delivery roadmap

Accepted direction: 2026-11-08. This is a delivery plan, not a report of completed
training or hardware experiments. [Validation status](VALIDATION_STATUS.md)
records the evidence; [validation protocols](VALIDATION_PLAN.md) define research
budgets and comparison rules.

## Flagship goal

**Evolve a repertoire of Bittle gaits, then compare their speed, stability and
robustness under changed conditions.** Show the baseline, the search process,
the resulting behavior archive, and what survives held-out evaluation. Report
unsuccessful runs alongside successful ones.

The intended presentation is a replayable archive with behavior clips and
metrics, followed by matched nominal/perturbed trials. Preselect displayed
policies using validation data and a recorded selection rule. Show all seed-level
results and failure rates; a best-run video alone is not completion evidence.
Physical transfer is a later milestone with its own measurements.

## Supporting demos

| Demo | Question it answers | Delivery target |
|---|---|---|
| QDax planar arm | How does evolution fill a behavior archive? | Keep the existing analytic MAP-Elites/random-search example reproducible; add an archive/progress visualization |
| Bittle gait repertoire | Which gaits remain useful when conditions change? | Shared evaluator, OpenCat reference, matched search baselines and held-out robustness report |
| Playground Go1 | Can we reproduce an established locomotion reference? | Pin the RSS 2025/0.2 task, learner, model and checkpoint lineage; reproduce upstream evaluation before testing our methods |
| Revolve2 modular robot | What changes when bodies evolve with controllers? | Isolated worker, fixed body/CPG fixture, then B6-R co-design pilot |

Go1's upstream [RSS 2025 demonstrations](https://playground.mujoco.org/rss2025/)
include joystick locomotion and recovery. The local `--playground-go1` command
only inspects an environment. Current Playground 1.0 has a different runtime;
keep it a separate comparison. Revolve2 is planned, not installed. EvoGym
continues as the soft-body co-design reference under B5/B6.

## Milestones and completion evidence

All unchecked milestones below remain open. Work can proceed on independent
references, but Bittle search comparisons require its evaluator gates to pass.

- [ ] **D1 — One Bittle policy contract (B0 / PR-A).** Export a versioned bundle
  containing architecture, observation names/order/units/normalization, clocks,
  action scaling/limits, joint map, control period, model hash and weights.
  Training and deployment must consume the same contract. Replay recorded
  observations through JAX training inference and NumPy deployment inference;
  require action agreement at `atol=1e-6, rtol=1e-5`. Reject incompatible shapes,
  missing required observations and incompatible contract versions explicitly.
  A no-hardware test must exercise serialization and the actual inference path.
- [ ] **D2 — Audited model and OpenCat baseline (B0/B1 / PR-A/B).** Declare the
  exact hardware variant, firmware and servo type. Compare our MJCF with the
  pinned community model below. Preserve both as separate fixtures until model
  selection is justified. Verify stand/trot/walk joint ordering, signs, offsets,
  units and timing; replay OpenCat motion data and record displacement, foot
  slip, body contact, actuator tracking and warnings. Measure actual-unit mass
  and actuator response before claiming hardware calibration. Save source
  licenses/attribution, hashes and the chosen model/configuration.
- [ ] **D3 — Full-length simulation evaluation (B0/B1 / PR-A/B).** Extend beyond
  the current 512 physics steps (1.024 s): use 10 s nominal episodes and 30 s
  held-out robustness episodes at the frozen task control rate. At the current
  2 ms timestep these are 5,000 and 15,000 physics steps. Charge settling
  separately. Test native MuJoCo, serial MJX and batched MJX; preserve short
  numerical checks and preregister long-horizon behavioral tolerances using
  reference fixtures. Save first divergence, falls, finite-state checks, contact
  traces, progress by episode quarter, speed, drift and actuator work.
- [ ] **D4 — Matched search and robustness study (B1-QD / PR-E).** Add QDax to
  the shared Bittle evaluator. Compare MAP-Elites, random search and hill climbing
  on the same policy/initial population/archive; compare evosax weight search
  through B1. Include zero, periodic and OpenCat reference controls. Freeze the
  left/right contact-duty descriptors and QD-score offset. Reevaluate every
  occupied elite on the same held-out panel and rebuild corrected archives.
  Test friction, added load, actuator response/noise and timed disturbances
  individually before combinations; freeze ranges and seeds before confirmation.
  Correct the example 11 parent/candidate evaluation-panel bug first. Use the
  B1-QD pilot budget, independent seeds and charged reevaluation costs.
- [ ] **D5 — Reproducible demo package (PR-B/E).** Publish a versioned run config,
  policy bundles, archive, nominal/perturbed clips, per-seed tables, failure
  ledger, measured runtime and reproduction commands. Add an archive/progress
  figure to the analytic QDax demo. Release the simulation demo when it passes
  its declared behavioral criteria; make a competitive claim only after the
  confirmation study meets the validation plan's comparison rule.
- [ ] **D6 — Measured hardware transfer.** After D1-D3, verify sensor availability,
  joint calibration, servo response and sustained serial/control timing on the
  identified unit. Use the same deployable observation subset in simulation;
  reserve privileged simulation state for metrics, or label it as an ablation.
  Compare an OpenCat baseline and frozen selected policies on recorded trials,
  including failures. Missing hardware leaves this milestone open; a dry run
  does not count as a transfer result. See [SIM2REAL.md](SIM2REAL.md).

## Bittle audit at revision 5460ca6

| Area | Observed implementation | Required resolution |
|---|---|---|
| Architecture/clocks | Example 10 uses 32 hidden units and 8 clock inputs; deployment defaults to 64 and 2 | D1: shared policy bundle and parity fixture; changing one CLI default is insufficient |
| Observations | Training consumes all 28 model sensor values, including virtual touch and torso state; deployment constructs joint angles plus optional Euler IMU values | D1/D6: one observable input contract; quaternion/Euler conversion, order, timing and missing-data behavior must be explicit |
| Output/timing | Training scales `tanh` by 1.5 at 25 Hz; deployment uses unscaled `tanh` and defaults to 20 Hz | D1/D6: shared action semantics and measured sustained control cadence |
| Model mass | Loaded MJCF totals 0.269 kg; 0.177 kg is the torso, not the robot | D2: compare actual-unit measurements and document all body inertias |
| Servo dynamics | The 40 ms `servo` default is declared but unused; all 8 actuators have `dyntype=none`, `na=0` | D2: identify an actuator model and validate its step response before adopting delay randomization |
| Physics evidence | Tested zero/sinusoidal replay passes through 512 steps | D3: complete policy-horizon and batch evaluation; short replay is not walking validation |

The mass/actuator facts were checked by loading the unchanged XML with MuJoCo
4.0 on 2026-11-08. No training, full-length replay or hardware trial was run for
this roadmap update. Changes to model/observation semantics require retraining
and separately labeled results; retain historical checkpoints and evidence.

## New comparison source

[MarcHesse/bittle-mujoco](https://github.com/MarcHesse/bittle-mujoco/tree/e00a6ab1495c421c0ed522cdf4eb1d3fc436c249),
reviewed 2026-11-08 at `e00a6ab1495c421c0ed522cdf4eb1d3fc436c249`, provides an
MJCF, OpenCat motion fixtures and contact diagnostics. The author reports a
273.5 g measured **Bittle X V1 / BiBoard V0_1 / plastic P1S-servo** unit and
documents model corrections. That is external evidence for that variant;
we have not run or calibrated the model locally. It is a comparison candidate,
not a replacement silently incorporated into our results.

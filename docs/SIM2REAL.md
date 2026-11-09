# Bittle simulation-to-hardware transfer plan

Updated 2026-11-08. Bittle is the project's flagship locomotion demo and future
hardware target. The goal is to compare a repertoire of gaits on speed,
stability and robustness, then measure transfer on an identified physical unit.
The [demo roadmap](DEMO_ROADMAP.md) defines D1-D6 and their acceptance criteria.

**Current status:** the simulator examples and serial deployment prototype do
not share a compatible policy interface. No physical transfer has been validated
here. The commands and interfaces in `sim2real/deploy_bittle.py` are prototype
development tools, not a working export/deploy path for example 10 checkpoints.

## Policy contract to implement first

| Contract field | Example 10 training | Current deployment prototype |
|---|---|---|
| Hidden units | 32 | Defaults to 64 |
| Clock inputs | 8 phase-shifted sine/cosine channels | 2 channels |
| Observations | All 28 model sensor scalars | 8 joint angles plus optional 3 Euler IMU values |
| Action output | `1.5 * tanh(...)` | `tanh(...)`, then radians-to-degrees and offsets |
| Control rate | 25 Hz | Defaults to 20 Hz; serial timing unmeasured |

Changing `--n-hidden` or `--ctrl-hz` alone cannot reconcile these differences.
Implement one versioned policy bundle with the observation order, units,
normalization, clock construction, architecture, action interpretation, joint
calibration, period, model hash and weights. Training and deployment must use
that bundle. Compare both inference paths on the same recorded observations
and require action agreement at `atol=1e-6, rtol=1e-5` before hardware trials.

The simulation includes virtual touch, torso position and torso orientation
information. The current training policy reads them even though some XML
comments describe them as reward-only. Define a deployable observation subset
from the actual hardware's available sensors; keep privileged simulation
observations for metrics or a separately labeled ablation. Specify quaternion
versus Euler conventions, sensor latency and missing-data behavior. A missing
required observation must not silently become a zero-valued measurement.

## Identify the hardware and model

Record robot revision, controller board, firmware, servo type, battery and any
payload. Verify which commands return measured joint positions and which return
commanded targets on that configuration. Confirm IMU units, axes and update rate.
Do not assume that every Bittle/Bittle X variant exposes the same feedback.
Consult the current [Petoi documentation](https://docs.petoi.com/) and
[Python API](https://docs.petoi.com/apis/python-api) for the identified unit.

The local `models/bittle/bittle.xml` is a simplified eight-actuator model. Loading
it with MuJoCo 4.0 gives **269 g total mass**; 177 g is only its torso. It uses
position actuators without activation dynamics: the declared 40 ms `servo`
default is not used by the actuators. Servo delay must be implemented and checked
against measured response before treating it as a calibrated model parameter.

Compare the local model with
[MarcHesse's Bittle X V1 model](https://github.com/MarcHesse/bittle-mujoco/tree/e00a6ab1495c421c0ed522cdf4eb1d3fc436c249).
That source reports a measured 273.5 g V1/P1S-servo unit and includes OpenCat
gaits and contact diagnostics. It is not yet validated in evo-embodied and must
not silently replace our model or be assumed to describe another hardware
revision. Preserve source attribution and separately version every model change.

### Joint map to verify

```text
          FRONT
    FL (8,12)  FR (9,13)
         +------+
         |BITTLE|
         +------+
    BL (11,15) BR (10,14)
          BACK
```

Pairs denote Petoi shoulder/knee indices. Check each joint's sign, zero offset,
mechanical limit and command unit against the selected model and firmware. Then
replay known OpenCat stand/trot/walk sequences using the same conversion and
timing in both environments.

## Planned progression

1. **D1: inference parity.** Export/import the policy bundle, compare actions on
   saved observations, and reject incompatible bundles or unavailable inputs.
2. **D2: model and baseline.** Audit mass/inertia, actuator response, joint map
   and OpenCat fixture traces. Log tracking error, foot slip and body contact.
3. **D3: full-length simulation.** Evaluate 10 s nominal and 30 s perturbed
   episodes across native MuJoCo and MJX. The current passing 512-step replay
   spans only 1.024 s and does not establish walking or transfer.
4. **D4/D5: repertoire and robustness.** Run matched search comparisons, recheck
   occupied elites on held-out scenarios, and publish policy-linked videos,
   metrics, seeds, failures and compute costs.
5. **D6: hardware trials.** Measure sustained command/feedback timing and servo
   tracking on the identified unit. Compare the OpenCat reference with policies
   selected on validation data, then report physical displacement, speed, falls,
   drift and measured-versus-simulated response under recorded conditions.

Domain randomization is a hypothesis to test against deterministic training.
Choose ranges from measurements or label them as synthetic stress tests. Correct
example 11's parent/candidate evaluation-panel mismatch before interpreting a
comparison. Randomization does not guarantee transfer, and a command-magnitude
penalty is not measured actuator energy.

## Current prototype tools

Inspect the deployment interface without opening a serial connection:

```bash
.venv/bin/python sim2real/deploy_bittle.py --help
```

Example 10 remains a simulation training prototype. Its saved policies require
the D1 interface work and, when observation/model semantics change, retraining.
A prototype `--dry-run` only avoids sending serial commands; it is not an
inference-parity test, sensor calibration, or evidence of physical transfer.

## Related work and broader scope

- [opencat-gym](https://github.com/ger01d/opencat-gym) and
  [its deployment project](https://github.com/ger01d/opencat-gym-sim2real) are
  candidate external control/transfer references; pin and reproduce their
  protocols before comparing results.
- [Bittle URDF](https://github.com/AIWintermuteAI/Bittle_URDF) is the source cited
  by the local MJCF. Its provenance does not validate our simplified parameters.
- The [demo roadmap](DEMO_ROADMAP.md) retains Go1 as a published locomotion
  reference and Revolve2 as the modular body–controller co-design example.

## Sim2Real in Context

Bittle addresses rigid-body locomotion with position-controlled servos. EvoGym
is the separate soft-voxel reference; SoftZoo/RoboDiff address different material
and design problems. Each needs its own model, observations, actuator semantics
and transfer evidence. The virtual-rat pathway adds biological and motion-data
validation rather than inheriting validity from a robot gait result. See
[project choices](PROJECT_CHOICES.md), [research comparisons](RESEARCH_COMPARISONS.md)
and [research gaps](RESEARCH_GAPS.md) for those tracks.

# Fitness Function Design for Locomotion

The single most important thing in evolutionary robotics is the fitness function.
A bad fitness function produces robots that exploit loopholes instead of walking.
This document explains why, with examples from this project's experiments.

## The Leap-and-Flail Problem

Our first fitness function was:

```python
fitness = x_distance - 10 * height_penalty - 0.5 * y_drift
```

This produced robots that scored +11.27 fitness by **launching themselves forward
in a single ballistic leap**, then flailing their legs in midair. The height
penalty kept them from falling flat, but the distance reward was so dominant that
a single powerful jump beat any amount of careful walking.

**Why this happens:** The fitness function has no concept of *how* the robot
moved — only *where it ended up*. A 2-meter leap and a 2-meter walk both score
equally. But a leap is far easier to discover by random mutation than a
coordinated gait.

## Three Principles of Locomotion Fitness

### 1. Measure Sustained, Healthy Progress

```python
# Displacement alone does not distinguish walking from a launch or slide.
displacement = final_x_position - initial_x_position

# An example task-specific reward: progress only while healthy.
fitness = mean(where(healthy_at_each_timestep, forward_velocity, 0.0))
```

At a fixed horizon T, time-averaged forward velocity is displacement divided by
T (up to numerical sampling). Replacing distance by mean velocity alone does
not change the ordering of trajectories and does not establish sustained gait.
A 2 m leap and 2 m of walking over the same horizon have the same average speed.

Example 09 changes the objective through a torso-height/uprightness health gate
and a permanent unhealthy flag. Those are additional assumptions, not a property
of velocity itself. Its height window is not a direct foot-contact test. Record
progress in each quarter of the episode, contact type, airborne fraction, survival
and uprightness to distinguish walking, hopping and sliding. Define the intended
behavior before tuning: running can legitimately contain aerial phases.

Reset conditions also matter: the current quadruped begins with ground penetration.
Resolve that initial impulse before attributing launch behavior entirely to the
reward. See [the validation findings](VALIDATION_STATUS.md).

### 2. Penalize Effort and Measure Physical Work Separately

```python
energy_cost = mean(sum(ctrl ** 2))  # command-effort proxy, not joules
fitness -= ENERGY_WEIGHT * energy_cost
```

Without an energy penalty, the optimizer discovers that maxing out all motors
all the time produces unpredictable chaotic motion — and sometimes that chaos
happens to go forward. This is "flailing."

The proxy makes large commands expensive; it does not guarantee an efficient
gait. Here motor gear ratios differ, and Bittle uses position/servo commands, so
`ctrl ** 2` is not measured torque or energy. Log actual actuator generalized
forces and velocities and integrate their absolute power for a mechanical-work
diagnostic. Electrical energy requires an actuator/electrical model.

**How to tune the weight:** Start with `ENERGY_WEIGHT = 0.005`. If the robot
barely moves, reduce it. If it still flails, increase it. The energy penalty
and velocity reward must be balanced — too much energy penalty produces a robot
that stands still (minimum energy = zero torque).

### 3. Penalize Command Variation

```python
ctrl_change = sum((ctrl_t - ctrl_{t-1}) ** 2)
fitness -= SMOOTHNESS_WEIGHT * mean(ctrl_change)
```

Even with an energy penalty, the optimizer can produce rapid alternating
commands (bang-bang control) that are technically low-mean-energy but
physically violent. The smoothness penalty makes rapid control changes
expensive, forcing the controller to produce smooth, gradual commands.

This discourages abrupt commands; it does not ensure walking or sinusoidal joint
motion. A command-difference penalty is not mechanical jerk (the derivative of
acceleration). Normalize time-dependent metrics when changing control frequency
and report actuator/joint kinematics separately.

## The Full Fitness Function

This is an illustrative decomposition, not the exact current implementation of
every example. Example 09 adds a health gate and uprightness term, while the sweep
runner still uses the older distance-based objective. Research comparisons need
one versioned evaluator and separate component metrics.

```python
fitness = (
    VELOCITY_WEIGHT * mean_forward_velocity      # go forward
    - ENERGY_WEIGHT * mean_squared_command         # effort proxy
    - SMOOTHNESS_WEIGHT * mean_ctrl_change         # don't twitch
    - HEIGHT_PENALTY_WEIGHT * mean_height_violation # don't fall
    + ALIVE_BONUS * n_alive_steps                  # survive
    - DRIFT_WEIGHT * abs(final_y)                  # go straight
)
```

Each component addresses a specific failure mode:

| Component | What it prevents | Typical weight |
|-----------|-----------------|----------------|
| `mean_forward_velocity` | Standing still | 1.0 |
| `mean_squared_command` | Large commands (proxy) | 0.005 |
| `mean_ctrl_change` | Twitching, bang-bang control | 0.1 |
| `mean_height_violation` | Falling over | 5.0 |
| `n_alive_steps` | Dying early (falling and not recovering) | 0.1 |
| `abs(final_y)` | Circling or veering off course | 0.3 |

## Common Failure Modes and Fixes

### "The Leap"
**Symptom:** Robot jumps forward, lands, doesn't move again.
**Possible causes:** Reward loophole, penetrating reset, or large initial forces.
**Check:** Reset/contact dynamics and time-resolved progress. Mean velocity alone
does not solve this at a fixed horizon; validate any added health/contact gate.

### "The Flail"
**Symptom:** All joints move at maximum speed in random directions.
**Cause:** No energy penalty.
**Fix:** Add `- ENERGY_WEIGHT * mean(ctrl²)`.

### "The Twitch"
**Symptom:** Robot vibrates in place or makes tiny spastic movements.
**Cause:** No smoothness penalty, or energy penalty too high.
**Fix:** Add smoothness penalty, reduce energy penalty slightly.

### "The Statue"
**Symptom:** Robot stands perfectly still.
**Cause:** Energy penalty too high relative to velocity reward.
**Fix:** Reduce `ENERGY_WEIGHT`, increase `VELOCITY_WEIGHT`.

### "The Spinner"
**Symptom:** Robot rotates in circles.
**Cause:** No lateral drift penalty.
**Fix:** Add `- DRIFT_WEIGHT * abs(final_y)`.

### "The Scooter"
**Symptom:** Robot slides forward on its belly without using legs.
**Cause:** Height penalty too low, or belly has low friction.
**Fix:** Increase `MIN_TORSO_HEIGHT`, or add contact reward for feet.

### "The Bunny Hop"
**Symptom:** Robot bounces forward using all four legs simultaneously.
**Possible cause:** The learned policy synchronizes contacts because the objective
rewards that behavior. A shared clock does not force all output joints in phase.
**Check:** Contact phase relationships and task criteria; compare controller
structure and genuinely different frequencies under controlled ablations.

## Advanced: Multi-Frequency Clocks

A single 2Hz sine/cosine clock gives the controller a sense of rhythm,
but all four legs receive the same signal. Real quadruped gaits have
specific phase relationships:

| Gait | Phase pattern | Speed |
|------|--------------|-------|
| Walk | FL→HR→FR→HL (each 90° apart) | Slow |
| Trot | Diagonal pairs in phase (FL+HR, FR+HL) | Medium |
| Bound | Front pair in phase, rear pair in phase | Fast |
| Gallop | FL→FR→HL→HR (sequential) | Very fast |

You can encourage specific gaits by providing multiple clock signals at
different phases:

```python
# 4 phase-offset clocks (one per leg pair)
for i, phase_offset in enumerate([0, π/2, π, 3π/2]):
    clock_sin_i = sin(2π * freq * t + phase_offset)
    clock_cos_i = cos(2π * freq * t + phase_offset)
```

For a fully connected controller, shifted sine/cosine pairs at the same frequency
are linear combinations of the original pair. They do not add independent timing
information. Different frequencies or explicit leg-specific connectivity can
change the representation. Any learning-speed benefit from redundant phase
channels needs an ablation controlling parameter count and initialization.

## Comparison: Our Experiments

| Experiment | Fitness function | Best score | Behavior |
|-----------|-----------------|-----------|----------|
| 05 (distance) | x_dist - height_pen - y_drift | 11.27 | Leap and flail |
| 09 (velocity) | velocity - energy - smooth + alive - drift | TBD | Expected: coordinated gait |
| 10 (Bittle gait) | same as 09, position ctrl, phase clocks | TBD | Bittle-specific gait for sim2real |
| 11 (domain rand) | same as 10 + sensor/action noise | TBD | Robust policy for real hardware |

The absolute fitness numbers are not comparable across fitness functions
(they measure different things). What matters is the *behavior*.

## Further Reading

- Brax `ant.py` reward: [source](https://github.com/google/brax/blob/main/brax/envs/ant.py) — the standard formulation
- Tunyasuvunakool et al. (2020) "dm_control: Software and Tasks for Continuous Control" — DeepMind's locomotion rewards
- Heess et al. (2017) "Emergence of Locomotion Behaviours in Rich Environments" — velocity + energy + alive
- MuJoCo Gymnasium `Ant-v5` reward breakdown: [docs](https://gymnasium.farama.org/environments/mujoco/ant/)

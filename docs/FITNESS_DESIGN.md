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

### 1. Reward Velocity, Not Distance

```python
# BAD: rewards a single leap
fitness = final_x_position

# GOOD: rewards sustained forward movement
fitness = mean(forward_velocity_at_each_timestep)
```

Distance rewards a single burst of speed. Velocity rewards *consistent* forward
movement across the entire simulation. A robot that walks at 0.5 m/s for 8
seconds scores the same as one that runs at 0.5 m/s — but a robot that leaps
2m in the first second and sits still for 7 seconds scores much lower (mean
velocity ≈ 0.25 m/s).

This single change is the most important fix.

### 2. Penalize Energy (No Free Flailing)

```python
energy_cost = mean(sum(ctrl ** 2))  # squared torque at each timestep
fitness -= ENERGY_WEIGHT * energy_cost
```

Without an energy penalty, the optimizer discovers that maxing out all motors
all the time produces unpredictable chaotic motion — and sometimes that chaos
happens to go forward. This is "flailing."

The energy penalty makes large torques expensive. The optimizer must find
*efficient* movements — which turn out to be rhythmic gaits, because gaits
exploit the pendulum dynamics of swinging legs rather than fighting them.

**How to tune the weight:** Start with `ENERGY_WEIGHT = 0.005`. If the robot
barely moves, reduce it. If it still flails, increase it. The energy penalty
and velocity reward must be balanced — too much energy penalty produces a robot
that stands still (minimum energy = zero torque).

### 3. Penalize Jerk (Smoothness Matters)

```python
ctrl_change = sum((ctrl_t - ctrl_{t-1}) ** 2)
fitness -= SMOOTHNESS_WEIGHT * mean(ctrl_change)
```

Even with an energy penalty, the optimizer can produce rapid alternating
commands (bang-bang control) that are technically low-mean-energy but
physically violent. The smoothness penalty makes rapid control changes
expensive, forcing the controller to produce smooth, gradual commands.

This is what turns twitching into walking. Smooth control naturally produces
sinusoidal joint trajectories — which is exactly what a gait looks like.

## The Full Fitness Function

```python
fitness = (
    VELOCITY_WEIGHT * mean_forward_velocity      # go forward
    - ENERGY_WEIGHT * mean_squared_torque          # don't flail
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
| `mean_squared_torque` | Flailing, chaotic motion | 0.005 |
| `mean_ctrl_change` | Twitching, bang-bang control | 0.1 |
| `mean_height_violation` | Falling over | 5.0 |
| `n_alive_steps` | Dying early (falling and not recovering) | 0.1 |
| `abs(final_y)` | Circling or veering off course | 0.3 |

## Common Failure Modes and Fixes

### "The Leap"
**Symptom:** Robot jumps forward, lands, doesn't move again.
**Cause:** Distance reward, no velocity reward.
**Fix:** Replace `x_distance` with `mean(x_velocity)`.

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
**Cause:** All legs are in phase (no phase diversity in clock signal).
**Fix:** Add per-leg phase offsets to the clock signal, or increase
the number of clock frequencies.

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

This gives the controller enough information to produce trotting or walking
gaits. Without phase offsets, the controller must discover these relationships
purely from sensor feedback — which is possible but much slower.

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

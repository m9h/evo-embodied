# Sim2Real: From MuJoCo to a Walking Robot on Your Desk

This guide covers deploying evolved/trained controllers from simulation
to a real Petoi Bittle quadruped robot (~$300).

## Hardware: What to Buy

| Item | Price | Why |
|------|-------|-----|
| **Petoi Bittle X V2** | $299 | Feedback servos (reads actual joint angles), ESP32 WiFi/BT |
| Alloy Servo Set (optional) | $150 | More durable, less backlash — recommended for repeated experiments |
| USB-C cable | — | For serial communication (included with Bittle) |

**Total: $299-450**

The **feedback servos** (Bittle X V2, post-May 2024) are critical — they let the
robot report its actual joint positions back to your controller, closing the
control loop. Without feedback, you're running open-loop and the sim2real gap
is much larger.

### Alternatives

| Robot | Price | Pros | Cons |
|-------|-------|------|------|
| Petoi Bittle X V2 | $299 | Cheapest with feedback servos, proven sim2real | 8 DOF (no abduction) |
| RealAnt | ~$410 DIY | Purpose-built for RL, MuJoCo model included | Requires assembly, external tracking |
| Mini Pupper 2 | $649 | 12 DOF, ROS2, camera | More complex setup |
| 3D-printed quadruped | ~$150 | Cheapest possible | No feedback servos, fragile |
| Unitree Go2 Air | $1,600 | Professional quality | SDK locked — can't deploy custom policies! |
| Unitree Go2 EDU | $12,000+ | Gold standard sim2real | Way too expensive for classroom |

## The MuJoCo Bittle Model

`models/bittle/bittle.xml` is a hand-authored MJCF model of the Bittle with:

- **8 position-controlled actuators** matching the real servo layout
- **IMU sensor** (accelerometer + gyro) matching the MPU6050
- **Joint position sensors** matching feedback servo readings
- **Foot contact sensors** for reward shaping (virtual — not on real robot)
- **Servo dynamics** modeled as first-order filter (40ms time constant)
- Measured masses and inertias from the community URDF

### Joint Map

```
          FRONT
    FL (8,12)  FR (9,13)
         ┌──────┐
         │BITTLE│
         └──────┘
    BL (11,15) BR (10,14)
          BACK

Petoi joint indices: shoulder, knee
MuJoCo actuators: servo_shoulder_XX, servo_knee_XX
```

## Pipeline: Sim → Real

### Step 1: Train in Simulation

Use any of the training scripts with the Bittle model:

```bash
# Evolve a walking gait (evolutionary approach)
uv run python examples/09_coordinated_gait.py --output-dir /data/evo-embodied

# Or train with PPO (RL approach, faster convergence)
# (requires modifying to use bittle.xml instead of quadruped.xml)
```

The key difference from the generic quadruped: the Bittle model uses
**position actuators** (not torque), matching how real hobby servos work.
The controller outputs target joint angles, not torques.

### Step 2: Test in Simulation

```bash
# Render video of trained controller
uv run python experiments/render_from_weights.py /data/evo-embodied/YYYYMMDD/
```

Watch the video. Does it look like walking? If it's leaping or flailing,
see `docs/FITNESS_DESIGN.md` for fixes.

### Step 3: Deploy to Real Bittle

```bash
# Plug in Bittle via USB
# List serial ports
uv run python sim2real/deploy_bittle.py --list-ports

# Dry run (see commands without sending to robot)
uv run python sim2real/deploy_bittle.py \
    --weights /data/evo-embodied/YYYYMMDD/best_weights.npy \
    --dry-run

# Deploy for real (30 seconds, 20 Hz control)
uv run python sim2real/deploy_bittle.py \
    --weights /data/evo-embodied/YYYYMMDD/best_weights.npy \
    --port /dev/ttyUSB0 \
    --duration 30
```

### Step 4: Debug the Reality Gap

The first deployment will probably not walk well. This is normal and is the
most educational part. Common issues:

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Robot vibrates/oscillates | Servo PD gains too high in sim | Reduce `kp` in bittle.xml, retrain |
| Robot falls immediately | Mass/inertia mismatch | Weigh your actual Bittle, update XML |
| Legs move but no traction | Friction too high in sim | Reduce ground friction in XML |
| Gait is too fast | Control freq mismatch | Match `--ctrl-hz` to training rate (25 Hz) |
| One leg doesn't move | Servo offset wrong | Calibrate with `--dry-run`, check joint map |
| Robot walks but drifts | No drift penalty in training | Add `y_drift` penalty, retrain |

## Domain Randomization

The most effective technique for closing the sim2real gap is training with
randomized dynamics so the policy is robust to real-world variation.

Add these to your training loop:

```python
# Randomize per-episode
friction = default_friction * jax.random.uniform(key, (), 0.5, 1.5)
servo_delay = jax.random.uniform(key, (), 0.02, 0.06)  # 20-60ms
mass_scale = jax.random.uniform(key, (), 0.8, 1.2)
imu_noise = jax.random.normal(key, (3,)) * 0.05  # rad
joint_noise = jax.random.normal(key, (8,)) * 0.02  # rad
```

The idea: if the policy works across a wide range of simulated dynamics,
it's more likely to work on the real robot (which has specific but unknown
dynamics within that range).

## Sim2Real Checklist

Before deploying, verify:

- [ ] **Position control**: Actuators output target angles (not torques)
- [ ] **Control frequency**: Training and deployment use the same rate (20-25 Hz)
- [ ] **Joint limits**: Sim limits match real servo range (-125° to +125°)
- [ ] **Joint map**: MuJoCo actuator indices → Petoi servo indices are correct
- [ ] **Servo offsets**: Calibrate zero position for each joint
- [ ] **Safety**: Set conservative joint limits, start with slow/small motions
- [ ] **Power**: Bittle battery charged, servos not overheating
- [ ] **Catch the robot**: Have a hand ready to catch it when it falls

## Existing Work

These projects have successfully done Bittle sim2real:

- [opencat-gym](https://github.com/ger01d/opencat-gym) — PyBullet + SB3 PPO
- [opencat-gym-sim2real](https://github.com/ger01d/opencat-gym-sim2real) — deployment code
- [Bittle URDF](https://github.com/AIWintermuteAI/Bittle_URDF) — measured robot model
- [MuJoCo MPC Bittle](https://github.com/gravesreid/autonomous-bittle) — model predictive control
- [Jabbour et al. RSS 2022](https://a2r-lab.org/publication/bittlesim2real/) — academic paper on Bittle sim2real
- [Akgun et al. 2024](https://arxiv.org/html/2402.13201) — Decision Transformers on Bittle

## Further Reading

- [Petoi Serial Protocol](https://docs.petoi.com/apis/serial-protocol)
- [Petoi Joint Index](https://docs.petoi.com/petoi-robot-joint-index)
- [Petoi Python API](https://docs.petoi.com/apis/python-api)
- [Feedback Servo Docs](https://docs.petoi.com/apis/serial-protocol/feedback-servos)
- [sim2real overview](https://www.reinforcementlearningpath.com/sim2real)

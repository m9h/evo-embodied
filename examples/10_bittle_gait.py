"""Evolve a walking gait on the Petoi Bittle model (sim2real ready).

Key differences from the generic quadruped (09_coordinated_gait.py):

1. POSITION CONTROL — the Bittle uses hobby servos that accept target angles,
   not torques. The controller outputs desired joint angles, not forces.

2. PHASE-OFFSET CLOCKS — four clock signals (one per leg) with 90-degree
   phase offsets. This gives the controller enough information to produce
   a trot or walk without discovering the phase relationship from scratch.

3. TINY ROBOT — the Bittle weighs 177g and stands 9cm tall. Fitness weights,
   height thresholds, and mutation scales are tuned for this scale.

4. RICH SENSORS — IMU (accel + gyro + quat), 8 joint positions, 4 foot
   contacts, torso pose. 28 sensor values vs. 11 on the generic quadruped.

This script produces a controller suitable for sim2real transfer (Assignment 18).
Train here, then deploy with sim2real/deploy_bittle.py.

Run: uv run python examples/10_bittle_gait.py --output-dir /data/evo-embodied
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from mujoco import mjx

try:
    import mediapy as media
    HAS_MEDIAPY = True
except ImportError:
    HAS_MEDIAPY = False

MODEL_PATH = Path(__file__).parent.parent / "models" / "bittle" / "bittle.xml"

# ── Config ──────────────────────────────────────────────────────────
N_POPULATION = 256
N_GENERATIONS = 500
N_HIDDEN = 32             # smaller network for 8-DOF robot
CONTROL_STEPS = 250       # 250 control decisions = 10 seconds at 25 Hz
PHYSICS_PER_CTRL = 20     # 25 Hz control (timestep=0.002)
MUTATION_SCALE = 0.08
MUTATION_DECAY = 0.9995
SEED = 42

# Fitness weights — tuned for the Bittle's scale
VELOCITY_WEIGHT = 2.0       # must be strong to overcome penalties at this scale
ENERGY_WEIGHT = 0.01        # penalize squared torques (position control = smaller values)
SMOOTHNESS_WEIGHT = 0.05    # penalize jerky angle changes
HEIGHT_PENALTY_WEIGHT = 10.0  # penalize torso below threshold
ALIVE_BONUS = 0.05         # per-step reward for staying upright
DRIFT_WEIGHT = 0.5         # penalize lateral drift
MIN_TORSO_HEIGHT = 0.05    # Bittle torso starts at 0.12m, legs ~0.08m
CLOCK_FREQ_HZ = 2.0        # gait oscillator frequency

# Phase offsets for four legs (walk gait: each leg 90 degrees apart)
#   FL=0, FR=pi/2, BL=pi, BR=3*pi/2
LEG_PHASE_OFFSETS = [0.0, jnp.pi / 2, jnp.pi, 3 * jnp.pi / 2]
N_CLOCK_INPUTS = len(LEG_PHASE_OFFSETS) * 2  # sin + cos per leg = 8

# Position control: scale network output to servo range
CTRL_SCALE = 1.5  # tanh * 1.5 covers approx [-1.5, 1.5] rad servo range

RENDER_CTRL_STEPS = 375   # 15 seconds of video at 25 Hz
RENDER_FPS = 30


def make_run_dir(base_dir):
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base_dir) / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(run_dir):
    config = {
        "experiment": "bittle_gait",
        "model": "bittle/bittle.xml",
        "actuator_type": "position (target angles, not torques)",
        "n_population": N_POPULATION,
        "n_generations": N_GENERATIONS,
        "n_hidden": N_HIDDEN,
        "control_steps": CONTROL_STEPS,
        "physics_per_ctrl": PHYSICS_PER_CTRL,
        "control_freq_hz": 1.0 / (0.002 * PHYSICS_PER_CTRL),
        "sim_duration_s": CONTROL_STEPS * PHYSICS_PER_CTRL * 0.002,
        "mutation_scale": MUTATION_SCALE,
        "mutation_decay": MUTATION_DECAY,
        "ctrl_scale": CTRL_SCALE,
        "clock_freq_hz": CLOCK_FREQ_HZ,
        "leg_phase_offsets": ["0", "pi/2", "pi", "3*pi/2"],
        "n_clock_inputs": N_CLOCK_INPUTS,
        "fitness_weights": {
            "velocity": VELOCITY_WEIGHT,
            "energy": ENERGY_WEIGHT,
            "smoothness": SMOOTHNESS_WEIGHT,
            "height_penalty": HEIGHT_PENALTY_WEIGHT,
            "alive_bonus": ALIVE_BONUS,
            "drift": DRIFT_WEIGHT,
        },
        "min_torso_height": MIN_TORSO_HEIGHT,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    (run_dir / "experiment.json").write_text(
        json.dumps({"experiment": "bittle_gait"}))
    return config


def build_model():
    mj_model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    mjx_model = mjx.put_model(mj_model)
    return mj_model, mjx_model


def make_clock_inputs(t):
    """Phase-offset clock signals: sin/cos per leg for walk gait."""
    clocks = []
    for phase in LEG_PHASE_OFFSETS:
        clocks.append(jnp.sin(2.0 * jnp.pi * CLOCK_FREQ_HZ * t + phase))
        clocks.append(jnp.cos(2.0 * jnp.pi * CLOCK_FREQ_HZ * t + phase))
    return jnp.array(clocks)


def make_evaluate_fn(mjx_model, n_sensors, n_motors):
    n_inputs = n_sensors + N_CLOCK_INPUTS
    n_w = n_inputs * N_HIDDEN + N_HIDDEN * n_motors

    @jax.jit
    def evaluate_one(weights_flat, mjx_model):
        w1 = weights_flat[:n_inputs * N_HIDDEN].reshape(n_inputs, N_HIDDEN)
        w2 = weights_flat[n_inputs * N_HIDDEN:].reshape(N_HIDDEN, n_motors)

        mjx_data = mjx.make_data(mjx_model)
        dt = mjx_model.opt.timestep
        ctrl_dt = dt * PHYSICS_PER_CTRL

        def physics_step(data, _):
            return mjx.step(mjx_model, data), None

        def control_step(carry, ctrl_idx):
            data, prev_ctrl = carry

            t = ctrl_idx * ctrl_dt
            clock = make_clock_inputs(t)

            sensor_input = jnp.concatenate([data.sensordata, clock])
            hidden = jnp.tanh(sensor_input @ w1)
            # Position control: output target joint angles
            ctrl = jnp.tanh(hidden @ w2) * CTRL_SCALE

            data = data.replace(ctrl=ctrl)
            data, _ = jax.lax.scan(physics_step, data, None,
                                   length=PHYSICS_PER_CTRL)

            torso_z = data.qpos[2]
            x_vel = data.qvel[0]
            energy = jnp.sum(ctrl ** 2)
            ctrl_change = jnp.sum((ctrl - prev_ctrl) ** 2)
            alive = (torso_z > MIN_TORSO_HEIGHT).astype(jnp.float32)

            metrics = jnp.array([x_vel, energy, ctrl_change, torso_z, alive])
            return (data, ctrl), metrics

        init_ctrl = jnp.zeros(n_motors)
        (final_data, _), all_metrics = jax.lax.scan(
            control_step,
            (mjx_data, init_ctrl),
            jnp.arange(CONTROL_STEPS),
        )

        x_vels = all_metrics[:, 0]
        energies = all_metrics[:, 1]
        ctrl_changes = all_metrics[:, 2]
        torso_heights = all_metrics[:, 3]
        alive_steps = all_metrics[:, 4]

        mean_fwd_vel = jnp.mean(x_vels)
        mean_energy = jnp.mean(energies)
        mean_smoothness = jnp.mean(ctrl_changes)
        height_violations = jnp.maximum(0.0, MIN_TORSO_HEIGHT - torso_heights)
        mean_height_pen = jnp.mean(height_violations)
        total_alive = jnp.sum(alive_steps)
        y_drift = jnp.abs(final_data.qpos[1])

        fitness = (
            VELOCITY_WEIGHT * mean_fwd_vel
            - ENERGY_WEIGHT * mean_energy
            - SMOOTHNESS_WEIGHT * mean_smoothness
            - HEIGHT_PENALTY_WEIGHT * mean_height_pen
            + ALIVE_BONUS * total_alive
            - DRIFT_WEIGHT * y_drift
        )
        return fitness

    evaluate_batch = jax.jit(jax.vmap(evaluate_one, in_axes=(0, None)))
    return evaluate_batch, n_w


def evolve(mjx_model, evaluate_batch, n_w, run_dir):
    key = jax.random.PRNGKey(SEED)
    key, init_key = jax.random.split(key)
    population = jax.random.normal(init_key, (N_POPULATION, n_w)) * 0.3

    print("JIT compiling...", flush=True)
    t0 = time.time()
    fitnesses = evaluate_batch(population, mjx_model)
    fitnesses.block_until_ready()
    jit_time = time.time() - t0
    print(f"JIT done in {jit_time:.1f}s\n", flush=True)

    best_history, mean_history = [], []
    mutation_scale = MUTATION_SCALE

    print(f"{'Gen':>5s}  {'Best':>8s}  {'Mean':>8s}  {'Mut':>8s}", flush=True)
    print("-" * 36, flush=True)

    t_start = time.time()
    for gen in range(N_GENERATIONS):
        key, mk = jax.random.split(key)
        candidates = population + jax.random.normal(mk, population.shape) * mutation_scale
        cand_fit = evaluate_batch(candidates, mjx_model)

        improved = cand_fit > fitnesses
        population = jnp.where(improved[:, None], candidates, population)
        fitnesses = jnp.where(improved, cand_fit, fitnesses)

        gb, gm = float(fitnesses.max()), float(fitnesses.mean())
        best_history.append(gb)
        mean_history.append(gm)
        mutation_scale *= MUTATION_DECAY

        if (gen + 1) % 50 == 0 or gen == 0:
            print(f"{gen+1:5d}  {gb:+8.4f}  {gm:+8.4f}  {mutation_scale:8.5f}", flush=True)

        if (gen + 1) % 100 == 0:
            best_idx = int(jnp.argmax(fitnesses))
            np.save(run_dir / f"weights_gen{gen+1:04d}.npy",
                    np.array(population[best_idx]))

    elapsed = time.time() - t_start
    best_idx = int(jnp.argmax(fitnesses))
    best_weights = np.array(population[best_idx])
    total_sims = N_POPULATION * N_GENERATIONS

    print(f"\nEvolution: {elapsed:.1f}s, {total_sims:,} sims ({total_sims/elapsed:.0f}/sec)",
          flush=True)
    print(f"Best fitness: {float(fitnesses[best_idx]):+.4f}", flush=True)

    np.save(run_dir / "best_weights.npy", best_weights)
    np.save(run_dir / "population.npy", np.array(population))
    np.save(run_dir / "fitnesses.npy", np.array(fitnesses))
    np.savez(run_dir / "history.npz",
             best=np.array(best_history), mean=np.array(mean_history))

    summary = {
        "jit_time_s": round(jit_time, 1),
        "evolution_time_s": round(elapsed, 1),
        "total_simulations": total_sims,
        "sims_per_sec": round(total_sims / elapsed),
        "best_fitness": round(float(fitnesses[best_idx]), 4),
        "mean_fitness": round(float(fitnesses.mean()), 4),
        "jax_backend": str(jax.default_backend()),
        "jax_devices": [str(d) for d in jax.devices()],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Results saved to {run_dir}/", flush=True)

    # RESULT line for adapter integration
    print(f"RESULT|fitness={summary['best_fitness']}|sims_per_sec={summary['sims_per_sec']}|",
          flush=True)

    return best_weights, best_history, mean_history


def plot_fitness(best_hist, mean_hist, run_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(best_hist, label="Best", linewidth=2)
    ax.plot(mean_hist, label="Mean", linewidth=1, alpha=0.7)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (velocity - energy - smoothness + alive)")
    ax.set_title(f"Bittle Gait — pop={N_POPULATION}, gens={N_GENERATIONS}, "
                 f"h={N_HIDDEN}, phase-offset clocks, position ctrl")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(run_dir / "fitness_curve.png", dpi=150)
    plt.close(fig)
    print(f"Saved {run_dir / 'fitness_curve.png'}", flush=True)


def render_video(mj_model, best_weights, n_sensors, n_motors, run_dir):
    if not HAS_MEDIAPY:
        print("Skipping video: mediapy not installed", flush=True)
        return

    n_inputs = n_sensors + N_CLOCK_INPUTS
    w1 = best_weights[:n_inputs * N_HIDDEN].reshape(n_inputs, N_HIDDEN)
    w2 = best_weights[n_inputs * N_HIDDEN:].reshape(N_HIDDEN, n_motors)

    data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, data)

    try:
        renderer = mujoco.Renderer(mj_model, height=720, width=1280)
    except Exception as e:
        print(f"Renderer failed: {e}", flush=True)
        return

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    cam.distance = 0.6   # closer camera for tiny robot
    cam.azimuth = 150
    cam.elevation = -25

    dt = mj_model.opt.timestep
    ctrl_dt = dt * PHYSICS_PER_CTRL
    steps_per_frame = max(1, int(1.0 / (RENDER_FPS * dt)))

    frames = []
    physics_step = 0
    prev_ctrl = np.zeros(n_motors)

    print(f"Rendering {RENDER_CTRL_STEPS} ctrl steps...", flush=True)
    for ctrl_i in range(RENDER_CTRL_STEPS):
        t = ctrl_i * ctrl_dt

        # Phase-offset clocks (numpy version for CPU rendering)
        clock = []
        for phase in [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]:
            clock.append(np.sin(2.0 * np.pi * CLOCK_FREQ_HZ * t + phase))
            clock.append(np.cos(2.0 * np.pi * CLOCK_FREQ_HZ * t + phase))
        clock = np.array(clock)

        sensor_input = np.concatenate([data.sensordata.copy(), clock])
        hidden = np.tanh(sensor_input @ w1)
        ctrl = np.tanh(hidden @ w2) * CTRL_SCALE
        data.ctrl[:] = ctrl
        prev_ctrl = ctrl

        for _ in range(PHYSICS_PER_CTRL):
            mujoco.mj_step(mj_model, data)
            if physics_step % steps_per_frame == 0:
                renderer.update_scene(data, cam)
                frames.append(renderer.render().copy())
            physics_step += 1

    renderer.close()

    video_path = run_dir / "bittle_gait.mp4"
    media.write_video(str(video_path), frames, fps=RENDER_FPS)
    print(f"Saved {video_path} ({len(frames)} frames, "
          f"{video_path.stat().st_size / 1024:.0f} KB)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evolve a walking gait on the Petoi Bittle model")
    parser.add_argument("--output-dir", default="/data/evo-embodied")
    args = parser.parse_args()

    run_dir = make_run_dir(args.output_dir)
    print(f"Run: {run_dir}", flush=True)
    print(f"JAX: {jax.default_backend()}, {jax.devices()}\n", flush=True)
    save_config(run_dir)

    mj_model, mjx_model = build_model()
    n_s, n_m = mj_model.nsensordata, mj_model.nu
    print(f"Bittle: {n_s} sensors, {n_m} motors (position control)", flush=True)
    print(f"  Actuator type: position (target angles, not torques)", flush=True)
    print(f"  Phase-offset clocks: {N_CLOCK_INPUTS} inputs "
          f"(4 legs x sin/cos)", flush=True)

    evaluate_batch, n_w = make_evaluate_fn(mjx_model, n_s, n_m)
    print(f"Network: {n_s}+{N_CLOCK_INPUTS} → {N_HIDDEN} → {n_m} "
          f"({n_w} weights)", flush=True)
    print(f"\nFitness = {VELOCITY_WEIGHT}*velocity "
          f"- {ENERGY_WEIGHT}*energy "
          f"- {SMOOTHNESS_WEIGHT}*smoothness "
          f"- {HEIGHT_PENALTY_WEIGHT}*height_pen "
          f"+ {ALIVE_BONUS}*alive "
          f"- {DRIFT_WEIGHT}*drift\n", flush=True)

    best_weights, best_hist, mean_hist = evolve(
        mjx_model, evaluate_batch, n_w, run_dir)
    plot_fitness(best_hist, mean_hist, run_dir)
    render_video(mj_model, best_weights, n_s, n_m, run_dir)

    print(f"\nDone! Results: {run_dir}/", flush=True)
    print(f"Next: deploy to real Bittle with sim2real/deploy_bittle.py", flush=True)
    print(f"  Or: add domain randomization with examples/11_domain_randomization.py",
          flush=True)

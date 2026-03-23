"""Evolve a coordinated quadruped gait (not a leap-and-flail).

The key insight: reward VELOCITY not DISTANCE, and penalize everything
that isn't smooth, efficient locomotion.

Fitness = forward_velocity
        - energy_cost        (penalize large torques → no flailing)
        - action_smoothness  (penalize jerky control changes)
        - height_penalty     (must stay upright)
        - alive_bonus        (per-step reward for not falling)

This is the standard formulation from DeepMind's locomotion work and
Brax's ant/humanoid environments.

Run: uv run python examples/09_coordinated_gait.py --output-dir /data/evo-embodied
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

MODEL_PATH = Path(__file__).parent.parent / "models" / "quadruped.xml"

# ── Config ──────────────────────────────────────────────────────────
N_POPULATION = 256
N_GENERATIONS = 500
N_HIDDEN = 64
CONTROL_STEPS = 200       # 200 control decisions = 8 seconds
PHYSICS_PER_CTRL = 20     # 25 Hz control
MUTATION_SCALE = 0.1
MUTATION_DECAY = 0.9995
SEED = 42

# Fitness weights — the key to getting gaits instead of leaps
VELOCITY_WEIGHT = 1.0       # reward forward velocity (m/s)
ENERGY_WEIGHT = 0.005       # penalize sum of squared torques
SMOOTHNESS_WEIGHT = 0.1     # penalize |ctrl_t - ctrl_{t-1}|
HEIGHT_PENALTY_WEIGHT = 5.0 # penalize torso below threshold
ALIVE_BONUS = 0.1           # per-step reward for staying upright
DRIFT_WEIGHT = 0.3          # penalize lateral drift
MIN_TORSO_HEIGHT = 0.2      # below this = falling
CLOCK_FREQ_HZ = 2.0         # gait oscillator frequency

RENDER_CTRL_STEPS = 300     # 12 seconds of video
RENDER_FPS = 30


def make_run_dir(base_dir):
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base_dir) / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(run_dir):
    config = {
        "experiment": "coordinated_gait",
        "n_population": N_POPULATION,
        "n_generations": N_GENERATIONS,
        "n_hidden": N_HIDDEN,
        "control_steps": CONTROL_STEPS,
        "physics_per_ctrl": PHYSICS_PER_CTRL,
        "control_freq_hz": 1.0 / (0.002 * PHYSICS_PER_CTRL),
        "sim_duration_s": CONTROL_STEPS * PHYSICS_PER_CTRL * 0.002,
        "mutation_scale": MUTATION_SCALE,
        "mutation_decay": MUTATION_DECAY,
        "fitness_weights": {
            "velocity": VELOCITY_WEIGHT,
            "energy": ENERGY_WEIGHT,
            "smoothness": SMOOTHNESS_WEIGHT,
            "height_penalty": HEIGHT_PENALTY_WEIGHT,
            "alive_bonus": ALIVE_BONUS,
            "drift": DRIFT_WEIGHT,
        },
        "min_torso_height": MIN_TORSO_HEIGHT,
        "clock_freq_hz": CLOCK_FREQ_HZ,
        "fitness": ("velocity_weight * mean_forward_vel "
                    "- energy_weight * mean_squared_torque "
                    "- smoothness_weight * mean_ctrl_change "
                    "- height_penalty_weight * mean_height_violation "
                    "+ alive_bonus * n_alive_steps "
                    "- drift_weight * abs(final_y)"),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    (run_dir / "experiment.json").write_text(
        json.dumps({"experiment": "coordinated_gait"}))
    return config


def build_model():
    mj_model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    mjx_model = mjx.put_model(mj_model)
    return mj_model, mjx_model


def make_evaluate_fn(mjx_model, n_sensors, n_motors):
    n_inputs = n_sensors + 2  # sensors + sin/cos clock
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

            # Clock signal
            t = ctrl_idx * ctrl_dt
            clock_sin = jnp.sin(2.0 * jnp.pi * CLOCK_FREQ_HZ * t)
            clock_cos = jnp.cos(2.0 * jnp.pi * CLOCK_FREQ_HZ * t)

            # Neural network controller
            sensor_input = jnp.concatenate([
                data.sensordata,
                jnp.array([clock_sin, clock_cos])
            ])
            ctrl = jnp.tanh(jnp.tanh(sensor_input @ w1) @ w2)
            data = data.replace(ctrl=ctrl)

            # Run physics
            data, _ = jax.lax.scan(physics_step, data, None,
                                   length=PHYSICS_PER_CTRL)

            # Collect per-step metrics
            torso_z = data.qpos[2]
            x_vel = data.qvel[0]  # forward velocity
            energy = jnp.sum(ctrl ** 2)  # squared torque proxy
            ctrl_change = jnp.sum((ctrl - prev_ctrl) ** 2)  # smoothness
            alive = (torso_z > MIN_TORSO_HEIGHT).astype(jnp.float32)

            metrics = jnp.array([x_vel, energy, ctrl_change, torso_z, alive])
            return (data, ctrl), metrics

        init_ctrl = jnp.zeros(n_motors)
        (final_data, _), all_metrics = jax.lax.scan(
            control_step,
            (mjx_data, init_ctrl),
            jnp.arange(CONTROL_STEPS),
        )

        # Unpack metrics: [ctrl_steps, 5]
        x_vels = all_metrics[:, 0]
        energies = all_metrics[:, 1]
        ctrl_changes = all_metrics[:, 2]
        torso_heights = all_metrics[:, 3]
        alive_steps = all_metrics[:, 4]

        # Fitness components
        # 1. Mean forward velocity (not distance!) — rewards sustained movement
        mean_fwd_vel = jnp.mean(x_vels)

        # 2. Energy cost — penalizes flailing
        mean_energy = jnp.mean(energies)

        # 3. Smoothness — penalizes jerky control changes
        mean_smoothness = jnp.mean(ctrl_changes)

        # 4. Height penalty — penalizes being below threshold
        height_violations = jnp.maximum(0.0, MIN_TORSO_HEIGHT - torso_heights)
        mean_height_pen = jnp.mean(height_violations)

        # 5. Alive bonus — per-step reward for staying upright
        total_alive = jnp.sum(alive_steps)

        # 6. Lateral drift
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

    return best_weights, best_history, mean_history


def plot_fitness(best_hist, mean_hist, run_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(best_hist, label="Best", linewidth=2)
    ax.plot(mean_hist, label="Mean", linewidth=1, alpha=0.7)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (velocity - energy - smoothness + alive)")
    ax.set_title(f"Coordinated Gait — pop={N_POPULATION}, gens={N_GENERATIONS}, "
                 f"h={N_HIDDEN}, {CONTROL_STEPS * PHYSICS_PER_CTRL * 0.002:.0f}s sim")
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

    n_inputs = n_sensors + 2
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
    cam.distance = 3.0
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
        clock = np.array([
            np.sin(2.0 * np.pi * CLOCK_FREQ_HZ * t),
            np.cos(2.0 * np.pi * CLOCK_FREQ_HZ * t),
        ])
        sensor_input = np.concatenate([data.sensordata.copy(), clock])
        ctrl = np.tanh(np.tanh(sensor_input @ w1) @ w2)
        data.ctrl[:] = ctrl
        prev_ctrl = ctrl

        for _ in range(PHYSICS_PER_CTRL):
            mujoco.mj_step(mj_model, data)
            if physics_step % steps_per_frame == 0:
                renderer.update_scene(data, cam)
                frames.append(renderer.render().copy())
            physics_step += 1

    renderer.close()

    video_path = run_dir / "video.mp4"
    media.write_video(str(video_path), frames, fps=RENDER_FPS)
    print(f"Saved {video_path} ({len(frames)} frames, "
          f"{video_path.stat().st_size / 1024:.0f} KB)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/data/evo-embodied")
    args = parser.parse_args()

    run_dir = make_run_dir(args.output_dir)
    print(f"Run: {run_dir}", flush=True)
    print(f"JAX: {jax.default_backend()}, {jax.devices()}\n", flush=True)
    save_config(run_dir)

    mj_model, mjx_model = build_model()
    n_s, n_m = mj_model.nsensordata, mj_model.nu
    print(f"Quadruped: {n_s} sensors, {n_m} motors", flush=True)

    evaluate_batch, n_w = make_evaluate_fn(mjx_model, n_s, n_m)
    print(f"Network: {n_s}+2 → {N_HIDDEN} → {n_m} ({n_w} weights)", flush=True)
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

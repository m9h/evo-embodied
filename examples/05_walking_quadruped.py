"""Evolve a quadruped that actually WALKS (not just falls forward).

Key improvements over 03/04:
1. Fitness = distance - penalty for low torso (no more "controlled collapse")
2. Control at 25 Hz (not 500 Hz) — locomotion needs slow, smooth commands
3. Oscillatory clock inputs — gives the controller a sense of rhythm
4. Longer evolution with adaptive mutation
5. All results saved to /data/evo-embodied/<run_id>/ for easy visualization

Run: uv run python examples/05_walking_quadruped.py
     uv run python examples/05_walking_quadruped.py --output-dir /data/evo-embodied
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapy as media
import mujoco
import numpy as np
from mujoco import mjx

# ── Config ──────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent.parent / "models" / "quadruped.xml"

N_POPULATION = 256
N_GENERATIONS = 500
N_HIDDEN = 16
CONTROL_STEPS = 100       # 100 control decisions per evaluation
PHYSICS_PER_CTRL = 20     # 20 physics steps per control step (= 25 Hz at dt=0.002)
MUTATION_SCALE = 0.1
MUTATION_DECAY = 0.9995   # slowly reduce mutation over generations
MIN_TORSO_HEIGHT = 0.25   # penalize below this height
RENDER_CTRL_STEPS = 200   # 200 control steps for video (8 seconds)
RENDER_FPS = 30


def make_run_dir(base_dir):
    """Create timestamped run directory."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base_dir) / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(run_dir):
    """Save experiment config as JSON."""
    config = {
        "n_population": N_POPULATION,
        "n_generations": N_GENERATIONS,
        "n_hidden": N_HIDDEN,
        "control_steps": CONTROL_STEPS,
        "physics_per_ctrl": PHYSICS_PER_CTRL,
        "control_freq_hz": 1.0 / (0.002 * PHYSICS_PER_CTRL),
        "sim_duration_s": CONTROL_STEPS * PHYSICS_PER_CTRL * 0.002,
        "mutation_scale": MUTATION_SCALE,
        "mutation_decay": MUTATION_DECAY,
        "min_torso_height": MIN_TORSO_HEIGHT,
        "model": "quadruped.xml",
        "fitness": "x_distance - 10*height_penalty - 0.5*y_drift",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    return config


def build_model():
    """Load MuJoCo and MJX models."""
    mj_model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    mjx_model = mjx.put_model(mj_model)
    return mj_model, mjx_model


def make_evolution_fns(mjx_model, n_sensors, n_motors):
    """Build JIT-compiled fitness evaluation with walking-specific improvements."""
    n_inputs = n_sensors + 2  # sensors + oscillatory clock
    n_w = n_inputs * N_HIDDEN + N_HIDDEN * n_motors

    @jax.jit
    def evaluate_one(weights_flat, mjx_model):
        w1 = weights_flat[:n_inputs * N_HIDDEN].reshape(n_inputs, N_HIDDEN)
        w2 = weights_flat[n_inputs * N_HIDDEN:].reshape(N_HIDDEN, n_motors)

        mjx_data = mjx.make_data(mjx_model)
        dt = mjx_model.opt.timestep
        ctrl_dt = dt * PHYSICS_PER_CTRL

        def physics_step(data, _):
            """Inner loop: run physics at full rate with held control."""
            data = mjx.step(mjx_model, data)
            return data, None

        def control_step(carry, ctrl_idx):
            """Outer loop: compute control at 25 Hz, then run physics."""
            data = carry

            t = ctrl_idx * ctrl_dt
            clock_sin = jnp.sin(2.0 * jnp.pi * 2.0 * t)
            clock_cos = jnp.cos(2.0 * jnp.pi * 2.0 * t)

            sensor_input = jnp.concatenate([
                data.sensordata,
                jnp.array([clock_sin, clock_cos])
            ])
            ctrl = jnp.tanh(jnp.tanh(sensor_input @ w1) @ w2)
            data = data.replace(ctrl=ctrl)

            data, _ = jax.lax.scan(physics_step, data, None,
                                   length=PHYSICS_PER_CTRL)

            torso_z = data.qpos[2]
            return data, torso_z

        final_data, torso_heights = jax.lax.scan(
            control_step,
            mjx_data,
            jnp.arange(CONTROL_STEPS),
        )

        x_distance = final_data.qpos[0]
        height_penalty = jnp.mean(
            jnp.maximum(0.0, MIN_TORSO_HEIGHT - torso_heights)
        ) * 10.0
        y_drift = jnp.abs(final_data.qpos[1]) * 0.5

        fitness = x_distance - height_penalty - y_drift
        return fitness

    evaluate_batch = jax.jit(jax.vmap(evaluate_one, in_axes=(0, None)))
    return evaluate_batch, n_w


def evolve(mjx_model, evaluate_batch, n_w, run_dir):
    """Parallel hill climber with adaptive mutation. Saves checkpoints."""
    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    population = jax.random.normal(init_key, (N_POPULATION, n_w)) * 0.3

    # JIT warmup
    print("JIT compiling (one-time cost)...", flush=True)
    t0 = time.time()
    fitnesses = evaluate_batch(population, mjx_model)
    fitnesses.block_until_ready()
    jit_time = time.time() - t0
    print(f"JIT done in {jit_time:.1f}s\n", flush=True)

    best_history = []
    mean_history = []
    mutation_scale = MUTATION_SCALE

    print(f"{'Gen':>5s}  {'Best':>8s}  {'Mean':>8s}  {'Mut σ':>8s}", flush=True)
    print("-" * 36, flush=True)

    t_start = time.time()
    for gen in range(N_GENERATIONS):
        key, mk = jax.random.split(key)
        mutations = jax.random.normal(mk, population.shape) * mutation_scale
        candidates = population + mutations
        cand_fit = evaluate_batch(candidates, mjx_model)

        improved = cand_fit > fitnesses
        population = jnp.where(improved[:, None], candidates, population)
        fitnesses = jnp.where(improved, cand_fit, fitnesses)

        gb = float(fitnesses.max())
        gm = float(fitnesses.mean())
        best_history.append(gb)
        mean_history.append(gm)

        mutation_scale *= MUTATION_DECAY

        if (gen + 1) % 50 == 0 or gen == 0:
            print(f"{gen+1:5d}  {gb:+8.4f}  {gm:+8.4f}  {mutation_scale:8.5f}", flush=True)

        # Checkpoint every 100 generations
        if (gen + 1) % 100 == 0:
            best_idx = int(jnp.argmax(fitnesses))
            np.save(run_dir / f"weights_gen{gen+1:04d}.npy",
                    np.array(population[best_idx]))

    elapsed = time.time() - t_start
    best_idx = int(jnp.argmax(fitnesses))
    best_weights = np.array(population[best_idx])
    total_sims = N_POPULATION * N_GENERATIONS

    print(f"\nEvolution: {elapsed:.1f}s, {total_sims:,} sims ({total_sims/elapsed:.0f}/sec)", flush=True)
    print(f"Best fitness: {float(fitnesses[best_idx]):+.4f}", flush=True)

    # Save final results
    np.save(run_dir / "best_weights.npy", best_weights)
    np.save(run_dir / "population.npy", np.array(population))
    np.save(run_dir / "fitnesses.npy", np.array(fitnesses))
    np.savez(run_dir / "history.npz",
             best=np.array(best_history),
             mean=np.array(mean_history))

    # Save summary
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
    """Save fitness curve."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(best_hist, label="Best", linewidth=2)
    ax.plot(mean_hist, label="Mean", linewidth=1, alpha=0.7)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (x_dist - height_penalty - y_drift)")
    ax.set_title(
        f"Walking Quadruped — {N_POPULATION} pop, {N_GENERATIONS} gen, "
        f"25 Hz control, oscillatory clock"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    fig.tight_layout()
    plot_path = run_dir / "fitness_curve.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved {plot_path}", flush=True)


def render_video(mj_model, best_weights, n_sensors, n_motors, run_dir):
    """Render the best evolved walker to MP4."""
    n_inputs = n_sensors + 2
    w1 = best_weights[:n_inputs * N_HIDDEN].reshape(n_inputs, N_HIDDEN)
    w2 = best_weights[n_inputs * N_HIDDEN:].reshape(N_HIDDEN, n_motors)

    data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, data)

    renderer = mujoco.Renderer(mj_model, height=720, width=1280)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso"
    )
    cam.distance = 3.0
    cam.azimuth = 150
    cam.elevation = -25

    dt = mj_model.opt.timestep
    ctrl_dt = dt * PHYSICS_PER_CTRL
    steps_per_frame = max(1, int(1.0 / (RENDER_FPS * dt)))

    frames = []
    total_physics_steps = RENDER_CTRL_STEPS * PHYSICS_PER_CTRL
    sim_time = total_physics_steps * dt

    print(f"Rendering {RENDER_CTRL_STEPS} control steps ({sim_time:.1f}s)...", flush=True)
    physics_step = 0
    for ctrl_i in range(RENDER_CTRL_STEPS):
        t = ctrl_i * ctrl_dt
        clock = np.array([
            np.sin(2.0 * np.pi * 2.0 * t),
            np.cos(2.0 * np.pi * 2.0 * t),
        ])
        sensor_input = np.concatenate([data.sensordata.copy(), clock])
        ctrl = np.tanh(np.tanh(sensor_input @ w1) @ w2)
        data.ctrl[:] = ctrl

        for _ in range(PHYSICS_PER_CTRL):
            mujoco.mj_step(mj_model, data)
            if physics_step % steps_per_frame == 0:
                renderer.update_scene(data, cam)
                frames.append(renderer.render().copy())
            physics_step += 1

    renderer.close()
    video_path = run_dir / "walking_quadruped.mp4"
    media.write_video(str(video_path), frames, fps=RENDER_FPS)
    print(f"Saved {video_path} ({len(frames)} frames, {video_path.stat().st_size / 1024:.0f} KB)",
          flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evolve a walking quadruped")
    parser.add_argument("--output-dir", type=str, default="/data/evo-embodied",
                        help="Base directory for results (default: /data/evo-embodied)")
    args = parser.parse_args()

    run_dir = make_run_dir(args.output_dir)
    print(f"Run directory: {run_dir}", flush=True)
    print(f"JAX: {jax.default_backend()}, devices: {jax.devices()}\n", flush=True)

    config = save_config(run_dir)

    mj_model, mjx_model = build_model()
    n_s = mj_model.nsensordata
    n_m = mj_model.nu
    print(f"Quadruped: {n_s} sensors, {n_m} motors", flush=True)

    evaluate_batch, n_w = make_evolution_fns(mjx_model, n_s, n_m)
    print(f"Network: {n_s}+2 inputs → {N_HIDDEN} hidden → {n_m} outputs ({n_w} weights)", flush=True)
    print(f"Control: 25 Hz ({CONTROL_STEPS} steps x {PHYSICS_PER_CTRL} physics/step), "
          f"2 Hz gait clock\n", flush=True)

    best_weights, best_hist, mean_hist = evolve(mjx_model, evaluate_batch, n_w, run_dir)
    plot_fitness(best_hist, mean_hist, run_dir)
    render_video(mj_model, best_weights, n_s, n_m, run_dir)

    print(f"\nDone! All results in {run_dir}/", flush=True)
    print(f"  config.json         — experiment parameters", flush=True)
    print(f"  summary.json        — timing, fitness, hardware info", flush=True)
    print(f"  best_weights.npy    — best evolved controller", flush=True)
    print(f"  population.npy      — full final population", flush=True)
    print(f"  fitnesses.npy       — all final fitness values", flush=True)
    print(f"  history.npz         — per-generation best/mean fitness", flush=True)
    print(f"  fitness_curve.png   — fitness plot", flush=True)
    print(f"  walking_quadruped.mp4 — video of best controller", flush=True)
    print(f"  weights_gen*.npy    — checkpoints every 100 gens", flush=True)

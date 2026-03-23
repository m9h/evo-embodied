"""Experiment sweep runner for evo-embodied.

Runs a grid of hyperparameter experiments, saving results to /data/evo-embodied/.
Handles headless rendering (EGL), checkpointing, and summary generation.

Usage:
    # Run all experiments in the default sweep
    uv run python experiments/sweep.py

    # Run a specific experiment by name
    uv run python experiments/sweep.py --only hidden64_8s

    # Custom output directory
    uv run python experiments/sweep.py --output-dir /data/evo-embodied

    # List available experiments without running
    uv run python experiments/sweep.py --list
"""
import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Force EGL for headless rendering (must be before mujoco import)
os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from mujoco import mjx

# Conditional mediapy import (may not have ffmpeg on all systems)
try:
    import mediapy as media
    HAS_MEDIAPY = True
except ImportError:
    HAS_MEDIAPY = False


MODEL_PATH = Path(__file__).parent.parent / "models" / "quadruped.xml"


# ── Experiment Configuration ───────────────────────────────────────


@dataclass
class ExperimentConfig:
    """All hyperparameters for one evolution run."""
    name: str = "default"
    n_population: int = 256
    n_generations: int = 500
    n_hidden: int = 16
    control_steps: int = 100
    physics_per_ctrl: int = 20
    mutation_scale: float = 0.1
    mutation_decay: float = 0.9995
    min_torso_height: float = 0.25
    clock_freq_hz: float = 2.0
    render_ctrl_steps: int = 200
    render_fps: int = 30
    seed: int = 42

    @property
    def control_freq_hz(self):
        return 1.0 / (0.002 * self.physics_per_ctrl)

    @property
    def sim_duration_s(self):
        return self.control_steps * self.physics_per_ctrl * 0.002

    def to_dict(self):
        d = asdict(self)
        d["control_freq_hz"] = self.control_freq_hz
        d["sim_duration_s"] = self.sim_duration_s
        d["fitness"] = "x_distance - 10*height_penalty - 0.5*y_drift"
        d["model"] = "quadruped.xml"
        return d


# ── Default Sweep ──────────────────────────────────────────────────


SWEEP = {
    # Baseline
    "baseline": ExperimentConfig(
        name="baseline",
    ),
    # Network size sweep
    "hidden32": ExperimentConfig(
        name="hidden32", n_hidden=32, n_generations=300,
    ),
    "hidden64": ExperimentConfig(
        name="hidden64", n_hidden=64, n_generations=300,
    ),
    "hidden128": ExperimentConfig(
        name="hidden128", n_hidden=128, n_generations=300,
    ),
    # Population size sweep
    "pop512": ExperimentConfig(
        name="pop512", n_population=512, n_generations=300,
    ),
    "pop1024": ExperimentConfig(
        name="pop1024", n_population=1024, n_generations=200,
    ),
    # Simulation duration sweep
    "sim8s": ExperimentConfig(
        name="sim8s", control_steps=200, n_generations=300,
    ),
    # Mutation sweep
    "fine_mut": ExperimentConfig(
        name="fine_mut", mutation_scale=0.05, mutation_decay=0.999,
        n_generations=300,
    ),
    "coarse_mut": ExperimentConfig(
        name="coarse_mut", mutation_scale=0.2, mutation_decay=0.9998,
        n_generations=300,
    ),
    # Combined best from initial sweep
    "combined_best": ExperimentConfig(
        name="combined_best", n_hidden=64, control_steps=200,
        n_generations=500,
    ),
    # Push further: larger net + longer sim + more gens
    "h64_8s_pop512": ExperimentConfig(
        name="h64_8s_pop512", n_hidden=64, control_steps=200,
        n_population=512, n_generations=500,
    ),
    # Clock frequency experiments
    "clock_1hz": ExperimentConfig(
        name="clock_1hz", n_hidden=64, control_steps=200,
        n_generations=300, clock_freq_hz=1.0,
    ),
    "clock_4hz": ExperimentConfig(
        name="clock_4hz", n_hidden=64, control_steps=200,
        n_generations=300, clock_freq_hz=4.0,
    ),
}


# ── Evolution Core ─────────────────────────────────────────────────


def build_model():
    mj_model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    mjx_model = mjx.put_model(mj_model)
    return mj_model, mjx_model


def make_evaluate_fn(mjx_model, n_sensors, n_motors, cfg):
    n_inputs = n_sensors + 2
    n_w = n_inputs * cfg.n_hidden + cfg.n_hidden * n_motors

    @jax.jit
    def evaluate_one(weights_flat, mjx_model):
        w1 = weights_flat[:n_inputs * cfg.n_hidden].reshape(n_inputs, cfg.n_hidden)
        w2 = weights_flat[n_inputs * cfg.n_hidden:].reshape(cfg.n_hidden, n_motors)

        mjx_data = mjx.make_data(mjx_model)
        dt = mjx_model.opt.timestep
        ctrl_dt = dt * cfg.physics_per_ctrl

        def physics_step(data, _):
            return mjx.step(mjx_model, data), None

        def control_step(carry, ctrl_idx):
            data = carry
            t = ctrl_idx * ctrl_dt
            clock_sin = jnp.sin(2.0 * jnp.pi * cfg.clock_freq_hz * t)
            clock_cos = jnp.cos(2.0 * jnp.pi * cfg.clock_freq_hz * t)

            sensor_input = jnp.concatenate([
                data.sensordata,
                jnp.array([clock_sin, clock_cos])
            ])
            ctrl = jnp.tanh(jnp.tanh(sensor_input @ w1) @ w2)
            data = data.replace(ctrl=ctrl)
            data, _ = jax.lax.scan(physics_step, data, None,
                                   length=cfg.physics_per_ctrl)
            return data, data.qpos[2]

        final_data, torso_heights = jax.lax.scan(
            control_step, mjx_data, jnp.arange(cfg.control_steps),
        )

        x_dist = final_data.qpos[0]
        height_pen = jnp.mean(
            jnp.maximum(0.0, cfg.min_torso_height - torso_heights)
        ) * 10.0
        y_drift = jnp.abs(final_data.qpos[1]) * 0.5

        return x_dist - height_pen - y_drift

    evaluate_batch = jax.jit(jax.vmap(evaluate_one, in_axes=(0, None)))
    return evaluate_batch, n_w


def evolve(mjx_model, evaluate_batch, n_w, cfg, run_dir):
    key = jax.random.PRNGKey(cfg.seed)
    key, init_key = jax.random.split(key)
    population = jax.random.normal(init_key, (cfg.n_population, n_w)) * 0.3

    print("JIT compiling...", flush=True)
    t0 = time.time()
    fitnesses = evaluate_batch(population, mjx_model)
    fitnesses.block_until_ready()
    jit_time = time.time() - t0
    print(f"JIT done in {jit_time:.1f}s\n", flush=True)

    best_history, mean_history = [], []
    mutation_scale = cfg.mutation_scale

    print(f"{'Gen':>5s}  {'Best':>8s}  {'Mean':>8s}  {'Mut':>8s}", flush=True)
    print("-" * 36, flush=True)

    t_start = time.time()
    for gen in range(cfg.n_generations):
        key, mk = jax.random.split(key)
        candidates = population + jax.random.normal(mk, population.shape) * mutation_scale
        cand_fit = evaluate_batch(candidates, mjx_model)

        improved = cand_fit > fitnesses
        population = jnp.where(improved[:, None], candidates, population)
        fitnesses = jnp.where(improved, cand_fit, fitnesses)

        gb, gm = float(fitnesses.max()), float(fitnesses.mean())
        best_history.append(gb)
        mean_history.append(gm)
        mutation_scale *= cfg.mutation_decay

        if (gen + 1) % 50 == 0 or gen == 0:
            print(f"{gen+1:5d}  {gb:+8.4f}  {gm:+8.4f}  {mutation_scale:8.5f}", flush=True)

        if (gen + 1) % 100 == 0:
            best_idx = int(jnp.argmax(fitnesses))
            np.save(run_dir / f"weights_gen{gen+1:04d}.npy",
                    np.array(population[best_idx]))

    elapsed = time.time() - t_start
    best_idx = int(jnp.argmax(fitnesses))
    best_weights = np.array(population[best_idx])
    total_sims = cfg.n_population * cfg.n_generations

    print(f"\nEvolution: {elapsed:.1f}s, {total_sims:,} sims ({total_sims/elapsed:.0f}/sec)", flush=True)
    print(f"Best fitness: {float(fitnesses[best_idx]):+.4f}", flush=True)

    # Save everything
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

    return best_weights, best_history, mean_history, summary


# ── Visualization ──────────────────────────────────────────────────


def plot_fitness(best_hist, mean_hist, cfg, run_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(best_hist, label="Best", linewidth=2)
    ax.plot(mean_hist, label="Mean", linewidth=1, alpha=0.7)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (x_dist - height_penalty - y_drift)")
    ax.set_title(f"{cfg.name} — pop={cfg.n_population}, gens={cfg.n_generations}, "
                 f"h={cfg.n_hidden}, sim={cfg.sim_duration_s:.0f}s")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(run_dir / "fitness_curve.png", dpi=150)
    plt.close(fig)
    print(f"Saved {run_dir / 'fitness_curve.png'}", flush=True)


def render_video(mj_model, best_weights, n_sensors, n_motors, cfg, run_dir):
    """Render video using EGL (works headless on DGX/servers)."""
    if not HAS_MEDIAPY:
        print("Skipping video: mediapy not installed", flush=True)
        return

    n_inputs = n_sensors + 2
    w1 = best_weights[:n_inputs * cfg.n_hidden].reshape(n_inputs, cfg.n_hidden)
    w2 = best_weights[n_inputs * cfg.n_hidden:].reshape(cfg.n_hidden, n_motors)

    data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, data)

    try:
        renderer = mujoco.Renderer(mj_model, height=720, width=1280)
    except Exception as e:
        print(f"Renderer init failed: {e}", flush=True)
        print("Hint: set MUJOCO_GL=egl for headless rendering", flush=True)
        return

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    cam.distance = 3.0
    cam.azimuth = 150
    cam.elevation = -25

    dt = mj_model.opt.timestep
    ctrl_dt = dt * cfg.physics_per_ctrl
    steps_per_frame = max(1, int(1.0 / (cfg.render_fps * dt)))

    frames = []
    physics_step = 0
    total_steps = cfg.render_ctrl_steps * cfg.physics_per_ctrl
    print(f"Rendering {cfg.render_ctrl_steps} ctrl steps "
          f"({total_steps * dt:.1f}s)...", flush=True)

    for ctrl_i in range(cfg.render_ctrl_steps):
        t = ctrl_i * ctrl_dt
        clock = np.array([
            np.sin(2.0 * np.pi * cfg.clock_freq_hz * t),
            np.cos(2.0 * np.pi * cfg.clock_freq_hz * t),
        ])
        sensor_input = np.concatenate([data.sensordata.copy(), clock])
        ctrl = np.tanh(np.tanh(sensor_input @ w1) @ w2)
        data.ctrl[:] = ctrl

        for _ in range(cfg.physics_per_ctrl):
            mujoco.mj_step(mj_model, data)
            if physics_step % steps_per_frame == 0:
                renderer.update_scene(data, cam)
                frames.append(renderer.render().copy())
            physics_step += 1

    renderer.close()

    video_path = run_dir / "video.mp4"
    media.write_video(str(video_path), frames, fps=cfg.render_fps)
    print(f"Saved {video_path} ({len(frames)} frames, "
          f"{video_path.stat().st_size / 1024:.0f} KB)", flush=True)


# ── Runner ─────────────────────────────────────────────────────────


def run_experiment(cfg, output_base):
    """Run one full experiment: evolve → plot → render."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_base) / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    (run_dir / "experiment.json").write_text(json.dumps({"experiment": cfg.name}))

    print(f"\n{'='*60}", flush=True)
    print(f"Experiment: {cfg.name}", flush=True)
    print(f"Output: {run_dir}", flush=True)
    print(f"JAX: {jax.default_backend()}, {jax.devices()}", flush=True)
    print(f"{'='*60}\n", flush=True)

    mj_model, mjx_model = build_model()
    n_s, n_m = mj_model.nsensordata, mj_model.nu

    print(f"Quadruped: {n_s} sensors, {n_m} motors", flush=True)
    evaluate_batch, n_w = make_evaluate_fn(mjx_model, n_s, n_m, cfg)
    print(f"Network: {n_s}+2 → {cfg.n_hidden} → {n_m} ({n_w} weights)", flush=True)
    print(f"Sim: {cfg.sim_duration_s:.0f}s @ {cfg.control_freq_hz:.0f}Hz ctrl, "
          f"{cfg.clock_freq_hz:.0f}Hz clock", flush=True)
    print(f"Evolution: pop={cfg.n_population}, gens={cfg.n_generations}, "
          f"mut={cfg.mutation_scale}\n", flush=True)

    best_weights, best_hist, mean_hist, summary = evolve(
        mjx_model, evaluate_batch, n_w, cfg, run_dir)
    plot_fitness(best_hist, mean_hist, cfg, run_dir)
    render_video(mj_model, best_weights, n_s, n_m, cfg, run_dir)

    print(f"\nDone: {cfg.name} → fitness={summary['best_fitness']}", flush=True)
    print(f"Results: {run_dir}/\n", flush=True)

    return run_dir, summary


def run_sweep(experiments, output_base):
    """Run multiple experiments and produce a comparison summary."""
    results = {}

    for name, cfg in experiments.items():
        try:
            run_dir, summary = run_experiment(cfg, output_base)
            results[name] = {
                "run_dir": str(run_dir),
                "fitness": summary["best_fitness"],
                "mean_fitness": summary["mean_fitness"],
                "evolution_time_s": summary["evolution_time_s"],
                "sims_per_sec": summary["sims_per_sec"],
            }
        except Exception as e:
            print(f"\nERROR in {name}: {e}\n", flush=True)
            results[name] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("SWEEP RESULTS", flush=True)
    print("=" * 60, flush=True)

    sorted_results = sorted(
        [(k, v) for k, v in results.items() if "fitness" in v],
        key=lambda x: x[1]["fitness"],
        reverse=True,
    )
    for name, r in sorted_results:
        print(f"  {name:25s}  fitness={r['fitness']:+8.4f}  "
              f"({r['evolution_time_s']:.0f}s, {r['sims_per_sec']} sims/sec)",
              flush=True)

    # Save sweep summary
    summary_path = Path(output_base) / "sweep_summary.json"
    (summary_path).write_text(json.dumps(results, indent=2))
    print(f"\nSweep summary: {summary_path}", flush=True)

    return results


# ── Comparison Plot ────────────────────────────────────────────────


def plot_comparison(output_base):
    """Generate comparison plot from all runs in output_base."""
    base = Path(output_base)
    runs = []

    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        exp_file = d / "experiment.json"
        hist_file = d / "history.npz"
        if exp_file.exists() and hist_file.exists():
            exp = json.loads(exp_file.read_text())
            hist = np.load(hist_file)
            runs.append((exp.get("experiment", d.name), hist["best"], hist["mean"]))

    if not runs:
        print("No runs found for comparison plot", flush=True)
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for name, best, mean in runs:
        ax1.plot(best, label=name, linewidth=1.5)
        ax2.plot(mean, label=name, linewidth=1.5)

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Best Fitness")
    ax1.set_title("Best Fitness by Experiment")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Mean Fitness")
    ax2.set_title("Mean Fitness by Experiment")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = base / "comparison.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved {plot_path}", flush=True)


# ── CLI ────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evo-embodied experiment sweep")
    parser.add_argument("--output-dir", default="/data/evo-embodied",
                        help="Base output directory")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only this experiment (comma-separated names)")
    parser.add_argument("--list", action="store_true",
                        help="List available experiments and exit")
    parser.add_argument("--compare", action="store_true",
                        help="Generate comparison plot from existing results")
    args = parser.parse_args()

    if args.list:
        print("Available experiments:")
        for name, cfg in SWEEP.items():
            print(f"  {name:25s}  pop={cfg.n_population:4d}  gens={cfg.n_generations:4d}  "
                  f"h={cfg.n_hidden:3d}  sim={cfg.sim_duration_s:.0f}s  "
                  f"mut={cfg.mutation_scale}")
        raise SystemExit(0)

    if args.compare:
        plot_comparison(args.output_dir)
        raise SystemExit(0)

    if args.only:
        names = [n.strip() for n in args.only.split(",")]
        experiments = {n: SWEEP[n] for n in names if n in SWEEP}
        missing = [n for n in names if n not in SWEEP]
        if missing:
            print(f"Unknown experiments: {missing}")
            print(f"Available: {list(SWEEP.keys())}")
            raise SystemExit(1)
    else:
        experiments = SWEEP

    run_sweep(experiments, args.output_dir)
    plot_comparison(args.output_dir)

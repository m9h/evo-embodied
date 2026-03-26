"""Domain randomization for sim2real transfer on the Petoi Bittle.

The idea: if a policy works across many *different* simulations, it's more
likely to work on the *real* robot (which has specific but unknown dynamics
somewhere in that range).

This script extends 10_bittle_gait.py with two kinds of randomization:

1. MODEL-LEVEL (pool of MuJoCo models with different dynamics)
   - Friction: +/-50%
   - Body mass: +/-20% per link
   - Joint damping: +/-30%
   Each generation trains against a different model from the pool.

2. STEP-LEVEL (noise injected inside the JIT-compiled evaluation)
   - Sensor noise: Gaussian noise on IMU and joint position readings
   - Action noise: Gaussian noise on control outputs (servo imprecision)
   - External perturbation: random forces on torso (bumps, uneven ground)

The result: lower peak fitness than deterministic training, but a policy
that transfers to real hardware instead of exploiting simulation quirks.

Run: uv run python examples/11_domain_randomization.py --output-dir /data/evo-embodied
Compare: run 10_bittle_gait.py and this script, then deploy both to the
real Bittle (sim2real/deploy_bittle.py). The randomized policy should
perform better on hardware despite worse sim fitness.
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
N_HIDDEN = 32
CONTROL_STEPS = 250
PHYSICS_PER_CTRL = 20
MUTATION_SCALE = 0.08
MUTATION_DECAY = 0.9995
SEED = 42

# Fitness weights (same as 10_bittle_gait.py)
VELOCITY_WEIGHT = 2.0
ENERGY_WEIGHT = 0.01
SMOOTHNESS_WEIGHT = 0.05
HEIGHT_PENALTY_WEIGHT = 10.0
ALIVE_BONUS = 0.05
DRIFT_WEIGHT = 0.5
MIN_TORSO_HEIGHT = 0.05
CLOCK_FREQ_HZ = 2.0
LEG_PHASE_OFFSETS = [0.0, jnp.pi / 2, jnp.pi, 3 * jnp.pi / 2]
N_CLOCK_INPUTS = len(LEG_PHASE_OFFSETS) * 2
CTRL_SCALE = 1.5

# ── Domain randomization config ────────────────────────────────────
N_MODEL_VARIANTS = 16     # pool of randomized MuJoCo models
SENSOR_NOISE_STD = 0.02   # radians for joint pos, m/s² for IMU
ACTION_NOISE_STD = 0.03   # radians of servo imprecision
PERTURBATION_FORCE = 0.05 # Newtons (random push on torso every N steps)
PERTURBATION_INTERVAL = 50  # apply perturbation every N control steps

RENDER_CTRL_STEPS = 375
RENDER_FPS = 30


def make_run_dir(base_dir):
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base_dir) / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(run_dir):
    config = {
        "experiment": "domain_randomization",
        "model": "bittle/bittle.xml",
        "n_population": N_POPULATION,
        "n_generations": N_GENERATIONS,
        "n_hidden": N_HIDDEN,
        "control_steps": CONTROL_STEPS,
        "physics_per_ctrl": PHYSICS_PER_CTRL,
        "sim_duration_s": CONTROL_STEPS * PHYSICS_PER_CTRL * 0.002,
        "fitness_weights": {
            "velocity": VELOCITY_WEIGHT,
            "energy": ENERGY_WEIGHT,
            "smoothness": SMOOTHNESS_WEIGHT,
            "height_penalty": HEIGHT_PENALTY_WEIGHT,
            "alive_bonus": ALIVE_BONUS,
            "drift": DRIFT_WEIGHT,
        },
        "domain_randomization": {
            "n_model_variants": N_MODEL_VARIANTS,
            "friction_range": [0.5, 1.5],
            "mass_range": [0.8, 1.2],
            "damping_range": [0.7, 1.3],
            "sensor_noise_std": SENSOR_NOISE_STD,
            "action_noise_std": ACTION_NOISE_STD,
            "perturbation_force_n": PERTURBATION_FORCE,
            "perturbation_interval": PERTURBATION_INTERVAL,
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    (run_dir / "experiment.json").write_text(
        json.dumps({"experiment": "domain_randomization"}))
    return config


# ── Model-level randomization (CPU side) ───────────────────────────

def create_randomized_model(xml_path, rng):
    """Create a MuJoCo model with randomized dynamics.

    Modifies the MjModel arrays directly. This is the standard approach
    for domain randomization — create many model variants, train on each.
    """
    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))

    # Friction: scale all geom friction uniformly (+/-50%)
    friction_scale = rng.uniform(0.5, 1.5)
    mj_model.geom_friction[:] *= friction_scale

    # Mass: scale each body independently (+/-20%)
    for i in range(mj_model.nbody):
        mass_scale = rng.uniform(0.8, 1.2)
        mj_model.body_mass[i] *= mass_scale
        # Scale inertia proportionally to keep physics consistent
        mj_model.body_inertia[i] *= mass_scale

    # Joint damping: scale each DOF (+/-30%)
    for i in range(mj_model.nv):
        damp_scale = rng.uniform(0.7, 1.3)
        mj_model.dof_damping[i] *= damp_scale

    return mj_model


def build_model_pool():
    """Create a pool of randomized MJX models for training."""
    rng = np.random.default_rng(SEED)

    # Base model (deterministic, for comparison)
    mj_base = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

    # Randomized variants
    print(f"Creating {N_MODEL_VARIANTS} randomized Bittle models...", flush=True)
    mj_models = []
    mjx_models = []
    randomization_params = []

    for i in range(N_MODEL_VARIANTS):
        mj_rand = create_randomized_model(str(MODEL_PATH), rng)
        mjx_rand = mjx.put_model(mj_rand)
        mj_models.append(mj_rand)
        mjx_models.append(mjx_rand)
        randomization_params.append({
            "variant": i,
            "friction_mean": float(np.mean(mj_rand.geom_friction[:, 0])),
            "mass_total": float(np.sum(mj_rand.body_mass)),
            "damping_mean": float(np.mean(mj_rand.dof_damping)),
        })

    print(f"  Mass range: "
          f"{min(p['mass_total'] for p in randomization_params):.3f} - "
          f"{max(p['mass_total'] for p in randomization_params):.3f} kg",
          flush=True)
    print(f"  Friction range: "
          f"{min(p['friction_mean'] for p in randomization_params):.3f} - "
          f"{max(p['friction_mean'] for p in randomization_params):.3f}",
          flush=True)

    return mj_base, mj_models, mjx_models, randomization_params


# ── Step-level randomization (inside JIT) ──────────────────────────

def make_clock_inputs(t):
    clocks = []
    for phase in LEG_PHASE_OFFSETS:
        clocks.append(jnp.sin(2.0 * jnp.pi * CLOCK_FREQ_HZ * t + phase))
        clocks.append(jnp.cos(2.0 * jnp.pi * CLOCK_FREQ_HZ * t + phase))
    return jnp.array(clocks)


def make_evaluate_fn(mjx_model_ref, n_sensors, n_motors):
    """Build evaluation with step-level domain randomization.

    Model-level randomization is handled outside the JIT (different
    mjx_model per generation). Step-level noise is inside the JIT.
    """
    n_inputs = n_sensors + N_CLOCK_INPUTS
    n_w = n_inputs * N_HIDDEN + N_HIDDEN * n_motors

    @jax.jit
    def evaluate_one(weights_flat, mjx_model, noise_key):
        w1 = weights_flat[:n_inputs * N_HIDDEN].reshape(n_inputs, N_HIDDEN)
        w2 = weights_flat[n_inputs * N_HIDDEN:].reshape(N_HIDDEN, n_motors)

        mjx_data = mjx.make_data(mjx_model)
        dt = mjx_model.opt.timestep
        ctrl_dt = dt * PHYSICS_PER_CTRL

        def physics_step(data, _):
            return mjx.step(mjx_model, data), None

        def control_step(carry, ctrl_idx):
            data, prev_ctrl, step_key = carry

            k1, k2, k3, next_key = jax.random.split(step_key, 4)

            t = ctrl_idx * ctrl_dt
            clock = make_clock_inputs(t)

            # Sensor noise: corrupt observations (sim2real gap)
            noisy_sensors = (
                data.sensordata
                + jax.random.normal(k1, data.sensordata.shape) * SENSOR_NOISE_STD
            )

            sensor_input = jnp.concatenate([noisy_sensors, clock])
            hidden = jnp.tanh(sensor_input @ w1)
            ctrl = jnp.tanh(hidden @ w2) * CTRL_SCALE

            # Action noise: servo imprecision
            ctrl = ctrl + jax.random.normal(k2, ctrl.shape) * ACTION_NOISE_STD

            data = data.replace(ctrl=ctrl)

            # External perturbation: random push on torso
            apply_perturbation = (ctrl_idx % PERTURBATION_INTERVAL == 0)
            force = jax.random.normal(k3, (3,)) * PERTURBATION_FORCE
            # xfrc_applied shape: (nbody, 6) — [fx, fy, fz, tx, ty, tz]
            # Body 1 is typically the torso (body 0 is world)
            perturbation = jnp.zeros_like(data.xfrc_applied)
            perturbation = perturbation.at[1, :3].set(
                force * apply_perturbation.astype(jnp.float32)
            )
            data = data.replace(xfrc_applied=perturbation)

            data, _ = jax.lax.scan(physics_step, data, None,
                                   length=PHYSICS_PER_CTRL)

            # Clear perturbation after physics (don't persist across steps)
            data = data.replace(
                xfrc_applied=jnp.zeros_like(data.xfrc_applied))

            torso_z = data.qpos[2]
            x_vel = data.qvel[0]
            energy = jnp.sum(ctrl ** 2)
            ctrl_change = jnp.sum((ctrl - prev_ctrl) ** 2)
            alive = (torso_z > MIN_TORSO_HEIGHT).astype(jnp.float32)

            metrics = jnp.array([x_vel, energy, ctrl_change, torso_z, alive])
            return (data, ctrl, next_key), metrics

        init_ctrl = jnp.zeros(n_motors)
        (final_data, _, _), all_metrics = jax.lax.scan(
            control_step,
            (mjx_data, init_ctrl, noise_key),
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

    def evaluate_batch(population, mjx_model, batch_key):
        """Evaluate population with per-individual noise keys."""
        keys = jax.random.split(batch_key, population.shape[0])
        return jax.jit(jax.vmap(evaluate_one, in_axes=(0, None, 0)))(
            population, mjx_model, keys)

    return evaluate_batch, n_w


def evolve(mjx_models, evaluate_batch, n_w, run_dir):
    key = jax.random.PRNGKey(SEED)
    key, init_key = jax.random.split(key)
    population = jax.random.normal(init_key, (N_POPULATION, n_w)) * 0.3

    # JIT compile on first model
    print("JIT compiling...", flush=True)
    t0 = time.time()
    key, noise_key = jax.random.split(key)
    fitnesses = evaluate_batch(population, mjx_models[0], noise_key)
    fitnesses.block_until_ready()
    jit_time = time.time() - t0
    print(f"JIT done in {jit_time:.1f}s\n", flush=True)

    best_history, mean_history = [], []
    model_usage = []
    mutation_scale = MUTATION_SCALE

    print(f"{'Gen':>5s}  {'Best':>8s}  {'Mean':>8s}  {'Mut':>8s}  {'Model':>5s}",
          flush=True)
    print("-" * 44, flush=True)

    t_start = time.time()
    for gen in range(N_GENERATIONS):
        # Select a randomized model for this generation
        model_idx = gen % len(mjx_models)
        mjx_model = mjx_models[model_idx]

        key, mk, noise_key = jax.random.split(key, 3)
        candidates = population + jax.random.normal(mk, population.shape) * mutation_scale
        cand_fit = evaluate_batch(candidates, mjx_model, noise_key)

        improved = cand_fit > fitnesses
        population = jnp.where(improved[:, None], candidates, population)
        fitnesses = jnp.where(improved, cand_fit, fitnesses)

        gb, gm = float(fitnesses.max()), float(fitnesses.mean())
        best_history.append(gb)
        mean_history.append(gm)
        model_usage.append(model_idx)
        mutation_scale *= MUTATION_DECAY

        if (gen + 1) % 50 == 0 or gen == 0:
            print(f"{gen+1:5d}  {gb:+8.4f}  {gm:+8.4f}  "
                  f"{mutation_scale:8.5f}  {model_idx:5d}", flush=True)

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
        "n_model_variants": N_MODEL_VARIANTS,
        "sensor_noise_std": SENSOR_NOISE_STD,
        "action_noise_std": ACTION_NOISE_STD,
        "jax_backend": str(jax.default_backend()),
        "jax_devices": [str(d) for d in jax.devices()],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Results saved to {run_dir}/", flush=True)

    print(f"RESULT|fitness={summary['best_fitness']}|sims_per_sec={summary['sims_per_sec']}|",
          flush=True)

    return best_weights, best_history, mean_history


def plot_fitness(best_hist, mean_hist, run_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(best_hist, label="Best", linewidth=2)
    ax.plot(mean_hist, label="Mean", linewidth=1, alpha=0.7)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title(f"Domain Randomization — {N_MODEL_VARIANTS} model variants, "
                 f"sensor noise={SENSOR_NOISE_STD}, action noise={ACTION_NOISE_STD}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(run_dir / "fitness_curve.png", dpi=150)
    plt.close(fig)
    print(f"Saved {run_dir / 'fitness_curve.png'}", flush=True)


def render_video(mj_model, best_weights, n_sensors, n_motors, run_dir):
    """Render on the BASE (unrandomized) model to see true performance."""
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
    cam.distance = 0.6
    cam.azimuth = 150
    cam.elevation = -25

    dt = mj_model.opt.timestep
    ctrl_dt = dt * PHYSICS_PER_CTRL
    steps_per_frame = max(1, int(1.0 / (RENDER_FPS * dt)))

    frames = []
    physics_step = 0

    print(f"Rendering {RENDER_CTRL_STEPS} ctrl steps (base model, no noise)...",
          flush=True)
    for ctrl_i in range(RENDER_CTRL_STEPS):
        t = ctrl_i * ctrl_dt
        clock = []
        for phase in [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]:
            clock.append(np.sin(2.0 * np.pi * CLOCK_FREQ_HZ * t + phase))
            clock.append(np.cos(2.0 * np.pi * CLOCK_FREQ_HZ * t + phase))
        clock = np.array(clock)

        sensor_input = np.concatenate([data.sensordata.copy(), clock])
        hidden = np.tanh(sensor_input @ w1)
        ctrl = np.tanh(hidden @ w2) * CTRL_SCALE
        data.ctrl[:] = ctrl

        for _ in range(PHYSICS_PER_CTRL):
            mujoco.mj_step(mj_model, data)
            if physics_step % steps_per_frame == 0:
                renderer.update_scene(data, cam)
                frames.append(renderer.render().copy())
            physics_step += 1

    renderer.close()

    video_path = run_dir / "bittle_domrand.mp4"
    media.write_video(str(video_path), frames, fps=RENDER_FPS)
    print(f"Saved {video_path} ({len(frames)} frames, "
          f"{video_path.stat().st_size / 1024:.0f} KB)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Domain randomization for Bittle sim2real")
    parser.add_argument("--output-dir", default="/data/evo-embodied")
    args = parser.parse_args()

    run_dir = make_run_dir(args.output_dir)
    print(f"Run: {run_dir}", flush=True)
    print(f"JAX: {jax.default_backend()}, {jax.devices()}\n", flush=True)
    save_config(run_dir)

    mj_base, mj_models, mjx_models, rand_params = build_model_pool()
    (run_dir / "randomization_params.json").write_text(
        json.dumps(rand_params, indent=2))

    n_s = mj_base.nsensordata
    n_m = mj_base.nu
    print(f"\nBittle: {n_s} sensors, {n_m} motors", flush=True)
    print(f"Domain randomization:", flush=True)
    print(f"  {N_MODEL_VARIANTS} model variants (friction, mass, damping)", flush=True)
    print(f"  Sensor noise: {SENSOR_NOISE_STD} std", flush=True)
    print(f"  Action noise: {ACTION_NOISE_STD} std", flush=True)
    print(f"  Perturbation: {PERTURBATION_FORCE}N every {PERTURBATION_INTERVAL} steps\n",
          flush=True)

    evaluate_batch, n_w = make_evaluate_fn(mjx_models[0], n_s, n_m)
    print(f"Network: {n_s}+{N_CLOCK_INPUTS} → {N_HIDDEN} → {n_m} "
          f"({n_w} weights)\n", flush=True)

    best_weights, best_hist, mean_hist = evolve(
        mjx_models, evaluate_batch, n_w, run_dir)
    plot_fitness(best_hist, mean_hist, run_dir)
    render_video(mj_base, best_weights, n_s, n_m, run_dir)

    print(f"\nDone! Results: {run_dir}/", flush=True)
    print(f"Compare with deterministic training (10_bittle_gait.py).", flush=True)
    print(f"Deploy both to real Bittle and see which transfers better.", flush=True)

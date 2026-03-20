"""Evolve a quadruped and render the best one walking as an MP4 video.

Runs a short evolution (or loads saved weights), then renders the best
controller to video using MuJoCo's offscreen renderer + mediapy.

Run: uv run python examples/04_render_evolved_quadruped.py
Output: evolved_quadruped.mp4
"""
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mediapy as media
import mujoco
import numpy as np
from mujoco import mjx

# ── Config ──────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent.parent / "models" / "quadruped.xml"
VIDEO_PATH = Path(__file__).parent.parent / "evolved_quadruped.mp4"
FITNESS_PLOT_PATH = Path(__file__).parent.parent / "fitness_curve.png"

N_POPULATION = 128
N_GENERATIONS = 300
N_HIDDEN = 8
SIM_STEPS = 500
MUTATION_SCALE = 0.05
RENDER_STEPS = 2000      # longer sim for video
RENDER_FPS = 30
RENDER_WIDTH = 1280
RENDER_HEIGHT = 720


def evolve():
    """Run parallel hill climber, return best weights and fitness history."""
    mj_model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    mjx_model = mjx.put_model(mj_model)
    n_s = mj_model.nsensordata
    n_m = mj_model.nu
    n_w = n_s * N_HIDDEN + N_HIDDEN * n_m

    @jax.jit
    def eval_one(wf, mm):
        w1 = wf[: n_s * N_HIDDEN].reshape(n_s, N_HIDDEN)
        w2 = wf[n_s * N_HIDDEN :].reshape(N_HIDDEN, n_m)
        d = mjx.make_data(mm)

        def step_fn(d, _):
            c = jnp.tanh(jnp.tanh(d.sensordata @ w1) @ w2)
            return mjx.step(mm, d.replace(ctrl=c)), None

        fd, _ = jax.lax.scan(step_fn, d, None, length=SIM_STEPS)
        return fd.qpos[0]

    eval_batch = jax.jit(jax.vmap(eval_one, in_axes=(0, None)))

    # Initialize
    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    population = jax.random.normal(init_key, (N_POPULATION, n_w)) * 0.1

    print("Compiling (JIT)...")
    fitnesses = eval_batch(population, mjx_model)
    fitnesses.block_until_ready()
    print("JIT done.\n")

    best_history = []
    mean_history = []

    print(f"Evolving {N_POPULATION} parallel hill climbers x {N_GENERATIONS} generations...")
    print(f"{'Gen':>5s}  {'Best':>8s}  {'Mean':>8s}")
    print("-" * 28)

    t0 = time.time()
    for gen in range(N_GENERATIONS):
        key, mk = jax.random.split(key)
        mutations = jax.random.normal(mk, population.shape) * MUTATION_SCALE
        candidates = population + mutations
        cand_fit = eval_batch(candidates, mjx_model)

        improved = cand_fit > fitnesses
        population = jnp.where(improved[:, None], candidates, population)
        fitnesses = jnp.where(improved, cand_fit, fitnesses)

        gb = float(fitnesses.max())
        gm = float(fitnesses.mean())
        best_history.append(gb)
        mean_history.append(gm)

        if (gen + 1) % 50 == 0 or gen == 0:
            print(f"{gen+1:5d}  {gb:+8.4f}  {gm:+8.4f}")

    elapsed = time.time() - t0
    best_idx = int(jnp.argmax(fitnesses))
    best_weights = np.array(population[best_idx])

    print(f"\nEvolution took {elapsed:.1f}s")
    print(f"Best fitness: {float(fitnesses[best_idx]):+.4f}")

    return mj_model, best_weights, best_history, mean_history


def plot_fitness(best_history, mean_history):
    """Save fitness curve plot."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(best_history, label="Best", linewidth=2)
    ax.plot(mean_history, label="Mean", linewidth=1, alpha=0.7)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (x-distance)")
    ax.set_title(f"Parallel Hill Climber — {N_POPULATION} individuals, {N_GENERATIONS} generations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FITNESS_PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"Fitness curve saved: {FITNESS_PLOT_PATH}")


def render_video(mj_model, best_weights):
    """Render the best evolved controller to MP4."""
    n_s = mj_model.nsensordata
    n_m = mj_model.nu
    w1 = best_weights[: n_s * N_HIDDEN].reshape(n_s, N_HIDDEN)
    w2 = best_weights[n_s * N_HIDDEN :].reshape(N_HIDDEN, n_m)

    data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, data)

    renderer = mujoco.Renderer(mj_model, height=RENDER_HEIGHT, width=RENDER_WIDTH)

    # Camera that tracks the robot
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    cam.distance = 3.0
    cam.azimuth = 135
    cam.elevation = -20

    # Determine frame interval to match desired FPS
    dt = mj_model.opt.timestep
    steps_per_frame = max(1, int(1.0 / (RENDER_FPS * dt)))

    frames = []
    print(f"\nRendering {RENDER_STEPS} timesteps at {RENDER_FPS} fps...")

    for step in range(RENDER_STEPS):
        # Neural network controller
        sensors = data.sensordata.copy()
        hidden = np.tanh(sensors @ w1)
        commands = np.tanh(hidden @ w2)
        data.ctrl[:] = commands
        mujoco.mj_step(mj_model, data)

        # Capture frame at the desired rate
        if step % steps_per_frame == 0:
            renderer.update_scene(data, cam)
            frame = renderer.render()
            frames.append(frame.copy())

    renderer.close()

    print(f"Captured {len(frames)} frames")
    print(f"Writing video to {VIDEO_PATH}...")
    media.write_video(str(VIDEO_PATH), frames, fps=RENDER_FPS)
    print(f"Video saved: {VIDEO_PATH} ({VIDEO_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}\n")

    mj_model, best_weights, best_hist, mean_hist = evolve()
    plot_fitness(best_hist, mean_hist)
    render_video(mj_model, best_weights)

    print("\nDone! Files created:")
    print(f"  Video: {VIDEO_PATH}")
    print(f"  Plot:  {FITNESS_PLOT_PATH}")

"""Assignment 11: Parallel hill climber with MJX + JAX.

The key upgrade over PyBullet: instead of running N hill climbers
sequentially, MJX + JAX vmap runs them ALL simultaneously on GPU (or
parallelized on CPU). This is 10-1000x faster depending on hardware.

Run: uv run python examples/03_mjx_parallel_evolution.py
"""
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Check available backends
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")
print()

# MJX imports
import mujoco
from mujoco import mjx


def load_model():
    """Load quadruped model for both MuJoCo and MJX."""
    model_path = Path(__file__).parent.parent / "models" / "quadruped.xml"
    mj_model = mujoco.MjModel.from_xml_path(str(model_path))
    mjx_model = mjx.put_model(mj_model)
    return mj_model, mjx_model


def make_evaluate_fn(mjx_model, n_sensors, n_motors, n_hidden, sim_steps):
    """Create a JIT-compiled evaluation function."""

    @jax.jit
    def evaluate_one(weights_flat, mjx_model):
        # Unpack weights
        w1 = weights_flat[: n_sensors * n_hidden].reshape(n_sensors, n_hidden)
        w2 = weights_flat[n_sensors * n_hidden :].reshape(n_hidden, n_motors)

        mjx_data = mjx.make_data(mjx_model)

        def step_fn(data, _):
            sensors = data.sensordata
            hidden = jnp.tanh(sensors @ w1)
            commands = jnp.tanh(hidden @ w2)
            data = data.replace(ctrl=commands)
            data = mjx.step(mjx_model, data)
            return data, None

        final_data, _ = jax.lax.scan(step_fn, mjx_data, None, length=sim_steps)
        return final_data.qpos[0]  # x-distance

    # Vectorize across population
    evaluate_batch = jax.jit(jax.vmap(evaluate_one, in_axes=(0, None)))

    return evaluate_one, evaluate_batch


def parallel_hill_climber(
    n_population=64,
    n_generations=200,
    n_hidden=8,
    sim_steps=500,
    mutation_scale=0.05,
):
    """Run n_population independent hill climbers in parallel via MJX."""
    mj_model, mjx_model = load_model()
    n_sensors = mj_model.nsensordata
    n_motors = mj_model.nu
    n_weights = n_sensors * n_hidden + n_hidden * n_motors

    print(f"Quadruped: {n_sensors} sensors, {n_motors} motors")
    print(f"Neural network: {n_sensors} -> {n_hidden} -> {n_motors} ({n_weights} weights)")
    print(f"Population: {n_population} parallel hill climbers")
    print(f"Generations: {n_generations} x {sim_steps} timesteps")
    print()

    evaluate_one, evaluate_batch = make_evaluate_fn(
        mjx_model, n_sensors, n_motors, n_hidden, sim_steps
    )

    # Initialize population
    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    population = jax.random.normal(init_key, (n_population, n_weights)) * 0.1

    # Initial evaluation (also triggers JIT compilation)
    print("Compiling (first evaluation triggers JIT)...")
    t_compile = time.time()
    fitnesses = evaluate_batch(population, mjx_model)
    fitnesses.block_until_ready()  # wait for async JAX computation
    t_compile = time.time() - t_compile
    print(f"JIT compilation took {t_compile:.1f}s (one-time cost)")
    print()

    # Evolution loop
    best_ever = float(fitnesses.max())
    print(f"{'Gen':>5s}  {'Best':>8s}  {'Mean':>8s}  {'Best Ever':>10s}")
    print("-" * 40)

    t_start = time.time()
    for gen in range(n_generations):
        # Mutate
        key, mutate_key = jax.random.split(key)
        mutations = jax.random.normal(mutate_key, population.shape) * mutation_scale
        candidates = population + mutations

        # Evaluate candidates
        candidate_fitnesses = evaluate_batch(candidates, mjx_model)

        # Selection: keep candidate if it improved
        improved = candidate_fitnesses > fitnesses
        population = jnp.where(improved[:, None], candidates, population)
        fitnesses = jnp.where(improved, candidate_fitnesses, fitnesses)

        gen_best = float(fitnesses.max())
        gen_mean = float(fitnesses.mean())
        best_ever = max(best_ever, gen_best)

        if (gen + 1) % 20 == 0 or gen == 0:
            print(f"{gen+1:5d}  {gen_best:+8.4f}  {gen_mean:+8.4f}  {best_ever:+10.4f}")

    elapsed = time.time() - t_start
    total_sims = n_population * n_generations
    print()
    print(f"Evolution took {elapsed:.1f}s")
    print(f"Total simulations: {total_sims:,} ({total_sims/elapsed:.0f} sims/sec)")
    print(f"Best fitness: {best_ever:+.4f}")

    # For comparison: estimate sequential time
    seq_estimate = total_sims * (sim_steps * 0.002 / 10)  # rough: 10x realtime
    print(f"\nEstimated sequential PyBullet time: ~{seq_estimate:.0f}s ({seq_estimate/60:.0f} min)")
    print(f"MJX speedup: ~{seq_estimate/elapsed:.0f}x")

    return population, fitnesses


if __name__ == "__main__":
    parallel_hill_climber()

"""LLM-Guided Evolution: an LLM steers the evolutionary search.

Based on "LLM Guided Evolution" (Morris et al., 2024) and "Evolution of Thought."
Every K generations, the LLM reviews fitness history, analyzes population
diversity, and suggests adaptive changes to mutation rate, selection pressure,
and which individuals to perturb. The LLM acts as a meta-optimizer that
*reflects* on the evolutionary trajectory.

Requires: NVIDIA_API_KEY environment variable (from build.nvidia.com)
Run: uv run python examples/06_llm_guided_evolution.py --output-dir /data/evo-embodied
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mujoco import mjx
import mujoco
from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent.parent / "models" / "quadruped.xml"

N_POPULATION = 256
N_GENERATIONS = 500
N_HIDDEN = 64
CONTROL_STEPS = 200
PHYSICS_PER_CTRL = 20
MUTATION_SCALE = 0.1
MUTATION_DECAY = 0.9995
MIN_TORSO_HEIGHT = 0.25
LLM_INTERVAL = 25          # consult LLM every N generations
LLM_MODEL = "nvidia/nemotron-3-super-120b-a12b"

# ── LLM Client ─────────────────────────────────────────────────────
def make_llm_client():
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        print("WARNING: NVIDIA_API_KEY not set, LLM guidance disabled", flush=True)
        return None
    return OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)


def llm_guide(client, gen, best_hist, mean_hist, mutation_scale, population_std,
              prev_advice=""):
    """Ask the LLM to analyze evolution progress and suggest adjustments."""
    if client is None:
        return mutation_scale, ""

    recent_best = best_hist[-LLM_INTERVAL:] if len(best_hist) >= LLM_INTERVAL else best_hist
    recent_mean = mean_hist[-LLM_INTERVAL:] if len(mean_hist) >= LLM_INTERVAL else mean_hist

    best_improvement = recent_best[-1] - recent_best[0] if len(recent_best) > 1 else 0
    mean_improvement = recent_mean[-1] - recent_mean[0] if len(recent_mean) > 1 else 0

    prompt = f"""You are an evolutionary optimization expert advising on a walking quadruped evolution.

## Current State (Generation {gen}/{N_GENERATIONS})
- Best fitness: {recent_best[-1]:.4f} (improved {best_improvement:+.4f} over last {len(recent_best)} gens)
- Mean fitness: {recent_mean[-1]:.4f} (improved {mean_improvement:+.4f})
- Current mutation scale: {mutation_scale:.6f}
- Population weight std (diversity): {population_std:.4f}
- Fitness = x_distance - 10*height_penalty - 0.5*y_drift

## Full History (sampled)
Best: {[round(b, 2) for b in best_hist[::max(1, len(best_hist)//10)]]}
Mean: {[round(m, 2) for m in mean_hist[::max(1, len(mean_hist)//10)]]}

## Previous Advice
{prev_advice if prev_advice else "None (first consultation)"}

## Your Task
Analyze the evolutionary trajectory and respond with EXACTLY this JSON format:
{{
  "mutation_scale": <float between 0.001 and 0.5>,
  "reasoning": "<1-2 sentences: what you observe and why you suggest this change>",
  "diagnosis": "<one of: stagnating, exploring, converging, plateaued, improving>"
}}

Rules:
- If fitness is stagnating, INCREASE mutation to escape local optima
- If fitness is improving steadily, keep mutation similar or slightly decrease
- If population diversity is very low, INCREASE mutation significantly
- If best is much higher than mean, the population hasn't caught up — moderate mutation"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        text = response.choices[0].message.content.strip()
        # Extract JSON from response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(text[start:end])
            new_scale = float(result["mutation_scale"])
            new_scale = max(0.001, min(0.5, new_scale))
            advice = f"Gen {gen}: {result.get('diagnosis', '?')} — {result.get('reasoning', '?')}"
            return new_scale, advice
    except Exception as e:
        print(f"  LLM error: {e}", flush=True)

    return mutation_scale, ""


# ── Simulation (same as 05) ────────────────────────────────────────
def build_model():
    mj_model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    mjx_model = mjx.put_model(mj_model)
    return mj_model, mjx_model


def make_evolution_fns(mjx_model, n_sensors, n_motors):
    n_inputs = n_sensors + 2
    n_w = n_inputs * N_HIDDEN + N_HIDDEN * n_motors

    @jax.jit
    def evaluate_one(weights_flat, mjx_model):
        w1 = weights_flat[:n_inputs * N_HIDDEN].reshape(n_inputs, N_HIDDEN)
        w2 = weights_flat[n_inputs * N_HIDDEN:].reshape(N_HIDDEN, n_motors)
        mjx_data = mjx.make_data(mjx_model)
        dt = mjx_model.opt.timestep
        ctrl_dt = dt * PHYSICS_PER_CTRL

        def physics_step(data, _):
            data = mjx.step(mjx_model, data)
            return data, None

        def control_step(carry, ctrl_idx):
            data = carry
            t = ctrl_idx * ctrl_dt
            clock_sin = jnp.sin(2.0 * jnp.pi * 2.0 * t)
            clock_cos = jnp.cos(2.0 * jnp.pi * 2.0 * t)
            sensor_input = jnp.concatenate([data.sensordata, jnp.array([clock_sin, clock_cos])])
            ctrl = jnp.tanh(jnp.tanh(sensor_input @ w1) @ w2)
            data = data.replace(ctrl=ctrl)
            data, _ = jax.lax.scan(physics_step, data, None, length=PHYSICS_PER_CTRL)
            torso_z = data.qpos[2]
            return data, torso_z

        final_data, torso_heights = jax.lax.scan(control_step, mjx_data, jnp.arange(CONTROL_STEPS))
        x_distance = final_data.qpos[0]
        height_penalty = jnp.mean(jnp.maximum(0.0, MIN_TORSO_HEIGHT - torso_heights)) * 10.0
        y_drift = jnp.abs(final_data.qpos[1]) * 0.5
        return x_distance - height_penalty - y_drift

    evaluate_batch = jax.jit(jax.vmap(evaluate_one, in_axes=(0, None)))
    return evaluate_batch, n_w


def evolve(mjx_model, evaluate_batch, n_w, run_dir):
    client = make_llm_client()
    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    population = jax.random.normal(init_key, (N_POPULATION, n_w)) * 0.3

    print("JIT compiling...", flush=True)
    t0 = time.time()
    fitnesses = evaluate_batch(population, mjx_model)
    fitnesses.block_until_ready()
    print(f"JIT done in {time.time() - t0:.1f}s\n", flush=True)

    best_history, mean_history, llm_log = [], [], []
    mutation_scale = MUTATION_SCALE
    prev_advice = ""

    print(f"{'Gen':>5s}  {'Best':>8s}  {'Mean':>8s}  {'Mut σ':>8s}  {'LLM':>6s}", flush=True)
    print("-" * 44, flush=True)

    t_start = time.time()
    for gen in range(N_GENERATIONS):
        # LLM consultation
        llm_tag = ""
        if gen > 0 and gen % LLM_INTERVAL == 0:
            pop_std = float(jnp.std(population))
            new_scale, advice = llm_guide(
                client, gen, best_history, mean_history, mutation_scale, pop_std, prev_advice
            )
            if advice:
                old_scale = mutation_scale
                mutation_scale = new_scale
                prev_advice = advice
                llm_log.append({"gen": gen, "old_scale": old_scale, "new_scale": new_scale, "advice": advice})
                llm_tag = "<<LLM"
                print(f"  LLM: {advice}", flush=True)

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

        if not llm_tag:
            mutation_scale *= MUTATION_DECAY

        if (gen + 1) % 50 == 0 or gen == 0 or llm_tag:
            print(f"{gen+1:5d}  {gb:+8.4f}  {gm:+8.4f}  {mutation_scale:8.5f}  {llm_tag}", flush=True)

        if (gen + 1) % 100 == 0:
            best_idx = int(jnp.argmax(fitnesses))
            np.save(run_dir / f"weights_gen{gen+1:04d}.npy", np.array(population[best_idx]))

    elapsed = time.time() - t_start
    best_idx = int(jnp.argmax(fitnesses))
    best_weights = np.array(population[best_idx])

    # Save results
    np.save(run_dir / "best_weights.npy", best_weights)
    np.save(run_dir / "population.npy", np.array(population))
    np.savez(run_dir / "history.npz", best=np.array(best_history), mean=np.array(mean_history))
    (run_dir / "llm_guidance_log.json").write_text(json.dumps(llm_log, indent=2))

    summary = {
        "experiment": "llm_guided_evolution",
        "jit_time_s": round(time.time() - t0, 1),
        "evolution_time_s": round(elapsed, 1),
        "total_simulations": N_POPULATION * N_GENERATIONS,
        "best_fitness": round(float(fitnesses[best_idx]), 4),
        "mean_fitness": round(float(fitnesses.mean()), 4),
        "llm_consultations": len(llm_log),
        "llm_model": LLM_MODEL,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nBest fitness: {float(fitnesses[best_idx]):+.4f}", flush=True)
    print(f"LLM consultations: {len(llm_log)}", flush=True)
    print(f"Results saved to {run_dir}/", flush=True)
    return best_weights, best_history, mean_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/data/evo-embodied")
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_dir) / f"{ts}-llm-guided"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}", flush=True)

    mj_model, mjx_model = build_model()
    n_s, n_m = mj_model.nsensordata, mj_model.nu
    evaluate_batch, n_w = make_evolution_fns(mjx_model, n_s, n_m)
    print(f"Network: {n_s}+2 → {N_HIDDEN} → {n_m} ({n_w} weights)", flush=True)

    evolve(mjx_model, evaluate_batch, n_w, run_dir)

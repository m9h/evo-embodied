"""LLaMEA: LLM-generated evolutionary strategies.

Based on "LLaMEA: A Large Language Model Evolutionary Algorithm" (van Stein & Bäck, 2024).
The LLM generates Python code for the evolutionary strategy itself — mutation operators,
selection mechanisms, and population management. Each strategy is evaluated by running
evolution for N generations, and the LLM iterates on the code based on results.

Meta-evolution: evolving the algorithm that evolves robots.

Requires: NVIDIA_API_KEY environment variable
Run: uv run python examples/08_llamea_strategy.py --output-dir /data/evo-embodied
"""
import argparse
import json
import os
import time
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import numpy as np
import mujoco
from mujoco import mjx
from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent.parent / "models" / "quadruped.xml"

N_STRATEGIES = 6            # how many strategies to try
N_POPULATION = 128
N_GENERATIONS_EVAL = 100    # generations per strategy evaluation
N_HIDDEN = 64
CONTROL_STEPS = 200
PHYSICS_PER_CTRL = 20
MIN_TORSO_HEIGHT = 0.25
LLM_MODEL = "nvidia/nemotron-3-super-120b-a12b"

# ── LLM Client ─────────────────────────────────────────────────────
def make_llm_client():
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        raise SystemExit("NVIDIA_API_KEY required")
    return OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)


# ── Fitness Evaluation (shared by all strategies) ──────────────────
def build_evaluator():
    mj_model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    mjx_model = mjx.put_model(mj_model)
    n_s = mj_model.nsensordata
    n_m = mj_model.nu
    n_inputs = n_s + 2
    n_w = n_inputs * N_HIDDEN + N_HIDDEN * n_m

    @jax.jit
    def evaluate_one(weights_flat, mjx_model):
        w1 = weights_flat[:n_inputs * N_HIDDEN].reshape(n_inputs, N_HIDDEN)
        w2 = weights_flat[n_inputs * N_HIDDEN:].reshape(N_HIDDEN, n_m)
        mjx_data = mjx.make_data(mjx_model)
        dt = mjx_model.opt.timestep
        ctrl_dt = dt * PHYSICS_PER_CTRL

        def physics_step(data, _):
            data = mjx.step(mjx_model, data)
            return data, None

        def control_step(carry, ctrl_idx):
            data = carry
            t = ctrl_idx * ctrl_dt
            sensor_input = jnp.concatenate([
                data.sensordata,
                jnp.array([jnp.sin(2*jnp.pi*2*t), jnp.cos(2*jnp.pi*2*t)])
            ])
            ctrl = jnp.tanh(jnp.tanh(sensor_input @ w1) @ w2)
            data = data.replace(ctrl=ctrl)
            data, _ = jax.lax.scan(physics_step, data, None, length=PHYSICS_PER_CTRL)
            return data, data.qpos[2]

        final_data, torso_heights = jax.lax.scan(control_step, mjx_data, jnp.arange(CONTROL_STEPS))
        x_dist = final_data.qpos[0]
        h_pen = jnp.mean(jnp.maximum(0.0, MIN_TORSO_HEIGHT - torso_heights)) * 10.0
        y_drift = jnp.abs(final_data.qpos[1]) * 0.5
        return x_dist - h_pen - y_drift

    evaluate_batch = jax.jit(jax.vmap(evaluate_one, in_axes=(0, None)))
    return mjx_model, evaluate_batch, n_w


# ── Baseline Strategy ─────────────────────────────────────────────
BASELINE_STRATEGY = textwrap.dedent("""\
def evolve_step(key, population, fitnesses, gen, config):
    \"\"\"Parallel hill climber with adaptive mutation.\"\"\"
    key, mk = jax.random.split(key)
    mutation_scale = config['initial_mutation'] * (config['decay'] ** gen)
    mutations = jax.random.normal(mk, population.shape) * mutation_scale
    candidates = population + mutations
    return key, candidates

def select(population, fitnesses, candidates, cand_fitnesses, config):
    \"\"\"Keep improvements.\"\"\"
    improved = cand_fitnesses > fitnesses
    new_pop = jnp.where(improved[:, None], candidates, population)
    new_fit = jnp.where(improved, cand_fitnesses, fitnesses)
    return new_pop, new_fit
""")


def propose_strategy(client, iteration, previous_results):
    """Ask LLM to generate a new evolutionary strategy as Python code."""
    results_summary = ""
    if previous_results:
        results_summary = "## Previous Strategies and Results\n"
        for r in previous_results:
            results_summary += f"\n### {r['name']} — fitness: {r['best_fitness']:.2f}\n"
            results_summary += f"```python\n{r['code'][:500]}\n```\n"

    prompt = f"""You are an evolutionary algorithm researcher. Generate a novel evolutionary strategy as Python code.

## Available Imports (already imported)
- jax, jax.numpy as jnp, jax.random
- numpy as np

## Required Interface
You MUST define exactly these two functions:

```python
def evolve_step(key, population, fitnesses, gen, config):
    \"\"\"Generate candidate solutions from current population.

    Args:
        key: JAX PRNGKey
        population: jnp.array shape (pop_size, n_weights)
        fitnesses: jnp.array shape (pop_size,)
        gen: int, current generation number
        config: dict with 'initial_mutation' (0.1) and 'decay' (0.999)

    Returns:
        key: updated PRNGKey
        candidates: jnp.array shape (pop_size, n_weights)
    \"\"\"
    ...

def select(population, fitnesses, candidates, cand_fitnesses, config):
    \"\"\"Select next generation from parents + candidates.

    Returns:
        new_population: jnp.array shape (pop_size, n_weights)
        new_fitnesses: jnp.array shape (pop_size,)
    \"\"\"
    ...
```

## Design Principles
- ALL operations must use JAX (jnp, jax.random) — no Python loops over individuals
- Population shape is (pop_size, n_weights) where n_weights ~= 4000
- Higher fitness is better
- Be creative: try tournament selection, crossover, differential evolution, CMA-ES ideas, novelty search, etc.

{results_summary}

## Iteration {iteration + 1}/{N_STRATEGIES}
Generate a NOVEL strategy different from all previous ones. Respond with:

NAME: <short_snake_case_name>
IDEA: <1-2 sentences explaining the approach>
```python
<your code here>
```"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=1500,
        )
        text = response.choices[0].message.content

        # Extract code
        code_blocks = []
        in_block = False
        current = []
        for line in text.split("\n"):
            if line.strip().startswith("```python"):
                in_block = True
                current = []
            elif line.strip() == "```" and in_block:
                in_block = False
                code_blocks.append("\n".join(current))
            elif in_block:
                current.append(line)

        if not code_blocks:
            return None, "failed", "No code block found", ""

        code = code_blocks[0]

        # Extract metadata
        name = f"strategy_{iteration}"
        idea = "LLM-generated strategy"
        for line in text.split("\n"):
            if line.strip().startswith("NAME:"):
                name = line.split("NAME:")[-1].strip().replace(" ", "_")[:30]
            if line.strip().startswith("IDEA:"):
                idea = line.split("IDEA:")[-1].strip()[:200]

        # Validate code compiles
        compile(code, f"<strategy_{name}>", "exec")
        return code, name, idea, text

    except Exception as e:
        return None, "failed", f"Error: {e}", ""


def run_strategy(code, mjx_model, evaluate_batch, n_w, name):
    """Execute a strategy for N_GENERATIONS_EVAL and return best fitness."""
    # Create a namespace with the strategy functions
    namespace = {"jax": jax, "jnp": jnp, "np": np}
    try:
        exec(code, namespace)
    except Exception as e:
        print(f"    Code exec failed: {e}", flush=True)
        return -999.0, []

    evolve_step = namespace.get("evolve_step")
    select_fn = namespace.get("select")
    if not evolve_step or not select_fn:
        print("    Missing evolve_step or select function", flush=True)
        return -999.0, []

    config = {"initial_mutation": 0.1, "decay": 0.999}
    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    population = jax.random.normal(init_key, (N_POPULATION, n_w)) * 0.3
    fitnesses = evaluate_batch(population, mjx_model)
    fitnesses.block_until_ready()

    best_hist = []
    for gen in range(N_GENERATIONS_EVAL):
        try:
            key, candidates = evolve_step(key, population, fitnesses, gen, config)
            cand_fit = evaluate_batch(candidates, mjx_model)
            population, fitnesses = select_fn(population, fitnesses, candidates, cand_fit, config)

            best_hist.append(float(fitnesses.max()))
            if (gen + 1) % 25 == 0:
                print(f"    Gen {gen+1:4d}: best={float(fitnesses.max()):+.2f}", flush=True)
        except Exception as e:
            print(f"    Strategy crashed at gen {gen}: {e}", flush=True)
            return max(best_hist) if best_hist else -999.0, best_hist

    return float(fitnesses.max()), best_hist


# ── Main Loop ─────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/data/evo-embodied")
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_dir) / f"{ts}-llamea"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}", flush=True)

    client = make_llm_client()
    mjx_model, evaluate_batch, n_w = build_evaluator()

    # JIT warmup
    print("JIT compiling...", flush=True)
    dummy = jax.random.normal(jax.random.PRNGKey(0), (N_POPULATION, n_w)) * 0.3
    _ = evaluate_batch(dummy, mjx_model)
    _.block_until_ready()
    print("JIT done.\n", flush=True)

    # Baseline strategy
    print("=== Baseline: parallel hill climber ===", flush=True)
    baseline_fitness, baseline_hist = run_strategy(BASELINE_STRATEGY, mjx_model, evaluate_batch, n_w, "baseline")
    print(f"  Baseline fitness: {baseline_fitness:+.2f}\n", flush=True)

    results = [{
        "name": "baseline_hill_climber",
        "idea": "Parallel hill climber with adaptive Gaussian mutation",
        "best_fitness": baseline_fitness,
        "code": BASELINE_STRATEGY,
        "history": baseline_hist,
    }]

    # LLM-generated strategies
    for i in range(N_STRATEGIES):
        print(f"\n=== Strategy {i+1}/{N_STRATEGIES}: asking LLM... ===", flush=True)
        code, name, idea, raw = propose_strategy(client, i, results)

        if code is None:
            print(f"  SKIP: {idea}", flush=True)
            results.append({"name": name, "idea": idea, "best_fitness": -999, "code": "", "history": []})
            continue

        print(f"  Name: {name}", flush=True)
        print(f"  Idea: {idea}", flush=True)
        (run_dir / f"{name}.py").write_text(code)

        fitness, hist = run_strategy(code, mjx_model, evaluate_batch, n_w, name)
        print(f"  Fitness: {fitness:+.2f} (baseline: {baseline_fitness:+.2f})", flush=True)

        results.append({
            "name": name, "idea": idea, "best_fitness": fitness,
            "code": code, "history": hist,
        })

    # Final summary
    results.sort(key=lambda r: r["best_fitness"], reverse=True)
    print("\n" + "=" * 60, flush=True)
    print("STRATEGY RESULTS (ranked)", flush=True)
    print("=" * 60, flush=True)
    for r in results:
        marker = " *** WINNER ***" if r == results[0] else ""
        print(f"  {r['best_fitness']:+8.2f}  {r['name']:30s}  {r.get('idea', '')[:50]}{marker}", flush=True)

    # Save without code in summary (code saved as separate files)
    summary_results = [{k: v for k, v in r.items() if k != "code"} for r in results]
    (run_dir / "results.json").write_text(json.dumps(summary_results, indent=2, default=str))
    print(f"\nAll results saved to {run_dir}/", flush=True)

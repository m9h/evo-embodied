"""RoboMorph: LLM-driven morphology evolution.

Based on "RoboMorph: Evolving Robot Morphology using Large Language Models" (2024).
The LLM proposes new body designs (MuJoCo XML), each gets a short evolution run
to evaluate its potential, and the LLM iterates based on results.

Two nested loops:
  Outer: LLM proposes morphology variations (body plan)
  Inner: GPU-accelerated evolution of controllers for each morphology

Requires: NVIDIA_API_KEY environment variable
Run: uv run python examples/07_robomorph.py --output-dir /data/evo-embodied
"""
import argparse
import json
import os
import time
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mujoco
from mujoco import mjx
from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────
N_MORPHOLOGIES = 8          # how many body plans to try
N_POPULATION = 128          # smaller pop for faster inner loop
N_GENERATIONS_INNER = 100   # short evolution per morphology
N_HIDDEN = 32
CONTROL_STEPS = 100
PHYSICS_PER_CTRL = 20
MUTATION_SCALE = 0.1
MUTATION_DECAY = 0.999
MIN_TORSO_HEIGHT = 0.2
LLM_MODEL = "nvidia/nemotron-3-super-120b-a12b"

BASELINE_XML = (Path(__file__).parent.parent / "models" / "quadruped.xml").read_text()

# ── LLM Morphology Generator ──────────────────────────────────────
def make_llm_client():
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        print("ERROR: NVIDIA_API_KEY required for morphology generation", flush=True)
        return None
    return OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)


def propose_morphology(client, iteration, previous_results):
    """Ask the LLM to propose a new robot morphology as MuJoCo XML."""
    results_summary = ""
    if previous_results:
        results_summary = "## Previous Morphology Results\n"
        for r in previous_results:
            results_summary += f"- {r['name']}: fitness={r['best_fitness']:.2f}, {r['description']}\n"
        results_summary += "\nBuild on what worked. Avoid repeating what didn't.\n"

    prompt = f"""You are a robot morphology designer. Generate a MuJoCo MJCF XML for a walking robot.

## Rules
- The robot must be a plausible walking creature with a torso and legs
- The torso body MUST be named "torso" and have a freejoint named "root"
- Each joint needs a matching motor in <actuator> with reasonable gear ratios
- Each joint needs a <jointpos> sensor in <sensor>
- Include a <framepos objtype="body" objname="torso"/> sensor
- Use timestep="0.002" and gravity="0 0 -9.81"
- Joint ranges should be physically reasonable (e.g., -60 to 60 degrees)
- Keep total mass between 3 and 20 kg
- Include a ground plane and light

## Iteration {iteration + 1}/{N_MORPHOLOGIES}

{results_summary}

## Design Brief
Try something different from a standard quadruped. Consider:
- Hexapod (6 legs) — more stable
- Asymmetric legs (front longer than back)
- Three-segment legs (hip + knee + ankle)
- Wide vs narrow stance
- Heavy torso vs light torso
- Different joint axes (add lateral hip joints)

Respond with EXACTLY this format:
```xml
<mujoco model="your_model_name">
...complete valid MJCF XML...
</mujoco>
```

DESCRIPTION: <one line describing the morphology>
NAME: <short snake_case name>"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000,
        )
        text = response.choices[0].message.content

        # Extract XML
        xml_start = text.find("<mujoco")
        xml_end = text.find("</mujoco>") + len("</mujoco>")
        if xml_start < 0 or xml_end <= xml_start:
            return None, "failed", "XML extraction failed"
        xml = text[xml_start:xml_end]

        # Extract name and description
        name = "morph_" + str(iteration)
        desc = "LLM-generated morphology"
        for line in text.split("\n"):
            if line.strip().startswith("NAME:"):
                name = line.split("NAME:")[-1].strip().replace(" ", "_")[:30]
            if line.strip().startswith("DESCRIPTION:"):
                desc = line.split("DESCRIPTION:")[-1].strip()[:100]

        # Validate XML by trying to load it
        mujoco.MjModel.from_xml_string(xml)
        return xml, name, desc

    except Exception as e:
        return None, "failed", f"Error: {e}"


# ── Inner Evolution Loop ──────────────────────────────────────────
def evaluate_morphology(xml_string, run_dir, morph_name):
    """Run a short evolution on a given morphology. Returns best fitness."""
    try:
        mj_model = mujoco.MjModel.from_xml_string(xml_string)
        mjx_model = mjx.put_model(mj_model)
    except Exception as e:
        print(f"    Model load failed: {e}", flush=True)
        return -999.0, {}

    n_s = mj_model.nsensordata
    n_m = mj_model.nu
    if n_s == 0 or n_m == 0:
        print(f"    No sensors ({n_s}) or motors ({n_m})", flush=True)
        return -999.0, {}

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

    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    population = jax.random.normal(init_key, (N_POPULATION, n_w)) * 0.3

    # JIT warmup
    fitnesses = evaluate_batch(population, mjx_model)
    fitnesses.block_until_ready()

    mutation_scale = MUTATION_SCALE
    best_hist = []
    for gen in range(N_GENERATIONS_INNER):
        key, mk = jax.random.split(key)
        mutations = jax.random.normal(mk, population.shape) * mutation_scale
        candidates = population + mutations
        cand_fit = evaluate_batch(candidates, mjx_model)
        improved = cand_fit > fitnesses
        population = jnp.where(improved[:, None], candidates, population)
        fitnesses = jnp.where(improved, cand_fit, fitnesses)
        mutation_scale *= MUTATION_DECAY
        best_hist.append(float(fitnesses.max()))

        if (gen + 1) % 25 == 0:
            print(f"    Gen {gen+1:4d}: best={float(fitnesses.max()):+.2f} mean={float(fitnesses.mean()):+.2f}", flush=True)

    best_fitness = float(fitnesses.max())
    best_idx = int(jnp.argmax(fitnesses))
    np.save(run_dir / f"{morph_name}_weights.npy", np.array(population[best_idx]))

    info = {
        "n_sensors": n_s, "n_motors": n_m, "n_weights": n_w,
        "best_fitness": round(best_fitness, 4),
        "mean_fitness": round(float(fitnesses.mean()), 4),
    }
    return best_fitness, info


# ── Main Loop ─────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/data/evo-embodied")
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_dir) / f"{ts}-robomorph"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}", flush=True)

    client = make_llm_client()
    if not client:
        raise SystemExit("NVIDIA_API_KEY required")

    # First evaluate the baseline quadruped
    print("=== Baseline: standard quadruped ===", flush=True)
    baseline_fitness, baseline_info = evaluate_morphology(BASELINE_XML, run_dir, "baseline_quadruped")
    print(f"  Baseline fitness: {baseline_fitness:+.2f}\n", flush=True)

    results = [{
        "name": "baseline_quadruped",
        "description": "Standard 4-leg quadruped, 2-segment legs, 8 motors",
        "best_fitness": baseline_fitness,
        **baseline_info,
    }]

    # Save baseline XML
    (run_dir / "baseline_quadruped.xml").write_text(BASELINE_XML)

    # Iterate with LLM-proposed morphologies
    for i in range(N_MORPHOLOGIES):
        print(f"\n=== Morphology {i+1}/{N_MORPHOLOGIES}: asking LLM... ===", flush=True)
        xml, name, desc = propose_morphology(client, i, results)

        if xml is None:
            print(f"  SKIP: {desc}", flush=True)
            results.append({"name": name, "description": desc, "best_fitness": -999})
            continue

        print(f"  Name: {name}", flush=True)
        print(f"  Description: {desc}", flush=True)
        (run_dir / f"{name}.xml").write_text(xml)

        fitness, info = evaluate_morphology(xml, run_dir, name)
        print(f"  Fitness: {fitness:+.2f} (baseline: {baseline_fitness:+.2f})", flush=True)

        results.append({"name": name, "description": desc, "best_fitness": fitness, **info})

    # Final summary
    results.sort(key=lambda r: r["best_fitness"], reverse=True)
    print("\n" + "=" * 60, flush=True)
    print("MORPHOLOGY RESULTS (ranked)", flush=True)
    print("=" * 60, flush=True)
    for r in results:
        marker = " *** WINNER ***" if r == results[0] else ""
        print(f"  {r['best_fitness']:+8.2f}  {r['name']:30s}  {r.get('description', '')}{marker}", flush=True)

    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nAll results saved to {run_dir}/", flush=True)

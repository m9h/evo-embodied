"""Check example 03's float32 batch/individual fitness agreement.

Defaults cover two population sizes, three seeds and pre/post-contact horizons.
Use --steps 500 for the longer-horizon diagnostic (currently fails on both the
old and modern dependency stacks). No training or tolerance adjustment occurs.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import jax
import mujoco
import numpy as np
from mujoco import mjx


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/03_mjx_parallel_evolution.py"
ATOL, RTOL = 1e-6, 1e-5


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_batches(model_path, steps=(16, 128), seeds=(0, 1, 42), populations=(2, 4)):
    if jax.config.x64_enabled:
        raise ValueError("This gate must run in float32, with JAX_ENABLE_X64=false")
    spec = importlib.util.spec_from_file_location("parallel_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    mm = mjx.put_model(model, impl="jax")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    distances = [float(c.dist) for c in data.contact]
    reset_valid = all(distance >= -1e-8 for distance in distances)
    cases = []
    for horizon in steps:
        one, batch = module.make_evaluate_fn(mm, model.nsensordata, model.nu, 8, horizon)
        for population in populations:
            for seed in seeds:
                weights = jax.random.normal(jax.random.key(seed),
                    (population, 8 * (model.nsensordata + model.nu))) * 0.1
                actual = np.asarray(batch(weights, mm))
                repeat = np.asarray(batch(weights, mm))
                expected = np.asarray([one(w, mm) for w in weights])
                finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
                equivalent = bool(np.allclose(actual, expected,
                    atol=ATOL, rtol=RTOL, equal_nan=False))
                repeat_exact = bool(np.array_equal(actual, repeat))
                case = {
                    "steps": horizon, "seed": seed, "population": population,
                    "weights_sha256": hashlib.sha256(np.asarray(weights).tobytes()).hexdigest(),
                    "finite": finite, "within_tolerance": equivalent,
                    "repeat_exact": repeat_exact,
                    "batch_fitness": actual.tolist() if finite else None,
                    "individual_fitness": expected.tolist() if finite else None,
                    "max_abs_error": float(np.max(np.abs(actual - expected))) if finite else None,
                    "passed": finite and equivalent and repeat_exact,
                }
                cases.append(case)
                print(f"steps={horizon} population={population} seed={seed}: "
                      f"{'PASS' if case['passed'] else 'FAIL'} "
                      f"max_abs_error={case['max_abs_error']}", flush=True)
    return {
        "model_sha256": sha256(model_path), "example_sha256": sha256(EXAMPLE),
        "atol": ATOL, "rtol": RTOL, "dtype": "float32",
        "reset_nonpenetrating": reset_valid,
        "initial_min_contact_distance_m": min(distances, default=None),
        "cases": cases, "passed": reset_valid and all(c["passed"] for c in cases),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/quadruped.xml")
    parser.add_argument("--steps", type=int, action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--population", type=int, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    steps, seeds, populations = args.steps or [16, 128], args.seed or [0, 1, 42], args.population or [2, 4]
    if args.output.exists() or min(steps + populations) < 1 or min(seeds) < 0:
        parser.error("Use a new output path, positive steps/populations and nonnegative seeds")
    jax.config.update("jax_enable_x64", False)
    report = {
        "schema_version": 1, "scope": "float32 evaluator agreement, not research validation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(),
        "runner_sha256": sha256(__file__), "lock_sha256": sha256(ROOT / "uv.lock"),
        "platform": platform.platform(), "python": platform.python_version(),
        "backend": jax.default_backend(), "devices": [str(d) for d in jax.devices()],
        "versions": {name: importlib.metadata.version(name) for name in
                     ("jax", "jaxlib", "mujoco", "mujoco-mjx", "numpy")},
        "model": str(args.model),
        **check_batches(args.model, steps, seeds, populations),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

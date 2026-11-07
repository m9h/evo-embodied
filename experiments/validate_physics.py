"""Small CPU MuJoCo / MJX-JAX replay check; not a training benchmark.

Run from the repository root with the existing project environment:
    JAX_PLATFORMS=cpu .venv/bin/python experiments/validate_physics.py \
        --output /tmp/evo-physics.json

The JSON records model/runner hashes, versions, tolerances, contacts, repeatability,
and synchronized timings. Full research validation is in docs/VALIDATION_PLAN.md.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("qpos", "qvel", "sensordata")
ABS_TOL = {"qpos": 1e-6, "qvel": 1e-5, "sensordata": 1e-4}
REL_TOL = 1e-5


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_value(*args):
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True
    ).strip()


def check_model(path, steps, solver_tolerance=None):
    model = mujoco.MjModel.from_xml_path(str(path))
    if solver_tolerance is not None:
        model.opt.tolerance = solver_tolerance
    mjx_model = mjx.put_model(model, impl="jax")
    initial = mjx.make_data(mjx_model, impl="jax")

    @jax.jit
    def rollout(actions):
        def step(data, ctrl):
            data = mjx.step(mjx_model, data.replace(ctrl=ctrl))
            return data, tuple(getattr(data, field) for field in FIELDS)

        return jax.lax.scan(step, initial, actions)[1]

    inspection = mujoco.MjData(model)
    mujoco.mj_forward(model, inspection)
    initial_distances = [float(c.dist) for c in inspection.contact]
    reset_valid = all(distance >= -1e-8 for distance in initial_distances)
    t = np.arange(steps)[:, None] * model.opt.timestep
    phase = np.arange(model.nu)[None, :] * np.pi / 4
    inputs = {
        "zero": np.zeros((steps, model.nu)),
        "sine": 0.05 * np.sin(2 * np.pi * t + phase),
    }
    # Respect the model's declared control limits without changing its dynamics.
    limited = model.actuator_ctrllimited.astype(bool)
    for actions in inputs.values():
        actions[:, limited] = np.clip(
            actions[:, limited],
            model.actuator_ctrlrange[limited, 0],
            model.actuator_ctrlrange[limited, 1],
        )

    started = time.perf_counter()
    compiled = rollout.lower(jnp.asarray(inputs["zero"])).compile()
    compile_s = time.perf_counter() - started
    cases = []
    for name, actions in inputs.items():
        data = mujoco.MjData(model)
        reference = {field: [] for field in FIELDS}
        max_contacts = 0
        started = time.perf_counter()
        for ctrl in actions:
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
            max_contacts = max(max_contacts, data.ncon)
            for field in FIELDS:
                reference[field].append(getattr(data, field).copy())
        cpu_s = time.perf_counter() - started
        actions_device = jnp.asarray(actions)
        jax.block_until_ready(compiled(actions_device))  # warm before timing
        started = time.perf_counter()
        actual_device = jax.block_until_ready(compiled(actions_device))
        mjx_s = time.perf_counter() - started
        repeat_device = jax.block_until_ready(compiled(actions_device))
        checks = {}
        for field, actual, repeat in zip(FIELDS, actual_device, repeat_device):
            ref = np.asarray(reference[field])
            actual, repeat = np.asarray(actual), np.asarray(repeat)
            finite = bool(np.isfinite(ref).all() and np.isfinite(actual).all())
            bad_steps = np.flatnonzero(~np.all(np.isclose(
                actual, ref, atol=ABS_TOL[field], rtol=REL_TOL
            ), axis=1))
            checks[field] = {
                "finite": finite,
                "max_abs_error": float(np.max(np.abs(actual - ref), initial=0))
                if finite else None,
                "within_tolerance": finite and bool(np.allclose(
                    actual, ref, atol=ABS_TOL[field], rtol=REL_TOL
                )),
                "repeat_exact": bool(np.array_equal(actual, repeat)),
                "first_failing_step": int(bad_steps[0]) + 1 if bad_steps.size else None,
            }
        warning_counts = [int(w.number) for w in data.warning]
        passed = not any(warning_counts) and all(
            c["within_tolerance"] and c["repeat_exact"] for c in checks.values()
        )
        cases.append({
            "input": name,
            "actions_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
            "checks": checks,
            "cpu_warning_counts": warning_counts,
            "cpu_max_contacts": max_contacts,
            "cpu_rollout_s": cpu_s,
            "mjx_warm_rollout_s": mjx_s,
            "passed": passed,
        })
    return {
        "model": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "model_sha256": sha256(path),
        "steps": steps,
        "timestep_s": float(model.opt.timestep),
        "solver_tolerance": float(model.opt.tolerance),
        "solver_tolerance_override": solver_tolerance,
        "duration_s": steps * float(model.opt.timestep),
        "nq": model.nq, "nv": model.nv, "nu": model.nu,
        "initial_min_contact_distance_m": min(initial_distances, default=None),
        "reset_nonpenetrating": reset_valid,
        "compile_s": compile_s,
        "cases": cases,
        "passed": reset_valid and all(case["passed"] for case in cases),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", type=Path,
                        help="Repeat for several XMLs; default: models/quadruped.xml")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--solver-tolerance", type=float,
                        help="Diagnostic convergence override; does not change comparison tolerances")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.solver_tolerance is not None and (
        not np.isfinite(args.solver_tolerance) or args.solver_tolerance < 0
    ):
        parser.error("--solver-tolerance must be finite and nonnegative")
    if args.output.exists():
        parser.error("--output already exists; choose a new result path")
    # Reference precision. Float32/GPU validation is a separate benchmark gate.
    jax.config.update("jax_enable_x64", True)
    versions = {}
    for package in ("jax", "jaxlib", "mujoco", "mujoco-mjx", "numpy"):
        versions[package] = importlib.metadata.version(package)
    report = {
        "schema_version": 1,
        "scope": "short open-loop replay; not paper reproduction or throughput evidence",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status": git_value("status", "--porcelain"),
        "runner_sha256": sha256(__file__),
        "lock_sha256": sha256(ROOT / "uv.lock"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": versions,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "x64_enabled": bool(jax.config.x64_enabled),
        "atol": ABS_TOL,
        "rtol": REL_TOL,
        "models": [],
    }
    for path in args.model or [ROOT / "models/quadruped.xml"]:
        path = path.resolve()
        print(f"Checking {path.name} ({args.steps} steps)...", flush=True)
        try:
            result = check_model(path, args.steps, args.solver_tolerance)
        except Exception as exc:
            result = {"model": str(path), "passed": False,
                      "error": f"{type(exc).__name__}: {exc}"}
        report["models"].append(result)
        print(f"  {'PASS' if result['passed'] else 'FAIL'}", flush=True)
    report["passed"] = all(model["passed"] for model in report["models"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Results: {args.output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

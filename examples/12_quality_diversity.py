"""QDax MAP-Elites and equal-budget random search on the upstream planar arm.

Run: uv run --locked --extra qd python examples/12_quality_diversity.py \
    --output /tmp/qd-arm.json

This analytic forward-kinematics task teaches a behavior archive independently
of the open MuJoCo replay gates. It is not a locomotion or co-design benchmark.
Fitness is negative standard deviation of normalized joint angles; descriptors
are normalized end-effector x/y. Both methods maximize fitness, share the initial
population and grid, and pay for initialization plus every candidate evaluation.
"""

import argparse
import functools
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def run_comparison(seed=0, population=32, generations=5, joints=6, bins=8):
    from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids
    from qdax.core.emitters.mutation_operators import isoline_variation
    from qdax.core.emitters.standard_emitters import MixingEmitter
    from qdax.core.map_elites import MAPElites
    from qdax.tasks.arm import arm_scoring_function
    from qdax.utils.metrics import default_qd_metrics

    # Independent kinematic fixtures catch descriptor/fitness convention changes.
    fixtures = jnp.array([[0.5] * 4, [0.75, 0.5, 0.5, 0.5]])
    fitness, descriptors, _ = arm_scoring_function(fixtures, jax.random.key(0))
    np.testing.assert_allclose(descriptors, [[1.0, 0.5], [0.5, 1.0]], atol=1e-6)
    np.testing.assert_allclose(fitness, [0.0, -np.sqrt(3) / 16], atol=1e-6)

    centroids = compute_euclidean_centroids((bins, bins), 0.0, 1.0)
    emitter = MixingEmitter(
        mutation_fn=lambda x, key: (x, {}),
        variation_fn=functools.partial(isoline_variation,
            iso_sigma=0.05, line_sigma=0.1, minval=0.0, maxval=1.0),
        variation_percentage=1.0, batch_size=population,
    )
    # Fitness lies in [-0.5, 0]; a fixed offset makes occupied-cell QD scores positive.
    metrics_fn = functools.partial(default_qd_metrics, qd_offset=0.5)
    algorithm = MAPElites(arm_scoring_function, emitter, metrics_fn)
    key = jax.random.key(seed)
    init_key, search_key, random_key = jax.random.split(key, 3)
    initial = jax.random.uniform(init_key, (population, joints))
    archive, state, metrics = algorithm.init(initial, centroids, init_key)
    random_archive = archive  # Immutable repertoire: both start with identical elites.

    def snapshot(repertoire):
        return {name: float(value) for name, value in metrics_fn(repertoire).items()}

    histories = {"map_elites": [snapshot(archive)], "random_search": [snapshot(archive)]}
    update = jax.jit(algorithm.update)

    @jax.jit
    def random_update(repertoire, rng):
        candidates = jax.random.uniform(rng, (population, joints))
        fits, descs, extra = arm_scoring_function(candidates, rng)
        return repertoire.add(candidates, descs, fits, extra)

    for _ in range(generations):
        search_key, step_key = jax.random.split(search_key)
        archive, state, metrics = update(archive, state, step_key)
        random_key, step_key = jax.random.split(random_key)
        random_archive = random_update(random_archive, step_key)
        histories["map_elites"].append(snapshot(archive))
        histories["random_search"].append(snapshot(random_archive))

    methods = {}
    for name, repertoire in [("map_elites", archive), ("random_search", random_archive)]:
        # QDax stores scalar fitness as a (cells, 1) column.
        archive_fitness = np.asarray(repertoire.fitnesses)[:, 0]
        occupied = archive_fitness != -np.inf
        assert occupied.any(), "An archive must contain evaluated elites"
        genomes = np.asarray(repertoire.genotypes)[occupied]
        fits = archive_fitness[occupied]
        descs = np.asarray(repertoire.descriptors)[occupied]
        assert np.isfinite(genomes).all() and np.isfinite(fits).all() and np.isfinite(descs).all()
        assert ((genomes >= 0) & (genomes <= 1)).all()
        assert ((descs >= -1e-6) & (descs <= 1 + 1e-6)).all()
        # Replay the saved elites, not merely a metric computed from the same archive.
        replay_fits, replay_descs, _ = arm_scoring_function(jnp.asarray(genomes), key)
        np.testing.assert_allclose(replay_fits, fits, atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(replay_descs, descs, atol=1e-6, rtol=1e-5)
        for metric in ("coverage", "qd_score", "max_fitness"):
            values = np.array([item[metric] for item in histories[name]])
            assert np.isfinite(values).all()
            assert (np.diff(values) >= -1e-5).all(), f"Elitism violated: {name}/{metric}"
        methods[name] = {
            "candidate_evaluations": population * (generations + 1),
            "validation_evaluations": int(occupied.sum()),
            "history": histories[name],
            "elites": [
                {"cell": int(cell), "genotype": g.tolist(),
                 "fitness": float(f), "descriptor": d.tolist()}
                for cell, g, f, d in zip(np.flatnonzero(occupied), genomes, fits, descs)
            ],
        }
    return {
        "config": {"seed": seed, "population": population, "generations": generations,
                   "joints": joints, "grid": [bins, bins], "qd_offset": 0.5,
                   "iso_sigma": 0.05, "line_sigma": 0.1},
        "initial_population_sha256": hashlib.sha256(np.asarray(initial).tobytes()).hexdigest(),
        "methods": methods,
        "checks": {"kinematic_fixtures": True, "elite_replay": True,
                   "finite_occupied_archive": True, "elitism": True},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--joints", type=int, default=6)
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = args.seed if args.seed is not None else [0]
    if min(seeds) < 0 or min(args.population, args.joints, args.bins) < 1 or args.generations < 0:
        parser.error("Seeds/generations must be nonnegative; population/joints/bins positive")
    if args.output.exists():
        parser.error("Choose a new output path; existing evidence is not overwritten")
    jax.config.update("jax_enable_x64", False)
    report = {
        "schema_version": 1,
        "scope": "QDax analytic planar-arm integration; not dynamic robot validation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        "versions": {p: importlib.metadata.version(p) for p in ("qdax", "jax", "jaxlib", "numpy")},
        "backend": jax.default_backend(), "devices": [str(d) for d in jax.devices()],
        "python": platform.python_version(), "platform": platform.platform(),
        "dtype": "float32", "fixture_evaluations_per_run": 2, "runs": [],
    }
    for seed in seeds:
        run = run_comparison(seed, args.population, args.generations, args.joints, args.bins)
        report["runs"].append(run)
        for method, result in run["methods"].items():
            print(f"seed={seed} {method}: {result['history'][-1]}", flush=True)
    report["passed"] = True  # All checks above raise on failure.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

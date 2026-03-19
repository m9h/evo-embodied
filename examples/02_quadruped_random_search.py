"""Assignment 9: Random search on a quadruped.

Generates random neural network weights, evaluates each one by simulating
the quadruped and measuring how far it walks. Keeps the best.

This is the CPU MuJoCo version (no MJX). Assignment 11 upgrades to MJX
for GPU-parallel search.

Run: uv run python examples/02_quadruped_random_search.py
"""
import mujoco
import numpy as np
import time
from pathlib import Path


def load_quadruped():
    """Load the quadruped model."""
    model_path = Path(__file__).parent.parent / "models" / "quadruped.xml"
    return mujoco.MjModel.from_xml_path(str(model_path))


class NeuralNetwork:
    """Simple feedforward network: sensors → hidden → motors."""

    def __init__(self, n_sensors, n_hidden, n_motors, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        self.w1 = rng.standard_normal((n_sensors, n_hidden)) * 0.1
        self.w2 = rng.standard_normal((n_hidden, n_motors)) * 0.1

    def forward(self, sensors):
        hidden = np.tanh(sensors @ self.w1)
        return np.tanh(hidden @ self.w2)

    def get_weights(self):
        return (self.w1.copy(), self.w2.copy())

    def set_weights(self, weights):
        self.w1, self.w2 = weights[0].copy(), weights[1].copy()


def evaluate(model, brain, steps=1000):
    """Run one simulation and return fitness (x-distance traveled)."""
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    for _ in range(steps):
        sensors = data.sensordata.copy()
        commands = brain.forward(sensors)
        data.ctrl[:] = commands
        mujoco.mj_step(model, data)

    # Fitness = distance traveled in x direction
    return data.qpos[0]


def random_search(n_candidates=100, n_hidden=8, steps=1000):
    """Evaluate n_candidates random neural networks, return the best."""
    model = load_quadruped()
    n_sensors = model.nsensordata
    n_motors = model.nu

    print(f"Quadruped: {n_sensors} sensors, {n_motors} motors")
    print(f"Neural network: {n_sensors} → {n_hidden} → {n_motors}")
    print(f"Evaluating {n_candidates} random candidates ({steps} timesteps each)...")
    print()

    rng = np.random.default_rng(42)
    best_fitness = -float("inf")
    best_weights = None

    t0 = time.time()
    for i in range(n_candidates):
        brain = NeuralNetwork(n_sensors, n_hidden, n_motors, rng=rng)
        fitness = evaluate(model, brain, steps=steps)

        if fitness > best_fitness:
            best_fitness = fitness
            best_weights = brain.get_weights()
            marker = " *** NEW BEST ***"
        else:
            marker = ""

        if (i + 1) % 10 == 0 or marker:
            print(f"  [{i+1:3d}/{n_candidates}] fitness = {fitness:+.4f}{marker}")

    elapsed = time.time() - t0
    print()
    print(f"Done in {elapsed:.1f}s ({elapsed/n_candidates:.2f}s per evaluation)")
    print(f"Best fitness: {best_fitness:+.4f} (distance in x)")
    print()
    print("Next step: Assignment 10 (hill climber) would mutate the best weights")
    print("           instead of generating new random ones each time.")

    return best_weights, best_fitness


if __name__ == "__main__":
    random_search()

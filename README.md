# evo-embodied

A modern evolutionary robotics environment using **MuJoCo + MJX + JAX**, designed as a contemporary replacement for the pyrosim/PyBullet stack used in Josh Bongard's [CS 3060 Evolutionary Robotics](https://www.reddit.com/r/ludobots/wiki/index) course at UVM.

## Why Replace pyrosim/PyBullet?

| | PyBullet (current) | MuJoCo + MJX (this repo) |
|---|---|---|
| Maintainer | Unmaintained | Google DeepMind (active) |
| Install | `pip install pybullet` | `pip install mujoco mujoco-mjx jax` |
| API | C-style integer handles | Pythonic objects (`model.body`, `data.qpos`) |
| Parallel sims | Sequential only | GPU-vectorized via JAX `vmap` (100-1000x speedup) |
| Differentiable | No | Yes (MJX supports `jax.grad` through physics) |
| Visualization | Basic OpenGL | Built-in viewer + `mediapy` for notebooks |
| Docs | Sparse | Excellent ([mujoco.readthedocs.io](https://mujoco.readthedocs.io)) |
| Research adoption | Declining | Standard in robotics/RL research |

The key upgrade: **the parallel hill climber** (assignment 11) currently runs N simulations sequentially. MJX runs them all on GPU simultaneously, turning "wait 10 minutes" into "watch it happen live."

## Quickstart

```bash
# Fedora
bash setup.sh
uv run jupyter lab

# Any platform with uv installed
uv sync
uv run jupyter lab

# With NVIDIA GPU (CUDA-accelerated evolution)
uv sync --extra gpu
```

## Mapping to Bongard's 13 Assignments

The course builds incrementally from "drop a box" to "evolve a walking robot." Every assignment maps directly to MuJoCo/MJX — the concepts are identical, the API is better.

### Phase 1: Simulation Fundamentals (CPU MuJoCo)

Assignments 1-8 use standard MuJoCo. Students learn physics simulation, robot design, and neural controllers with clear, debuggable, visual feedback.

| # | Assignment | PyBullet (old) | MuJoCo (new) | What Changes |
|---|-----------|---------------|-------------|-------------|
| 1 | **Simulation** | `p.connect()`, `loadURDF` | `mujoco.MjModel.from_xml_string()`, `mujoco.viewer` | MJCF XML replaces URDF; declarative model definition |
| 2 | **Objects** | `createCollisionShape()`, `createMultiBody()` | `<body><geom type="box"/></body>` in MJCF | XML bodies vs. imperative API — cleaner, easier to read |
| 3 | **Joints** | `createConstraint()`, `JOINT_REVOLUTE` | `<joint type="hinge"/>` in MJCF | Joint types declared in XML, not constructed in code |
| 4 | **Motors** | `setJointMotorControl2()` | `data.ctrl[i] = value` | Direct array assignment vs. function call per motor |
| 5 | **Sensors** | `getContactPoints()`, `getJointState()` | `data.sensordata` + `<sensor>` tags in MJCF | Sensors declared in XML, read from flat array |
| 6 | **Neurons** | Hand-built with numpy | Hand-built with numpy (identical) | No change — this is pure Python/numpy |
| 7 | **Synapses** | Hand-built weight matrices | Hand-built weight matrices (identical) | No change |
| 8 | **Refactoring** | Classes wrapping PyBullet calls | Classes wrapping MuJoCo calls | Same OOP exercise, cleaner underlying API |

### Phase 2: Evolutionary Search (MJX + JAX)

Assignments 9-13 unlock MJX for **GPU-parallel evolution**. Students experience firsthand why vectorized computation matters — their own evolutionary algorithms run 100-1000x faster.

| # | Assignment | PyBullet (old) | MJX + JAX (new) | What Changes |
|---|-----------|---------------|----------------|-------------|
| 9 | **Random Search** | Sequential: loop over N random genomes | `jax.vmap`: evaluate N genomes in one GPU call | First taste of vectorization |
| 10 | **Hill Climber** | Sequential: mutate, simulate, compare | Same logic, but `jax.jit`-compiled simulation | JIT compilation concept introduced |
| 11 | **Parallel Hill Climber** | N sequential hill climbers (slow!) | `jax.vmap` over population — all run simultaneously | **The key upgrade** — orders of magnitude faster |
| 12 | **Quadruped** | Design URDF by hand | Design MJCF by hand (same exercise) | MJCF is actually easier for articulated bodies |
| 13 | **GA / Phototaxis** | Sequential fitness evaluation | Batched evaluation, `jax.random` for crossover | Full evolutionary algorithm at GPU speed |

### Phase 3: Final Projects (Optional Packages)

Students doing ambitious final projects can draw on the optional extras:

| Extra | Install | What it enables |
|-------|---------|----------------|
| `strategies` | `uv sync --extra strategies` | `evosax` — GPU-accelerated CMA-ES, OpenES, PGPE, etc. Go beyond hand-built hill climbers |
| `rl` | `uv sync --extra rl` | `brax` + `dm_control` — compare evolution vs. reinforcement learning on the same robots |
| `gpu` | `uv sync --extra gpu` | JAX CUDA backend — required for serious MJX speedups |
| `full` | `uv sync --extra full` | Everything above |

## Key Concepts Introduced

This stack introduces students to ideas that are increasingly central to scientific computing and ML, naturally through the course material:

| Concept | Where it appears | Why it matters beyond this course |
|---------|-----------------|----------------------------------|
| **Declarative models** (MJCF XML) | Assignments 1-5 | Same paradigm as config-driven ML pipelines, IaC |
| **JIT compilation** (`jax.jit`) | Assignment 10 | Foundation of modern ML frameworks (JAX, PyTorch 2.0) |
| **Vectorization** (`jax.vmap`) | Assignments 9-13 | Core technique in scientific computing, GPU programming |
| **Functional programming** (JAX's pure-function model) | Assignments 9-13 | Reproducibility, parallelism, debugging |
| **Differentiable simulation** (optional) | Final projects | Gradient-based design optimization, sim-to-real transfer |

## Package Details

### Core (always installed)

| Package | Why |
|---------|-----|
| `mujoco>=3.2` | Physics engine — rigid bodies, joints, motors, sensors, contact |
| `mujoco-mjx>=3.2` | JAX-accelerated MuJoCo for GPU-parallel simulation |
| `jax>=0.4.35` | Numerical computing framework — `vmap`, `jit`, `grad` |
| `numpy>=1.26` | Array operations, neural network weight matrices |
| `matplotlib>=3.8` | Fitness curves, population statistics, sensor visualization |
| `mediapy>=1.2` | Render MuJoCo frames to video in Jupyter notebooks |
| `jupyterlab>=4.0` | Notebook environment |

### Not included (by design)

| Package | Why excluded |
|---------|-------------|
| PyBullet | The thing we're replacing |
| DEAP | Course pedagogy requires implementing EA from scratch |
| PyTorch / TensorFlow | JAX is the natural fit for MJX; adding torch creates confusion |
| Isaac Gym/Lab | Requires NVIDIA GPU, closed-source core, overkill for pedagogy |
| Genesis | Too young (2024), API still changing |
| Drake | Contact-implicit optimization focus, wrong abstraction level |
| pyrosim | Legacy wrapper — students should learn the real API |

## Resources

### MuJoCo / MJX
- [MuJoCo documentation](https://mujoco.readthedocs.io)
- [MJX tutorial](https://mujoco.readthedocs.io/en/stable/mjx.html)
- [MuJoCo MJCF modeling guide](https://mujoco.readthedocs.io/en/stable/XMLreference.html)
- [DeepMind MuJoCo GitHub](https://github.com/google-deepmind/mujoco)

### JAX
- [JAX quickstart](https://jax.readthedocs.io/en/latest/quickstart.html)
- [JAX vmap tutorial](https://jax.readthedocs.io/en/latest/automatic-vectorization.html) (most relevant for this course)
- [JAX JIT tutorial](https://jax.readthedocs.io/en/latest/jit-compilation.html)

### Evolutionary Robotics (Bongard's course)
- [r/ludobots wiki](https://www.reddit.com/r/ludobots/wiki/index) — original course content
- [r/ludobots subreddit](https://www.reddit.com/r/ludobots/) — community + assignment help
- [Josh Bongard's YouTube](https://www.youtube.com/@joshbongard3314) — lecture recordings
- [Bongard lab (MEC Lab)](https://meclab.org/)

### Evolutionary Strategies on GPU
- [evosax](https://github.com/RobertTLange/evosax) — JAX-native ES library
- [EvoJAX](https://github.com/google/evojax) — Google's hardware-accelerated neuroevolution

### Related Courses
- [HF Deep RL course](https://huggingface.co/learn/deep-rl-course) — RL perspective on the same robot control problems
- [HF Robotics / LeRobot](https://huggingface.co/learn/robotics-course) — real-world robot learning

## Requirements

- Python 3.12+
- `uv` (Fedora: `sudo dnf install uv`, or `pip install uv`)
- OpenGL runtime for MuJoCo viewer (present on most desktops)
- Optional: NVIDIA GPU + CUDA for `--extra gpu`

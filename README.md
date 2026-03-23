# evo-embodied

A learning environment for **embodied intelligence** using **MuJoCo + MJX + JAX**. Start with evolutionary robotics (evolve a walking quadruped), then follow the pathway into the full virtualrat research stack — biomechanically accurate rodent simulation, imitation learning, and active inference.

## Two Purposes

**1. Evolutionary Robotics Course** — a modern replacement for the pyrosim/PyBullet stack used in Josh Bongard's [CS 3060](https://www.reddit.com/r/ludobots/wiki/index) at UVM. GPU-parallel evolution via MJX, declarative MJCF models, and an API that matches current research practice.

**2. On-Ramp to the Virtual Rat** — every concept learned here (MuJoCo physics, MJX GPU parallelism, JAX `vmap`/`jit`/`grad`, neural controllers, MJCF morphologies) reappears in the virtualrat pipeline. Students who complete the 13 assignments have the foundation to work with STAC-MJX, track-mjx, and the full embodied intelligence stack.

## The Virtual Rat Pipeline

The virtualrat project connects several components into a pipeline for biomechanically accurate rodent simulation and control:

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  evo-embodied (YOU ARE HERE)                                        │
│  ═══════════════════════════                                        │
│  MuJoCo basics → MJX GPU parallelism → evolutionary search         │
│  → neural controllers → fitness landscapes                         │
│                                                                     │
│         ┌──────────────────────────┐                                │
│         ▼                          │                                │
│  ┌─────────────┐  real rat    ┌────┴────────┐                      │
│  │ RatInABox   │  video  ──▶  │  STAC-MJX   │                      │
│  │ synthetic   │              │  inverse     │                      │
│  │ trajectories│              │  kinematics  │                      │
│  │ + place/grid│              │  (MJX + IK)  │                      │
│  │ cells       │              └──────┬───────┘                      │
│  └─────────────┘                     │                              │
│                              reference motion clips                 │
│                                      │                              │
│                              ┌───────▼───────┐                     │
│                              │   track-mjx   │                     │
│                              │   imitation    │                     │
│                              │   learning     │                     │
│                              │   (PPO/SAC     │                     │
│                              │   in MJX)      │                     │
│                              └───────┬───────┘                     │
│                                      │                              │
│                         ┌────────────┼────────────┐                │
│                         ▼            ▼            ▼                │
│                    ┌─────────┐  ┌─────────┐  ┌─────────┐          │
│                    │ jaxctrl │  │   alf   │  │   hgx   │          │
│                    │ control │  │ active  │  │ hyper-  │          │
│                    │ theory  │  │inference│  │ graph   │          │
│                    │ (LQR,   │  │ (FEP,  │  │ neural  │          │
│                    │ Riccati)│  │  EFE)  │  │ nets    │          │
│                    └─────────┘  └─────────┘  └─────────┘          │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  bl1 — in-silico cortical culture (DishBrain-inspired)      │   │
│  │  Spiking networks + STDP + virtual MEA + closed-loop games  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  spinning-up-alf — 17-module curriculum                     │   │
│  │  Animal behavior → RL → Active Inference → Embodied AIF     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Map

| Component | What it does | Shared with evo-embodied |
|-----------|-------------|-------------------------|
| [STAC-MJX](https://github.com/talmolab/stac-mjx) | Markerless motion capture → MuJoCo body registration via GPU-parallel IK | MuJoCo, MJX, JAX, `jax.lax.scan` |
| [track-mjx](https://github.com/talmolab/track-mjx) | Train locomotion policies from reference clips (PPO in MJX) | MuJoCo, MJX, JAX, `vmap`, `brax` |
| [RatInABox](https://github.com/TomGeorge1234/RatInABox) | Synthetic trajectories + neural encoding (place cells, grid cells) | Environment modeling |
| [jaxctrl](https://github.com/m9h/jaxctrl) | Differentiable control theory (LQR, Lyapunov, Koopman) in JAX | JAX, `equinox`, `diffrax` |
| [alf](https://github.com/m9h/alf) | Active inference agents (generative models, expected free energy) | JAX, `equinox`, agent architectures |
| [hgx](https://github.com/m9h/hgx) | Hypergraph neural networks for higher-order interactions | JAX, `equinox`, neural dynamics |
| [bl1](https://github.com/m9h/bl1) | In-silico cortical cultures (spiking networks, STDP, virtual MEA) | JAX, `jax.lax.scan`, neural simulation |
| [spinning-up-alf](https://github.com/m9h/spinning-up-alf) | 17-module curriculum: animal behavior → RL → active inference | Full stack |

### Concept Bridge: evo-embodied → virtualrat

| You learn in evo-embodied | You use it again in |
|--------------------------|-------------------|
| MJCF XML models (quadruped) | STAC-MJX (rodent morphology), track-mjx (task environments) |
| `mujoco.mj_step()` | All MuJoCo-based components |
| `mjx.step()` + `jax.lax.scan` | STAC-MJX (IK optimization), track-mjx (PPO rollouts), bl1 (spiking dynamics) |
| `jax.vmap` (parallel populations) | track-mjx (parallel environment rollouts), bl1 (batched cultures) |
| `jax.jit` (compiled simulation) | Every JAX component in the stack |
| Neural network controllers | track-mjx (policy networks), alf (generative models), bl1 (spiking nets) |
| Fitness functions / reward shaping | track-mjx (imitation reward), alf (expected free energy) |
| Evolutionary search (hill climber, GA) | evosax strategies, neuroevolution baselines |
| `jax.grad` through physics (optional) | jaxctrl (differentiable LQR), track-mjx (policy gradients through MJX) |

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

# Run evolution with results saved to /data
uv run python examples/05_walking_quadruped.py --output-dir /data/evo-embodied
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

### Phase 3: Beyond Bongard (virtualrat on-ramp)

After the 13 assignments, students have the foundation for the full stack:

| Extension | What to try | Component it leads to |
|-----------|------------|----------------------|
| Replace hand-built NN with `equinox.Module` | Typed neural nets with pytree structure | alf, jaxctrl, bl1 |
| Replace evolution with PPO (`brax`) | Policy gradient RL on same quadruped | track-mjx |
| Load rodent MJCF instead of quadruped | Biomechanically accurate morphology | STAC-MJX, MIMIC-MJX |
| Add `jax.grad` through `mjx.step` | Gradient-based controller optimization | jaxctrl (differentiable LQR) |
| Replace reward with expected free energy | Active inference agent | alf |
| Record joint trajectories as "mocap" data | Inverse kinematics pipeline | STAC-MJX |

## Optional Extras

| Extra | Install | What it enables |
|-------|---------|----------------|
| `strategies` | `uv sync --extra strategies` | `evosax` — GPU-accelerated CMA-ES, OpenES, PGPE |
| `rl` | `uv sync --extra rl` | `brax` + `dm_control` — compare evolution vs. RL |
| `gpu` | `uv sync --extra gpu` | JAX CUDA backend for MJX |
| `full` | `uv sync --extra full` | Everything above |

## Key Concepts Introduced

| Concept | Where it appears | Why it matters beyond this course |
|---------|-----------------|----------------------------------|
| **Declarative models** (MJCF XML) | Assignments 1-5 | Same paradigm as config-driven ML pipelines, IaC |
| **JIT compilation** (`jax.jit`) | Assignment 10 | Foundation of modern ML frameworks (JAX, PyTorch 2.0) |
| **Vectorization** (`jax.vmap`) | Assignments 9-13 | Core technique in scientific computing, GPU programming |
| **Functional programming** (JAX's pure-function model) | Assignments 9-13 | Reproducibility, parallelism, debugging |
| **`jax.lax.scan`** (compiled loops) | Assignments 9-13 | Used in STAC-MJX, track-mjx, bl1 for long rollouts |
| **Differentiable simulation** (optional) | Final projects | jaxctrl, track-mjx policy gradients, sim-to-real |

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
| pyrosim | Legacy wrapper — students should learn the real API |

## Resources

### MuJoCo / MJX
- [MuJoCo documentation](https://mujoco.readthedocs.io)
- [MJX tutorial](https://mujoco.readthedocs.io/en/stable/mjx.html)
- [MJCF modeling guide](https://mujoco.readthedocs.io/en/stable/XMLreference.html)
- [DeepMind MuJoCo GitHub](https://github.com/google-deepmind/mujoco)

### JAX
- [JAX quickstart](https://jax.readthedocs.io/en/latest/quickstart.html)
- [JAX vmap tutorial](https://jax.readthedocs.io/en/latest/automatic-vectorization.html)
- [JAX JIT tutorial](https://jax.readthedocs.io/en/latest/jit-compilation.html)

### Virtual Rat / MIMIC-MJX
- [MIMIC-MJX](https://mimic-mjx.talmolab.org/) — biomechanically accurate rodent simulation (Talmo Lab)
- [MIMIC-MJX paper](https://arxiv.org/abs/2511.20532) — Zhang et al. 2025
- [MIMIC-MJX datasets](https://huggingface.co/datasets/talmolab/MIMIC-MJX) — rodent reference motion clips

### Evolutionary Robotics
- [r/ludobots wiki](https://www.reddit.com/r/ludobots/wiki/index) — Bongard's course content
- [Josh Bongard's YouTube](https://www.youtube.com/@joshbongard3314) — lecture recordings
- [evosax](https://github.com/RobertTLange/evosax) — JAX-native evolutionary strategies

### Active Inference & RL
- [spinning-up-alf](https://github.com/m9h/spinning-up-alf) — 17-module RL → AIF curriculum
- [HF Deep RL course](https://huggingface.co/learn/deep-rl-course) — RL perspective on robot control

## Requirements

- Python 3.12+
- `uv` (Fedora: `sudo dnf install uv`, or `pip install uv`)
- OpenGL runtime for MuJoCo viewer (present on most desktops)
- Optional: NVIDIA GPU + CUDA for `--extra gpu`

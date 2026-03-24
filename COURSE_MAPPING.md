# Course Mapping: Josh Bongard's CS 3060 → evo-embodied

A complete 20-assignment curriculum that extends Bongard's 13-assignment evolutionary robotics course with fitness function design, sim2real transfer, and connections to the virtualrat research stack.

## Overview

```
Phase 1: Simulation Fundamentals     (Assignments 1-8)   — Bongard originals
Phase 2: Evolutionary Search          (Assignments 9-13)  — Bongard originals + GPU upgrade
Phase 3: Fitness Engineering          (Assignments 14-15)  — NEW: why your robot leaps not walks
Phase 4: Real Robot                   (Assignments 16-18)  — NEW: sim2real with Petoi Bittle
Phase 5: Beyond Evolution             (Assignments 19-20)  — NEW: RL, active inference, virtualrat
```

**Bongard's course ends at assignment 13.** Phases 3-5 are our extensions. A semester course covers Phases 1-3 (15 assignments). A year-long course or independent study adds Phase 4 (real hardware) and Phase 5 (research connections).

---

## Phase 1: Simulation Fundamentals (Assignments 1-8)

These are direct translations of Josh's assignments from pyrosim/PyBullet to MuJoCo. The concepts are identical; the API is better.

### Assignment 1: Simulation

**Goal:** Set up physics simulation, create a ground plane, drop a box.

**Josh's version:** `p.connect(p.GUI)`, `loadURDF("plane.urdf")`, `createMultiBody()`

**evo-embodied version:**
```python
import mujoco
import mujoco.viewer

xml = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"/>
    <body pos="0 0 3">
      <freejoint/>
      <geom type="box" size="0.5 0.5 0.5" mass="1" rgba="0.2 0.6 1 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
```

**Script:** `examples/01_hello_mujoco.py`

**What's different:** MJCF XML is declarative — the world is defined as data, not built imperatively. Students learn to separate model from code.

---

### Assignment 2: Objects

**Goal:** Add multiple physical bodies to the simulation.

**evo-embodied:** Add named bodies in MJCF XML. Access by name: `model.body('block1')` instead of tracking integer IDs.

---

### Assignment 3: Joints

**Goal:** Connect bodies with articulated joints.

**evo-embodied:** Joint hierarchy expressed by XML nesting — parent-child is obvious from indentation. No more `createConstraint()` with cryptic arguments.

---

### Assignment 4: Motors

**Goal:** Add motor control to joints.

**evo-embodied:**
```xml
<actuator>
  <motor name="shoulder_motor" joint="shoulder" gear="100"/>
</actuator>
```
```python
data.ctrl[0] = 0.5  # one line per motor
```

---

### Assignment 5: Sensors

**Goal:** Add touch/position sensors to body parts.

**evo-embodied:** Sensors declared in XML, read from `data.sensordata`. MuJoCo has 30+ sensor types (touch, accelerometer, gyro, rangefinder, camera...) vs. PyBullet's contact points and joint states.

**Sim2real connection:** The sensors here (joint position, IMU, touch) are the same ones available on the real Bittle robot in Phase 4.

---

### Assignment 6: Neurons

**Goal:** Build a simple neural network (sensor neurons → motor neurons).

**Identical to Josh's version.** Pure Python/numpy, hand-built:
```python
class NeuralNetwork:
    def __init__(self, n_sensors, n_motors):
        self.weights = np.random.randn(n_sensors, n_motors) * 0.1

    def forward(self, sensor_values):
        return np.tanh(sensor_values @ self.weights)
```

No PyTorch, no TensorFlow. The pedagogy requires building it yourself.

---

### Assignment 7: Synapses

**Goal:** Add hidden layer, weighted connections.

**Identical to Josh's version.** Add `w1`, `w2`, hidden layer with tanh activation.

---

### Assignment 8: Refactoring

**Goal:** Restructure into classes (Robot, Simulation, NeuralNetwork).

**evo-embodied version:**
```python
class Robot:
    def __init__(self, mjcf_xml):
        self.model = mujoco.MjModel.from_xml_string(mjcf_xml)
        self.data = mujoco.MjData(self.model)
        self.brain = NeuralNetwork(self.model.nsensordata, 8, self.model.nu)

    def sense(self): return self.data.sensordata.copy()
    def think(self, s): return self.brain.forward(s)
    def act(self, c): self.data.ctrl[:] = c
```

**Key difference:** `model.nsensordata`, `model.nu` give clean introspection. No magic numbers.

---

## Phase 2: Evolutionary Search (Assignments 9-13)

This is where evo-embodied diverges most from Josh's course. The concepts (random search, hill climber, parallel hill climber, GA) are identical. The implementation uses MJX + JAX for GPU parallelism — turning hours of waiting into seconds of computation.

### Assignment 9: Random Search

**Josh's version:** Sequential Python loop over N random genomes.

**evo-embodied version:** `jax.vmap` evaluates all candidates in one GPU call.

**Script:** `examples/02_quadruped_random_search.py`

**New concept:** Vectorization (`jax.vmap`). Students see 10-100x speedup on their first try.

---

### Assignment 10: Hill Climber

**Josh's version:** Mutate best, simulate, compare, keep if better.

**evo-embodied:** Same logic, but `jax.jit`-compiled. Students learn JIT compilation.

---

### Assignment 11: Parallel Hill Climber

**This is the assignment that benefits most from the stack upgrade.**

**Josh's version:** N hill climbers running sequentially. Painful to wait for.

**evo-embodied:** All N hill climbers run simultaneously via `jax.vmap`. 256,000 simulations in minutes instead of hours.

**Script:** `examples/03_mjx_parallel_evolution.py`

**Benchmark:** 64 parallel hill climbers, 200 generations: RTX 2080 → 195 sims/sec. DGX Spark → ~27 sims/sec at pop 512.

---

### Assignment 12: Quadruped

**Josh's version:** Design a four-legged robot in URDF, evolve its controller.

**evo-embodied:** Design in MJCF (much more readable):

**Model:** `models/quadruped.xml`

Students modify the XML directly — add legs, change proportions, try hexapods, experiment with different joint types. No Python code changes needed.

---

### Assignment 13: Genetic Algorithm / Phototaxis

**Josh's version:** Implement crossover + selection, or evolve light-seeking behavior.

**evo-embodied:** GA with tournament selection, uniform crossover, and elitism — all in JAX, fully vectorized. Or implement phototaxis using MuJoCo rangefinder sensors.

**Script:** `examples/05_walking_quadruped.py` (full evolution with rendering)

---

## Phase 3: Fitness Engineering (Assignments 14-15) — NEW

**Why this phase exists:** Every student who completes Phase 2 produces a robot that leaps forward and flails its legs. This looks cool but isn't walking. Phase 3 teaches the hardest lesson in evolutionary robotics: **the fitness function is everything.**

### Assignment 14: The Leap-and-Flail Problem

**Goal:** Understand why your evolved robot leaps instead of walks. Fix it.

**Reading:** `docs/FITNESS_DESIGN.md`

**Exercise:**
1. Run `examples/05_walking_quadruped.py` — observe the behavior (leap + flail)
2. Watch the DGX results: fitness 11.27, but the video shows a ballistic launch
3. Diagnose: the fitness function (`x_distance`) rewards a single leap equally to 4 seconds of walking
4. Study the failure mode catalog: The Leap, The Flail, The Twitch, The Statue, The Spinner
5. Write a 1-page analysis: which failure mode does your robot exhibit? Why?

**What students learn:** The optimization algorithm is doing exactly what you asked. If you reward distance, you get leaps. The problem is always the fitness function, never the algorithm.

---

### Assignment 15: Coordinated Gait

**Goal:** Redesign the fitness function to produce actual walking.

**Script:** `examples/09_coordinated_gait.py`

**Three principles to implement:**
1. **Reward velocity, not distance** — `mean(x_velocity)` not `final_x_position`
2. **Penalize energy** — `- weight * mean(ctrl²)` makes flailing expensive
3. **Penalize jerk** — `- weight * mean(|ctrl_t - ctrl_{t-1}|²)` makes twitching expensive

**Exercise:**
1. Start from the broken fitness (assignment 14)
2. Add one component at a time, rerun evolution, record behavior:
   - Add velocity reward → does it still leap?
   - Add energy penalty → does it stop flailing?
   - Add smoothness penalty → does it stop twitching?
3. Tune the weights. Too much energy penalty → The Statue. Too little → The Flail.
4. Compare your best gait video to the Brax PPO ant: `examples/00_reference_demos.py --brax-ant`

**What students learn:** Reward/fitness design is the core skill of RL and evolutionary robotics. The algorithm only does what you measure.

---

## Phase 4: Real Robot (Assignments 16-18) — NEW

**Hardware required:** Petoi Bittle X V2 with feedback servos (~$300)

### Assignment 16: The Bittle Model

**Goal:** Understand the difference between simulation and reality.

**Model:** `models/bittle/bittle.xml`

**Exercise:**
1. Load the Bittle MJCF model, inspect it: `mujoco.MjModel.from_xml_path('models/bittle/bittle.xml')`
2. Compare to `models/quadruped.xml`:
   - Bittle uses **position actuators** (target angles) vs. quadruped's **torque motors**
   - Bittle has **servo dynamics** (40ms filter delay) — commands don't execute instantly
   - Bittle is **tiny** (177g, 9cm tall) vs. the generic quadruped (multi-kg)
3. Run your gait evolution from assignment 15 but on the Bittle model
4. What changes? Does the same fitness function work? What breaks?

**What students learn:** The sim2real gap starts in simulation. Position control ≠ torque control. Servo delay matters. Mass matters.

---

### Assignment 17: Domain Randomization

**Goal:** Train a policy robust enough to transfer to real hardware.

**Exercise:**
1. Train a gait on the exact Bittle model — it works in sim
2. Randomize dynamics per-episode:
   - Friction: ±50%
   - Servo delay: 20-60ms
   - Mass: ±20% per link
   - IMU noise: ±5° orientation, ±0.5 m/s² acceleration
   - Joint damping: ±30%
3. Retrain with randomization — fitness will be lower, but the policy is more robust
4. Compare videos: deterministic training vs. randomized training

**What students learn:** A policy that only works in one simulation is brittle. Domain randomization trades peak performance for robustness — exactly what you need for real hardware.

---

### Assignment 18: Deploy to Bittle

**Goal:** Transfer your evolved controller to the real robot.

**Script:** `sim2real/deploy_bittle.py`
**Guide:** `docs/SIM2REAL.md`

**Exercise:**
1. Connect Bittle via USB, verify: `--list-ports`
2. Dry run: `--dry-run` — watch the commands, check they're reasonable
3. Deploy: `--weights best_weights.npy --duration 10`
4. Document what happens. It will probably not walk well. That's the point.
5. Debug using the checklist in `docs/SIM2REAL.md`:
   - Vibrating? → Lower servo gains in bittle.xml, retrain
   - Falls immediately? → Weigh your actual Bittle, update the XML
   - Legs move but no traction? → Floor friction mismatch
6. Iterate: adjust model → retrain → redeploy → observe → adjust
7. Write up: What was the biggest gap? How did you close it?

**What students learn:** Sim2real is an iterative process of identifying and closing gaps between simulation and reality. No simulation is perfect — the skill is diagnosing which imperfections matter.

---

## Phase 5: Beyond Evolution (Assignments 19-20) — NEW

### Assignment 19: Evolution vs. Reinforcement Learning

**Goal:** Train the same robot with PPO and compare to evolution.

**Exercise:**
1. Train a Brax ant with PPO: `examples/00_reference_demos.py --brax-ant` (~3 min)
2. Compare: RL produces smooth gaits in minutes. Evolution took hours and fitness engineering.
3. Why? RL uses **gradients** — it knows which direction to improve. Evolution uses **random mutation** — it has to stumble upon improvements.
4. But evolution has advantages: no differentiability required, can optimize morphology (not just controllers), naturally parallel, works with any fitness function
5. Read: the evo-embodied README's "Phase 3: Beyond Bongard" table for concrete next steps

**What students learn:** Evolution and RL are complementary, not competing. Evolution excels at open-ended design spaces (morphology, novel fitness landscapes). RL excels at policy optimization for known embodiments.

---

### Assignment 20: The Virtualrat Pipeline (Independent Study)

**Goal:** Connect evo-embodied skills to the full embodied intelligence research stack.

**Choose one:**

**A. Biomechanical motion tracking** — Load a real rodent morphology from MIMIC-MJX into MuJoCo. Use STAC-MJX to register motion capture keypoints. Train a controller to replay the motion with track-mjx (PPO).

**B. Active inference** — Replace the evolutionary fitness function with expected free energy from the alf framework. Instead of optimizing a number, the agent has a generative model of its environment and acts to minimize surprise.

**C. LLM-augmented evolution** — Use the LLM experiments (examples 06-08):
  - 06: LLM steers mutation rate based on fitness trajectory
  - 07: LLM proposes robot morphologies (outer loop: body, inner loop: controller)
  - 08: LLM generates the evolutionary algorithm itself (meta-evolution)

**D. Cortical control** — Connect bl1 (spiking neural network with STDP) as the controller instead of a feedforward network. The "brain" learns via spike-timing-dependent plasticity rather than evolved weights.

---

## Assignment ↔ Script/File Map

| # | Assignment | Script/File | Model |
|---|-----------|-------------|-------|
| 1 | Simulation | `examples/01_hello_mujoco.py` | inline XML |
| 2 | Objects | (modify 01) | inline XML |
| 3 | Joints | (modify 01) | inline XML |
| 4 | Motors | (modify 01) | inline XML |
| 5 | Sensors | (modify 01) | inline XML |
| 6 | Neurons | (new file) | — |
| 7 | Synapses | (modify 06) | — |
| 8 | Refactoring | (new file) | `models/single_link.xml` |
| 9 | Random Search | `examples/02_quadruped_random_search.py` | `models/quadruped.xml` |
| 10 | Hill Climber | (modify 09) | `models/quadruped.xml` |
| 11 | Parallel Hill Climber | `examples/03_mjx_parallel_evolution.py` | `models/quadruped.xml` |
| 12 | Quadruped | `models/quadruped.xml` | `models/quadruped.xml` |
| 13 | GA / Phototaxis | `examples/05_walking_quadruped.py` | `models/quadruped.xml` |
| 14 | Leap-and-Flail | `docs/FITNESS_DESIGN.md` | — |
| 15 | Coordinated Gait | `examples/09_coordinated_gait.py` | `models/quadruped.xml` |
| 16 | Bittle Model | `models/bittle/bittle.xml` | `models/bittle/bittle.xml` |
| 17 | Domain Randomization | (modify 15 for Bittle) | `models/bittle/bittle.xml` |
| 18 | Deploy to Bittle | `sim2real/deploy_bittle.py` | `models/bittle/bittle.xml` |
| 19 | Evolution vs. RL | `examples/00_reference_demos.py` | (brax envs) |
| 20 | Virtualrat Pipeline | `examples/06-08_*.py` | (varies) |

---

## Semester Schedule (15 weeks)

| Week | Assignments | Topic |
|------|------------|-------|
| 1 | **0** (reference demos) | See the destination: watch a trained robot walk |
| 1-2 | **1-3** | Simulation, objects, joints |
| 3 | **4-5** | Motors and sensors |
| 4 | **6-7** | Neural networks from scratch |
| 5 | **8** | Refactoring + code review |
| 6 | **9** | Random search |
| 7 | **10** | Hill climber |
| 8 | **11** | Parallel hill climber (GPU) |
| 9 | **12** | Quadruped design |
| 10 | **13** | Genetic algorithm |
| 11 | **14** | Fitness function analysis (why it leaps) |
| 12 | **15** | Coordinated gait (velocity-based fitness) |
| 13-14 | **16-18** | Bittle model, domain randomization, deployment |
| 15 | **19** + presentations | Evolution vs. RL comparison |

---

## Skills Comparison

| Skill | Josh's PyBullet course | evo-embodied (Phases 1-2) | + Phase 3-4 | + Phase 5 |
|-------|----------------------|--------------------------|-------------|-----------|
| Physics simulation | Yes | Yes (better API) | — | — |
| Robot design (MJCF) | URDF | MJCF (readable) | Bittle model | Rodent model |
| Neural networks | Hand-built | Hand-built | — | equinox/flax |
| Evolutionary algorithms | Yes | Yes (identical) | — | LLM-augmented |
| GPU parallelism | No | `jax.vmap`, `jax.jit` | — | MJX rollouts |
| Fitness function design | Implicit | — | **Explicit curriculum** | EFE (active inference) |
| Sim2real | No | No | **Bittle deployment** | MIMIC-MJX |
| Domain randomization | No | No | **Yes** | — |
| Reinforcement learning | No | No | — | **Brax PPO** |
| Differentiable physics | No | Available | — | jaxctrl |
| Research connection | No | No | — | **virtualrat stack** |

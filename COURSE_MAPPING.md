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

**Sim2real connection:** Verify joint-position and IMU availability on the selected Bittle hardware. Virtual touch and torso state are simulation measurements; a deployable policy needs an explicitly matched observation subset.

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

**New concept:** Vectorization (`jax.vmap`). Measure how population size and
compilation affect throughput; a speedup is workload- and hardware-dependent.

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

**Historical observations:** 64 parallel hill climbers, 200 generations:
RTX 2080 → 195 sims/sec; DGX Spark → ~27 sims/sec at pop 512. These differently
configured observations have no complete run manifests here and are not a
controlled hardware comparison. Re-measure under B1 in the validation plan.

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

**Demo goal:** evolve a repertoire of Bittle gaits and compare speed, stability
and robustness. Follow [D1-D6 in the demo roadmap](docs/DEMO_ROADMAP.md).
Simulation exercises need no robot. The later hardware exercise requires an
identified Bittle variant with verified sensor/servo capabilities and calibration.
The current deployment prototype does not yet match the training interface.

### Assignment 16: The Bittle Model

**Goal:** Understand the difference between simulation and reality.

**Model:** `models/bittle/bittle.xml`

**Exercise:**
1. Load the Bittle MJCF model, inspect it: `mujoco.MjModel.from_xml_path('models/bittle/bittle.xml')`
2. Compare to `models/quadruped.xml`:
   - Bittle uses **position actuators** (target angles) vs. quadruped's **torque motors**
   - The model declares a **40 ms servo filter**, but its active position actuators do not use it; inspect `actuator_dyntype` and `na`
   - Loaded model mass is **269 g total**; 177 g describes only the torso
3. Audit the joint map and compare the pinned community Bittle X V1 model and OpenCat stand/trot/walk fixtures from the roadmap
4. Validate the shared policy contract and full-length evaluator before search comparisons; example 10 remains a single-policy prototype
5. Compare the fixed gait baseline with evolved behaviors. Does forward progress come from walking, sliding or falling?

**What students learn:** The sim2real gap starts in simulation. Position control ≠ torque control. Servo delay matters. Mass matters.

---

### Assignment 17: Domain Randomization

**Goal:** Test whether randomized training improves held-out robustness.

**Exercise:**
1. Correct example 11's parent/candidate evaluation-panel mismatch; unchanged candidates must tie parents under the same sampled conditions
2. Freeze nominal, validation and test conditions, then evaluate deterministic training
3. Choose randomization ranges from measurements or label them synthetic: friction, added load, actuator response, sensor noise and timed disturbances. Implement and validate delay dynamics before randomizing them
4. Retrain at matched budgets and compare all seeds on the same held-out panel; report speed, falls, drift, actuator work and uncertainty
5. Reevaluate the behavior archive, not just its best gait. Publish corrected coverage and QD score alongside clips

**What students learn:** Robustness and transfer are measurable hypotheses. Randomization can help or hurt; nominal fitness and a selected video cannot establish its benefit.

---

### Assignment 18: Deploy to Bittle

**Goal:** Transfer your evolved controller to the real robot.

**Script:** `sim2real/deploy_bittle.py`
**Guide:** `docs/SIM2REAL.md`

**Status:** planned D6 exercise; the script is a prototype, not a compatible
deployment path for example 10 checkpoints.

**Exercise after D1-D3:**
1. Verify exported-policy inference parity on recorded observations before opening a serial connection
2. Record the actual robot/firmware/servo variant; calibrate joint signs/offsets, observable sensors and actuator response
3. Measure sustained command/feedback timing against the policy's declared control rate
4. Compare a verified OpenCat baseline with frozen policies selected using validation data
5. Record displacement, speed, drift, falls and tracking error across repeated trials, including failures
6. Diagnose the measured gap, version model/interface changes and retrain where needed
7. Report the physical results separately from the simulation comparison; a dry run is not hardware validation

**What students learn:** Sim2real is an iterative process of identifying and closing gaps between simulation and reality. No simulation is perfect — the skill is diagnosing which imperfections matter.

---

## Phase 5: Beyond Evolution (Assignments 19-20) — NEW

### Assignment 19: Evolution vs. Reinforcement Learning

**Goal:** Train the same robot with PPO and compare to evolution.

**Exercise:**
1. Inspect or train the pinned reference with `uv run --locked --extra rl python examples/00_reference_demos.py --brax-ant`. Its full budget is 20M steps; runtime is measured on your hardware.
2. Build the matched-task adapter described in B1 of [the validation plan](docs/VALIDATION_PLAN.md). Compare PPO, random search, hill climbing and evosax using the same robot, observations, actions, reset, reward and charged transition budget. A Brax Ant versus our custom quadruped is not that comparison.
3. Explain the gradient distinction: PPO differentiates its policy/value losses, not the simulator. Evolution estimates or searches for improvements from sampled evaluations; its efficiency must be measured rather than assumed.
4. Run `examples/12_quality_diversity.py` with the `qd` extra and compare MAP-Elites with equal-budget random search. Explain why archive coverage, QD score and best fitness answer different questions. This analytic arm is an integration exercise, not evidence of robot locomotion.
5. Propose an EvoRL evolution-plus-learning arm with explicitly charged inner updates, then follow [project choices](docs/PROJECT_CHOICES.md) for the reference runtime boundary.

**What students learn:** Search, quality-diversity and reinforcement learning can
be combined. Establish their task-specific tradeoffs with controlled experiments
and independent seeds before making efficiency or adaptation claims.

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

**D. Development and cortical control** — Plan a devograph/hgx or bl1 adapter with
the same sensory/motor interface as fixed-network and TensorNEAT comparators.
Separate inherited weights/topology from within-lifetime growth or STDP. Include
growth-disabled and plasticity-disabled controls; these adapters are future work.

**E. Soft-body co-design with control analysis** — A planned research extension,
tracked as B5-B7 in [the validation plan](docs/VALIDATION_PLAN.md). The pairing is
soft-body simulation, evolutionary search and control theory. The EvoGym adapter
and controlled-system identification experiment are not yet implemented.

The setup:

1. **Soft-body sim** — Evolution Gym for the voxel co-design benchmark ([EvolutionGym/evogym](https://github.com/EvolutionGym/evogym)). It does not provide JAX autodiff through its simulator. Use [SoftZoo](https://github.com/zswang666/softzoo) for a separate differentiable-physics study; different material models prevent treating it as an exact numerical oracle for EvoGym.
2. **evosax** — an optional dependency already declared; a voxel morphology encoding and EvoGym search adapter still need implementation
3. **[jaxctrl](https://github.com/m9h/jaxctrl)** — network control theory, applied to evolved morphologies:
   - Fit a controlled surrogate from state/input trajectories using DMDc; autonomous `KoopmanEstimator`/exact DMD does not identify the actuator matrix B
   - Analyze finite-horizon reachability with normalized state/input units and inspect conditioning
   - Use discrete-time LQR for a discrete fitted model, with bounded corrections around a nominal trajectory and nonlinear evaluation in EvoGym

**Research question:** Does controllability-informed body selection or feedback
improve held-out disturbance recovery at the same training budget? Validate the
identified model and the behavior separately; a well-conditioned Gramian alone
does not establish a useful robot.

**Reading:**
- [`docs/RESEARCH_GAPS.md`](docs/RESEARCH_GAPS.md) — the math primitives this exercise stresses (saltation matrices, hybrid Floquet, randomized DMD) and which are still missing from the JAX ecosystem
- [`docs/SIM2REAL.md`](docs/SIM2REAL.md) "Sim2Real in Context" — the broader landscape this assignment sits in
- Matthews, Spielberg, Rus, Kriegman, Bongard, "Efficient automatic design of robots," [*PNAS* 2023](https://www.pnas.org/doi/abs/10.1073/pnas.2305180120) — RoboDiff, the differentiable successor to voxcraft from the xenobot authors
- Kriegman et al., "A scalable pipeline for designing reconfigurable organisms," [*PNAS* 2020](https://www.pnas.org/doi/10.1073/pnas.1910837117) — the original xenobot paper: voxel evolution realized as living frog cells
- Bruder et al., "Modeling and control of soft robots using the Koopman operator," [*RSS* 2019](https://www.roboticsproceedings.org/rss15/p60.pdf)

**Exercise:**

1. Use Evolution Gym to evolve a voxel locomotion morphology on a flat-terrain task
2. Roll out the evolved best-of-generation; collect state trajectories (voxel positions + velocities)
3. Collect excitation rollouts with recorded inputs, fit `(A, B)` using DMDc, and compare held-out prediction against persistence and [PyDMD](https://pydmd.github.io/PyDMD/dmdc.html)
4. Check stabilizability and finite-horizon reachability after consistent state/input normalization
5. Design bounded discrete-time feedback around a nominal trajectory, then evaluate in the original nonlinear Evolution Gym simulator
6. Compare nominal control, zero correction, fitted feedback and wrong-model/null feedback on identical held-out perturbations. Charge identification rollouts to the training budget

**What students learn:** Three orthogonal optimization layers — morphology, controller-via-evolution, and controller-via-control-theory — and how they compose. Also: that the network-control-theoretic primitives in jaxctrl are domain-general. The same `controllability_gramian` that answers "is this evolved voxel robot steerable" answers "is this gene regulatory network controllable by transcription factor perturbations" and "which brain regions must we stimulate to drive a target dynamics."

**Caveats to surface explicitly:**
- Koopman/DMD on voxel sim data is only valid around a specific operating point (a stance, a gait limit cycle). It doesn't generalize to arbitrary maneuvers without re-fitting.
- Evolution Gym uses smooth penalty-based contact, not hybrid contact. The saltation/hybrid-Floquet machinery (which would be needed for a rigorous controllability analysis through impacts) is currently missing from the JAX ecosystem — see RESEARCH_GAPS.md.
- The control-theoretic surrogate is linear; the underlying dynamics are not. Multilinear LQR (jaxctrl L2) is the next step but expects more from the user.

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
| 16 | Bittle Model | `examples/10_bittle_gait.py` | `models/bittle/bittle.xml` |
| 17 | Domain Randomization | `examples/11_domain_randomization.py` | `models/bittle/bittle.xml` |
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
| Differentiable physics | No | Available (MJX) | — | JAX-MPM (soft-body) |
| Differentiable control theory | No | No | — | jaxctrl (Lyapunov, Riccati, Koopman, controllability) |
| Research connection | No | No | — | **virtualrat stack** |

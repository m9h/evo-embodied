# Course Mapping: Bongard CS 3060 → MuJoCo/MJX

Detailed translation guide from pyrosim/PyBullet to MuJoCo/MJX for each of Josh Bongard's 13 assignments. Each section shows the old approach, the new approach, and code snippets.

---

## Assignment 1: Simulation

**Goal:** Set up physics simulation, create a ground plane, drop a box.

### PyBullet (old)
```python
import pybullet as p
import pybullet_data

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

planeId = p.loadURDF("plane.urdf")
boxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
bodyId = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=boxId,
                           basePosition=[0, 0, 3])

for _ in range(1000):
    p.stepSimulation()
```

### MuJoCo (new)
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

**Key difference:** MuJoCo uses declarative MJCF XML to define the world. The physics model is separate from the simulation code. Students learn to think about models as data, not imperative construction sequences.

---

## Assignment 2: Objects

**Goal:** Add multiple physical bodies to the simulation.

### MuJoCo approach
```xml
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="5 5 0.1"/>

    <!-- Stack of blocks -->
    <body name="block1" pos="0 0 0.5">
      <freejoint/>
      <geom type="box" size="0.5 0.5 0.5" mass="1" rgba="1 0 0 1"/>
    </body>
    <body name="block2" pos="0 0 2">
      <freejoint/>
      <geom type="box" size="0.3 0.3 0.3" mass="0.5" rgba="0 1 0 1"/>
    </body>
    <body name="sphere1" pos="1 0 1">
      <freejoint/>
      <geom type="sphere" size="0.4" mass="0.5" rgba="0 0 1 1"/>
    </body>
  </worldbody>
</mujoco>
```

**Key difference:** Bodies are named (`name="block1"`) and can be accessed by name in Python: `model.body('block1')`. No more tracking integer IDs.

---

## Assignment 3: Joints

**Goal:** Connect bodies with articulated joints.

### MuJoCo approach
```xml
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="5 5 0.1"/>

    <!-- Simple arm: base fixed to world, two hinged segments -->
    <body name="base" pos="0 0 1">
      <geom type="box" size="0.2 0.2 0.2" rgba="0.5 0.5 0.5 1"/>
      <joint name="shoulder" type="hinge" axis="0 1 0"/>
      <body name="upper_arm" pos="0 0 0.6">
        <geom type="box" size="0.1 0.1 0.4" rgba="1 0 0 1"/>
        <joint name="elbow" type="hinge" axis="0 1 0"/>
        <body name="lower_arm" pos="0 0 0.6">
          <geom type="box" size="0.08 0.08 0.3" rgba="0 0 1 1"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
```

**Key difference:** Joint hierarchy is expressed by XML nesting. Parent-child relationships are obvious from indentation. In PyBullet, students had to mentally track which `createConstraint` connected which body IDs.

---

## Assignment 4: Motors

**Goal:** Add motor control to joints.

### MuJoCo approach
```xml
<mujoco>
  <worldbody>
    <!-- ... bodies and joints as above ... -->
  </worldbody>

  <actuator>
    <motor name="shoulder_motor" joint="shoulder" gear="100"/>
    <motor name="elbow_motor" joint="elbow" gear="50"/>
  </actuator>
</mujoco>
```

```python
# In simulation loop:
data.ctrl[0] = 0.5   # shoulder_motor torque
data.ctrl[1] = -0.3  # elbow_motor torque
mujoco.mj_step(model, data)
```

**Key difference:** `data.ctrl` is a simple numpy array. One line per motor, vs. PyBullet's verbose `setJointMotorControl2(bodyId, jointIndex, controlMode, targetVelocity, force)`.

---

## Assignment 5: Sensors

**Goal:** Add touch/position sensors to body parts.

### MuJoCo approach
```xml
<mujoco>
  <worldbody>
    <!-- ... -->
    <body name="foot" pos="0 0 0">
      <geom name="foot_geom" type="sphere" size="0.1"/>
    </body>
  </worldbody>

  <sensor>
    <touch name="foot_touch" site="foot_site"/>
    <jointpos name="shoulder_pos" joint="shoulder"/>
    <jointvel name="shoulder_vel" joint="shoulder"/>
  </sensor>
</mujoco>
```

```python
# Read all sensor data as a flat array:
touch_value = data.sensordata[0]   # foot_touch
joint_pos = data.sensordata[1]     # shoulder_pos
joint_vel = data.sensordata[2]     # shoulder_vel

# Or by name:
touch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "foot_touch")
touch_value = data.sensordata[model.sensor_adr[touch_id]]
```

**Key difference:** Sensors are declared in XML and read from `data.sensordata`. MuJoCo supports 30+ sensor types (touch, accelerometer, gyro, force, torque, rangefinder, camera...). PyBullet only had contact points and joint states.

---

## Assignment 6: Neurons

**Goal:** Build a simple neural network (sensor neurons → motor neurons).

### Same in both stacks
```python
import numpy as np

class NeuralNetwork:
    def __init__(self, n_sensors, n_motors):
        self.weights = np.random.randn(n_sensors, n_motors) * 0.1

    def forward(self, sensor_values):
        return np.tanh(sensor_values @ self.weights)

# Usage:
nn = NeuralNetwork(n_sensors=3, n_motors=2)
motor_commands = nn.forward(data.sensordata[:3])
data.ctrl[:] = motor_commands
```

**No change.** This assignment is pure Python/numpy. The course intentionally has students build neural networks by hand, not use PyTorch/TensorFlow.

---

## Assignment 7: Synapses

**Goal:** Wire neurons with weighted connections, add hidden layers.

### Same in both stacks
```python
class NeuralNetwork:
    def __init__(self, n_sensors, n_hidden, n_motors):
        self.w1 = np.random.randn(n_sensors, n_hidden) * 0.1
        self.w2 = np.random.randn(n_hidden, n_motors) * 0.1

    def forward(self, sensor_values):
        hidden = np.tanh(sensor_values @ self.w1)
        return np.tanh(hidden @ self.w2)
```

**No change.** Pure numpy.

---

## Assignment 8: Refactoring

**Goal:** Restructure code into clean classes.

### MuJoCo approach
```python
class Robot:
    def __init__(self, mjcf_xml):
        self.model = mujoco.MjModel.from_xml_string(mjcf_xml)
        self.data = mujoco.MjData(self.model)
        n_sensors = self.model.nsensordata
        n_motors = self.model.nu
        self.brain = NeuralNetwork(n_sensors, 8, n_motors)

    def sense(self):
        return self.data.sensordata.copy()

    def think(self, sensor_values):
        return self.brain.forward(sensor_values)

    def act(self, motor_commands):
        self.data.ctrl[:] = motor_commands

class Simulation:
    def __init__(self, robot):
        self.robot = robot

    def run(self, steps=1000):
        for _ in range(steps):
            sensors = self.robot.sense()
            commands = self.robot.think(sensors)
            self.robot.act(commands)
            mujoco.mj_step(self.robot.model, self.robot.data)

    def get_fitness(self):
        # Distance traveled in x direction
        return self.robot.data.qpos[0]
```

**Key difference:** `model.nsensordata`, `model.nu` (number of actuators) give clean introspection. No more magic numbers.

---

## Assignment 9: Random Search

**Goal:** Generate random neural network weights, evaluate fitness.

### PyBullet (old) — sequential
```python
best_fitness = -float('inf')
for i in range(100):
    robot = Robot()
    robot.brain.randomize()
    sim = Simulation(robot)
    sim.run()
    fitness = sim.get_fitness()
    if fitness > best_fitness:
        best_fitness = fitness
        best_weights = robot.brain.get_weights()
```

### MJX + JAX (new) — vectorized
```python
import jax
import jax.numpy as jnp
from mujoco import mjx

# Compile model for JAX
mjx_model = mjx.put_model(model)

@jax.jit
def evaluate_one(weights, mjx_model):
    mjx_data = mjx.make_data(mjx_model)
    def step_fn(data, _):
        sensors = data.sensordata
        commands = jnp.tanh(sensors @ weights)
        data = data.replace(ctrl=commands)
        data = mjx.step(mjx_model, data)
        return data, None
    final_data, _ = jax.lax.scan(step_fn, mjx_data, None, length=1000)
    return final_data.qpos[0]  # x-distance

# Evaluate ALL candidates at once
evaluate_batch = jax.vmap(evaluate_one, in_axes=(0, None))

key = jax.random.PRNGKey(42)
population = jax.random.normal(key, (100, n_sensors, n_motors)) * 0.1
fitnesses = evaluate_batch(population, mjx_model)
best_idx = jnp.argmax(fitnesses)
```

**Key upgrade:** 100 evaluations happen in one GPU call instead of a Python loop. Students see a 10-100x speedup immediately.

---

## Assignment 10: Hill Climber

### MJX + JAX
```python
@jax.jit
def hill_climber_step(key, weights, best_fitness, mjx_model):
    key, subkey = jax.random.split(key)
    mutation = jax.random.normal(subkey, weights.shape) * 0.05
    candidate = weights + mutation
    fitness = evaluate_one(candidate, mjx_model)
    improved = fitness > best_fitness
    new_weights = jnp.where(improved, candidate, weights)
    new_fitness = jnp.where(improved, fitness, best_fitness)
    return key, new_weights, new_fitness

# Run 1000 generations — entire loop is JIT-compiled
key = jax.random.PRNGKey(0)
weights = jax.random.normal(key, (n_sensors, n_motors)) * 0.1
fitness = evaluate_one(weights, mjx_model)
for gen in range(1000):
    key, weights, fitness = hill_climber_step(key, weights, fitness, mjx_model)
```

**Key concept introduced:** `jax.jit` compiles the entire step function. Students learn that "write it in Python, run it at C speed" is possible.

---

## Assignment 11: Parallel Hill Climber

**This is the assignment that benefits most from the stack upgrade.**

### PyBullet (old) — painfully slow
```python
# N independent hill climbers, each running sequentially
population = [HillClimber() for _ in range(10)]
for gen in range(500):
    for hc in population:
        hc.mutate()
        hc.evaluate()  # <-- each evaluation is a full simulation
        hc.select()
# Total: 10 * 500 * sim_time = very slow
```

### MJX + JAX — massively parallel
```python
@jax.jit
def parallel_hill_climber_step(keys, all_weights, all_fitnesses, mjx_model):
    keys, subkeys = jax.vmap(jax.random.split)(keys)
    mutations = jax.vmap(
        lambda k, w: jax.random.normal(k, w.shape) * 0.05
    )(subkeys, all_weights)
    candidates = all_weights + mutations
    new_fitnesses = evaluate_batch(candidates, mjx_model)
    improved = new_fitnesses > all_fitnesses
    all_weights = jnp.where(improved[:, None, None], candidates, all_weights)
    all_fitnesses = jnp.where(improved, new_fitnesses, all_fitnesses)
    return keys, all_weights, all_fitnesses

# 512 parallel hill climbers, each evaluating simultaneously
n_population = 512
keys = jax.random.split(jax.random.PRNGKey(0), n_population)
all_weights = jax.random.normal(keys[0], (n_population, n_sensors, n_motors)) * 0.1
all_fitnesses = evaluate_batch(all_weights, mjx_model)

for gen in range(500):
    keys, all_weights, all_fitnesses = parallel_hill_climber_step(
        keys, all_weights, all_fitnesses, mjx_model)
```

**Key upgrade:** 512 hill climbers * 500 generations = 256,000 full simulations. On GPU, this takes seconds. On PyBullet, it takes hours.

---

## Assignment 12: Quadruped

### MuJoCo MJCF quadruped
```xml
<mujoco model="quadruped">
  <compiler angle="degree"/>
  <option gravity="0 0 -9.81" timestep="0.002"/>

  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="10 10 0.1" rgba="0.8 0.8 0.8 1"/>

    <body name="torso" pos="0 0 0.5">
      <freejoint name="root"/>
      <geom type="box" size="0.4 0.2 0.1" mass="5" rgba="0.3 0.3 0.8 1"/>

      <!-- Front-left leg -->
      <body name="fl_upper" pos="0.3 0.2 0">
        <joint name="fl_hip" type="hinge" axis="0 1 0" range="-60 60"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.04" mass="0.5"/>
        <body name="fl_lower" pos="0 0 -0.3">
          <joint name="fl_knee" type="hinge" axis="0 1 0" range="-90 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03" mass="0.3"/>
        </body>
      </body>

      <!-- Front-right leg -->
      <body name="fr_upper" pos="0.3 -0.2 0">
        <joint name="fr_hip" type="hinge" axis="0 1 0" range="-60 60"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.04" mass="0.5"/>
        <body name="fr_lower" pos="0 0 -0.3">
          <joint name="fr_knee" type="hinge" axis="0 1 0" range="-90 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03" mass="0.3"/>
        </body>
      </body>

      <!-- Back-left leg -->
      <body name="bl_upper" pos="-0.3 0.2 0">
        <joint name="bl_hip" type="hinge" axis="0 1 0" range="-60 60"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.04" mass="0.5"/>
        <body name="bl_lower" pos="0 0 -0.3">
          <joint name="bl_knee" type="hinge" axis="0 1 0" range="-90 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03" mass="0.3"/>
        </body>
      </body>

      <!-- Back-right leg -->
      <body name="br_upper" pos="-0.3 -0.2 0">
        <joint name="br_hip" type="hinge" axis="0 1 0" range="-60 60"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.04" mass="0.5"/>
        <body name="br_lower" pos="0 0 -0.3">
          <joint name="br_knee" type="hinge" axis="0 1 0" range="-90 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03" mass="0.3"/>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor joint="fl_hip" gear="100"/><motor joint="fl_knee" gear="50"/>
    <motor joint="fr_hip" gear="100"/><motor joint="fr_knee" gear="50"/>
    <motor joint="bl_hip" gear="100"/><motor joint="bl_knee" gear="50"/>
    <motor joint="br_hip" gear="100"/><motor joint="br_knee" gear="50"/>
  </actuator>

  <sensor>
    <jointpos joint="fl_hip"/><jointpos joint="fl_knee"/>
    <jointpos joint="fr_hip"/><jointpos joint="fr_knee"/>
    <jointpos joint="bl_hip"/><jointpos joint="bl_knee"/>
    <jointpos joint="br_hip"/><jointpos joint="br_knee"/>
    <framepos objtype="body" objname="torso"/>
  </sensor>
</mujoco>
```

**Key difference:** The entire quadruped is a readable XML file. In PyBullet, this same robot requires 50+ lines of imperative `createMultiBody` and `createConstraint` calls. Students can modify the robot's morphology by editing XML — add legs, change proportions, try hexapods — without touching Python code.

---

## Assignment 13: Genetic Algorithm / Phototaxis

### Genetic Algorithm with JAX
```python
@jax.jit
def ga_step(key, population, fitnesses, mjx_model):
    key, k1, k2, k3 = jax.random.split(key, 4)
    pop_size = population.shape[0]

    # Tournament selection
    idx_a = jax.random.randint(k1, (pop_size,), 0, pop_size)
    idx_b = jax.random.randint(k2, (pop_size,), 0, pop_size)
    parents = jnp.where(
        (fitnesses[idx_a] > fitnesses[idx_b])[:, None, None],
        population[idx_a], population[idx_b]
    )

    # Crossover (uniform)
    mask = jax.random.bernoulli(k3, 0.5, population.shape)
    offspring = jnp.where(mask, parents, population[jnp.roll(idx_a, 1)])

    # Mutation
    key, k4 = jax.random.split(key)
    mutation = jax.random.normal(k4, offspring.shape) * 0.05
    mutate_mask = jax.random.bernoulli(key, 0.1, offspring.shape)
    offspring = offspring + mutation * mutate_mask

    # Evaluate
    new_fitnesses = evaluate_batch(offspring, mjx_model)

    # Elitism — keep best from previous generation
    best_idx = jnp.argmax(fitnesses)
    worst_idx = jnp.argmin(new_fitnesses)
    offspring = offspring.at[worst_idx].set(population[best_idx])
    new_fitnesses = new_fitnesses.at[worst_idx].set(fitnesses[best_idx])

    return key, offspring, new_fitnesses
```

### Phototaxis with MuJoCo
```xml
<!-- Add a light source as a target -->
<body name="light_target" pos="5 5 0.5">
  <geom type="sphere" size="0.3" rgba="1 1 0 1"/>
  <light pos="0 0 0" dir="0 0 -1"/>
</body>

<sensor>
  <!-- Rangefinder pointing toward light -->
  <rangefinder name="light_sensor" site="eye_site"/>
</sensor>
```

```python
def phototaxis_fitness(data):
    robot_pos = data.qpos[:2]   # x, y of robot
    target_pos = jnp.array([5.0, 5.0])
    distance = jnp.linalg.norm(robot_pos - target_pos)
    return -distance  # minimize distance = maximize negative distance
```

---

## Summary: What Students Gain

| Skill | PyBullet course | MuJoCo/MJX course |
|-------|----------------|-------------------|
| Physics simulation | Yes | Yes (better API) |
| Robot design | URDF (verbose) | MJCF (readable, modifiable) |
| Neural networks from scratch | Yes | Yes (identical) |
| Evolutionary algorithms | Yes | Yes (identical concepts) |
| Understanding parallelism | No (everything serial) | Yes (`vmap` is the core upgrade) |
| JIT compilation | No | Yes (`jax.jit`) |
| GPU programming concepts | No | Yes (transparent CPU→GPU transition) |
| Industry-standard tools | Declining (PyBullet) | Active (MuJoCo is the standard) |
| Differentiable simulation | No | Available for final projects |
| Functional programming | No | Yes (JAX's pure-function model) |

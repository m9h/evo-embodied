"""Assignment 1: Hello MuJoCo — drop a box, watch it fall.

Equivalent to the first pyrosim/PyBullet assignment: set up a physics
simulation with a ground plane and a single falling body.

Run: uv run python examples/01_hello_mujoco.py
"""
import mujoco
import numpy as np

# Define the world in MJCF XML
xml = """
<mujoco model="hello">
  <option gravity="0 0 -9.81" timestep="0.002"/>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="5 5 0.1" rgba="0.9 0.9 0.9 1"/>

    <body name="box" pos="0 0 3">
      <freejoint/>
      <geom type="box" size="0.5 0.5 0.5" mass="1" rgba="0.2 0.6 1 1"/>
    </body>
  </worldbody>
</mujoco>
"""

# Create model and data
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Simulate 2 seconds
print("Simulating box drop...")
print(f"{'Time':>8s}  {'Height':>8s}  {'Velocity':>10s}")
print("-" * 30)

for step in range(1000):
    mujoco.mj_step(model, data)

    if step % 100 == 0:
        t = data.time
        z = data.qpos[2]  # z-position of the box
        vz = data.qvel[2]  # z-velocity
        print(f"{t:8.3f}  {z:8.3f}  {vz:10.3f}")

print(f"\nFinal position: z = {data.qpos[2]:.3f}")
print("(Box should have landed on the ground plane at z ≈ 0.5)")

# Render a single frame to verify visualization works
try:
    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.update_scene(data)
    pixels = renderer.render()
    print(f"\nRenderer works: got {pixels.shape} image")
    renderer.close()
except Exception as e:
    print(f"\nRenderer unavailable (headless?): {e}")
    print("This is fine — simulation still works without visualization.")

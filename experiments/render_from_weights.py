"""Re-render video from saved weights.

Use this to generate videos from DGX runs that failed at the render step
(missing DISPLAY / no OpenGL context).

Usage:
    # Render the best run
    uv run python experiments/render_from_weights.py /data/evo-embodied/20260323-020138/

    # Render all runs that are missing videos
    uv run python experiments/render_from_weights.py --all /data/evo-embodied/
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mediapy as media
import mujoco
import numpy as np

MODEL_PATH = Path(__file__).parent.parent / "models" / "quadruped.xml"


def render_from_run(run_dir, render_ctrl_steps=200, fps=30):
    """Load weights and config from a run directory, render video."""
    run_dir = Path(run_dir)
    weights_path = run_dir / "best_weights.npy"
    config_path = run_dir / "config.json"

    if not weights_path.exists():
        print(f"  No best_weights.npy in {run_dir}, skipping")
        return

    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    n_hidden = config.get("n_hidden", 16)
    physics_per_ctrl = config.get("physics_per_ctrl", 20)
    clock_freq = config.get("clock_freq_hz", 2.0)
    # Use config's control_steps for render duration, or override
    ctrl_steps = render_ctrl_steps

    best_weights = np.load(weights_path)
    mj_model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    n_sensors = mj_model.nsensordata
    n_motors = mj_model.nu
    n_inputs = n_sensors + 2

    w1 = best_weights[:n_inputs * n_hidden].reshape(n_inputs, n_hidden)
    w2 = best_weights[n_inputs * n_hidden:].reshape(n_hidden, n_motors)

    data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, data)

    renderer = mujoco.Renderer(mj_model, height=720, width=1280)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    cam.distance = 3.0
    cam.azimuth = 150
    cam.elevation = -25

    dt = mj_model.opt.timestep
    ctrl_dt = dt * physics_per_ctrl
    steps_per_frame = max(1, int(1.0 / (fps * dt)))

    frames = []
    physics_step = 0

    for ctrl_i in range(ctrl_steps):
        t = ctrl_i * ctrl_dt
        clock = np.array([
            np.sin(2.0 * np.pi * clock_freq * t),
            np.cos(2.0 * np.pi * clock_freq * t),
        ])
        sensor_input = np.concatenate([data.sensordata.copy(), clock])
        ctrl = np.tanh(np.tanh(sensor_input @ w1) @ w2)
        data.ctrl[:] = ctrl

        for _ in range(physics_per_ctrl):
            mujoco.mj_step(mj_model, data)
            if physics_step % steps_per_frame == 0:
                renderer.update_scene(data, cam)
                frames.append(renderer.render().copy())
            physics_step += 1

    renderer.close()

    video_path = run_dir / "video.mp4"
    media.write_video(str(video_path), frames, fps=fps)
    print(f"  Saved {video_path} ({len(frames)} frames, "
          f"{video_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render video from saved weights")
    parser.add_argument("path", help="Run directory or base directory (with --all)")
    parser.add_argument("--all", action="store_true",
                        help="Render all runs missing videos in the base directory")
    parser.add_argument("--force", action="store_true",
                        help="Re-render even if video.mp4 already exists")
    parser.add_argument("--ctrl-steps", type=int, default=200,
                        help="Number of control steps to render (default: 200 = 8s)")
    args = parser.parse_args()

    if args.all:
        base = Path(args.path)
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            video_exists = (d / "video.mp4").exists() or (d / "walking_quadruped.mp4").exists()
            if video_exists and not args.force:
                print(f"Skipping {d.name} (video exists, use --force to re-render)")
                continue
            if not (d / "best_weights.npy").exists():
                continue
            print(f"Rendering {d.name}...")
            try:
                render_from_run(d, render_ctrl_steps=args.ctrl_steps)
            except Exception as e:
                print(f"  ERROR: {e}")
    else:
        render_from_run(args.path, render_ctrl_steps=args.ctrl_steps)

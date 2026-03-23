"""Reference demonstrations: see what good locomotion looks like.

Before evolving your own controllers, watch trained agents walk.
This script provides three levels of pre-trained locomotion demos,
from easiest to most advanced.

## Quick start (no extra deps):
    uv run python examples/00_reference_demos.py --brax-ant

## With demo extras:
    uv sync --extra demos
    uv run python examples/00_reference_demos.py --playground-go1

## Available demos:

  --brax-ant         Train a walking ant in ~3 min with PPO (brax, in rl extra)
  --brax-humanoid    Train a walking humanoid with PPO (brax, in rl extra)
  --playground-go1   Run pre-trained Unitree Go1 quadruped (playground, in demos extra)
  --playground-g1    Run pre-trained Unitree G1 humanoid (playground, in demos extra)
  --list             List all available demos
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")


def demo_brax_ant(render=True):
    """Train a Brax ant to walk in ~3 minutes, then render it.

    This is the fastest path to "see a robot learn to walk."
    Brax's PPO implementation + MJX backend = 50M timesteps in minutes.
    """
    try:
        import brax
    except ImportError:
        print("Install brax: uv sync --extra rl")
        return

    import jax
    from brax import envs
    from brax.training.agents.ppo import train as ppo_train
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"JAX: {jax.default_backend()}, {jax.devices()}")
    print("\n=== Training Brax Ant with PPO ===")
    print("This takes ~3 min on GPU, ~10 min on CPU.\n")

    # Create environment
    env = envs.get_environment("ant")

    # Track progress
    progress = []
    def progress_fn(num_steps, metrics):
        progress.append((num_steps, metrics["eval/episode_reward"]))
        reward = metrics["eval/episode_reward"]
        print(f"  Step {num_steps:>10,d}  reward={reward:.1f}", flush=True)

    # Train
    t0 = time.time()
    make_inference_fn, params, _ = ppo_train(
        environment=env,
        num_timesteps=20_000_000,  # 20M steps — enough for decent walking
        episode_length=1000,
        num_evals=10,
        reward_scaling=10.0,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=5,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        num_envs=2048,
        batch_size=1024,
        seed=0,
        progress_fn=progress_fn,
    )
    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed:.0f}s")

    # Save progress plot
    if progress:
        steps, rewards = zip(*progress)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(steps, rewards, "o-", linewidth=2)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Episode Reward")
        ax.set_title(f"Brax Ant PPO — trained in {elapsed:.0f}s")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig("brax_ant_training.png", dpi=150)
        plt.close(fig)
        print(f"Saved brax_ant_training.png")
        print(f"Final reward: {rewards[-1]:.1f}")

    if render:
        _render_brax(env, make_inference_fn, params, "brax_ant_demo.mp4")


def demo_brax_humanoid(render=True):
    """Train a Brax humanoid to walk."""
    try:
        import brax
    except ImportError:
        print("Install brax: uv sync --extra rl")
        return

    import jax
    from brax import envs
    from brax.training.agents.ppo import train as ppo_train

    print(f"JAX: {jax.default_backend()}, {jax.devices()}")
    print("\n=== Training Brax Humanoid with PPO ===")
    print("This takes ~5-10 min on GPU.\n")

    env = envs.get_environment("humanoid")

    def progress_fn(num_steps, metrics):
        reward = metrics["eval/episode_reward"]
        print(f"  Step {num_steps:>10,d}  reward={reward:.1f}", flush=True)

    t0 = time.time()
    make_inference_fn, params, _ = ppo_train(
        environment=env,
        num_timesteps=50_000_000,
        episode_length=1000,
        num_evals=10,
        reward_scaling=0.1,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=10,
        num_minibatches=32,
        num_updates_per_batch=8,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-3,
        num_envs=2048,
        batch_size=1024,
        seed=0,
        progress_fn=progress_fn,
    )
    print(f"\nTraining done in {time.time() - t0:.0f}s")

    if render:
        _render_brax(env, make_inference_fn, params, "brax_humanoid_demo.mp4")


def _render_brax(env, make_inference_fn, params, filename):
    """Render a trained Brax policy to MP4."""
    try:
        import mediapy as media
    except ImportError:
        print("Skipping render: mediapy not installed")
        return

    import jax
    from brax.io import html as brax_html

    print(f"Rendering to {filename}...")
    inference_fn = make_inference_fn(params)
    jit_inference = jax.jit(inference_fn)

    rng = jax.random.PRNGKey(0)
    state = jax.jit(env.reset)(rng)
    rollout = [state.pipeline_state]

    for _ in range(1000):
        act_rng, rng = jax.random.split(rng)
        action, _ = jit_inference(state.obs, act_rng)
        state = jax.jit(env.step)(state, action)
        rollout.append(state.pipeline_state)

    # Use brax's built-in rendering
    try:
        from brax.io import image
        frames = image.render_array(env.sys, rollout, height=480, width=640)
        media.write_video(filename, frames, fps=30)
        print(f"Saved {filename} ({len(frames)} frames)")
    except Exception as e:
        print(f"Video render failed: {e}")
        print("Try viewing in notebook: brax.io.html.render(env.sys, rollout)")


def demo_playground_go1():
    """Run the pre-trained Unitree Go1 quadruped from MuJoCo Playground.

    This is a real quadruped robot walking with a trained neural controller.
    No training required — the ONNX policy is pre-packaged.
    """
    try:
        from mujoco_playground import registry
    except ImportError:
        print("Install playground: uv sync --extra demos")
        return

    print("\n=== MuJoCo Playground: Unitree Go1 Locomotion ===")
    print("Pre-trained policy — no training needed.\n")

    # List available environments
    go1_envs = [e for e in registry.list_environments() if "go1" in e.lower()]
    print(f"Go1 environments: {go1_envs}")

    # Create environment
    env_name = "Go1JoystickFlatTerrain"
    if env_name not in registry.list_environments():
        # Try alternative names
        for name in go1_envs:
            if "flat" in name.lower() or "joystick" in name.lower():
                env_name = name
                break

    print(f"Loading {env_name}...")
    env = registry.load(env_name)
    print(f"Observation: {env.observation_size}, Action: {env.action_size}")
    print("\nTo run interactively with a viewer:")
    print(f"  python -m mujoco_playground.play --env {env_name}")
    print("\nSee: https://playground.mujoco.org/")


def demo_playground_g1():
    """Run the pre-trained Unitree G1 humanoid from MuJoCo Playground."""
    try:
        from mujoco_playground import registry
    except ImportError:
        print("Install playground: uv sync --extra demos")
        return

    print("\n=== MuJoCo Playground: Unitree G1 Humanoid ===")
    g1_envs = [e for e in registry.list_environments() if "g1" in e.lower()]
    print(f"G1 environments: {g1_envs}")
    print("\nTo run interactively:")
    print("  python -m mujoco_playground.play --env G1Joystick")


def list_demos():
    """List all available demos and their requirements."""
    print("Available demos:\n")
    print(f"  {'Demo':<25s} {'Extra needed':<20s} {'Description'}")
    print(f"  {'-'*25} {'-'*20} {'-'*40}")
    demos = [
        ("--brax-ant", "rl", "Train walking ant in ~3 min (PPO)"),
        ("--brax-humanoid", "rl", "Train walking humanoid in ~10 min (PPO)"),
        ("--playground-go1", "demos", "Pre-trained Go1 quadruped (instant)"),
        ("--playground-g1", "demos", "Pre-trained G1 humanoid (instant)"),
    ]
    for flag, extra, desc in demos:
        print(f"  {flag:<25s} {extra:<20s} {desc}")

    print("\nInstall extras:")
    print("  uv sync --extra rl       # brax + dm_control")
    print("  uv sync --extra demos    # mujoco-playground + onnxruntime")
    print("  uv sync --extra full     # everything")

    print("\nExternal resources (no install needed, just browse):")
    print("  HuggingFace SB3 Zoo: https://huggingface.co/sb3")
    print("    - sb3/sac-Ant-v3: trained walking ant (video on page)")
    print("    - sb3/sac-Humanoid-v3: trained walking humanoid")
    print("  MuJoCo Playground: https://playground.mujoco.org/")
    print("    - Interactive demos in browser")
    print("  Brax Colab: https://colab.research.google.com/github/google/brax/blob/main/notebooks/training.ipynb")
    print("    - Train any locomotion task interactively")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reference locomotion demos — see what good walking looks like")
    parser.add_argument("--brax-ant", action="store_true",
                        help="Train a Brax ant to walk (~3 min)")
    parser.add_argument("--brax-humanoid", action="store_true",
                        help="Train a Brax humanoid to walk (~10 min)")
    parser.add_argument("--playground-go1", action="store_true",
                        help="Pre-trained Unitree Go1 quadruped")
    parser.add_argument("--playground-g1", action="store_true",
                        help="Pre-trained Unitree G1 humanoid")
    parser.add_argument("--list", action="store_true",
                        help="List all available demos")
    parser.add_argument("--no-render", action="store_true",
                        help="Skip video rendering")
    args = parser.parse_args()

    if args.list or not any([args.brax_ant, args.brax_humanoid,
                             args.playground_go1, args.playground_g1]):
        list_demos()
        sys.exit(0)

    if args.brax_ant:
        demo_brax_ant(render=not args.no_render)
    if args.brax_humanoid:
        demo_brax_humanoid(render=not args.no_render)
    if args.playground_go1:
        demo_playground_go1()
    if args.playground_g1:
        demo_playground_g1()

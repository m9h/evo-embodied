#!/bin/bash
#SBATCH --job-name=evo-gait
#SBATCH --output=/data/evo-embodied/gait_%j.log
#SBATCH --error=/data/evo-embodied/gait_%j.log
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Coordinated gait evolution with GATED velocity fitness
# Velocity only counts when healthy (on ground + upright)
# Prevents leap-and-flail exploit that plagued distance and naive velocity fitness
#
# Submit: sbatch slurm/coordinated_gait.sbatch
# Monitor: tail -f /data/evo-embodied/gait_<jobid>.log
# Cancel:  scancel <jobid>

set -e

echo "=== Coordinated Gait Evolution ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo ""

cd /home/mhough/dev/evo-embodied

# Ensure environment is set up
if [ ! -d .venv ]; then
    echo "Setting up venv..."
    uv sync --extra gpu
fi

# Force EGL for headless rendering
export MUJOCO_GL=egl

# Run the coordinated gait evolution
echo "=== Running coordinated gait evolution ==="
uv run python examples/09_coordinated_gait.py \
    --output-dir /data/evo-embodied

echo ""
echo "=== Job complete ==="
echo "End: $(date)"

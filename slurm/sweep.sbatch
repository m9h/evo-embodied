#!/bin/bash
#SBATCH --job-name=evo-sweep
#SBATCH --output=/data/evo-embodied/sweep_%j.log
#SBATCH --error=/data/evo-embodied/sweep_%j.log
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8

# Full parameter sweep across all experiment configurations
#
# Submit:  sbatch slurm/sweep.sbatch
# Subset:  sbatch --export=EXPERIMENTS="hidden64,combined_best" slurm/sweep.sbatch
# Monitor: tail -f /data/evo-embodied/sweep_<jobid>.log

set -e

echo "=== Evo-Embodied Parameter Sweep ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo ""

cd /home/mhough/evo-embodied

if [ ! -d .venv ]; then
    echo "Setting up venv..."
    uv sync --extra gpu
fi

export MUJOCO_GL=egl

# Run specific experiments or full sweep
if [ -n "${EXPERIMENTS}" ]; then
    echo "Running selected experiments: ${EXPERIMENTS}"
    uv run python experiments/sweep.py \
        --output-dir /data/evo-embodied \
        --only "${EXPERIMENTS}"
else
    echo "Running full sweep (all experiments)"
    uv run python experiments/sweep.py \
        --output-dir /data/evo-embodied
fi

echo ""
echo "=== Sweep complete ==="
echo "End: $(date)"

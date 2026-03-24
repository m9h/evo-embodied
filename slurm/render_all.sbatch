#!/bin/bash
#SBATCH --job-name=evo-render
#SBATCH --output=/data/evo-embodied/render_%j.log
#SBATCH --error=/data/evo-embodied/render_%j.log
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Render videos for all runs that are missing them
# Quick job — usually finishes in minutes
#
# Submit: sbatch slurm/render_all.sbatch

set -e

echo "=== Rendering all missing videos ==="
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

cd /home/mhough/evo-embodied
export MUJOCO_GL=egl

uv run python experiments/render_from_weights.py \
    --all /data/evo-embodied/

echo ""
echo "=== Render complete ==="
echo "End: $(date)"

#!/bin/bash
# Setup evolutionary robotics environment on Fedora
set -e
cd "$(dirname "$0")"

echo "=== Fedora System Dependencies ==="
sudo dnf install -y uv gcc gcc-c++ python3-devel git \
    mesa-libGL mesa-libEGL mesa-dri-drivers \
    libglvnd-glx libglvnd-opengl \
    libxcb libxkbcommon libxkbcommon-x11 \
    xcb-util xcb-util-cursor xcb-util-image \
    xcb-util-keysyms xcb-util-renderutil xcb-util-wm \
    fontconfig freetype libX11 libXext

# GPU detection
echo ""
if lspci 2>/dev/null | grep -qi nvidia; then
    echo ">>> NVIDIA GPU detected."
    echo "    For GPU-accelerated evolution (MJX on CUDA):"
    echo "      uv sync --extra gpu"
    echo "    Ensure drivers are installed: sudo dnf install akmod-nvidia (RPM Fusion)"
elif lspci 2>/dev/null | grep -qi 'amd.*radeon\|amd.*navi\|amd.*vega'; then
    echo ">>> AMD GPU detected. JAX ROCm support is experimental."
    echo "    CPU MJX is still fast — 100x speedup over serial PyBullet."
else
    echo ">>> No discrete GPU detected — CPU-only mode."
    echo "    MJX on CPU still parallelizes via JAX vmap — significant speedup over PyBullet."
fi
echo ""

echo "=== Installing Python packages ==="
uv sync

echo "=== Registering Jupyter kernel ==="
uv run python -m ipykernel install --user \
    --name="evo-embodied" \
    --display-name="Evolutionary Robotics (MuJoCo/MJX)"

echo ""
echo "Ready!"
echo "  Jupyter:  cd $(pwd) && uv run jupyter lab"
echo "  Verify:   uv run python -c \"import mujoco; print(f'MuJoCo {mujoco.__version__}')\""

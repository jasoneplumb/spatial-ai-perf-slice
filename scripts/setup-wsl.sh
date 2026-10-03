#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
grep -qi microsoft /proc/version || { echo "ERROR: run inside WSL2"; exit 1; }
command -v nvidia-smi >/dev/null || { echo "ERROR: GPU not visible in WSL"; exit 2; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
sudo apt-get update
sudo apt-get install -y git python3 python3-venv python3-pip build-essential ninja-build
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
# Pin the PyTorch runtime independently of the host driver's CUDA 13.0 ceiling.
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
python tools/check_environment.py
cat <<'EOF'

Base GPU/PyTorch environment is ready.
CUDA toolkit/nvcc and spconv are intentionally NOT installed here yet.
OpenPCDet builds CUDA extensions, and current spconv prebuilt packages top out at
CUDA 12.0; we will pin the development-toolchain layer separately after this
base smoke test succeeds.
EOF

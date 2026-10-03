#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

grep -qi microsoft /proc/version || { echo "ERROR: run inside WSL2"; exit 1; }
command -v nvidia-smi >/dev/null || { echo "ERROR: GPU not visible in WSL"; exit 2; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

sudo apt-get update
sudo apt-get install -y git curl ca-certificates build-essential ninja-build

# Keep the benchmark independent of Ubuntu's system Python.
MINIFORGE_DIR="$HOME/miniforge3"
if [[ ! -x "$MINIFORGE_DIR/bin/conda" ]]; then
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  curl -fsSL -o "$TMP/miniforge.sh" \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash "$TMP/miniforge.sh" -b -p "$MINIFORGE_DIR"
fi

CONDA="$MINIFORGE_DIR/bin/conda"
ENV_DIR="$ROOT/.conda-env"

# Recreate the local environment so a failed/incompatible prior setup cannot leak in.
rm -rf "$ENV_DIR" .venv
"$CONDA" create -y -p "$ENV_DIR" python=3.11 pip
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126

python tools/check_environment.py

cat <<'EOF'

Base GPU/PyTorch environment is ready.
Expected: Python 3.11, PyTorch 2.7.0, CUDA runtime 12.6, RTX 2060, CC 7.5.
The repo environment is .conda-env; reactivate it later with:
  source ~/miniforge3/etc/profile.d/conda.sh
  conda activate ./.conda-env

CUDA toolkit/nvcc and spconv are intentionally deferred until this check passes.
EOF

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! grep -qi microsoft /proc/version; then
  echo "ERROR: This script is intended to run inside WSL2." >&2
  exit 1
fi

echo "== WSL / GPU preflight =="
uname -a
echo

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is not visible inside WSL."
  echo "Update/install the NVIDIA Windows driver with WSL CUDA support, then reopen Ubuntu."
  exit 2
fi
nvidia-smi
echo

echo "== Linux prerequisites =="
sudo apt-get update
sudo apt-get install -y git python3 python3-venv python3-pip build-essential ninja-build

if [[ ! -d "$ROOT/.venv" ]]; then
  python3 -m venv "$ROOT/.venv"
fi
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
python -m pip install --upgrade pip setuptools wheel

echo
echo "WSL and NVIDIA GPU are visible. Base Python environment is ready."
echo "Next run: ./scripts/setup.sh"

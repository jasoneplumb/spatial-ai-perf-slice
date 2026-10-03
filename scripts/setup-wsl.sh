#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

grep -qi microsoft /proc/version || { echo "ERROR: run inside WSL2"; exit 1; }
command -v nvidia-smi >/dev/null || { echo "ERROR: GPU not visible in WSL"; exit 2; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

sudo apt-get update
sudo apt-get install -y git build-essential ninja-build curl ca-certificates

# Ubuntu on this host currently defaults to Python 3.14. Keep the benchmark
# independent of the distro Python by installing a repo-local CPython 3.11.
PYTHON_VERSION=3.11.14
PYTHON_DIR="$HOME/.local/python-$PYTHON_VERSION"
if [[ ! -x "$PYTHON_DIR/bin/python3.11" ]]; then
  sudo apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev libffi-dev liblzma-dev tk-dev xz-utils \
    libncursesw5-dev uuid-dev
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  curl -fsSL "https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz" -o "$TMP/python.tgz"
  tar -xzf "$TMP/python.tgz" -C "$TMP"
  pushd "$TMP/Python-$PYTHON_VERSION" >/dev/null
  ./configure --prefix="$PYTHON_DIR" --with-ensurepip=install
  make -j"$(nproc)"
  make install
  popd >/dev/null
fi

rm -rf .venv
"$PYTHON_DIR/bin/python3.11" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126

python tools/check_environment.py

cat <<'EOF'

Base GPU/PyTorch environment is ready.
Expected: Python 3.11, PyTorch 2.7.0, CUDA runtime 12.6, RTX 2060, CC 7.5.
CUDA toolkit/nvcc and spconv are intentionally deferred until this check passes.
EOF

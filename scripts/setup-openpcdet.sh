#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "$ROOT/.conda-env"

python tools/check_environment.py

# spconv 2.3.8 publishes a CUDA 12.6 wheel for Python 3.11.
python -m pip install "spconv-cu126==2.3.8"

# OpenPCDet's own CUDA extensions require nvcc. Install a toolkit in the
# isolated conda environment; do not install an NVIDIA display driver in WSL.
conda install -y -p "$ROOT/.conda-env" -c nvidia cuda-toolkit=12.6

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="7.5"

nvcc --version
python tools/check_environment.py

bash ./scripts/setup.sh

PCDET="$ROOT/third_party/OpenPCDet"
cd "$PCDET"
git checkout master
git pull --ff-only
OPENPCDET_COMMIT="$(git rev-parse HEAD)"
echo "$OPENPCDET_COMMIT" > "$ROOT/third_party/OPENPCDET_COMMIT"

python -m pip install -e .

cd "$ROOT"
python - <<'PY'
import torch
import spconv.pytorch as spconv
import pcdet
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("spconv:", getattr(__import__("spconv"), "__version__", "installed"))
print("pcdet import: OK")
print("gpu:", torch.cuda.get_device_name(0))
PY

cat <<EOF

OpenPCDet development layer is ready.
Pinned OpenPCDet commit:
  $OPENPCDET_COMMIT

Reactivate later with:
  source ~/miniforge3/etc/profile.d/conda.sh
  conda activate "$ROOT/.conda-env"
  export CUDA_HOME="\$CONDA_PREFIX"
  export TORCH_CUDA_ARCH_LIST=7.5
EOF

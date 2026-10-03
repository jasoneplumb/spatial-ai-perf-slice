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
OPENPCDET_COMMIT="233f849829b6ac19afb8af8837a0246890908755"
cd "$PCDET"
git fetch origin "$OPENPCDET_COMMIT"
git checkout --detach "$OPENPCDET_COMMIT"
git reset --hard "$OPENPCDET_COMMIT"
echo "$OPENPCDET_COMMIT" > "$ROOT/third_party/OPENPCDET_COMMIT"

# OpenPCDet 233f849 names this spconv 2.x argument max_num_points, while
# spconv 2.3.8 exposes it as max_num_points_per_voxel. This is an API-name
# compatibility patch only; the value and voxelization semantics are unchanged.
python - <<'PY'
from pathlib import Path
p = Path("pcdet/datasets/processor/data_processor.py")
s = p.read_text()
old = "max_num_points=max_num_points_per_voxel,"
new = "max_num_points_per_voxel=max_num_points_per_voxel,"
if old in s:
    p.write_text(s.replace(old, new, 1))
elif new not in s:
    raise SystemExit("Unexpected Point2VoxelCPU3d constructor form; refusing to patch")
PY

python -m pip install --no-build-isolation -e .

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

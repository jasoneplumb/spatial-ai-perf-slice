#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "$ROOT/.conda-env"

# Optional: ONNX export and CPU reference execution of the dense segment
# (tools/export_dense_onnx.py). Not needed by the CUDA smoke/profiling path.
python -m pip install "onnx==1.17.0" "onnxruntime==1.20.1"

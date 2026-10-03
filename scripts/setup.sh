#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/third_party" "$ROOT/artifacts" "$ROOT/results"

if [[ ! -d "$ROOT/third_party/OpenPCDet/.git" ]]; then
  git clone https://github.com/open-mmlab/OpenPCDet.git "$ROOT/third_party/OpenPCDet"
fi

echo "OpenPCDet commit: $(git -C "$ROOT/third_party/OpenPCDet" rev-parse HEAD)"
echo
echo "Next:"
echo "  1. Install OpenPCDet using its INSTALL.md for your CUDA/PyTorch environment."
echo "  2. Prepare Waymo data according to OpenPCDet docs."
echo "  3. Obtain a checkpoint compatible with tools/cfgs/waymo_models/centerpoint.yaml."
echo "  4. Run tools/freeze_frame.py, then tools/profile_single_frame.py."

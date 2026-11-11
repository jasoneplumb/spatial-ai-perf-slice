#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/third_party" "$ROOT/artifacts" "$ROOT/results"

if [[ ! -d "$ROOT/third_party/OpenPCDet/.git" ]]; then
  git clone https://github.com/open-mmlab/OpenPCDet.git "$ROOT/third_party/OpenPCDet"
fi

echo "OpenPCDet commit: $(git -C "$ROOT/third_party/OpenPCDet" rev-parse HEAD)"
echo
echo "scripts/setup-openpcdet.sh pins, patches and builds this clone."
echo "The smoke test and the studies need no dataset or checkpoint."
echo "The optional reference baseline (tools/freeze_frame.py, tools/profile_single_frame.py)"
echo "needs Waymo data and a compatible checkpoint that you are entitled to use; neither is included."

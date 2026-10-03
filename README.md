# Spatial AI Performance Slice

A reproducible, measurement-first performance study of a single-frame LiDAR 3D-detection workload.

The initial workload is **CenterPoint (voxel/ResNet) on Waymo**, using OpenPCDet as the reference implementation. The purpose is not to reimplement CenterPoint. It is to establish a portable performance slice:

**fixed input -> reference output -> stage-aware timing -> profile -> optimization -> remeasurement**

No vendor-specific hardware, SDK, or confidential material belongs in this repository.

## Status

This initial scaffold covers the first four steps:

1. **Freeze the workload** — model/config/input/timing boundary are explicit.
2. **Run the unmodified reference path** — OpenPCDet is used without source changes.
3. **Measure a single frozen frame** — warmup + repeated CUDA-event timing, with machine-readable JSON output.
4. **Profile the same frame** — PyTorch Profiler captures CPU/CUDA operator activity and exports a Chrome/Perfetto-compatible trace.

See [docs/WORKLOAD.md](docs/WORKLOAD.md).

## Quick start

Prerequisites: Linux, NVIDIA GPU, CUDA-compatible PyTorch, and the normal OpenPCDet build dependencies.

```bash
./scripts/setup.sh
# Follow OpenPCDet's Waymo preparation instructions and obtain a compatible checkpoint.

python tools/freeze_frame.py \
  --openpcdet third_party/OpenPCDet \
  --cfg third_party/OpenPCDet/tools/cfgs/waymo_models/centerpoint.yaml \
  --index 0 \
  --out artifacts/frame-000000.pt

python tools/profile_single_frame.py \
  --openpcdet third_party/OpenPCDet \
  --cfg third_party/OpenPCDet/tools/cfgs/waymo_models/centerpoint.yaml \
  --checkpoint /path/to/checkpoint.pth \
  --frame artifacts/frame-000000.pt \
  --warmup 20 --iterations 100 \
  --out results/baseline
```

The profiler writes:

- `results/baseline/timing.json` — end-to-end inference distribution
- `results/baseline/operators.txt` — operator table sorted by CUDA time
- `results/baseline/trace.json` — timeline trace
- `results/baseline/prediction.pt` — reference prediction for later correctness checks

## Measurement boundary

The primary number is **model-call latency for one already-prepared frame**. Dataset I/O and Waymo decoding are intentionally outside the timed region. The frozen artifact preserves the model input so subsequent experiments operate on exactly the same frame.

A later phase should split the model call into voxelization / sparse 3D backbone / BEV mapping / 2D backbone / CenterHead / decode+NMS, then add target-device measurements.

## Why this shape

The artifact is deliberately small. It separates workload definition from target hardware so the same input, correctness reference, and measurement protocol can be carried to another CPU/GPU/accelerator implementation without changing the experiment's question.

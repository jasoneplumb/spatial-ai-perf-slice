# Spatial AI Performance Slice

A reproducible, measurement-first performance study of a single-frame LiDAR 3D-detection workload.

The initial workload is **CenterPoint (voxel/ResNet) on Waymo**, using OpenPCDet as the reference implementation. The purpose is not to reimplement CenterPoint. It is to establish a portable performance slice:

**fixed input -> reference output -> stage-aware timing -> profile -> optimization -> remeasurement**

No vendor-specific hardware, SDK, or confidential material belongs in this repository.

## Status

The development environment is pinned and the OpenPCDet reference build is working. The next two milestones are deliberately separate:

1. **Structural smoke test** — prove the exact Waymo CenterPoint graph executes on the local GPU using deterministic synthetic LiDAR-like points and random weights.
2. **Reference baseline** — freeze one licensed Waymo frame, run a compatible trained checkpoint, save the reference prediction, and measure the same model call repeatedly.

The smoke test is not a performance baseline.

OpenPCDet does **not** distribute its pretrained Waymo models because of the Waymo Dataset License Agreement. This repository therefore does not fetch an unofficial checkpoint. The reference baseline requires a checkpoint that the user is entitled to use.

See [docs/WORKLOAD.md](docs/WORKLOAD.md).

## Environment

On Windows 11, use the scripted WSL2 path in [docs/WINDOWS_WSL.md](docs/WINDOWS_WSL.md).

The current reference environment is:

- Python 3.11
- PyTorch 2.7.0 + CUDA 12.6
- CUDA toolkit / nvcc 12.6
- spconv 2.3.8
- OpenPCDet commit `233f849829b6ac19afb8af8837a0246890908755`
- NVIDIA RTX 2060, compute capability 7.5

Setup:

```bash
bash ./scripts/setup-wsl.sh
bash ./scripts/setup-openpcdet.sh
```

## Current next step: structural smoke test

```bash
bash ./scripts/smoke-centerpoint.sh
```

This uses the production CenterPoint graph and the Waymo configuration, but **randomly initialized weights** and **synthetic LiDAR-like points**. It answers only: does the configured graph execute successfully, and does it fit on this GPU?

The result is written to:

```text
results/smoke/centerpoint.json
```

It is explicitly marked `"valid_for_baseline": false`.

## Reference baseline

Once licensed Waymo data and a compatible trained checkpoint are available:

```bash
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

- `results/baseline/timing.json` — model-call latency distribution
- `results/baseline/operators.txt` — operator table sorted by CUDA time
- `results/baseline/trace.json` — timeline trace
- `results/baseline/prediction.pt` — reference prediction for later correctness checks

## Measurement boundary

The primary number is **`model(batch_dict)` latency for one already-prepared frame**. Frame loading, host copies, host-to-device transfer, dataset I/O, and Waymo decoding are outside the primary timed region.

A later phase will split the model call into sparse 3D backbone / BEV mapping / 2D backbone / CenterHead / decode+NMS and add target-device measurements.

## Why this shape

The artifact separates workload definition from target hardware so the same input, correctness reference, and measurement protocol can be carried to another CPU/GPU/accelerator implementation without changing the experiment's question.

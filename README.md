# Spatial AI Performance Slice

A reproducible, measurement-first performance study of a single-frame LiDAR 3D-detection workload.

The initial workload is **CenterPoint (voxel/ResNet) on Waymo**, using OpenPCDet as the reference implementation. The purpose is not to reimplement CenterPoint. It is to establish a portable performance slice:

**fixed input -> reference output -> stage-aware timing -> profile -> optimization -> remeasurement**

No vendor-specific hardware, SDK, or confidential material belongs in this repository.

## Status

**Milestone `v0.2.0-boundaries` is complete.** The workload's stage boundaries are characterized without licensed data:
- a verified tensor contract
- a license-free parametric lidar generator
- density, NMS and CPU sweeps with cost models tested on held-out inputs
- deterministic static ONNX exports of the dense segment and of the pillar variant

See [docs/MILESTONE-0.2.md](docs/MILESTONE-0.2.md). One of its findings corrects 0.1: the 0.1 smoke input overstated the sparse backbone's share of the frame (80% against 29–43% on surface-structured scans).

Milestone `v0.1.0-structural` ([docs/MILESTONE-0.1.md](docs/MILESTONE-0.1.md)) established the runnable, layered-profiling vertical slice. [CHANGELOG.md](CHANGELOG.md) has both.

The next milestone is still the **representative reference baseline**: one real point cloud, then a compatible trained checkpoint and frozen reference prediction. It is waiting on the data-licensing question.

No result in this repository is a performance baseline. Inputs are synthetic and weights are random.

OpenPCDet does **not** distribute its pretrained Waymo models because of the Waymo Dataset License Agreement. This repository therefore does not fetch an unofficial checkpoint. The reference baseline requires a checkpoint that the user is entitled to use.

See [docs/WORKLOAD.md](docs/WORKLOAD.md). The static tensor contract and stage boundaries are in [docs/WORKLOAD-CONTRACT.md](docs/WORKLOAD-CONTRACT.md).

### Studies

| Question | Tool | Write-up |
| --- | --- | --- |
| What crosses each stage boundary? | `tools/derive_contract.py`, `tools/inventory_boundaries.py` | [WORKLOAD-CONTRACT.md](docs/WORKLOAD-CONTRACT.md), [BOUNDARY-INVENTORY.md](docs/BOUNDARY-INVENTORY.md) |
| How does input structure drive cost? | `tools/scan_generator.py`, `tools/density_sweep.py` | [DENSITY-SWEEP.md](docs/DENSITY-SWEEP.md) |
| What do decode and NMS cost, and where do they run? | `tools/nms_sweep.py` | [NMS-SWEEP.md](docs/NMS-SWEEP.md) |
| Can the dense part leave PyTorch? | `tools/export_dense_onnx.py` | [ONNX-DENSE.md](docs/ONNX-DENSE.md) |
| What would a CPU fallback for the sparse backbone cost? | `tools/cpu_sparse_backbone.py` | [CPU-SPARSE-BACKBONE.md](docs/CPU-SPARSE-BACKBONE.md) |
| What if the sparse backbone is removed? | `tools/pillar_variant.py` | [PILLAR-VARIANT.md](docs/PILLAR-VARIANT.md) |

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
bash ./scripts/setup-onnx.sh        # optional: ONNX export tools only
```

## Structural smoke test

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

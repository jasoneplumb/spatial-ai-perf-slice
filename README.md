# Spatial AI Performance Slice

[![ci](https://github.com/jasoneplumb/spatial-ai-perf-slice/actions/workflows/ci.yml/badge.svg?branch=mainline)](https://github.com/jasoneplumb/spatial-ai-perf-slice/actions/workflows/ci.yml)
[![release](https://img.shields.io/github/v/release/jasoneplumb/spatial-ai-perf-slice)](https://github.com/jasoneplumb/spatial-ai-perf-slice/releases)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![python](https://img.shields.io/badge/python-3.11-blue)](docs/ENVIRONMENT.md)
[![status](https://img.shields.io/badge/results-structural%20profiling%2C%20not%20a%20benchmark-orange)](#what-the-results-are-and-are-not)

A reproducible, measurement-first performance study of a single-frame LiDAR 3D-detection workload.

The workload is **CenterPoint (voxel/ResNet) with the Waymo configuration**, using OpenPCDet as the reference implementation. The purpose is not to reimplement CenterPoint. It is to establish a portable performance slice:

**fixed input -> reference output -> stage-aware timing -> profile -> optimization -> remeasurement**

The study is hardware-neutral by design: it contains no vendor-specific code or SDKs, and nothing that is not publicly reproducible.

## Status

**Release `v0.2.0`.** The workload's stage boundaries are characterized without licensed data:
- a verified tensor contract
- a license-free parametric lidar generator
- density, NMS and CPU sweeps with cost models tested on held-out inputs
- deterministic static ONNX exports of the dense segment and of the pillar variant

See [docs/MILESTONE-0.2.md](docs/MILESTONE-0.2.md). One of its findings corrects 0.1: the 0.1 smoke input overstated the sparse backbone's share of the frame (80% against 29–43% on surface-structured scans).

Milestone `v0.1.0-structural` ([docs/MILESTONE-0.1.md](docs/MILESTONE-0.1.md)) established the runnable, layered-profiling vertical slice. [CHANGELOG.md](CHANGELOG.md) has both.

The next milestone is still the **representative reference baseline**: one real point cloud, then a compatible trained checkpoint and frozen reference prediction. It is waiting on the data-licensing question.

## What the results are, and are not

**This is structural profiling, not a validated performance benchmark.** No result in this repository is a performance baseline.

- **Inputs are synthetic.** Every scan comes from `tools/scan_generator.py` (numpy only) or from the older uniform-point generator in the smoke test. Real frames probably have more scattered occupancy.
- **Weights are random.** Detection counts, NMS load under real predictions, accuracy and quantization behavior are untested. The exported ONNX graphs must not be used for calibration or accuracy work.
- **One machine.** All latencies are from one NVIDIA RTX 2060 and one desktop x86-64 CPU under WSL2. They say nothing about other GPUs, embedded CPUs or accelerators.
- **What does transfer:** tensor shapes and dtypes at each boundary, which stages fall outside a standard dense graph, and which quantities drive cost and how they scale.

Every tool that writes measurements from synthetic input or random weights marks them `"valid_for_baseline": false`. The documented numbers were measured on the reference machine in October 2026. This release's automated checks do not re-run them: they cover syntax, the synthetic generator and repository invariants, and they need no GPU.

### No Waymo data or pretrained checkpoints

This repository contains **no Waymo data and no model weights**, and none is needed to reproduce anything in it. Waymo data and Waymo-trained weights fall under the Waymo Dataset License Agreement for Non-Commercial Use. OpenPCDet does **not** distribute its pretrained Waymo models for that reason, and this repository does not fetch an unofficial checkpoint. The reference-baseline path below needs a frame and a checkpoint that you are entitled to use. Anything derived from them must stay out of version control.

## Studies

| Question | Tool | Write-up |
| --- | --- | --- |
| What crosses each stage boundary? | `tools/derive_contract.py`, `tools/inventory_boundaries.py` | [WORKLOAD-CONTRACT.md](docs/WORKLOAD-CONTRACT.md), [BOUNDARY-INVENTORY.md](docs/BOUNDARY-INVENTORY.md) |
| How does input structure drive cost? | `tools/scan_generator.py`, `tools/density_sweep.py` | [DENSITY-SWEEP.md](docs/DENSITY-SWEEP.md) |
| What do decode and NMS cost, and where do they run? | `tools/nms_sweep.py` | [NMS-SWEEP.md](docs/NMS-SWEEP.md) |
| Can the dense part leave PyTorch? | `tools/export_dense_onnx.py` | [ONNX-DENSE.md](docs/ONNX-DENSE.md) |
| What would a CPU fallback for the sparse backbone cost? | `tools/cpu_sparse_backbone.py` | [CPU-SPARSE-BACKBONE.md](docs/CPU-SPARSE-BACKBONE.md) |
| What if the sparse backbone is removed? | `tools/pillar_variant.py` | [PILLAR-VARIANT.md](docs/PILLAR-VARIANT.md) |

The workload definition and measurement protocol are in [docs/WORKLOAD.md](docs/WORKLOAD.md).

## Quick check without a GPU

Needs only Python 3.11 or newer and numpy. In a Git checkout the repository checks cover the tracked files; in a source archive they cover every file present:

```bash
python -m pip install numpy
python -m unittest discover -s tests -v
python tools/scan_generator.py --seed 7 --out results/scans/scan_seed7.npy
```

The tests check Python and shell syntax, that the scan generator is deterministic and reproduces the documented seed-7 scan (`points_sha256` `c9d636f3…`), and repository invariants: no datasets, checkpoints or generated artifacts tracked, consistent pins and versions, and no broken documentation references. CI runs the same command with numpy 2.4.6 on Linux; the hash was also reproduced with numpy 2.5.4 on macOS. Another numpy version could in principle change the generated bits, which the hash test would report.

## Environment

The profiling tools need Linux on x86-64 with an NVIDIA GPU. The reference environment is:

- Windows 11 + WSL2 (Ubuntu); see [docs/WINDOWS_WSL.md](docs/WINDOWS_WSL.md)
- Python 3.11
- PyTorch 2.7.0 + CUDA 12.6
- CUDA toolkit / nvcc 12.6
- spconv 2.3.8
- OpenPCDet commit `233f849829b6ac19afb8af8837a0246890908755`
- NVIDIA RTX 2060, compute capability 7.5

Setup, from the repository root inside WSL2:

```bash
bash ./scripts/setup-wsl.sh         # Miniforge, .conda-env, PyTorch
bash ./scripts/setup-openpcdet.sh   # spconv, CUDA toolkit, pinned OpenPCDet in third_party/
bash ./scripts/setup-onnx.sh        # optional: ONNX export tools only
```

Reproducibility limits:

- `setup-wsl.sh` refuses to run outside WSL2, installs the latest Miniforge into `~/miniforge3`, and recreates `.conda-env` from scratch.
- PyTorch, torchvision, spconv, ONNX, onnxruntime and the OpenPCDet commit are pinned. numpy and other transitive dependencies are not.
- `setup-openpcdet.sh` builds OpenPCDet's CUDA extensions for compute capability 7.5 (`TORCH_CUDA_ARCH_LIST`). Change it for another GPU.
- `setup-openpcdet.sh` resets `third_party/OpenPCDet` to the pinned commit and renames one keyword argument for spconv 2.3.8.

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

## Reproducing the studies

Each write-up has a "Reproduce" section. The tools run from the repository root with `.conda-env/bin/python`. Two outputs are inputs to later tools, so produce them first:

```bash
.conda-env/bin/python tools/derive_contract.py --openpcdet third_party/OpenPCDet \
    --cfg third_party/OpenPCDet/tools/cfgs/waymo_models/centerpoint.yaml \
    --out results/contract/centerpoint_waymo.json
.conda-env/bin/python tools/scan_generator.py --seed 7 --out results/scans/scan_seed7.npy
```

Generated output goes to `results/` and `artifacts/`, which Git ignores ([results/README.md](results/README.md)).

## Reference baseline

Not part of this release. Once licensed Waymo data and a compatible trained checkpoint are available:

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

This path has not been run on real data.

## Measurement boundary

The primary number is **`model(batch_dict)` latency for one already-prepared frame**. Frame loading, host copies, host-to-device transfer, dataset I/O, and Waymo decoding are outside the primary timed region. Host voxelization, which also runs before the model call, is timed separately in the sweeps.

## Why this shape

The artifact separates workload definition from target hardware so the same input, correctness reference, and measurement protocol can be carried to another CPU/GPU/accelerator implementation without changing the experiment's question.

## License

Apache-2.0: see [LICENSE](LICENSE) and [NOTICE](NOTICE). Upstream projects, dependencies and data-licensing constraints are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). OpenPCDet and every other dependency are downloaded at setup time and are not redistributed here.

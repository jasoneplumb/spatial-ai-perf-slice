# Changelog

This project uses milestone versions to mark reproducible experimental states. A milestone version does not imply that the workload is yet suitable for cross-platform performance comparison.

## [Unreleased]

### Added

- `tools/derive_contract.py` and [docs/WORKLOAD-CONTRACT.md](docs/WORKLOAD-CONTRACT.md): the static input/output contract, stage list and per-boundary tensor shapes of the pinned CenterPoint workload, derived from the config and module source with provenance hashes. Frame-dependent dimensions stay `null`; execution and memory placement stay `unknown`.
- `tools/inventory_boundaries.py` and [docs/BOUNDARY-INVENTORY.md](docs/BOUNDARY-INVENTORY.md): Phase 1 inventory of every stage-boundary tensor, with CUDA allocation per stage, checked against the contract (19/19 match).
- `tools/scan_generator.py`: a deterministic parametric 64-beam lidar scan of a procedural street scene, with tunable sensor and scene parameters and no dataset content.

- `tools/density_sweep.py` and [docs/DENSITY-SWEEP.md](docs/DENSITY-SWEEP.md): a 90-scan sweep of clutter, buildings and azimuth density, with cost-model fits tested against held-out legacy inputs.
- `tools/pcdet_harness.py`: OpenPCDet setup shared by the inventory and sweep tools.

### Changed

- The contract now records that `load_data_to_gpu` converts the INT32 voxel coordinates and point counts to FP32.

### Observations

- Compared with the surface-structured parametric scan, the legacy synthetic smoke input (uniform points with no surfaces) inflates active sites at the strided sparse-backbone levels by 6–15× and model-call latency from 47 ms to 165 ms. Legacy smoke stage proportions should not be used to reason about where time goes.
- Across 90 parametric scans (77k–319k points), model-call latency stays within 44–54 ms. The dense 2D part (31 ± 1 ms, independent of input) is 57–71% of it. Sparse-backbone time is predicted across inputs only by channel-weighted active sites per level, not by voxel or raw site counts.

## [0.1.0-structural] - 2026-10-03

### Established

- Built and validated a reproducible Windows 11 / WSL2 development path for the CenterPoint workload.
- Pinned the reference stack:
  - Python 3.11.16
  - PyTorch 2.7.0 + CUDA 12.6
  - CUDA toolkit / nvcc 12.6
  - spconv 2.3.8
  - OpenPCDet commit `233f849829b6ac19afb8af8837a0246890908755`
  - NVIDIA GeForce RTX 2060, compute capability 7.5
- Added compatibility fixes needed to run the pinned OpenPCDet revision with the current environment:
  - editable install without pip build isolation
  - spconv 2.3.8 `Point2VoxelCPU3d` argument-name compatibility
  - bypass of unrelated optional dataset imports during CenterPoint smoke/replay paths
- Added a deterministic structural smoke test using the real Waymo CenterPoint graph with synthetic LiDAR-like input and random weights.
- Added stage timing, sparse-backbone substage timing, detailed `conv3` / `conv4` timing, active-site counts, operator profiling, and Chrome/Perfetto trace export.
- Added support for replaying a real model-ready `N x 5` point cloud or OpenPCDet-processed Waymo `N x 6` point cloud without requiring trained weights.
- Added frozen-frame and trained-checkpoint profiling tools for the later reference-baseline phase.
- Corrected the primary benchmark timing boundary to measure `model(batch_dict)` only, excluding frame loading, host copies, and host-to-device transfer.

### Structural observations

These measurements are from the deterministic synthetic smoke workload and are **not a benchmark baseline**.

- Model-call median: approximately **162.95 ms**
- Peak allocated VRAM: approximately **0.654 GiB**
- Input: 80,000 synthetic points
- Occupied input voxels: 79,295

Median stage latency:

| Stage | Median |
| --- | ---: |
| MeanVFE | 0.08 ms |
| VoxelResBackBone8x | 130.62 ms |
| HeightCompression | 2.94 ms |
| BaseBEVBackbone | 21.12 ms |
| CenterHead, including decode/NMS | 7.70 ms |
| Detector post-processing bookkeeping | ~0 ms |

Sparse-backbone median latency:

| Stage | Median | Active sites | Channels |
| --- | ---: | ---: | ---: |
| conv_input | 0.66 ms | 79,295 | 16 |
| conv1 | 1.35 ms | 79,295 | 16 |
| conv2 | 17.60 ms | 277,807 | 32 |
| conv3 | 50.22 ms | 267,820 | 64 |
| conv4 | 58.89 ms | 91,026 | 128 |
| conv_out | 1.92 ms | 44,435 | 128 |

The detailed pass showed that the repeated SubMConv3d operations inside the `conv3` and `conv4` residual blocks dominate those stages. Operator profiling independently showed the largest CUDA consumers to be CUMM sparse-convolution kernels rather than sparse-index construction alone.

### Why this matters

This milestone turns the workload from a paper/configuration exercise into a working vertical slice:

1. **The real execution path is proven.** The pinned CenterPoint graph executes through OpenPCDet, spconv, native CUDA extensions, decode, and NMS on the local GPU.
2. **The measurement hierarchy is established.** Whole-model, architectural-stage, sparse-backbone, individual-layer, and operator views now agree closely enough to localize cost.
3. **The dominant structural cost is understood.** For the synthetic occupancy pattern, the wide sparse residual convolutions in `conv3` and `conv4` dominate execution.
4. **The synthetic workload is kept separate from the benchmark baseline.** Random weights and unusually dispersed synthetic occupancy are useful for structural understanding but are explicitly excluded from performance claims.
5. **The next experiment is small and well defined.** A single real Waymo point cloud can now test whether realistic occupancy preserves the same sparse-backbone behavior even before a trained checkpoint is available.
6. **The methodology is portable.** The fixed workload boundary and layered profiling approach can later be carried to another CPU/GPU/accelerator target without changing the experimental question.

### Known limitations

- The current structural measurements use random model weights.
- The synthetic 80k-point input creates nearly one occupied input voxel per point and should not be treated as representative Waymo occupancy.
- Detection counts from random weights have no semantic meaning.
- CenterHead/NMS behavior will change with trained weights and realistic predictions.
- A licensed representative Waymo frame and compatible trained CenterPoint checkpoint are still required for the reference baseline.
- No target-device optimization has been attempted yet.

### Next milestone

The next milestone is the **representative reference baseline**:

1. profile one real Waymo point cloud through the same graph and compare occupancy/stage behavior with the synthetic structural case;
2. acquire a compatible trained CenterPoint checkpoint under appropriate dataset/model terms;
3. freeze a representative input and reference prediction;
4. run at least 100 model-call measurements plus an operator trace;
5. establish correctness tolerances;
6. only then choose and measure the first optimization.

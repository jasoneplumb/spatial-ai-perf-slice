# Changelog

This project uses milestone versions to mark reproducible experimental states. A milestone version does not imply that the workload is yet suitable for cross-platform performance comparison.

## [Unreleased]

### Added

- Release workflow: pushing a plain `vX.Y.Z` tag re-runs the release checks against the tagged tree, checks the tag against the changelog, and publishes a GitHub release from `.github/release-notes/<tag>.md`. It can also be dispatched by hand for an existing tag.
- Claude review workflow, triggered by the `review-requested` label on a pull request.

### Changed

- CI pins its actions to commit SHAs, cancels superseded runs, and can be dispatched by hand.
- Release notes moved from `docs/` to `.github/release-notes/`.
- README improvements: the README and `results/README.md` state the scope directly: a portable protocol, reference measurements from one NVIDIA GPU, and nothing that is not publicly reproducible.

## [0.2.0] - 2026-11-11

First public release. Boundary characterization without licensed data; see [docs/MILESTONE-0.2.md](docs/MILESTONE-0.2.md). The measurements were completed on 2026-10-07 and are unchanged by the release preparation.

Nothing in this release is a performance baseline: inputs are synthetic, weights are random, and all timings come from one RTX 2060 / desktop x86-64 machine. No Waymo data or pretrained checkpoint is included.

### Release preparation

- Licensed the repository under Apache-2.0 (`LICENSE`, `NOTICE`) and added [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md): upstream projects, dependencies, their licenses, and the Waymo data constraint.
- Added `tests/test_release.py` and a CI workflow. They need no GPU and check Python and shell syntax, scan-generator determinism against the documented seed-7 hash, and repository invariants (nothing restricted tracked, consistent OpenPCDet pin and version strings, resolvable documentation references). They do not run the CUDA tools.
- README: stated what the results are and are not, added a GPU-free quick check, the order in which the studies must be reproduced, and the limits of the setup scripts' reproducibility.
- Corrected stale setup instructions in [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) and [docs/WINDOWS_WSL.md](docs/WINDOWS_WSL.md), which still described a `.venv` that the setup script replaced with `.conda-env`.

### Added

- `tools/derive_contract.py` and [docs/WORKLOAD-CONTRACT.md](docs/WORKLOAD-CONTRACT.md): the static input/output contract, stage list and per-boundary tensor shapes of the pinned CenterPoint workload, derived from the config and module source with provenance hashes. Frame-dependent dimensions stay `null`; execution and memory placement stay `unknown`.
- `tools/inventory_boundaries.py` and [docs/BOUNDARY-INVENTORY.md](docs/BOUNDARY-INVENTORY.md): Phase 1 inventory of every stage-boundary tensor, with CUDA allocation per stage, checked against the contract (19/19 match).
- `tools/scan_generator.py`: a deterministic parametric 64-beam lidar scan of a procedural street scene, with tunable sensor and scene parameters and no dataset content.
- `tools/density_sweep.py` and [docs/DENSITY-SWEEP.md](docs/DENSITY-SWEEP.md): a 90-scan sweep of clutter, buildings and azimuth density, with cost-model fits tested against held-out legacy inputs.
- `tools/nms_sweep.py` and [docs/NMS-SWEEP.md](docs/NMS-SWEEP.md): decode and rotated-NMS cost against candidate count, driven by constructed inputs independent of the weights, with GPU-busy time and CUDA runtime calls per frame.
- `tools/export_dense_onnx.py`, `scripts/setup-onnx.sh` and [docs/ONNX-DENSE.md](docs/ONNX-DENSE.md): a deterministic, fully static ONNX export of the dense segment (`spatial_features` → raw head outputs) with constant folding, shape inventory, per-convolution MAC counts, and an onnxruntime-versus-PyTorch numeric check.
- `tools/cpu_sparse_backbone.py` and [docs/CPU-SPARSE-BACKBONE.md](docs/CPU-SPARSE-BACKBONE.md): the sparse 3D backbone on the CPU (spconv Native algorithm, bias applied by hook, checked against the GPU), split into rulebook construction and gather–GEMM–scatter, at 1, 4 and 16 threads.
- `tools/pillar_variant.py` and [docs/PILLAR-VARIANT.md](docs/PILLAR-VARIANT.md): stage timing of the pinned `centerpoint_pillar_1x.yaml`, and a static ONNX export of its whole network from voxelizer output to raw head outputs (pillars padded to P_MAX, batch-1 scatter and PillarVFE features rewritten, identical to OpenPCDet's modules).
- `tools/scan_generator.py` version 2: optional short-range lidars, ego body and edge second returns, with `--preset waymo-like` calibrated to the published 177K average points per frame. Default parameters reproduce version 1 bit for bit.
- `tools/pcdet_harness.py`: OpenPCDet setup shared by the inventory and sweep tools.

### Changed

- `fold_constants` no longer stores generated tensors (`ConstantOfShape`, `Expand`) as weights, and the ONNX inventory understands BOOL tensors and zero-length dimensions. The dense-segment export is unchanged.
- The contract now records that `load_data_to_gpu` converts the INT32 voxel coordinates and point counts to FP32.

### Observations

- Compared with the surface-structured parametric scan, the legacy synthetic smoke input (uniform points with no surfaces) inflates active sites at the strided sparse-backbone levels by 6–15× and model-call latency from 47 ms to 165 ms. Legacy smoke stage proportions should not be used to reason about where time goes.
- The pinned rotated NMS runs its greedy suppression on the host CPU after a blocking device-to-host copy. Decode + NMS costs about 2 ms per frame, nearly independent of candidate count (M ≤ 500, capped by top-K). 58–82% of it is host-side and synchronization overhead (5 stream syncs, 1 blocking copy, 1 device allocation per frame).
- The dense segment is 85.78 GMAC per frame and uses only Conv, Relu, Pad, ConvTranspose, BatchNormalization and Concat. As exported, `nn.ZeroPad2d` becomes a `Pad` with runtime-computed pads, which leaves 50 tensors with symbolic shapes until the constant chain is folded.
- On a desktop CPU the sparse backbone takes 171–389 ms on realistic scans even at 16 threads, 12–17× its GPU time. Rulebook construction does not speed up with threads (28–33% of time at 16 threads), and the whole stage speeds up only 1.5–1.7× from 1 to 16 threads. The channel-weighted MAC proxy predicts CPU time too (R² 0.93–0.97; held-out error −8.5% to −1%).
- The pillar CenterPoint variant has no sparse 3D backbone and exports as one static ONNX graph with a single scatter. It does 2.6× the dense work (223.9 vs 85.8 GMAC, with the head at stride 1), takes 90 ms on the reference GPU against 44–54 ms, and is documented upstream at 2.7–5.8 points lower Waymo L2 mAPH.
- Adding four short-range lidars, the ego body and edge second returns (seed 7, 183k points) raises voxels by 7.5%, active sites by 6–11% and sparse-backbone time by 13%. The near-field points mostly fall into voxels that are already occupied.
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

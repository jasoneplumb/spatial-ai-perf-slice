# Milestone 0.2: Boundary characterization without licensed data

**Date:** 2026-10-07
**Milestone name:** `v0.2.0-boundaries`
**Reference implementation commit before milestone documentation:** `b30c74f362993b606c94b081a67ad27cf9c52ff5`

## Why this milestone took a different route

Milestone 0.1 named one real Waymo point cloud as the next experiment. That is still the right experiment, but the Waymo Dataset License Agreement (March 2025) restricts use to non-commercial purposes. Whether a given use qualifies has to be settled before Waymo frames or Waymo-trained weights enter this work. Until then, this milestone answers what can be answered **without** licensed data:

- the exact tensor contract
- how input structure drives cost
- which stages fall outside a standard graph export
- what a CPU fallback would cost

Every input here is generated (`tools/scan_generator.py`, numpy only). Every model uses random weights.

## What exists at this point

```text
parametric 64-beam scan (+ optional short-range lidars, ego body, second returns)
        ↓  host: range mask, shuffle, voxelize            ← CPU, scales with points
MeanVFE → VoxelResBackBone8x (spconv)                      ← no standard-ONNX form
        ↓  HeightCompression: sparse → dense [1,256,188,188]
BaseBEVBackbone → CenterHead convs                         ← static ONNX, 85.8 GMAC
        ↓  decode / top-K / rotated NMS                    ← greedy loop on the host CPU
detections
```

| Tool | What it establishes | Write-up |
| --- | --- | --- |
| `derive_contract.py` | Static I/O contract and per-boundary shapes, derived from config and source, with hashes | [WORKLOAD-CONTRACT.md](WORKLOAD-CONTRACT.md) |
| `inventory_boundaries.py` | Every boundary tensor observed and checked against the contract (19/19), with allocations | [BOUNDARY-INVENTORY.md](BOUNDARY-INVENTORY.md) |
| `scan_generator.py` | Deterministic, tunable, license-free lidar scans; v2 adds short-range lidars, ego body, edge second returns | [BOUNDARY-INVENTORY.md](BOUNDARY-INVENTORY.md) |
| `density_sweep.py` | 90-scan sweep of clutter, buildings and point density; cost-model fits tested on held-out inputs | [DENSITY-SWEEP.md](DENSITY-SWEEP.md) |
| `nms_sweep.py` | Decode/NMS cost by candidate count, independent of weights | [NMS-SWEEP.md](NMS-SWEEP.md) |
| `export_dense_onnx.py` | Deterministic static ONNX of the dense segment, with inventory, MACs and a numeric check | [ONNX-DENSE.md](ONNX-DENSE.md) |
| `cpu_sparse_backbone.py` | Sparse backbone on the CPU: rulebook vs gather–GEMM–scatter, by thread count | [CPU-SPARSE-BACKBONE.md](CPU-SPARSE-BACKBONE.md) |
| `pillar_variant.py` | Pillar CenterPoint (no sparse backbone): timing and a static whole-network ONNX | [PILLAR-VARIANT.md](PILLAR-VARIANT.md) |

`pcdet_harness.py` holds the OpenPCDet setup these tools share. The pinned environment is unchanged from 0.1. `scripts/setup-onnx.sh` adds `onnx==1.17.0` and `onnxruntime==1.20.1` for the export tools only.

## What was learned

**1. Milestone 0.1's headline cost split was an artifact of its synthetic input.** The 0.1 smoke input spread 30% of its points uniformly through the air, so nearly every point formed its own voxel. Isolated voxels are the worst case for strided sparse convolution.

| | 0.1 smoke input (80k points) | Surface-structured scans (90, 77k–319k points) |
| --- | ---: | ---: |
| Model call, RTX 2060 | 165 ms | 44–54 ms |
| Sparse backbone share | 80% | 29–43% |
| Dense 2D part (input-independent) | ~30 ms | 30–32 ms (57–71%) |

The 0.1 measurements remain correct for that input. They should not be used to reason about where time goes.

**2. Sparse-backbone cost follows channel-weighted active sites per level.** Voxel counts and raw site counts miss it by 2.5–6.5× on held-out inputs. Active sites weighted by kernel volume × channels predict held-out cost within −8% to +20% on the GPU, and within −8.5% to −1% on the CPU.

**3. Three stages fall outside a standard dense graph:**

| Stage | Where it runs in the reference | Cost |
| --- | --- | --- |
| Voxelization | host CPU, before `model()` | 2–8 ms, scales with point count |
| Sparse 3D backbone | GPU (spconv) | no standard-ONNX operator |
| Rotated NMS | GPU mask, then greedy loop on the host | ~2 ms per frame for decode + NMS, 58–82% of it fixed host and sync overhead |

The rotated NMS round trip is: GPU builds the overlap mask, blocking copy to the host, greedy suppression on the CPU, copy back. Each frame makes 5 stream syncs, 1 blocking copy and 1 device allocation, largely independent of candidate count.

**4. A stock CPU fallback for the sparse backbone would dominate the frame.** On a 16-thread desktop CPU it takes 171–389 ms, 12–17× its GPU time. Rulebook construction does not speed up with threads at all, and the whole stage speeds up only 1.5–1.7× from 1 to 16 threads.

**5. The dense segment exports cleanly once one quirk is handled.** `nn.ZeroPad2d` exports with runtime-computed pads, which leaves 50 tensors with symbolic shapes. After bounded constant folding the graph is fully static: Conv/ReLU/Pad/ConvTranspose/BN/Concat, 85.78 GMAC. It matches PyTorch within 4.3 × 10⁻⁶.

**6. The pillar variant removes the sparse stage at a price.** `centerpoint_pillar_1x.yaml` exports as one static graph from voxelizer output to heads, with a single scatter. It does 2.6× the dense work (223.9 GMAC, the head at full 468 × 468 resolution), carries a 336 MB FP32 BEV feature, and takes 90–92 ms on the GPU. Upstream it is documented at 2.7–5.8 points lower Waymo L2 mAPH.

**7. More sensors change little.** Short-range lidars, the ego body and edge second returns (183k points, calibrated to the published 177K average) raise active sites by 6–11% and sparse-backbone time by 13%. The near-field points mostly land in voxels that are already occupied.

## Why this is a useful reference point

- **Every boundary has a verified contract.** Shapes, dtypes and conversions are derived, observed and cross-checked, including one the derivation missed: INT32 → FP32 at host-to-device.
- **Cost now has a model, not just a measurement.** Per-level active sites predict sparse cost across very different inputs, on GPU and CPU alike.
- **The CPU-dependent work is located and sized.** Host voxelization, the NMS host round trip and the sparse backbone's rulebook construction are each measured. Each is separable from the accelerator-friendly dense work.
- **There are portable artifacts.** Two static ONNX graphs (dense segment; whole pillar network) are reproducible bit for bit from this commit.
- **All of it is license-free**, so it can be shared and reproduced without dataset access.

## What this milestone does not claim

- **Nothing here is a performance baseline.** Weights are random, inputs synthetic, and all latencies are for an RTX 2060 and a desktop x86 CPU.
- **The synthetic scans are probably still optimistic.** They lack vegetation and other irregular clutter, so real frames likely have more active sites. Where a real frame falls between the scans and the 0.1 smoke input is unknown.
- **Detection counts, NMS load under real predictions, accuracy and quantization behavior are untested.** They need trained weights.
- **The ONNX graphs are not calibrated.** They must not be used for INT8 calibration or accuracy work.
- **The CPU results come from a stock library path on a desktop CPU.** They say nothing about any embedded core.

## Next milestone

The **representative reference baseline** from 0.1 still stands, once the data-licensing question is settled:

1. one real point cloud through the same tools: inventory, per-level active sites, and its position relative to the synthetic sweep;
2. a trained checkpoint the user is entitled to use, a frozen reference prediction, and real detection and NMS load.

Independent of licensing:

- **target-side evidence:** compiler placement reports for the two ONNX graphs, and the CPU-side stages measured on an AArch64 core;
- **a vegetation-like clutter model,** to test whether scattered occupancy is what separates synthetic from real frames.

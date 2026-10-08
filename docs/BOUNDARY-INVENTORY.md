# Boundary inventory and synthetic-input realism

Phase 1 of boundary characterization. Every tensor that crosses a stage boundary was observed on one forward pass and checked against the static contract in [WORKLOAD-CONTRACT.md](WORKLOAD-CONTRACT.md). The same instrumentation also showed that the original synthetic input misrepresents the workload, so a parametric scan generator was added.

All results use **random weights** and **synthetic input**. Shapes, dtypes, sparse active sites and allocations are valid for the inputs used. Detection counts are not representative, and none of this is a performance baseline. Bytes and allocations describe the PyTorch/CUDA reference implementation on an RTX 2060, not any other target.

## Reproduce

```bash
# static contract (Phase 0)
.conda-env/bin/python tools/derive_contract.py --openpcdet third_party/OpenPCDet \
    --cfg third_party/OpenPCDet/tools/cfgs/waymo_models/centerpoint.yaml \
    --out results/contract/centerpoint_waymo.json

# deterministic parametric scan (writes .npy plus a .json sidecar with params and sha256)
.conda-env/bin/python tools/scan_generator.py --seed 7 --out results/scans/scan_seed7.npy

# boundary inventory; exits non-zero on any contract mismatch
.conda-env/bin/python tools/inventory_boundaries.py --points-npy results/scans/scan_seed7.npy --label scan-seed7
.conda-env/bin/python tools/inventory_boundaries.py --legacy-synthetic 80000 --label legacy-80k

# timing on the same input (unchanged smoke test)
.conda-env/bin/python tools/smoke_centerpoint.py --points-npy results/scans/scan_seed7.npy \
    --warmup 5 --iterations 20 --out results/smoke-scan/centerpoint.json
```

## Inventory method

The inventory follows the dataset path (range mask, seeded test-time shuffle, `Point2VoxelCPU3d`, collate, `load_data_to_gpu`). It then runs `model.module_list` one module at a time. For every new or replaced entry in `batch_dict`, it records:

- shape, dtype, device and strides
- contiguity
- logical bytes and backing-storage bytes
- whether the tensor shares storage with another boundary tensor

It also wraps CenterHead's decode to capture the raw head outputs, and wraps class-agnostic NMS to capture its inputs. Forward hooks on the sparse backbone record active sites per level. Each stage also records how much CUDA allocated memory changed, and its peak above the starting level.

**Result: all 19 contract boundaries matched on both inputs** (rank, every static dimension, configured bounds, dtype). Every boundary tensor is contiguous, and none aliases another boundary tensor.

## Findings

1. **Integer-to-float round trip.** `load_data_to_gpu` converts the voxelizer's INT32 coordinates and point counts to FP32, and the sparse backbone converts the coordinates back to INT32. The static derivation missed this; the contract has been corrected. Details are in [WORKLOAD-CONTRACT.md](WORKLOAD-CONTRACT.md#integer-to-float-round-trip-at-host-to-device).
2. **The sparse backbone keeps far more memory than its output.** On the legacy input it adds 436 MB of allocated memory (peak 508 MB above start), against a 23 MB output. About 179 MB of that is the per-level feature tensors, including the output and those kept in `multi_scale_3d_features` (computed from the active-site counts × channels × 4 bytes). The remaining ~257 MB is presumably spconv index/rulebook state, which stays referenced for the rest of the forward pass. That split has not been verified. On the scan input the same figures are 66 MB and 74 MB.
3. **Allocation after the sparse-to-dense step does not depend on the input.** HeightCompression retains 36.2 MB and briefly peaks at twice that inside `dense()`. BaseBEVBackbone retains 73.4 MB with a peak of 172.9 MB. These are identical across inputs, as the static shapes predict.
4. **The voxel cap drops points.** On the scan, 24,118 of 158,206 in-range points (15%) exceed `MAX_POINTS_PER_VOXEL = 5` and are discarded.

## How realistic is the legacy generator?

`smoke_centerpoint.synthetic_points` has three problems:

- 70% of its points are spread evenly across the ground area, with no beam rings and no falloff with range.
- The other 30% are drawn uniformly through a volume 1.2–3.5 m above the ground: floating noise that lies on no surface.
- There is no occlusion.

Real lidar returns lie on surfaces, along beam rings, and stop at the first object they hit.

The result is that almost every point gets its own voxel. 99.1% of voxels hold one point, and 80,000 points produce 79,295 voxels. Isolated voxels are the worst case for strided sparse convolution, because each one turns on a full 3×3×3 neighborhood of output sites.

| | Legacy (80k uniform) | Parametric scan (seed 7) |
| --- | ---: | ---: |
| Points in range | 80,000 | 158,206 |
| Voxels V | 79,295 | 55,186 |
| Voxels with 1 / 2 / 3 / 4 / 5 points | 78,598 / 689 / 8 / 0 / 0 | 22,429 / 10,755 / 7,149 / 5,563 / 9,290 |
| Active sites, conv2 (relative to V) | 277,807 (3.50) | 45,088 (0.82) |
| Active sites, conv3 | 267,820 (3.38) | 18,248 (0.33) |
| Active sites, conv4 | 91,026 (1.15) | 7,110 (0.13) |
| Active sites, conv_out | 44,435 (0.56) | 5,592 (0.10) |
| BEV cells occupied | 62.9% | 7.9% |
| Model-call median / p95 | 165.0 / 166.2 ms | 47.3 / 48.2 ms |
| VoxelResBackBone8x median | 132.5 ms | 16.5 ms |
| BaseBEVBackbone + CenterHead | 30.1 ms | 27.7 ms |
| Peak allocated VRAM | 0.94 GiB | 0.35 GiB |

Both rows come from 5 warmup and 20 measured iterations on the same GPU and software.

The legacy input makes the sparse backbone about 8× more expensive and makes it dominate the frame (80%). With surface-structured input, sparse 3D work (35%) and dense 2D work (59%) are comparable. The dense part doesn't depend on the input, so **the CPU-versus-accelerator balance of the workload depends heavily on input realism.** Conclusions drawn from the legacy smoke numbers about where time goes should not be carried forward.

## The parametric scan generator

`tools/scan_generator.py` casts one ring of rays from a single sensor into a procedural street scene and keeps the first-return hits. The scene contains:

- a ground plane
- lane-aligned vehicles
- cyclists
- pedestrians on sidewalks
- poles
- building blocks with gaps

Every parameter is exposed as a CLI flag. The same parameters and seed give bit-identical output, recorded as `points_sha256` in the sidecar file. It uses only numpy and no dataset content.

| Parameter | Default | Basis |
| --- | --- | --- |
| beams | 64 | design choice |
| inclination range | −17.6° to +2.4° | published: Waymo Open Dataset paper, Table 2 (top lidar VFOV) |
| max range | 75 m | published: same table |
| azimuth samples | 2650 | **assumption**: commonly cited range-image width; not verified against an official source |
| beam spacing | uniform in inclination (`inclination_gamma = 1`) | **assumption**: real per-beam inclinations are not used |
| sensor height / ground z | 2.0 m / 0.0 m | **assumption** |
| dropout / range noise | 5% / 0.02 m | **assumption** |
| scene counts and sizes | see `ScanParams` | **assumption** |

What it doesn't model:

- **The four short-range lidars and second returns.** OpenPCDet's Waymo preprocessing merges all five lidars and both returns (`use_two_returns=True`). The paper reports an average of 177K points per frame across all returns. One first-return ring gives about 158K here, but with a different spatial distribution.
- **The ego vehicle body.**
- **Vegetation and other irregular clutter.**
- **Terrain slope.**
- **Motion during the sweep.**
- **Physical intensity or elongation.** These values are placeholders, and with random weights they don't affect any measured quantity.

The missing clutter and extra sensors would add voxels and active sites. So the scan is probably an *optimistic* (sparse) bound and the legacy input a *pessimistic* (dense, isolated) one. Where a real frame falls between them is unknown without real data.

## Next steps

1. Sweep the generator's scene density (objects, buildings, dropout) and point count to turn the sparse backbone's cost into curves of V and S, rather than single points.
2. Add an optional short-range-lidar and second-return model, so that point counts and near-field density approach the published 177K average.
3. Export the dense segment (`spatial_features` → head outputs) to ONNX. Its contract is fully static and has now been verified.

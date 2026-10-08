# Workload contract — static shapes, boundaries, and provenance

Phase 0 of boundary characterization: what the pinned CenterPoint workload consumes and produces, which tensors cross each stage boundary, and which of those facts are known versus still missing.

Everything here is **derived** from the pinned configuration and module source. Nothing here is measured, and no stage is assigned to any processor, memory, or accelerator. Those fields are `unknown` until a partition report or measurement supports them.

## Reproduce

```bash
.conda-env/bin/python tools/derive_contract.py \
    --openpcdet third_party/OpenPCDet \
    --cfg third_party/OpenPCDet/tools/cfgs/waymo_models/centerpoint.yaml \
    --out results/contract/centerpoint_waymo.json
```

The script loads the config through OpenPCDet's own loader, so `_BASE_CONFIG_` is merged exactly as the model sees it. It then computes the grid and every static shape using the same formulas as the module source. It asserts two independent consistency checks:

- channels × height after the sparse backbone equals `NUM_BEV_FEATURES` (128 × 2 = 256)
- the BEV extent equals the grid divided by `FEATURE_MAP_STRIDE` (1504 / 8 = 188)

It also runs the installed voxelizer on three points to observe its output dtypes.

The JSON output uses the `workload` / `stages` / `boundaries` layout with evidence grades. Unknown dimensions are `null`, and logical bytes are computed only for fully determined shapes.

## Provenance

| Item | Value |
| --- | --- |
| OpenPCDet commit | `233f849829b6ac19afb8af8837a0246890908755` (worktree unmodified) |
| Model config | `tools/cfgs/waymo_models/centerpoint.yaml`, sha256 `db86c835…e87da2` |
| Dataset config | `tools/cfgs/dataset_configs/waymo_dataset.yaml`, sha256 `c68049ce…09eb5b` |
| spconv | 2.3.8 |
| License | OpenPCDet code is Apache-2.0. Waymo data and Waymo-trained weights fall under the Waymo Dataset License Agreement, and no pretrained Waymo checkpoint is distributed. |
| ONNX | None. There is no export path at this commit. |

## Input contract

| Parameter | Value | Source |
| --- | --- | --- |
| Batch | 1 | protocol |
| Point features | x, y, z, intensity, elongation (FP32) | `waymo_dataset.yaml`; `waymo_utils.py` saves `float32` |
| Intensity transform | `tanh` | `waymo_dataset.py get_lidar` |
| Point-cloud range | [-75.2, -75.2, -2, 75.2, 75.2, 4] m | dataset config |
| Voxel size | [0.1, 0.1, 0.15] m | dataset config |
| Grid (x, y, z) | 1504 × 1504 × 40 | derived (`np.round(range / voxel)`) |
| Max points per voxel | 5 | dataset config |
| Max voxels (test) | 150000 | dataset config |
| Point shuffle at test | **enabled** | dataset config: point order is nondeterministic unless seeded |

## Output contract

At most 500 detections, each `[x, y, z, dx, dy, dz, heading]` FP32, with an FP32 score and an INT64 label (1-based: Vehicle, Pedestrian, Cyclist).

Decoding steps:

1. Sigmoid on the heatmap.
2. Top-K with K = 500.
3. Drop candidates scoring 0.1 or below, and boxes centered outside the post-center range.
4. Class-agnostic rotated NMS (`nms_gpu`): IoU 0.7, keep the top 4096 before NMS, at most 500 after.

## Stages and boundaries

The timing boundary is the existing primary boundary: `model(batch_dict)`. Frame load and voxelization run in the dataset path **before** that call, so the current latency numbers exclude them.

| Stage | Inside `model()` | Standard-ONNX expressible | Note |
| --- | :-: | :-: | --- |
| Frame load, range mask, shuffle | no | n/a | dataset code (numpy) |
| Voxelization (`Point2VoxelCPU3d`) | no | no (data-dependent hashing) | host CPU in this implementation |
| MeanVFE | yes | yes | elementwise mean |
| VoxelResBackBone8x | yes | **no** | spconv sparse 3D convolutions; data-dependent rulebook |
| HeightCompression | yes | partly | sparse → dense scatter, then reshape |
| BaseBEVBackbone | yes | yes | dense, static shape |
| CenterHead convolutions | yes | yes | dense, static shape |
| Decode, top-K, mask | yes | partly | fixed-size top-K, then a data-dependent mask |
| Rotated NMS | yes | **no** | OpenPCDet custom CUDA extension; the overlap mask is computed on the GPU, but the greedy suppression runs on the host CPU after a blocking copy ([NMS-SWEEP.md](NMS-SWEEP.md)) |

"Standard-ONNX expressible" is a judgment from the operator types. No export was attempted. It identifies where an exported graph would have to stop, or need custom operators.

Symbols for frame-dependent counts: N points, V voxels, S active sparse sites, M masked candidates, D final detections.

| Boundary | Shape | Type | Logical bytes | Upper bound |
| --- | --- | --- | ---: | ---: |
| points | N × 5 | FP32 | — | — |
| voxels | V × 5 × 5 | FP32 | — | 15,000,000 |
| voxel_num_points | V | INT32 on host → **FP32** in model | — | 600,000 |
| voxel_coords | V × 4 | INT32 on host → **FP32** in model → INT32 in backbone | — | 2,400,000 |
| voxel_features | V × 5 | FP32 | — | 3,000,000 |
| encoded sparse features | S × 128 | FP32 | — | 36,192,256 |
| encoded sparse indices | S × 4 | INT32 | — | 1,131,008 |
| spatial_features | 1 × 256 × 188 × 188 | FP32 | 36,192,256 | — |
| spatial_features_2d | 1 × 512 × 188 × 188 | FP32 | 72,384,512 | — |
| head outputs (hm 3, center 2, center_z 1, dim 3, rot 2) | 1 × 11 × 188 × 188 in total | FP32 | 1,555,136 | — |
| candidate boxes / scores | M × 7 / M | FP32 | — | 14,000 / 2,000 |
| pred boxes / scores / labels | D × 7 / D / D | FP32 / FP32 / INT64 | — | 14,000 / 2,000 / 4,000 |

Upper bounds come from configured caps (V ≤ 150000; S ≤ the full 2 × 188 × 188 output grid; M, D ≤ 500). They are not representative sizes.

Logical bytes are not allocated bytes or transferred bytes. Strides, padding, allocator rounding, and copies are all unknown.

Sparse-backbone spatial extents (z, y, x) by level:

- input: 41 × 1504 × 1504 (16 channels)
- conv2: 21 × 752 × 752 (32 channels)
- conv3: 11 × 376 × 376 (64 channels)
- conv4: 5 × 188 × 188 (128 channels)
- conv_out: 2 × 188 × 188 (128 channels)

The inline shape comments in `spconv_backbone.py` describe a KITTI-sized grid and do not apply to this config.

## Structural observations this contract makes visible

These are derived facts about the reference implementation, not claims about any deployment:

1. **Two of the pipeline's stages fall outside any standard dense-graph export:** the sparse 3D backbone and rotated NMS. Voxelization also runs outside the model call, in host code. Any target that compiles a standard graph must handle these three some other way: custom operators, a dense or pillar substitute, or general-purpose CPU code.
2. **The format change is concentrated at HeightCompression.** Upstream of it, every tensor has a data-dependent leading dimension (N, V, S). Downstream of it, everything is static and dense until the post-top-K mask. This is the natural partition point for a static-shape compiler, and its dense output is 36.2 MB in FP32.
3. **Point shuffling is enabled at test time.** With identical input, voxel order (and therefore sparse-convolution work order) varies between runs unless seeded. This matters for reproducible timing and for bitwise output comparison.

## Known versus missing

| Item | Status |
| --- | --- |
| Source, commit, config hashes, license | known |
| Static dense shapes (BEV, 2D backbone, heads) | derived |
| Voxelizer output dtypes | observed on host (spconv 2.3.8 probe) |
| N, V, S, M, D for a representative frame | **missing**: N, V, S need a licensed Waymo frame (active sparse sites depend on geometry, not weights); M and D also need a trained checkpoint |
| Trained checkpoint | **missing**: must be one the user is entitled to use |
| ONNX graph and shape inference | **missing**: no export path; would cover a subset of stages only |
| Execution domain, memory placement, transfers per boundary | **unknown**: needs a target partition report, runtime trace, or measurement |
| Allocated bytes, strides | **unknown** |

## Integer-to-float round trip at host-to-device

`load_data_to_gpu` (`pcdet/models/__init__.py`) converts every numpy array with `.float()`. The voxelizer's INT32 coordinates and point counts therefore enter the model as FP32, and `VoxelResBackBone8x` casts the coordinates back with `.int()`. The values are exact (grid indices are far below 2^24) and the transfer size is unchanged (both are 4 bytes). The cost is one extra conversion on the host (`.float()` before `.cuda()`) and one conversion back on the device for each frame. This was first missed by the static derivation and then caught by the Phase 1 inventory ([BOUNDARY-INVENTORY.md](BOUNDARY-INVENTORY.md)).

## Next increment

Phase 1 is recorded in [BOUNDARY-INVENTORY.md](BOUNDARY-INVENTORY.md).

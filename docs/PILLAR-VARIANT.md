# Pillar variant: removing the sparse 3D backbone

The pinned OpenPCDet ships `tools/cfgs/waymo_models/centerpoint_pillar_1x.yaml` (sha256 `35ecb682…0979`). It is CenterPoint with the spconv sparse 3D backbone replaced by:

1. **PillarVFE:** a small point network per 0.32 × 0.32 m pillar, at most 20 points each.
2. **PointPillarScatter:** writes pillar features into a dense 468 × 468 × 64 BEV grid.

These are followed by a larger 3-level `BaseBEVBackbone` and `CenterHead` at **stride 1** (468 × 468 maps). A dynamic-voxelization twin, `centerpoint_dyn_pillar_1x.yaml`, differs only by moving voxelization into the model (`DynPillarVFE`); it was not measured.

This note compares the pillar variant with the current voxel model (`centerpoint.yaml`). It uses the same reference conditions as the other studies: RTX 2060, i9-14900KF, random weights, synthetic scans. **It is not a performance baseline.**

## Reproduce

```bash
.conda-env/bin/python tools/pillar_variant.py     # ~5 min
# -> results/onnx/pillar_variant.json
# -> artifacts/onnx/centerpoint_pillar_static.onnx (+ .raw.onnx)
```

## Comparison

| | Voxel model (`centerpoint.yaml`) | Pillar model (`centerpoint_pillar_1x.yaml`) |
| --- | --- | --- |
| Stage with no standard-ONNX form inside the model | spconv sparse 3D backbone | none (rotated NMS remains, in both) |
| Static ONNX coverage | dense segment only, from `spatial_features` | **voxelizer output → raw head outputs** |
| BEV / head resolution | 188 × 188 (stride 8) | 468 × 468 (stride 1) |
| Largest dense activation (FP32) | 72.4 MB (`[1,512,188,188]`) | 336 MB (`[1,384,468,468]`) |
| Dense conv work per frame | 85.8 GMAC | 223.9 GMAC (2.6×) |
| Point-network work | — | 13.3 GMAC at P_MAX = 150,000; about 0.3–0.9 GMAC at the scans' actual 3.7k–10k pillars |
| Model call, RTX 2060 (synthetic scans) | 44–54 ms, varies with input | **89.8–92.3 ms**, nearly constant |
| Peak GPU allocation in the model call | 256–328 MB | 832–837 MB |
| Static ONNX on onnxruntime CPU | 210 ms (dense segment only) | 1,187 ms (whole network at P_MAX) |
| Waymo L2 mAPH, Veh / Ped / Cyc (documented upstream, 20% training data) | 64.42 / 60.34 / 67.28 | 61.69 / 55.00 / 61.46 |

The voxel figures come from [DENSITY-SWEEP.md](DENSITY-SWEEP.md) and [ONNX-DENSE.md](ONNX-DENSE.md). The pillar figures come from 18 parametric scans: clutter 0 / 1 / 4 × azimuth 1325 / 2650 / 5300 × buildings 0 / 10, seed 1. The accuracy row is quoted from the OpenPCDet README at the pinned commit (Waymo val v1.2) and has not been reproduced here.

Pillar stage times across the 18 scans (median of 10, GPU):

| Stage | Time |
| --- | --- |
| PillarVFE | 1.5–3.3 ms |
| PointPillarScatter | 1.0–1.3 ms |
| BaseBEVBackbone | 51.6–52.7 ms |
| CenterHead (convolutions + decode/NMS) | 34.7–35.7 ms |
| Host voxelization | 2.3–9.3 ms |

## The static whole-network export

`StaticPillarNet` in `tools/pillar_variant.py` makes the graph exportable with fixed shapes. It makes four changes, and the wrapped network's output is **identical (difference 0)** to OpenPCDet's own modules on the GPU:

1. **Pillars are padded to a fixed P_MAX** (default: the configured cap of 150,000). Padding pillars have `voxel_num_points = 0`.
2. **The scatter is rewritten for batch 1** with no host `.item()` and no boolean-mask indexing. Each padding pillar writes to its own dummy cell beyond the grid, which is sliced off. This gives one ONNX `ScatterND` with no duplicate indices.
3. **PillarVFE's feature construction is written out of place.** The original `torch.zeros_like` plus in-place column writes exports as a baked-in `[150000, 20, 3]` constant and `ScatterND` chains that shape inference cannot resolve. OpenPCDet's own PFN layers are then called unchanged.
4. **PFNLayer's 50k-row chunking of its Linear layer is turned off.** The chunking is a memory workaround, and traced at P_MAX it would bake in the chunk slices. One chunk gives the same result.

Result:

| Property | Value |
| --- | --- |
| Opset | 17 |
| Parameters | 5,221,835 (22.7 MB file; the only stored constants are two 150k-entry index vectors, 1.8 MB) |
| Static shapes | every tensor fully static after constant folding (the raw export has 68 non-static tensors) |
| `ScatterND` nodes | exactly one (the pillar scatter) |
| Bulk operators | 27 Conv, 3 ConvTranspose, 27 Relu, 5 BatchNormalization, 2 MatMul, 2 ReduceMax, 3 Pad |
| onnxruntime CPU vs OpenPCDet | within 1.6 × 10⁻⁶ of output peak |
| Determinism | two exports produced identical files |

| File | sha256 |
| --- | --- |
| `centerpoint_pillar_static.onnx` | `3211b72e7c2d61a842aaed4eae16a952071d7fc110a57ce7da77a792775008ea` |
| `centerpoint_pillar_static.raw.onnx` | `146d2d8913e6b059dda3aafdf36b8fdc52cb7be701532b51a02c90b36b1fc656` |

Two tooling notes:

- **onnxruntime 1.20.1 fails to load the model as exported.** Its `MatmulBNFusion` rewrite rule turns PillarVFE's rank-3 MatMul + BatchNormalization into a `Gemm`, which requires rank-2 input. The model is valid ONNX and passes the checker. The CPU reference run disables ORT's two rule-based transformers (`Level1/2_RuleBasedTransformer`), because that rule cannot be disabled on its own; all other ORT optimizations stay on. A target compiler attempting the same fusion would hit the same issue.
- **The scatter canvas must not become a stored constant.** A `torch.zeros` with a fixed shape is baked by the tracer into a 94 MB `Constant` (`[cells + P_MAX, 64]`), even with the exporter's folding off. Turning folding off also stops BatchNorm being fused into the convolutions. The wrapper therefore derives the canvas from a runtime tensor (`zeros_like` of one feature row, then `Expand`), which took the file from 117 MB to 22.7 MB.
- **The shared constant folder no longer materializes generated tensors.** `fold_constants` in `tools/export_dense_onnx.py` now leaves in place nodes whose output is much larger than their inputs, such as `ConstantOfShape` or `Expand`. Folding them had stored a 94 MB zero canvas as a weight. The dense-segment export is unchanged (same sha256).

## Findings

1. **The pillar variant removes the stage with no standard-ONNX form, at the cost of 2.6× the dense work.** Everything between the host voxelizer and the raw head outputs becomes one static ONNX graph with one scatter. The price is 223.9 GMAC of dense convolution against 85.8 GMAC, a 336 MB FP32 BEV feature against 72 MB, and about 1.8× the reference GPU latency (90–92 ms against 44–54 ms).
2. **Most of the extra work is the full-resolution head.** `FEATURE_MAP_STRIDE: 1` runs the shared head conv (384 → 64 channels, 48.4 GMAC, the single largest layer) and the five separate heads (41.8 GMAC) at 468 × 468. Together that is 40% of the conv MACs. The BEV blocks are 121.1 GMAC and the deconvolutions 12.6 GMAC.
3. **Pillar cost is almost independent of the input.** Latency varied by 2.5 ms across scans with 3.7k–10k pillars and 77k–318k points. The voxel model's cost moves with active sparse sites. A fixed, data-independent cost suits a static accelerator schedule.
4. **Choosing P_MAX is a real deployment trade-off.** At the configured cap of 150,000, every PillarVFE intermediate is `[150000, 20, 64]` FP32 = 768 MB, and the point network costs 13.3 GMAC. The scans used only 3.7k–10k pillars (0.3–0.9 GMAC of real work). A static compile must pick P_MAX from the real pillar-count distribution, and synthetic scans cannot supply that.
5. **The 20-point pillar cap discards a lot of points.** It drops 27–81% of in-range points in these scans; the voxel model's 5-point cap dropped 0–51% across the 90 density-sweep scans. Whether that loss matters is an accuracy question.
6. **The upstream accuracy cost is 2.7–5.8 mAPH (L2).** The documented numbers put the pillar model 2.7 / 5.3 / 5.8 points below the ResNet voxel config for vehicle / pedestrian / cyclist, both trained on 20% of the data.

## Limitations

- Pillar counts and point drops come from synthetic scenes, which concentrate returns on nearby walls and omit short-range lidars and second returns. Real frames will occupy more pillars.
- The rewritten scatter and VFE are verified for batch 1 and this config (`USE_ABSLOTE_XYZ: True`, `WITH_DISTANCE: False`); the wrapper asserts both.
- Accuracy figures are upstream claims. With random weights, nothing here measures detection quality.
- Host voxelization and decode/NMS remain outside the export, as for the voxel model.

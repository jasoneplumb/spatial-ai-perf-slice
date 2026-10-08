# ONNX export of the dense segment

The part of CenterPoint with a fully static contract, exported to ONNX as a portable artifact that a graph compiler can consume:

`spatial_features [1, 256, 188, 188]` → BaseBEVBackbone → CenterHead shared conv + separate heads → raw head outputs `hm, center, center_z, dim, rot`, each `[1, C, 188, 188]`.

On surface-structured input this segment accounts for 57–71% of the model call ([DENSITY-SWEEP.md](DENSITY-SWEEP.md)). The stages before and after it are outside the export because standard ONNX cannot express them:

- voxelization (host)
- the spconv sparse backbone
- the sparse-to-dense scatter
- decode, top-K, the score mask, and rotated NMS (which partly runs on the host; see [NMS-SWEEP.md](NMS-SWEEP.md))

**The weights are random.** Graph structure, shapes, operator set and MAC counts are valid. The outputs are not meaningful detections, and the model must not be used for INT8 calibration, accuracy or quantization-sensitivity work. That needs a trained checkpoint (see the licensing constraint in [WORKLOAD.md](WORKLOAD.md)).

## Reproduce

```bash
bash scripts/setup-onnx.sh                      # onnx==1.17.0, onnxruntime==1.20.1 into .conda-env
.conda-env/bin/python tools/export_dense_onnx.py
# -> artifacts/onnx/centerpoint_dense.raw.onnx   (as exported)
# -> artifacts/onnx/centerpoint_dense.onnx       (constant-folded, fully static)
# -> results/onnx/dense_export.json              (inventory, MACs, numerics, timing)
```

The export is deterministic: two runs produced identical files.

| File | sha256 |
| --- | --- |
| `centerpoint_dense.onnx` | `b320aab304aa737dbd75575e13b855f97804b9c118679e55b2c7406177e751e5` |
| `centerpoint_dense.raw.onnx` | `5d3d8effa2aa36132857f6e82e9597a051ec5c7641de5e3ea52025438aac779a` |

Export settings: `torch.onnx.export(dynamo=False)` with PyTorch 2.7.0, opset 17, static input shape, `torch.manual_seed(7)` weights. The folded model is 20.3 MB with 5,061,469 parameters.

## Validation

| Check | Result |
| --- | --- |
| `onnx.checker` (full check) | pass |
| Strict shape inference with data propagation | all 45 intermediate tensors and all 5 outputs fully static |
| Input/output shapes against [the contract](WORKLOAD-CONTRACT.md) | match |
| Folded model against raw export (onnxruntime) | bitwise-identical outputs |
| onnxruntime CPU against PyTorch CPU | max abs error ≤ 4.3 × 10⁻⁶ of each output's peak |
| onnxruntime CPU against PyTorch CUDA | max abs error ≤ 8.5 × 10⁻⁷ of each output's peak |

The numeric comparison uses a realistic input: `spatial_features` produced by the real sparse path from the parametric scan `results/scans/scan_seed7.npy`.

## Findings

1. **As exported, the graph is not static-shape.** `nn.ZeroPad2d(1)` (the first layer of each BEV block) is emitted as `Pad` whose `pads` input is computed at runtime by a constant chain (`ConstantOfShape`, `Concat`, `Reshape`, `Slice`, `Transpose`, `Cast`). Shape inference can't see through it, so 50 tensors, all outputs included, get symbolic dimensions. A compiler that requires static shapes may reject the raw file or fall back on these nodes. `fold_constants` in the export tool evaluates the 53 constant-only nodes with onnx's reference evaluator; after that, every shape resolves.
2. **The folded graph uses only common dense operators:**

   | Op | Count |
   | --- | ---: |
   | Conv | 23 |
   | Relu | 20 |
   | Pad | 2 (constant pads, each feeding a padding-0 Conv) |
   | ConvTranspose | 2 (1×1 stride 1, and 2×2 stride 2) |
   | BatchNormalization | 2 (after ConvTranspose; not fused by the exporter) |
   | Concat | 1 |

   There is no data-dependent control flow and nothing custom. Whether a given compiler fuses the explicit Pad and the post-ConvTranspose BatchNorm is a question to confirm with that compiler.
3. **The segment computes 85.78 GMAC per frame.** This was computed from the static shapes and checked against a per-layer hand calculation:

   | Part | GMAC |
   | --- | ---: |
   | BEV level 0 (188², 6 convs) | 36.5 |
   | BEV level 1 (94², 6 convs) | 28.7 |
   | Deconvolutions | 3.5 |
   | Shared head conv (512→64 channels) | 10.4 |
   | Five separate heads | 6.7 |

   The largest single layers are the BEV level-0 entry conv (256→128) and the shared head conv (512→64), at 10.4 GMAC each.
4. **Activation memory is large and static.** The input is 36.2 MB in FP32 and the concatenated BEV feature is 72.4 MB. All 45 intermediate tensors and their logical sizes are listed in `dense_export.json`; that is the basis for any on-chip-memory or tiling question.
5. **Run on a CPU, this segment is about 7× slower than on the reference GPU.** onnxruntime 1.20.1, CPU execution provider, default threading, on an Intel Core i9-14900KF (32 logical CPUs): median 209.6 ms (min 195.9 ms) over 10 runs, about 410 GMAC/s. The same segment takes about 28–30 ms on the RTX 2060: the density sweep's dense-part time of 30–32 ms, less the ~2 ms of decode and NMS it includes. This shows what falling back on a desktop-class CPU would cost; it says nothing about any embedded CPU or accelerator.

## Next steps

1. Give `centerpoint_dense.onnx` to a target graph compiler and record which operators it supports, how it partitions the graph, and which operators fall back to the CPU. That is the first compiler-observed evidence for the execution-domain question.
2. To quantize, replace the random weights with a trained checkpoint and calibrate on representative `spatial_features`. Neither can be produced from the current license-free inputs.

# v0.2.0

First public release: a measurement-first study of where one frame of LiDAR 3D detection (CenterPoint, Waymo configuration, OpenPCDet reference implementation) spends its time, done entirely without licensed data.

**This is structural profiling, not a performance benchmark.** Inputs are synthetic, weights are random, and every timing comes from one RTX 2060 and one desktop x86-64 CPU. No Waymo data or pretrained checkpoint is included or needed.

## Methodology

- One timing boundary: `model(batch_dict)` for an already-prepared frame, timed with CUDA events after warmup.
- Layered views that have to agree: whole model, model stage, sparse-backbone level, individual layer, operator trace.
- A static tensor contract derived from the pinned config and source, then checked against observed tensors (19 of 19 boundaries match).
- Cost models fitted on one family of inputs and tested on held-out inputs.
- Every result file from synthetic input or random weights is marked `"valid_for_baseline": false`.

## What is in it

- `tools/scan_generator.py`: deterministic, numpy-only parametric lidar scans (64-beam ring, optional short-range lidars, ego body, edge second returns).
- Boundary contract and inventory, a 90-scan density sweep, a decode/NMS sweep, and a CPU run of the sparse backbone.
- Deterministic static ONNX exports of the dense segment and of the whole pillar variant, with shape inventory, MAC counts and numeric checks.
- Pinned WSL2 environment scripts (PyTorch 2.7.0, CUDA 12.6, spconv 2.3.8, OpenPCDet `233f849`).
- GPU-free release checks (`python -m unittest discover -s tests -v`) and CI.

## Findings

- On surface-structured scans the model call takes 44–54 ms. The input-independent dense 2D part is 57–71% of it; the sparse backbone is 29–43%, not the 80% that the 0.1 uniform-point input suggested.
- Sparse-backbone cost follows channel-weighted active sites per level, on GPU and CPU. Voxel counts miss held-out inputs by 2.5–6.5×.
- Three stages fall outside a standard dense graph: host voxelization, the spconv sparse backbone, and rotated NMS, whose greedy loop runs on the host after a blocking copy.
- A stock CPU fallback for the sparse backbone takes 171–389 ms at 16 threads, 12–17× its GPU time.
- The dense segment exports as a fully static ONNX graph (85.78 GMAC) once runtime-computed pads are folded.
- The pillar variant removes the sparse stage and exports whole, at 2.6× the dense work and about 90 ms.

Details and conditions are in [MILESTONE-0.2.md](MILESTONE-0.2.md) and the per-study write-ups.

## Limitations

- No real point cloud has been run. The synthetic scans lack vegetation and irregular clutter and are probably optimistic.
- Detection counts, NMS load under real predictions, accuracy and quantization are untested; the ONNX graphs must not be used for calibration.
- The numbers were measured in October 2026 on the reference machine. The release checks do not re-run the GPU tools.
- The setup path is Windows 11 + WSL2 with an NVIDIA GPU. Transitive Python dependencies are not pinned.

## License

Apache-2.0. See `LICENSE`, `NOTICE` and `THIRD_PARTY_NOTICES.md`. OpenPCDet and all other dependencies are downloaded at setup time and are not redistributed.

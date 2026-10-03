# Milestone 0.1 — Working CenterPoint structural vertical slice

**Date:** 2026-10-03  
**Milestone name:** `v0.1.0-structural`  
**Reference implementation commit before milestone documentation:** `a6ee4391613ff8cb35cbec6db28e890d71b01447`

## What exists at this point

A complete structural path for single-frame CenterPoint inference now runs locally:

```text
deterministic or external LiDAR points
        ↓
OpenPCDet-compatible voxelization
        ↓
MeanVFE
        ↓
VoxelResBackBone8x
        ↓
HeightCompression
        ↓
BaseBEVBackbone
        ↓
CenterHead + decode + rotated NMS
        ↓
predictions
```

The environment and OpenPCDet revision are pinned, the native CUDA extensions build, and the workload can be profiled at progressively finer levels without modifying the reference OpenPCDet source.

The current tooling provides:

- whole-model CUDA-event timing;
- top-level model-stage timing;
- sparse-backbone `conv_input` / `conv1` ... `conv_out` timing;
- detailed `conv3` and `conv4` residual-convolution timing;
- sparse active-site and channel counts;
- PyTorch operator summaries;
- Chrome/Perfetto traces;
- peak GPU-memory reporting;
- replay support for synthetic points and external real point-cloud `.npy` files;
- a separate frozen-frame + trained-checkpoint path for the later reference baseline.

## What was learned

The structural run on the RTX 2060 takes about 163 ms for the chosen synthetic occupancy pattern. Roughly 80% of that time is inside `VoxelResBackBone8x`.

The sparse-backbone timing narrows the dominant work further:

```text
conv2          ~17.6 ms
conv3          ~50.2 ms
conv4          ~58.9 ms
```

Within `conv3` and `conv4`, most time is spent in the repeated SubMConv3d residual convolutions, not in BN/ReLU/residual-add work. The operator profile independently shows the largest CUDA consumers as CUMM sparse-convolution kernels.

That agreement across three levels of measurement is the important result:

```text
whole model
    ↓
sparse 3D backbone
    ↓
conv3 / conv4
    ↓
wide residual sparse convolutions
```

## Why this is a useful reference point

Before this milestone, the main uncertainty was whether the intended workload could be made concrete and measurable on available hardware without access to the eventual target system.

That uncertainty is now resolved.

The project has demonstrated that it is possible to:

- reproduce a real inference graph on ordinary development hardware;
- define an explicit timing boundary;
- retain a stable reference implementation;
- localize latency from whole-model behavior down to individual sparse layers;
- separate workload understanding from target-specific optimization;
- distinguish structural smoke measurements from measurements that can support performance claims.

This is therefore a good checkpoint before introducing representative data, trained weights, or target-specific code. If later work becomes noisy or changes direction, this commit state preserves a small, understandable execution path that can be returned to.

## What this milestone does not claim

This is deliberately **not** the reference performance baseline.

The current input is synthetic and has an unusually dispersed occupancy pattern: 80,000 generated points produced 79,295 occupied input voxels. The model weights are random. Consequently:

- the ~163 ms latency is not a Waymo benchmark result;
- the current detection count is meaningless;
- the relative cost of sparse stages may change with real occupancy;
- CenterHead/decode/NMS cost may change with trained predictions.

These constraints are part of the experiment, not caveats to hide.

## Next experiment

The immediate next experiment requires only one representative real Waymo LiDAR point cloud. Trained weights are not required to test the sparse-coordinate behavior because occupancy propagation is determined by input coordinates and convolution geometry.

The comparison should answer:

```text
synthetic occupancy             real Waymo occupancy
-------------------             --------------------
input point count        vs.    input point count
input voxel count        vs.    input voxel count
conv2 active sites       vs.    conv2 active sites
conv3 active sites       vs.    conv3 active sites
conv4 active sites       vs.    conv4 active sites
conv3 latency            vs.    conv3 latency
conv4 latency            vs.    conv4 latency
whole-model latency      vs.    whole-model latency
```

If the bottleneck survives realistic occupancy, it becomes a credible optimization target. If it does not, the experiment has prevented optimizing the wrong thing.

The trained-checkpoint baseline should follow only after that occupancy experiment.

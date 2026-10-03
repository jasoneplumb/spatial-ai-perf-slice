# Workload contract — CenterPoint single-frame LiDAR inference

## Question

Where does the latency of one representative LiDAR frame go, and which change measurably reduces that latency without changing the accepted output?

## Reference implementation

- Framework: OpenPCDet
- Model config: `tools/cfgs/waymo_models/centerpoint.yaml`
- Model: CenterPoint with MeanVFE, VoxelResBackBone8x, HeightCompression, BaseBEVBackbone, and CenterHead
- Dataset: Waymo Open Dataset
- Classes: Vehicle, Pedestrian, Cyclist
- Batch size: 1
- Temporal context: one frame

The exact OpenPCDet commit must be recorded in every result.

## Frozen input

The benchmark input is a serialized **post-dataset/pre-model batch dictionary** produced by `tools/freeze_frame.py`.

This choice intentionally removes:
- TFRecord/file-system latency
- dataset parsing
- host-side sample selection
- augmentation

It retains the tensors/arrays that the unmodified OpenPCDet model consumes. The frame file SHA-256 is recorded with every benchmark.

For a more portable target later, add a second canonical representation containing the raw LiDAR point tensor plus the minimal metadata required to reproduce preprocessing.

## Primary timing boundary

Start immediately before `model(batch_dict)`; stop after GPU completion.

This includes work performed by the model path, including preprocessing implemented inside the model, network execution, prediction decoding, and NMS. It excludes checkpoint loading, dataset construction, file I/O, and one-time CUDA initialization.

## Protocol

- inference mode / no gradients
- batch size 1
- 20 warmup iterations by default
- 100 measured iterations by default
- CUDA events for device-synchronized elapsed time
- report min, median, mean, p90, p95, p99, max, and standard deviation
- run PyTorch Profiler separately on the identical frozen input
- retain the first prediction as the correctness reference

Do not compare optimization results unless model config, checkpoint, frozen-frame SHA-256, software revision, and measurement boundary match.

## Correctness contract

Phase 1 stores the reference prediction. Before accepting an optimization, compare:
- number of detected boxes
- class labels
- confidence scores
- box parameters

Exact tolerances should be chosen before the first non-reference implementation is measured. Do not silently weaken tolerances to accept an optimization.

## Stage taxonomy

Use these categories when interpreting the trace:

1. voxel/input preparation
2. sparse 3D backbone
3. height compression / BEV mapping
4. dense 2D BEV backbone
5. CenterHead
6. decode and rotated NMS
7. synchronization/data movement
8. framework/runtime overhead

The initial profiler observes these through operators. A subsequent instrumentation pass should add explicit stage ranges around the OpenPCDet modules without changing computation.

## Baseline record

Each run records:
- UTC timestamp
- host/platform
- Python version
- PyTorch version
- CUDA version
- GPU name
- OpenPCDet commit
- config path
- checkpoint path
- frozen-frame SHA-256
- warmup and iteration counts
- timing distribution

## Exit criteria for the baseline phase

The baseline is complete when:
- the frozen frame can be replayed repeatedly
- a reference prediction is saved
- timing JSON is produced from >=100 iterations
- a profiler operator table and trace are produced
- rerunning the benchmark yields a stable enough distribution to support an optimization comparison

Only then choose the first optimization.

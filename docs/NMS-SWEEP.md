# Decode and NMS sweep

Cost of CenterHead post-processing (heatmap decode, top-K, score/range mask, rotated NMS) as a function of candidate count. It is measured on the reference implementation (RTX 2060, PyTorch 2.7.0, CUDA 12.6, x86-64 host). **It is not a performance baseline.**

Random weights give a fixed detection count no matter the input (313 in every density-sweep run). So this sweep drives the post-processing directly, with constructed inputs that produce a chosen number of candidates M.

## Reproduce

```bash
.conda-env/bin/python tools/nms_sweep.py     # ~1 min; -> results/sweep/nms.json
```

There are two parts:

- **A, `nms_gpu` alone:** M synthetic car-sized boxes, either *spread* (uniform over the range, almost no overlap) or *clustered* (groups of 8 near-duplicates within ±0.2 m, as around a heatmap peak). Sizes go up to the configured limit and beyond it (1000, 2000, 4096).
- **B, the whole `CenterHead.generate_predicted_boxes`:** constructed head outputs in which exactly M heatmap cells exceed the score threshold. Their regression channels decode to spread boxes, or to 3×3 cell blocks that decode to the same position.

Each point is timed by wall clock, synchronized, as the median of 50 runs after 10 warmup runs. One profiled run per point gives GPU-busy time (all recorded CUDA activity) and the CUDA runtime calls made by the host. "Host remainder" is wall time minus GPU-busy time: launches, runtime calls, synchronization waits and host computation. These serialize here, so the remainder approximates host-side time, but it is not a direct measurement of it.

## Structure of the pinned NMS

`iou3d_nms_utils.nms_gpu` → `src/iou3d_nms.cpp nms_gpu`:

1. Sort the scores on the GPU and gather boxes in score order.
2. `cudaMalloc` an M × ⌈M/64⌉ 64-bit overlap mask, then launch the BEV-IoU mask kernel.
3. **Blocking `cudaMemcpy` of the mask to the host** (32,000 bytes at M = 500), then `cudaFree`.
4. **Greedy suppression loop on the host CPU**, writing kept indices into a CPU tensor.
5. Copy the kept indices back to the GPU (`keep[:num_out].cuda()`).

So the "GPU" NMS is a GPU-to-CPU-to-GPU round trip inside `model()` on every frame. It is one of the stages that would remain on the application CPU, or need porting, on another target.

Also, `MAX_OBJ_PER_SAMPLE = 500` top-K runs **before** NMS, so M ≤ 500 always, and the configured `NMS_PRE_MAXSIZE = 4096` never takes effect in this configuration.

## Results

| M | Layout | `nms_gpu` wall | GPU-busy | Host remainder | Kept |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | spread | 0.49 ms | 0.02 ms | 0.47 ms (96%) | 1 |
| 50 | spread | 0.93 ms | 0.41 ms | 0.52 ms (56%) | 50 |
| 500 | spread | 0.93 ms | 0.43 ms | 0.50 ms (54%) | 500 |
| 500 | clustered | 1.15 ms | 0.63 ms | 0.52 ms (45%) | 151 |
| 1000 † | spread | 1.08 ms | 0.48 ms | 0.60 ms | 998 |
| 2000 † | spread | 2.39 ms | 1.75 ms | 0.64 ms | 1994 |
| 4096 † | spread | 7.61 ms | 7.37 ms | 0.24 ms | 4078 |
| 4096 † | clustered | 8.48 ms | 7.70 ms | 0.78 ms | 1316 |

† Above the configured top-K; unreachable with this config. Shown to characterize the kernel.

| M | Layout | Decode + NMS wall | GPU-busy | Host remainder | Detections |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | — | 0.79–0.81 ms | 0.22 ms | 0.58 ms (72%) | 0 |
| 1 | spread | 1.45 ms | 0.27 ms | 1.18 ms (81%) | 1 |
| 100 | spread | 1.81 ms | 0.66 ms | 1.15 ms (63%) | 100 |
| 500 | spread | 1.85 ms | 0.70 ms | 1.16 ms (62%) | 500 |
| 500 | clustered | 1.97 ms | 0.82 ms | 1.16 ms (59%) | 124 |

CUDA runtime calls per decode + NMS call are the same for every M ≥ 1: 11 `cudaMemcpyAsync`, 5 `cudaStreamSynchronize`, 1 blocking `cudaMemcpy`, and 1 `cudaMalloc` / `cudaFree` pair. With M = 0, NMS is skipped, leaving 4 `cudaMemcpyAsync` and 4 `cudaStreamSynchronize`. Two `cudaDeviceSynchronize` calls in every row come from the measurement itself and are excluded.

## Findings

1. **Within the configured range, post-processing costs about 2 ms and barely depends on M.** Decode + NMS goes from 0.8 ms (no candidates) to 1.5 ms (one candidate) to 1.85–2.0 ms (500 candidates). That is about 4% of the 44–54 ms model call measured in the density sweep. The step from 0 to 1 candidate (+0.65 ms) is the fixed cost of entering NMS at all. Going from 1 to 500 candidates adds only about 0.4–0.5 ms.
2. **Post-processing is limited by latency and synchronization, not compute.** In 17 of the 18 profiled runs with M ≥ 1, the host-side remainder is a constant 1.13–1.21 ms, which is 58–82% of decode + NMS wall time. The exception is one clustered run at M = 400, whose single profile recorded unusually high GPU time. Each frame makes five stream synchronizations, a blocking device-to-host copy, and a device allocation and free. The host greedy loop over at most 500 boxes × 8 mask words is a negligible amount of arithmetic. On a slower application CPU, it is this fixed host and runtime overhead that would grow, not the work proportional to M.
3. **Overlap matters little.** Clustered candidates (70–75% suppressed) cost 0.1–0.25 ms more than spread ones at the same M, from extra GPU time in the overlap-mask stage.
4. **The NMS kernel only becomes compute-bound well above this configuration.** Up to about 1000 candidates its GPU time is nearly flat (0.4–0.5 ms, dominated by fixed costs). Above that it grows steeply, by 4.2× from 2000 to 4096, consistent with the M² overlap mask: 1.75 ms at 2000 and 7.4 ms at 4096. A configuration that raised top-K toward the 4096 pre-NMS cap would make NMS GPU-bound at about 8 ms.
5. **The smoke test's `post_processing` stage is not NMS.** Its 0.02 ms is `model.post_processing` (recall bookkeeping). Decode and NMS run inside the CenterHead stage time.

## Limitations

- The host remainder mixes launch overhead, runtime calls, synchronization waits and the greedy loop. Separating them would need native profiling (Nsight Systems) or instrumenting the extension.
- Synthetic candidates have no realistic score distribution or class mix. The candidate count M, the overlap and the call structure are what this sweep controls.
- All host figures are for a desktop x86-64 CPU and CUDA driver.

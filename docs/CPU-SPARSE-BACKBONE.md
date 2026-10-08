# Sparse 3D backbone on a CPU

What VoxelResBackBone8x costs when it runs on a general-purpose CPU instead of a GPU. It is the one model stage with no standard-ONNX form, so where it can run is the largest open question about CPU dependence ([DENSITY-SWEEP.md](DENSITY-SWEEP.md), [ONNX-DENSE.md](ONNX-DENSE.md)).

**Conditions:**

- Intel Core i9-14900KF, WSL2, PyTorch 2.7.0, spconv 2.3.8
- random weights; synthetic input
- median of 3 runs after 1 warmup

This is a desktop x86-64 CPU running a stock library path. The result is a ranking and scaling signal, **not a prediction for any embedded CPU**.

## Reproduce

```bash
.conda-env/bin/python tools/cpu_sparse_backbone.py     # ~10 min
# -> results/sweep/cpu_sparse.jsonl and cpu_sparse.summary.json
```

The inputs are 9 parametric scans (clutter 0 / 1 / 4 × azimuth 1325 / 2650 / 5300, buildings on, seed 1) plus the legacy 80k-point input. Each runs at 1, 4 and 16 PyTorch threads.

### How the CPU path runs

- spconv's default implicit-GEMM algorithm is CUDA-only, so every sparse convolution is switched to `ConvAlgo.Native`. That is rulebook (index-pair) construction followed by gather → GEMM → scatter per kernel offset.
- spconv's CPU path does not support a fused bias, so 16 of the 21 sparse convs have their bias applied by a forward hook.
- Every input's CPU output is checked against the GPU backbone. The maximum difference is **≤ 1.2 × 10⁻⁶** of the output's peak, with identical active sites.
- Time is split by wrapping spconv's `get_indice_pairs` (rulebook) and `indice_conv` (gather–GEMM–scatter). The remainder is BatchNorm, ReLU, residual adds and framework overhead.

## Results

| Input | Voxels | MAC proxy | CPU, 1 thread | CPU, 16 threads | GPU (density sweep) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scans, range of 9 | 32k–86k | 24–46 G | 282–666 ms | 171–389 ms | 13–24 ms |
| Seed-1, clutter 1, azimuth 2650 | 53.6k | 29.6 G | 383 ms | 238 ms | — |
| Legacy 80k (held out) | 79.3k | 354 G | 6,079 ms | 3,185 ms | 135 ms |

Split for the 9 scans:

| | 1 thread | 4 threads | 16 threads |
| --- | ---: | ---: | ---: |
| Rulebook construction | 50–129 ms (16–20%) | 49–129 ms (26–31%) | 52–127 ms (28–33%) |
| Gather–GEMM–scatter | 221–507 ms | 122–290 ms | 114–253 ms |
| Whole backbone, speedup vs 1 thread | — | 1.49–1.66× | 1.51–1.74× |
| Throughput, MAC proxy | 69–86 GMAC/s | 107–137 GMAC/s | 110–142 GMAC/s |

Cost-model fits on the 9 scans (time = slope × MAC proxy + intercept), tested on the held-out legacy input:

| Threads | ms per GMAC-proxy | R² | Legacy predicted / observed |
| ---: | ---: | ---: | --- |
| 1 | 16.0 | 0.97 | 5,563 / 6,079 ms (−8.5%) |
| 4 | 9.9 | 0.93 | 3,448 / 3,494 ms (−1%) |
| 16 | 8.4 | 0.93 | 2,958 / 3,185 ms (−7%) |

## Findings

1. **A stock CPU fallback for this stage would dominate the frame.** On realistic-shaped scans the backbone takes 171–389 ms even on 16 desktop threads. That is 12–17× its GPU time on the same inputs (13–24 ms), and on its own 3.8–7.1× the entire GPU model call for those inputs. For comparison, the dense segment on the same CPU takes 210 ms in onnxruntime.
2. **Rulebook construction does not use more threads.** It is 50–129 ms at every thread count (speedup 0.93–1.04). Its share therefore grows from 16–20% at 1 thread to 28–33% at 16. This is irregular, hashing-style index work, and it caps what more cores can do. It is also the kind of kernel where single-core performance, memory latency and any vector gather/scatter or hashing support would matter.
3. **The convolution itself also scales poorly.** Gather–GEMM–scatter speeds up only 1.7–2.0× from 1 to 16 threads, with most of the gain by 4 threads. Native spconv runs one small GEMM per kernel offset (up to 27 per layer), bracketed by gathers and scatters, so it is likely memory-bound rather than compute-bound. That is a hypothesis; no hardware counters were read.
4. **The channel-weighted MAC proxy predicts CPU time too.** R² is 0.93–0.97 on the scans, and the error on the held-out legacy input, with 8–15× more proxy work, is −8.5% to −1%. The same per-level site counts that predict GPU time predict CPU time. The CPU-to-GPU cost ratio per proxy GMAC is about 22× at 16 threads and 42× at 1 thread (against 0.38 ms/GMAC on the RTX 2060).
5. **conv2 to conv4 carry the cost.** These are the strided level transitions and their residual blocks, 32 to 128 channels. They take 82–90% of single-thread backbone time on the scans; conv_input, conv1 and conv_out together take the rest.

## Limitations

- This is spconv's reference CPU path, not a tuned CPU implementation. A fused, cache-aware gather–GEMM–scatter or a better-parallelized rulebook could be substantially faster. These numbers show what a stock library fallback costs, not what a CPU can do.
- The CPU and GPU runs use different spconv algorithms (Native vs implicit GEMM). Outputs match, but the arithmetic order differs.
- Thread count is set with `torch.set_num_threads`. Whether spconv's CPU kernels follow that setting was not examined; the flat rulebook time suggests that part is single-threaded.
- Desktop x86-64 with large caches and AVX2. An embedded core will differ in clock, cache, memory bandwidth and SIMD width, by amounts that cannot be inferred from this.

## Next steps

1. Read hardware counters (`perf stat`: instructions, cache misses, memory bandwidth) for the rulebook and gather–GEMM–scatter phases, to test the memory-bound hypothesis.
2. Run the same tool on an AArch64 CPU for the first scaling data on Arm.
3. Check whether a pillar-based CenterPoint variant (dense backbone only) is available in the pinned OpenPCDet. It would remove this stage entirely.

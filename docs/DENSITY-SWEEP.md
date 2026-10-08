# Scene-density sweep

How input structure drives per-stage cost. This sweep runs on parametric scans from `tools/scan_generator.py`, with random weights, on the reference PyTorch/CUDA implementation (RTX 2060, PyTorch 2.7.0, CUDA 12.6, x86-64 host). **It is not a performance baseline**, and none of these numbers transfer to another target. What does transfer is *which quantities drive cost* and how they scale.

## Reproduce

```bash
.conda-env/bin/python tools/density_sweep.py      # ~10 min on the reference host
# -> results/sweep/density.jsonl (one row per configuration)
# -> results/sweep/density.summary.json (fits)
```

The grid has 90 scan configurations:

- clutter ∈ {0, 0.5, 1, 2, 4}: a multiplier on the default numbers of vehicles, pedestrians, cyclists and poles
- buildings per side ∈ {0, 10}
- azimuth samples ∈ {1325, 2650, 5300}
- seeds {1, 2, 3}

Three legacy inputs (`smoke_centerpoint.synthetic_points` with 40k, 80k and 160k points) are held out to test the cost models.

Each configuration records the median of 5 host voxelization calls. On the GPU it runs 3 warmup calls, then 10 timed model calls and 10 per-stage instrumented passes.

## Results by axis

Each value is the median over the configurations that share that axis value.

| Axis | Value | Points N | Voxels V | Sites conv2 | Sites conv_out | Points lost to voxel cap | Voxelize (CPU) | Sparse backbone | Dense part | Model median |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clutter | 0 | 158k | 64.0k | 57.3k | 5.9k | 4.5% | 4.0 ms | 18.4 ms | 30.6 ms | 49.3 ms |
| | 1 | 159k | 52.4k | 42.9k | 4.9k | 17.9% | 3.9 ms | 16.9 ms | 31.3 ms | 48.2 ms |
| | 4 | 159k | 41.9k | 33.1k | 4.6k | 30.9% | 3.8 ms | 15.2 ms | 31.5 ms | 47.0 ms |
| buildings | 0 | 159k | 49.5k | 40.1k | 4.6k | 19.1% | 3.8 ms | 16.0 ms | 30.9 ms | 47.1 ms |
| | 10 | 158k | 50.8k | 43.8k | 5.2k | 19.2% | 4.1 ms | 16.6 ms | 31.0 ms | 48.2 ms |
| azimuth | 1325 | 79k | 40.2k | 37.5k | 4.8k | 9.5% | 2.1 ms | 14.9 ms | 31.0 ms | 46.3 ms |
| | 2650 | 159k | 52.4k | 42.9k | 5.1k | 17.9% | 3.9 ms | 16.8 ms | 31.0 ms | 48.0 ms |
| | 5300 | 317k | 64.4k | 48.1k | 5.2k | 38.1% | 6.6 ms | 18.0 ms | 31.0 ms | 49.5 ms |

"Dense part" is HeightCompression + BaseBEVBackbone + CenterHead, including decode and NMS.

| Range across all 90 scans | Min | Median | Max |
| --- | ---: | ---: | ---: |
| Model call | 44.3 ms | — | 54.4 ms |
| Sparse backbone | 13.2 ms | 16.3 ms | 23.6 ms |
| Dense part | 30.3 ms | 31.0 ms | 32.1 ms |
| Sparse backbone share of model | 29% | — | 43% |

Repeating a configuration with different seeds varies model latency by up to 4.0 ms.

## Findings

1. **For surface-structured input, the dense part dominates, and its cost does not depend on the input.** The input varied 4× in point count (77k–319k) and 2.8× in voxel count (31k–86k), yet the model call stayed within 44–54 ms. The dense 2D part held at 31 ± 1 ms (57–71% of the call), and the sparse backbone moved over 13–24 ms. All the variation comes from the sparse 3D stage.
2. **More objects make the frame cheaper.** Clutter blocks the beams: rays that would have spread across ground and building faces instead pile onto nearby object surfaces. Voxels drop from 64k to 42k, and points lost to the voxel cap rise from 4.5% to 31%. Point count alone is a poor proxy for cost.
3. **The voxel cap absorbs extra density.** Doubling and quadrupling the azimuth samples raises voxels by only 1.3× and 1.6×, and active sites by even less (conv_out +8%). The share of points lost to the 5-points-per-voxel cap rises from 9.5% to 38%. Host voxelization, by contrast, scales with points: 2.1 → 3.9 → 6.6 ms, fit about 0.97 ms + 18.8 ns per point (R² 0.78). It is the one stage outside the model call, and the only stage whose cost grows with the raw point count rather than with occupancy.
4. **Sparse-backbone cost follows channel-weighted active sites, not voxels.** Three linear models were fitted to the 90 scans and tested on the held-out legacy inputs:

   | Predictor of backbone time | R² on scans | Legacy 40k / 80k / 160k (observed 83 / 135 / 199 ms) |
   | --- | ---: | --- |
   | voxels V | 0.88 | 15 / 21 / 31 ms (5.6–6.5× too low) |
   | Σ active sites across levels | 0.94 | 33 / 50 / 74 ms (2.5–2.7× too low) |
   | per-level sites × kernel volume × Cin × Cout (MAC proxy) | 0.87 | 99 / 141 / 183 ms (−8% to +20%) |

   Only the MAC proxy extrapolates. The legacy input's expensive levels are the deep ones (conv3 at 64 channels, conv4 at 128). Raw voxel or site counts can't see that, so a cost model for this stage needs active sites per level weighted by channel width. That means the per-level counts S2 to S5 are what any target-side estimate needs, not V alone.
5. **The legacy 160k input reaches the 150,000-voxel cap.** Nothing in the scan sweep comes close; the highest is 86k.
6. **Detection counts are meaningless with random weights.** Every configuration, scan and legacy alike, produced exactly 313 detections. The heatmap is dominated by the initial bias, not by the input. NMS load therefore cannot be studied through this path. It needs a direct sweep of candidate count M, or trained weights.

## Limitations

- The scan omits the short-range lidars, second returns, vegetation and the ego vehicle body (see [BOUNDARY-INVENTORY.md](BOUNDARY-INVENTORY.md)). Real frames are likely to have more voxels and active sites at the same point count, so the backbone share measured here is probably a lower bound.
- The MAC proxy assumes every kernel tap is occupied. It is a ranking device and should not be read as a FLOP count.
- Voxelization time is for one call of spconv's `Point2VoxelCPU3d` on a desktop x86-64 CPU. Its threading behavior was not examined, and it says nothing about an embedded CPU beyond the linear scaling with N.

## Next steps

1. Sweep NMS and decode directly over candidate count M (up to the configured 4096 pre-NMS cap), independent of weights.
2. Export the dense segment to ONNX. It is now shown to account for most of the frame on surface-structured input, and its contract is static.
3. Time host voxelization and NMS on an AArch64 CPU to get the first CPU-side scaling data on representative hardware.

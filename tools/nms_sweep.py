#!/usr/bin/env python3
"""Decode/NMS cost as a function of candidate count, independent of weights.

Random weights produce a fixed, input-independent detection count, so the
post-processing path is driven directly here:

A. iou3d_nms_utils.nms_gpu alone on synthetic candidate boxes (M boxes,
   spread or clustered), including counts beyond the configured top-K.
B. CenterHead.generate_predicted_boxes (heatmap decode, top-K, mask,
   class-agnostic NMS) on constructed head outputs that yield exactly M
   candidates above the score threshold.

The pinned nms_gpu computes the pairwise BEV-overlap bitmask on the GPU, copies
it to the host with a blocking cudaMemcpy, runs the greedy suppression loop on
the CPU, and copies the kept indices back (src/iou3d_nms.cpp nms_gpu). Wall
time therefore includes host work and synchronization; GPU-busy time is taken
from the CUDA activity recorded by torch.profiler.

Reference implementation on this host only; not a performance baseline.
"""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from pcdet_harness import Workload

SYNC_OPS = ("cudaMemcpy", "cudaMemcpyAsync", "cudaStreamSynchronize", "cudaDeviceSynchronize")
ALLOC_OPS = ("cudaMalloc", "cudaFree")


def wall_ms(fn, warmup, reps):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    xs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        xs.append((time.perf_counter() - t0) * 1e3)
    xs.sort()
    return {"median": statistics.median(xs), "p95": xs[int(round(0.95 * (len(xs) - 1)))], "min": xs[0]}


def profile_once(fn):
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    gpu_us, kernels, host_calls = 0.0, {}, {}
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            gpu_us += e.device_time_total
            kernels[e.name] = kernels.get(e.name, 0) + 1
        elif e.name in SYNC_OPS + ALLOC_OPS:
            host_calls[e.name] = host_calls.get(e.name, 0) + 1
    return {"gpu_busy_ms": gpu_us / 1e3, "cuda_activities": sum(kernels.values()), "host_runtime_calls": host_calls}


def candidate_boxes(m, layout, rng):
    """M car-sized boxes [x, y, z, dx, dy, dz, heading] and descending-ish scores."""
    if layout == "spread":
        xy = rng.uniform(-74.0, 74.0, (m, 2))
        yaw = rng.uniform(-np.pi, np.pi, m)
    else:  # clustered: groups of 8 near-duplicates, as around one heatmap peak
        centers = rng.uniform(-74.0, 74.0, ((m + 7) // 8, 2))
        xy = np.repeat(centers, 8, axis=0)[:m] + rng.normal(0.0, 0.2, (m, 2))
        yaw = np.repeat(rng.uniform(-np.pi, np.pi, len(centers)), 8)[:m] + rng.normal(0.0, 0.05, m)
    dims = np.column_stack([rng.uniform(4.0, 5.0, m), rng.uniform(1.8, 2.0, m), rng.uniform(1.5, 1.8, m)])
    boxes = np.column_stack([xy, np.full(m, 1.0), dims, yaw]).astype(np.float32)
    scores = rng.uniform(0.11, 1.0, m).astype(np.float32)
    return torch.from_numpy(boxes).cuda(), torch.from_numpy(scores).cuda()


def head_outputs(m, layout, rng, shape, num_class, stride, voxel_size):
    """Head tensors whose decode yields exactly M candidates above threshold."""
    _, h, w = shape
    hm = torch.full((1, num_class, h, w), -10.0)
    center = torch.full((1, 2, h, w), 0.5)
    if m:
        if layout == "spread":
            flat = rng.choice(h * w, m, replace=False)
            ys, xs = flat // w, flat % w
            offx = offy = np.full(m, 0.5)
        else:
            # 3x3 cell blocks whose center offsets decode to near-identical positions.
            n_clusters = (m + 8) // 9
            rows_, cols_ = np.arange(2, h - 2, 4), np.arange(2, w - 2, 4)
            block = rng.choice(len(rows_) * len(cols_), n_clusters, replace=False)
            cy, cx = rows_[block // len(cols_)], cols_[block % len(cols_)]
            dy, dx = np.meshgrid([-1, 0, 1], [-1, 0, 1], indexing="ij")
            ys = (cy[:, None] + dy.ravel()).ravel()[:m]
            xs = (cx[:, None] + dx.ravel()).ravel()[:m]
            jitter = 0.2 / (stride * voxel_size)  # 0.2 m in cells
            offx = (np.repeat(cx, 9)[:m] - xs) + 0.5 + rng.normal(0.0, jitter, m)
            offy = (np.repeat(cy, 9)[:m] - ys) + 0.5 + rng.normal(0.0, jitter, m)
        cls = rng.integers(0, num_class, m)
        hm[0, cls, ys, xs] = torch.from_numpy(rng.uniform(0.5, 4.0, m)).float()
        center[0, 0, ys, xs] = torch.from_numpy(offx).float()
        center[0, 1, ys, xs] = torch.from_numpy(offy).float()
    d = {
        "hm": hm,
        "center": center,
        "center_z": torch.full((1, 1, h, w), 1.0),
        "dim": torch.log(torch.tensor([4.5, 1.9, 1.6])).view(1, 3, 1, 1).expand(1, 3, h, w).contiguous(),
        "rot": torch.cat([torch.ones(1, 1, h, w), torch.zeros(1, 1, h, w)], dim=1),
    }
    return {k: v.cuda() for k, v in d.items()}


def ints(s):
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openpcdet", default="third_party/OpenPCDet")
    ap.add_argument("--cfg", default="third_party/OpenPCDet/tools/cfgs/waymo_models/centerpoint.yaml")
    ap.add_argument("--counts", type=ints, default=ints("0,1,10,50,100,200,300,400,500"))
    ap.add_argument("--beyond-config", type=ints, default=ints("1000,2000,4096"),
                    help="extra nms_gpu-only counts above the configured top-K")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results/sweep/nms.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    wl = Workload(args.openpcdet, args.cfg)
    from pcdet.ops.iou3d_nms import iou3d_nms_utils

    torch.manual_seed(args.seed)
    model = wl.build_model().cuda().eval()
    head = model.dense_head
    post = head.model_cfg.POST_PROCESSING
    thresh = float(post.NMS_CONFIG.NMS_THRESH)
    stride = int(head.feature_map_stride)
    shape = (len(wl.cfg.CLASS_NAMES), int(wl.grid_size[1] // stride), int(wl.grid_size[0] // stride))
    top_k = int(post.MAX_OBJ_PER_SAMPLE)

    rows_a, rows_b = [], []
    for layout in ("spread", "clustered"):
        for m in [c for c in args.counts if c > 0] + args.beyond_config:
            rng = np.random.default_rng(args.seed + m)
            boxes, scores = candidate_boxes(m, layout, rng)
            fn = lambda: iou3d_nms_utils.nms_gpu(boxes, scores, thresh)  # noqa: E731
            kept = int(fn()[0].shape[0])
            col_blocks = -(-m // 64)
            rows_a.append({"layout": layout, "candidates": m, "within_config": m <= top_k, "kept": kept,
                           "mask_bytes_device_to_host": m * col_blocks * 8,
                           "wall_ms": wall_ms(fn, args.warmup, args.reps), **profile_once(fn)})
            r = rows_a[-1]
            print(f"A {layout:9} M={m:5d} kept={kept:5d} wall={r['wall_ms']['median']:.3f}ms "
                  f"gpu={r['gpu_busy_ms']:.3f}ms calls={r['host_runtime_calls']}", flush=True)

        for m in args.counts:
            if m > top_k:
                continue
            rng = np.random.default_rng(args.seed + 1000 + m)
            preds = [head_outputs(m, layout, rng, shape, len(wl.cfg.CLASS_NAMES), stride, wl.voxel_size[0])]
            with torch.inference_mode():
                fn = lambda: head.generate_predicted_boxes(1, preds)  # noqa: E731
                out = fn()
                rows_b.append({"layout": layout, "candidates_requested": m,
                               "detections_after_nms": int(out[0]["pred_boxes"].shape[0]),
                               "wall_ms": wall_ms(fn, args.warmup, args.reps), **profile_once(fn)})
            r = rows_b[-1]
            print(f"B {layout:9} M={m:5d} dets={r['detections_after_nms']:4d} wall={r['wall_ms']['median']:.3f}ms "
                  f"gpu={r['gpu_busy_ms']:.3f}ms calls={r['host_runtime_calls']}", flush=True)

    result = {
        "valid_for_baseline": False,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "gpu": torch.cuda.get_device_name(0), "cpu": platform.processor() or platform.machine()},
        "config": {"nms_type": post.NMS_CONFIG.NMS_TYPE, "nms_thresh": thresh, "top_k": top_k,
                   "nms_pre_maxsize": int(post.NMS_CONFIG.NMS_PRE_MAXSIZE),
                   "nms_post_maxsize": int(post.NMS_CONFIG.NMS_POST_MAXSIZE),
                   "score_thresh": float(post.SCORE_THRESH), "heatmap_shape": list(shape)},
        "protocol": {"warmup": args.warmup, "reps": args.reps, "seed": args.seed},
        "nms_gpu_only": rows_a,
        "decode_and_nms": rows_b,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

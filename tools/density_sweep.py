#!/usr/bin/env python3
"""Scene-density sweep: how input structure drives per-stage cost.

Generates parametric scans (tools/scan_generator.py) over a grid of object
clutter, building presence, azimuth density and seeds. For each scan it records
the host-side voxelization time, voxel count, sparse-backbone active sites per
level, and GPU stage latencies. It then fits simple cost models.

Weights are random and inputs synthetic: valid for structural cost
(shapes, sparsity, latency of the reference implementation on this host),
not for detection quality. Not a performance baseline.
"""
import argparse
import dataclasses
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from pcdet_harness import Workload
from scan_generator import ScanParams, generate

LEVELS = ("conv_input", "conv1", "conv2", "conv3", "conv4", "conv_out")

# Per-site multiply-accumulate proxy for each sparse level, assuming every
# kernel tap is occupied (an upper bound; real neighbor occupancy is lower).
# Layer lists follow spconv_backbone.py VoxelResBackBone8x.
MAC_PER_SITE = {
    "conv_input": 27 * 5 * 16,
    "conv1": 4 * 27 * 16 * 16,
    "conv2": 27 * 16 * 32 + 4 * 27 * 32 * 32,
    "conv3": 27 * 32 * 64 + 4 * 27 * 64 * 64,
    "conv4": 27 * 64 * 128 + 4 * 27 * 128 * 128,
    "conv_out": 3 * 128 * 128,
}


def median_p95(xs):
    xs = sorted(xs)
    return statistics.median(xs), xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))]


def cuda_time(fn):
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    stop.record()
    return out, start, stop


def measure(wl, model, levels, raw, seed, warmup, iterations):
    points = wl.mask_and_shuffle(raw, seed)
    vox_ms = []
    for _ in range(5):
        t0 = time.perf_counter()
        voxels, coords, num = wl.voxelize(points)
        vox_ms.append((time.perf_counter() - t0) * 1e3)
    base = wl.to_gpu(wl.collate(voxels, coords, num))

    def fresh():
        return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base.items()}

    with torch.inference_mode():
        for _ in range(warmup):
            model(fresh())
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_alloc = torch.cuda.memory_allocated()
        total = []
        for _ in range(iterations):
            batch = fresh()
            (pred, _), a, b = cuda_time(lambda: model(batch))
            b.synchronize()
            total.append(a.elapsed_time(b))
        peak = torch.cuda.max_memory_allocated() - start_alloc

        stage = {}
        for _ in range(iterations):
            batch = fresh()
            events = []
            for m in model.module_list:
                batch, a, b = cuda_time(lambda m=m, batch=batch: m(batch))
                events.append((type(m).__name__, a, b))
            torch.cuda.synchronize()
            for name, a, b in events:
                stage.setdefault(name, []).append(a.elapsed_time(b))

    occ = np.bincount(num, minlength=wl.max_points_per_voxel + 1)[1:]
    t_med, t_p95 = median_p95(total)
    return {
        "points_in_range": int(points.shape[0]),
        "voxels": int(voxels.shape[0]),
        "points_dropped_by_voxel_cap": int(points.shape[0] - num.sum()),
        "voxels_by_occupancy_1_to_max": occ.tolist(),
        "active_sites": {k: levels[k] for k in LEVELS},
        "voxelize_cpu_ms_median": statistics.median(vox_ms),
        "model_ms_median": t_med,
        "model_ms_p95": t_p95,
        "stage_ms_median": {k: statistics.median(v) for k, v in stage.items()},
        "peak_allocated_above_inputs_bytes": int(peak),
        "detections_random_weights": int(pred[0]["pred_boxes"].shape[0]),
    }


def fit(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    slope, intercept = np.polyfit(xs, ys, 1)
    pred = slope * xs + intercept
    r2 = 1 - ((ys - pred) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2)}


def analyze(rows):
    scans = [r for r in rows if r["input"]["kind"] == "scan"]
    legacy = [r for r in rows if r["input"]["kind"] == "legacy"]

    def backbone(r):
        return r["stage_ms_median"]["VoxelResBackBone8x"]

    def dense(r):
        s = r["stage_ms_median"]
        return s["HeightCompression"] + s["BaseBEVBackbone"] + s["CenterHead"]

    predictors = {
        "voxels": lambda r: r["voxels"],
        "sum_active_sites": lambda r: sum(r["active_sites"].values()),
        "mac_proxy_gmac": lambda r: sum(MAC_PER_SITE[k] * r["active_sites"][k] for k in LEVELS) / 1e9,
    }
    out = {"n_scan": len(scans), "n_legacy": len(legacy), "backbone_ms_fits": {}}
    for name, f in predictors.items():
        m = fit([f(r) for r in scans], [backbone(r) for r in scans])
        m["legacy_check"] = [{
            "label": r["input"]["label"], "x": f(r), "observed_ms": backbone(r),
            "predicted_ms": m["slope"] * f(r) + m["intercept"],
        } for r in legacy]
        out["backbone_ms_fits"][name] = m
    out["voxelize_cpu_ms_vs_points"] = fit([r["points_in_range"] for r in rows],
                                           [r["voxelize_cpu_ms_median"] for r in rows])
    d = [dense(r) for r in scans]
    out["dense_part_ms"] = {"min": min(d), "median": statistics.median(d), "max": max(d)}
    b = [backbone(r) for r in scans]
    out["backbone_ms"] = {"min": min(b), "median": statistics.median(b), "max": max(b)}
    out["backbone_share_of_model"] = {
        "min": min(backbone(r) / r["model_ms_median"] for r in scans),
        "max": max(backbone(r) / r["model_ms_median"] for r in scans),
    }
    return out


def floats(s):
    return [float(x) for x in s.split(",")]


def ints(s):
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openpcdet", default="third_party/OpenPCDet")
    ap.add_argument("--cfg", default="third_party/OpenPCDet/tools/cfgs/waymo_models/centerpoint.yaml")
    ap.add_argument("--clutter", type=floats, default=floats("0,0.5,1,2,4"),
                    help="multiplier on default vehicle/pedestrian/cyclist/pole counts")
    ap.add_argument("--buildings", type=ints, default=ints("0,10"), help="buildings_per_side values")
    ap.add_argument("--azimuth", type=ints, default=ints("1325,2650,5300"), help="azimuth_samples values")
    ap.add_argument("--seeds", type=ints, default=ints("1,2,3"))
    ap.add_argument("--legacy", type=ints, default=ints("40000,80000,160000"),
                    help="legacy smoke_centerpoint.synthetic_points sizes, for out-of-sample checks")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--out", default="results/sweep/density.jsonl")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    wl = Workload(args.openpcdet, args.cfg)
    torch.manual_seed(7)
    model = wl.build_model().cuda().eval()
    sparse = next(m for m in model.module_list if type(m).__name__ == "VoxelResBackBone8x")
    levels = {}
    for name in LEVELS:
        getattr(sparse, name).register_forward_hook(
            lambda _m, _i, out, name=name: levels.__setitem__(name, int(out.features.shape[0])))

    defaults = ScanParams()
    jobs = []
    for c in args.clutter:
        for bld in args.buildings:
            for az in args.azimuth:
                for seed in args.seeds:
                    p = dataclasses.replace(
                        defaults, buildings_per_side=bld, azimuth_samples=az,
                        vehicles=round(defaults.vehicles * c), pedestrians=round(defaults.pedestrians * c),
                        cyclists=round(defaults.cyclists * c), poles=round(defaults.poles * c))
                    jobs.append(("scan", {"clutter": c, "buildings_per_side": bld, "azimuth_samples": az}, p, seed))
    for n in args.legacy:
        jobs.append(("legacy", {"points": n}, n, 7))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with out.open("w") as f:
        for i, (kind, axes, p, seed) in enumerate(jobs):
            if kind == "scan":
                raw, scene = generate(p, seed)
                inp = {"kind": "scan", "seed": seed, "axes": axes, "params": dataclasses.asdict(p),
                       "boxes_placed": scene["boxes"], "label": f"scan c={axes['clutter']} b={axes['buildings_per_side']} "
                                                               f"az={axes['azimuth_samples']} s={seed}"}
            else:
                from smoke_centerpoint import synthetic_points
                raw = synthetic_points(p, seed)
                inp = {"kind": "legacy", "seed": seed, "axes": axes, "label": f"legacy {p}"}
            inp["raw_points"] = int(raw.shape[0])
            row = {"input": inp, **measure(wl, model, levels, raw, seed, args.warmup, args.iterations)}
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"[{i + 1}/{len(jobs)}] {inp['label']:32} N={row['points_in_range']:7d} V={row['voxels']:7d} "
                  f"S_out={row['active_sites']['conv_out']:6d} vox={row['voxelize_cpu_ms_median']:6.1f}ms "
                  f"model={row['model_ms_median']:6.1f}ms bb={row['stage_ms_median']['VoxelResBackBone8x']:6.1f}ms",
                  flush=True)

    summary = {
        "valid_for_baseline": False,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "gpu": torch.cuda.get_device_name(0), "cpu": platform.processor() or platform.machine(),
                        "torch_cpu_threads": torch.get_num_threads()},
        "protocol": {"warmup": args.warmup, "iterations": args.iterations, "voxelize_repeats": 5},
        "analysis": analyze(rows),
    }
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["analysis"], indent=1))


if __name__ == "__main__":
    main()

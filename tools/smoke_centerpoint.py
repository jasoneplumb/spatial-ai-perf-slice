#!/usr/bin/env python3
"""Structural CenterPoint smoke test.

This deliberately uses deterministic synthetic LiDAR points and randomly
initialized weights. It validates that the pinned Waymo CenterPoint graph can
execute on the local GPU. Its latency is NOT a benchmark baseline.
"""
import argparse
import copy
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def percentile(values, q):
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def git_head(path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def load_openpcdet_cfg(root: Path, cfg_path: Path):
    from pcdet.config import cfg, cfg_from_yaml_file

    old_cwd = Path.cwd()
    try:
        os.chdir(root / "tools")
        cfg_from_yaml_file(str(cfg_path), cfg)
    finally:
        os.chdir(old_cwd)
    return cfg


def synthetic_points(count, seed):
    rng = np.random.default_rng(seed)
    ground_count = int(count * 0.70)
    elevated_count = count - ground_count

    # Ground-like returns distributed radially around the sensor.
    theta = rng.uniform(-np.pi, np.pi, ground_count)
    radius = 70.0 * np.sqrt(rng.random(ground_count))
    ground = np.empty((ground_count, 5), dtype=np.float32)
    ground[:, 0] = radius * np.cos(theta)
    ground[:, 1] = radius * np.sin(theta)
    ground[:, 2] = -1.45 + rng.normal(0.0, 0.05, ground_count)
    ground[:, 3] = rng.uniform(0.0, 1.0, ground_count)
    ground[:, 4] = rng.uniform(0.0, 1.0, ground_count)

    # Elevated returns provide sparse occupancy above the road plane.
    theta = rng.uniform(-np.pi, np.pi, elevated_count)
    radius = 65.0 * np.sqrt(rng.random(elevated_count))
    elevated = np.empty((elevated_count, 5), dtype=np.float32)
    elevated[:, 0] = radius * np.cos(theta)
    elevated[:, 1] = radius * np.sin(theta)
    elevated[:, 2] = rng.uniform(-1.2, 3.5, elevated_count)
    elevated[:, 3] = rng.uniform(0.0, 1.0, elevated_count)
    elevated[:, 4] = rng.uniform(0.0, 1.0, elevated_count)

    points = np.concatenate([ground, elevated], axis=0)
    rng.shuffle(points)
    return points


def clone_batch(batch):
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.clone()
        elif isinstance(value, np.ndarray):
            out[key] = value.copy()
        else:
            out[key] = copy.deepcopy(value)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openpcdet", default="third_party/OpenPCDet")
    ap.add_argument(
        "--cfg",
        default="third_party/OpenPCDet/tools/cfgs/waymo_models/centerpoint.yaml",
    )
    ap.add_argument("--points", type=int, default=80000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--out", default="results/smoke/centerpoint.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    root = Path(args.openpcdet).resolve()
    cfg_path = Path(args.cfg).resolve()
    sys.path.insert(0, str(root))

    from pcdet.datasets import DatasetTemplate
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils import common_utils

    cfg = load_openpcdet_cfg(root, cfg_path)
    logger = common_utils.create_logger()

    dataset = DatasetTemplate(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=Path("."),
        logger=logger,
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    points = synthetic_points(args.points, args.seed)
    prepared = dataset.prepare_data(
        {"points": points, "frame_id": np.array("synthetic-smoke")}
    )
    cpu_batch = dataset.collate_batch([prepared])
    voxel_count = int(cpu_batch["voxels"].shape[0])

    model = build_network(
        model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset
    )
    model.cuda().eval()

    base_batch = copy.deepcopy(cpu_batch)
    load_data_to_gpu(base_batch)
    torch.cuda.synchronize()

    def run_once(timed=False):
        batch = clone_batch(base_batch)
        start = stop = None
        if timed:
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
        with torch.inference_mode():
            pred, _ = model(batch)
        if timed:
            stop.record()
            stop.synchronize()
            return pred, float(start.elapsed_time(stop))
        torch.cuda.synchronize()
        return pred, None

    for _ in range(args.warmup):
        run_once(False)

    torch.cuda.reset_peak_memory_stats()
    times = []
    first_pred = None
    for i in range(args.iterations):
        pred, elapsed = run_once(True)
        times.append(elapsed)
        if i == 0:
            first_pred = pred

    detections = 0
    if first_pred and "pred_boxes" in first_pred[0]:
        detections = int(first_pred[0]["pred_boxes"].shape[0])

    result = {
        "valid_for_baseline": False,
        "purpose": "structural smoke test only",
        "weights": "randomly initialized",
        "input": "deterministic synthetic lidar-like points",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "openpcdet_commit": git_head(root),
        "config": str(cfg_path),
        "requested_points": args.points,
        "voxel_count": voxel_count,
        "detections_random_weights": detections,
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "latency_ms": {
            "min": min(times),
            "median": statistics.median(times),
            "mean": statistics.mean(times),
            "p95": percentile(times, 0.95),
            "max": max(times),
        },
        "peak_allocated_vram_gib": round(
            torch.cuda.max_memory_allocated() / (1024 ** 3), 3
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

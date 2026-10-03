#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
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


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def load_openpcdet_cfg(root: Path, cfg_path: Path):
    from pcdet.config import cfg, cfg_from_yaml_file

    old_cwd = Path.cwd()
    try:
        os.chdir(root / "tools")
        cfg_from_yaml_file(str(cfg_path), cfg)
    finally:
        os.chdir(old_cwd)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openpcdet", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for this baseline harness")

    root = Path(args.openpcdet).resolve()
    cfg_path = Path(args.cfg).resolve()
    sys.path.insert(0, str(root))

    from types import ModuleType, SimpleNamespace

    datasets_pkg = ModuleType("pcdet.datasets")
    datasets_pkg.__path__ = [str(root / "pcdet" / "datasets")]
    datasets_pkg.__package__ = "pcdet.datasets"
    sys.modules.setdefault("pcdet.datasets", datasets_pkg)

    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils import common_utils

    cfg = load_openpcdet_cfg(root, cfg_path)
    logger = common_utils.create_logger()

    point_cloud_range = np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE, dtype=np.float32)
    voxel_cfg = next(
        x for x in cfg.DATA_CONFIG.DATA_PROCESSOR if x.NAME == "transform_points_to_voxels"
    )
    voxel_size = np.asarray(voxel_cfg.VOXEL_SIZE, dtype=np.float32)
    grid_size = np.round(
        (point_cloud_range[3:6] - point_cloud_range[0:3]) / voxel_size
    ).astype(np.int64)
    dataset = SimpleNamespace(
        class_names=cfg.CLASS_NAMES,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        grid_size=grid_size,
        depth_downsample_factor=None,
        point_feature_encoder=SimpleNamespace(num_point_features=5),
    )
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(
        filename=str(Path(args.checkpoint).resolve()), logger=logger, to_cpu=True
    )
    model.cuda().eval()

    saved = torch.load(args.frame, map_location="cpu", weights_only=False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def prepare_gpu_batch():
        batch = copy.deepcopy(saved)
        load_data_to_gpu(batch)
        torch.cuda.synchronize()
        return batch

    def run_model(batch):
        with torch.inference_mode():
            pred, _ = model(batch)
        return pred

    for _ in range(args.warmup):
        batch = prepare_gpu_batch()
        run_model(batch)
        torch.cuda.synchronize()

    times_ms = []
    prediction = None
    for i in range(args.iterations):
        batch = prepare_gpu_batch()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        pred = run_model(batch)
        stop.record()
        stop.synchronize()
        times_ms.append(float(start.elapsed_time(stop)))
        if i == 0:
            prediction = pred

    torch.save(prediction, out / "prediction.pt")

    # Profiling is separate so profiler overhead never contaminates latency.
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(5):
            batch = prepare_gpu_batch()
            run_model(batch)
            torch.cuda.synchronize()
            prof.step()

    (out / "operators.txt").write_text(
        prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=100) + "\n"
    )
    prof.export_chrome_trace(str(out / "trace.json"))

    timing = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "openpcdet_commit": git_head(root),
        "config": str(cfg_path),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "frame": str(Path(args.frame).resolve()),
        "frame_sha256": sha256(args.frame),
        "timing_boundary": "model(batch_dict) only; frame load/copy/H2D excluded",
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "latency_ms": {
            "min": min(times_ms),
            "median": statistics.median(times_ms),
            "mean": statistics.mean(times_ms),
            "p90": percentile(times_ms, 0.90),
            "p95": percentile(times_ms, 0.95),
            "p99": percentile(times_ms, 0.99),
            "max": max(times_ms),
            "stdev": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        },
        "samples_ms": times_ms,
    }
    (out / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

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


def fresh_batch(saved):
    # load_data_to_gpu mutates the dictionary, so every iteration gets a copy.
    return copy.deepcopy(saved)


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
    sys.path.insert(0, str(root))

    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.models import build_network, load_data_to_gpu

    cfg_from_yaml_file(str(Path(args.cfg).resolve()), cfg)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=None)
    model.load_params_from_file(filename=str(Path(args.checkpoint).resolve()), logger=None, to_cpu=False)
    model.cuda().eval()

    saved = torch.load(args.frame, map_location="cpu", weights_only=False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def run_once():
        batch = fresh_batch(saved)
        load_data_to_gpu(batch)
        with torch.inference_mode():
            pred, _ = model(batch)
        return pred

    for _ in range(args.warmup):
        run_once()
    torch.cuda.synchronize()

    times_ms = []
    prediction = None
    for i in range(args.iterations):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        pred = run_once()
        stop.record()
        stop.synchronize()
        times_ms.append(float(start.elapsed_time(stop)))
        if i == 0:
            prediction = pred

    torch.save(prediction, out / "prediction.pt")

    # Profile a small number of iterations separately so profiler overhead
    # never contaminates the primary latency distribution.
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(5):
            run_once()
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
        "config": str(Path(args.cfg).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "frame": str(Path(args.frame).resolve()),
        "frame_sha256": sha256(args.frame),
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

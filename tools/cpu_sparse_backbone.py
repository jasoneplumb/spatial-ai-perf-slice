#!/usr/bin/env python3
"""Cost of the sparse 3D backbone when it runs on a general-purpose CPU.

spconv's default implicit-GEMM algorithm is CUDA-only. On the CPU this tool
switches every sparse convolution to spconv's Native algorithm (rulebook
construction, then gather-GEMM-scatter). spconv's CPU path does not support a
fused bias, so each conv's bias is applied by a forward hook instead; the
output is checked against the GPU backbone for every input.

Time is split per backbone level and, across the whole backbone, into
rulebook (index-pair) construction, gather-GEMM-scatter convolution, and the
remainder (BatchNorm, ReLU, residual adds, framework overhead), at several
CPU thread counts.

Desktop x86-64 CPU, random weights, synthetic input: a ranking and scaling
signal for a CPU fallback, not a prediction for any embedded CPU.
"""
import argparse
import copy
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
# Same per-site MAC proxy as tools/density_sweep.py.
MAC_PER_SITE = {
    "conv_input": 27 * 5 * 16,
    "conv1": 4 * 27 * 16 * 16,
    "conv2": 27 * 16 * 32 + 4 * 27 * 32 * 32,
    "conv3": 27 * 32 * 64 + 4 * 27 * 64 * 64,
    "conv4": 27 * 64 * 128 + 4 * 27 * 128 * 128,
    "conv_out": 3 * 128 * 128,
}


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


class Timers:
    """Wall-clock accumulators around spconv's rulebook and convolution ops."""

    def __init__(self):
        import spconv.pytorch.ops as ops

        self.ops = ops
        self.acc = {"rulebook": 0.0, "gather_gemm_scatter": 0.0}
        self.originals = {}
        for name, key in (("get_indice_pairs", "rulebook"), ("indice_conv", "gather_gemm_scatter")):
            fn = getattr(ops, name)
            self.originals[name] = fn

            def wrapped(*a, _fn=fn, _key=key, **k):
                t0 = time.perf_counter()
                try:
                    return _fn(*a, **k)
                finally:
                    self.acc[_key] += time.perf_counter() - t0

            setattr(ops, name, wrapped)

    def reset(self):
        for k in self.acc:
            self.acc[k] = 0.0


def make_cpu_backbone(bb):
    from spconv.core import ConvAlgo
    from spconv.pytorch.conv import SparseConvolution

    bb = copy.deepcopy(bb).cpu()
    for m in bb.modules():
        if isinstance(m, SparseConvolution):
            m.algo = ConvAlgo.Native
            if m.bias is not None:
                bias = m.bias.detach().clone()
                m.bias = None
                m.register_forward_hook(lambda _m, _i, out, bias=bias: out.replace_feature(out.features + bias))
    return bb


def run_levels(bb, batch, level_ms, sites):
    import spconv.pytorch as spconv

    x = spconv.SparseConvTensor(features=batch["voxel_features"], indices=batch["voxel_coords"].int(),
                                spatial_shape=bb.sparse_shape, batch_size=1)
    for name in LEVELS:
        t0 = time.perf_counter()
        x = getattr(bb, name)(x)
        level_ms[name] = (time.perf_counter() - t0) * 1e3
        sites[name] = int(x.features.shape[0])
    return x


def ints(s):
    return [int(x) for x in s.split(",")]


def floats(s):
    return [float(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openpcdet", default="third_party/OpenPCDet")
    ap.add_argument("--cfg", default="third_party/OpenPCDet/tools/cfgs/waymo_models/centerpoint.yaml")
    ap.add_argument("--clutter", type=floats, default=floats("0,1,4"))
    ap.add_argument("--azimuth", type=ints, default=ints("1325,2650,5300"))
    ap.add_argument("--buildings", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--legacy", type=ints, default=ints("80000"))
    ap.add_argument("--threads", type=ints, default=ints("1,4,16"))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="results/sweep/cpu_sparse.jsonl")
    args = ap.parse_args()

    wl = Workload(args.openpcdet, args.cfg)
    torch.manual_seed(7)
    model = wl.build_model().eval()
    vfe, bb = model.module_list[0], model.module_list[1]
    gpu_bb = copy.deepcopy(bb).cuda() if torch.cuda.is_available() else None
    cpu_bb = make_cpu_backbone(bb)
    timers = Timers()

    defaults = ScanParams()
    inputs = []
    for c in args.clutter:
        for az in args.azimuth:
            p = dataclasses.replace(defaults, buildings_per_side=args.buildings, azimuth_samples=az,
                                    vehicles=round(defaults.vehicles * c), pedestrians=round(defaults.pedestrians * c),
                                    cyclists=round(defaults.cyclists * c), poles=round(defaults.poles * c))
            inputs.append(("scan", f"scan c={c} az={az} s={args.seed}", generate(p, args.seed)[0]))
    for n in args.legacy:
        from smoke_centerpoint import synthetic_points
        inputs.append(("legacy", f"legacy {n}", synthetic_points(n, 7)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with out.open("w") as f:
        for kind, label, raw in inputs:
            pts = wl.mask_and_shuffle(raw, args.seed)
            b = wl.collate(*wl.voxelize(pts))
            cpu = {k: (torch.from_numpy(v).float() if isinstance(v, np.ndarray) else v) for k, v in b.items()}
            with torch.inference_mode():
                cpu = vfe(cpu)
                check = None
                if gpu_bb is not None:
                    gpu = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in cpu.items()}
                    ref = gpu_bb(dict(gpu))["encoded_spconv_tensor"].dense().cpu().numpy()
                    got = cpu_bb(dict(cpu))["encoded_spconv_tensor"].dense().numpy()
                    check = float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-12))
                for nt in args.threads:
                    torch.set_num_threads(nt)
                    sites = {}
                    run_levels(cpu_bb, cpu, {}, sites)  # warmup
                    totals, levels, parts = [], {k: [] for k in LEVELS}, {k: [] for k in timers.acc}
                    for _ in range(args.reps):
                        timers.reset()
                        lv = {}
                        t0 = time.perf_counter()
                        run_levels(cpu_bb, cpu, lv, sites)
                        totals.append((time.perf_counter() - t0) * 1e3)
                        for k in LEVELS:
                            levels[k].append(lv[k])
                        for k, v in timers.acc.items():
                            parts[k].append(v * 1e3)
                    total = statistics.median(totals)
                    part = {k: statistics.median(v) for k, v in parts.items()}
                    part["other"] = total - sum(part.values())
                    row = {
                        "kind": kind, "label": label, "threads": nt, "points_in_range": int(pts.shape[0]),
                        "voxels": int(cpu["voxels"].shape[0]), "active_sites": sites,
                        "mac_proxy_gmac": sum(MAC_PER_SITE[k] * sites[k] for k in LEVELS) / 1e9,
                        "backbone_ms_median": total, "backbone_ms_min": min(totals),
                        "level_ms_median": {k: statistics.median(v) for k, v in levels.items()},
                        "split_ms_median": part, "max_rel_diff_vs_gpu": check,
                    }
                    rows.append(row)
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    print(f"{label:24} t={nt:2d} V={row['voxels']:6d} proxy={row['mac_proxy_gmac']:6.1f}G "
                          f"cpu={total:8.1f}ms rulebook={part['rulebook']:7.1f} gemm={part['gather_gemm_scatter']:7.1f} "
                          f"other={part['other']:6.1f} diff={check:.1e}", flush=True)
    torch.set_num_threads(max(args.threads))

    summary = {"valid_for_baseline": False, "cpu": cpu_model(), "torch": torch.__version__,
               "algorithm": "spconv ConvAlgo.Native on CPU; bias applied by hook", "reps": args.reps, "fits": {}}
    for nt in args.threads:
        scans = [r for r in rows if r["kind"] == "scan" and r["threads"] == nt]
        xs = np.array([r["mac_proxy_gmac"] for r in scans])
        ys = np.array([r["backbone_ms_median"] for r in scans])
        slope, icpt = np.polyfit(xs, ys, 1)
        pred = slope * xs + icpt
        r2 = 1 - ((ys - pred) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()
        legacy = [{"label": r["label"], "observed_ms": r["backbone_ms_median"],
                   "predicted_ms": slope * r["mac_proxy_gmac"] + icpt}
                  for r in rows if r["kind"] == "legacy" and r["threads"] == nt]
        summary["fits"][str(nt)] = {"ms_per_gmac_proxy": float(slope), "intercept_ms": float(icpt), "r2": float(r2),
                                    "scan_ms_range": [float(ys.min()), float(ys.max())], "legacy_check": legacy}
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["fits"], indent=1))


if __name__ == "__main__":
    main()

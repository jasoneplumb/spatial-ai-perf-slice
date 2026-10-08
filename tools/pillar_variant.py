#!/usr/bin/env python3
"""Pillar CenterPoint variant: stage cost and a static whole-network ONNX export.

OpenPCDet's centerpoint_pillar_1x.yaml replaces the spconv sparse 3D backbone
with PillarVFE (per-pillar point network) + PointPillarScatter (scatter into a
dense BEV grid), followed by BaseBEVBackbone and CenterHead, all dense.

A. GPU stage timing over parametric scans (same grid as tools/cpu_sparse_backbone.py).
B. A static-shape ONNX export of everything between the host voxelizer and the
   raw head outputs: pillars are padded to a fixed P_MAX, and the batch-1
   scatter is re-expressed without host syncs or data-dependent shapes (each
   padding pillar writes to its own dummy cell, which is sliced off). The
   rewrite is checked against OpenPCDet's own modules before export.

Random weights, synthetic input; reference GPU/CPU only. Not a performance baseline.
"""
import argparse
import dataclasses
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from export_dense_onnx import conv_macs, cpu_model, fold_constants, sha256, static_report
from pcdet_harness import Workload
from scan_generator import ScanParams, generate


class StaticPillarNet(nn.Module):
    """voxels [P,K,5], voxel_num_points [P] (0 = padding), voxel_coords [P,4] -> raw head outputs."""

    def __init__(self, model, grid_size, p_max):
        super().__init__()
        self.p_max = p_max
        # PFNLayer splits its Linear into 50k-pillar chunks (a memory workaround); with a static
        # P_MAX above that, tracing would bake in the chunk slices. One chunk is numerically identical.
        for layer in model.vfe.pfn_layers:
            layer.part = max(layer.part, p_max)
        self.vfe = model.vfe
        self.backbone_2d = model.backbone_2d
        self.shared_conv = model.dense_head.shared_conv
        self.head = model.dense_head.heads_list[0]
        self.output_names = list(self.head.sep_head_dict)
        self.nx, self.ny = int(grid_size[0]), int(grid_size[1])
        self.channels = model.map_to_bev_module.num_bev_features
        assert model.vfe.use_absolute_xyz and not model.vfe.with_distance, "rewrite covers this config only"

    def pillar_features(self, voxels, num, coords):
        """PillarVFE.forward written out of place.

        The original builds f_center with torch.zeros_like plus three in-place column writes, which export
        as a [P, K, 3] constant and ScatterND chains whose shapes inference cannot resolve. Same arithmetic,
        then OpenPCDet's own PFN layers.
        """
        v = self.vfe
        xyz = voxels[:, :, :3]
        f_cluster = xyz - xyz.sum(dim=1, keepdim=True) / num.type_as(voxels).view(-1, 1, 1)
        c = coords.to(voxels.dtype)
        f_center = torch.stack([
            voxels[:, :, 0] - (c[:, 3].unsqueeze(1) * v.voxel_x + v.x_offset),
            voxels[:, :, 1] - (c[:, 2].unsqueeze(1) * v.voxel_y + v.y_offset),
            voxels[:, :, 2] - (c[:, 1].unsqueeze(1) * v.voxel_z + v.z_offset),
        ], dim=-1)
        features = torch.cat([voxels, f_cluster, f_center], dim=-1)
        k = voxels.shape[1]
        mask = (num.int().unsqueeze(1) > torch.arange(k, dtype=torch.int, device=voxels.device).view(1, -1))
        features = features * mask.unsqueeze(-1).type_as(voxels)
        for pfn in v.pfn_layers:
            features = pfn(features)
        return features.squeeze(1)

    def forward(self, voxels, voxel_num_points, voxel_coords):
        valid = voxel_num_points > 0
        # Padding pillars get one point so PillarVFE's mean is finite; their output is discarded below.
        num = torch.where(valid, voxel_num_points, torch.ones_like(voxel_num_points))
        feats = self.pillar_features(voxels, num, voxel_coords)
        p = self.p_max
        cells = self.ny * self.nx
        coords = voxel_coords.long()
        real = coords[:, 2] * self.nx + coords[:, 3]
        dummy = cells + torch.arange(p, device=voxels.device)
        idx = torch.where(valid, real, dummy)
        # Derived from a runtime tensor so the exporter cannot bake a [cells + P_MAX, C] zero
        # constant (~94 MB) into the graph.
        canvas = torch.zeros_like(feats[:1]).expand(cells + p, self.channels)
        canvas = canvas.index_put((idx,), feats)
        spatial = canvas[:cells].t().reshape(1, self.channels, self.ny, self.nx)
        x = self.backbone_2d({"spatial_features": spatial})["spatial_features_2d"]
        out = self.head(self.shared_conv(x))
        return tuple(out[k] for k in self.output_names)


def pad(voxels, num, coords, p_max):
    p = voxels.shape[0]
    if p > p_max:
        raise SystemExit(f"{p} pillars exceed P_MAX {p_max}")
    v = np.zeros((p_max,) + voxels.shape[1:], np.float32)
    n = np.zeros((p_max,), np.int32)
    c = np.zeros((p_max, 4), np.int32)
    v[:p], n[:p], c[:p, 1:] = voxels, num, coords
    return v, n, c


def matmul_macs(model, shapes):
    inits = {i.name: list(i.dims) for i in model.graph.initializer}
    total = 0
    for node in model.graph.node:
        if node.op_type in ("MatMul", "Gemm") and node.input[1] in inits:
            k = inits[node.input[1]][0]
            total += int(np.prod(shapes[node.output[0]])) * k
    return total


def scans(args, buildings):
    d = ScanParams()
    for c in args.clutter:
        for az in args.azimuth:
            p = dataclasses.replace(d, buildings_per_side=buildings, azimuth_samples=az,
                                    vehicles=round(d.vehicles * c), pedestrians=round(d.pedestrians * c),
                                    cyclists=round(d.cyclists * c), poles=round(d.poles * c))
            yield f"scan c={c} az={az} s={args.seed}", generate(p, args.seed)[0]


def stage_timing(wl, model, raw, seed, warmup, iters):
    pts = wl.mask_and_shuffle(raw, seed)
    t0 = time.perf_counter()
    v, c, n = wl.voxelize(pts)
    vox_ms = (time.perf_counter() - t0) * 1e3
    base = wl.to_gpu(wl.collate(v, c, n))

    def fresh():
        return {k: (x.clone() if torch.is_tensor(x) else x) for k, x in base.items()}

    with torch.inference_mode():
        for _ in range(warmup):
            model(fresh())
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.memory_allocated()
        total, stage = [], {}
        for _ in range(iters):
            b = fresh()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            model(b)
            e1.record()
            e1.synchronize()
            total.append(e0.elapsed_time(e1))
        peak = torch.cuda.max_memory_allocated() - start
        for _ in range(iters):
            b = fresh()
            ev = []
            for m in model.module_list:
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                b = m(b)
                e1.record()
                ev.append((type(m).__name__, e0, e1))
            torch.cuda.synchronize()
            for name, e0, e1 in ev:
                stage.setdefault(name, []).append(e0.elapsed_time(e1))
    return {
        "points_in_range": int(pts.shape[0]), "pillars": int(v.shape[0]),
        "points_dropped_by_pillar_cap": int(pts.shape[0] - n.sum()),
        "voxelize_cpu_ms": vox_ms, "model_ms_median": statistics.median(total),
        "stage_ms_median": {k: statistics.median(x) for k, x in stage.items()},
        "peak_allocated_above_inputs_bytes": int(peak),
    }


def floats(s):
    return [float(x) for x in s.split(",")]


def ints(s):
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openpcdet", default="third_party/OpenPCDet")
    ap.add_argument("--cfg", default="third_party/OpenPCDet/tools/cfgs/waymo_models/centerpoint_pillar_1x.yaml")
    ap.add_argument("--clutter", type=floats, default=floats("0,1,4"))
    ap.add_argument("--azimuth", type=ints, default=ints("1325,2650,5300"))
    ap.add_argument("--buildings", type=ints, default=ints("0,10"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--p-max", type=int, default=None, help="static pillar count; default = configured voxel cap")
    ap.add_argument("--export-input", default="results/scans/scan_seed7.npy")
    ap.add_argument("--onnx-out", default="artifacts/onnx/centerpoint_pillar_static.onnx")
    ap.add_argument("--report", default="results/onnx/pillar_variant.json")
    ap.add_argument("--ort-reps", type=int, default=5)
    args = ap.parse_args()

    import onnx
    import onnxruntime as ort

    wl = Workload(args.openpcdet, args.cfg)
    p_max = args.p_max or int(wl.voxel_cfg.MAX_NUMBER_OF_VOXELS["test"])
    torch.manual_seed(7)
    model = wl.build_model().cuda().eval()

    # A. stage timing
    rows = []
    for b in args.buildings:
        for label, raw in scans(args, b):
            r = {"label": f"{label} b={b}", **stage_timing(wl, model, raw, args.seed, args.warmup, args.iterations)}
            rows.append(r)
            s = r["stage_ms_median"]
            print(f"{r['label']:30} N={r['points_in_range']:7d} P={r['pillars']:6d} "
                  f"dropped={100 * r['points_dropped_by_pillar_cap'] / r['points_in_range']:4.1f}% "
                  f"vox={r['voxelize_cpu_ms']:5.1f}ms model={r['model_ms_median']:6.1f}ms "
                  + " ".join(f"{k}={v:.1f}" for k, v in s.items()), flush=True)

    # B. static export of the whole network after the host voxelizer
    pts = wl.mask_and_shuffle(np.load(args.export_input).astype(np.float32), 7)
    v, c, n = wl.voxelize(pts)
    ref_batch = wl.to_gpu(wl.collate(v, c, n))
    net = StaticPillarNet(model, wl.grid_size, p_max).eval()
    vp, np_, cp = pad(v, n, c, p_max)
    feeds = {"voxels": vp, "voxel_num_points": np_, "voxel_coords": cp}
    tin = tuple(torch.from_numpy(x).cuda() for x in (vp, np_, cp))
    with torch.inference_mode():
        b = model.vfe(ref_batch)
        b = model.map_to_bev_module(b)
        b = model.backbone_2d(b)
        ref = model.dense_head.heads_list[0](model.dense_head.shared_conv(b["spatial_features_2d"]))
        ref = [ref[k].float().cpu().numpy() for k in net.output_names]
        got = [t.float().cpu().numpy() for t in net(*tin)]
    rewrite_diff = max(float(np.abs(a - r).max() / max(np.abs(r).max(), 1e-12)) for a, r in zip(got, ref))
    print(f"static rewrite vs OpenPCDet modules (GPU): max rel diff {rewrite_diff:.2e}")

    net_cpu = net.cpu()
    tin_cpu = tuple(torch.from_numpy(x) for x in (vp, np_, cp))
    out = Path(args.onnx_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw_path = out.with_name(out.stem + ".raw.onnx")
    torch.onnx.export(net_cpu, tin_cpu, str(raw_path), dynamo=False, opset_version=17,
                      input_names=list(feeds), output_names=[f"head_{k}" for k in net.output_names],
                      do_constant_folding=True)
    raw_model, _, _, raw_dynamic = static_report(onnx, raw_path)
    folded, folded_ops = fold_constants(raw_model)
    onnx.save(folded, str(out))
    m, inferred, infos, dynamic = static_report(onnx, out)
    shapes = {i["name"]: i["shape"] for i in infos}
    convs = conv_macs(inferred, shapes)
    conv_total = sum(x["macs"] for x in convs)
    mm_total = matmul_macs(inferred, shapes)

    # onnxruntime 1.20.1's MatmulBNFusion rewrite rule turns PillarVFE's rank-3 MatMul + BatchNormalization
    # into a Gemm, which requires rank-2 input, and the session fails to load. The model itself is valid
    # ONNX (checker passes). The rule cannot be disabled on its own, so the rule-based transformers that
    # host it are disabled; every other ORT optimization stays on.
    disabled = ["Level1_RuleBasedTransformer", "Level2_RuleBasedTransformer"]
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"], disabled_optimizers=disabled)
    ort_out = sess.run(None, feeds)
    ort_diff = max(float(np.abs(a - r).max() / max(np.abs(r).max(), 1e-12)) for a, r in zip(ort_out, ref))
    times = []
    for _ in range(args.ort_reps):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        times.append((time.perf_counter() - t0) * 1e3)

    report = {
        "valid_for_baseline": False,
        "config": {"path": args.cfg, "sha256": sha256(args.cfg), "grid_xyz": [int(x) for x in wl.grid_size],
                   "pillar_size": list(wl.voxel_cfg.VOXEL_SIZE), "max_points_per_pillar": wl.max_points_per_voxel,
                   "max_pillars": int(wl.voxel_cfg.MAX_NUMBER_OF_VOXELS["test"])},
        "gpu": torch.cuda.get_device_name(0), "cpu": cpu_model(),
        "stage_timing": rows,
        "export": {
            "path": str(out), "sha256": sha256(out), "raw_path": str(raw_path), "raw_sha256": sha256(raw_path),
            "p_max": p_max, "export_input": args.export_input, "export_input_pillars": int(v.shape[0]),
            "inputs": [i for i in infos[:3]], "outputs": infos[3:3 + len(net.output_names)],
            "raw_non_static_tensors": len(raw_dynamic), "folded_nodes": folded_ops,
            "non_static_tensors": dynamic,
            "op_histogram": dict(Counter(nd.op_type for nd in m.graph.node).most_common()),
            "parameters": int(sum(np.prod(i.dims) for i in m.graph.initializer)),
            "file_bytes": out.stat().st_size,
            "constant_node_bytes": int(sum(onnx.numpy_helper.to_array(a.t).nbytes for nd in m.graph.node
                                           if nd.op_type == "Constant" for a in nd.attribute if a.name == "value")),
            "conv_gmac": conv_total / 1e9, "matmul_gmac_at_p_max": mm_total / 1e9,
            "largest_tensors": sorted((i for i in infos if i["logical_bytes"]), key=lambda i: -i["logical_bytes"])[:5],
            "rewrite_vs_openpcdet_max_rel_diff": rewrite_diff, "ort_vs_openpcdet_max_rel_diff": ort_diff,
            "onnxruntime_cpu_ms_median": statistics.median(times), "onnxruntime_cpu_ms_min": min(times),
            "ort_reps": args.ort_reps, "ort_disabled_optimizers": disabled,
        },
    }
    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2) + "\n")
    e = report["export"]
    print(f"wrote {out} and {rp}")
    print(f"P_MAX={p_max}; raw non-static {e['raw_non_static_tensors']}; folded {folded_ops}; non-static after fold: {len(dynamic)}")
    print("ops:", e["op_histogram"])
    print(f"conv {e['conv_gmac']:.1f} GMAC, matmul {e['matmul_gmac_at_p_max']:.2f} GMAC at P_MAX; params {e['parameters']:,}")
    print(f"ort vs openpcdet {ort_diff:.2e}; ort CPU median {e['onnxruntime_cpu_ms_median']:.0f} ms")
    if dynamic or rewrite_diff > 1e-4 or ort_diff > 1e-4:
        raise SystemExit("export check failed")


if __name__ == "__main__":
    main()

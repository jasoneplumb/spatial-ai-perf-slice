#!/usr/bin/env python3
"""Export the static dense segment of CenterPoint to ONNX and inventory it.

Segment: spatial_features [1, 256, 188, 188] (HeightCompression output)
-> BaseBEVBackbone -> CenterHead shared conv + separate heads
-> raw head outputs (hm, center, center_z, dim, rot), each [1, C, 188, 188].

This is the part of the graph with a fully static contract. Voxelization, the
spconv sparse backbone, the sparse-to-dense scatter, decode and rotated NMS are
outside it (see docs/WORKLOAD-CONTRACT.md).

Weights are random: the graph structure, shapes, operator set and MAC counts
are valid; outputs are not meaningful detections and the model must not be
used for INT8 calibration or accuracy work.

Validation:
  1. onnx.checker and strict shape inference; every value_info shape must be
     fully static.
  2. onnxruntime (CPU) vs PyTorch on a realistic input: spatial_features
     produced by the real sparse path from a parametric scan.
"""
import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from pcdet_harness import Workload

ONNX_DTYPES = {1: "FP32", 6: "INT32", 7: "INT64", 10: "FP16"}
BYTES = {"FP32": 4, "INT32": 4, "INT64": 8, "FP16": 2}


class DenseSegment(nn.Module):
    def __init__(self, backbone_2d, dense_head):
        super().__init__()
        self.backbone_2d = backbone_2d
        self.shared_conv = dense_head.shared_conv
        self.head = dense_head.heads_list[0]
        self.output_names = list(self.head.sep_head_dict)

    def forward(self, spatial_features):
        x = self.backbone_2d({"spatial_features": spatial_features})["spatial_features_2d"]
        out = self.head(self.shared_conv(x))
        return tuple(out[k] for k in self.output_names)


def fold_constants(model):
    """Evaluate every node whose inputs are all constants and replace it with an initializer.

    The legacy exporter emits nn.ZeroPad2d as Pad whose `pads` input is computed
    by a chain of constant ops; shape inference cannot see through it. Uses the
    onnx reference evaluator, so no extra dependency is needed.
    """
    import onnx
    from onnx import numpy_helper
    from onnx.reference import ReferenceEvaluator

    g = model.graph
    const = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    kept, folded = [], Counter()
    for node in g.node:
        if all(i == "" or i in const for i in node.input):
            outs = ReferenceEvaluator(node).run(None, {i: const[i] for i in node.input if i})
            for name, value in zip(node.output, outs):
                const[name] = np.asarray(value)
            folded[node.op_type] += 1
        else:
            kept.append(node)
    used = {i for n in kept for i in n.input} | {o.name for o in g.output}
    new = onnx.helper.make_graph(
        kept, g.name, list(g.input), list(g.output),
        [numpy_helper.from_array(v, k) for k, v in const.items() if k in used])
    out = onnx.helper.make_model(new, opset_imports=model.opset_import, producer_name=model.producer_name,
                                 producer_version=model.producer_version)
    out.ir_version = model.ir_version
    return out, dict(folded)


def static_report(onnx, path):
    m = onnx.load(str(path))
    onnx.checker.check_model(m, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(m, check_type=True, strict_mode=True, data_prop=True)
    infos = [tensor_info(v) for v in list(inferred.graph.input) + list(inferred.graph.output)
             + list(inferred.graph.value_info)]
    return m, inferred, infos, [v["name"] for v in infos if v["logical_bytes"] is None]


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_info(vi):
    t = vi.type.tensor_type
    dims = [d.dim_value if d.HasField("dim_value") and d.dim_value > 0 else None for d in t.shape.dim]
    dtype = ONNX_DTYPES.get(t.elem_type, str(t.elem_type))
    static = all(d is not None for d in dims)
    return {"name": vi.name, "shape": dims, "dtype": dtype,
            "logical_bytes": int(np.prod(dims)) * BYTES[dtype] if static and dtype in BYTES else None}


def conv_macs(model, shapes):
    """MACs per Conv/ConvTranspose from static shapes and weight initializers."""
    inits = {i.name: list(i.dims) for i in model.graph.initializer}
    rows = []
    for node in model.graph.node:
        if node.op_type not in ("Conv", "ConvTranspose"):
            continue
        w = inits[node.input[1]]
        group = next((a.i for a in node.attribute if a.name == "group"), 1)
        if node.op_type == "Conv":
            # out elems x (Cin/group x kH x kW); weight is [Cout, Cin/group, kH, kW]
            macs = int(np.prod(shapes[node.output[0]])) * w[1] * w[2] * w[3]
        else:
            # every input element scatters into Cout/group x kH x kW outputs; weight is [Cin, Cout/group, kH, kW]
            macs = int(np.prod(shapes[node.input[0]])) * w[1] * w[2] * w[3]
        rows.append({"node": node.name, "op": node.op_type, "weight": w, "group": group,
                     "input": shapes[node.input[0]], "output": shapes[node.output[0]], "macs": macs})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openpcdet", default="third_party/OpenPCDet")
    ap.add_argument("--cfg", default="third_party/OpenPCDet/tools/cfgs/waymo_models/centerpoint.yaml")
    ap.add_argument("--points-npy", default="results/scans/scan_seed7.npy",
                    help="parametric scan used to produce a realistic spatial_features input")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--onnx-out", default="artifacts/onnx/centerpoint_dense.onnx")
    ap.add_argument("--report", default="results/onnx/dense_export.json")
    ap.add_argument("--ort-reps", type=int, default=10)
    args = ap.parse_args()

    import onnx
    import onnxruntime as ort

    wl = Workload(args.openpcdet, args.cfg)
    torch.manual_seed(args.seed)
    model = wl.build_model().eval()
    segment = DenseSegment(model.backbone_2d, model.dense_head).eval()

    # Realistic input: run the real sparse path on the scan up to HeightCompression.
    model.cuda()
    points = wl.mask_and_shuffle(np.load(args.points_npy).astype(np.float32), args.seed)
    batch = wl.to_gpu(wl.collate(*wl.voxelize(points)))
    with torch.inference_mode():
        for m in model.module_list:
            batch = m(batch)
            if type(m).__name__ == "HeightCompression":
                break
    x = batch["spatial_features"].detach().clone()
    occupied = float((x.abs().sum(dim=1) > 0).float().mean())

    with torch.inference_mode():
        ref_gpu = [t.float().cpu().numpy() for t in segment(x)]
    segment.cpu()
    x_cpu = x.cpu()
    with torch.inference_mode():
        ref_cpu = [t.numpy() for t in segment(x_cpu)]

    out = Path(args.onnx_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_name(out.stem + ".raw.onnx")
    torch.onnx.export(segment, (x_cpu,), str(raw), dynamo=False, opset_version=args.opset,
                      input_names=["spatial_features"], output_names=[f"head_{n}" for n in segment.output_names],
                      do_constant_folding=True)
    raw_model, _, _, raw_dynamic = static_report(onnx, raw)
    raw_ops = dict(Counter(n.op_type for n in raw_model.graph.node).most_common())

    folded_model, folded_ops = fold_constants(raw_model)
    onnx.save(folded_model, str(out))
    m, inferred, infos, dynamic = static_report(onnx, out)
    inputs, outputs = infos[:len(m.graph.input)], infos[len(m.graph.input):len(m.graph.input) + len(m.graph.output)]
    values = infos[len(m.graph.input) + len(m.graph.output):]
    shapes = {v["name"]: v["shape"] for v in inputs + outputs + values}
    convs = conv_macs(inferred, shapes)
    n_params = int(sum(np.prod(i.dims) for i in m.graph.initializer))

    so = ort.SessionOptions()
    sess = ort.InferenceSession(str(out), so, providers=["CPUExecutionProvider"])
    feed = {"spatial_features": x_cpu.numpy()}
    got = sess.run(None, feed)
    raw_got = ort.InferenceSession(str(raw), providers=["CPUExecutionProvider"]).run(None, feed)
    folded_identical = all(np.array_equal(a, b) for a, b in zip(got, raw_got))
    times = []
    for _ in range(args.ort_reps):
        t0 = time.perf_counter()
        sess.run(None, feed)
        times.append((time.perf_counter() - t0) * 1e3)

    def diff(a, b):
        scale = float(np.abs(b).max()) or 1.0
        return {"max_abs": float(np.abs(a - b).max()), "max_abs_over_ref_max": float(np.abs(a - b).max()) / scale,
                "ref_max_abs": scale}

    numerics = {name: {"ort_vs_torch_cpu": diff(g, rc), "ort_vs_torch_cuda": diff(g, rg)}
                for name, g, rc, rg in zip(segment.output_names, got, ref_cpu, ref_gpu)}
    contract = {b["id"]: b for b in json.loads(Path("results/contract/centerpoint_waymo.json").read_text())["boundaries"]}
    contract_ok = (inputs[0]["shape"] == contract["spatial_features"]["shape"]
                   and all(o["shape"] == contract[o["name"]]["shape"] for o in outputs))

    report = {
        "valid_for_baseline": False,
        "weights": f"randomly initialized (torch.manual_seed({args.seed}))",
        "onnx": {"path": str(out), "sha256": sha256(out), "bytes": out.stat().st_size, "opset": args.opset,
                 "exporter": f"torch.onnx.export(dynamo=False), torch {torch.__version__}",
                 "onnx_version": onnx.__version__, "parameters": n_params},
        "raw_export": {"path": str(raw), "sha256": sha256(raw), "op_histogram": raw_ops,
                       "non_static_tensor_count": len(raw_dynamic),
                       "note": "nn.ZeroPad2d exports as Pad with runtime-computed pads; shape inference "
                               "cannot resolve downstream shapes until the constant chain is folded"},
        "constant_folding": {"folded_nodes": folded_ops, "outputs_bitwise_identical_to_raw": folded_identical},
        "segment": "spatial_features -> BaseBEVBackbone -> CenterHead convolutions -> raw head outputs",
        "inputs": inputs,
        "outputs": outputs,
        "matches_contract": contract_ok,
        "value_info_count": len(values),
        "non_static_tensors": dynamic,
        "op_histogram": dict(Counter(n.op_type for n in m.graph.node).most_common()),
        "conv_macs_total": int(sum(c["macs"] for c in convs)),
        "convs": convs,
        "intermediate_tensors": values,
        "input_sample": {"source": args.points_npy, "bev_cells_nonzero_fraction": occupied},
        "numerics": numerics,
        "onnxruntime_cpu": {"version": ort.__version__, "providers": sess.get_providers(),
                            "intra_op_threads_default": so.intra_op_num_threads, "reps": args.ort_reps,
                            "ms_median": statistics.median(times), "ms_min": min(times),
                            "cpu": cpu_model(), "logical_cpus": os.cpu_count()},
    }
    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2) + "\n")

    print(f"wrote {out} ({out.stat().st_size:,} bytes, {n_params:,} params) and {rp}")
    print(f"raw export: {len(raw_dynamic)} non-static tensors; ops {raw_ops}")
    print(f"folded: {folded_ops}; outputs bitwise identical to raw: {folded_identical}")
    print("inputs ", [(i["name"], i["shape"]) for i in inputs])
    print("outputs", [(o["name"], o["shape"]) for o in outputs])
    print("matches contract:", contract_ok, "| value_info:", len(values), "| non-static:", dynamic)
    print("ops:", report["op_histogram"])
    print(f"conv MACs: {report['conv_macs_total'] / 1e9:.2f} G over {len(convs)} conv nodes")
    for k, v in numerics.items():
        print(f"  {k:9} ort-vs-cpu {v['ort_vs_torch_cpu']['max_abs_over_ref_max']:.2e}  "
              f"ort-vs-cuda {v['ort_vs_torch_cuda']['max_abs_over_ref_max']:.2e}  (ref max {v['ort_vs_torch_cpu']['ref_max_abs']:.3g})")
    print(f"onnxruntime CPU median {report['onnxruntime_cpu']['ms_median']:.1f} ms over {args.ort_reps} runs")
    if dynamic or not contract_ok or not folded_identical:
        raise SystemExit("export check failed")


if __name__ == "__main__":
    main()

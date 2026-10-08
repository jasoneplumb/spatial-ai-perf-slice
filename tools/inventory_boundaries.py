#!/usr/bin/env python3
"""Phase 1 boundary inventory: observe every stage-boundary tensor.

Runs the pinned CenterPoint graph once, stage by stage, and records for each
tensor crossing a stage boundary its shape, dtype, device, strides,
contiguity, logical bytes and backing-storage bytes, plus CUDA allocator
deltas per stage. Observed shapes are checked against the static contract
from tools/derive_contract.py.

Weights are random: tensor shapes, sparse active sites and allocations are
valid; detection counts are not representative. Byte and allocation figures
describe this PyTorch/CUDA reference implementation on this host, not any
other target.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import torch


def load_openpcdet_cfg(root: Path, cfg_path: Path):
    from pcdet.config import cfg, cfg_from_yaml_file

    old_cwd = Path.cwd()
    try:
        os.chdir(root / "tools")
        cfg_from_yaml_file(str(cfg_path), cfg)
    finally:
        os.chdir(old_cwd)
    return cfg


def describe(x):
    """Describe a numpy array or torch tensor at a boundary."""
    if isinstance(x, np.ndarray):
        return {
            "kind": "numpy", "device": "host", "shape": list(x.shape), "dtype": str(x.dtype),
            "strides_bytes": list(x.strides), "contiguous": bool(x.flags["C_CONTIGUOUS"]),
            "logical_bytes": int(x.nbytes), "storage_bytes": int(x.base.nbytes if x.base is not None else x.nbytes),
        }
    return {
        "kind": "torch", "device": str(x.device), "shape": list(x.shape), "dtype": str(x.dtype).replace("torch.", ""),
        "strides_elems": list(x.stride()), "contiguous": bool(x.is_contiguous()),
        "logical_bytes": int(x.numel() * x.element_size()),
        "storage_bytes": int(x.untyped_storage().nbytes()),
        "storage_offset": int(x.storage_offset()),
        "_ptr": int(x.untyped_storage().data_ptr()),
    }


class Allocator:
    """Allocated-bytes delta and peak above start for one stage."""

    def __enter__(self):
        torch.cuda.synchronize()
        self.start = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *exc):
        torch.cuda.synchronize()
        self.delta = torch.cuda.memory_allocated() - self.start
        self.peak_above_start = torch.cuda.max_memory_allocated() - self.start
        return False

    def record(self):
        return {"allocated_delta_bytes": int(self.delta), "peak_above_start_bytes": int(self.peak_above_start)}


def sha256_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# Observed name -> contract boundary id (tools/derive_contract.py).
CONTRACT_MAP = {
    "points": "points",
    "voxels@cuda": "voxels",
    "voxel_num_points@cuda": "voxel_num_points",
    "voxel_coords@cuda": "voxel_coords",
    "voxel_features": "voxel_features",
    "encoded_spconv_tensor.features": "encoded_sparse_features",
    "encoded_spconv_tensor.indices": "encoded_sparse_indices",
    "spatial_features": "spatial_features",
    "spatial_features_2d": "spatial_features_2d",
    "head.hm": "head_hm",
    "head.center": "head_center",
    "head.center_z": "head_center_z",
    "head.dim": "head_dim",
    "head.rot": "head_rot",
    "nms_in.box_preds": "candidate_boxes",
    "nms_in.box_scores": "candidate_scores",
    "pred_boxes": "pred_boxes",
    "pred_scores": "pred_scores",
    "pred_labels": "pred_labels",
}
DTYPE_NAME = {"float32": "FP32", "int32": "INT32", "int64": "INT64"}


def check_contract(observed, contract):
    by_id = {b["id"]: b for b in contract["boundaries"]}
    checks = []
    for name, cid in CONTRACT_MAP.items():
        b = by_id[cid]
        obs = observed.get(name)
        if obs is None:
            checks.append({"name": name, "contract_id": cid, "status": "not_observed"})
            continue
        problems = []
        if len(obs["shape"]) != len(b["shape"]):
            problems.append(f"rank {len(obs['shape'])} != {len(b['shape'])}")
        else:
            for i, (o, c) in enumerate(zip(obs["shape"], b["shape"])):
                if c is not None and o != c:
                    problems.append(f"dim {i}: {o} != {c}")
            ub = b.get("upper_bound_shape")
            if ub:
                for i, (o, u) in enumerate(zip(obs["shape"], ub)):
                    if o > u:
                        problems.append(f"dim {i}: {o} exceeds bound {u}")
        if DTYPE_NAME.get(obs["dtype"]) != b["dtype_exact"]:
            problems.append(f"dtype {obs['dtype']} != {b['dtype_exact']}")
        resolved = {i: o for i, (o, c) in enumerate(zip(obs["shape"], b["shape"])) if c is None}
        checks.append({"name": name, "contract_id": cid, "status": "mismatch" if problems else "ok",
                       "problems": problems, "frame_dependent_dims": resolved})
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openpcdet", default="third_party/OpenPCDet")
    ap.add_argument("--cfg", default="third_party/OpenPCDet/tools/cfgs/waymo_models/centerpoint.yaml")
    ap.add_argument("--contract", default="results/contract/centerpoint_waymo.json")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--points-npy", help="model-ready N x 5 float32 points (e.g. from tools/scan_generator.py)")
    src.add_argument("--legacy-synthetic", type=int, metavar="N",
                     help="use smoke_centerpoint.synthetic_points(N, seed)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", default="results/inventory")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    root = Path(args.openpcdet).resolve()
    cfg_path = Path(args.cfg).resolve()
    sys.path.insert(0, str(root))

    # Same import shim as tools/smoke_centerpoint.py: avoid executing
    # pcdet/datasets/__init__.py and its optional dataset backends.
    datasets_pkg = ModuleType("pcdet.datasets")
    datasets_pkg.__path__ = [str(root / "pcdet" / "datasets")]
    datasets_pkg.__package__ = "pcdet.datasets"
    sys.modules.setdefault("pcdet.datasets", datasets_pkg)

    import cumm.tensorview as tv
    from spconv.utils import Point2VoxelCPU3d
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.models.model_utils import model_nms_utils

    cfg = load_openpcdet_cfg(root, cfg_path)
    pc_range = np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE, dtype=np.float32)
    voxel_cfg = next(x for x in cfg.DATA_CONFIG.DATA_PROCESSOR if x.NAME == "transform_points_to_voxels")
    voxel_size = np.asarray(voxel_cfg.VOXEL_SIZE, dtype=np.float32)
    grid_size = np.round((pc_range[3:6] - pc_range[0:3]) / voxel_size).astype(np.int64)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.points_npy:
        raw = np.load(args.points_npy).astype(np.float32, copy=False)
        if raw.ndim != 2 or raw.shape[1] != 5:
            raise SystemExit(f"--points-npy must be N x 5; got {raw.shape}")
        input_desc = {"source": str(Path(args.points_npy)), "kind": "points_npy"}
        sidecar = Path(args.points_npy).with_suffix(".json")
        if sidecar.exists():
            input_desc["generator_metadata"] = json.loads(sidecar.read_text())
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from smoke_centerpoint import synthetic_points
        raw = synthetic_points(args.legacy_synthetic, args.seed)
        input_desc = {"source": f"smoke_centerpoint.synthetic_points({args.legacy_synthetic}, {args.seed})",
                      "kind": "legacy_synthetic"}
    input_desc["raw_points"] = int(raw.shape[0])
    input_desc["raw_sha256"] = sha256_array(raw)

    # Dataset path, outside model(): range mask, test-time shuffle (seeded), voxelize, collate.
    mask = np.all((raw[:, :3] >= pc_range[:3]) & (raw[:, :3] < pc_range[3:6]), axis=1)
    points = raw[mask]
    if next(x for x in cfg.DATA_CONFIG.DATA_PROCESSOR if x.NAME == "shuffle_points").SHUFFLE_ENABLED["test"]:
        points = points[np.random.permutation(points.shape[0])]

    voxelizer = Point2VoxelCPU3d(
        vsize_xyz=voxel_size.tolist(), coors_range_xyz=pc_range.tolist(), num_point_features=5,
        max_num_points_per_voxel=int(voxel_cfg.MAX_POINTS_PER_VOXEL),
        max_num_voxels=int(voxel_cfg.MAX_NUMBER_OF_VOXELS["test"]),
    )
    tv_voxels, tv_coords, tv_num = voxelizer.point_to_voxel(tv.from_numpy(points))
    voxels, coords, num_points = tv_voxels.numpy().copy(), tv_coords.numpy().copy(), tv_num.numpy().copy()
    voxel_coords = np.pad(coords, ((0, 0), (1, 0)), constant_values=0)

    observed = {"points": describe(points)}
    host = {"voxels": voxels, "voxel_num_points": num_points, "voxel_coords": voxel_coords}
    observed.update({k: describe(v) for k, v in host.items()})
    stages = [
        {"stage": "frame_load", "inside_model": False, "outputs": ["points"]},
        {"stage": "voxelize", "inside_model": False, "outputs": list(host)},
    ]

    dataset = SimpleNamespace(
        class_names=cfg.CLASS_NAMES, point_cloud_range=pc_range, voxel_size=voxel_size,
        grid_size=grid_size, depth_downsample_factor=None,
        point_feature_encoder=SimpleNamespace(num_point_features=5),
    )
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset).cuda().eval()

    batch = {**host, "batch_size": 1}
    with Allocator() as a:
        load_data_to_gpu(batch)
    stages.append({"stage": "host_to_device", "inside_model": False,
                   "outputs": [f"{k}@cuda" for k in host], **a.record()})
    for k in host:
        observed[f"{k}@cuda"] = describe(batch[k])

    # Sparse-backbone levels: active sites per level via forward hooks.
    sparse = next(m for m in model.module_list if type(m).__name__ == "VoxelResBackBone8x")
    levels = {}
    for name in ("conv_input", "conv1", "conv2", "conv3", "conv4", "conv_out"):
        def hook(_m, _i, out, name=name):
            levels[name] = {"active_sites": int(out.features.shape[0]), "channels": int(out.features.shape[1]),
                            "spatial_shape_zyx": [int(s) for s in out.spatial_shape]}
        getattr(sparse, name).register_forward_hook(hook)

    # Separate the dense head's convolutions from its decode/NMS, and capture NMS inputs.
    head = model.dense_head
    original_decode = head.generate_predicted_boxes
    decode_alloc = {}

    def wrapped_decode(batch_size, pred_dicts):
        for k, v in pred_dicts[0].items():
            observed[f"head.{k}"] = describe(v)
        with Allocator() as a:
            out = original_decode(batch_size, pred_dicts)
        decode_alloc.update(a.record())
        return out

    head.generate_predicted_boxes = wrapped_decode
    original_nms = model_nms_utils.class_agnostic_nms

    def wrapped_nms(box_scores, box_preds, nms_config, score_thresh=None):
        observed["nms_in.box_scores"] = describe(box_scores)
        observed["nms_in.box_preds"] = describe(box_preds)
        return original_nms(box_scores, box_preds, nms_config, score_thresh)

    model_nms_utils.class_agnostic_nms = wrapped_nms

    with torch.inference_mode():
        for module in model.module_list:
            name = type(module).__name__
            before = {k: (id(v), v) for k, v in batch.items()}
            with Allocator() as a:
                batch = module(batch)
            outputs = []
            for k, v in batch.items():
                if k in before and before[k][0] == id(v):
                    continue
                if torch.is_tensor(v):
                    observed[k] = describe(v)
                    outputs.append(k)
                elif type(v).__name__ == "SparseConvTensor":
                    observed[f"{k}.features"] = describe(v.features)
                    observed[f"{k}.indices"] = describe(v.indices)
                    outputs += [f"{k}.features", f"{k}.indices"]
            if name == "CenterHead":
                outputs = [k for k in observed if k.startswith("head.") or k.startswith("nms_in.")]
                for k in ("pred_boxes", "pred_scores", "pred_labels"):
                    observed[k] = describe(batch["final_box_dicts"][0][k])
                    outputs.append(k)
            stages.append({"stage": name, "inside_model": True, "outputs": outputs, **a.record()})
            if name == "CenterHead":
                stages[-1]["of_which_decode_and_nms"] = dict(decode_alloc)
    model_nms_utils.class_agnostic_nms = original_nms

    # Aliasing: tensors sharing one storage (views, not copies).
    by_ptr = {}
    for k, v in observed.items():
        if "_ptr" in v:
            by_ptr.setdefault(v["_ptr"], []).append(k)
    for k, v in observed.items():
        ptr = v.pop("_ptr", None)
        if ptr is not None and len(by_ptr[ptr]) > 1:
            v["shares_storage_with"] = [o for o in by_ptr[ptr] if o != k]

    contract = json.loads(Path(args.contract).read_text())
    checks = check_contract(observed, contract)

    occ = np.bincount(num_points, minlength=int(voxel_cfg.MAX_POINTS_PER_VOXEL) + 1)[1:]
    V = int(voxels.shape[0])
    summary = {
        "points_in_range": int(points.shape[0]),
        "voxels": V,
        "points_per_voxel_hist_1_to_max": occ.tolist(),
        "points_dropped_by_voxel_cap": int(points.shape[0] - num_points.sum()),
        "sparse_levels": levels,
        "active_site_ratio_to_voxels": {k: round(v["active_sites"] / V, 3) for k, v in levels.items()},
        "bev_cells_occupied_fraction": round(levels["conv_out"]["active_sites"]
                                             / np.prod(levels["conv_out"]["spatial_shape_zyx"]), 4),
        "detections_random_weights": observed["pred_boxes"]["shape"][0],
    }

    result = {
        "valid_for_baseline": False,
        "evidence_grade": "measured (reference implementation on this host; random weights)",
        "label": args.label,
        "input": input_desc,
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "openpcdet_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
            "contract": args.contract,
        },
        "summary": summary,
        "stages": stages,
        "tensors": observed,
        "contract_checks": checks,
    }
    out = Path(args.out_dir) / f"{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    print(f"wrote {out}")
    print(json.dumps(summary, indent=1))
    print(f"\n{'tensor':34} {'shape':24} {'dtype':8} {'logical':>12} {'storage':>12} contig")
    for k, v in observed.items():
        print(f"{k:34} {str(v['shape']):24} {v['dtype']:8} {v['logical_bytes']:12,d} {v['storage_bytes']:12,d} "
              f"{v['contiguous']}{'  shares:' + ','.join(v['shares_storage_with']) if 'shares_storage_with' in v else ''}")
    print(f"\n{'stage':20} {'alloc delta':>14} {'peak above start':>17}")
    for s in stages:
        if "allocated_delta_bytes" in s:
            print(f"{s['stage']:20} {s['allocated_delta_bytes']:14,d} {s['peak_above_start_bytes']:17,d}")
    bad = [c for c in checks if c["status"] != "ok"]
    print(f"\ncontract checks: {len(checks) - len(bad)} ok, {len(bad)} not ok")
    for c in bad:
        print("  ", c)
    sys.exit(1 if any(c["status"] == "mismatch" for c in bad) else 0)


if __name__ == "__main__":
    main()

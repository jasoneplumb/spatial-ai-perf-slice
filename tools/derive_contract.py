#!/usr/bin/env python3
"""Derive the static input/output contract of the pinned CenterPoint workload.

Every shape here follows from the pinned OpenPCDet config and module source.
Dimensions that depend on the input frame (point count, voxel count, active
sparse sites, detections) are recorded as null rather than estimated.

The output contains no execution-domain or memory-placement claims: those are
recorded as "unknown" until a partition report or measurement exists.
"""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

BITS = {"FP32": 32, "INT32": 32, "INT64": 64}

# Element types allowed by the portable boundary schema. Integer index tensors
# (INT32/INT64) are outside that enum and are emitted as "unknown" there, with
# the exact type kept in `dtype_exact`.
PORTABLE_DTYPES = {"FP32", "FP16", "BF16", "INT16", "INT8", "UINT8", "INT4"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_openpcdet_cfg(root: Path, cfg_path: Path):
    from pcdet.config import cfg, cfg_from_yaml_file

    # OpenPCDet resolves _BASE_CONFIG_ relative to its tools directory.
    old_cwd = Path.cwd()
    try:
        os.chdir(root / "tools")
        cfg_from_yaml_file(str(cfg_path), cfg)
    finally:
        os.chdir(old_cwd)
    return cfg


def conv_out(n, k, s, p):
    """Output extent of a strided (non-submanifold) sparse or dense convolution."""
    return (n + 2 * p - (k - 1) - 1) // s + 1


def logical_bytes(shape, dtype):
    if not shape or any(d is None for d in shape) or dtype not in BITS:
        return None
    count = 1
    for d in shape:
        count *= d
    return (count * BITS[dtype] + 7) // 8


def probe_voxel_generator(cfg):
    """Observe output dtypes of the voxelizer the pinned dataset path uses."""
    import cumm.tensorview as tv
    from spconv.utils import Point2VoxelCPU3d

    proc = next(p for p in cfg.DATA_CONFIG.DATA_PROCESSOR if p.NAME == "transform_points_to_voxels")
    gen = Point2VoxelCPU3d(
        vsize_xyz=proc.VOXEL_SIZE,
        coors_range_xyz=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
        num_point_features=len(cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.used_feature_list),
        max_num_points_per_voxel=proc.MAX_POINTS_PER_VOXEL,
        max_num_voxels=proc.MAX_NUMBER_OF_VOXELS["test"],
    )
    # Two points in one voxel and one in another; values are irrelevant except
    # that they lie inside the configured range.
    pts = np.array([[0.01, 0.01, 0.01, 0.5, 0.1],
                    [0.02, 0.02, 0.02, 0.5, 0.1],
                    [5.0, 5.0, 1.0, 0.5, 0.1]], dtype=np.float32)
    voxels, coords, num = gen.point_to_voxel(tv.from_numpy(pts))
    voxels, coords, num = voxels.numpy(), coords.numpy(), num.numpy()
    return {
        "voxels": {"dtype": str(voxels.dtype), "trailing_shape": list(voxels.shape[1:])},
        "voxel_coords": {"dtype": str(coords.dtype), "trailing_shape": list(coords.shape[1:])},
        "voxel_num_points": {"dtype": str(num.dtype), "trailing_shape": list(num.shape[1:])},
        "voxel_count_for_probe": int(voxels.shape[0]),
    }


NP_TO_DTYPE = {"float32": "FP32", "int32": "INT32", "int64": "INT64"}


def derive(cfg, probe):
    data = cfg.DATA_CONFIG
    model = cfg.MODEL
    pc_range = np.array(data.POINT_CLOUD_RANGE, dtype=np.float64)
    proc = next(p for p in data.DATA_PROCESSOR if p.NAME == "transform_points_to_voxels")
    voxel_size = np.array(proc.VOXEL_SIZE, dtype=np.float64)

    # data_processor.py: np.round((range_max - range_min) / voxel_size).astype(int64), order x, y, z
    gx, gy, gz = (int(v) for v in np.round((pc_range[3:6] - pc_range[0:3]) / voxel_size))
    num_point_features = len(data.POINT_FEATURE_ENCODING.used_feature_list)
    max_voxels = int(proc.MAX_NUMBER_OF_VOXELS["test"])
    max_pts_per_voxel = int(proc.MAX_POINTS_PER_VOXEL)

    # spconv_backbone.py VoxelResBackBone8x: sparse_shape = grid_size[::-1] + [1, 0, 0]  (z, y, x)
    z, y, x = gz + 1, gy, gx
    sparse_levels = [{"name": "conv_input/conv1", "shape_zyx": [z, y, x], "channels": 16}]
    # conv2, conv3: SparseConv3d k=3 s=2 p=1 ; conv4: k=3 s=2 p=(0,1,1)
    for name, ch, pz in (("conv2", 32, 1), ("conv3", 64, 1), ("conv4", 128, 0)):
        z, y, x = conv_out(z, 3, 2, pz), conv_out(y, 3, 2, 1), conv_out(x, 3, 2, 1)
        sparse_levels.append({"name": name, "shape_zyx": [z, y, x], "channels": ch})
    # conv_out: SparseConv3d k=(3,1,1) s=(2,1,1) padding=last_pad (default 0)
    last_pad = int(model.BACKBONE_3D.get("last_pad", 0))
    z = conv_out(z, 3, 2, last_pad)
    sparse_levels.append({"name": "conv_out", "shape_zyx": [z, y, x], "channels": 128})
    bev_c, bev_h, bev_w = 128 * z, y, x

    # Consistency with values stated independently in the config.
    assert bev_c == model.MAP_TO_BEV.NUM_BEV_FEATURES, (bev_c, model.MAP_TO_BEV.NUM_BEV_FEATURES)
    stride = model.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.FEATURE_MAP_STRIDE
    assert gx // stride == bev_w and gy // stride == bev_h, (gx, gy, stride, bev_h, bev_w)

    # base_bev_backbone.py: ZeroPad2d(1) + Conv2d(k=3, stride=s, padding=0); deblocks ConvTranspose2d(k=s, stride=s)
    b2d = model.BACKBONE_2D
    h, w = bev_h, bev_w
    bev_levels, ups = [], []
    for nf, s, us, unf in zip(b2d.NUM_FILTERS, b2d.LAYER_STRIDES, b2d.UPSAMPLE_STRIDES, b2d.NUM_UPSAMPLE_FILTERS):
        h, w = conv_out(h + 2, 3, s, 0), conv_out(w + 2, 3, s, 0)
        bev_levels.append({"block_out": [1, nf, h, w], "deblock_out": [1, unf, h * us, w * us]})
        ups.append((unf, h * us, w * us))
    assert len({(u[1], u[2]) for u in ups}) == 1, ups
    bev2d_c = sum(u[0] for u in ups)
    out_h, out_w = ups[0][1], ups[0][2]

    head = model.DENSE_HEAD
    num_class = len(cfg.CLASS_NAMES)
    head_channels = {"hm": num_class}
    head_channels.update({k: v["out_channels"] for k, v in head.SEPARATE_HEAD_CFG.HEAD_DICT.items()})
    box_dim = 7 + (2 if "vel" in head_channels else 0)
    post = head.POST_PROCESSING
    k_top = int(post.MAX_OBJ_PER_SAMPLE)

    vdt = {k: NP_TO_DTYPE[probe[k]["dtype"]] for k in ("voxels", "voxel_coords", "voxel_num_points")}

    tensors = [
        # (id, from, to, shape, dtype, layout, upper_bound_shape, grade, source, notes)
        ("points", "frame_load", "voxelize", [None, num_point_features], "FP32",
         "N x [x, y, z, intensity, elongation], row-major", None,
         "derived", "waymo_utils.py:188-190 astype(float32); waymo_dataset.yaml used_feature_list",
         "N = points remaining after mask_points_and_boxes_outside_range; frame-dependent."),
        ("voxels", "voxelize", "vfe", [None, max_pts_per_voxel, num_point_features], vdt["voxels"],
         "V x P x F, zero-padded beyond voxel_num_points", [max_voxels, max_pts_per_voxel, num_point_features],
         "derived", "waymo_dataset.yaml transform_points_to_voxels; dtype from spconv Point2VoxelCPU3d probe",
         "V <= MAX_NUMBER_OF_VOXELS[test]; frame-dependent."),
        ("voxel_num_points", "voxelize", "vfe", [None], vdt["voxel_num_points"], "V", [max_voxels],
         "derived", "spconv Point2VoxelCPU3d probe", "Per-voxel occupied point count."),
        ("voxel_coords", "voxelize", "sparse_backbone_3d", [None, 4], vdt["voxel_coords"],
         "V x [batch, z, y, x]; batch column prepended by collate_batch", [max_voxels, 4],
         "derived", "dataset.py collate_batch np.pad(coor, ((0,0),(1,0))); probe gives V x 3 before collate",
         "Index tensor; also defines the sparse-convolution rulebook."),
        ("voxel_features", "vfe", "sparse_backbone_3d", [None, num_point_features], "FP32", "V x F (mean of points)",
         [max_voxels, num_point_features], "derived", "mean_vfe.py forward", "Arithmetic mean over occupied points."),
        ("encoded_sparse_features", "sparse_backbone_3d", "height_compression", [None, 128], "FP32",
         f"S x C sparse features over spatial shape z,y,x={sparse_levels[-1]['shape_zyx']}",
         [sparse_levels[-1]["shape_zyx"][0] * bev_h * bev_w, 128],
         "derived", "spconv_backbone.py VoxelResBackBone8x conv_out",
         "S active sites is frame-dependent; upper bound is the full grid."),
        ("encoded_sparse_indices", "sparse_backbone_3d", "height_compression", [None, 4], "INT32",
         "S x [batch, z, y, x]", [sparse_levels[-1]["shape_zyx"][0] * bev_h * bev_w, 4],
         "derived", "spconv_backbone.py voxel_coords.int(); spconv SparseConvTensor indices",
         "Sparse-to-dense scatter happens at this boundary (SparseConvTensor.dense())."),
        ("spatial_features", "height_compression", "bev_backbone_2d", [1, bev_c, bev_h, bev_w], "FP32",
         "NCHW, C = 128 channels x D height slices", None,
         "derived", "height_compression.py dense().view(N, C*D, H, W)",
         "First fully dense, static-shape tensor in the pipeline."),
        ("spatial_features_2d", "bev_backbone_2d", "center_head", [1, bev2d_c, out_h, out_w], "FP32",
         "NCHW, concat of upsampled levels", None, "derived", "base_bev_backbone.py forward torch.cat(ups, dim=1)", ""),
    ]
    for name, ch in head_channels.items():
        tensors.append((f"head_{name}", "center_head", "decode_topk", [1, ch, out_h, out_w], "FP32", "NCHW", None,
                        "derived", "center_head.py SeparateHead + HEAD_DICT/NUM_HM_CONV",
                        "Raw logits; sigmoid applied in decode." if name == "hm" else
                        ("exp() applied in decode." if name == "dim" else "")))
    tensors += [
        ("candidate_boxes", "decode_topk", "nms", [None, box_dim], "FP32", "M x [x, y, z, dx, dy, dz, heading]",
         [k_top, box_dim], "derived", "centernet_utils.py decode_bbox_from_heatmap K=MAX_OBJ_PER_SAMPLE",
         "Top-K is fixed-size [1, K, 7]; score/range mask then makes M data-dependent."),
        ("candidate_scores", "decode_topk", "nms", [None], "FP32", "M", [k_top],
         "derived", "centernet_utils.py decode_bbox_from_heatmap", ""),
        ("pred_boxes", "nms", "result", [None, box_dim], "FP32", "D x [x, y, z, dx, dy, dz, heading]",
         [int(post.NMS_CONFIG.NMS_POST_MAXSIZE), box_dim],
         "derived", "model_nms_utils.py class_agnostic_nms; NMS_POST_MAXSIZE", "D detections; frame-dependent."),
        ("pred_scores", "nms", "result", [None], "FP32", "D", [int(post.NMS_CONFIG.NMS_POST_MAXSIZE)],
         "derived", "model_nms_utils.py class_agnostic_nms", ""),
        ("pred_labels", "nms", "result", [None], "INT64", "D, 1-based class id", [int(post.NMS_CONFIG.NMS_POST_MAXSIZE)],
         "derived", "center_head.py generate_predicted_boxes (+1, long)", ""),
    ]

    boundaries = []
    for tid, src, dst, shape, dt, layout, ub, grade, source, notes in tensors:
        boundaries.append({
            "id": tid, "from": src, "to": dst, "shape": shape,
            "dtype": dt if dt in PORTABLE_DTYPES else "unknown", "dtype_exact": dt,
            "layout": layout,
            "logical_bytes": logical_bytes(shape, dt),
            "upper_bound_shape": ub,
            "upper_bound_logical_bytes": logical_bytes(ub, dt) if ub else None,
            "memory_domain": "unknown", "transfer": "unknown",
            "physical_bytes": None, "latency_us": None,
            "evidence": {"grade": grade, "source": source},
            "notes": notes,
        })

    S = "OpenPCDet module source at pinned commit"
    stages = [
        ("frame_load", "Frame load + range mask + shuffle", "outside", "waymo_dataset.py get_lidar; data_processor.py",
         "Dataset path, before model(). shuffle_points is enabled for test: point order is non-deterministic unless seeded."),
        ("voxelize", "Voxelization (Point2VoxelCPU3d)", "outside", "data_processor.py transform_points_to_voxels",
         "Runs in the dataset on CPU via numpy/cumm, not inside model()."),
        ("vfe", "MeanVFE", "inside", "mean_vfe.py", "Elementwise; ONNX-expressible ops."),
        ("sparse_backbone_3d", "VoxelResBackBone8x (spconv)", "inside", "spconv_backbone.py",
         "Submanifold/strided sparse 3D convolutions; no standard ONNX operator. Rulebook (index) construction is data-dependent."),
        ("height_compression", "HeightCompression (sparse to dense)", "inside", "height_compression.py",
         "Scatter of active sites into a dense grid, then reshape."),
        ("bev_backbone_2d", "BaseBEVBackbone", "inside", "base_bev_backbone.py", "Dense Conv2d/BN/ReLU/ConvTranspose2d; static shape."),
        ("center_head", "CenterHead convs", "inside", "center_head.py", "Dense Conv2d; static shape."),
        ("decode_topk", "Heatmap decode + top-K + mask", "inside", "centernet_utils.py",
         "sigmoid/topk/gather/atan2 then boolean mask: output size data-dependent."),
        ("nms", "Rotated BEV NMS (nms_gpu)", "inside", "model_nms_utils.py; iou3d_nms_utils",
         "Custom CUDA extension in OpenPCDet; no standard ONNX operator."),
        ("result", "Detections", "inside", "centerpoint.py post_processing", "final_box_dicts returned by model()."),
    ]
    stage_records = [{
        "id": sid, "label": label, "execution_domain": "unknown",
        "timing_boundary": f"{where} model(batch_dict)",
        "evidence": {"grade": "derived", "source": f"{S}: {src}"},
        "notes": notes,
    } for sid, label, where, src, notes in stages]

    return {
        "grid_size_xyz": [gx, gy, gz],
        "sparse_levels": sparse_levels,
        "bev_levels": bev_levels,
        "head_channels": head_channels,
        "stages": stage_records,
        "boundaries": boundaries,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openpcdet", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.openpcdet).resolve()
    cfg_path = Path(args.cfg).resolve()
    base_cfg = (root / "tools" / "cfgs" / "dataset_configs" / "waymo_dataset.yaml").resolve()

    cfg = load_openpcdet_cfg(root, cfg_path)
    probe = probe_voxel_generator(cfg)
    derived = derive(cfg, probe)

    import spconv
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip()

    out = {
        "workload": {
            "name": "CenterPoint (VoxelResBackBone8x) on Waymo, OpenPCDet - static contract",
            "status": "derived",
            "source": f"OpenPCDet {commit} {cfg_path.relative_to(root)}",
        },
        "provenance": {
            "openpcdet_commit": commit,
            "openpcdet_worktree_modified": bool(dirty),
            "openpcdet_worktree_status": dirty.splitlines(),
            "model_cfg": str(cfg_path.relative_to(root)),
            "model_cfg_sha256": sha256(cfg_path),
            "dataset_cfg": str(base_cfg.relative_to(root)),
            "dataset_cfg_sha256": sha256(base_cfg),
            "spconv_version": spconv.__version__,
            "license_note": "OpenPCDet code is Apache-2.0. Waymo data and Waymo-trained weights are governed by the "
                            "Waymo Dataset License Agreement; OpenPCDet does not distribute pretrained Waymo models.",
        },
        "input_contract": {
            "batch_size": 1,
            "point_features": list(cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.used_feature_list),
            "point_cloud_range": list(cfg.DATA_CONFIG.POINT_CLOUD_RANGE),
            "voxel_size": next(p for p in cfg.DATA_CONFIG.DATA_PROCESSOR
                               if p.NAME == "transform_points_to_voxels").VOXEL_SIZE,
            "max_points_per_voxel": next(p for p in cfg.DATA_CONFIG.DATA_PROCESSOR
                                         if p.NAME == "transform_points_to_voxels").MAX_POINTS_PER_VOXEL,
            "max_voxels_test": next(p for p in cfg.DATA_CONFIG.DATA_PROCESSOR
                                    if p.NAME == "transform_points_to_voxels").MAX_NUMBER_OF_VOXELS["test"],
            "intensity_transform": "tanh (waymo_dataset.py get_lidar, POINTS_TANH_DIM unset)",
            "shuffle_points_test": bool(next(p for p in cfg.DATA_CONFIG.DATA_PROCESSOR
                                             if p.NAME == "shuffle_points").SHUFFLE_ENABLED["test"]),
        },
        "output_contract": {
            "classes": list(cfg.CLASS_NAMES),
            "score_thresh": cfg.MODEL.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH,
            "top_k": cfg.MODEL.DENSE_HEAD.POST_PROCESSING.MAX_OBJ_PER_SAMPLE,
            "post_center_limit_range": list(cfg.MODEL.DENSE_HEAD.POST_PROCESSING.POST_CENTER_LIMIT_RANGE),
            "nms": dict(cfg.MODEL.DENSE_HEAD.POST_PROCESSING.NMS_CONFIG),
        },
        "voxelizer_probe": probe,
        **derived,
        "onnx": {
            "available": False,
            "upstream_export_path": None,
            "note": "No ONNX export exists at the pinned commit. The spconv backbone and the rotated-NMS CUDA "
                    "extension have no standard ONNX operators, so any ONNX graph would cover a subset of stages.",
        },
        "unknown": [
            "N points, V voxels, S active sparse sites and D detections for a representative frame",
            "Trained checkpoint (required for representative S and D)",
            "Execution domain, memory domain and transfer behavior of every stage and boundary",
            "Physical/allocated bytes and strides of every boundary",
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {out_path}")
    bev = next(b for b in derived["boundaries"] if b["id"] == "spatial_features")
    print(f"grid xyz {derived['grid_size_xyz']}  BEV {bev['shape']}")


if __name__ == "__main__":
    main()

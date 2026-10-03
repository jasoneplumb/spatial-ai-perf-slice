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
    ap.add_argument(
        "--points-npy",
        help=(
            "Optional real point cloud. Accepts N x 5 model-ready "
            "[x,y,z,intensity,elongation] or OpenPCDet Waymo N x 6 "
            "[x,y,z,intensity,elongation,NLZ_flag]."
        ),
    )
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

    from types import ModuleType, SimpleNamespace
    import cumm.tensorview as tv
    from spconv.utils import Point2VoxelCPU3d

    # OpenPCDet eagerly imports every dataset backend when pcdet.datasets is
    # initialized. Some detector modules import dataset submodules even though
    # CenterPoint itself does not need those optional backends. Register a
    # namespace package so those submodules remain importable without executing
    # pcdet/datasets/__init__.py.
    datasets_pkg = ModuleType("pcdet.datasets")
    datasets_pkg.__path__ = [str(root / "pcdet" / "datasets")]
    datasets_pkg.__package__ = "pcdet.datasets"
    sys.modules.setdefault("pcdet.datasets", datasets_pkg)

    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils.spconv_utils import replace_feature, spconv

    cfg = load_openpcdet_cfg(root, cfg_path)

    point_cloud_range = np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE, dtype=np.float32)
    voxel_cfg = next(
        x for x in cfg.DATA_CONFIG.DATA_PROCESSOR if x.NAME == "transform_points_to_voxels"
    )
    voxel_size = np.asarray(voxel_cfg.VOXEL_SIZE, dtype=np.float32)
    grid_size = np.round(
        (point_cloud_range[3:6] - point_cloud_range[0:3]) / voxel_size
    ).astype(np.int64)

    # Avoid importing pcdet.datasets here: its package __init__ imports optional
    # dataset backends (NuScenes/Argo2/etc.) irrelevant to this Waymo graph.
    dataset = SimpleNamespace(
        class_names=cfg.CLASS_NAMES,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        grid_size=grid_size,
        depth_downsample_factor=None,
        point_feature_encoder=SimpleNamespace(num_point_features=5),
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.points_npy:
        points_path = Path(args.points_npy).resolve()
        loaded = np.load(points_path)
        if loaded.ndim != 2 or loaded.shape[1] not in (5, 6):
            raise SystemExit(
                f"--points-npy must contain N x 5 or N x 6 float data; got {loaded.shape}"
            )
        loaded = loaded.astype(np.float32, copy=False)
        if loaded.shape[1] == 6:
            # OpenPCDet's processed Waymo files store NLZ in column 5. Its
            # get_lidar() path keeps NLZ == -1 and tanh-normalizes intensity.
            loaded = loaded[loaded[:, 5] == -1]
            points = loaded[:, :5].copy()
            points[:, 3] = np.tanh(points[:, 3])
            input_description = "OpenPCDet-processed Waymo N x 6 point cloud"
        else:
            points = loaded
            input_description = "external model-ready N x 5 point cloud"
        source_point_count = int(points.shape[0])
        input_source = str(points_path)
    else:
        points = synthetic_points(args.points, args.seed)
        source_point_count = int(points.shape[0])
        input_description = "deterministic synthetic lidar-like points"
        input_source = None

    mask = np.all(
        (points[:, :3] >= point_cloud_range[:3])
        & (points[:, :3] < point_cloud_range[3:6]),
        axis=1,
    )
    points = points[mask].astype(np.float32, copy=False)
    in_range_point_count = int(points.shape[0])

    voxelizer = Point2VoxelCPU3d(
        vsize_xyz=voxel_size.tolist(),
        coors_range_xyz=point_cloud_range.tolist(),
        num_point_features=5,
        max_num_points_per_voxel=int(voxel_cfg.MAX_POINTS_PER_VOXEL),
        max_num_voxels=int(voxel_cfg.MAX_NUMBER_OF_VOXELS["test"]),
    )
    tv_voxels, tv_coords, tv_num_points = voxelizer.point_to_voxel(tv.from_numpy(points))
    voxels = tv_voxels.numpy().copy()
    coords = tv_coords.numpy().copy()
    num_points = tv_num_points.numpy().copy()
    voxel_count = int(voxels.shape[0])

    # OpenPCDet collate_batch prepends the batch index to [z,y,x] voxel coords.
    voxel_coords = np.pad(coords, ((0, 0), (1, 0)), constant_values=0)
    cpu_batch = {
        "voxels": voxels,
        "voxel_coords": voxel_coords,
        "voxel_num_points": num_points,
        "batch_size": 1,
    }

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

    # Run a separate instrumented pass so stage events do not contaminate the
    # aggregate smoke-test latency above.
    stage_samples = {}
    stage_names = [type(module).__name__ for module in model.module_list] + ["post_processing"]
    for name in stage_names:
        stage_samples[name] = []

    for _ in range(args.iterations):
        batch = clone_batch(base_batch)
        events = []
        with torch.inference_mode():
            for module in model.module_list:
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                batch = module(batch)
                stop.record()
                events.append((type(module).__name__, start, stop))

            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            model.post_processing(batch)
            stop.record()
            events.append(("post_processing", start, stop))

        torch.cuda.synchronize()
        for name, start, stop in events:
            stage_samples[name].append(float(start.elapsed_time(stop)))

    stage_latency = {}
    for name, values in stage_samples.items():
        stage_latency[name] = {
            "median": statistics.median(values),
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
        }

    # Decompose the dominant sparse backbone without modifying OpenPCDet.
    # MeanVFE first creates voxel_features, then we replay the exact
    # VoxelResBackBone8x submodules and time each one independently.
    vfe_module = next(
        module for module in model.module_list if type(module).__name__ == "MeanVFE"
    )
    sparse_backbone = next(
        module
        for module in model.module_list
        if type(module).__name__ == "VoxelResBackBone8x"
    )
    sparse_submodules = [
        ("conv_input", sparse_backbone.conv_input),
        ("conv1", sparse_backbone.conv1),
        ("conv2", sparse_backbone.conv2),
        ("conv3", sparse_backbone.conv3),
        ("conv4", sparse_backbone.conv4),
        ("conv_out", sparse_backbone.conv_out),
    ]
    sparse_samples = {name: [] for name, _ in sparse_submodules}
    sparse_active_sites = {}

    for iteration in range(args.iterations):
        batch = clone_batch(base_batch)
        with torch.inference_mode():
            batch = vfe_module(batch)
            input_sp_tensor = spconv.SparseConvTensor(
                features=batch["voxel_features"],
                indices=batch["voxel_coords"].int(),
                spatial_shape=sparse_backbone.sparse_shape,
                batch_size=batch["batch_size"],
            )
            x = input_sp_tensor
            events = []
            for name, module in sparse_submodules:
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                x = module(x)
                stop.record()
                events.append((name, start, stop))
                if iteration == 0:
                    sparse_active_sites[name] = {
                        "active_sites": int(x.features.shape[0]),
                        "channels": int(x.features.shape[1]),
                    }

        torch.cuda.synchronize()
        for name, start, stop in events:
            sparse_samples[name].append(float(start.elapsed_time(stop)))

    sparse_substage_latency = {}
    for name, values in sparse_samples.items():
        sparse_substage_latency[name] = {
            "median": statistics.median(values),
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
        }

    # Go one level deeper in conv3/conv4. Time the strided transition and each
    # individual SubMConv inside the two residual blocks. Non-convolution work
    # (BN/ReLU/add) is timed separately so the expensive CUMM kernels can be
    # associated with exact network layers.
    detail_samples = {}

    def record_event(events, name, fn):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        value = fn()
        stop.record()
        events.append((name, start, stop))
        return value

    def run_post_act(stage_name, stage, x, events):
        children = list(stage.children())
        x = record_event(
            events, f"{stage_name}.sparse_conv", lambda: children[0](x)
        )
        x = record_event(
            events,
            f"{stage_name}.bn_relu",
            lambda: replace_feature(x, children[2](children[1](x.features))),
        )
        return x

    def run_residual_block(stage_name, block, x, events):
        identity = x
        out = record_event(
            events, f"{stage_name}.conv1", lambda: block.conv1(x)
        )
        out = record_event(
            events,
            f"{stage_name}.bn1_relu",
            lambda: replace_feature(out, block.relu(block.bn1(out.features))),
        )
        out = record_event(
            events, f"{stage_name}.conv2", lambda: block.conv2(out)
        )
        out = record_event(
            events,
            f"{stage_name}.bn2_add_relu",
            lambda: replace_feature(
                out,
                block.relu(block.bn2(out.features) + identity.features),
            ),
        )
        return out

    for _ in range(args.iterations):
        batch = clone_batch(base_batch)
        events = []
        with torch.inference_mode():
            batch = vfe_module(batch)
            x = spconv.SparseConvTensor(
                features=batch["voxel_features"],
                indices=batch["voxel_coords"].int(),
                spatial_shape=sparse_backbone.sparse_shape,
                batch_size=batch["batch_size"],
            )
            x = sparse_backbone.conv_input(x)
            x = sparse_backbone.conv1(x)
            x = sparse_backbone.conv2(x)

            conv3_children = list(sparse_backbone.conv3.children())
            x = run_post_act("conv3.downsample", conv3_children[0], x, events)
            x = run_residual_block("conv3.res1", conv3_children[1], x, events)
            x = run_residual_block("conv3.res2", conv3_children[2], x, events)

            conv4_children = list(sparse_backbone.conv4.children())
            x = run_post_act("conv4.downsample", conv4_children[0], x, events)
            x = run_residual_block("conv4.res1", conv4_children[1], x, events)
            x = run_residual_block("conv4.res2", conv4_children[2], x, events)

        torch.cuda.synchronize()
        for name, start, stop in events:
            detail_samples.setdefault(name, []).append(
                float(start.elapsed_time(stop))
            )

    sparse_detail_latency = {}
    for name, values in detail_samples.items():
        sparse_detail_latency[name] = {
            "median": statistics.median(values),
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
        }

    # Capture one operator-level profile separately from all timing passes.
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        batch = clone_batch(base_batch)
        with torch.inference_mode():
            model(batch)
        torch.cuda.synchronize()

    operator_path = out.parent / "operators.txt"
    trace_path = out.parent / "trace.json"
    operator_path.write_text(
        prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=100) + "\n"
    )
    prof.export_chrome_trace(str(trace_path))

    detections = 0
    if first_pred and "pred_boxes" in first_pred[0]:
        detections = int(first_pred[0]["pred_boxes"].shape[0])

    result = {
        "valid_for_baseline": False,
        "purpose": "structural smoke test only",
        "weights": "randomly initialized",
        "input": input_description,
        "input_source": input_source,
        "source_point_count": source_point_count,
        "in_range_point_count": in_range_point_count,
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
        "stage_latency_ms": stage_latency,
        "sparse_backbone_substage_latency_ms": sparse_substage_latency,
        "sparse_backbone_active_sites": sparse_active_sites,
        "sparse_backbone_detail_latency_ms": sparse_detail_latency,
        "profile_artifacts": {
            "operators": str(operator_path),
            "trace": str(trace_path),
        },
        "peak_allocated_vram_gib": round(
            torch.cuda.max_memory_allocated() / (1024 ** 3), 3
        ),
    }

    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

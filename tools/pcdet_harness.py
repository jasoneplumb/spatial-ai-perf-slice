"""Shared setup for running the pinned OpenPCDet CenterPoint graph outside its dataset.

Mirrors the dataset path used by tools/smoke_centerpoint.py: range mask,
test-time shuffle (seeded), Point2VoxelCPU3d voxelization, collate (batch
index column), then load_data_to_gpu.
"""
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np


def load_openpcdet_cfg(root: Path, cfg_path: Path):
    from pcdet.config import cfg, cfg_from_yaml_file

    old_cwd = Path.cwd()
    try:
        os.chdir(root / "tools")
        cfg_from_yaml_file(str(cfg_path), cfg)
    finally:
        os.chdir(old_cwd)
    return cfg


class Workload:
    def __init__(self, openpcdet, cfg_path):
        root = Path(openpcdet).resolve()
        sys.path.insert(0, str(root))
        # Avoid executing pcdet/datasets/__init__.py and its optional dataset backends.
        datasets_pkg = ModuleType("pcdet.datasets")
        datasets_pkg.__path__ = [str(root / "pcdet" / "datasets")]
        datasets_pkg.__package__ = "pcdet.datasets"
        sys.modules.setdefault("pcdet.datasets", datasets_pkg)

        import cumm.tensorview as tv
        from spconv.utils import Point2VoxelCPU3d

        self.root = root
        self.tv = tv
        self.cfg = cfg = load_openpcdet_cfg(root, Path(cfg_path).resolve())
        self.pc_range = np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE, dtype=np.float32)
        procs = {p.NAME: p for p in cfg.DATA_CONFIG.DATA_PROCESSOR}
        self.voxel_cfg = procs["transform_points_to_voxels"]
        self.voxel_size = np.asarray(self.voxel_cfg.VOXEL_SIZE, dtype=np.float32)
        self.grid_size = np.round((self.pc_range[3:6] - self.pc_range[0:3]) / self.voxel_size).astype(np.int64)
        self.shuffle_test = bool(procs["shuffle_points"].SHUFFLE_ENABLED["test"])
        self.max_points_per_voxel = int(self.voxel_cfg.MAX_POINTS_PER_VOXEL)
        self.voxelizer = Point2VoxelCPU3d(
            vsize_xyz=self.voxel_size.tolist(), coors_range_xyz=self.pc_range.tolist(), num_point_features=5,
            max_num_points_per_voxel=self.max_points_per_voxel,
            max_num_voxels=int(self.voxel_cfg.MAX_NUMBER_OF_VOXELS["test"]),
        )

    def mask_and_shuffle(self, raw, seed):
        r = self.pc_range
        points = raw[np.all((raw[:, :3] >= r[:3]) & (raw[:, :3] < r[3:6]), axis=1)]
        if self.shuffle_test:
            points = points[np.random.RandomState(seed).permutation(points.shape[0])]
        return np.ascontiguousarray(points, dtype=np.float32)

    def voxelize(self, points):
        v, c, n = self.voxelizer.point_to_voxel(self.tv.from_numpy(points))
        return v.numpy().copy(), c.numpy().copy(), n.numpy().copy()

    @staticmethod
    def collate(voxels, coords, num_points):
        return {
            "voxels": voxels,
            "voxel_coords": np.pad(coords, ((0, 0), (1, 0)), constant_values=0),
            "voxel_num_points": num_points,
            "batch_size": 1,
        }

    def build_model(self):
        from pcdet.models import build_network

        dataset = SimpleNamespace(
            class_names=self.cfg.CLASS_NAMES, point_cloud_range=self.pc_range, voxel_size=self.voxel_size,
            grid_size=self.grid_size, depth_downsample_factor=None,
            point_feature_encoder=SimpleNamespace(num_point_features=5),
        )
        return build_network(model_cfg=self.cfg.MODEL, num_class=len(self.cfg.CLASS_NAMES), dataset=dataset)

    @staticmethod
    def to_gpu(batch):
        from pcdet.models import load_data_to_gpu

        load_data_to_gpu(batch)
        return batch

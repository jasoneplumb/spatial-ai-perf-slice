#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import torch


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openpcdet", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.openpcdet).resolve()
    cfg_path = Path(args.cfg).resolve()
    sys.path.insert(0, str(root))

    from types import ModuleType

    datasets_pkg = ModuleType("pcdet.datasets")
    datasets_pkg.__path__ = [str(root / "pcdet" / "datasets")]
    datasets_pkg.__package__ = "pcdet.datasets"
    sys.modules.setdefault("pcdet.datasets", datasets_pkg)

    from pcdet.datasets.waymo.waymo_dataset import WaymoDataset
    from pcdet.utils import common_utils

    cfg = load_openpcdet_cfg(root, cfg_path)
    logger = common_utils.create_logger()

    dataset = WaymoDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        logger=logger,
    )

    sample = dataset[args.index]
    batch = dataset.collate_batch([sample])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(copy.deepcopy(batch), out)

    meta = {
        "dataset_index": args.index,
        "config": str(cfg_path),
        "frame_file": str(out),
        "sha256": sha256(out),
    }
    out.with_suffix(out.suffix + ".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

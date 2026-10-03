#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openpcdet", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.openpcdet).resolve()
    sys.path.insert(0, str(root))

    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import build_dataloader

    cfg_from_yaml_file(str(Path(args.cfg).resolve()), cfg)

    dataset, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=0,
        logger=None,
        training=False,
    )

    sample = dataset[args.index]
    batch = dataset.collate_batch([sample])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(copy.deepcopy(batch), out)

    meta = {
        "dataset_index": args.index,
        "config": str(Path(args.cfg).resolve()),
        "frame_file": str(out),
        "sha256": sha256(out),
    }
    out.with_suffix(out.suffix + ".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

# Third-party notices

This repository is licensed under Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). It **does not redistribute** any third-party source file, dataset, model weight or binary; the only third-party-derived code is the adapted OpenPCDet logic listed below. The setup scripts download the software below onto the user's machine, where each component stays under its own license.

This file is an engineering inventory, not legal advice. Licenses were read from each project's repository in November 2026. Transitive dependencies are not listed; check them in the installed environment (for example with `pip-licenses`).

## Reference implementation

| Project | Version | License | How it is used |
| --- | --- | --- | --- |
| [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) (OpenPCDet Development Team, OpenMMLab) | commit `233f849829b6ac19afb8af8837a0246890908755` | Apache-2.0 | Cloned into `third_party/OpenPCDet` (ignored by Git) by `scripts/setup.sh`. The tools import its `pcdet` package and read its model configs. |

Two things here go beyond calling OpenPCDet's API:

- **Local patch.** `scripts/setup-openpcdet.sh` renames one keyword argument in the local clone's `pcdet/datasets/processor/data_processor.py` (`max_num_points` → `max_num_points_per_voxel`) for spconv 2.3.8. The patched file stays in the user's clone and is not redistributed.
- **Adapted logic.** A few functions re-express OpenPCDet module logic so that it can be timed or exported:
  - `StaticPillarNet.pillar_features` in `tools/pillar_variant.py` rewrites `PillarVFE.forward` out of place, and `StaticPillarNet.forward` rewrites the `PointPillarScatter` scatter for batch 1.
  - `run_residual_block` in `tools/smoke_centerpoint.py` replays the forward pass of the sparse residual block one operation at a time.
  - `tools/pcdet_harness.py` mirrors the dataset path (range mask, shuffle, voxelize, collate).

  These are treated as modified portions of an Apache-2.0 work, distributed under the same license. OpenPCDet ships no `NOTICE` file at the pinned commit, and its `LICENSE` leaves the copyright line as the unfilled template.

The model itself is CenterPoint (Yin, Zhou and Krähenbühl, "Center-based 3D Object Detection and Tracking", CVPR 2021; reference code at [tianweiy/CenterPoint](https://github.com/tianweiy/CenterPoint), MIT). Only OpenPCDet's implementation is used. Accuracy figures in [docs/PILLAR-VARIANT.md](docs/PILLAR-VARIANT.md) are quoted from the OpenPCDet README and were not reproduced.

## Runtime and build dependencies

Installed by the setup scripts; none is committed here.

| Project | Version | License | Installed by |
| --- | --- | --- | --- |
| [PyTorch](https://github.com/pytorch/pytorch) | 2.7.0 (CUDA 12.6 wheel) | BSD-3-Clause; the wheel bundles NVIDIA CUDA runtime libraries under NVIDIA's license terms | `scripts/setup-wsl.sh` |
| [torchvision](https://github.com/pytorch/vision) | 0.22.0 | BSD-3-Clause | `scripts/setup-wsl.sh` |
| [NumPy](https://github.com/numpy/numpy) | as resolved by pip | BSD-3-Clause | `scripts/setup-wsl.sh`, as a dependency of torchvision; the only dependency of `tools/scan_generator.py` and the tests |
| [spconv](https://github.com/traveller59/spconv) | 2.3.8 (`spconv-cu126`) | Apache-2.0 | `scripts/setup-openpcdet.sh` |
| [cumm](https://github.com/FindDefinition/cumm) | as required by spconv | Apache-2.0 | dependency of spconv |
| NVIDIA CUDA Toolkit | 12.6 (conda, `nvidia` channel) | NVIDIA Software License Agreement (proprietary) | `scripts/setup-openpcdet.sh` |
| [ONNX](https://github.com/onnx/onnx) | 1.17.0 | Apache-2.0 | `scripts/setup-onnx.sh` (optional) |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | 1.20.1 | MIT | `scripts/setup-onnx.sh` (optional) |
| [Miniforge](https://github.com/conda-forge/miniforge) and Python 3.11 | latest installer; Python 3.11 | BSD-3-Clause; PSF License | `scripts/setup-wsl.sh` |

The NVIDIA components are proprietary. Installing them means accepting NVIDIA's terms; this repository neither grants nor changes them.

## Data and model weights

**None is included, and none is needed for any result in this repository.**

- **Waymo Open Dataset.** Waymo data, and model weights trained on it, fall under the Waymo Dataset License Agreement for Non-Commercial Use. This repository contains no Waymo data, nothing derived from Waymo data, and no pretrained checkpoint. OpenPCDet does not distribute its Waymo checkpoints either.
- **Bringing your own.** `tools/freeze_frame.py`, `tools/profile_single_frame.py` and the `--points-npy` options can read Waymo frames or a trained checkpoint that you supply. Whether you may use them is for you to establish. Anything produced from them (frozen frames, predictions, traces) inherits those terms and must stay out of version control.
- **Published sensor figures.** `tools/scan_generator.py` takes a few sensor parameters (vertical field of view, range limits, average points per frame) from the Waymo Open Dataset paper (Sun et al., "Scalability in Perception for Autonomous Driving: Waymo Open Dataset", CVPR 2020, arXiv:1912.04838). Those are marked "Published" in `PARAM_NOTES`; the other documented parameters are marked as assumptions or design choices. They are cited facts, not dataset content. The `waymo-like` preset name is descriptive only.
- **Generated artifacts.** The ONNX files that the export tools write carry random weights and the OpenPCDet architecture. They are written to `artifacts/`, which Git ignores.

## Trademarks

Waymo, NVIDIA, CUDA, GeForce, Intel and other names belong to their owners and are used only to identify the products and publications involved. This project is not affiliated with, sponsored by or endorsed by Waymo, OpenMMLab, NVIDIA or any other organization named here.

# Windows 11 + WSL2 setup

The goal is to use WSL2 as plumbing, not as a separate thing to learn. The benchmark itself remains a Linux/Python/PyTorch workflow.

## 1. Windows: install WSL2 + Ubuntu

Open **PowerShell as Administrator** from the repository checkout and run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap-wsl.ps1
```

Reboot if requested. The first Ubuntu launch asks for a Linux username and password.

## 2. Ubuntu: keep the repo in the Linux filesystem

Clone the repo under your Linux home directory (for example `~/src/spatial-ai-perf-slice`), rather than running the workload from `/mnt/c`.

Then:

```bash
cd ~/src/spatial-ai-perf-slice
bash ./scripts/setup-wsl.sh
```

This verifies that WSL can see the NVIDIA GPU with `nvidia-smi`, installs basic build prerequisites and Miniforge, and creates the conda environment `.conda-env` with the pinned PyTorch.

## 3. Reference-workload setup

Install spconv and the CUDA toolkit, then clone, pin and build the OpenPCDet reference:

```bash
bash ./scripts/setup-openpcdet.sh
bash ./scripts/smoke-centerpoint.sh
```

The smoke test and the studies in the top-level README run from here with synthetic input and random weights. No dataset or checkpoint is needed. The two ONNX export studies (`tools/export_dense_onnx.py`, `tools/pillar_variant.py`) also need:

```bash
bash ./scripts/setup-onnx.sh
```

The `scripts/*.sh` launchers activate the environment themselves. To run tools directly, either call `.conda-env/bin/python` or activate it first:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ./.conda-env
```

## 4. If something fails

Verify these from Ubuntu, with the environment activated:

```bash
nvidia-smi
python --version        # expect 3.11
git --version
```

If `nvidia-smi` shows the NVIDIA GPU, the Windows/WSL boundary is working. Do not install a separate Linux NVIDIA display driver inside WSL.

## 5. Reference baseline (not part of this release)

The baseline needs prepared Waymo data and a CenterPoint checkpoint compatible with `tools/cfgs/waymo_models/centerpoint.yaml`, both of which you must be entitled to use. Neither is included. With them, use the commands in the top-level README to freeze one frame and collect the baseline, which produces:

- `results/baseline/timing.json`
- `results/baseline/operators.txt`
- `results/baseline/trace.json`
- `results/baseline/prediction.pt`

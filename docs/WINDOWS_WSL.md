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
./scripts/setup-wsl.sh
```

This verifies that WSL can see the NVIDIA GPU with `nvidia-smi`, installs basic build/Python prerequisites, and creates `.venv`.

## 3. Reference-workload setup

Activate the environment and clone the OpenPCDet reference:

```bash
source .venv/bin/activate
./scripts/setup.sh
```

At that point the remaining external prerequisites are intentionally explicit: a CUDA-compatible PyTorch/OpenPCDet environment, prepared Waymo data, and a CenterPoint checkpoint compatible with `tools/cfgs/waymo_models/centerpoint.yaml`.

## 4. First useful checkpoint

Before downloading/preparing Waymo, verify these from Ubuntu:

```bash
nvidia-smi
python --version
git --version
```

If `nvidia-smi` shows the NVIDIA GPU, the Windows/WSL boundary is working. Do not install a separate Linux NVIDIA display driver inside WSL.

## 5. Run the slice

After OpenPCDet, Waymo and the checkpoint are ready, use the commands in the top-level README to freeze one frame and collect the baseline.

The first experimental milestone is the appearance of:

- `results/baseline/timing.json`
- `results/baseline/operators.txt`
- `results/baseline/trace.json`
- `results/baseline/prediction.pt`

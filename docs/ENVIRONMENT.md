# Pinned WSL environment

Target development host: Windows 11 + WSL2 + RTX 2060 6 GiB.

## Base layer

- Windows NVIDIA driver: existing 581.15 (keep it)
- WSL2: existing installation
- Ubuntu: current WSL Ubuntu
- Python: 3.11 from Miniforge, in the repository-local conda environment `.conda-env`
- PyTorch: 2.7.0
- torchvision: 0.22.0
- PyTorch CUDA runtime: 12.6

The Windows driver reports CUDA 13.0 capability. That is a driver compatibility
ceiling, not a requirement to use a CUDA 13 toolkit.

Run:

```bash
bash ./scripts/setup-wsl.sh
```

Then:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ./.conda-env
python tools/check_environment.py
```

Expected on the current machine: RTX 2060, about 6 GiB VRAM, compute capability
7.5, CUDA available, and a successful CUDA matrix operation.

## Development toolkit / OpenPCDet layer

Do not install a Linux NVIDIA display driver in WSL.

The CUDA development toolkit is separate from the PyTorch runtime. OpenPCDet
requires compiled CUDA extensions, so after the base check succeeds run:

```bash
bash ./scripts/setup-openpcdet.sh
```

It installs `spconv-cu126==2.3.8` and the CUDA 12.6 toolkit (nvcc) into
`.conda-env`, clones OpenPCDet into `third_party/`, checks out the pinned
commit `233f849829b6ac19afb8af8837a0246890908755`, applies a one-keyword
compatibility patch for spconv 2.3.8, and builds the extensions. Verify the
result with `bash ./scripts/smoke-centerpoint.sh`.

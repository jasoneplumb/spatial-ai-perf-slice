# Pinned WSL environment

Target development host: Windows 11 + WSL2 + RTX 2060 6 GiB.

## Base layer

- Windows NVIDIA driver: existing 581.15 (keep it)
- WSL2: existing installation
- Ubuntu: current WSL Ubuntu
- Python: Ubuntu python3 in repository-local .venv
- PyTorch: 2.7.0
- torchvision: 0.22.0
- PyTorch CUDA runtime: 12.6

The Windows driver reports CUDA 13.0 capability. That is a driver compatibility
ceiling, not a requirement to use a CUDA 13 toolkit.

Run:

```bash
./scripts/setup-wsl.sh
```

Then:

```bash
source .venv/bin/activate
python tools/check_environment.py
```

Expected on the current machine: RTX 2060, about 6 GiB VRAM, compute capability
7.5, CUDA available, and a successful CUDA matrix operation.

## Development toolkit / OpenPCDet layer

Do not install a Linux NVIDIA display driver in WSL.

The CUDA development toolkit is a separate decision from the PyTorch runtime.
OpenPCDet requires compiled CUDA extensions, while upstream spconv 2.x currently
publishes Linux CUDA packages through CUDA 12.0. Therefore the repo deliberately
does not pretend that a CUDA-12.6 PyTorch wheel implies a matching spconv-cu126
package.

After the base check succeeds, pin the OpenPCDet commit, toolkit and spconv
combination together and verify it with the CenterPoint reference workload.

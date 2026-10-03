#!/usr/bin/env python3
import json, os, platform, shutil, subprocess, sys

def cmd(name):
    path=shutil.which(name)
    if not path: return None
    try: return subprocess.check_output([name, "--version"], text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e: return str(e)

r={"python":sys.version.split()[0],"platform":platform.platform(),"wsl": "microsoft" in platform.release().lower(),
   "nvcc":cmd("nvcc")}
try:
    import torch
    r["torch"]=torch.__version__; r["torch_cuda_runtime"]=torch.version.cuda
    r["cuda_available"]=torch.cuda.is_available()
    if r["cuda_available"]:
        p=torch.cuda.get_device_properties(0)
        r.update(gpu=p.name, compute_capability=".".join(map(str,torch.cuda.get_device_capability(0))),
                 vram_gib=round(p.total_memory/2**30,2))
        a=torch.randn((1024,1024),device="cuda"); b=a@a; torch.cuda.synchronize()
        r["cuda_smoke_test"]=float(b[0,0].item())==float(b[0,0].item())
except Exception as e:
    r["torch_error"]=repr(e)
try:
    import spconv
    r["spconv"]=getattr(spconv,"__version__","installed")
except Exception as e: r["spconv_error"]=repr(e)
print(json.dumps(r,indent=2))
ok=r.get("cuda_available") and r.get("cuda_smoke_test")
sys.exit(0 if ok else 1)

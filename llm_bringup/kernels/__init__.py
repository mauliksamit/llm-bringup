"""Custom CUDA kernels, JIT-built with torch.utils.cpp_extension on first import.

Needs nvcc and a host C++ compiler on PATH (on Windows: run inside a Visual
Studio developer prompt, e.g. `VsDevCmd.bat -arch=amd64`). Build output goes
to build/torch_extensions/ (gitignored).
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_HERE = Path(__file__).resolve().parent
_BUILD = _HERE.parents[1] / "build" / "torch_extensions"
_ext = None


def _module():
    global _ext
    if _ext is None:
        _BUILD.mkdir(parents=True, exist_ok=True)
        major, minor = torch.cuda.get_device_capability()
        _ext = load(
            name="llm_bringup_rmsnorm",
            sources=[str(_HERE / "rmsnorm.cu")],
            extra_cuda_cflags=["-O3", "-lineinfo", f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"],
            build_directory=str(_BUILD),
            verbose=False,
        )
    return _ext


def fused_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return _module().rmsnorm_forward(x, w, eps)

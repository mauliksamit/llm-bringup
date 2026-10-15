#!/usr/bin/env python3
"""Small driver for Compute Sanitizer: runs the fused RMSNorm kernel on the
awkward shapes from tests/test_fused_rmsnorm.py (vector and scalar paths,
unaligned view, 1-element rows) and checks the results.

Usage (inside a shell where the extension is already built):
  compute-sanitizer --tool memcheck  python scripts/sanitize_rmsnorm.py
  compute-sanitizer --tool racecheck python scripts/sanitize_rmsnorm.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_bringup.kernels import fused_rmsnorm  # noqa: E402
from llm_bringup.model import rmsnorm_torch  # noqa: E402

bad = 0
for dtype in (torch.float32, torch.bfloat16, torch.float16):
    for rows, hidden in ((1, 1), (3, 7), (2, 8), (5, 2047), (4, 2048), (2, 5632)):
        x = torch.randn(rows, hidden, device="cuda").to(dtype)
        w = torch.randn(hidden, device="cuda").to(dtype)
        y = fused_rmsnorm(x, w, 1e-5)
        ok = torch.allclose(y.float(), rmsnorm_torch(x, w, 1e-5).float(), rtol=2 * 2**-7, atol=1e-6)
        bad += not ok
        print(f"{str(dtype):15s} rows={rows} hidden={hidden}: {'PASS' if ok else 'FAIL'}")
base = torch.randn(4 * 2048 + 1, device="cuda", dtype=torch.bfloat16)
x = base[1:].view(4, 2048)
w = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
ok = torch.allclose(fused_rmsnorm(x, w, 1e-5).float(), rmsnorm_torch(x, w, 1e-5).float(), rtol=2 * 2**-7)
bad += not ok
print(f"unaligned view: {'PASS' if ok else 'FAIL'}")
torch.cuda.synchronize()
print("ALL PASS" if bad == 0 else f"{bad} FAILED")
sys.exit(1 if bad else 0)

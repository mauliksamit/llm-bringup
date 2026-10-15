#!/usr/bin/env python3
"""Kernel-level benchmark: fused RMSNorm vs the composed PyTorch RMSNorm.

Same protocol as kern: CUDA events around the op only, 10 warmup + 100
timed launches per shape, median / min / p95, identical inputs for both,
raw timings saved. Shapes: rows {1, 4, 128, 4096} x hidden 2048, bfloat16
(1 and 4 rows are decode at batch 1 and 4; 4096 rows is a b4 x p1024 prefill).
Derived: GB/s = (read x + read w + write y) bytes / time
              = (2 * rows * hidden + hidden) * 2 bytes / time   (bf16)
This is the minimum traffic; the composed version moves more (intermediate
float32 tensors), so its GB/s is "effective", not achieved bandwidth.
Run twice (--run 1, --run 2). Needs the extension to build (see
llm_bringup/kernels/__init__.py).

Usage: python scripts/bench_rmsnorm.py --run 1
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_bringup.common import RESULTS, environment  # noqa: E402
from llm_bringup.kernels import fused_rmsnorm  # noqa: E402
from llm_bringup.model import rmsnorm_torch  # noqa: E402

ROWS = [1, 4, 128, 4096]
HIDDEN = 2048
WARMUP, TIMED = 10, 100


def time_op(fn) -> list[float]:
    for _ in range(WARMUP):
        fn()
    out = []
    for _ in range(TIMED):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        out.append(s.elapsed_time(e) * 1000)  # microseconds
    return out


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    rows_out = []
    for rows in ROWS:
        x = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(HIDDEN, device="cuda", dtype=torch.bfloat16)
        assert torch.allclose(fused_rmsnorm(x, w, 1e-5).float(), rmsnorm_torch(x, w, 1e-5).float(), rtol=2 * 2**-7, atol=0)
        nbytes = (2 * rows * HIDDEN + HIDDEN) * 2
        rec = {"rows": rows, "hidden": HIDDEN, "dtype": "bfloat16", "min_bytes": nbytes}
        for name, fn in (("torch_composed", lambda: rmsnorm_torch(x, w, 1e-5)), ("fused", lambda: fused_rmsnorm(x, w, 1e-5))):
            t = time_op(fn)
            med = statistics.median(t)
            rec[name] = {"median_us": med, "min_us": min(t), "p95_us": sorted(t)[int(0.95 * (len(t) - 1))],
                         "gbps_at_median": nbytes / (med * 1e-6) / 1e9, "iters_us": t}
        rec["speedup_median"] = rec["torch_composed"]["median_us"] / rec["fused"]["median_us"]
        rows_out.append(rec)
        print(f"rows={rows:5d}: torch {rec['torch_composed']['median_us']:8.2f} us  fused {rec['fused']['median_us']:8.2f} us  "
              f"({rec['speedup_median']:.2f}x)  fused {rec['fused']['gbps_at_median']:.1f} GB/s", flush=True)
    d = RESULTS / "kernels"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"rmsnorm_run{a.run}.json").write_text(json.dumps({
        "run": a.run, "protocol": "CUDA events per launch, 10 warmup + 100 timed", "warmup": WARMUP, "timed": TIMED,
        "environment": environment(), "shapes": rows_out}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

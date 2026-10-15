#!/usr/bin/env python3
"""Run 1 vs run 2 of scripts/bench_rmsnorm.py: flag any median that moves by
more than 5% (relative to the larger). Writes results/kernels/rmsnorm_drift.json.

Usage: python scripts/check_kernel_drift.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
d = ROOT / "results" / "kernels"
r1 = json.loads((d / "rmsnorm_run1.json").read_text())
r2 = json.loads((d / "rmsnorm_run2.json").read_text())
cells, flagged = [], 0
for a, b in zip(r1["shapes"], r2["shapes"]):
    for mode in ("eager", "graph"):
        for impl in ("torch_composed", "fused"):
            m1, m2 = a[mode][impl]["median_us"], b[mode][impl]["median_us"]
            drift = 100 * abs(m1 - m2) / max(m1, m2)
            flagged += drift > 5
            cells.append({"rows": a["rows"], "mode": mode, "impl": impl, "run1": m1, "run2": m2,
                          "drift_pct": drift, "flagged": drift > 5})
            print(f"rows={a['rows']:5d} {mode:5s} {impl:15s} {m1:9.2f} {m2:9.2f}  {drift:5.2f}%" + ("  FLAGGED" if drift > 5 else ""))
(d / "rmsnorm_drift.json").write_text(json.dumps({"limit_pct": 5, "flagged": flagged, "cells": cells}, indent=2) + "\n")
sys.exit(1 if flagged else 0)

#!/usr/bin/env python3
"""Compare run 1 and run 2 of a step: flag any median (TTFT or decode) that
differs by more than 5% (relative to the larger value). Writes
results/bench/<step>_drift.json. Exit 1 if anything is flagged.

Usage: python scripts/check_drift.py --step s0_fp32_baseline
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMIT_PCT = 5.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True)
    a = ap.parse_args()
    d = ROOT / "results" / "bench"
    r1 = json.loads((d / f"{a.step}_run1.json").read_text())
    r2 = json.loads((d / f"{a.step}_run2.json").read_text())
    cells, flagged = [], 0
    for c1, c2 in zip(r1["configs"], r2["configs"]):
        assert (c1["batch"], c1["prompt_len"]) == (c2["batch"], c2["prompt_len"])
        for metric in ("ttft_ms", "decode_ms"):
            m1, m2 = c1[metric]["median"], c2[metric]["median"]
            drift = 100 * abs(m1 - m2) / max(m1, m2)
            bad = drift > LIMIT_PCT
            flagged += bad
            cells.append({"batch": c1["batch"], "prompt_len": c1["prompt_len"], "metric": metric,
                          "run1_median": m1, "run2_median": m2, "drift_pct": drift, "flagged": bad})
            print(f"b={c1['batch']} p={c1['prompt_len']} {metric:9s} {m1:10.3f} {m2:10.3f}  drift {drift:5.2f}%"
                  + ("  FLAGGED" if bad else ""))
    out = {"step": a.step, "limit_pct": LIMIT_PCT, "flagged": flagged, "cells": cells}
    (d / f"{a.step}_drift.json").write_text(json.dumps(out, indent=2) + "\n")
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())

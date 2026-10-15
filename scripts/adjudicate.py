#!/usr/bin/env python3
"""Adjudicate a step whose run-1 vs run-2 drift check flagged cells.

Reads every results/bench/<step>_run<N>.json, and per cell reports each
run's median and min, the spread of the medians, and the spread of the
mins. Writes results/bench/<step>_adjudication.json. Reporting rule used
by scripts/summarize.py for every step: the reported value of a cell is the
median of the per-run medians over all runs present (with two runs, their
mean), so an adjudicating third run counts like the others.

Usage: python scripts/adjudicate.py --step s1_bf16
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True)
    a = ap.parse_args()
    d = ROOT / "results" / "bench"
    runs = sorted(d.glob(f"{a.step}_run*.json"), key=lambda p: int(p.stem.rsplit("run", 1)[1]))
    data = [json.loads(p.read_text()) for p in runs]
    cells = []
    for i, c0 in enumerate(data[0]["configs"]):
        for metric in ("ttft_ms", "decode_ms"):
            meds = [r["configs"][i][metric]["median"] for r in data]
            mins = [r["configs"][i][metric]["min"] for r in data]
            cell = {"batch": c0["batch"], "prompt_len": c0["prompt_len"], "metric": metric,
                    "runs": [r["run"] for r in data], "medians": meds, "mins": mins,
                    "median_spread_pct": 100 * (max(meds) - min(meds)) / max(meds),
                    "min_spread_pct": 100 * (max(mins) - min(mins)) / max(mins),
                    "reported_median_of_medians": statistics.median(meds)}
            cells.append(cell)
            print(f"b={cell['batch']} p={cell['prompt_len']} {metric:9s} medians {['%.2f' % m for m in meds]} "
                  f"spread {cell['median_spread_pct']:.2f}% | mins spread {cell['min_spread_pct']:.2f}%")
    out = {"step": a.step, "runs": [p.name for p in runs], "cells": cells,
           "rule": "reported value = median of per-run medians; min spread shows the noise floor"}
    (d / f"{a.step}_adjudication.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Summarize the Nsight Compute profile of decode-step GEMVs.

Input: the raw `ncu --csv --page raw` output of scripts/ncu_decode_step.py
run with `-k regex:gemv --kernel-name-base demangled` (see that script's
docstring; the NVTX range filter did not match on this setup, so every GEMV
launch is profiled and the last decode step is selected here).
Structure check: launches = 1 (prefill LM head) + 4 decode steps x 89
(22 layers x {qkv, o, gate/up, down} + LM head); the last 89 rows are the
profiled step.

For each projection: kernel time, DRAM bytes read, dram__throughput
(% of peak sustained), and the analytical weight bytes (out x in x 2, bf16)
for comparison. Aggregates are time-weighted.
Writes results/ncu/decode_step_gemv.csv (paths redacted) and
results/ncu/decode_step_gemv_summary.json.

Usage: python scripts/ncu_decode_summary.py results/_cache/ncu_raw.txt
"""
import csv
import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEP_GEMVS = 89
# grid size -> (projection, out_features, in_features) for TinyLlama at batch 1 (bf16)
SHAPES = {"(640, 1, 1)": ("qkv", 2560, 2048), "(2816, 1, 1)": ("gate_up", 11264, 2048),
          "(8000, 1, 1)": ("lm_head", 32000, 2048)}


def num(s: str) -> float:
    return float(s.replace(",", ""))


def main() -> int:
    raw = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace").splitlines()
    start = next(i for i, l in enumerate(raw) if l.startswith('"ID"'))
    rows = list(csv.reader(io.StringIO("\n".join(raw[start:]))))
    head, data = rows[0], rows[2:]
    if len(data) != 1 + 4 * STEP_GEMVS:
        print(f"unexpected launch count {len(data)} (expected {1 + 4 * STEP_GEMVS}); not summarizing")
        return 1
    step = data[-STEP_GEMVS:]
    col = {k: head.index(k) for k in ("Grid Size", "gpu__time_duration.sum", "dram__bytes_op_read.sum",
                                      "dram__throughput.avg.pct_of_peak_sustained_elapsed")}
    groups: dict[str, dict] = {}
    o_or_down = 0
    for r in step:
        grid = r[col["Grid Size"]]
        if grid in SHAPES:
            name, out_f, in_f = SHAPES[grid]
        else:  # grid 512: o_proj (2048x2048) and down_proj (2048x5632) alternate within each layer
            name, out_f, in_f = (("o", 2048, 2048) if o_or_down % 2 == 0 else ("down", 2048, 5632))
            o_or_down += 1
        g = groups.setdefault(name, {"launches": 0, "time_ns": 0.0, "dram_read_bytes": 0.0, "weighted_pct": 0.0,
                                     "weight_bytes": 0})
        t = num(r[col["gpu__time_duration.sum"]])
        g["launches"] += 1
        g["time_ns"] += t
        g["dram_read_bytes"] += num(r[col["dram__bytes_op_read.sum"]])
        g["weighted_pct"] += t * num(r[col["dram__throughput.avg.pct_of_peak_sustained_elapsed"]])
        g["weight_bytes"] += out_f * in_f * 2
    total_t = sum(g["time_ns"] for g in groups.values())
    for g in groups.values():
        g["dram_throughput_pct_time_weighted"] = g.pop("weighted_pct") / g["time_ns"]
        g["dram_read_over_weight_bytes"] = g["dram_read_bytes"] / g["weight_bytes"]
        g["effective_gbps_weight_bytes"] = g["weight_bytes"] / g["time_ns"]  # bytes/ns == GB/s
    summary = {
        "source": "scripts/ncu_decode_step.py under ncu 2025.3.1, --clock-control none, one decode step (s2 config, b1, position 131)",
        "metrics": ["gpu__time_duration.sum", "dram__bytes_op_read.sum", "dram__throughput.avg.pct_of_peak_sustained_elapsed"],
        "launches_in_step": STEP_GEMVS,
        "projections": groups,
        "total": {
            "time_us": total_t / 1000,
            "dram_read_bytes": sum(g["dram_read_bytes"] for g in groups.values()),
            "weight_bytes": sum(g["weight_bytes"] for g in groups.values()),
            "dram_throughput_pct_time_weighted": sum(g["dram_throughput_pct_time_weighted"] * g["time_ns"] for g in groups.values()) / total_t,
        },
        "note": "weight_bytes is analytical (out x in x 2 bytes); durations are under the profiler with clocks unlocked",
    }
    out = ROOT / "results" / "ncu"
    out.mkdir(parents=True, exist_ok=True)
    text = "\n".join(raw[start:]) + "\n"
    for p in sorted({str(ROOT), os.path.expanduser("~")}, key=len, reverse=True):
        text = text.replace(p, "<redacted>").replace(p.replace("\\", "/"), "<redacted>")
    (out / "decode_step_gemv.csv").write_text(text, encoding="utf-8")
    (out / "decode_step_gemv_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for k, g in groups.items():
        print(f"{k:8s} n={g['launches']:3d} time {g['time_ns'] / 1000:8.1f} us  DRAM read/weights {g['dram_read_over_weight_bytes']:.3f}  "
              f"dram% {g['dram_throughput_pct_time_weighted']:5.1f}  {g['effective_gbps_weight_bytes']:.0f} GB/s (weight bytes / time)")
    t = summary["total"]
    print(f"total time {t['time_us']:.1f} us, DRAM read {t['dram_read_bytes'] / 1e6:.1f} MB vs weights {t['weight_bytes'] / 1e6:.1f} MB, "
          f"dram% (time-weighted) {t['dram_throughput_pct_time_weighted']:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

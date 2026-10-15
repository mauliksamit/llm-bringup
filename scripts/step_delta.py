#!/usr/bin/env python3
"""Print a step's before/after lines from results/summary.json (for commit
messages and the write-up, so no number is retyped by hand).

Usage: python scripts/step_delta.py --step s5_fused_rmsnorm
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True)
    a = ap.parse_args()
    steps = json.loads((ROOT / "results" / "summary.json").read_text())["steps"]
    names = [s["step"] for s in steps]
    i = names.index(a.step)
    cur, prev = steps[i], steps[i - 1] if i else None
    print(f"{a.step} vs {prev['step'] if prev else '(none)'} (results/summary.json):")
    for j, c in enumerate(cur["cells"]):
        p = prev["cells"][j] if prev else None
        cell = f"b{c['batch']} p{c['prompt_len']}"
        if p:
            print(f"  {cell}: TTFT {p['ttft_ms']:.2f} -> {c['ttft_ms']:.2f} ms ({p['ttft_ms'] / c['ttft_ms']:.2f}x); "
                  f"decode {p['ms_per_token']:.3f} -> {c['ms_per_token']:.3f} ms/token ({p['decode_ms'] / c['decode_ms']:.2f}x); "
                  f"peak {p['peak_mem_gib']:.2f} -> {c['peak_mem_gib']:.2f} GiB")
        else:
            print(f"  {cell}: TTFT {c['ttft_ms']:.2f} ms; decode {c['ms_per_token']:.3f} ms/token; peak {c['peak_mem_gib']:.2f} GiB")
    n = cur["numerics"]
    if n:
        print(f"  numerics: max |logit diff| {n['max_abs_logit_diff']:.4g}, mean {n['mean_abs_logit_diff']:.3g}, "
              f"teacher-forced match {100 * n['teacher_forced_token_match_rate']:.1f}%, free-run {100 * n['free_run_match_rate']:.1f}%, "
              f"ppl {n['ppl']:.4f} (ref {n['ppl_reference']:.4f})")
    print(f"  drift flagged (run1 vs run2): {cur['drift_flagged_cells_run1_vs_run2']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Build results/summary.json and the generated tables in README.md and
docs/FACTS.md from results/bench/*_run*.json and results/numerics/*.json.

Reported value per cell = median of the per-run medians over every run of
that step (see scripts/adjudicate.py). Speedups are previous-step / this-step
(and baseline / this-step) on those reported values. Nothing is typed by hand.

Usage: python scripts/summarize.py            # write
       python scripts/summarize.py --check    # exit 1 if README/docs blocks are stale
"""
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_bringup.common import STEPS  # noqa: E402

BEGIN, END = "<!-- RESULTS-BEGIN -->", "<!-- RESULTS-END -->"
TBEGIN, TEND = "<!-- TRACE-BEGIN -->", "<!-- TRACE-END -->"


def load_step(step: str):
    runs = sorted((ROOT / "results" / "bench").glob(f"{step}_run*.json"),
                  key=lambda p: int(p.stem.rsplit("run", 1)[1]))
    if not runs:
        return None
    data = [json.loads(p.read_text()) for p in runs]
    cells = []
    for i, c0 in enumerate(data[0]["configs"]):
        rs = [r["configs"][i] for r in data]
        dec = statistics.median(r["decode_ms"]["median"] for r in rs)
        n = c0["new_tokens"] - 1
        cells.append({
            "batch": c0["batch"], "prompt_len": c0["prompt_len"],
            "ttft_ms": statistics.median(r["ttft_ms"]["median"] for r in rs),
            "decode_ms": dec,
            "ms_per_token": dec / n,
            "decode_tok_per_s": c0["batch"] * n / (dec / 1000),
            "peak_mem_gib": max(r["peak_mem_gib"] for r in rs),
            "runs": len(rs),
        })
    num_p = ROOT / "results" / "numerics" / f"{step}.json"
    num = json.loads(num_p.read_text())["summary"] if num_p.exists() else None
    drift_p = ROOT / "results" / "bench" / f"{step}_drift.json"
    flagged = json.loads(drift_p.read_text())["flagged"] if drift_p.exists() else None
    return {"step": step, "runs": [p.name for p in runs], "cells": cells, "numerics": num,
            "drift_flagged_cells_run1_vs_run2": flagged}


def main() -> int:
    check = "--check" in sys.argv
    steps = [s for s in (load_step(k) for k in STEPS) if s]
    base = steps[0]
    for i, s in enumerate(steps):
        prev = steps[i - 1] if i else None
        for j, c in enumerate(s["cells"]):
            b0 = base["cells"][j]
            c["ttft_speedup_vs_baseline"] = b0["ttft_ms"] / c["ttft_ms"]
            c["decode_speedup_vs_baseline"] = b0["decode_ms"] / c["decode_ms"]
            if prev:
                p = prev["cells"][j]
                c["ttft_speedup_vs_prev"] = p["ttft_ms"] / c["ttft_ms"]
                c["decode_speedup_vs_prev"] = p["decode_ms"] / c["decode_ms"]
    kernels = []
    kruns = sorted((ROOT / "results" / "kernels").glob("rmsnorm_run*.json"))
    if kruns:
        kd = [json.loads(p.read_text()) for p in kruns]
        for i, sh in enumerate(kd[0]["shapes"]):
            row = {"rows": sh["rows"], "hidden": sh["hidden"], "runs": len(kd)}
            for impl in ("torch_composed", "fused"):
                meds = [r["shapes"][i]["graph"][impl]["median_us"] for r in kd]
                row[f"{impl}_graph_us"] = statistics.median(meds)
                row[f"{impl}_graph_us_range"] = [min(meds), max(meds)]
            sp = [r["shapes"][i]["graph"]["speedup_median"] for r in kd]
            row["graph_speedup"] = row["torch_composed_graph_us"] / row["fused_graph_us"]
            row["graph_speedup_range_over_runs"] = [min(sp), max(sp)]
            kernels.append(row)
    summary = {"generated_by": "scripts/summarize.py", "rule": "median of per-run medians", "steps": steps,
               "rmsnorm_kernel_graph_mode": kernels}

    rows = ["| Step | batch | prompt | TTFT ms | decode ms/token | decode tok/s | peak GiB | vs prev (TTFT / decode) | vs baseline (TTFT / decode) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for s in steps:
        for c in s["cells"]:
            vp = (f"{c['ttft_speedup_vs_prev']:.2f}x / {c['decode_speedup_vs_prev']:.2f}x"
                  if "ttft_speedup_vs_prev" in c else "–")
            rows.append(f"| {s['step']} | {c['batch']} | {c['prompt_len']} | {c['ttft_ms']:.2f} | {c['ms_per_token']:.3f} | "
                        f"{c['decode_tok_per_s']:.1f} | {c['peak_mem_gib']:.2f} | {vp} | "
                        f"{c['ttft_speedup_vs_baseline']:.2f}x / {c['decode_speedup_vs_baseline']:.2f}x |")
    nrows = ["| Step | max abs logit diff | mean abs logit diff | teacher-forced token match | free-run greedy match | perplexity (ref) |",
             "|---|---|---|---|---|---|"]
    for s in steps:
        n = s["numerics"]
        if n:
            nrows.append(f"| {s['step']} | {n['max_abs_logit_diff']:.4g} | {n['mean_abs_logit_diff']:.3g} | "
                         f"{100 * n['teacher_forced_token_match_rate']:.1f}% | {100 * n['free_run_match_rate']:.1f}% | "
                         f"{n['ppl']:.4f} ({n['ppl_reference']:.4f}) |")
    krows = ["| rows x 2048 (bf16) | composed PyTorch, us | fused kernel, us | speedup (range over runs) |", "|---|---|---|---|"]
    for k in kernels:
        lo, hi = k["graph_speedup_range_over_runs"]
        krows.append(f"| {k['rows']} | {k['torch_composed_graph_us']:.2f} | {k['fused_graph_us']:.2f} | "
                     f"{k['graph_speedup']:.2f}x ({lo:.2f}-{hi:.2f}x) |")
    kblock = ("\n\n### Fused RMSNorm kernel vs composed PyTorch (graph-replayed GPU time, median of run medians)\n\n"
              + "\n".join(krows)) if kernels else ""
    block = (f"{BEGIN}\n### Performance (median of per-run medians; 64 new tokens; RTX 5070 Laptop)\n\n" + "\n".join(rows)
             + "\n\n### Numerics vs the FP32 CPU reference (transformers)\n\n" + "\n".join(nrows) + kblock + f"\n{END}")

    # Per-step trace breakdown (perflens), if scripts/trace_breakdown.py has run.
    tb_path = ROOT / "results" / "trace_breakdown.json"
    tblock = ""
    if tb_path.exists():
        tb = json.loads(tb_path.read_text(encoding="utf-8"))
        trows = ["| Step | GPU span ms | GPU busy ms | idle (launch gaps) | kernels per new token | GEMV/GEMM share (heuristic) |",
                 "|---|---|---|---|---|---|"]
        for r in tb["steps"]:
            i = r["idle"][0]
            gemm = next((b["share_pct"] for b in r["buckets_heuristic"] if b["bucket"] == "gemm"), 0.0)
            trows.append(f"| {r['step']} | {r['gpu_span_us'] / 1000:.1f} | {i['busy_us'] / 1000:.1f} | {i['idle_pct']:.1f}% | "
                         f"{r['kernels_per_new_token']:.1f} | {gemm:.1f}% |")
        tblock = (f"{TBEGIN}\nTrace: {tb['workload']}; analysis by perflens `{tb['perflens_commit'][:7]}` "
                  f"(results/trace_breakdown.json).\n\n" + "\n".join(trows) + f"\n{TEND}")

    out = ROOT / "results" / "summary.json"
    new_json = json.dumps(summary, indent=2) + "\n"
    stale = not out.exists() or out.read_text() != new_json
    for path in (ROOT / "README.md", ROOT / "docs" / "WRITEUP.md"):
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        new_text = text
        for b, e, content in ((BEGIN, END, block), (TBEGIN, TEND, tblock)):
            if b in new_text and content:
                new_text = re.sub(re.escape(b) + r"[\s\S]*?" + re.escape(e), lambda _, c=content: c, new_text)
        if new_text != text:
            stale = True
            if not check:
                path.write_text(new_text, encoding="utf-8")
    if check:
        print("summary stale: run python scripts/summarize.py" if stale else "summary up to date")
        return 1 if stale else 0
    out.write_text(new_json)
    print(block)
    print(tblock)
    return 0


if __name__ == "__main__":
    sys.exit(main())

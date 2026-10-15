#!/usr/bin/env python3
"""Generate results/facts.json and docs/FACTS.md from committed results.

Every value is read from a results/ file (or counted by running pytest /
reading the weights file header); the source file is recorded next to each.
Derived figures that the docs quote (speedups, "N% fewer") are computed here
so the claims audit can trace them.

Usage: python scripts/facts.py [--skip-tests]
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def param_count() -> int | None:
    st = ROOT / "models" / "TinyLlama-1.1B-Chat-v1.0" / "model.safetensors"
    if not st.exists():
        return None
    from safetensors import safe_open

    total = 0
    with safe_open(str(st), "pt") as f:
        for k in f.keys():
            n = 1
            for d in f.get_slice(k).get_shape():
                n *= d
            total += n
    return total


def pytest_counts():
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=ROOT,
                       capture_output=True, text=True)
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    m = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|skipped|errors?)", tail)}
    return {"passed": m.get("passed", 0), "failed": m.get("failed", 0), "skipped": m.get("skipped", 0),
            "summary_line": tail, "command": "python -m pytest -q"}


def cell(step, b, p):
    return next(c for c in step["cells"] if c["batch"] == b and c["prompt_len"] == p)


def main() -> int:
    s = load("results/summary.json")
    steps = {x["step"]: x for x in s["steps"]}
    base, best = steps["s0_fp32_baseline"], steps["s5_fused_rmsnorm"]
    tb = {r["step"]: r for r in load("results/trace_breakdown.json")["steps"]}
    ncu = load("results/ncu/decode_step_gemv_summary.json")
    model = load("results/model_files.json")
    sanit = {t: "0 errors" in (ROOT / f"results/sanitizer/rmsnorm_{t}.log").read_text(encoding="utf-8", errors="replace")
             or "0 hazards" in (ROOT / f"results/sanitizer/rmsnorm_{t}.log").read_text(encoding="utf-8", errors="replace")
             for t in ("memcheck", "racecheck", "synccheck", "initcheck")}
    cells = []
    for c0 in base["cells"]:
        c5 = cell(best, c0["batch"], c0["prompt_len"])
        cells.append({"batch": c0["batch"], "prompt_len": c0["prompt_len"],
                      "ttft_ms_baseline": c0["ttft_ms"], "ttft_ms_final": c5["ttft_ms"],
                      "ttft_speedup": c0["ttft_ms"] / c5["ttft_ms"],
                      "ms_per_token_baseline": c0["ms_per_token"], "ms_per_token_final": c5["ms_per_token"],
                      "decode_speedup": c0["decode_ms"] / c5["decode_ms"],
                      "decode_tok_per_s_final": c5["decode_tok_per_s"],
                      "peak_mem_gib_baseline": c0["peak_mem_gib"], "peak_mem_gib_final": c5["peak_mem_gib"]})
    s3, s4 = steps["s3_static_kv"], steps["s4_cuda_graph"]
    ttft_up = [100 * (cell(s4, c["batch"], c["prompt_len"])["ttft_ms"] / c["ttft_ms"] - 1) for c in s3["cells"]]
    ratios = [r["dram_read_over_weight_bytes"] for r in ncu["projections"].values()]
    pcts = [r["dram_throughput_pct_time_weighted"] for r in ncu["projections"].values()]
    facts = {
        "generated_by": "scripts/facts.py",
        "model": {"repo": model["repo"], "revision": model["revision"], "license": model["license"],
                  "parameters": param_count(), "source": "results/model_files.json + model.safetensors header"},
        "bringup_numerics_fp32": {**steps["s0_fp32_baseline"]["numerics"], "source": "results/numerics/s0_fp32_baseline.json"},
        "final_numerics_bf16": {**best["numerics"], "source": "results/numerics/s5_fused_rmsnorm.json"},
        "baseline_vs_final": {"from": "s0_fp32_baseline", "to": "s5_fused_rmsnorm", "cells": cells,
                              "source": "results/summary.json"},
        "launch_gaps": {
            "idle_pct_eager_bf16_s3": tb["s3_static_kv"]["idle"][0]["idle_pct"],
            "idle_pct_graph_s4": tb["s4_cuda_graph"]["idle"][0]["idle_pct"],
            "idle_pct_final_s5": tb["s5_fused_rmsnorm"]["idle"][0]["idle_pct"],
            "busy_ms_fp32_s0": tb["s0_fp32_baseline"]["idle"][0]["busy_us"] / 1000,
            "busy_ms_bf16_s1": tb["s1_bf16"]["idle"][0]["busy_us"] / 1000,
            "idle_pct_fp32_s0": tb["s0_fp32_baseline"]["idle"][0]["idle_pct"],
            "idle_pct_bf16_s1": tb["s1_bf16"]["idle"][0]["idle_pct"],
            "kernels_per_token_s4": tb["s4_cuda_graph"]["kernels_per_new_token"],
            "kernels_per_token_s5": tb["s5_fused_rmsnorm"]["kernels_per_new_token"],
            "kernels_per_token_reduction_pct_s4_to_s5": 100 * (1 - tb["s5_fused_rmsnorm"]["kernels_per_new_token"] / tb["s4_cuda_graph"]["kernels_per_new_token"]),
            "source": "results/trace_breakdown.json (perflens)"},
        "s4_ttft_increase_pct_vs_s3": {"min": min(ttft_up), "max": max(ttft_up), "per_cell": ttft_up,
                                       "source": "results/summary.json (s4 ttft / s3 ttft - 1)"},
        "decode_gemv_counters": {"dram_read_over_weight_bytes_min": min(ratios), "dram_read_over_weight_bytes_max": max(ratios),
                                 "dram_throughput_pct_min": min(pcts), "dram_throughput_pct_max": max(pcts),
                                 "dram_throughput_pct_time_weighted": ncu["total"]["dram_throughput_pct_time_weighted"],
                                 "weight_gb": ncu["total"]["weight_bytes"] / 1e9, "gemv_time_us": ncu["total"]["time_us"],
                                 "source": "results/ncu/decode_step_gemv_summary.json"},
        "fused_rmsnorm_kernel": {"graph_mode": s["rmsnorm_kernel_graph_mode"], "sanitizer_clean": sanit,
                                 "source": "results/summary.json, results/kernels/, results/sanitizer/"},
    }
    if "--skip-tests" not in sys.argv:
        facts["tests"] = pytest_counts()
    else:
        old = ROOT / "results" / "facts.json"
        if old.exists():
            facts["tests"] = json.loads(old.read_text())["tests"]
    (ROOT / "results" / "facts.json").write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")

    def f(x, d=2):
        return f"{x:.{d}f}"

    lines = ["# Verified facts (generated — do not edit)", "",
             "Generated by `scripts/facts.py` from `results/`. Only verified numbers, each with its source file.",
             "",
             "| Fact | Value | Source |", "|---|---|---|",
             f"| Model | {model['repo']} @ `{model['revision'][:12]}`, {facts['model']['parameters']:,} parameters, {model['license']} | results/model_files.json |",
             f"| Bring-up numerics (from-scratch FP32 GPU vs transformers FP32 CPU) | max abs logit diff {facts['bringup_numerics_fp32']['max_abs_logit_diff']:.3g}, teacher-forced token match {100 * facts['bringup_numerics_fp32']['teacher_forced_token_match_rate']:.1f}%, perplexity {facts['bringup_numerics_fp32']['ppl']:.4f} vs {facts['bringup_numerics_fp32']['ppl_reference']:.4f} | results/numerics/s0_fp32_baseline.json |"]
    for c in cells:
        lines.append(f"| b{c['batch']} p{c['prompt_len']}: FP32 baseline -> final (BF16, SDPA, static KV, CUDA graphs, fused RMSNorm) | "
                     f"TTFT {f(c['ttft_ms_baseline'])} -> {f(c['ttft_ms_final'])} ms ({f(c['ttft_speedup'])}x); decode "
                     f"{f(c['ms_per_token_baseline'], 3)} -> {f(c['ms_per_token_final'], 3)} ms/token ({f(c['decode_speedup'])}x); "
                     f"peak {f(c['peak_mem_gib_baseline'])} -> {f(c['peak_mem_gib_final'])} GiB | results/summary.json |")
    lg = facts["launch_gaps"]
    lines += [
        f"| GPU idle (launch gaps) in the traced generation | eager static-KV {f(lg['idle_pct_eager_bf16_s3'], 1)}% -> CUDA graphs {f(lg['idle_pct_graph_s4'], 1)}% -> final {f(lg['idle_pct_final_s5'], 1)}% | results/trace_breakdown.json |",
        f"| Kernels per new token (traced) | {f(lg['kernels_per_token_s4'], 1)} -> {f(lg['kernels_per_token_s5'], 1)} with the fused RMSNorm ({f(lg['kernels_per_token_reduction_pct_s4_to_s5'], 1)}% fewer) | results/trace_breakdown.json |",
        f"| Decode GEMVs (Nsight Compute) | DRAM read / weight bytes {f(facts['decode_gemv_counters']['dram_read_over_weight_bytes_min'], 3)}-{f(facts['decode_gemv_counters']['dram_read_over_weight_bytes_max'], 3)}; dram__throughput {f(facts['decode_gemv_counters']['dram_throughput_pct_min'], 1)}-{f(facts['decode_gemv_counters']['dram_throughput_pct_max'], 1)}% of peak | results/ncu/decode_step_gemv_summary.json |",
    ]
    for k in s["rmsnorm_kernel_graph_mode"]:
        lo, hi = k["graph_speedup_range_over_runs"]
        lines.append(f"| Fused RMSNorm kernel, {k['rows']} x {k['hidden']} bf16 (graph-replayed GPU time, median of {k['runs']} runs) | "
                     f"{f(k['torch_composed_graph_us'])} -> {f(k['fused_graph_us'])} us ({f(k['graph_speedup'])}x; per-run {f(lo)}-{f(hi)}x) | results/summary.json |")
    lines.append(f"| Compute Sanitizer on the fused kernel | " + ", ".join(f"{t}: {'clean' if ok else 'NOT CLEAN'}" for t, ok in sanit.items()) + " | results/sanitizer/ |")
    if "tests" in facts:
        lines.append(f"| Tests (pytest) | {facts['tests']['summary_line']} | results/facts.json |")
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "FACTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())

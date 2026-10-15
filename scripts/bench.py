#!/usr/bin/env python3
"""Benchmark one ladder step: TTFT, decode throughput, peak GPU memory.

Grid: batch {1, 4} x prompt length {128, 1024}, NEW_TOKENS greedy tokens each.
Protocol (same discipline as kern): CUDA events (see llm_bringup/generate.py
for the exact timing definitions), WARMUP untimed generations then TIMED
timed generations per config, median / p95 / min reported, raw per-iteration
values saved. Run the script twice per step (--run 1, --run 2) and compare
with scripts/check_drift.py (flags any median that moves more than 5%).
Prompts are real text: BOS + slices of data/eval_text.txt, a different
offset per batch row; identical across steps.

Derived metrics:
  decode_tok_per_s = batch * (NEW_TOKENS - 1) / (decode_ms / 1000)
  ms_per_token     = decode_ms / (NEW_TOKENS - 1)       (per sequence)
  peak_mem_gib     = torch.cuda.max_memory_allocated() over warmup + timed
                     iterations, divided by 2^30 (allocator view; excludes
                     the CUDA context and other processes)

Usage: python scripts/bench.py --step s0_fp32_baseline --run 1
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_bringup.common import MODEL_DIR, RESULTS, STEPS, environment, eval_ids, no_tf32, tokenizer  # noqa: E402
from llm_bringup.generate import GraphDecoder, generate  # noqa: E402
from llm_bringup.model import Llama  # noqa: E402

BATCHES = [1, 4]
PROMPTS = [128, 1024]
NEW_TOKENS = 64
WARMUP = 10
TIMED = 30


def pct(xs, p):
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def gpu_sample():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,pstate",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"unavailable ({e})"


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=list(STEPS))
    ap.add_argument("--run", type=int, required=True)
    a = ap.parse_args()
    no_tf32()
    step = STEPS[a.step]
    tok = tokenizer()
    ids = eval_ids(tok)
    model = Llama.load(MODEL_DIR, step["opts"], "cuda")
    configs = []
    for b in BATCHES:
        for p in PROMPTS:
            rows = [[tok.bos_token_id] + ids[r * 1500: r * 1500 + p - 1] for r in range(b)]
            prompt = torch.tensor(rows, device="cuda")
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            g, capture_ms = None, None
            if step["graph"]:
                t0 = time.perf_counter()
                g = GraphDecoder(model, b, p, NEW_TOKENS)
                g.prefill(prompt)
                g.capture()
                torch.cuda.synchronize()
                capture_ms = (time.perf_counter() - t0) * 1000
            env_before = gpu_sample()
            first = None
            for _ in range(WARMUP):
                out, _ = generate(model, prompt, NEW_TOKENS, graph=g)
                first = out.clone() if first is None else first
            ttft, dec = [], []
            for _ in range(TIMED):
                out, t = generate(model, prompt, NEW_TOKENS, graph=g, timed=True)
                ttft.append(t.ttft_ms)
                dec.append(t.decode_ms)
            env_after = gpu_sample()
            # every timed generation must produce the same greedy tokens as the first warmup one
            deterministic = bool(torch.equal(out, first))
            med_dec = statistics.median(dec)
            cfg = {
                "batch": b, "prompt_len": p, "new_tokens": NEW_TOKENS, "warmup": WARMUP, "timed": TIMED,
                "ttft_ms": {"median": statistics.median(ttft), "p95": pct(ttft, 95), "min": min(ttft)},
                "decode_ms": {"median": med_dec, "p95": pct(dec, 95), "min": min(dec)},
                "ms_per_token_median": med_dec / (NEW_TOKENS - 1),
                "decode_tok_per_s_median": b * (NEW_TOKENS - 1) / (med_dec / 1000),
                "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
                "graph_capture_ms": capture_ms,
                "tokens_identical_across_iterations": deterministic,
                "env_before": env_before, "env_after": env_after,
                "iters": {"ttft_ms": ttft, "decode_ms": dec},
            }
            configs.append(cfg)
            print(f"{a.step} run{a.run} b={b} p={p}: TTFT {cfg['ttft_ms']['median']:.2f} ms, "
                  f"decode {cfg['decode_tok_per_s_median']:.1f} tok/s ({cfg['ms_per_token_median']:.3f} ms/tok), "
                  f"peak {cfg['peak_mem_gib']:.2f} GiB, same tokens={deterministic}", flush=True)
            del g
    out = {"step": a.step, "run": a.run, "options": {k: str(v) for k, v in step["opts"].__dict__.items()},
           "cuda_graph": step["graph"], "protocol": "CUDA events; see scripts/bench.py docstring",
           "environment": environment(), "configs": configs}
    d = RESULTS / "bench"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{a.step}_run{a.run}.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

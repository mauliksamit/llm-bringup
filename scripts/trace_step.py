#!/usr/bin/env python3
"""Save a torch.profiler Chrome trace of one generation for a ladder step.

Config: batch 1, prompt 128, NEW_TOKENS greedy tokens (decode-dominated, so
launch gaps and per-step kernels are visible), after 3 untraced warmup
generations. Output: data/traces/<step>_b1_p128.json.gz plus a .meta.json
(GPU, driver, torch, command, commit). Local paths and the host name are
stripped. Readable in perflens (github.com/mauliksamit/perflens).

Usage: python scripts/trace_step.py --step s2_sdpa
"""
import argparse
import gzip
import json
import os
import platform
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_bringup.common import MODEL_DIR, ROOT, STEPS, environment, eval_ids, no_tf32, tokenizer  # noqa: E402
from llm_bringup.generate import GraphDecoder, generate  # noqa: E402
from llm_bringup.model import Llama  # noqa: E402

NEW_TOKENS = 64


def redact(text: str) -> str:
    variants = set()
    for p in (str(ROOT), os.path.expanduser("~")):
        variants |= {p, p.replace("\\", "/"), p.replace("\\", "\\\\")}
    for v in sorted(variants, key=len, reverse=True):
        text = text.replace(v, "<redacted>")
    return text.replace(platform.node(), "<host>") if platform.node() else text


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=list(STEPS))
    a = ap.parse_args()
    no_tf32()
    step = STEPS[a.step]
    tok = tokenizer()
    model = Llama.load(MODEL_DIR, step["opts"], "cuda")
    prompt = torch.tensor([[tok.bos_token_id] + eval_ids(tok)[:127]], device="cuda")
    g = None
    if step["graph"]:
        g = GraphDecoder(model, 1, 128, NEW_TOKENS)
        g.prefill(prompt)
        g.capture()
    for _ in range(3):
        generate(model, prompt, NEW_TOKENS, graph=g)
    torch.cuda.synchronize()
    acts = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=acts) as prof:
        with torch.profiler.record_function(f"generate_{a.step}"):
            generate(model, prompt, NEW_TOKENS, graph=g)
        torch.cuda.synchronize()
    out = ROOT / "data" / "traces" / f"{a.step}_b1_p128.json.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    prof.export_chrome_trace(str(tmp))
    text = redact(tmp.read_text(encoding="utf-8"))
    tmp.unlink()
    json.loads(text)
    with gzip.open(out, "wt", encoding="utf-8", compresslevel=9) as f:
        f.write(text)
    meta = {"file": f"data/traces/{out.name}", "step": a.step,
            "options": {k: str(v) for k, v in step["opts"].__dict__.items()}, "cuda_graph": step["graph"],
            "workload": {"batch": 1, "prompt_len": 128, "new_tokens": NEW_TOKENS, "untraced_warmup": 3},
            "command": f"python scripts/trace_step.py --step {a.step}", "environment": environment()}
    (out.parent / (out.name + ".meta.json")).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {meta['file']} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

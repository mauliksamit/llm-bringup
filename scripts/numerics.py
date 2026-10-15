#!/usr/bin/env python3
"""Numerics of one ladder step against the FP32 CPU reference (scripts/reference.py).

For each reference sequence (prompt + reference greedy continuation):
  teacher-forced logits on the GPU -> max / mean |logit - ref logit| over all
  positions, and the argmax agreement rate ("token match rate").
Free-running greedy generation from each prompt -> exact-prefix match length
  and match rate against the reference tokens.
Perplexity over the first PPL_TOKENS tokens of data/eval_text.txt.

Writes results/numerics/<step>.json. Usage: python scripts/numerics.py --step s1_bf16
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_bringup.common import (CACHE, MODEL_DIR, PPL_TOKENS, REF_NEW_TOKENS, RESULTS, STEPS,  # noqa: E402
                                environment, eval_ids, no_tf32, tokenizer)
from llm_bringup.generate import GraphDecoder, generate  # noqa: E402
from llm_bringup.model import Llama  # noqa: E402


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=list(STEPS))
    a = ap.parse_args()
    no_tf32()
    step = STEPS[a.step]
    ref = json.loads((RESULTS / "reference" / "reference.json").read_text(encoding="utf-8"))
    model = Llama.load(MODEL_DIR, step["opts"], "cuda")
    tok = tokenizer()
    rows = []
    for i, s in enumerate(ref["sequences"]):
        prompt, gen = s["prompt_ids"], s["generated_ids"]
        ref_logits = torch.load(CACHE / f"ref_logits_{i}.pt").cuda()
        full = torch.tensor([prompt + gen], device="cuda")
        logits = model.forward(full, None, 0)[0].float()
        diff = (logits - ref_logits).abs()
        agree = (logits.argmax(-1) == ref_logits.argmax(-1)).float().mean().item()
        p = torch.tensor([prompt], device="cuda")
        if step["graph"]:
            g = GraphDecoder(model, 1, len(prompt), REF_NEW_TOKENS)
            g.prefill(p)
            g.capture()
            out, _ = generate(model, p, REF_NEW_TOKENS, graph=g)
        else:
            out, _ = generate(model, p, REF_NEW_TOKENS)
        out = out[0].tolist()
        prefix = next((k for k, (x, y) in enumerate(zip(out, gen)) if x != y), len(gen))
        rows.append({
            "sequence": i, "positions": logits.shape[0],
            "max_abs_logit_diff": diff.max().item(), "mean_abs_logit_diff": diff.mean().item(),
            "teacher_forced_token_match_rate": agree,
            "free_run_exact_prefix": prefix, "free_run_match_rate": sum(x == y for x, y in zip(out, gen)) / len(gen),
            "free_run_text": tok.decode(out),
        })
        print(f"seq {i}: max|d|={rows[-1]['max_abs_logit_diff']:.4g} mean|d|={rows[-1]['mean_abs_logit_diff']:.4g} "
              f"tf-match={agree:.4f} prefix={prefix}/{len(gen)}", flush=True)
    ids = torch.tensor([[tok.bos_token_id] + eval_ids(tok)[: PPL_TOKENS - 1]], device="cuda")
    nll = torch.nn.functional.cross_entropy(model.forward(ids, None, 0)[0, :-1].float(), ids[0, 1:]).item()
    result = {
        "step": a.step, "options": {k: str(v) for k, v in step["opts"].__dict__.items()}, "cuda_graph": step["graph"],
        "reference": "results/reference/reference.json (transformers, float32, CPU)",
        "sequences": rows,
        "summary": {
            "max_abs_logit_diff": max(r["max_abs_logit_diff"] for r in rows),
            "mean_abs_logit_diff": sum(r["mean_abs_logit_diff"] * r["positions"] for r in rows) / sum(r["positions"] for r in rows),
            "teacher_forced_token_match_rate": sum(r["teacher_forced_token_match_rate"] * r["positions"] for r in rows) / sum(r["positions"] for r in rows),
            "free_run_match_rate": sum(r["free_run_match_rate"] for r in rows) / len(rows),
            "ppl": math.exp(nll), "ppl_reference": ref["perplexity"]["ppl"], "ppl_tokens": PPL_TOKENS,
        },
        "environment": environment(),
    }
    out_dir = RESULTS / "numerics"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{a.step}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

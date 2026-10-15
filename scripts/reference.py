#!/usr/bin/env python3
"""Reference outputs from an independent implementation: Hugging Face
transformers LlamaForCausalLM, float32, on the CPU (no TF32, no GPU kernels).

Writes:
  results/reference/reference.json  prompts, greedy tokens + decoded text,
                                    reference perplexity, SHA-256 of the logits files
  results/_cache/ref_logits_<i>.pt  teacher-forced logits [len, vocab] per
                                    sequence (gitignored; large)

Deterministic: greedy decoding, fixed prompts, float32.
Usage: python scripts/reference.py
"""
import hashlib
import json
import math
import sys
import time

import torch
from transformers import LlamaForCausalLM

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from llm_bringup.common import (CACHE, EVAL_TEXT, MODEL_DIR, PPL_TOKENS, REF_NEW_TOKENS, RESULTS,  # noqa: E402
                                environment, eval_ids, no_tf32, reference_prompts, tokenizer)


def main() -> int:
    no_tf32()
    torch.manual_seed(0)
    tok = tokenizer()
    t0 = time.time()
    hf = LlamaForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()
    CACHE.mkdir(parents=True, exist_ok=True)
    out = {"implementation": "transformers LlamaForCausalLM, float32, CPU", "environment": environment(),
           "eval_text": str(EVAL_TEXT.relative_to(EVAL_TEXT.parents[1])).replace("\\", "/"),
           "sequences": []}
    with torch.no_grad():
        for i, prompt in enumerate(reference_prompts(tok)):
            ids = torch.tensor([prompt])
            gen = hf.generate(ids, max_new_tokens=REF_NEW_TOKENS, do_sample=False,
                              pad_token_id=tok.eos_token_id)[0, len(prompt):].tolist()
            full = torch.tensor([prompt + gen])
            logits = hf(full).logits[0].float()
            path = CACHE / f"ref_logits_{i}.pt"
            torch.save(logits, path)
            out["sequences"].append({
                "prompt_ids": prompt, "generated_ids": gen,
                "prompt_text": tok.decode(prompt), "generated_text": tok.decode(gen),
                "logits_file": f"results/_cache/{path.name}",
                "logits_sha256": hashlib.sha256(logits.numpy().tobytes()).hexdigest(),
                "logits_shape": list(logits.shape),
            })
            print(f"seq {i}: {tok.decode(gen)!r}", flush=True)
        ids = torch.tensor([[tok.bos_token_id] + eval_ids(tok)[: PPL_TOKENS - 1]])
        nll = torch.nn.functional.cross_entropy(hf(ids).logits[0, :-1].float(), ids[0, 1:])
        out["perplexity"] = {"tokens": PPL_TOKENS, "mean_nll": nll.item(), "ppl": math.exp(nll.item()),
                             "definition": "exp(mean next-token cross-entropy) over the first PPL_TOKENS tokens "
                                           "of data/eval_text.txt (BOS + text), one forward pass"}
    out["wall_s"] = round(time.time() - t0, 1)
    (RESULTS / "reference").mkdir(parents=True, exist_ok=True)
    (RESULTS / "reference" / "reference.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"ppl={out['perplexity']['ppl']:.4f}  wall={out['wall_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

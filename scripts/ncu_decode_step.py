#!/usr/bin/env python3
"""Target program for Nsight Compute: one NVTX-marked decode step (s2 config:
bf16, SDPA, dynamic cache, eager) at batch 1 after a 128-token prompt.

Run under ncu (metric names confirmed with `ncu --query-metrics` on ncu
2025.3.1; clocks never locked):
  ncu --clock-control none --nvtx --nvtx-include "decode_step/" -k regex:gemv \
      --metrics gpu__time_duration.sum,dram__bytes_op_read.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed \
      --csv --page raw python scripts/ncu_decode_step.py > results/ncu/decode_step_gemv.csv
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_bringup.common import MODEL_DIR, STEPS, eval_ids, no_tf32, tokenizer  # noqa: E402
from llm_bringup.model import Llama  # noqa: E402


@torch.no_grad()
def main() -> int:
    no_tf32()
    model = Llama.load(MODEL_DIR, STEPS["s2_sdpa"]["opts"], "cuda")
    tok = tokenizer()
    prompt = torch.tensor([[tok.bos_token_id] + eval_ids(tok)[:127]], device="cuda")
    cache = model.new_cache(1, 160)
    nxt = model.forward(prompt, cache, 0, last_only=True)[:, -1].argmax(-1, keepdim=True)
    for p in range(128, 131):  # warm decode steps (not profiled)
        nxt = model.forward(nxt, cache, p, last_only=True)[:, -1].argmax(-1, keepdim=True)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("decode_step")
    model.forward(nxt, cache, 131, last_only=True)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    return 0


if __name__ == "__main__":
    sys.exit(main())

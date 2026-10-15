# LOG.md — engineering log (plain language)

## 2026-10-15 — Bring-up
- Picked TinyLlama-1.1B-Chat-v1.0 (Llama architecture, public, Apache-2.0,
  fits 8 GB in FP32) at a pinned revision; wrote the forward pass from
  scratch; validated it against a random-weight `transformers` Llama (unit
  tests, including a deliberately broken RoPE that the test catches) and then
  the real weights against `transformers` FP32 on the CPU: 100.0% token
  agreement, perplexity equal to 4 decimals.

## 2026-10-15 — Baseline and BF16
- FP32 baseline is very stable (drift within 1.34%). BF16 sped prefill up a
  lot but decode barely moved, and decode took about the same time at batch 1
  and batch 4: a strong hint that decode was launch-bound, later confirmed
  by the traces (GPU idle rose when busy time halved).

## 2026-10-15 — SDPA, static KV cache, CUDA graphs
- SDPA: clear wins on long prompts and fewer kernels per token. Fixed the SDPA
  path first so it only passes a mask when needed.
- Static KV cache: first version was slower everywhere. The trace showed the
  decode mask being rebuilt in every layer (704 extra arange kernels in 16
  steps). Fixed; still slower in eager decode; kept because graphs need it.
- CUDA graphs: the big decode win, with near-zero run-to-run drift. TTFT went
  up a little; the GPU sat at its power cap in those runs (likely cause).

## 2026-10-15 — Fused RMSNorm kernel
- Wrote the kernel from scratch, matched PyTorch's rounding sequence, gated it
  with 69 correctness cases and all four Compute Sanitizer tools, then
  benchmarked it graph-replayed (eager timings were launch noise). In the
  model it removes about 315 kernels per token and helps both TTFT and decode.

## 2026-10-15 — Analysis
- perflens over all six traces gives the per-step breakdown (busy vs idle,
  kernels per token). Nsight Compute counters work on this machine now, so
  the decode GEMVs were profiled: DRAM-bound, weights streamed every token.
- Process miss: hand-typed numbers in three commit messages were wrong in the
  third decimal or a range bound; fixed before pushing, and from then on every
  number is printed by a script. The claims audit is strict (no free ratio
  search) so it cannot pass by coincidence.

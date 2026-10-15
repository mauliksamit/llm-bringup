# Negative results and misses (kept on purpose)

1. **BF16 did not speed up decode much (s1).** Hypothesis: halving weight bytes
   halves decode time. Measured: decode only 1.11x at b1 p128 while TTFT
   improved 2.10x. The traces show GPU busy time halving while idle time grew:
   decode was launch-bound. This miss is what pointed at CUDA graphs.
2. **Static KV cache, first version, was slower everywhere (s3, commit
   `e1593b3`).** Two causes found in the traces and code: the decode mask was
   rebuilt in every layer (704 extra `arange` and 704 extra compare kernels in
   16 steps), and prefill used a masked pass over the whole preallocated
   length. Fixed in `2edff08` (mask once per forward, causal prefill).
3. **Static KV cache, fixed, is still slower for eager decode (s3).** 0.82x at
   b1 p128 vs s2: masked attention over the fixed attend length plus in-place
   writes cost more than `torch.cat` saves. Kept in the ladder only because
   CUDA graphs need it; the eager regression is reported in the tables.
4. **TTFT got worse with CUDA graphs (s4)** by 5–13% although prefill code did
   not change. nvidia-smi samples show the GPU at its 75 W power cap with
   lower SM clocks; likely cause, not proven (two samples per cell).
5. **Greedy free-running agreement drops to 69.8% in BF16.** Two text prompts
   diverge from the FP32 reference after 12 tokens at a near-tie; perplexity
   barely moves (6.7468 vs 6.7475). Reported as is; not tuned away.
6. **Eager-mode kernel timings were unusable for claims.** Events around one
   eager call include CPU launch time; small-row timings drifted 15–37%
   between runs. Graph-replayed timing replaced them. An early "GB/s" figure
   above DRAM bandwidth was cache-warm throughput and is labeled so.
7. **First trace length was too large.** A 64-token trace was several times
   the size of today's 16-token traces (mostly CPU-op events), too large to
   commit for six steps; it was replaced and not kept, so its size is not
   quoted here. Traces now cover 16 decode tokens.
8. **Hand-typed numbers in commit messages were wrong three times** (ranges
   and third decimals) and were corrected before pushing; since then commit
   and doc numbers are printed by `scripts/step_delta.py` / `summarize.py`.
9. **NVTX range filtering in Nsight Compute matched nothing** on this setup;
   the decode-step profile selects the last step by launch structure instead
   (checked: 1 + 4 x 89 GEMV launches).
10. **`torch.compile`** — NOT RUN: Triton is not available on native Windows
    and WSL is out of scope here.

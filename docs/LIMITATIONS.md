# Limitations

- **One machine.** RTX 5070 Laptop GPU (8 GB, 75 W power cap, display-attached
  WDDM, Windows). Clocks move with power and temperature; medians, repeats,
  drift checks and recorded nvidia-smi samples carry that load, and the
  numbers compare runs on this machine only.
- **Batch-1 decode is CPU-sensitive** until CUDA graphs: eager decode timing
  depends on the host CPU and OS scheduling, which is why some eager cells
  needed a third run.
- **Small grid.** Batch {1, 4} x prompt {128, 1024}, 64 new tokens. No large
  batches, long contexts, or throughput-oriented serving (continuous batching,
  paged KV cache).
- **One model, one precision family.** FP32 and BF16 only; no quantization.
- **Reference.** `transformers` in FP32 on the CPU is an independent
  implementation, not ground truth; agreement with it shows the bring-up is
  correct to FP32 rounding, not that the model is "right".
- **Heuristic categories.** The GEMV/GEMM shares in the trace table come from
  perflens's name-based classifier.
- **Counters.** Nsight Compute data covers the decode GEMVs of one step in
  the s2 configuration; other kernels and steps are timing- and trace-based.
- **`torch.compile`**: NOT RUN (no Triton on native Windows).
- **GPU only.** This is a CUDA project; nothing here is a claim about other
  accelerators.

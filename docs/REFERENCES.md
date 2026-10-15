# References

No code was copied from any of these. Where an idea is used, the code says so.

Model and data
1. TinyLlama/TinyLlama-1.1B-Chat-v1.0, revision
   fe8a4ea1ffedaf415f4da2f062534de366a451e6, Apache-2.0:
   https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
2. P. Zhang, G. Zeng, T. Wang, W. Lu, "TinyLlama: An Open-Source Small
   Language Model", 2024, arXiv:2401.02385.
3. Project Gutenberg eBook #1342, *Pride and Prejudice* (public domain in the
   USA): https://www.gutenberg.org/cache/epub/1342/pg1342.txt

Architecture (what `llm_bringup/model.py` implements)
4. H. Touvron et al., "LLaMA: Open and Efficient Foundation Language Models",
   2023, arXiv:2302.13971 (pre-norm decoder, RMSNorm, SwiGLU, RoPE).
5. B. Zhang, R. Sennrich, "Root Mean Square Layer Normalization", NeurIPS 2019.
6. J. Su et al., "RoFormer: Enhanced Transformer with Rotary Position
   Embedding", 2021, arXiv:2104.09864.
7. J. Ainslie et al., "GQA: Training Generalized Multi-Query Transformer
   Models from Multi-Head Checkpoints", EMNLP 2023.
8. N. Shazeer, "GLU Variants Improve Transformer", 2020, arXiv:2002.05202.

Reference implementation (used only to validate, never imported by the model)
9. Hugging Face transformers 5.19.0, `LlamaForCausalLM`:
   https://github.com/huggingface/transformers

Optimizations and tools
10. T. Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention
    with IO-Awareness", NeurIPS 2022 (background for fused attention).
11. PyTorch `torch.nn.functional.scaled_dot_product_attention` documentation.
12. PyTorch CUDA Graphs documentation (`torch.cuda.CUDAGraph`,
    `torch.cuda.graph`, side-stream warmup before capture).
13. PyTorch C++/CUDA extensions (`torch.utils.cpp_extension.load`).
14. NVIDIA CUDA C++ Programming Guide — warp shuffle functions and
    block-level reductions (used for the RMSNorm reduction pattern).
15. NVIDIA Nsight Compute CLI and Compute Sanitizer documentation.
16. PyTorch profiler (`torch.profiler`) documentation.
17. perflens (trace analysis used for section 5 of the write-up):
    https://github.com/mauliksamit/perflens
18. kern (benchmark and correctness-gate discipline reused here):
    https://github.com/mauliksamit/kern

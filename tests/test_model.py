"""Bring-up tests on a tiny random Llama (CPU, FP32; GPU tests skip without CUDA).

The independent reference is Hugging Face transformers' LlamaForCausalLM with
the same random weights. Tolerances: FP32 on CPU, same math in a different
op order; observed differences are ~1e-6, 1e-4 leaves margin without hiding
an indexing or RoPE bug (those produce O(1) differences).
"""
import pytest
import torch
from transformers import LlamaConfig as HFConfig
from transformers import LlamaForCausalLM

from llm_bringup.generate import GraphDecoder, generate
from llm_bringup.model import Llama, LlamaConfig, RunOptions, rmsnorm_torch, rope_tables

TOL = 1e-4


def tiny(seed=0):
    torch.manual_seed(seed)
    hf_cfg = HFConfig(vocab_size=97, hidden_size=64, intermediate_size=160, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                      rms_norm_eps=1e-5, tie_word_embeddings=False)
    hf = LlamaForCausalLM(hf_cfg).eval()
    cfg = LlamaConfig(hidden_size=64, intermediate_size=160, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, vocab_size=97, rms_norm_eps=1e-5, rope_theta=10000.0,
                      max_position_embeddings=64)
    return hf, cfg, {k: v.detach().clone() for k, v in hf.state_dict().items()}


@torch.no_grad()
@pytest.mark.parametrize("attn", ["naive", "sdpa"])
def test_full_forward_matches_hf(attn):
    hf, cfg, sd = tiny()
    ids = torch.randint(0, 97, (2, 17))
    ref = hf(ids).logits
    ours = Llama(cfg, sd, RunOptions(attn=attn), "cpu").forward(ids, None, 0)
    assert (ours - ref).abs().max().item() < TOL


@torch.no_grad()
@pytest.mark.parametrize("cache", ["dynamic", "static"])
@pytest.mark.parametrize("attn", ["naive", "sdpa"])
def test_incremental_decode_matches_full_forward(attn, cache):
    """Prefill 9 tokens then feed 6 one at a time; each step's logits must equal
    the full-sequence forward at that position (KV cache + RoPE offsets correct)."""
    _, cfg, sd = tiny(1)
    m = Llama(cfg, sd, RunOptions(attn=attn, kv_cache=cache), "cpu")
    ids = torch.randint(0, 97, (3, 15))
    full = m.forward(ids, None, 0)
    kv = m.new_cache(3, 15)
    attend = 15 if cache == "static" else None
    steps = [m.forward(ids[:, :9], kv, 0, attend)]
    for p in range(9, 15):
        steps.append(m.forward(ids[:, p:p + 1], kv, p, attend))
    inc = torch.cat(steps, dim=1)
    assert (inc - full).abs().max().item() < TOL


@torch.no_grad()
def test_tensor_position_matches_int_position():
    _, cfg, sd = tiny(2)
    m = Llama(cfg, sd, RunOptions(kv_cache="static"), "cpu")
    ids = torch.randint(0, 97, (1, 6))
    a_cache, b_cache = m.new_cache(1, 8), m.new_cache(1, 8)
    m.forward(ids[:, :5], a_cache, 0, 8)
    m.forward(ids[:, :5], b_cache, 0, 8)
    a = m.forward(ids[:, 5:6], a_cache, 5, 8)
    b = m.forward(ids[:, 5:6], b_cache, torch.tensor(5), 8)
    assert torch.equal(a, b)


def test_rmsnorm_and_rope_definitions():
    x = torch.randn(4, 64)
    w = torch.randn(64)
    expect = w * x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-5)
    assert (rmsnorm_torch(x, w, 1e-5) - expect).abs().max().item() < 1e-5
    cfg = LlamaConfig(64, 160, 1, 4, 2, 97, 1e-5, 10000.0, 64)
    cos, sin = rope_tables(cfg, 8, "cpu", torch.float32)
    assert cos.shape == (8, 16) and torch.allclose(cos[0], torch.ones(16)) and torch.allclose(sin[0], torch.zeros(16))
    # position 3, frequency index 1: angle = 3 / theta^(2/16)
    assert abs(cos[3, 1].item() - torch.cos(torch.tensor(3 / 10000 ** (2 / 16))).item()) < 1e-6


@torch.no_grad()
def test_greedy_matches_hf_generate():
    hf, cfg, sd = tiny(3)
    ids = torch.randint(0, 97, (1, 7))
    ref = hf.generate(ids, max_new_tokens=8, do_sample=False, pad_token_id=0)[:, 7:]
    m = Llama(cfg, sd, RunOptions(), "cpu")
    kv = m.new_cache(1, 15)
    tok = m.forward(ids, kv, 0, last_only=True)[:, -1].argmax(-1, keepdim=True)
    out = [tok]
    for p in range(7, 14):
        tok = m.forward(tok, kv, p, last_only=True)[:, -1].argmax(-1, keepdim=True)
        out.append(tok)
    assert torch.equal(torch.cat(out, 1), ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
@torch.no_grad()
def test_cuda_graph_decode_matches_eager():
    _, cfg, sd = tiny(4)
    opts = RunOptions(attn="sdpa", kv_cache="static")
    m = Llama(cfg, sd, opts, "cuda")
    ids = torch.randint(0, 97, (2, 10), device="cuda")
    eager, _ = generate(m, ids, 12)
    g = GraphDecoder(m, 2, 10, 12)
    g.prefill(ids)
    g.capture()
    g.prefill(ids)  # capture advanced nothing, but re-prefill to be explicit
    g.decode()
    assert torch.equal(g.out, eager)

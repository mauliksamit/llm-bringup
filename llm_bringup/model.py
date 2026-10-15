"""Llama-architecture decoder written from scratch (no transformers import).

Covers what TinyLlama-1.1B needs: token embedding, RMSNorm, rotary position
embedding (RoPE, rotate-half form), grouped-query attention, SwiGLU MLP, and
an untied LM head. Weights load from the checkpoint's model.safetensors.

Every optimization in the ladder is an option here so each step changes one
thing (see RunOptions). Numerics follow the reference implementation's
conventions where they matter (RMSNorm and softmax computed in float32,
RoPE tables computed in float32); docs/WRITEUP.md records the comparison
against Hugging Face transformers.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


@dataclass(frozen=True)
class LlamaConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool = False

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @staticmethod
    def from_json(path: str | Path) -> "LlamaConfig":
        c = json.loads(Path(path).read_text(encoding="utf-8"))
        return LlamaConfig(
            hidden_size=c["hidden_size"],
            intermediate_size=c["intermediate_size"],
            num_hidden_layers=c["num_hidden_layers"],
            num_attention_heads=c["num_attention_heads"],
            num_key_value_heads=c.get("num_key_value_heads", c["num_attention_heads"]),
            vocab_size=c["vocab_size"],
            rms_norm_eps=c["rms_norm_eps"],
            rope_theta=c.get("rope_theta", 10000.0),
            max_position_embeddings=c["max_position_embeddings"],
            tie_word_embeddings=c.get("tie_word_embeddings", False),
        )


@dataclass(frozen=True)
class RunOptions:
    """One field per optimization-ladder step."""

    dtype: torch.dtype = torch.float32
    attn: str = "naive"  # "naive": explicit matmul + masked softmax; "sdpa": F.scaled_dot_product_attention
    kv_cache: str = "dynamic"  # "dynamic": torch.cat per step; "static": preallocated, written in place
    rmsnorm: str = "torch"  # "torch": composed ops; "fused": custom CUDA kernel (llm_bringup/kernels)


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------

def rmsnorm_torch(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """y = w * x / sqrt(mean(x^2) + eps); statistics in float32, output in x.dtype."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(x.dtype)


def rope_tables(cfg: LlamaConfig, max_len: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables [max_len, head_dim], computed in float32 then cast."""
    d = cfg.head_dim
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.int64, device=device).float() / d))
    pos = torch.arange(max_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [b, h, t, d]; cos/sin [t, d] or [b, 1, t, d]."""
    return x * cos + rotate_half(x) * sin


# ---------------------------------------------------------------------------
# KV caches
# ---------------------------------------------------------------------------

class DynamicCache:
    """Grows by torch.cat each step (allocates new K/V tensors every token)."""

    def __init__(self, n_layers: int):
        self.k: list[torch.Tensor | None] = [None] * n_layers
        self.v: list[torch.Tensor | None] = [None] * n_layers
        self.length = 0

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor, pos: int):
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k, v
        else:
            self.k[layer] = torch.cat([self.k[layer], k], dim=2)
            self.v[layer] = torch.cat([self.v[layer], v], dim=2)
        if layer == len(self.k) - 1:
            self.length = pos + k.shape[2]
        return self.k[layer], self.v[layer]


class StaticCache:
    """Preallocated [layers, b, kv_heads, max_len, d]; new K/V are written in place.

    Attention always reads the first `attend_len` positions (fixed for a
    whole generation) and masks positions beyond the current one, so tensor
    shapes do not change from step to step (needed for CUDA graphs).
    """

    def __init__(self, cfg: LlamaConfig, batch: int, max_len: int, device, dtype):
        shape = (cfg.num_hidden_layers, batch, cfg.num_key_value_heads, max_len, cfg.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.max_len = max_len

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor, pos: torch.Tensor | int):
        t = k.shape[2]
        if isinstance(pos, int):
            self.k[layer, :, :, pos:pos + t] = k
            self.v[layer, :, :, pos:pos + t] = v
        else:  # tensor position (graph-safe): index_copy_ along the sequence dim
            idx = pos + torch.arange(t, device=k.device)
            self.k[layer].index_copy_(2, idx, k)
            self.v[layer].index_copy_(2, idx, v)
        return self.k[layer], self.v[layer]


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

class Llama:
    def __init__(self, cfg: LlamaConfig, weights: dict[str, torch.Tensor], opts: RunOptions, device):
        self.cfg, self.opts, self.device = cfg, opts, torch.device(device)
        dt = opts.dtype

        def w(name: str) -> torch.Tensor:
            return weights[name].to(device=self.device, dtype=dt).contiguous()

        self.embed = w("model.embed_tokens.weight")
        self.norm = w("model.norm.weight")
        self.lm_head = self.embed if cfg.tie_word_embeddings else w("lm_head.weight")
        self.layers = []
        for i in range(cfg.num_hidden_layers):
            p = f"model.layers.{i}."
            self.layers.append({
                "ln1": w(p + "input_layernorm.weight"),
                "ln2": w(p + "post_attention_layernorm.weight"),
                # q/k/v fused into one GEMM: one weight read, one launch.
                "wqkv": torch.cat([w(p + "self_attn.q_proj.weight"), w(p + "self_attn.k_proj.weight"),
                                   w(p + "self_attn.v_proj.weight")], dim=0).contiguous(),
                "wo": w(p + "self_attn.o_proj.weight"),
                # gate/up fused into one GEMM.
                "wgu": torch.cat([w(p + "mlp.gate_proj.weight"), w(p + "mlp.up_proj.weight")], dim=0).contiguous(),
                "wd": w(p + "mlp.down_proj.weight"),
            })
        self.cos, self.sin = rope_tables(cfg, cfg.max_position_embeddings, self.device, dt)
        self._fused_rmsnorm = None
        if opts.rmsnorm == "fused":
            from llm_bringup.kernels import fused_rmsnorm  # built on first use

            self._fused_rmsnorm = fused_rmsnorm

    @staticmethod
    def load(model_dir: str | Path, opts: RunOptions, device="cuda") -> "Llama":
        model_dir = Path(model_dir)
        cfg = LlamaConfig.from_json(model_dir / "config.json")
        weights = load_file(str(model_dir / "model.safetensors"))
        return Llama(cfg, weights, opts, device)

    def rmsnorm(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if self._fused_rmsnorm is not None:
            return self._fused_rmsnorm(x, w, self.cfg.rms_norm_eps)
        return rmsnorm_torch(x, w, self.cfg.rms_norm_eps)

    def new_cache(self, batch: int, max_len: int):
        if self.opts.kv_cache == "static":
            return StaticCache(self.cfg, batch, max_len, self.device, self.opts.dtype)
        return DynamicCache(self.cfg.num_hidden_layers)

    def _attention(self, q, k, v, pos, t, attend_len, static):
        """q [b,h,t,d]; k/v [b,kvh,S,d] (S = cached length or static attend_len)."""
        cfg = self.cfg
        groups = cfg.num_attention_heads // cfg.num_key_value_heads
        if static:
            k, v = k[:, :, :attend_len], v[:, :, :attend_len]
        S = k.shape[2]
        # Causal mask over absolute positions: query i (at pos+i) sees keys j <= pos+i.
        q_pos = pos + torch.arange(t, device=q.device).view(t, 1)
        allowed = torch.arange(S, device=q.device).view(1, S) <= q_pos
        if self.opts.attn == "sdpa":
            if t > 1 and isinstance(pos, int) and pos == 0 and S == t:
                return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            return F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, enable_gqa=True)
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(cfg.head_dim)
        scores = scores.masked_fill(~allowed, float("-inf"))
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        return probs @ v

    def forward(self, tokens: torch.Tensor, cache, pos, attend_len: int | None = None,
                last_only: bool = False) -> torch.Tensor:
        """tokens [b, t] at absolute positions pos..pos+t-1 -> logits [b, t, vocab].

        `pos` is an int, or a 0-d int64 tensor (CUDA-graph path with a static cache).
        `last_only` computes the LM head for the last position only (generation
        needs nothing else; avoids a [b, t, vocab] tensor during prefill).
        """
        cfg = self.cfg
        b, t = tokens.shape
        h, kvh, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        static = isinstance(cache, StaticCache)
        if static and attend_len is None:
            attend_len = cache.max_len
        if isinstance(pos, int):
            cos, sin = self.cos[pos:pos + t], self.sin[pos:pos + t]
        else:
            idx = pos + torch.arange(t, device=tokens.device)
            cos, sin = self.cos.index_select(0, idx), self.sin.index_select(0, idx)
        x = F.embedding(tokens, self.embed)
        for i, L in enumerate(self.layers):
            hN = self.rmsnorm(x, L["ln1"])
            qkv = hN @ L["wqkv"].T
            q, k, v = qkv.split([h * d, kvh * d, kvh * d], dim=-1)
            q = q.view(b, t, h, d).transpose(1, 2)
            k = k.view(b, t, kvh, d).transpose(1, 2)
            v = v.view(b, t, kvh, d).transpose(1, 2)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            k_all, v_all = cache.update(i, k, v, pos) if cache is not None else (k, v)
            a = self._attention(q, k_all, v_all, pos, t, attend_len, static)
            x = x + a.transpose(1, 2).reshape(b, t, h * d) @ L["wo"].T
            hN = self.rmsnorm(x, L["ln2"])
            gate, up = (hN @ L["wgu"].T).chunk(2, dim=-1)
            x = x + (F.silu(gate) * up) @ L["wd"].T
        if last_only:
            x = x[:, -1:]
        x = self.rmsnorm(x, self.norm)
        return x @ self.lm_head.T

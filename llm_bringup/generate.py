"""Greedy generation with CUDA-event timing, plus a CUDA-graph decode path.

Timing definitions (all CUDA events on the current stream, inputs already
on the GPU, tokenization excluded):
  ttft_ms    prefill forward over the prompt + argmax of the last position
             (the first new token exists when this ends)
  decode_ms  the remaining (new_tokens - 1) single-token steps
  ms_per_token     decode_ms / (new_tokens - 1)          (per sequence)
  decode_tok_per_s batch * (new_tokens - 1) / decode_s   (all sequences)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from llm_bringup.model import Llama, StaticCache


@dataclass
class GenTiming:
    ttft_ms: float
    decode_ms: float


class GraphDecoder:
    """Captures one decode step (forward + argmax + bookkeeping) as a CUDA graph.

    Requires a static KV cache. All step state lives in GPU tensors the graph
    updates itself: the current token, the position, and the output buffer,
    so a replay needs no host-side work besides launching the graph.
    """

    def __init__(self, model: Llama, batch: int, prompt_len: int, new_tokens: int):
        assert model.opts.kv_cache == "static", "CUDA graphs need the static KV cache"
        self.model, self.batch, self.prompt_len, self.new_tokens = model, batch, prompt_len, new_tokens
        self.attend_len = prompt_len + new_tokens
        dev = model.device
        self.cache: StaticCache = model.new_cache(batch, self.attend_len)
        self.tok = torch.zeros(batch, 1, dtype=torch.long, device=dev)
        self.pos = torch.zeros((), dtype=torch.long, device=dev)
        self.out = torch.zeros(batch, new_tokens, dtype=torch.long, device=dev)
        self.graph: torch.cuda.CUDAGraph | None = None

    def _step(self):
        logits = self.model.forward(self.tok, self.cache, self.pos, self.attend_len, last_only=True)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        self.tok.copy_(nxt)
        self.out.index_copy_(1, (self.pos - self.prompt_len + 1).view(1), nxt)
        self.pos.add_(1)

    def capture(self):
        # Warm up on a side stream (required before capture), then capture.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        saved = self.pos.clone()
        with torch.cuda.stream(s):
            for _ in range(3):
                self.pos.copy_(saved)
                self._step()
        torch.cuda.current_stream().wait_stream(s)
        self.pos.copy_(saved)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step()
        self.pos.copy_(saved)

    def prefill(self, prompt: torch.Tensor):
        logits = self.model.forward(prompt, self.cache, 0, self.attend_len, last_only=True)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        self.tok.copy_(nxt)
        self.out[:, 0:1].copy_(nxt)
        self.pos.fill_(self.prompt_len)

    def decode(self):
        assert self.graph is not None
        for _ in range(self.new_tokens - 1):
            self.graph.replay()


@torch.no_grad()
def generate(model: Llama, prompt: torch.Tensor, new_tokens: int, graph: GraphDecoder | None = None,
             timed: bool = False) -> tuple[torch.Tensor, GenTiming | None]:
    """Greedy decode. Returns (new token ids [b, new_tokens], timing or None)."""
    b, p = prompt.shape
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)] if timed else None
    if ev:
        ev[0].record()
    if graph is not None:
        graph.prefill(prompt)
        if ev:
            ev[1].record()
        graph.decode()
        out = graph.out
    else:
        cache = model.new_cache(b, p + new_tokens)
        attend = p + new_tokens if model.opts.kv_cache == "static" else None
        logits = model.forward(prompt, cache, 0, attend, last_only=True)
        tok = logits[:, -1].argmax(-1, keepdim=True)
        if ev:
            ev[1].record()
        toks = [tok]
        for i in range(new_tokens - 1):
            logits = model.forward(tok, cache, p + i, attend, last_only=True)
            tok = logits[:, -1].argmax(-1, keepdim=True)
            toks.append(tok)
        out = torch.cat(toks, dim=1)
    if ev:
        ev[2].record()
        ev[2].synchronize()
        return out, GenTiming(ev[0].elapsed_time(ev[1]), ev[1].elapsed_time(ev[2]))
    return out, None

"""Shared paths, step configurations, prompts and environment metadata."""
from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import torch

from llm_bringup.model import RunOptions

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "TinyLlama-1.1B-Chat-v1.0"
EVAL_TEXT = ROOT / "data" / "eval_text.txt"
RESULTS = ROOT / "results"
CACHE = RESULTS / "_cache"  # large reference tensors, gitignored (hashes are committed)

# The optimization ladder. Each step changes exactly one option relative to the previous one.
STEPS: dict[str, dict] = {
    "s0_fp32_baseline": {"opts": RunOptions(torch.float32, "naive", "dynamic", "torch"), "graph": False},
    "s1_bf16": {"opts": RunOptions(torch.bfloat16, "naive", "dynamic", "torch"), "graph": False},
    "s2_sdpa": {"opts": RunOptions(torch.bfloat16, "sdpa", "dynamic", "torch"), "graph": False},
    "s3_static_kv": {"opts": RunOptions(torch.bfloat16, "sdpa", "static", "torch"), "graph": False},
    "s4_cuda_graph": {"opts": RunOptions(torch.bfloat16, "sdpa", "static", "torch"), "graph": True},
    "s5_fused_rmsnorm": {"opts": RunOptions(torch.bfloat16, "sdpa", "static", "fused"), "graph": True},
}

# Reference prompts: two slices of the public-domain eval text + one chat-style question.
REF_PROMPT_TOKENS = 64
REF_NEW_TOKENS = 32
PPL_TOKENS = 1024


def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_DIR)


def eval_ids(tok) -> list[int]:
    return tok(EVAL_TEXT.read_text(encoding="utf-8"), add_special_tokens=False)["input_ids"]


def reference_prompts(tok) -> list[list[int]]:
    ids = eval_ids(tok)
    bos = [tok.bos_token_id]
    chat = tok.apply_chat_template([{"role": "user", "content": "Explain in two sentences why the sky is blue."}],
                                   add_generation_prompt=True, tokenize=True)
    if hasattr(chat, "keys"):  # transformers 5 returns a BatchEncoding (dict-like, not a dict)
        chat = chat["input_ids"]
    chat = [int(t) for t in chat]
    return [bos + ids[: REF_PROMPT_TOKENS - 1], bos + ids[2000: 2000 + REF_PROMPT_TOKENS - 1], chat]


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def environment() -> dict:
    sha = git("rev-parse", "--verify", "-q", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    smi = ""
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap,power.limit",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except OSError:
        pass
    model_files = RESULTS / "model_files.json"
    rev = json.loads(model_files.read_text())["revision"] if model_files.exists() else "unknown"
    return {
        "git_commit": sha or "uncommitted",
        "tracked_files_dirty": dirty,
        "gpu": smi,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "model_revision": rev,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
    }


def no_tf32():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

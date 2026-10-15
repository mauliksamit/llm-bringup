#!/usr/bin/env python3
"""Download TinyLlama-1.1B-Chat-v1.0 at a pinned revision (public; no login).

Writes the files to models/<name>/ (gitignored) and records each file's
SHA-256 and size in results/model_files.json so every later result can name
the exact weights it used.

Usage: python scripts/download_model.py
"""
import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"  # pinned commit on the Hugging Face Hub
FILES = ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
         "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"]
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models" / "TinyLlama-1.1B-Chat-v1.0"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    snapshot_download(REPO, revision=REVISION, local_dir=DEST, allow_patterns=FILES)
    record = {"repo": REPO, "revision": REVISION, "license": "apache-2.0", "local_dir": "models/TinyLlama-1.1B-Chat-v1.0",
              "files": {f: {"bytes": (DEST / f).stat().st_size, "sha256": sha256(DEST / f)} for f in FILES}}
    out = ROOT / "results" / "model_files.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

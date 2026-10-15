#!/usr/bin/env python3
"""Fetch the public-domain evaluation text used for perplexity and prompts.

Source: Project Gutenberg eBook #1342, "Pride and Prejudice" by Jane Austen
(public domain in the USA). Only the novel's text from "Chapter I." on is
kept (no Project Gutenberg header, footer, or trademark), truncated to
MAX_CHARS characters. Writes data/eval_text.txt and data/eval_text.meta.json
(URL, byte range, SHA-256 of the saved text).

Usage: python scripts/fetch_eval_text.py
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

URL = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
START_MARK = "Chapter I.]"
MAX_CHARS = 40_000
ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    raw = urllib.request.urlopen(URL, timeout=60).read().decode("utf-8-sig")
    raw = raw.replace("\r\n", "\n")
    start = raw.find(START_MARK)
    if start < 0:
        print(f"start marker {START_MARK!r} not found; the source file changed. NOT RUN.")
        return 2
    text = raw[start + len(START_MARK):].lstrip()
    end_marker = text.find("*** END OF THE PROJECT GUTENBERG")
    if end_marker >= 0:
        text = text[:end_marker]
    text = text[:MAX_CHARS]
    out = ROOT / "data" / "eval_text.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    meta = {
        "source": "Project Gutenberg eBook #1342, Pride and Prejudice (Jane Austen), public domain in the USA",
        "url": URL,
        "selection": f"text after the first {START_MARK!r} marker, first {MAX_CHARS} characters, Gutenberg header/footer removed",
        "chars": len(text),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "command": "python scripts/fetch_eval_text.py",
    }
    (ROOT / "data" / "eval_text.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

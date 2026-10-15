#!/usr/bin/env python3
"""Claims audit: every measurement-like number in README.md and docs/*.md must
equal a value stored in results/ (or a data/ .meta.json) within the number's
own printed precision — or sit on a line containing NOT RUN, or be a
documented setting in scripts/audit_constants.json.

Deliberately strict:
- no free search over ratios of two values (with thousands of stored values
  that would let almost any percentage pass); derived figures the docs quote
  are precomputed by scripts/facts.py or scripts/summarize.py instead;
- raw per-iteration arrays are excluded (a claim must match a summary value,
  not happen to equal one of thousands of samples);
- allowed conversions only: fraction x 100 for "%", bytes -> GB/MB/GiB,
  microseconds <-> milliseconds.
Units checked: ms, us, µs, GiB, GB, MB, %, x / ×, tok/s.
Fenced code blocks are skipped. Exit 0 = pass, 1 = untraced number found.

Usage: python scripts/claims_audit.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_KEYS = {"iters", "iters_us", "prompt_ids", "generated_ids", "worker_total_ms_runs", "medians", "mins"}
UNIT = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s?(ms|µs|us|GiB|GB|MB|%|×|x(?![a-z0-9])|tok/s)")


def leaves(v, out, key=None):
    if key in SKIP_KEYS:
        return out
    if isinstance(v, bool):
        return out
    if isinstance(v, (int, float)):
        out.append(float(v))
    elif isinstance(v, list):
        for x in v:
            leaves(x, out)
    elif isinstance(v, dict):
        for k, x in v.items():
            leaves(x, out, k)
    return out


def traced_values():
    files = [p for p in (ROOT / "results").rglob("*.json") if "_cache" not in p.parts]
    files += list((ROOT / "data").rglob("*.meta.json"))
    vals = []
    for p in files:
        leaves(json.loads(p.read_text(encoding="utf-8")), vals)
    return sorted(set(vals)), len(files)


def decimals(s: str) -> int:
    return len(s.split(".")[1]) if "." in s else 0


def matches(v: float, unit: str, dec: int, vals) -> bool:
    tol = 0.5 * 10 ** -dec + 1e-9
    cands = []
    for t in vals:
        cands.append(t)
        if unit == "%":
            cands.append(100 * t)  # stored fractions (e.g. token match rate)
        elif unit in ("GB", "MB"):
            cands.append(t / (1e9 if unit == "GB" else 1e6))
        elif unit == "GiB":
            cands.append(t / 2**30)
        elif unit == "ms":
            cands.append(t / 1000)  # stored microseconds
        elif unit in ("us", "µs"):
            cands.append(t * 1000)  # stored milliseconds
    return any(abs(v - c) <= tol for c in cands)


def main() -> int:
    vals, nfiles = traced_values()
    consts = json.loads((ROOT / "scripts" / "audit_constants.json").read_text(encoding="utf-8"))["constants"]
    docs = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    fails, checked = [], 0
    for doc in docs:
        if not doc.exists():
            continue
        fence = False
        for i, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip().startswith("```"):
                fence = not fence
                continue
            if fence or "NOT RUN" in line:
                continue
            for m in UNIT.finditer(line):
                raw, unit = m.group(1), m.group(2)
                v = float(raw.replace(",", ""))
                if any(c["value"] == v and re.search(c["context"], line, re.I) for c in consts):
                    continue
                checked += 1
                if not matches(v, unit, decimals(raw.replace(",", "")), vals):
                    fails.append(f"{doc.relative_to(ROOT)}:{i}: {m.group(0)}  ...{line.strip()[:110]}")
    if fails:
        print(f"CLAIMS AUDIT FAILED: {len(fails)} untraced number(s)")
        for f in fails:
            print("  " + f)
        return 1
    print(f"CLAIMS AUDIT PASSED ({checked} numbers checked against {len(vals)} stored values from {nfiles} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

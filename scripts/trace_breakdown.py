#!/usr/bin/env python3
"""Where GPU time goes at each ladder step, measured from the saved traces
with perflens (github.com/mauliksamit/perflens, scripts/analyze_cli.ts).

Runs the perflens CLI on data/traces/<step>_b1_p128.json.gz for every step
and writes results/trace_breakdown.json with the perflens commit used.
Category shares come from perflens's name-based HEURISTIC classifier and are
labeled as such; idle = time in the GPU window with no kernel/memcpy/memset
recorded (CPU launch gaps), not SM utilization.

Usage: python scripts/trace_breakdown.py [path/to/perflens]   (default ../perflens)
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_bringup.common import STEPS  # noqa: E402


def main() -> int:
    perflens = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "perflens"
    traces = [ROOT / "data" / "traces" / f"{s}_b1_p128.json.gz" for s in STEPS]
    missing = [t.name for t in traces if not t.exists()]
    if missing or not (perflens / "scripts" / "analyze_cli.ts").exists():
        print(f"NOT RUN: missing traces {missing} or perflens checkout at {perflens}")
        return 2
    npx = shutil.which("npx") or "npx"
    res = subprocess.run([npx, "vite-node", "scripts/analyze_cli.ts", *map(str, traces)], cwd=perflens,
                         capture_output=True, text=True, encoding="utf-8")
    if res.returncode != 0:
        print(res.stderr)
        return res.returncode
    rows = json.loads(res.stdout)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=perflens, capture_output=True, text=True).stdout.strip()
    for step, r in zip(STEPS, rows):
        r["step"] = step
        r["kernels_per_new_token"] = r["gpu_kernels"] / 16  # 16 new tokens per traced generation (trace_step.py)
    out = {"generated_by": "scripts/trace_breakdown.py", "perflens_commit": commit,
           "workload": "batch 1, prompt 128, 16 new tokens (scripts/trace_step.py)",
           "note": "categories are perflens's name-based heuristic; idle = no GPU activity recorded",
           "steps": rows}
    (ROOT / "results" / "trace_breakdown.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for r in rows:
        i = r["idle"][0]
        b = {x["bucket"]: round(x["share_pct"], 1) for x in r["buckets_heuristic"]}
        print(f"{r['step']:18s} span {r['gpu_span_us'] / 1000:7.1f} ms  busy {i['busy_us'] / 1000:7.1f} ms  "
              f"idle {i['idle_pct']:5.1f}%  kernels/token {r['kernels_per_new_token']:6.1f}  {b}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

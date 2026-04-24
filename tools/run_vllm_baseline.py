#!/usr/bin/env python3
# ============================================================================
# tools/run_vllm_baseline.py
# ----------------------------------------------------------------------------
# Runs `vllm bench latency` once, parses the stdout report, and rewrites
# tools/artifacts/vllm_baseline.json so tools/eval_comparison.py picks up
# the fresh numbers automatically on the next invocation.
#
# The artifacts/ folder is git-ignored — this script produces a LOCAL cache,
# never a tracked artefact.  The committed source of truth for the baseline
# is the _EMBEDDED_DEFAULT dict inside tools/eval_comparison.py.
#
# This script REQUIRES a working vLLM environment — it is the only tool in
# the repo that does.  Run it inside your vLLM conda env / docker / venv,
# not in the reviewer bootstrap pipeline.
#
# Usage:
#     python tools/run_vllm_baseline.py                         # defaults
#     python tools/run_vllm_baseline.py --model /path/to/model  # custom
#     python tools/run_vllm_baseline.py --dry-run               # parse only
#
# What "parsing" means: we look for lines of the form
#
#     Avg latency: 5.0996 seconds
#     10% percentile latency: 4.9812 seconds
#     25% percentile latency: 5.0123 seconds
#     50% percentile latency: 5.2448 seconds
#     75% percentile latency: 5.3214 seconds
#     90% percentile latency: 5.4588 seconds
#     99% percentile latency: 5.4977 seconds
#
# and populate avg/p50/p99_latency_s from them.
# ============================================================================

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parent.parent
BASELINE   = REPO_ROOT / "tools" / "artifacts" / "vllm_baseline.json"

# ---------------------------------------------------------------------------
# Regex for vLLM's `bench latency` stdout format
# ---------------------------------------------------------------------------
_AVG_RE = re.compile(r"Avg\s+latency:\s*([\d.]+)\s*seconds", re.IGNORECASE)
_PCT_RE = re.compile(r"(\d+)%\s+percentile\s+latency:\s*([\d.]+)\s*seconds",
                     re.IGNORECASE)


def parse_vllm_stdout(text: str) -> dict:
    """Extract avg / p50 / p99 from the `vllm bench latency` report."""
    out: dict = {}
    m = _AVG_RE.search(text)
    if m:
        out["avg_latency_s"] = float(m.group(1))

    percentiles = {int(p): float(v) for p, v in _PCT_RE.findall(text)}
    if 50 in percentiles:
        out["p50_latency_s"] = percentiles[50]
    if 99 in percentiles:
        out["p99_latency_s"] = percentiles[99]

    if not out:
        raise RuntimeError(
            "Could not find latency lines in vLLM output — "
            "stdout format may have changed.  Dump:\n\n" + text[:2000]
        )
    return out


# ---------------------------------------------------------------------------
# vLLM command builder
# ---------------------------------------------------------------------------
def build_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "latency",
        "--model",                    args.model,
        "--batch-size",               str(args.batch_size),
        "--input-len",                str(args.input_len),
        "--output-len",               str(args.output_len),
        "--gpu-memory-utilization",   str(args.gpu_mem_util),
        "--max-model-len",            str(args.max_model_len),
        "--num-iters-warmup",         str(args.warmup),
        "--num-iters",                str(args.iters),
    ]


def probe_driver_and_gpu() -> tuple[str, str]:
    """Best-effort driver / GPU name lookup via nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        return "(nvidia-smi not found)", "(nvidia-smi not found)"
    try:
        res = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        line = res.stdout.strip().splitlines()[0]
        name, driver = [x.strip() for x in line.split(",")]
        return name, driver
    except (subprocess.SubprocessError, OSError, IndexError, ValueError):
        return "(unknown)", "(unknown)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    # Defaults match the reference command reported in the paper / REVIEWERS.md.
    # Override on the CLI for any one-off run.
    default_model = str(REPO_ROOT / "models" / "Llama-3.2-3B-Instruct")
    p.add_argument("--model",          default=default_model)
    p.add_argument("--batch-size",     type=int,   default=1)
    p.add_argument("--input-len",      type=int,   default=32)
    p.add_argument("--output-len",     type=int,   default=128)
    p.add_argument("--gpu-mem-util",   type=float, default=0.9)
    p.add_argument("--max-model-len",  type=int,   default=2048)
    # Warmup / iters mirror AtomFlow's benchmark harness (see src/main.cu:
    # BENCHMARK_WARMUP_ITERS=10, BENCHMARK_MEASURE_ITERS=20) so the two
    # engines are compared under an identical statistical protocol.
    p.add_argument("--warmup",         type=int,   default=10)
    p.add_argument("--iters",          type=int,   default=20)
    p.add_argument("--dry-run",        action="store_true",
                   help="Do not actually call vLLM; read stdin as pretend output.")
    p.add_argument("--output", type=Path, default=BASELINE,
                   help="Where to write the JSON (default: tools/artifacts/vllm_baseline.json).")
    p.add_argument("--notes", default="Refreshed automatically by tools/run_vllm_baseline.py")
    args = p.parse_args()

    # ── 1. Run vLLM (or read stdin in --dry-run) ────────────────────────────
    cmd = build_command(args)
    cmd_str = " ".join(cmd[1:])  # omit the sys.executable for readability

    if args.dry_run:
        print("[dry-run] reading fake vLLM stdout from stdin...")
        stdout = sys.stdin.read()
    else:
        print(f"[run] {cmd_str}")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except FileNotFoundError:
            sys.exit("[FATAL] `vllm` not importable — activate your vLLM env first.")
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout)
            sys.stderr.write(proc.stderr)
            sys.exit(f"[FATAL] vLLM exited with code {proc.returncode}")
        stdout = proc.stdout

    # ── 2. Parse ────────────────────────────────────────────────────────────
    parsed = parse_vllm_stdout(stdout)
    print(f"[parse] avg={parsed.get('avg_latency_s')}s  "
          f"p50={parsed.get('p50_latency_s')}s  "
          f"p99={parsed.get('p99_latency_s')}s")

    # ── 3. Merge with existing / defaults ──────────────────────────────────
    gpu_name, driver = probe_driver_and_gpu()

    record = {}
    if args.output.exists():
        try:
            record = json.loads(args.output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            record = {}

    record.update({
        "engine":          "vLLM (bench latency)",
        "command":         cmd_str,
        "hardware":        gpu_name,
        "driver":          driver,
        "date":            str(date.today()),
        "output_tokens":   args.output_len,
        "prefill_tokens":  args.input_len,
        "optimisations":   "CUDA Graph + torch.compile + PagedAttention",
        "notes":           args.notes,
    })
    record.update(parsed)

    # Reasonable default for the prefill estimate if we don't already have one.
    record.setdefault("prefill_ms_est", 35.0)
    record.setdefault("cuda",           "unknown")
    record.setdefault("version",        "vllm")

    # ── 4. Write JSON ──────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rel = args.output.relative_to(REPO_ROOT) if args.output.is_absolute() else args.output
    print(f"[write] {rel}")
    print(f"[next]  python tools/eval_comparison.py   # picks up new numbers")
    return 0


if __name__ == "__main__":
    sys.exit(main())

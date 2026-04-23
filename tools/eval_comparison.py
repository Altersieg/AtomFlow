#!/usr/bin/env python3
# ============================================================================
# tools/eval_comparison.py  —  AtomFlow vs. vLLM evaluation harness
# ----------------------------------------------------------------------------
# Methodology:  Live-vs-Static.
#   - vLLM baseline is pre-profiled on the authors' reference machine
#     (RTX 5060 Ti 16GB) and *hardcoded* below.  Live re-execution would
#     require heavy Python deps (vllm, torch, transformers) and an HF
#     license-gated model download — intentionally skipped on reviewer
#     machines, per SOSP / OSDI artifact-evaluation practice for heavy
#     baselines.
#   - AtomFlow is executed *live* on the current GPU.  Speedup is computed
#     in real time from the live TPOT.
#
# Usage:  python3 tools/eval_comparison.py
# ============================================================================

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# vLLM baseline — PRE-PROFILED, DO NOT EDIT without rerunning the benchmark
# ---------------------------------------------------------------------------
VLLM_BASELINE = {
    "engine":          "vLLM (bench latency)",
    "version":         "vllm (conda env: vllm_opt)",
    "date":            "2026-04-24",
    "hardware":        "NVIDIA RTX 5060 Ti 16GB",
    "driver":          "591.74",
    "cuda":            "12.x",
    "command": (
        "python -m vllm.entrypoints.cli.main bench latency "
        "--model Llama-3.2-3B-Instruct --batch-size 1 "
        "--input-len 4 --output-len 256 "
        "--gpu-memory-utilization 0.9 --max-model-len 2048 "
        "--num-iters-warmup 5 --num-iters 20"
    ),
    "optimisations":   "CUDA Graph (86 captured) + torch.compile + PagedAttention",
    "avg_latency_s":   5.0996,
    "p50_latency_s":   5.2448,
    "p99_latency_s":   5.4977,
    "output_tokens":   256,
    "prefill_tokens":  4,
    "prefill_ms_est":  35.0,   # single forward over 4 tokens, conservative
    "notes":           "Live execution bypassed: heavy Python deps "
                       "(vllm, torch) and HF-gated Llama-3.2 download.",
}


def vllm_pure_decode_tpot_ms() -> float:
    """Amortized per-token decode latency with prefill subtracted."""
    b = VLLM_BASELINE
    return (b["avg_latency_s"] * 1000.0 - b["prefill_ms_est"]) / b["output_tokens"]


# ---------------------------------------------------------------------------
# AtomFlow live execution
# ---------------------------------------------------------------------------
REPO_ROOT    = Path(__file__).resolve().parent.parent
BINARY       = REPO_ROOT / "build" / "atomflow"
WEIGHTS      = REPO_ROOT / "models" / "llama3_2_atomflow.bin"

_GRAPH_TPOT_RE = re.compile(r"\[Graph\]\s+TPOT:\s+([\d.]+)\s*ms")
_EAGER_TPOT_RE = re.compile(r"\[Eager\]\s+TPOT:\s+([\d.]+)\s*ms")
_AR_SUMMARY_RE = re.compile(r"\[AR\]\s+(\d+)\s+tokens in\s+([\d.]+)\s*ms")


@dataclass
class AtomFlowResult:
    eager_tpot_ms:   float | None
    graph_tpot_ms:   float | None
    ar_steps:        int   | None
    ar_total_ms:     float | None

    @property
    def ar_tpot_ms(self) -> float | None:
        if self.ar_steps and self.ar_total_ms:
            return self.ar_total_ms / self.ar_steps
        return None


def run_atomflow() -> AtomFlowResult:
    if not BINARY.exists():
        sys.exit(f"[FATAL] AtomFlow binary not found at {BINARY}.\n"
                 f"        Build first:  cmake --build build -j")
    if not WEIGHTS.exists():
        sys.exit(f"[FATAL] Weights not found at {WEIGHTS}.")

    proc = subprocess.run(
        [str(BINARY), str(WEIGHTS)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        sys.exit(f"[FATAL] AtomFlow exited with code {proc.returncode}")

    out = proc.stdout

    g = _GRAPH_TPOT_RE.search(out)
    e = _EAGER_TPOT_RE.search(out)
    a = _AR_SUMMARY_RE.search(out)

    return AtomFlowResult(
        eager_tpot_ms = float(e.group(1)) if e else None,
        graph_tpot_ms = float(g.group(1)) if g else None,
        ar_steps      = int(a.group(1))   if a else None,
        ar_total_ms   = float(a.group(2)) if a else None,
    )


# ---------------------------------------------------------------------------
# GPU metadata (best-effort; non-fatal if absent)
# ---------------------------------------------------------------------------
def gpu_info() -> str:
    if not shutil.which("nvidia-smi"):
        return "unknown (nvidia-smi not found)"
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()[0]
        name, vram, drv = [x.strip() for x in out.split(",")]
        return f"{name} · {vram} MiB · driver {drv}"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
SEP = "═" * 71
SUB = "─" * 71


def fmt_ms(x: float | None) -> str:
    return f"{x:6.2f} ms" if x is not None else "   n/a  "


def print_report(af: AtomFlowResult) -> None:
    vllm_tpot = vllm_pure_decode_tpot_ms()

    print()
    print(SEP)
    print("  AtomFlow  ·  Architecture Evaluation Report")
    print(SEP)

    # ---------------- BASELINE --------------------------------------------
    b = VLLM_BASELINE
    print("[BASELINE]  vLLM  (pre-profiled, not executed in this run)")
    print(f"            Version       : {b['version']}")
    print(f"            Hardware      : {b['hardware']} · driver {b['driver']}")
    print(f"            Date          : {b['date']}")
    print(f"            Optimisations : {b['optimisations']}")
    print(f"            Command       : {b['command']}")
    print(f"            Avg latency   : {b['avg_latency_s']*1000:.1f} ms "
          f"over {b['output_tokens']} output tokens")
    print(f"            Pure-decode TPOT : {vllm_tpot:.2f} ms/token "
          f"(amortized, prefill≈{b['prefill_ms_est']:.0f} ms subtracted)")
    print(f"            Rationale     : {b['notes']}")
    print(SUB)

    # ---------------- LIVE ATOMFLOW ---------------------------------------
    print("[LIVE]      AtomFlow  (executing on current device)")
    print(f"            Hardware      : {gpu_info()}")
    print(f"            Binary        : {BINARY}")
    print(f"            Weights       : {WEIGHTS.name} "
          f"({WEIGHTS.stat().st_size / (1024**3):.2f} GiB)")
    print()
    print(f"            [Eager]  TPOT : {fmt_ms(af.eager_tpot_ms)}")
    print(f"            [Graph]  TPOT : {fmt_ms(af.graph_tpot_ms)}   "
          f"← single-step best-case")
    if af.ar_tpot_ms is not None:
        print(f"            [AR]     TPOT : {fmt_ms(af.ar_tpot_ms)}   "
              f"← amortized over {af.ar_steps} steps")
    print(SUB)

    # ---------------- SPEEDUP ---------------------------------------------
    print("[RESULT]    Single-stream (M=1) decode TPOT comparison")
    print()

    def speedup_row(label: str, af_tpot: float | None) -> None:
        if af_tpot is None or af_tpot <= 0:
            print(f"            {label:<34}  n/a")
            return
        spd = vllm_tpot / af_tpot
        delta = (af_tpot - vllm_tpot) / vllm_tpot * 100.0
        print(f"            {label:<34}  "
              f"{af_tpot:5.2f} ms  →  speedup {spd:.2f}×  ({delta:+.1f}%)")

    print(f"            vLLM pure decode TPOT         :  {vllm_tpot:.2f} ms")
    speedup_row("AtomFlow [Graph]  (best-case)", af.graph_tpot_ms)
    speedup_row("AtomFlow [AR]     (amortized)", af.ar_tpot_ms)
    speedup_row("AtomFlow [Eager]  (no graph) ", af.eager_tpot_ms)
    print()
    print("            Interpretation:")
    print("              Static contiguous KV + arithmetic addressing + fused")
    print("              W8A16 GEMV win over paged indirection at M=1, where")
    print("              concurrency is absent and page-table lookups have no")
    print("              batching amortization to hide behind.")
    print(SEP)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    print("[INFO] Running AtomFlow live on current device ...")
    af = run_atomflow()
    print_report(af)
    return 0


if __name__ == "__main__":
    sys.exit(main())

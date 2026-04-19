#!/usr/bin/env bash
# run_inference.sh
# AtomFlow MVP single-step inference runner.
# AtomFlow MVP 单步推理运行脚本。
#
# Usage / 用法:
#   ./run_inference.sh [weights.bin]
#   Default: models/llama3_2_atomflow.bin

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${REPO_ROOT}/build"
BINARY="${BUILD_DIR}/atomflow"
WEIGHTS="${1:-${REPO_ROOT}/models/llama3_2_atomflow.bin}"

# ── 1. Sanity checks ────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  AtomFlow  ·  Inference Runner"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ ! -f "${BINARY}" ]]; then
    echo "[BUILD]  Binary not found — running cmake --build ..."
    cmake --build "${BUILD_DIR}" -j
fi

if [[ ! -f "${WEIGHTS}" ]]; then
    echo "[ERROR]  Weights file not found: ${WEIGHTS}"
    exit 1
fi

echo "[INFO]  Binary  : ${BINARY}"
echo "[INFO]  Weights : ${WEIGHTS}  ($(du -h "${WEIGHTS}" | cut -f1))"
echo ""

# ── 2. GPU info ─────────────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    echo "[GPU]"
    nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version \
               --format=csv,noheader,nounits | \
    awk -F',' '{printf "  Device : %s\n  VRAM   : %s MiB total  /  %s MiB free\n  Driver : %s\n", $1,$2,$3,$4}'
    echo ""
fi

# ── 3. Run inference ─────────────────────────────────────────────────────────
echo "[RUN]  Starting forward pass ..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
START_NS=$(date +%s%N)

"${BINARY}" "${WEIGHTS}"

END_NS=$(date +%s%N)
WALL_MS=$(( (END_NS - START_NS) / 1000000 ))

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[DONE]  Wall-clock total: ${WALL_MS} ms"
echo "        (includes weight upload, CUDA context init, and forward pass)"

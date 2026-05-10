#!/usr/bin/env bash
# ============================================================================
# scripts/bootstrap.sh  —  AtomFlow one-click evaluation pipeline
# ----------------------------------------------------------------------------
# Targeted at a FRESH reviewer machine.  Performs, in order:
#
#   [1/5] toolchain check    (CUDA >= 12.4, driver >= 550, cmake, gcc, python)
#   [2/5] GPU check          (Blackwell preferred; Ada/Ampere/Hopper fall back)
#   [3/5] fetch weights      (python3 tools/download_weights.py — stdlib only)
#   [4/5] build              (cmake --build build -j)
#   [5/5] run the demo       (./run_inference.sh — prints TPOT & tok/s)
#
# The whole pipeline has NO pip dependencies — download_weights.py is standard-library Python.
#
# Configure the HF repo containing the pre-exported weights via env var:
#   export ATOMFLOW_HF_REPO="Altersieg/test4AtomFlow"   (default)
#
# Usage:
#   bash scripts/bootstrap.sh                # full pipeline on a fresh clone
#   bash scripts/bootstrap.sh --skip-build   # reuse existing ./build/atomflow
#   bash scripts/bootstrap.sh --skip-fetch   # reuse existing ./models/
#   bash scripts/bootstrap.sh --skip-run     # stop after build (CI mode)
# ============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"

SKIP_BUILD=0
SKIP_FETCH=0
SKIP_RUN=0
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
        --skip-fetch) SKIP_FETCH=1 ;;
        --skip-run)   SKIP_RUN=1   ;;
        -h|--help)
            sed -n '2,25p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *) echo "[WARN] unknown flag: $arg" ;;
    esac
done

BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'
GRN=$'\033[32m'; YLW=$'\033[33m'; RST=$'\033[0m'

banner() { printf '\n%s━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━%s\n' "$BOLD" "$RST"; }
step()   { printf '\n%s[%s]%s %s\n' "$BOLD" "$1" "$RST" "$2"; }
ok()     { printf '   %s✓%s %s\n' "$GRN" "$RST" "$1"; }
warn()   { printf '   %s⚠%s %s\n' "$YLW" "$RST" "$1"; }
die()    { printf '   %s✗%s %s\n' "$RED" "$RST" "$1"; exit 1; }

# Numeric compare "a >= b" for X.Y version strings.
vge() { printf '%s\n%s\n' "$2" "$1" | sort -CV; }

banner
printf '%s  AtomFlow · One-Click Evaluation Pipeline%s\n' "$BOLD" "$RST"
printf '%s  repo : %s%s\n' "$DIM" "$REPO_ROOT" "$RST"
banner

# ── [1/5] Toolchain check ──────────────────────────────────────────────────
# Floors are intentionally loose — the engine builds cleanly with any CUDA /
# compiler combination that targets sm_70 or newer.  The recommended target
# is still Blackwell (sm_120); see the GPU check below for the hard gate.
step "1/5" "Toolchain check"
command -v cmake     >/dev/null || die "cmake not found — install cmake >= 3.18"
command -v gcc       >/dev/null || die "gcc not found — install gcc (any C++17-capable version, >= 9)"
command -v nvcc      >/dev/null || die "nvcc not found — install CUDA Toolkit >= 11.8 and put /usr/local/cuda/bin on PATH"
command -v python3   >/dev/null || die "python3 not found — install Python >= 3.8"

CMAKE_V=$(cmake --version | head -1 | awk '{print $3}')
GCC_V=$(gcc -dumpversion | cut -d. -f1)
NVCC_V=$(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')
PY_V=$(python3 --version | awk '{print $2}')

vge "$CMAKE_V" "3.18" || die "cmake ${CMAKE_V} is too old (need >= 3.18)"
[[ "$GCC_V" -ge 9 ]]  || die "gcc ${GCC_V} is too old (need >= 9 for C++17)"
vge "$NVCC_V" "11.8"  || die "CUDA ${NVCC_V} is too old (need >= 11.8)"

ok "cmake  ${CMAKE_V}"
ok "gcc    ${GCC_V}"
ok "nvcc   ${NVCC_V}   (>=12.4 recommended for native sm_120)"
ok "python ${PY_V}"

# ── [2/5] GPU & driver check ───────────────────────────────────────────────
# Hard requirement: Tensor-Core-capable GPU  ==  compute capability >= 7.0
# (Volta V100, Turing T4/RTX 20, Ampere RTX 30/A100, Ada RTX 40, Hopper H100,
#  Blackwell RTX 50).  Anything older has no Tensor Cores and the W8A16 path
# would fall off a performance cliff.
step "2/5" "GPU & driver check (Tensor-Core gate)"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found — NVIDIA driver not installed or not on PATH"

GPU_NAME=$(nvidia-smi --query-gpu=name          --format=csv,noheader | head -1)
GPU_MEM=$( nvidia-smi --query-gpu=memory.total  --format=csv,noheader,nounits | head -1)
DRV_V=$(   nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
CAP=$(     nvidia-smi --query-gpu=compute_cap   --format=csv,noheader 2>/dev/null | head -1 || echo "")

ok "GPU    ${GPU_NAME}  (${GPU_MEM} MiB, compute cap ${CAP:-unknown})"
ok "Driver ${DRV_V}"

# Enforce compute cap >= 7.0 (Tensor Cores).  Bash float compare via awk.
if [[ -n "$CAP" ]]; then
    if awk -v c="$CAP" 'BEGIN{exit !(c+0 >= 7.0)}'; then
        :  # ok, has Tensor Cores
    else
        die "GPU compute capability ${CAP} has no Tensor Cores — need >= 7.0 (Volta/Turing/Ampere/Ada/Hopper/Blackwell)"
    fi
    # Advisory: sm_120 native path is the reference config
    awk -v c="$CAP" 'BEGIN{exit !(c+0 >= 12.0)}' \
        && ok "compute cap ${CAP} — native sm_120 path available" \
        || warn "compute cap ${CAP} < 12.0 — driver will JIT from sm_89 PTX, numbers may differ from the report"
fi

if [[ "${GPU_MEM}" =~ ^[0-9]+$ ]] && [[ "${GPU_MEM}" -lt 8000 ]]; then
    warn "GPU has < 8 GiB — the 3B model needs ~5.5 GiB; leave other processes off the GPU"
fi

# ── [3/5] Fetch weights + tokenizer ────────────────────────────────────────
step "3/5" "Fetch weights (≈ 4.9 GiB)"
if [[ $SKIP_FETCH -eq 1 ]]; then
    ok "--skip-fetch: trusting existing models/"
else
    python3 "${REPO_ROOT}/tools/download_weights.py"
fi

# Ground-truth input embeddings are shipped with the repo (~12 KB) —
# refuse to run if they're missing so we fail loudly instead of segfaulting.
GT="${REPO_ROOT}/ground_truth/gt_input_embeddings.bin"
[[ -f "$GT" ]] || die "missing $GT — re-clone the repo (this file is version-controlled)"
ok "ground_truth/gt_input_embeddings.bin present"

# ── [4/5] Build ────────────────────────────────────────────────────────────
step "4/5" "Build"
if [[ $SKIP_BUILD -eq 1 && -x "${BUILD_DIR}/atomflow" ]]; then
    ok "--skip-build: reusing ${BUILD_DIR}/atomflow"
else
    if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
        (cd "$REPO_ROOT" && cmake -S . -B "$BUILD_DIR" -GNinja 2>/dev/null \
                          || cmake -S . -B "$BUILD_DIR")
    fi
    cmake --build "$BUILD_DIR" -j
    [[ -x "${BUILD_DIR}/atomflow" ]] || die "build produced no ${BUILD_DIR}/atomflow — check compiler errors above"
    ok "built ${BUILD_DIR}/atomflow"
fi

# ── [5/5] Run the demo ─────────────────────────────────────────────────────
if [[ $SKIP_RUN -eq 1 ]]; then
    step "5/5" "Run demo"
    ok "--skip-run: build artefacts ready; run ./run_inference.sh to benchmark"
else
    step "5/5" "Run demo (./run_inference.sh — prints TPOT and tok/s)"
    (cd "$REPO_ROOT" && ./run_inference.sh)
fi

banner
printf '%s  Pipeline complete.  Expected headline: ~65-70 tok/s on RTX 5060 Ti.%s\n' "$BOLD$GRN" "$RST"
banner

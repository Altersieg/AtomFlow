#!/usr/bin/env bash
# ============================================================================
# scripts/bootstrap.sh  —  AtomFlow one-click evaluation runner
# ----------------------------------------------------------------------------
# Four phases:
#   [1/4] toolchain check   (cuda, cmake, gcc, python)
#   [2/4] build             (cmake --build build -j)
#   [3/4] fetch weights     (python3 tools/fetch_weights.py)
#   [4/4] evaluate          (python3 tools/eval_comparison.py)
#
# Configure the HF repo containing the pre-exported weights via env:
#   export ATOMFLOW_HF_REPO="yourname/atomflow-llama3.2-3b"
#
# Usage:
#   bash scripts/bootstrap.sh                # full pipeline
#   bash scripts/bootstrap.sh --skip-build   # reuse existing ./build/atomflow
#   bash scripts/bootstrap.sh --skip-fetch   # reuse existing models/
# ============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"

SKIP_BUILD=0
SKIP_FETCH=0
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
        --skip-fetch) SKIP_FETCH=1 ;;
        -h|--help)
            sed -n '2,17p' "${BASH_SOURCE[0]}"
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

banner
printf '%s  AtomFlow · One-Click Evaluation Pipeline%s\n' "$BOLD" "$RST"
printf '%s  repo : %s%s\n' "$DIM" "$REPO_ROOT" "$RST"
banner

# ── [1/4] Toolchain check ──────────────────────────────────────────────────
step "1/4" "Toolchain check"
command -v cmake     >/dev/null || die "cmake not found (need >= 3.24)"
command -v gcc       >/dev/null || die "gcc   not found (need >= 11)"
command -v nvcc      >/dev/null || die "nvcc  not found (CUDA Toolkit >= 12.4)"
command -v python3   >/dev/null || die "python3 not found"
ok "cmake $(cmake --version | head -1 | awk '{print $3}')"
ok "gcc   $(gcc --version | head -1 | awk '{print $NF}')"
ok "nvcc  $(nvcc --version | grep release | awk '{print $6}')"
ok "python $(python3 --version | awk '{print $2}')"

if command -v nvidia-smi >/dev/null; then
    gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    ok "GPU   $gpu"
else
    warn "nvidia-smi not found; cannot verify GPU presence"
fi

# ── [2/4] Build ────────────────────────────────────────────────────────────
step "2/4" "Build"
if [[ $SKIP_BUILD -eq 1 && -x "${BUILD_DIR}/atomflow" ]]; then
    ok "--skip-build: reusing ${BUILD_DIR}/atomflow"
else
    if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
        (cd "$REPO_ROOT" && cmake -S . -B "$BUILD_DIR")
    fi
    cmake --build "$BUILD_DIR" -j
    [[ -x "${BUILD_DIR}/atomflow" ]] || die "build produced no ${BUILD_DIR}/atomflow"
    ok "built ${BUILD_DIR}/atomflow"
fi

# ── [3/4] Fetch weights ────────────────────────────────────────────────────
step "3/4" "Fetch weights"
if [[ $SKIP_FETCH -eq 1 ]]; then
    ok "--skip-fetch: trusting existing models/"
else
    python3 "${REPO_ROOT}/tools/fetch_weights.py"
fi

# ── [4/4] Evaluate ─────────────────────────────────────────────────────────
step "4/4" "Evaluate (AtomFlow live vs. vLLM pre-profiled)"
python3 "${REPO_ROOT}/tools/eval_comparison.py"

banner
printf '%s  Pipeline complete.%s\n' "$BOLD$GRN" "$RST"
banner

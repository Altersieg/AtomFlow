# AtomFlow — Reviewer Quickstart

> **One-line reproduction** on a fresh machine that meets the prerequisites below:
> ```bash
> git clone https://github.com/Altersieg/AtomFlow.git && cd AtomFlow && bash scripts/bootstrap.sh
> ```
> Expected wall-clock on the first run: **5–8 minutes** (4.9 GiB weight download + CMake/Ninja compile + benchmark).
> Expected headline output: **~65–70 tokens / second** on an RTX 5060 Ti.

---

## 1 · Hardware prerequisites

| Component   | Minimum (hard gate)                          | Recommended (reference run)            | Notes |
|-------------|-----------------------------------------------|----------------------------------------|-------|
| GPU         | **Any Tensor-Core GPU (compute cap ≥ 7.0)** — Volta, Turing, Ampere, Ada, Hopper, Blackwell | **Blackwell (sm_120, RTX 50-series)** | On anything older than Blackwell the driver JITs from sm_89 PTX — correctness unchanged, numbers differ |
| VRAM        | 8 GiB                                         | 16 GiB                                 | Model needs ~5.5 GiB; leave the card otherwise idle |
| Host RAM    | 8 GiB                                         | 16 GiB                                 | Only used during weight load |
| Disk        | 10 GiB free                                   | 20 GiB free                            | Weight cache + build dir |

> The benchmark runs on **any Tensor-Core NVIDIA GPU from Volta (2017) onwards**. Numbers in the report are for RTX 5060 Ti — older cards will be slower, newer ones faster.

## 2 · Software prerequisites

| Tool            | Minimum (hard gate) | Recommended   | Install hint (Ubuntu 22.04/24.04)                                 |
|-----------------|---------------------|---------------|--------------------------------------------------------------------|
| CUDA Toolkit    | **11.8**            | 12.4 or newer | `sudo apt install cuda-toolkit-12-6` (adds `nvcc`)                 |
| NVIDIA driver   | whatever ships with your CUDA Toolkit | R550+ for native sm_120 | `sudo ubuntu-drivers install` (reboot after) |
| CMake           | 3.18                | 3.24          | `sudo apt install cmake ninja-build`                               |
| GCC             | 9 (C++17)           | 11+           | `sudo apt install build-essential`                                 |
| Python          | 3.8                 | 3.10+         | `sudo apt install python3` (no pip deps needed)                    |
| `git-lfs`       | *(not required)*    | —             | Small files ship in the repo; weights come from Hugging Face Hub   |

**Zero pip dependencies for the basic pipeline.** `fetch_weights.py` and `eval_comparison.py` use only the Python standard library.

## 3 · What `bootstrap.sh` actually does

The script is a single `bash` entry point with five phases:

```
[1/5] Toolchain check  — verify CUDA / CMake / GCC / Python versions
[2/5] GPU check        — verify driver + compute capability via nvidia-smi
[3/5] Fetch weights    — download ~4.9 GiB from huggingface.co/Altersieg/test4AtomFlow
                         (stdlib urllib, resumable, SHA-256 verified)
[4/5] Build            — cmake -B build + cmake --build build -j
[5/5] Run demo         — ./run_inference.sh, which prints the TPOT / tok/s table
```

Every phase fails loudly with a human-readable hint. Partial runs can be resumed:

| Flag            | Use case                                                   |
|-----------------|------------------------------------------------------------|
| `--skip-fetch`  | Weights already in `models/` from a previous run           |
| `--skip-build`  | Binary already at `build/atomflow`                         |
| `--skip-run`    | CI / artefact evaluation — build only, don't benchmark     |
| `--with-eval`   | Also run `tools/eval_comparison.py` (vs. vLLM baseline)    |

## 4 · Expected output

After a successful run you will see (abbreviated):

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  AtomFlow · One-Click Evaluation Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[1/5] Toolchain check
   ✓ cmake  3.28.3
   ✓ gcc    13
   ✓ nvcc   12.6
   ✓ python 3.12.3
[2/5] GPU & driver check
   ✓ GPU    NVIDIA GeForce RTX 5060 Ti  (16311 MiB, compute cap 12.0)
   ✓ Driver 591.74
[3/5] Fetch weights (≈ 4.9 GiB)
   ✓ models/llama3_2_atomflow.bin  (5.17 GiB, SHA-256 OK)
   ✓ ground_truth/gt_input_embeddings.bin present
[4/5] Build
   ✓ built build/atomflow
[5/5] Run demo
   ...
   ║  [Eager]  TPOT:    15.148 ms  (  66.0 tok/s)            ║
   ║  [Graph]  TPOT:    15.188 ms  (  65.8 tok/s)            ║
   [AR]  20 tokens in 335.8 ms  (59.6 tok/s avg)

  Pipeline complete.  Expected headline: ~65-70 tok/s on RTX 5060 Ti.
```

## 5 · What to look at if you have only 5 minutes

1. `src/kernel/qkv_gemm.cu` — the fused W8A16 GEMV kernel (the biggest architectural win).
2. `src/kernel/tiled_attention.cu` — custom GQA attention for M = 1.
3. `src/core/engine.cu` — static three-pool arena (4 `cudaMalloc` calls total).
4. `tools/eval_comparison.py` — head-to-head vs. vLLM methodology.

## 6 · Common failure modes

| Symptom                                   | Cause & fix                                                      |
|-------------------------------------------|------------------------------------------------------------------|
| `nvcc not found`                          | CUDA Toolkit not installed or `/usr/local/cuda/bin` not on PATH   |
| `driver version is insufficient`          | Driver < R550; upgrade with `sudo ubuntu-drivers install`         |
| `SHA-256 mismatch` on weight download     | Partial/corrupt download — delete `models/*.bin` and rerun        |
| `CUDA error: invalid device function`     | GPU has no Tensor Cores (compute cap < 7.0); AtomFlow requires Volta or newer |
| `cudaErrorOutOfMemory`                    | GPU < 8 GiB VRAM, or another process is holding memory            |

## 7 · Optional — re-running the vLLM baseline yourself

**You do not need to do this to evaluate AtomFlow.**  The vLLM numbers in the comparison table were measured by the authors on the same RTX 5060 Ti used for AtomFlow and are committed to the repo in two places:

- `tools/eval_comparison.py` → `_EMBEDDED_DEFAULT`  (the source of truth, always available)
- `tools/artifacts/vllm_baseline.json`  (a local cache, regenerated by the script below)

The primary evaluation path (`bash scripts/bootstrap.sh`) reads these committed numbers and runs AtomFlow live.  **No vLLM installation is required.**

If you wish to verify the baseline independently — for instance because you want to re-measure on your own hardware — the steps are:

1. Obtain a Llama-3.2-3B-Instruct checkpoint in Hugging Face format (`config.json` + `.safetensors`).  We do not redistribute Meta's weights; access is governed by Meta's licence.  Options:
   - Apply for access on Hugging Face (gated model), or
   - Pull a community mirror such as `LLM-Research/Llama-3.2-3B-Instruct` on ModelScope.
   Place the files under `models/Llama-3.2-3B-Instruct/`.
2. Install vLLM in its own environment:
   ```bash
   conda create -n vllm_env python=3.12 -y && conda activate vllm_env
   pip install vllm
   ```
3. Run the wrapper, which mirrors AtomFlow's benchmark protocol (10 warm-up, 20 measured iterations):
   ```bash
   python tools/run_vllm_baseline.py
   ```
   This invokes `vllm bench latency` with the exact command printed in `eval_comparison.py`, parses the `Avg / p50 / p99 latency` lines, and rewrites `tools/artifacts/vllm_baseline.json`.  Re-running `python tools/eval_comparison.py` then uses the fresh numbers automatically.

If you prefer to skip this step, the embedded baseline remains valid — it was measured on the same hardware and driver listed in §1 of this document.

## 8 · Contact

Source: <https://github.com/Altersieg/AtomFlow>
Issues:  open a GitHub issue with the full bootstrap output attached.

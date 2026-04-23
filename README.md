# AtomFlow

**A Bare-Metal, Zero-Overhead LLM Inference Engine for Consumer GPUs**

**🌐 Language:** **English** | [简体中文](./README.zh-CN.md)

---

## Abstract

AtomFlow is a from-scratch CUDA C++ inference engine that achieves **68.7 tok/s** autoregressive decode of **Llama 3.2 3B** on a single **NVIDIA RTX 5060 Ti** (16 GB GDDR7, 448 GB/s, Blackwell).

The engine targets the physical reality of $M=1$ (single-token) decode: at arithmetic intensity < 1, performance is **entirely memory-bandwidth-bound**. Every architectural decision — static arenas, fused W8A16 GEMV, contiguous KV cache — is a direct consequence of this constraint. No dynamic allocation, no indirection, no framework.

| Metric | Value |
|--------|-------|
| Model | Llama 3.2 3B · 28L · GQA 24Q/8KV · V=128K |
| Hardware | RTX 5060 Ti · 16 GB GDDR7 · 448 GB/s |
| TPOT (CUDA Graph) | 13.3–15.2 ms → **66–75 tok/s** |
| AR Decode (KV cache) | **65–69 tok/s** |
| Total VRAM | 5.5 GiB (34% of 16 GB) |
| W8A16 GEMV Peak BW | 233 GB/s (52% of theoretical) |
| Runtime cudaMalloc | **0** |
| Numerical Accuracy | 37/37 layer probes PASS |

### Benchmark vs. vLLM

Head-to-head comparison against the industry reference **vLLM** (with full compilation optimisations enabled: `torch.compile` + CUDA Graph capture), on the same single **RTX 5060 Ti** hardware, Llama 3.2 3B, $M=1$ decode.

| Engine | TPOT (ms) | Notes |
|--------|-----------|-------|
| **AtomFlow** | **~14.5** | Static contiguous KV, fused W8A16 GEMV, CUDA Graph |
| vLLM (optimised) | ~16.1 | `torch.compile` + CUDA Graph + PagedAttention |
| **Advantage** | **−11.5%** | AtomFlow wins on per-step latency |

**Interpretation.** In the single-stream ($M=1$), non-concurrent regime, contiguous arithmetic addressing of the KV cache strictly dominates paged indirection: no page-table lookups, no TLB pressure, no scatter loads. The 11.5% latency advantage is the empirical cost of PagedAttention's flexibility when concurrency is not needed. PagedAttention remains optimal for multi-request batching; AtomFlow targets the opposite end of the design space.

---

## Architecture 1: AtomFlow Core — The Extreme Performance Zone

The production single-stream engine. Zero abstraction overhead. Every byte of VRAM accounted for at compile time.

### Key Architectural Decisions

- **Static Arena Allocator**
  - 4× `cudaMalloc` at init, 0 during inference.
  - Weight pool (5.17 GiB) + Activation arena (1.5 MiB) + KV Cache pool (224 MiB) + cuBLAS workspace (4 MiB).
  - All Views are POD descriptors pointing into pre-allocated arenas — no heap, no `new`, no `shared_ptr`.

- **IO-Aware Kernel Fusion**
  - Fused W8A16 GEMV: FP8-E4M3 dequant + FP16 dot-product in a single kernel pass.
  - Intermediate dequantized weights exist ONLY in registers — **zero global memory writes**.
  - Activation arena reduced from 49.5 MiB (with explicit dequant workspace) to **1.5 MiB**.

- **Static KV Cache**
  - Contiguous `[MAX_SEQ_LEN, KV_DIM]` per layer — pure arithmetic addressing: `cache[t × stride + head × HD + d]`.
  - No page tables, no indirect loads, no TLB pressure from scattered blocks.
  - Custom GQA attention kernel: tree-reduction Q·Kᵀ + shared-memory softmax + parallel V accumulation.

- **CUDA Graph Capture**
  - 280+ kernel launches recorded into a single `cudaGraphExec_t`.
  - CPU dispatch overhead eliminated entirely — measured 1.04–1.06× speedup over eager.

- **mmap Weight Loading**
  - Binary model file (`4.87 GiB`) mapped via `mmap` → single bulk `cudaMemcpy` to device pool.
  - Zero host-side allocation for weights.

---

## Architecture 2: Server Sandbox — The Experimental Zone

An **isolated CMake target** (`atomflow-server-test`) for evaluating server-class scheduling overhead.

**Purpose:**

Provide a headless simulation environment to quantify the physical cost of:

- **Paged KV Cache** — non-contiguous physical blocks + page-table indirection.
- **Block Managers** — free-list allocation/recycling with copy-on-write reference counting.
- **Continuous Batching** — dynamic request scheduling with variable sequence lengths.

**Design constraint:**

The sandbox shares ONLY the low-level math kernels (`src/kernel/*`) with the core engine. It does NOT link against `engine.cu` or `main.cu`. No inheritance bridges the two — they are physically separate compilation targets.

```bash
cmake --build build -j --target atomflow-server-test
./build/atomflow-server-test
```

---

## Build & Run

**Requirements**

- CUDA Toolkit ≥ 12.4
- CMake ≥ 3.24
- GCC ≥ 11 (C++17)
- GPU: Blackwell (`sm_120`)

**Quick start**

```bash
git clone https://github.com/altersieg/AtomFlow.git
cd AtomFlow
cmake -S . -B build
cmake --build build -j

./build/atomflow                # benchmark + AR generation
python tools/decode_tokens.py   # decode token IDs to text
```

**Validation mode** (compare every layer against HuggingFace ground truth):

```bash
cmake -S . -B build -DCMAKE_CUDA_FLAGS="-DENABLE_VALIDATOR=1"
cmake --build build -j
./build/atomflow    # Expect: 37/37 PASS, token 791
```

**Build modes**

| Mode | Command | Notes |
|------|---------|-------|
| MVP (default) | `cmake -S . -B build` | Separable compilation, fast iteration |
| Peak performance | `cmake -S . -B build -DATOMFLOW_MONOLITHIC_BUILD=ON` | LTO + cross-TU inlining |
| With profiler | `-DCMAKE_CUDA_FLAGS="-DENABLE_PROFILER=1"` | Per-kernel cudaEvent timing |

---

## Project Structure

`src/kernel/*` is the shared boundary between the Core engine and the Server Sandbox: both CMake targets (`atomflow` and `atomflow-server-test`) link the same low-level math kernels, but nothing else is shared.

```
AtomFlow/
├── CMakeLists.txt                  # Dual-target build: atomflow + atomflow-server-test
├── pyproject.toml                  # PEP 621 metadata (setuptools_scm dynamic version)
├── setup.py                        # torch.utils.cpp_extension CUDA build hook
├── run_inference.sh                # End-to-end convenience runner
├── include/
│   ├── core/
│   │   ├── engine.h                # AtomFlowEngine (init, forward_pass, KV cache)
│   │   ├── model_state.h           # LayerWeights & ActBuffers (POD structs)
│   │   ├── view.h                  # POD tensor descriptor (dims, strides, dtype)
│   │   ├── qkvview.h               # Zero-copy QKV slicing over qkv_out
│   │   ├── atom_context.h          # Global runtime context (handles, streams)
│   │   └── config.h                # Compile-time ModelConfig constants
│   ├── ops/
│   │   └── kernel.h                # All kernel launcher declarations
│   ├── memory/
│   │   ├── weight_loader.h         # mmap-based weight loading
│   │   ├── alloc.h                 # Arena bump-allocator helpers
│   │   ├── memory_planner.h        # Static arena size planning
│   │   └── block_manager.h         # Placeholder for sandbox block mgmt
│   └── utils/
│       ├── utils.h                 # CUDA_CHECK / CUBLAS_CHECK macros
│       ├── profiler.h              # EngineProfiler (deferred cudaEvent timing)
│       └── validator.h             # Ground-truth validation (cosine similarity)
├── src/
│   ├── main.cu                     # CLI + benchmark harness + AR loop + token logging
│   ├── core/
│   │   └── engine.cu               # AtomFlowEngine full implementation
│   ├── kernel/                     # ← SHARED BOUNDARY: linked by both targets
│   │   ├── helpers.cu              # Embed lookup, RoPE cache, KV cache writer
│   │   ├── qkv_gemm.cu             # Fused W8A16 GEMV + cuBLAS lm_head GEMM
│   │   ├── tiled_attention.cu      # GQA cached attention (M=1 optimised)
│   │   ├── rmsnorm.cu              # FP16 RMSNorm
│   │   ├── rope.cu                 # Rotary Position Embedding (in-place)
│   │   ├── residual_add.cu         # Element-wise residual add
│   │   ├── swiglu.cu               # SwiGLU activation
│   │   ├── argmax.cu               # Greedy argmax sampling
│   │   └── layer.cu                # Layer-level helpers
│   ├── experimental/
│   │   └── server_arch/            # Sandbox sources (target: atomflow-server-test)
│   │       ├── server_engine.h     # ServerEngine class (sandbox)
│   │       ├── server_engine.cu    # Entry point for atomflow-server-test
│   │       └── paged_kv_manager.h  # KVBlock, PageTable, BlockManager stubs
│   └── utils/
│       ├── profiler.cpp            # EngineProfiler implementation
│       └── validator.cpp           # Ground-truth comparison
├── tools/
│   ├── export_atomflow.py          # HF Llama 3.2 → AtomFlow .bin (AWQ + FP8 GS=128)
│   ├── dump_ground_truth.py        # Per-layer activation dump for validator
│   └── decode_tokens.py            # Offline token → text decoder (local tokenizer)
├── models/                         # Local model weights + tokenizer files
└── ground_truth/                   # Dumped per-layer FP32 activations for validator
```

---

## Roadmap

Scope is deliberately narrow: items below are either shipped or on the 2–3 month horizon. Long-horizon wishlist items have been removed to keep the roadmap honest.

### Done

- [x] Static arena allocator (weight + activation + KV cache pools)
- [x] Fused W8A16 GEMV (zero intermediate DRAM writes, 233 GB/s peak)
- [x] Static KV Cache + GQA cached attention (autoregressive decode)
- [x] CUDA Graph capture (280+ kernels → single graph dispatch)
- [x] Per-layer numerical validation (37/37 PASS)
- [x] Token logging + offline Python decoder
- [x] Isolated Server Sandbox CMake target (`atomflow-server-test`)
- [x] ServerEngine scaffolding + BlockManager / PageTable stubs

### Next (2–3 months)

- [ ] **PagedAttention kernel in the sandbox** — quantitative latency comparison vs. the static contiguous KV kernel (the comparison itself is the deliverable)
- [ ] **Additional prologue/epilogue fusions** — e.g. LayerNorm + MatMul, SwiGLU + down-projection; track the boundary where register pressure starts to negate fusion gains

---

## Design Philosophy

- **Physics First** — Every decision derived from hardware constraints (448 GB/s BW, 16 GB VRAM, $M=1$ arithmetic intensity < 1). For example, since $M=1$ decode is bandwidth-bound, any abstraction that adds DRAM round-trips is rejected by default.
- **Zero Runtime Allocation** — All memory statically partitioned at init; inference is pure kernel dispatch.
- **Fusion-First** — Default to kernel fusion. When two kernels share data, the burden of proof is on the decision to keep them separate. Current fusions include W8A16 GEMV (FP8 dequant + FP16 dot in registers) and the custom GQA attention kernel (Q·Kᵀ + softmax + V-accumulate in shared memory). Ongoing work tracks the boundary between "aggressive fusion pays" and "register pressure negates the gain."
- **Measure, Don't Assume** — Built-in profiler (`ENABLE_PROFILER=1`) and validator (`ENABLE_VALIDATOR=1`) for every claim.
- **Dual Architecture (in progress)** — The Production engine prioritizes $M=1$ single-stream decode. A parallel Server Sandbox target is being built to quantify the overhead of paging/batching vs. static contiguous layouts. The comparison itself is the deliverable; it is not a production serving system.

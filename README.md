# AtomFlow (AF) Engine

**🌐 Language:** **English** | [简体中文](./README.zh-CN.md)

---

A minimal, from-scratch LLM inference engine for **NVIDIA Ada / Blackwell** GPUs,
built to expose the physical mechanics of modern inference stacks (FP8 quantisation,
arena allocation, fused GEMV kernels) in a few thousand lines of readable C++/CUDA.

**Current status:** End-to-end **Llama 3.2 3B** single-token decode at **~58 tok/s**
(pure GPU, W8A16 fused GEMV, RTX 5060 Ti 16 GB).

## Build & Run

**Requirements**

- CUDA Toolkit ≥ 12.4
- CMake ≥ 3.24
- GCC ≥ 11 (C++17)
- GPU: Ada Lovelace (`sm_89`) or Blackwell (`sm_120`)

**Quick start**

```bash
git clone https://github.com/altersieg/AtomFlow.git
cd AtomFlow
cmake -S . -B build
cmake --build build -j
./build/atomflow                    # pure benchmark mode (ENABLE_VALIDATOR=0)
```

**Validation mode** (compare every layer against HuggingFace ground truth):

```bash
cmake -S . -B build -DCMAKE_CUDA_FLAGS="-DENABLE_VALIDATOR=1 -DENABLE_PROFILER=1"
cmake --build build -j
./build/atomflow
```

**Build modes**

| Mode              | Command                                                          | When to use                          |
| ----------------- | ---------------------------------------------------------------- | ------------------------------------ |
| MVP (default)     | `cmake -S . -B build`                                            | Fast iteration, separable compilation |
| Peak performance  | `cmake -S . -B build -DATOMFLOW_MONOLITHIC_BUILD=ON`             | LTO + aggressive cross-TU inlining    |

## Project Structure

```
AtomFlow/
├── CMakeLists.txt              # CMake: sm_89, MVP / Monolithic build switch
├── include/
│   ├── core/
│   │   ├── engine.h            # AtomFlowEngine class (init, forward_pass, cleanup)
│   │   ├── model_state.h       # LayerWeights & ActBuffers POD structs
│   │   ├── view.h              # POD tensor descriptor (dims, strides, dtype)
│   │   ├── atom_context.h      # Global runtime context
│   │   ├── qkvview.h           # Zero-copy QKV slicing
│   │   └── config.h            # ModelConfig
│   ├── ops/
│   │   └── kernel.h            # All kernel launcher declarations
│   ├── memory/
│   │   └── weight_loader.h     # mmap-based zero-copy weight loading
│   └── utils/
│       ├── utils.h             # CUDA_CHECK / CUBLAS_CHECK macros
│       ├── profiler.h          # EngineProfiler (deferred cudaEvent timing)
│       └── validator.h         # Ground-truth cosine-similarity validation
├── src/
│   ├── main.cu                 # Lightweight CLI + benchmark harness (~100 lines)
│   ├── core/
│   │   └── engine.cu           # AtomFlowEngine implementation
│   ├── kernel/
│   │   ├── helpers.cu          # Embed lookup, FP16↔FP32 cast, RoPE cache, file I/O
│   │   ├── qkv_gemm.cu         # Fused W8A16 GEMV kernel + cuBLAS GEMM fallback
│   │   ├── rmsnorm.cu          # FP16 RMSNorm
│   │   ├── rope.cu             # Rotary Position Embedding (in-place)
│   │   ├── tiled_attention.cu  # Shared-memory tiled attention
│   │   ├── residual_add.cu     # Element-wise residual add
│   │   ├── swiglu.cu           # SwiGLU activation
│   │   ├── argmax.cu           # Greedy argmax for next-token sampling
│   │   ├── dequant.cu          # FP8→FP16 dequantisation (legacy, fused path preferred)
│   │   └── layer.cu            # Layer-level helpers
│   └── utils/
│       ├── profiler.cpp        # EngineProfiler implementation
│       └── validator.cpp       # Ground-truth comparison utilities
└── tools/
    ├── export_atomflow.py      # Export HF Llama 3.2 → AtomFlow .bin (AWQ + FP8)
    └── dump_ground_truth.py    # Dump per-layer activations for validation
```

## Roadmap

- [x] `View` + zero-copy QKV slicing
- [x] RMSNorm, RoPE, SwiGLU, residual_add kernels
- [x] Tiled attention kernel (shared-memory)
- [x] FP8 weight export with AWQ smoothing (`export_atomflow.py`)
- [x] End-to-end Llama 3.2 3B single-token decode (28 layers + lm_head)
- [x] **Fused W8A16 GEMV kernel** (FP8 dequant + dot-product in one pass, zero intermediate writes)
- [x] Modular `AtomFlowEngine` class (init / forward / cleanup)
- [x] Pure GPU benchmark mode (`ENABLE_VALIDATOR=0`, `ENABLE_PROFILER=0`)
- [x] Per-layer ground-truth validation (37/37 PASS)
- [ ] KV cache for multi-token generation
- [ ] CUDA Graphs for decode loop
- [ ] RadixAttention KV-cache sharing
- [ ] Tensor Core `mma` attention path
- [ ] Python bindings (PyBind11)
- [ ] Multi-GPU via NCCL

## Design Philosophy

- **Simple Infrastructure** — POD-based `View` system for zero-overhead tensor management.
- **Cool Kernels** — fused W8A16 GEMV with vectorised FP8 loads, warp-shuffle reduction, zero intermediate global writes.
- **Minimalist Dispatch** — direct mapping from logical views to hardware grids, no framework overhead.
- **Industrial Modularity** — `AtomFlowEngine` encapsulates all GPU state; `main.cu` is a thin ~100-line harness.

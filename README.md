# AtomFlow (AF) Engine

**🌐 Language:** **English** | [简体中文](./README.zh-CN.md)

---

A minimal, from-scratch LLM inference engine for **NVIDIA Ada / Blackwell** GPUs,
built to expose the physical mechanics of modern inference stacks (FP8 Tensor Cores,
arena allocation, paged KV cache, RadixAttention) in a few thousand lines of readable
C++/CUDA.

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
├── CMakeLists.txt           # CMake config: sm_89, MVP / Monolithic build switch
├── include/
│   ├── view.h               # POD tensor descriptor (dims, strides, dtype)
│   ├── alloc.h              # ArenaAllocator: one-shot cudaMalloc pool
│   ├── memory_planner.h     # Stack-style activation allocator
│   ├── atom_context.h       # Global runtime context (model + arenas + streams)
│   ├── kernel.h             # Kernel launcher declarations
│   ├── qkvview.h            # Zero-copy QKV slicing
│   ├── config.h             # ModelConfig
│   ├── block_manager.h      # KV-cache block allocator (WIP)
│   └── operators.h          # Operator-level helpers
└── src/
    ├── main.cu              # Entry point / skeleton driver
    └── kernel/
        ├── rmsnorm.cu       # FP16 / FP8 RMSNorm with CUB block reduce
        ├── qkv_gemm.cu      # FP8 weights × FP16 activations via cuBLAS
        ├── rope.cu          # Rotary Position Embedding (in-place)
        ├── tiled_attention.cu  # Shared-memory tiled attention (FlashAttn-lite)
        ├── residual_add.cu  # Element-wise residual add
        ├── swiglu.cu        # SwiGLU activation
        └── layer.cu         # Single Transformer layer orchestration
```

## Roadmap

- [x] `ArenaAllocator` + `MemoryPlanner`
- [x] `View` + zero-copy QKV slicing
- [x] RMSNorm kernel (FP16 / FP8)
- [x] QKV fused GEMM (cuBLAS, FP8 weights)
- [x] RoPE kernel
- [x] CMake build system (MVP + Monolithic/LTO modes)
- [ ] Tiled attention kernel (shared-memory tiling done; tensor-core `mma` pending)
- [ ] SwiGLU + residual_add launcher wrappers
- [ ] Full single-layer forward pass (`layer_forward_naive`)
- [ ] End-to-end Llama 3.2 3B inference
- [ ] RadixAttention KV-cache sharing (atomic refcount + CoW)
- [ ] Structured decoding FSM fused into softmax
- [ ] Python bindings (PyBind11)
- [ ] Multi-GPU via NCCL

## Design Philosophy

- **Simple Infrastructure** — POD-based `View` system for zero-overhead tensor management.
- **Cool Kernels** — aggressive optimisation with FP8, vectorised memory access, CUB primitives.
- **Minimalist Dispatch** — direct mapping from logical views to hardware grids.

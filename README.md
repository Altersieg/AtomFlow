# AtomFlow (AF) Engine

High-performance LLM inference engine optimized for NVIDIA Blackwell GPUs.

## Design Philosophy
- **Simple Infrastructure**: POD-based `View` system for zero-overhead tensor management.
- **Cool Kernels**: Aggressive optimization using FP8, vectorized memory access, CUB primitives, and something else TBD.
- **Minimalist Dispatch**: Direct mapping from logical views to hardware grids.

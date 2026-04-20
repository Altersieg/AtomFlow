#pragma once
// ============================================================================
// server_engine.h — Experimental ServerEngine for continuous batching sandbox
// server_engine.h — 连续批处理沙箱的实验性 ServerEngine
//
// [EN] This header defines the ServerEngine class, an isolated prototype for
//      exploring continuous-batching and PagedAttention scheduling. It lives
//      in its own namespace and shares ONLY the low-level math kernels
//      (src/kernel/*) with the production single-stream engine.
//      It does NOT inherit from, reference, or link against AtomFlowEngine.
//
// [CN] 本头文件定义 ServerEngine 类——一个用于探索连续批处理和
//      PagedAttention 调度的独立原型。它位于独立命名空间中，仅与
//      生产级单流引擎共享底层数学 kernel（src/kernel/*）。
//      它不继承、不引用、也不链接 AtomFlowEngine。
// ============================================================================

#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace atomflow::server {

// ============================================================================
// ServerEngine — Experimental sandbox entry point
// ServerEngine — 实验性沙箱入口
//
// [EN] Minimal scaffolding. Will be expanded with:
//      - Multi-request queue management
//      - Continuous batching scheduler
//      - PagedAttention KV cache (via BlockManager in paged_kv_manager.h)
//
// [CN] 最小脚手架。后续将扩展：
//      - 多请求队列管理
//      - 连续批处理调度器
//      - PagedAttention KV 缓存（通过 paged_kv_manager.h 中的 BlockManager）
// ============================================================================
class ServerEngine {
public:
    ServerEngine()  = default;
    ~ServerEngine();

    // [EN] Load model weights and allocate GPU memory pools.
    // [CN] 加载模型权重并分配 GPU 内存池。
    void initialize(const char* weights_path);

    // [EN] Run one scheduling iteration: select a micro-batch, execute one
    //      decode step for each active request, and retire finished sequences.
    // [CN] 执行一次调度迭代：选择一个 micro-batch，为每个活跃请求
    //      执行一个解码步骤，并退出已完成的序列。
    void step();

private:
    // ── GPU handles / GPU 句柄 ──────────────────────────────────────────
    cudaStream_t    stream_  = nullptr;
    cublasHandle_t  cublas_  = nullptr;
};

} // namespace atomflow::server

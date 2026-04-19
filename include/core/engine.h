#pragma once
// ============================================================================
// engine.h — AtomFlowEngine: self-contained inference engine
// engine.h — AtomFlowEngine：自包含推理引擎
//
// Encapsulates all GPU resources (weight pool, activation arena, cuBLAS handle,
// CUDA stream, RoPE cache) and the forward pass logic.
//
// 封装所有 GPU 资源（权重池、激活 arena、cuBLAS 句柄、CUDA 流、RoPE 缓存）
// 以及前向传播逻辑。
// ============================================================================

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include "core/view.h"
#include "core/model_state.h"
#include "utils/profiler.h"

// ============================================================================
// Benchmark-mode switches / 基准测试模式开关
//
//   ENABLE_VALIDATOR  0 → strip ALL ground-truth validation from forward loop.
//   ENABLE_PROFILER   0 → strip per-kernel cudaEvent record/create overhead.
//
//   ENABLE_VALIDATOR  0 → 从前向循环中去除所有基准验证。
//   ENABLE_PROFILER   0 → 去除每个 kernel 的 cudaEvent 录制/创建开销。
// ============================================================================
#ifndef ENABLE_VALIDATOR
#define ENABLE_VALIDATOR 0
#endif
#ifndef ENABLE_PROFILER
#define ENABLE_PROFILER  0
#endif

class AtomFlowEngine {
public:
    AtomFlowEngine()  = default;
    ~AtomFlowEngine();

    // Non-copyable (owns GPU resources) / 不可拷贝（持有 GPU 资源）
    AtomFlowEngine(const AtomFlowEngine&)            = delete;
    AtomFlowEngine& operator=(const AtomFlowEngine&) = delete;

    // ── Lifecycle / 生命周期 ──────────────────────────────────────────────

    // [EN] Load weights from disk, allocate GPU memory, build RoPE cache.
    // [CN] 从磁盘加载权重、分配 GPU 内存、构建 RoPE 缓存。
    void initialize(const std::string& weights_path);

    // [EN] Inject input embeddings from a ground-truth binary file.
    //      Falls back to BOS embed lookup if file is absent.
    //      verbose=true prints a status message (first call); false is silent (warm-up).
    // [CN] 从基准真值二进制文件注入输入嵌入。
    //      文件不存在时回退到 BOS 嵌入查找。
    //      verbose=true 打印状态消息（首次调用）；false 静默（预热用）。
    void inject_input(const char* gt_embed_path, bool verbose = true);

    // [EN] Inject a token by ID: look up its row in the embedding table on GPU
    //      and write the FP16 vector into act.x. Used for autoregressive decode.
    // [CN] 按 token ID 注入：在 GPU 上查找嵌入表对应行，将 FP16 向量写入 act.x。
    //      用于自回归解码。
    void inject_token(int token_id);

    // [EN] Run one full forward pass: 28 layers + final_norm + lm_head + argmax.
    //      Writes result to act.d_token_id (device).
    // [CN] 执行一次完整前向传播：28 层 + final_norm + lm_head + argmax。
    //      结果写入 act.d_token_id（设备端）。
    void forward_pass();

    // [EN] Copy the output token_id from device to host and return it.
    // [CN] 将输出 token_id 从设备拷贝到主机并返回。
    int  get_output_token();

    // ── Validation (only active when ENABLE_VALIDATOR=1) / 验证 ─────────

#if ENABLE_VALIDATOR
    // [EN] Run validation checks against ground-truth files.
    //      Must be called AFTER forward_pass() with stream synchronized.
    // [CN] 对基准真值文件执行验证。
    //      必须在 forward_pass() 之后且流已同步时调用。
    void validate_logits();
    int  v_pass  = 0;
    int  v_total = 0;
#endif

    // ── Accessors / 访问器 ───────────────────────────────────────────────

    EngineProfiler& profiler() { return prof_; }

    // [EN] Access the engine's CUDA stream (needed for graph capture).
    // [CN] 获取引擎的 CUDA 流（CUDA Graph 捕获需要）。
    cudaStream_t stream() const { return stream_; }

    // Model hyperparameters (set during initialize) / 模型超参数
    int D = 0, FFN = 0, V = 0, NL = 0, GS = 0;
    int NH = 0, NKV = 0, HD = 0;
    int Q_DIM = 0, KV_DIM = 0, QKV_OUT = 0;

private:
    // ── GPU memory pools / GPU 内存池 ────────────────────────────────────
    void* d_weight_pool_ = nullptr;
    void* d_act_pool_    = nullptr;

    // ── Model state / 模型状态 ──────────────────────────────────────────
    std::vector<LayerWeights> lw_;
    ActBuffers act_{};
    View embed_v_{};
    View final_norm_v_{};
    View lm_head_v_{};

    // ── RoPE cache (device pointers inside act pool) / RoPE 缓存 ────────
    float* d_cos_ = nullptr;
    float* d_sin_ = nullptr;

    // ── CUDA context / CUDA 上下文 ──────────────────────────────────────
    cudaStream_t   stream_ = nullptr;
    cublasHandle_t cublas_ = nullptr;
    void*          d_cublas_ws_ = nullptr;  // pre-allocated cuBLAS workspace (graph-capture safe)
    EngineProfiler prof_;

    static constexpr size_t SEQ = 1;  // single decode step / 单步解码
};

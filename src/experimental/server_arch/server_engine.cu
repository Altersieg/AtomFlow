// ============================================================================
// server_engine.cu — Entry point for the atomflow-server-test executable
// server_engine.cu — atomflow-server-test 可执行文件的入口点
//
// [EN] This file contains:
//      1. The ServerEngine method implementations (init, step, dtor).
//      2. A standalone main() that exercises the experimental server loop.
//
//      It deliberately does NOT include or link against anything in src/core/
//      or src/main.cu.  The only shared code is the kernel math in src/kernel/*.
//
// [CN] 本文件包含：
//      1. ServerEngine 方法实现（初始化、调度步骤、析构）。
//      2. 独立的 main() 函数，用于运行实验性服务器循环。
//
//      本文件刻意不包含或链接 src/core/ 和 src/main.cu 中的任何内容。
//      唯一共享的代码是 src/kernel/* 中的数学 kernel。
// ============================================================================

#include "server_engine.h"
#include "paged_kv_manager.h"

#include <cstdio>
#include <cuda_runtime.h>

// [EN] Reuse the CUDA_CHECK macro from the shared utility header.
// [CN] 复用共享工具头文件中的 CUDA_CHECK 宏。
#include "utils/utils.h"

namespace atomflow::server {

// ── ServerEngine destructor / 析构函数 ──────────────────────────────────
ServerEngine::~ServerEngine() {
    if (cublas_) cublasDestroy(cublas_);
    if (stream_) cudaStreamDestroy(stream_);
}

// ── ServerEngine::initialize / 初始化 ──────────────────────────────────
// [EN] Placeholder: creates CUDA stream + cuBLAS handle.
//      Weight loading and memory pool allocation will be added when the
//      scheduler is ready to drive multi-request decode.
// [CN] 占位符：创建 CUDA 流 + cuBLAS 句柄。
//      权重加载和内存池分配将在调度器准备好驱动多请求解码时添加。
void ServerEngine::initialize(const char* weights_path) {
    (void)weights_path;  // [EN] unused for now / [CN] 暂未使用

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    cublasCreate(&cublas_);
    cublasSetStream(cublas_, stream_);

    std::printf("[server-engine]  CUDA stream + cuBLAS handle created.\n");
}

// ── ServerEngine::step / 调度步骤 ──────────────────────────────────────
// [EN] Placeholder scheduling iteration.  Will eventually:
//      1. Dequeue pending requests from an input queue.
//      2. Build a micro-batch (continuous batching).
//      3. For each active sequence, look up its PageTable → resolve physical
//         KV blocks → launch the batched GQA attention kernel.
//      4. Advance each sequence's token count, retire finished ones.
//
// [CN] 占位调度迭代。最终将：
//      1. 从输入队列中取出待处理请求。
//      2. 构建 micro-batch（连续批处理）。
//      3. 对每个活跃序列，查询其 PageTable → 解析物理 KV 块 →
//         启动批量 GQA 注意力 kernel。
//      4. 推进每个序列的 token 计数，退出已完成的序列。
void ServerEngine::step() {
    std::printf("[server-engine]  step() called (no-op stub).\n");
}

} // namespace atomflow::server

// ============================================================================
// main — Standalone entry point for the server architecture sandbox
// main — 服务器架构沙箱的独立入口点
// ============================================================================
int main(int argc, char* argv[]) {
    std::printf("╔══════════════════════════════════════════════════════════╗\n");
    std::printf("║  AtomFlow  ·  Server Architecture Sandbox                ║\n");
    std::printf("║  实验性服务器架构沙箱                                      ║\n");
    std::printf("╚══════════════════════════════════════════════════════════╝\n\n");

    const char* weights_path = (argc > 1) ? argv[1] : "models/llama3_2_atomflow.bin";

    atomflow::server::ServerEngine engine;
    engine.initialize(weights_path);

    // [EN] Run a few no-op scheduling steps to validate the loop skeleton.
    // [CN] 运行几个空操作调度步骤以验证循环骨架。
    constexpr int TEST_STEPS = 3;
    std::printf("\n[sandbox]  Running %d test scheduling steps...\n", TEST_STEPS);
    for (int i = 0; i < TEST_STEPS; ++i) {
        engine.step();
    }

    std::printf("\n[sandbox]  Done. Server sandbox exiting cleanly.\n");
    return 0;
}

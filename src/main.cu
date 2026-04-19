// ============================================================================
// main.cu — Lightweight CLI + benchmark harness for AtomFlowEngine
// main.cu — 轻量级 CLI + AtomFlowEngine 基准测试工具
//
// All GPU logic (weight loading, activation allocation, forward pass) lives in
// AtomFlowEngine (include/core/engine.h, src/core/engine.cu).
// This file only: parses args → initializes engine → runs warm-up → benchmarks.
//
// 所有 GPU 逻辑（权重加载、激活分配、前向传播）位于 AtomFlowEngine 中。
// 本文件仅负责：解析参数 → 初始化引擎 → 预热 → 基准测试。
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <string>
#include <chrono>
#include <cuda_runtime.h>

#include "core/engine.h"
#include "ops/kernel.h"
#include "utils/utils.h"
#if ENABLE_VALIDATOR
#include "utils/validator.h"
#endif

// Warm-up iterations for benchmark mode / 基准测试模式的预热迭代次数
#ifndef BENCHMARK_WARMUP_ITERS
#define BENCHMARK_WARMUP_ITERS 3
#endif

static constexpr const char* DEFAULT_WEIGHTS   = "models/llama3_2_atomflow.bin";
static constexpr const char* GT_EMBED_PATH     = "ground_truth/gt_input_embeddings.bin";

// ============================================================================
// main
// ============================================================================
int main(int argc, char* argv[]) {

    const std::string weights_path = (argc > 1) ? argv[1] : DEFAULT_WEIGHTS;

    // ── 1. Initialize engine / 初始化引擎 ────────────────────────────────
    AtomFlowEngine engine;
    engine.initialize(weights_path);

    // ── 2. Inject input embeddings / 注入输入嵌入 ────────────────────────
    engine.inject_input(GT_EMBED_PATH);

#if ENABLE_VALIDATOR
    validate_print_header();
#endif

    // ── 3. Warm-up / 预热 ───────────────────────────────────────────────
    std::printf("\n[Benchmark]  Warm-up: %d iterations...\n", BENCHMARK_WARMUP_ITERS);
    for (int wi = 0; wi < BENCHMARK_WARMUP_ITERS; ++wi) {
        // Re-inject embeddings silently (forward pass mutates act.x).
        // 静默重新注入嵌入（前向传播会修改 act.x）。
        engine.inject_input(GT_EMBED_PATH, /*verbose=*/false);
        engine.forward_pass();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("[Benchmark]  Warm-up done.\n");

    // ── 4. Timed run / 计时运行 ─────────────────────────────────────────
    engine.inject_input(GT_EMBED_PATH, /*verbose=*/false);
    CUDA_CHECK(cudaDeviceSynchronize());  // drain pipeline before timing

    auto t_start = std::chrono::steady_clock::now();
    engine.forward_pass();
    CUDA_CHECK(cudaDeviceSynchronize());  // wait for ALL GPU work
    auto t_end = std::chrono::steady_clock::now();

    double tpot_ms   = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    double tok_per_s  = 1000.0 / tpot_ms;

    std::printf("\n");
    std::printf("╔══════════════════════════════════════════════════════╗\n");
    std::printf("║  AtomFlow  ·  Pure GPU Benchmark (chrono)       ║\n");
    std::printf("╠══════════════════════════════════════════════════════╣\n");
    std::printf("║  Warm-up iterations:  %d                          ║\n", BENCHMARK_WARMUP_ITERS);
    std::printf("║  TPOT (chrono):       %8.3f ms                 ║\n", tpot_ms);
    std::printf("║  Generation speed:    %8.1f tok/s               ║\n", tok_per_s);
    std::printf("╚══════════════════════════════════════════════════════╝\n");

    // ── 5. Output + validation / 输出 + 验证 ────────────────────────────
    int token_id = engine.get_output_token();
    std::printf("\n[Output]  next token id = %d\n", token_id);

#if ENABLE_VALIDATOR
    engine.validate_logits();
    validate_print_footer(engine.v_pass, engine.v_total);
#endif

#if ENABLE_PROFILER
    engine.profiler().print_report();
#endif

    return EXIT_SUCCESS;
}

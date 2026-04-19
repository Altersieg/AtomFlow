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

    // ── 3. Eager warm-up (no graph) / 预热（无 Graph）─────────────────────
    //    Run a few forward passes eagerly to stabilize GPU clocks, fill
    //    caches, and JIT-compile any lazy kernels before graph capture.
    //    在 Graph 捕获之前，先以普通方式运行几次前向传播以稳定 GPU 时钟、
    //    填充缓存、并 JIT 编译所有惰性 kernel。
    std::printf("\n[Benchmark]  Warm-up: %d iterations (eager)...\n", BENCHMARK_WARMUP_ITERS);
    for (int wi = 0; wi < BENCHMARK_WARMUP_ITERS; ++wi) {
        engine.inject_input(GT_EMBED_PATH, /*verbose=*/false);
        engine.forward_pass();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("[Benchmark]  Warm-up done.\n");

#if !ENABLE_VALIDATOR
    // ── 4. CUDA Graph capture / CUDA Graph 捕获 ─────────────────────────
    //    Record the ENTIRE forward pass topology (28 layers + final_norm +
    //    lm_head + argmax ≈ 280 kernel launches) into a single graph.
    //    On replay, the driver submits all kernels with near-zero CPU
    //    launch overhead, crushing the inter-kernel gaps seen in Nsys.
    //
    //    NOTE: Graph capture is incompatible with ENABLE_VALIDATOR because
    //    the validator inserts cudaStreamSynchronize + cudaMemcpy D2H
    //    inside forward_pass(), which are illegal during stream capture.
    //
    //    将整个前向传播拓扑（28 层 + final_norm + lm_head + argmax ≈ 280 次
    //    kernel 启动）录制到单个 Graph 中。重放时，驱动以接近零的 CPU 启动
    //    开销提交所有 kernel，消除 Nsys 中观察到的 kernel 间隙。
    //    注意：Graph 捕获与 ENABLE_VALIDATOR 不兼容，因为验证器在
    //    forward_pass() 内插入了 cudaStreamSynchronize + cudaMemcpy D2H，
    //    这在流捕获期间是非法的。
    std::printf("[Graph]  Capturing forward pass...\n");
    engine.inject_input(GT_EMBED_PATH, /*verbose=*/false);

    cudaGraph_t     graph     = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    cudaStream_t    stream    = engine.stream();

    // Begin capture — all GPU work on `stream` is recorded, not executed.
    // 开始捕获 — 流上所有 GPU 工作被录制，不执行。
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    engine.forward_pass();
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    std::printf("[Graph]  Capture complete.\n");

    // Instantiate — compile the recorded graph into an executable.
    // 实例化 — 将录制的 Graph 编译为可执行对象。
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    std::printf("[Graph]  Instantiated.\n");

    // ── 5. Graph warm-up / Graph 预热 ───────────────────────────────────
    //    Run the graph a few times to warm up the graph executor itself.
    //    对 Graph 执行器本身进行预热。
    std::printf("[Graph]  Graph warm-up: %d iterations...\n", BENCHMARK_WARMUP_ITERS);
    for (int wi = 0; wi < BENCHMARK_WARMUP_ITERS; ++wi) {
        engine.inject_input(GT_EMBED_PATH, /*verbose=*/false);
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    std::printf("[Graph]  Graph warm-up done.\n");

    // ── 6. Timed run — graph replay / 计时运行 — Graph 重放 ─────────────
    engine.inject_input(GT_EMBED_PATH, /*verbose=*/false);
    CUDA_CHECK(cudaDeviceSynchronize());  // drain pipeline before timing

    auto t_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto t_end = std::chrono::steady_clock::now();

    double graph_ms   = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    double graph_tps  = 1000.0 / graph_ms;

    // Also time a single eager pass for comparison.
    // 同时计时一次普通前向传播用于对比。
    engine.inject_input(GT_EMBED_PATH, /*verbose=*/false);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t2_start = std::chrono::steady_clock::now();
    engine.forward_pass();
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t2_end = std::chrono::steady_clock::now();

    double eager_ms   = std::chrono::duration<double, std::milli>(t2_end - t2_start).count();
    double eager_tps  = 1000.0 / eager_ms;

    std::printf("\n");
    std::printf("╔══════════════════════════════════════════════════════════╗\n");
    std::printf("║  AtomFlow  ·  Pure GPU Benchmark                        ║\n");
    std::printf("╠══════════════════════════════════════════════════════════╣\n");
    std::printf("║  Warm-up iterations:  %d                                ║\n", BENCHMARK_WARMUP_ITERS);
    std::printf("║                                                          ║\n");
    std::printf("║  [Eager]  TPOT:  %8.3f ms  (%6.1f tok/s)            ║\n", eager_ms, eager_tps);
    std::printf("║  [Graph]  TPOT:  %8.3f ms  (%6.1f tok/s)            ║\n", graph_ms, graph_tps);
    std::printf("║  Speedup:        %5.2fx                                 ║\n", eager_ms / graph_ms);
    std::printf("╚══════════════════════════════════════════════════════════╝\n");

    // Cleanup CUDA Graph / 清理 CUDA Graph
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);

#else  // ENABLE_VALIDATOR — eager-only benchmark
    // When validator is enabled, cudaStreamSynchronize inside forward_pass()
    // makes graph capture impossible. Fall back to eager timing only.
    // 启用验证器时，forward_pass() 内的 cudaStreamSynchronize 使图捕获不可能。
    // 仅使用普通计时。
    engine.inject_input(GT_EMBED_PATH, /*verbose=*/false);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t_start = std::chrono::steady_clock::now();
    engine.forward_pass();
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t_end = std::chrono::steady_clock::now();

    double eager_ms  = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    double eager_tps = 1000.0 / eager_ms;

    std::printf("\n");
    std::printf("╔══════════════════════════════════════════════════════════╗\n");
    std::printf("║  AtomFlow  ·  Eager Benchmark (validator ON)             ║\n");
    std::printf("╠══════════════════════════════════════════════════════════╣\n");
    std::printf("║  [Eager]  TPOT:  %8.3f ms  (%6.1f tok/s)            ║\n", eager_ms, eager_tps);
    std::printf("╚══════════════════════════════════════════════════════════╝\n");
#endif // !ENABLE_VALIDATOR

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

#if !ENABLE_VALIDATOR
    // ── 6. Autoregressive generation loop / 自回归生成循环 ─────────────
    //    Feed the predicted token_id back as input embedding and repeat.
    //    Without KV cache, each step sees only seq_len=1 (no history),
    //    so the generated text will be incoherent — but this validates
    //    the full decode loop plumbing before we implement KV cache.
    //    Skipped in validator mode: AR changes the input, making GT probes
    //    fire spurious FATALs.
    //
    //    将预测的 token_id 的嵌入重新喂给引擎并重复。
    //    无 KV 缓存时，每步仅看到 seq_len=1（无历史），
    //    因此生成文本将不连贯 — 但这可在实现 KV 缓存前
    //    验证完整的解码循环管线。
    //    在验证器模式下跳过：AR 改变输入，会使 GT 探针产生虚假 FATAL。
    constexpr int AR_STEPS = 20;
    std::printf("\n[AR]  Autoregressive generation: %d steps (no KV cache)\n", AR_STEPS);
    std::printf("[AR]  step  0 → token %d\n", token_id);

    auto ar_start = std::chrono::steady_clock::now();
    for (int step = 1; step <= AR_STEPS; ++step) {
        engine.inject_token(token_id);
        engine.forward_pass();
        token_id = engine.get_output_token();
        std::printf("[AR]  step %2d → token %d\n", step, token_id);
    }
    auto ar_end = std::chrono::steady_clock::now();

    double ar_total_ms = std::chrono::duration<double, std::milli>(ar_end - ar_start).count();
    double ar_tps      = AR_STEPS * 1000.0 / ar_total_ms;

    std::printf("\n[AR]  %d tokens in %.1f ms  (%.1f tok/s avg)\n",
                AR_STEPS, ar_total_ms, ar_tps);
#endif // !ENABLE_VALIDATOR

    return EXIT_SUCCESS;
}

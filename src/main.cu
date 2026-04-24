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
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#include "core/engine.h"
#include "ops/kernel.h"
#include "utils/utils.h"
#if ENABLE_VALIDATOR
#include "utils/validator.h"
#endif

// Warm-up iterations for benchmark mode / 基准测试模式的预热迭代次数
//   10 matches common practice (vLLM uses 5-10) and is enough time for the
//   GPU boost clock to stabilise (>150 ms of sustained work).
#ifndef BENCHMARK_WARMUP_ITERS
#define BENCHMARK_WARMUP_ITERS 10
#endif

// Timed iterations whose median we report.  Using median (not mean) makes the
// reading robust to the occasional stutter from OS scheduling / driver GC.
#ifndef BENCHMARK_MEASURE_ITERS
#define BENCHMARK_MEASURE_ITERS 20
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

    // ◭─ 6. Timed runs — median of N iterations / 计时运行 — 取 N 次迭代的中位数 ───
    // Helper lambda that times a single trial and returns milliseconds.
    // 辅助 lambda：测一次 trial 并返回毫秒数。
    auto time_once = [&](auto &&launch) -> double {
        engine.inject_input(GT_EMBED_PATH, /*verbose=*/false);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto s = std::chrono::steady_clock::now();
        launch();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto e = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(e - s).count();
    };

    auto median_of = [&](std::vector<double>& v) -> double {
        const size_t mid = v.size() / 2;
        std::nth_element(v.begin(), v.begin() + mid, v.end());
        return v[mid];
    };

    // Graph timings
    std::vector<double> graph_times;
    graph_times.reserve(BENCHMARK_MEASURE_ITERS);
    for (int mi = 0; mi < BENCHMARK_MEASURE_ITERS; ++mi) {
        graph_times.push_back(time_once([&]{
            CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        }));
    }
    double graph_ms  = median_of(graph_times);
    double graph_tps = 1000.0 / graph_ms;

    // Eager timings
    std::vector<double> eager_times;
    eager_times.reserve(BENCHMARK_MEASURE_ITERS);
    for (int mi = 0; mi < BENCHMARK_MEASURE_ITERS; ++mi) {
        eager_times.push_back(time_once([&]{
            engine.forward_pass();
        }));
    }
    double eager_ms  = median_of(eager_times);
    double eager_tps = 1000.0 / eager_ms;

    std::printf("\n");
    std::printf("╔══════════════════════════════════════════════════════════╗\n");
    std::printf("║  AtomFlow  ·  Pure GPU Benchmark                        ║\n");
    std::printf("╠══════════════════════════════════════════════════════════╣\n");
    std::printf("║  Warm-up: %2d   Measured (median of): %2d              ║\n",
                BENCHMARK_WARMUP_ITERS, BENCHMARK_MEASURE_ITERS);
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
    // ── 6. Autoregressive generation loop (with KV cache) / 自回归生成循环（带 KV 缓存）
    //    Reset the KV cache, then iterate: inject token → set position → forward.
    //    The engine writes K,V into the cache at current_pos_ and attends over
    //    all cached history [0, current_pos_].
    //    Skipped in validator mode: AR changes the input, making GT probes
    //    fire spurious FATALs.
    //
    //    重置 KV 缓存，然后循环：注入 token → 设置位置 → 前向传播。
    //    引擎将 K,V 写入缓存的 current_pos_ 行，并对
    //    所有已缓存历史 [0, current_pos_] 进行注意力。
    //    在验证器模式下跳过：AR 改变输入，会使 GT 探针产生虚假 FATAL。
    constexpr int AR_STEPS = 20;
    std::printf("\n[AR]  Autoregressive generation: %d steps (KV cache)\n", AR_STEPS);

    // [EN] Open token log file for the Python decoder to consume.
    //      Each line contains one integer token ID.
    // [CN] 打开 token 日志文件供 Python 解码器消费。
    //      每行包含一个整数 token ID。
    static constexpr const char* TOKEN_LOG_PATH = "output_tokens.txt";
    FILE* f_out = std::fopen(TOKEN_LOG_PATH, "w");
    if (!f_out) {
        std::fprintf(stderr, "[AR]  ERROR: cannot open %s for writing\n", TOKEN_LOG_PATH);
        return EXIT_FAILURE;
    }

    engine.reset_kv_cache();

    // Step 0: re-inject GT embeddings at position 0.
    // 步骤 0：在位置 0 重新注入 GT 嵌入。
    engine.current_pos_ = 0;
    engine.inject_input(GT_EMBED_PATH, /*verbose=*/false);
    engine.forward_pass();
    token_id = engine.get_output_token();
    std::printf("[AR]  step  0 → token %d\n", token_id);
    // [EN] Log token to disk with immediate flush for real-time availability.
    // [CN] 将 token 写入磁盘并立即刷新，确保实时可用。
    std::fprintf(f_out, "%d\n", token_id);
    std::fflush(f_out);

    auto ar_start = std::chrono::steady_clock::now();
    for (int step = 1; step <= AR_STEPS; ++step) {
        engine.current_pos_ = step;
        engine.inject_token(token_id);
        engine.forward_pass();
        token_id = engine.get_output_token();
        std::printf("[AR]  step %2d → token %d\n", step, token_id);
        std::fprintf(f_out, "%d\n", token_id);
        std::fflush(f_out);
    }
    auto ar_end = std::chrono::steady_clock::now();

    // [EN] Close token log file. / [CN] 关闭 token 日志文件。
    std::fclose(f_out);
    std::printf("[AR]  Token IDs written to %s\n", TOKEN_LOG_PATH);

    double ar_total_ms = std::chrono::duration<double, std::milli>(ar_end - ar_start).count();
    double ar_tps      = AR_STEPS * 1000.0 / ar_total_ms;

    std::printf("\n[AR]  %d tokens in %.1f ms  (%.1f tok/s avg)\n",
                AR_STEPS, ar_total_ms, ar_tps);
#endif // !ENABLE_VALIDATOR

    return EXIT_SUCCESS;
}

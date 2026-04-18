#pragma once

// ============================================================================
// EngineProfiler — AtomFlow inference engine performance monitor
// EngineProfiler — AtomFlow 推理引擎性能监控器
//
// Design principles / 设计原则：
//
//  1. STRICT BACKEND SEPARATION
//     Host-side metrics (TTFT) use std::chrono::steady_clock.
//     Device-side metrics (TPOT, kernel durations, bandwidth) use cudaEvent_t.
//     They must NEVER be mixed, because chrono measures wall-clock time that
//     includes driver latency, PCIe overhead, and OS scheduling jitter — none
//     of which reflect true GPU execution time.
//
//     严格的后端分离
//     主机端指标（TTFT）使用 std::chrono::steady_clock。
//     设备端指标（TPOT、Kernel 耗时、带宽）使用 cudaEvent_t。
//     两者绝对不能混用，因为 chrono 测量的挂钟时间包含驱动延迟、
//     PCIe 开销和操作系统调度抖动，均不反映真实的 GPU 执行时间。
//
//  2. DEFERRED DEVICE SYNCHRONIZATION (critical path protection)
//     stop_device() does NOT call cudaEventSynchronize(). It only records
//     the stop event and appends the event pair to a pending queue.
//     flush_device_timers() (called by print_report) performs ONE batched
//     synchronization at the end of inference, reading all accumulated events.
//     This eliminates per-kernel CPU-GPU pipeline bubbles.
//
//     延迟设备同步（关键路径保护）
//     stop_device() 不调用 cudaEventSynchronize()。它只记录停止事件，
//     并将事件对追加到 pending 队列。
//     flush_device_timers()（由 print_report 调用）在推理结束时执行
//     一次批量同步，读取所有累积的事件。这消除了每个 Kernel 的
//     CPU-GPU 流水线气泡。
//
// ============================================================================

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>

// ============================================================================
// Internal data structures (defined here for inline RAII guards)
// 内部数据结构（在此定义以供内联 RAII 守卫使用）
// ============================================================================

// One pending (start, stop) CUDA event pair that has not yet been read.
// Optionally carries a bytes_rw count for bandwidth computation.
// 一个尚未读取的待处理 (start, stop) CUDA 事件对。
// 可选地携带 bytes_rw 用于带宽计算。
struct DevicePendingPair {
    cudaEvent_t start    = nullptr;
    cudaEvent_t stop     = nullptr;
    uint64_t    bytes_rw = 0;  // 0 = no bandwidth measurement / 0 = 不测带宽
};

// Accumulator for one named device timer.
// 一个命名设备计时器的累加器。
//
// [Bug/Imperfection: A new cudaEvent_t is created on every start_device() call.
//  cudaEventCreate has ~1-3 µs overhead per call and fragments the CUDA context's
//  internal event pool. A production profiler would maintain a free-list of
//  pre-allocated events and recycle them between measurements.
//  每次 start_device() 调用都会新建 cudaEvent_t。cudaEventCreate 每次调用
//  约有 1-3 µs 开销，会碎片化 CUDA 上下文的内部事件池。
//  生产级 profiler 应维护一个预分配事件的空闲链表，在测量之间复用。]
struct DeviceTimerEntry {
    std::vector<DevicePendingPair> pending;   // not yet flushed / 尚未刷新
    float    total_ms    = 0.0f;  // flushed & accumulated / 已刷新并累积
    uint64_t bytes_total = 0;     // accumulated annotated bytes for bandwidth / 带宽用字节数累计
    uint32_t count       = 0;     // completed measurements / 已完成的测量次数
    bool     running     = false; // start recorded, stop not yet / start 已记录但 stop 未记录
    cudaEvent_t cur_start = nullptr;  // start event awaiting its stop / 等待 stop 的 start 事件
};

// Accumulator for one named host timer.
// 一个命名主机计时器的累加器。
struct HostTimerEntry {
    using Clock = std::chrono::steady_clock;
    Clock::time_point start_tp;
    double   total_ms = 0.0;
    uint32_t count    = 0;
    bool     running  = false;
};

// ============================================================================
// EngineProfiler
// ============================================================================
class EngineProfiler {
public:
    EngineProfiler()  = default;
    ~EngineProfiler();

    // Profilers hold CUDA events; copying is undefined.
    // Profiler 持有 CUDA 事件，禁止拷贝。
    EngineProfiler(const EngineProfiler&)            = delete;
    EngineProfiler& operator=(const EngineProfiler&) = delete;

    // ------------------------------------------------------------------ //
    // HOST-SIDE timing — std::chrono::steady_clock                        //
    // 主机端计时 — std::chrono::steady_clock                              //
    //                                                                     //
    // Use for: TTFT and any host-orchestration latency.                   //
    // 用于：TTFT 及任何主机编排延迟。                                      //
    // ------------------------------------------------------------------ //

    void start_host(const std::string& name);
    void stop_host (const std::string& name);

    // ------------------------------------------------------------------ //
    // DEVICE-SIDE timing — cudaEvent_t                                    //
    // 设备端计时 — cudaEvent_t                                            //
    //                                                                     //
    // Use for: TPOT, per-kernel duration, memory bandwidth.               //
    // 用于：TPOT、单 Kernel 耗时、内存带宽。                              //
    //                                                                     //
    // IMPORTANT: stop_device() does NOT block the CPU. It records the     //
    // stop event on `stream` and enqueues the pair. Actual elapsed time   //
    // is only computed when flush_device_timers() is called.              //
    //                                                                     //
    // 重要：stop_device() 不会阻塞 CPU。它在 stream 上记录停止事件并      //
    // 入列事件对。实际耗时仅在调用 flush_device_timers() 时才计算。       //
    // ------------------------------------------------------------------ //

    void start_device(const std::string& name, cudaStream_t stream = 0);
    void stop_device (const std::string& name, cudaStream_t stream = 0);

    // ------------------------------------------------------------------ //
    // High-level inference-phase markers                                  //
    // 高层推理阶段标记                                                     //
    // ------------------------------------------------------------------ //

    // TTFT — measured HOST-SIDE (chrono), because TTFT is an end-to-end
    // wall-clock metric that includes tokenization, memory allocation, and
    // all async GPU work. A pure GPU timer would undercount it.
    //
    // TTFT 使用主机端 chrono 测量，因为它是端到端挂钟指标，包含分词、
    // 内存分配以及所有异步 GPU 工作。纯 GPU 计时器会低估它。
    void mark_prefill_start();
    void mark_first_token();

    // TPOT — measured DEVICE-SIDE (cudaEvent_t), because TPOT is a pure
    // GPU execution metric. Using chrono here would absorb CPU launch
    // overhead and scheduling jitter that don't reflect actual compute time.
    //
    // TPOT 使用设备端 cudaEvent_t 测量，因为它是纯 GPU 执行指标。
    // 此处使用 chrono 会吸收 CPU 启动开销和调度抖动，
    // 这些不能反映真实的计算时间。
    //
    // [Bug/Imperfection: If the decode step spans multiple CUDA streams, a
    //  single event pair only measures the stream it is recorded on. To measure
    //  the true wall-clock GPU time across streams, one would need a dependency
    //  event graph or Nsight Compute's "GPU Active Cycles" metric.
    //  如果解码步骤跨越多个 CUDA stream，单个事件对仅测量其所在 stream。
    //  要测量跨 stream 的真实 GPU 挂钟时间，需要依赖事件图或
    //  Nsight Compute 的"GPU Active Cycles"指标。]
    void mark_decode_start(cudaStream_t stream = 0);
    void mark_decode_end  (cudaStream_t stream = 0);

    // ------------------------------------------------------------------ //
    // Bandwidth annotation                                                 //
    // 带宽注释                                                             //
    //                                                                      //
    // Call IMMEDIATELY after stop_device() for the same kernel name to     //
    // annotate the last pending pair with its theoretical byte count.      //
    // 在同名 stop_device() 之后立即调用，以将理论字节数注入最后一个       //
    // pending 事件对中。                                                   //
    // ------------------------------------------------------------------ //

    void annotate_bandwidth(const std::string& kernel_name, uint64_t bytes_rw);

    // ------------------------------------------------------------------ //
    // Deferred flush + report                                              //
    // 延迟刷新 + 报告                                                      //
    // ------------------------------------------------------------------ //

    // Flush all pending device event pairs.
    // Performs ONE batched cudaEventSynchronize per timer (not per kernel).
    // This is the ONLY place where CPU-GPU synchronization occurs.
    //
    // 刷新所有待处理设备事件对。
    // 每个计时器执行一次批量 cudaEventSynchronize（而非每个 Kernel 一次）。
    // 这是 CPU-GPU 同步发生的唯一位置。
    //
    // [Bug/Imperfection: Even a single cudaEventSynchronize at the end of a
    //  decode step stalls the CPU until ALL preceding GPU work on that stream
    //  has completed. In a continuous batching scheduler (e.g., vLLM's engine
    //  loop), this sync would prevent the CPU from immediately scheduling the
    //  next batch. The correct production approach is to read events completely
    //  asynchronously using CUDA callbacks or a separate profiling thread.
    //  即使在解码步骤末尾只有一次 cudaEventSynchronize，它也会使 CPU 停顿，
    //  直到该 stream 上所有先前的 GPU 工作全部完成。在连续批处理调度器中
    //（如 vLLM 的引擎循环），此同步会阻止 CPU 立即调度下一批。
    //  正确的生产方案是通过 CUDA callbacks 或独立 profiling 线程完全
    //  异步地读取事件。]
    void flush_device_timers();

    // Print the full ASCII performance dashboard (calls flush_device_timers first).
    // 打印完整 ASCII 性能看板（先调用 flush_device_timers）。
    void print_report();

    // Reset all state for a new inference request.
    // 重置所有状态以开始新的推理请求。
    void reset();

    // ------------------------------------------------------------------ //
    // RAII scoped timers                                                   //
    // RAII 作用域计时器                                                    //
    // ------------------------------------------------------------------ //

    // Usage: { auto _ = profiler.scoped_host("tokenize"); ... }
    // 用法：{ auto _ = profiler.scoped_host("tokenize"); ... }
    struct ScopedHostTimer {
        EngineProfiler& p; std::string n;
        ScopedHostTimer(EngineProfiler& p_, std::string n_)
            : p(p_), n(std::move(n_)) { p.start_host(n); }
        ~ScopedHostTimer() { p.stop_host(n); }
    };

    // Usage: { auto _ = profiler.scoped_device("rmsnorm", stream); ... }
    // 用法：{ auto _ = profiler.scoped_device("rmsnorm", stream); ... }
    struct ScopedDeviceTimer {
        EngineProfiler& p; std::string n; cudaStream_t s;
        ScopedDeviceTimer(EngineProfiler& p_, std::string n_, cudaStream_t s_)
            : p(p_), n(std::move(n_)), s(s_) { p.start_device(n, s); }
        ~ScopedDeviceTimer() { p.stop_device(n, s); }
    };

    ScopedHostTimer   scoped_host  (const std::string& name)                       { return {*this, name};     }
    ScopedDeviceTimer scoped_device(const std::string& name, cudaStream_t stream=0) { return {*this, name, stream}; }

private:
    // Host timers indexed by name / 按名称索引的主机计时器
    std::unordered_map<std::string, HostTimerEntry>   host_timers_;

    // Device timers indexed by name / 按名称索引的设备计时器
    std::unordered_map<std::string, DeviceTimerEntry> device_timers_;

    // TTFT state (host-side) / TTFT 状态（主机端）
    std::chrono::steady_clock::time_point prefill_start_;
    double ttft_ms_ = -1.0;  // -1 = not yet measured / -1 = 尚未测量

    // Decode event pairs (device-side) — separate from named timers to keep
    // the per-step overhead out of the general timer map lookup.
    // 解码事件对（设备端）—— 与命名计时器分离，避免每步开销触发 map 查找。
    std::vector<DevicePendingPair> decode_pending_;
    bool    decode_step_running_ = false;
    float   tpot_total_ms_       = 0.0f;
    uint32_t decode_count_       = 0;

    // Internal helpers / 内部辅助
    DeviceTimerEntry& _ensure_device(const std::string& name);
    void _flush_one(DeviceTimerEntry& entry);

    // Report formatting / 报告格式化
    static void _box_top   ();
    static void _box_bottom();
    static void _box_divider();
    static void _box_row   (const char* label, const char* value);
    static void _box_section(const char* title);
};

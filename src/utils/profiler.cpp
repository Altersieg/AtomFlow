#include "utils/profiler.h"
#include "utils/utils.h"

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <numeric>

// ============================================================================
// Destructor
// 析构函数
// ============================================================================
EngineProfiler::~EngineProfiler() {
    // Destroy all pending events that were never flushed.
    // 销毁所有从未被刷新的待处理事件。
    for (auto& [name, entry] : device_timers_) {
        if (entry.cur_start) CUDA_CHECK(cudaEventDestroy(entry.cur_start));
        for (auto& pair : entry.pending) {
            if (pair.start) CUDA_CHECK(cudaEventDestroy(pair.start));
            if (pair.stop)  CUDA_CHECK(cudaEventDestroy(pair.stop));
        }
    }
    for (auto& pair : decode_pending_) {
        if (pair.start) CUDA_CHECK(cudaEventDestroy(pair.start));
        if (pair.stop)  CUDA_CHECK(cudaEventDestroy(pair.stop));
    }
}

// ============================================================================
// HOST-SIDE timing (std::chrono)
// 主机端计时（std::chrono）
// ============================================================================

void EngineProfiler::start_host(const std::string& name) {
    auto& e   = host_timers_[name];
    e.start_tp = std::chrono::steady_clock::now();
    e.running  = true;
}

void EngineProfiler::stop_host(const std::string& name) {
    auto now = std::chrono::steady_clock::now();
    auto it  = host_timers_.find(name);
    if (it == host_timers_.end() || !it->second.running) return;

    auto& e    = it->second;
    e.total_ms += std::chrono::duration<double, std::milli>(now - e.start_tp).count();
    e.count++;
    e.running  = false;
}

// ============================================================================
// DEVICE-SIDE timing (cudaEvent_t)
// 设备端计时（cudaEvent_t）
// ============================================================================

// Lazily initialise a DeviceTimerEntry (no CUDA calls at construction).
// 惰性初始化 DeviceTimerEntry（构造时不调用 CUDA）。
DeviceTimerEntry& EngineProfiler::_ensure_device(const std::string& name) {
    return device_timers_[name];   // default-constructs if absent / 不存在时默认构造
}

void EngineProfiler::start_device(const std::string& name, cudaStream_t stream) {
    auto& e = _ensure_device(name);

    // If a previous start was left without a matching stop, recycle the event
    // to avoid leaking it.
    // 如果上一个 start 没有匹配的 stop，回收该事件以避免泄漏。
    if (e.running && e.cur_start) {
        CUDA_CHECK(cudaEventDestroy(e.cur_start));
        e.cur_start = nullptr;
    }

    // [Bug/Imperfection: cudaEventCreate + cudaEventRecord are called per
    //  timing interval. This allocates a new driver-level event object each
    //  call (~1-3 µs on typical hardware), which accumulates into measurable
    //  overhead for very fine-grained per-kernel profiling.
    //  A production engine should pre-allocate an event ring-buffer and
    //  index into it rather than creating/destroying events dynamically.
    //  每次计时区间都会调用 cudaEventCreate + cudaEventRecord，
    //  每次调用约分配 1-3 µs 的驱动级事件对象，对于极细粒度的
    //  逐 Kernel profiling 会积累成可观的开销。
    //  生产引擎应预分配一个事件环形缓冲区并索引使用，
    //  而非动态创建/销毁事件。]
    CUDA_CHECK(cudaEventCreate(&e.cur_start));
    CUDA_CHECK(cudaEventRecord(e.cur_start, stream));
    e.running = true;
}

void EngineProfiler::stop_device(const std::string& name, cudaStream_t stream) {
    auto it = device_timers_.find(name);
    if (it == device_timers_.end() || !it->second.running) return;

    auto& e = it->second;

    cudaEvent_t stop_ev;
    CUDA_CHECK(cudaEventCreate(&stop_ev));
    CUDA_CHECK(cudaEventRecord(stop_ev, stream));

    // Enqueue the pair — NO cudaEventSynchronize here.
    // The CPU continues immediately after returning from this function.
    // 入列事件对 —— 此处不调用 cudaEventSynchronize。
    // 从本函数返回后 CPU 立即继续。
    e.pending.push_back({e.cur_start, stop_ev, 0});
    e.cur_start = nullptr;
    e.running   = false;
}

// ============================================================================
// Inference-phase markers
// 推理阶段标记
// ============================================================================

// TTFT: pure host-side chrono
// TTFT：纯主机端 chrono
void EngineProfiler::mark_prefill_start() {
    prefill_start_ = std::chrono::steady_clock::now();
}

void EngineProfiler::mark_first_token() {
    auto now = std::chrono::steady_clock::now();
    ttft_ms_ = std::chrono::duration<double, std::milli>(now - prefill_start_).count();
}

// TPOT: device-side cudaEvent_t
// TPOT：设备端 cudaEvent_t
void EngineProfiler::mark_decode_start(cudaStream_t stream) {
    // Recycle dangling start if stop was never called.
    // 回收从未调用 stop 的悬空 start 事件。
    if (decode_step_running_) {
        if (!decode_pending_.empty() && decode_pending_.back().stop == nullptr) {
            CUDA_CHECK(cudaEventDestroy(decode_pending_.back().start));
            decode_pending_.pop_back();
        }
    }
    DevicePendingPair pair{};
    CUDA_CHECK(cudaEventCreate(&pair.start));
    CUDA_CHECK(cudaEventRecord(pair.start, stream));
    decode_pending_.push_back(pair);
    decode_step_running_ = true;
}

void EngineProfiler::mark_decode_end(cudaStream_t stream) {
    if (!decode_step_running_ || decode_pending_.empty()) return;

    auto& pair = decode_pending_.back();
    CUDA_CHECK(cudaEventCreate(&pair.stop));
    CUDA_CHECK(cudaEventRecord(pair.stop, stream));
    decode_step_running_ = false;
    // No sync — pair is read during flush_device_timers().
    // 不同步 —— 事件对在 flush_device_timers() 时读取。
}

// ============================================================================
// Bandwidth annotation
// 带宽注释
// ============================================================================

void EngineProfiler::annotate_bandwidth(const std::string& kernel_name, uint64_t bytes_rw) {
    auto it = device_timers_.find(kernel_name);
    if (it == device_timers_.end() || it->second.pending.empty()) return;
    it->second.pending.back().bytes_rw = bytes_rw;
}

// ============================================================================
// Deferred flush
// 延迟刷新
// ============================================================================

void EngineProfiler::_flush_one(DeviceTimerEntry& entry) {
    for (auto& pair : entry.pending) {
        if (!pair.start || !pair.stop) continue;

        // [Bug/Imperfection: cudaEventSynchronize blocks the calling CPU thread
        //  until the GPU has processed past `pair.stop` on its stream.
        //  For a single stream this means all kernels queued before stop_device()
        //  must finish before we can read the elapsed time.
        //  In a multi-stream pipeline this may force unnecessary cross-stream
        //  serialization if the CPU is managing stream dependencies via events.
        //  Mitigation applied here: we call this ONLY once per flush (end of step),
        //  not per kernel, reducing the number of bubbles to O(1) per decode step.
        //  cudaEventSynchronize 阻塞调用 CPU 线程，直到 GPU 在其 stream 上
        //  处理超过 pair.stop。对于单 stream，这意味着 stop_device() 之前
        //  排队的所有 Kernel 都必须完成才能读取耗时。
        //  在多 stream 流水线中，若 CPU 通过事件管理 stream 依赖关系，
        //  这可能会强制不必要的跨 stream 序列化。
        //  此处的缓解措施：每次刷新仅调用一次（步骤末尾），
        //  而非每个 Kernel 一次，将气泡数量降至每解码步骤 O(1)。]
        CUDA_CHECK(cudaEventSynchronize(pair.stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, pair.start, pair.stop));
        entry.total_ms    += ms;
        entry.bytes_total += pair.bytes_rw;   // accumulate bandwidth bytes / 累加带宽字节数
        entry.count++;

        CUDA_CHECK(cudaEventDestroy(pair.start));
        CUDA_CHECK(cudaEventDestroy(pair.stop));
    }
    entry.pending.clear();
}

void EngineProfiler::flush_device_timers() {
    // Flush named device timers
    // 刷新命名设备计时器
    for (auto& [name, entry] : device_timers_)
        _flush_one(entry);

    // Flush decode (TPOT) event pairs
    // 刷新解码（TPOT）事件对
    for (auto& pair : decode_pending_) {
        if (!pair.start || !pair.stop) continue;
        CUDA_CHECK(cudaEventSynchronize(pair.stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, pair.start, pair.stop));
        tpot_total_ms_ += ms;
        decode_count_++;
        CUDA_CHECK(cudaEventDestroy(pair.start));
        CUDA_CHECK(cudaEventDestroy(pair.stop));
    }
    decode_pending_.clear();
}

// ============================================================================
// Reset
// 重置
// ============================================================================

void EngineProfiler::reset() {
    // Destroy any unread pending events before clearing
    // 清空前销毁所有未读的待处理事件
    for (auto& [name, entry] : device_timers_) {
        if (entry.cur_start) { CUDA_CHECK(cudaEventDestroy(entry.cur_start)); entry.cur_start = nullptr; }
        for (auto& pair : entry.pending) {
            if (pair.start) CUDA_CHECK(cudaEventDestroy(pair.start));
            if (pair.stop)  CUDA_CHECK(cudaEventDestroy(pair.stop));
        }
    }
    for (auto& pair : decode_pending_) {
        if (pair.start) CUDA_CHECK(cudaEventDestroy(pair.start));
        if (pair.stop)  CUDA_CHECK(cudaEventDestroy(pair.stop));
    }

    host_timers_.clear();
    device_timers_.clear();
    decode_pending_.clear();

    ttft_ms_           = -1.0;
    tpot_total_ms_     = 0.0f;
    decode_count_      = 0;
    decode_step_running_ = false;
}

// ============================================================================
// ASCII Report
// ASCII 报告
// ============================================================================

// Box dimensions / 表格尺寸
static constexpr int BOX_W = 60;  // total width including borders / 含边框的总宽度

static void _repeat(const char* s, int n) { for (int i = 0; i < n; ++i) fputs(s, stdout); }

void EngineProfiler::_box_top() {
    printf("╔"); _repeat("═", BOX_W - 2); printf("╗\n");
}
void EngineProfiler::_box_bottom() {
    printf("╚"); _repeat("═", BOX_W - 2); printf("╝\n");
}
void EngineProfiler::_box_divider() {
    printf("╠"); _repeat("═", BOX_W - 2); printf("╣\n");
}

void EngineProfiler::_box_section(const char* title) {
    _box_divider();
    int inner  = BOX_W - 4;                          // space between "║  " and "  ║"
    int tlen   = static_cast<int>(strlen(title));
    int pad_l  = (inner - tlen) / 2;
    int pad_r  = inner - tlen - pad_l;
    printf("║  %*s%s%*s  ║\n", pad_l, "", title, pad_r, "");
    _box_divider();
}

// Print one label-value row, dots fill the gap.
// 打印一行标签-值，点号填充间距。
void EngineProfiler::_box_row(const char* label, const char* value) {
    constexpr int inner = BOX_W - 4;   // "║  " + content + "  ║"
    int llen   = static_cast<int>(strlen(label));
    int vlen   = static_cast<int>(strlen(value));
    int gap    = inner - llen - vlen;
    if (gap < 1) gap = 1;
    printf("║  %s", label);
    _repeat(".", gap);
    printf("%s  ║\n", value);
}

// ============================================================================
// print_report
// ============================================================================
void EngineProfiler::print_report() {
    flush_device_timers();   // ensure all GPU timers are resolved first
                             // 先确保所有 GPU 计时器已解析

    // ── Header ──────────────────────────────────────────────────────── //
    _box_top();
    {
        constexpr const char* title = "AtomFlow Engine  ·  Performance Report";
        int inner = BOX_W - 4;
        int tlen  = static_cast<int>(strlen(title));
        int pad_l = (inner - tlen) / 2;
        int pad_r = inner - tlen - pad_l;
        printf("║  %*s%s%*s  ║\n", pad_l, "", title, pad_r, "");
    }

    // ── Latency ─────────────────────────────────────────────────────── //
    _box_section("Latency Metrics");

    // TTFT — host-side chrono
    {
        char vbuf[32];
        if (ttft_ms_ >= 0.0) snprintf(vbuf, sizeof(vbuf), "%8.2f ms  [host]", ttft_ms_);
        else                  snprintf(vbuf, sizeof(vbuf), "       N/A");
        _box_row("TTFT (Time To First Token)", vbuf);
    }

    // TPOT — device-side cudaEvent
    if (decode_count_ > 0) {
        double avg_tpot = tpot_total_ms_ / static_cast<double>(decode_count_);
        double tok_s    = (avg_tpot > 0.0) ? (1000.0 / avg_tpot) : 0.0;
        char v1[40], v2[40], v3[32];
        snprintf(v1, sizeof(v1), "%8.2f ms  [device]", avg_tpot);
        snprintf(v2, sizeof(v2), "%8.1f tok/s", tok_s);
        snprintf(v3, sizeof(v3), "%8u tokens", decode_count_);
        _box_row("TPOT avg (per output token)", v1);
        _box_row("Generation speed",            v2);
        _box_row("Tokens generated",            v3);
    } else {
        _box_row("TPOT", "       N/A  (no decode steps)");
    }

    // ── Kernel Timings ──────────────────────────────────────────────── //
    // Collect device timers that have data, sort by total descending.
    // 收集有数据的设备计时器，按总耗时降序排列。
    std::vector<std::pair<std::string, const DeviceTimerEntry*>> dev_rows;
    for (const auto& [name, entry] : device_timers_)
        if (entry.count > 0) dev_rows.emplace_back(name, &entry);
    std::sort(dev_rows.begin(), dev_rows.end(),
        [](const auto& a, const auto& b){ return a.second->total_ms > b.second->total_ms; });

    if (!dev_rows.empty()) {
        _box_section("Kernel Timings  (device · cudaEvent_t · avg per call)");
        for (const auto& [name, ep] : dev_rows) {
            double avg = ep->total_ms / static_cast<double>(ep->count);
            char   lbuf[40], vbuf[32];
            snprintf(lbuf, sizeof(lbuf),  "  %-22s x%-4u", name.c_str(), ep->count);
            snprintf(vbuf, sizeof(vbuf), "%7.3f ms", avg);
            _box_row(lbuf, vbuf);
        }
    }

    // ── Host Timings ────────────────────────────────────────────────── //
    // [Bug/Imperfection: Host-side chrono values include driver launch
    //  overhead, cuBLAS workspace lookups, and OS scheduling jitter.
    //  They represent wall-clock time, NOT pure compute time.
    //  For any metric where GPU execution time is the quantity of interest,
    //  always prefer the cudaEvent_t values printed above.
    //  主机端 chrono 值包含驱动启动开销、cuBLAS workspace 查找
    //  和操作系统调度抖动，代表挂钟时间，而非纯计算时间。
    //  对于任何关注 GPU 执行时间的指标，始终优先使用上方的 cudaEvent_t 值。]
    bool has_host = false;
    for (const auto& [n, e] : host_timers_) if (e.count > 0) { has_host = true; break; }

    if (has_host) {
        _box_section("Host Timings  (wall-clock · chrono · avg per call)");
        for (const auto& [name, entry] : host_timers_) {
            if (entry.count == 0) continue;
            double avg = entry.total_ms / entry.count;
            char   lbuf[40], vbuf[32];
            snprintf(lbuf, sizeof(lbuf), "  %-22s x%-4u", name.c_str(), entry.count);
            snprintf(vbuf, sizeof(vbuf), "%7.3f ms", avg);
            _box_row(lbuf, vbuf);
        }
    }

    // ── Memory Bandwidth ────────────────────────────────────────────── //
    // Collect bandwidth entries from device timers that have bytes_rw data.
    // 从具有 bytes_rw 数据的设备计时器中收集带宽条目。
    struct BwRow { std::string name; double gb_s; };
    std::vector<BwRow> bw_rows;

    // Build bandwidth rows from device timers that have bytes_rw annotations.
    // GB/s = (bytes / 1e9) / (total_ms / 1e3) = bytes / (total_ms * 1e6)
    // 从具有 bytes_rw 注释的设备计时器构建带宽行。
    // [Bug/Imperfection: bytes_total reflects the THEORETICAL I/O size passed
    //  by the caller via annotate_bandwidth(). It does NOT measure actual HBM
    //  traffic, which can be lower due to L2 hits or higher due to replays.
    //  Ground-truth HBM bandwidth requires Nsight Compute's
    //  l2_global_load / l2_global_store hardware performance counters.
    //  bytes_total 反映调用者通过 annotate_bandwidth() 传入的理论 I/O 大小。
    //  它不测量实际 HBM 流量——L2 命中会使其偏低，重放会使其偏高。
    //  精确的 HBM 带宽需要 Nsight Compute 的 l2_global_load/store 硬件计数器。]
    for (const auto& [name, entry] : device_timers_) {
        if (entry.count == 0 || entry.bytes_total == 0) continue;
        double gb_s = static_cast<double>(entry.bytes_total)
                      / (static_cast<double>(entry.total_ms) * 1e6);
        bw_rows.push_back({name, gb_s});
    }
    std::sort(bw_rows.begin(), bw_rows.end(),
        [](const BwRow& a, const BwRow& b){ return a.gb_s > b.gb_s; });

    if (!bw_rows.empty()) {
        _box_section("Memory Bandwidth  (theoretical · from annotate_bandwidth)");
        for (const auto& row : bw_rows) {
            char lbuf[40], vbuf[32];
            snprintf(lbuf, sizeof(lbuf), "  %-26s", row.name.c_str());
            snprintf(vbuf, sizeof(vbuf), "%7.1f GB/s", row.gb_s);
            _box_row(lbuf, vbuf);
        }
    }

    // ── Footer ──────────────────────────────────────────────────────── //
    _box_bottom();
}

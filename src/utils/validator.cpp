// validator.cpp
// Ground-truth cosine-similarity validator implementation.
// 基准真值余弦相似度验证器实现。

#include "utils/validator.h"
#include "utils/utils.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>

// ============================================================================
// ANSI colour codes / ANSI 颜色代码
// ============================================================================
static constexpr const char* COL_RESET  = "\033[0m";
static constexpr const char* COL_GREEN  = "\033[32;1m";
static constexpr const char* COL_YELLOW = "\033[33;1m";
static constexpr const char* COL_RED    = "\033[31;1m";
static constexpr const char* COL_CYAN   = "\033[36;1m";
static constexpr const char* COL_BOLD   = "\033[1m";

// ============================================================================
// Internal helpers / 内部辅助函数
// ============================================================================

// load_fp32_binary
//   Read a raw FP32 binary from disk; returns number of floats loaded.
//   从磁盘读取原始 FP32 二进制；返回加载的 float 数量。
//
//   [Bug/Imperfection: No endian check. Files are assumed to be little-endian
//    (x86/ARM default). Running on a big-endian host will silently corrupt data.
//    无端序检查。假定文件为小端序（x86/ARM 默认）。
//    在大端序主机上运行将静默损坏数据。]
static size_t load_fp32_binary(const std::string& path, std::vector<float>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        std::fprintf(stderr, "[Validator] ERROR: cannot open '%s'\n", path.c_str());
        return 0;
    }
    const std::streamsize byte_count = f.tellg();
    f.seekg(0, std::ios::beg);
    const size_t n = static_cast<size_t>(byte_count) / sizeof(float);
    out.resize(n);
    f.read(reinterpret_cast<char*>(out.data()), byte_count);
    return n;
}

// download_view_to_float
//   Synchronously copies a GPU View to a host float vector.
//   Handles both FP16 and FP32 engine tensors.
//   同步将 GPU View 复制到主机 float 向量。
//   处理 FP16 和 FP32 引擎张量。
//
//   [Bug/Imperfection: cudaMemcpy (DeviceToHost) is a blocking D2H transfer
//    that stalls any in-flight CUDA kernels until the copy completes.
//    For a 3072-element FP16 vector this is ~6 KB and costs ~5–20 µs of
//    PCIe latency on top of pipeline serialisation overhead.
//    cudaMemcpy D2H 会阻塞所有进行中的 CUDA kernel 直到复制完成。
//    对于 3072 元素 FP16 向量，约 6 KB，在流水线串行化开销之上额外产生
//    约 5–20 µs 的 PCIe 延迟。]
static size_t download_view_to_float(const View& v, std::vector<float>& out) {
    // Count total elements / 统计总元素数
    size_t n = 1;
    for (int i = 0; i < v.num_dims; ++i) n *= static_cast<size_t>(v.dims[i]);
    out.resize(n);

    if (v.dtype == DataType::FP32) {
        // Direct copy: device FP32 → host float
        // 直接复制：device FP32 → host float
        CUDA_CHECK(cudaMemcpy(out.data(), v.data_ptr, n * sizeof(float),
                              cudaMemcpyDeviceToHost));
    } else if (v.dtype == DataType::FP16) {
        // D2H in FP16, then convert to float on CPU
        // 以 FP16 做 D2H，然后在 CPU 上转换为 float
        //
        // [Bug/Imperfection: Converting FP16→float on the CPU introduces
        //  zero additional precision error but wastes bandwidth compared to
        //  running a cast kernel on the GPU before the D2H transfer.
        //  在 CPU 上 FP16→float 不引入额外精度误差，但相比在 GPU 上
        //  先做 cast kernel 再 D2H，浪费了带宽。]
        std::vector<uint16_t> tmp(n);
        CUDA_CHECK(cudaMemcpy(tmp.data(), v.data_ptr, n * sizeof(uint16_t),
                              cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < n; ++i) {
            // Reinterpret bits as __half, then convert
            // 将位重解释为 __half，然后转换
            __half h;
            memcpy(&h, &tmp[i], sizeof(uint16_t));
            out[i] = __half2float(h);
        }
    } else {
        std::fprintf(stderr,
            "[Validator] WARNING: unsupported dtype %d — skipping D2H\n",
            static_cast<int>(v.dtype));
        return 0;
    }
    return n;
}

// cosine_similarity + relative_l2
//   Both computed in a single pass over the data for cache efficiency.
//   单次遍历数据同时计算余弦相似度和相对 L2 误差，提高缓存效率。
struct Metrics {
    float cosine;   // [-1, 1]
    float rel_l2;   // >= 0
};
static Metrics compute_metrics(const float* a, const float* b, size_t n) {
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0, l2_diff = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double ai = a[i], bi = b[i];
        dot    += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
        double d = ai - bi;
        l2_diff += d * d;
    }
    const float denom = static_cast<float>(std::sqrt(norm_a) * std::sqrt(norm_b));
    const float cosine = (denom > 1e-12f)
                       ? static_cast<float>(dot / (double)denom)
                       : 0.0f;
    const float rel_l2 = (norm_b > 1e-24)
                       ? static_cast<float>(std::sqrt(l2_diff / norm_b))
                       : 0.0f;
    return {cosine, rel_l2};
}

// ============================================================================
// Public API implementation / 公共接口实现
// ============================================================================

ValidationResult validate_view(const View&        engine_view,
                                const std::string& gt_filepath,
                                const std::string& name) {
    ValidationResult res{};
    res.name    = name.empty() ? gt_filepath : name;
    res.gt_path = gt_filepath;
    res.passed  = false;

    // Step A: Download engine view to host float vector
    // 步骤 A：将引擎 View 下载到主机 float 向量
    std::vector<float> engine_f;
    const size_t n_engine = download_view_to_float(engine_view, engine_f);
    if (n_engine == 0) {
        res.cosine_sim = 0.0f;
        res.l2_rel_err = 9999.0f;
        res.numel      = 0;
        return res;
    }

    // Step B: Load ground-truth binary
    // 步骤 B：加载基准真值二进制
    std::vector<float> gt_f;
    const size_t n_gt = load_fp32_binary(gt_filepath, gt_f);
    if (n_gt == 0) {
        res.cosine_sim = 0.0f;
        res.l2_rel_err = 9999.0f;
        res.numel      = 0;
        return res;
    }

    // Step C: Element count mismatch check
    // 步骤 C：元素数量不匹配检查
    //
    // [Bug/Imperfection: AtomFlow captures only the LAST token's hidden state
    //  ([1, D]), while HuggingFace hooks capture all positions ([1, seq_len, D]).
    //  The Python script trims to the last position before saving, so n_engine
    //  and n_gt should match. If they do not, it indicates a shape disagreement
    //  between the Python export and the C++ engine that must be investigated.
    //  AtomFlow 仅捕获最后 token 的隐藏状态（[1, D]），而 HuggingFace hook
    //  捕获所有位置（[1, seq_len, D]）。Python 脚本在保存前截取最后位置，
    //  因此 n_engine 与 n_gt 应一致。若不一致，说明 Python 导出与 C++ 引擎
    //  之间存在形状分歧，必须排查。]
    const size_t n = std::min(n_engine, n_gt);
    if (n_engine != n_gt) {
        std::fprintf(stderr,
            "[Validator] %sWARN%s size mismatch for '%s': "
            "engine=%zu  gt=%zu  — comparing first %zu elements\n",
            COL_YELLOW, COL_RESET, res.name.c_str(), n_engine, n_gt, n);
    }
    res.numel = n;

    // Step D: Compute metrics
    // 步骤 D：计算指标
    const Metrics m = compute_metrics(engine_f.data(), gt_f.data(), n);
    res.cosine_sim = m.cosine;
    res.l2_rel_err = m.rel_l2;

    // Step E: Determine pass/warn/fail and print
    // 步骤 E：判断通过/警告/失败并打印
    const char* tag;
    const char* col;
    if (res.cosine_sim >= VALIDATOR_PASS_THRESHOLD) {
        tag     = "PASS ";
        col     = COL_GREEN;
        res.passed = true;
    } else if (res.cosine_sim >= VALIDATOR_WARN_THRESHOLD) {
        tag = "WARN ";
        col = COL_YELLOW;
    } else {
        tag = "FATAL";
        col = COL_RED;
    }

    std::printf("  %s[%s]%s  %-30s  cos=%.6f  rel_l2=%.4e  n=%zu\n",
                col, tag, COL_RESET,
                res.name.c_str(),
                res.cosine_sim,
                res.l2_rel_err,
                res.numel);

    return res;
}

void validate_print_header() {
    std::printf("\n%s", COL_BOLD);
    std::printf("╔══════════════════════════════════════════════════════════════════╗\n");
    std::printf("║       AtomFlow  ·  Ground Truth Cosine Similarity Report         ║\n");
    std::printf("║   Thresholds:  >=%.2f → PASS   >=%.2f → WARN   <%.2f → FATAL   ║\n",
                VALIDATOR_PASS_THRESHOLD,
                VALIDATOR_WARN_THRESHOLD,
                VALIDATOR_WARN_THRESHOLD);
    std::printf("╠══════════════════════════════════════════════════════════════════╣\n");
    std::printf("%s", COL_RESET);
    std::printf("  %-7s  %-30s  %-12s  %-12s  %s\n",
                "[Status]", "Name", "Cosine Sim", "Rel L2 Err", "Numel");
    std::printf("  %-7s  %-30s  %-12s  %-12s  %s\n",
                "-------", "------------------------------",
                "----------", "----------", "-----");
}

void validate_print_footer(int n_passed, int n_total) {
    const bool all_pass = (n_passed == n_total);
    const char* col = all_pass ? COL_GREEN : COL_RED;
    std::printf("\n%s", COL_BOLD);
    std::printf("╠══════════════════════════════════════════════════════════════════╣\n");
    std::printf("║  Result:  %s%d / %d passed%s",
                col, n_passed, n_total, COL_RESET);
    // Pad to fill the box
    // 填充以对齐箱子宽度
    int pad = 50 - std::snprintf(nullptr, 0, "%d / %d passed", n_passed, n_total);
    for (int i = 0; i < pad; ++i) std::printf(" ");
    std::printf("%s║\n", COL_BOLD);
    std::printf("╚══════════════════════════════════════════════════════════════════╝\n");
    std::printf("%s\n", COL_RESET);
}

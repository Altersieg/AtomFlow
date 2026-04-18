#pragma once
// validator.h
// Ground-truth cosine-similarity validator for AtomFlow internal tensors.
// 用于 AtomFlow 内部张量的基准真值余弦相似度验证器。

#include <string>
#include "core/view.h"

// ============================================================================
// Thresholds / 阈值
// ============================================================================
// Similarity >= PASS_THRESHOLD  → [PASS]   (green)
// Similarity <  WARN_THRESHOLD  → [WARN]   (yellow)
// Similarity <  FAIL_THRESHOLD  → [FATAL]  (red)
constexpr float VALIDATOR_PASS_THRESHOLD = 0.99f;
constexpr float VALIDATOR_WARN_THRESHOLD = 0.95f;

// ============================================================================
// ValidationResult
// ============================================================================
struct ValidationResult {
    std::string name;         // human label / 人类可读标签
    std::string gt_path;      // path to ground-truth .bin / 基准真值文件路径
    float       cosine_sim;   // cosine similarity [-1, 1] / 余弦相似度
    float       l2_rel_err;   // relative L2 error ||a-b||/||b|| / 相对 L2 误差
    size_t      numel;        // number of elements compared / 比较的元素数
    bool        passed;       // similarity >= PASS_THRESHOLD / 是否通过
};

// ============================================================================
// Primary API / 主接口
// ============================================================================

// validate_view
//   Downloads engine_view from GPU to CPU, loads the ground-truth binary,
//   computes cosine similarity and relative L2, prints a colour-coded report line.
//
//   将 engine_view 从 GPU 下载到 CPU，加载基准真值二进制，
//   计算余弦相似度和相对 L2 误差，打印彩色报告行。
//
// Parameters / 参数:
//   engine_view  — View pointing into GPU activation pool / 指向 GPU 激活池的 View
//   gt_filepath  — path to FP32 binary written by dump_ground_truth.py / FP32 二进制路径
//   name         — label for the report line / 报告行标签
//
// Returns / 返回:
//   ValidationResult with all computed fields / 包含所有计算字段的 ValidationResult
//
// [Bug/Imperfection: This function calls cudaMemcpy (device→host) synchronously,
//  which serialises the GPU pipeline and will dominate latency if called inside
//  the hot forward loop. Only call it during offline debug passes, not during
//  production profiling runs.
//  此函数同步调用 cudaMemcpy（device→host），这会串行化 GPU 管线，
//  如果在热前向循环中调用将主导延迟。仅在离线调试中调用，
//  不应在生产性能分析时调用。]
ValidationResult validate_view(const View&        engine_view,
                                const std::string& gt_filepath,
                                const std::string& name = "");

// validate_print_header / validate_print_footer
//   Print the ASCII table border around a validation session.
//   打印验证会话的 ASCII 表格边框。
void validate_print_header();
void validate_print_footer(int n_passed, int n_total);

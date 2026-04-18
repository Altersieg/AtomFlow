#pragma once
// utils.h
// Lightweight, zero-overhead CUDA and cuBLAS error-checking macros.
// 轻量级、零开销的 CUDA 和 cuBLAS 错误检查宏。
//
// Debug mode   (NDEBUG not defined): every call is checked; abort() on failure.
// Release mode (NDEBUG     defined): macros expand to a bare call with no branches.
//
// 调试模式  （未定义 NDEBUG）：每次调用都被检查；失败时 abort()。
// 发布模式  （已定义 NDEBUG）：宏展开为不含分支的裸调用。

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// ============================================================================
// cuBLAS error-to-string helper
// cuBLAS 错误转字符串辅助函数
//
// cuBLAS has no official cublasGetErrorString(); we provide our own switch.
// cuBLAS 没有官方的 cublasGetErrorString()，此处提供自定义 switch。
// ============================================================================
inline const char* cublas_status_string(cublasStatus_t s) {
    switch (s) {
        case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:    return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:    return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:    return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:    return "CUBLAS_STATUS_LICENSE_ERROR";
        default:                             return "CUBLAS_STATUS_UNKNOWN";
    }
}

// ============================================================================
#ifndef NDEBUG
// ── DEBUG mode: full checking with abort ─────────────────────────────────
// 调试模式：完整检查，失败时 abort
// ============================================================================

// CUDA_CHECK — wrap any cudaError_t-returning CUDA Runtime API call.
// CUDA_CHECK — 包裹任何返回 cudaError_t 的 CUDA Runtime API 调用。
#define CUDA_CHECK(call)                                                           \
    do {                                                                           \
        cudaError_t _e = (call);                                                   \
        if (_e != cudaSuccess) {                                                   \
            std::fprintf(stderr,                                                   \
                "\n[CUDA ERROR]  %s:%d\n"                                          \
                "  call  : " #call "\n"                                            \
                "  error : %s\n",                                                  \
                __FILE__, __LINE__, cudaGetErrorString(_e));                       \
            std::abort();                                                          \
        }                                                                          \
    } while (0)

// CUBLAS_CHECK — wrap any cublasStatus_t-returning cuBLAS API call.
// CUBLAS_CHECK — 包裹任何返回 cublasStatus_t 的 cuBLAS API 调用。
#define CUBLAS_CHECK(call)                                                         \
    do {                                                                           \
        cublasStatus_t _s = (call);                                                \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                         \
            std::fprintf(stderr,                                                   \
                "\n[CUBLAS ERROR]  %s:%d\n"                                        \
                "  call  : " #call "\n"                                            \
                "  error : %s\n",                                                  \
                __FILE__, __LINE__, cublas_status_string(_s));                     \
            std::abort();                                                          \
        }                                                                          \
    } while (0)

// CUDA_CHECK_LAST — call cudaGetLastError() to catch asynchronous kernel errors.
// Place IMMEDIATELY after a <<<>>> kernel launch in the same translation unit.
//
// CUDA_CHECK_LAST — 调用 cudaGetLastError() 捕获异步 kernel 错误。
// 在同一翻译单元中紧接 <<<>>> kernel 启动之后放置。
//
// [Bug/Imperfection: cudaGetLastError() only catches launch-configuration
//  errors (bad grid/block dims, insufficient shared memory, etc.) that the
//  driver detects at launch time. Actual kernel execution errors (e.g. invalid
//  memory access) are NOT caught here — they surface as Error 700
//  (cudaErrorIllegalAddress) on the NEXT synchronising API call.
//  To catch runtime errors inside kernels, also add CUDA_CHECK after the next
//  cudaStreamSynchronize or cudaMemcpy following the launch.
//  cudaGetLastError() 仅捕获驱动在启动时检测到的启动配置错误
//  （网格/线程块维度错误、共享内存不足等）。
//  Kernel 执行期间的实际错误（如非法内存访问）不在此处捕获——
//  它们会在下一个同步 API 调用时以 Error 700 浮现。
//  要捕获 Kernel 内部运行时错误，需在启动后的
//  cudaStreamSynchronize 或 cudaMemcpy 之后再加一次 CUDA_CHECK。]
#define CUDA_CHECK_LAST()                                                          \
    do {                                                                           \
        cudaError_t _e = cudaGetLastError();                                       \
        if (_e != cudaSuccess) {                                                   \
            std::fprintf(stderr,                                                   \
                "\n[CUDA KERNEL LAUNCH ERROR]  %s:%d\n"                           \
                "  error : %s\n",                                                  \
                __FILE__, __LINE__, cudaGetErrorString(_e));                       \
            std::abort();                                                          \
        }                                                                          \
    } while (0)

#else
// ── RELEASE mode (NDEBUG defined): zero overhead ─────────────────────────
// 发布模式（已定义 NDEBUG）：零开销
//
// The call is still executed; its return value is silently discarded.
// 调用仍然执行；其返回值被静默丢弃。
// ============================================================================
#define CUDA_CHECK(call)    (void)(call)
#define CUBLAS_CHECK(call)  (void)(call)
#define CUDA_CHECK_LAST()   do {} while (0)

#endif  // NDEBUG

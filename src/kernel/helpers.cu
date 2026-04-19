#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "utils/utils.h"
#include "ops/kernel.h"

// ============================================================================
// Small device-side helpers that avoid CPU↔GPU round-trips
// 避免 CPU↔GPU 往返的小型设备端辅助 kernel
// ============================================================================

// [EN] Embed lookup: read row `token_id` from FP32 embed table, write FP16 to dst.
// [CN] 嵌入查找：从 FP32 嵌入表中读取 token_id 对应行，写入 FP16 dst。
__global__ void k_embed_lookup_kernel(const float* table, int token_id,
                                      half* dst, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D) dst[i] = __float2half(table[token_id * D + i]);
}

void launch_embed_lookup(const float* table, int token_id,
                         half* dst, int D, cudaStream_t stream) {
    k_embed_lookup_kernel<<<(D + 255) / 256, 256, 0, stream>>>(
        table, token_id, dst, D);
    CUDA_CHECK_LAST();
}

// [EN] FP16 → FP32 cast for lm_head input (needs float path into cublasSgemm).
// [CN] FP16 转 FP32，供 lm_head 使用（cublasSgemm 需要 float 输入）。
__global__ void k_cast_fp16_to_fp32_kernel(const half* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

void launch_cast_fp16_to_fp32(const half* src, float* dst, int n,
                              cudaStream_t stream) {
    k_cast_fp16_to_fp32_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        src, dst, n);
    CUDA_CHECK_LAST();
}

// ============================================================================
// load_fp32_bin_to_fp16_device
//
// [EN] Reads a raw FP32 binary file from disk, converts each element to FP16
//      on the CPU, then uploads the converted data to GPU via cudaMemcpy.
// [CN] 从磁盘读取原始 FP32 二进制文件，在 CPU 端逐元素转换为 FP16，
//      然后通过 cudaMemcpy 同步上传到 GPU 目标地址。
//
// Returns number of elements loaded, or 0 on failure.
// 返回已加载的元素数量；失败时返回 0。
// ============================================================================
size_t load_fp32_bin_to_fp16_device(const char* path,
                                    void*       d_dst,
                                    int         expected_numel)
{
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return 0;

    std::vector<float> h_fp32(expected_numel);
    size_t n_read = std::fread(h_fp32.data(), sizeof(float), expected_numel, f);
    std::fclose(f);

    if (n_read != static_cast<size_t>(expected_numel)) {
        std::fprintf(stderr,
            "[load_fp32_bin] WARNING: %s — expected %d floats, got %zu\n",
            path, expected_numel, n_read);
        return 0;
    }

    std::vector<__half> h_fp16(expected_numel);
    for (int i = 0; i < expected_numel; ++i) {
        h_fp16[i] = __float2half(h_fp32[i]);
    }

    CUDA_CHECK(cudaMemcpy(d_dst, h_fp16.data(),
                          expected_numel * sizeof(__half),
                          cudaMemcpyHostToDevice));

    return static_cast<size_t>(expected_numel);
}

// ============================================================================
// RoPE precomputation (CPU → GPU)
// RoPE 预计算（CPU → GPU）
// ============================================================================
void build_rope_cache(float* d_cos, float* d_sin,
                      int max_seq, int head_dim, float base,
                      cudaStream_t stream) {
    std::vector<float> h_cos(max_seq * head_dim);
    std::vector<float> h_sin(max_seq * head_dim);
    for (int s = 0; s < max_seq; ++s) {
        for (int i = 0; i < head_dim / 2; ++i) {
            float theta = s / std::pow(base, 2.0f * i / head_dim);
            h_cos[s * head_dim + 2 * i]     = std::cos(theta);
            h_cos[s * head_dim + 2 * i + 1] = std::cos(theta);
            h_sin[s * head_dim + 2 * i]     = std::sin(theta);
            h_sin[s * head_dim + 2 * i + 1] = std::sin(theta);
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(d_cos, h_cos.data(),
                               max_seq * head_dim * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_sin, h_sin.data(),
                               max_seq * head_dim * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
}

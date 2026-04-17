#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>
#include <stdexcept>
#include "view.h"
#include "kernel.h"

// [EN] Rotary Position Embedding kernel. Rotates pairs (d, d+head_dim/2) in-place for Q and K.
// [CN] 旋转位置编码核函数。就地旋转 Q 和 K 的 (d, d+head_dim/2) 对。
template<typename T>
__global__ void rope_kernel(
    T* q_ptr,                   // [1, Seq, q_head_num * head_dim]
    T* k_ptr,                   // [1, Seq, kv_head_num * head_dim]
    const float* cos_cache,
    const float* sin_cache,
    int seq_len,
    int kv_head_num,
    int q_head_num,
    int head_dim)
{
    int pos = blockIdx.x;       // token index
    int d   = threadIdx.x;      // pair index (0 .. head_dim/2 - 1)
    int cache_id = pos * (head_dim / 2) + d;

    float cos_val = cos_cache[cache_id];
    float sin_val = sin_cache[cache_id];

    // --- Q heads ---
    for (int h = 0; h < q_head_num; ++h) {
        T* q_cur = q_ptr + pos * q_head_num * head_dim + h * head_dim;
        float x = (float)q_cur[d];
        float y = (float)q_cur[d + head_dim / 2];
        q_cur[d]                 = (T)(x * cos_val - y * sin_val);
        q_cur[d + head_dim / 2]  = (T)(x * sin_val + y * cos_val);
    }

    // --- K heads (GQA: fewer than Q) ---
    for (int h = 0; h < kv_head_num; ++h) {
        T* k_cur = k_ptr + pos * kv_head_num * head_dim + h * head_dim;
        float x = (float)k_cur[d];
        float y = (float)k_cur[d + head_dim / 2];
        k_cur[d]                 = (T)(x * cos_val - y * sin_val);
        k_cur[d + head_dim / 2]  = (T)(x * sin_val + y * cos_val);
    }
}

void launch_rope(
    const View& q_input,
    const View& k_input,
    const float* cos_cache,
    const float* sin_cache,
    int seq_len,
    int kv_head_num,
    int q_head_num,
    int head_dim,
    cudaStream_t stream)
{
    int threads_per_block = head_dim / 2;
    int blocks_per_grid   = seq_len;

    switch (q_input.dtype) {
        case DataType::FP16: {
            half* q_ptr = static_cast<half*>(q_input.data_ptr);
            half* k_ptr = static_cast<half*>(k_input.data_ptr);
            rope_kernel<half><<<blocks_per_grid, threads_per_block, 0, stream>>>(
                q_ptr, k_ptr, cos_cache, sin_cache,
                seq_len, kv_head_num, q_head_num, head_dim);
            break;
        }
        case DataType::FP8_E4M3: {
            __nv_fp8_e4m3* q_ptr = static_cast<__nv_fp8_e4m3*>(q_input.data_ptr);
            __nv_fp8_e4m3* k_ptr = static_cast<__nv_fp8_e4m3*>(k_input.data_ptr);
            rope_kernel<__nv_fp8_e4m3><<<blocks_per_grid, threads_per_block, 0, stream>>>(
                q_ptr, k_ptr, cos_cache, sin_cache,
                seq_len, kv_head_num, q_head_num, head_dim);
            break;
        }
        default:
            throw std::runtime_error("Unsupported data type in RoPE");
    }
}

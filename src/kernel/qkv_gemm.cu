#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "utils/utils.h"
#include "core/view.h"
#include "ops/kernel.h"

// ============================================================================
// Fused W8A16 GEMV Kernel — eliminates intermediate global-memory write-back.
// 融合 W8A16 GEMV 内核 — 消除中间全局显存写回。
//
// Operation / 运算:
//   Y[n] = sum_k( dequant(W_fp8[n,k], scale[n, k/GS]) * X_fp16[k] )
//   for n in [0, N),  where W is [N, K] FP8-E4M3 row-major.
//
// Design / 设计:
//   - One block computes one output element Y[n].
//     每个 block 计算一个输出元素 Y[n]。
//   - BLOCK_DIM threads collaborate on the K-dimension dot product.
//     BLOCK_DIM 个线程协作完成 K 维内积。
//   - Vectorized reads: 4 FP8 bytes via uint32_t, 4 FP16 via float2.
//     向量化读取：uint32_t 读 4 个 FP8，float2 读 4 个 FP16。
//   - In-register dequant: FP8→float × scale, NO global memory write.
//     寄存器内反量化：FP8→float × scale，不写全局显存。
//   - FP32 accumulation → warp shuffle + shared-memory block reduction.
//     FP32 累加 → warp shuffle + 共享内存块规约。
//
// Constraint: GS (group_size=128) must be divisible by VEC (4).
//   Since 128 / 4 = 32, vectorized reads never cross a scale-group boundary.
//   GS 必须能被 VEC 整除。128 / 4 = 32，向量化读取不会跨 scale 组边界。
// ============================================================================
template <int BLOCK_DIM = 256>
static __global__ void w8a16_gemv_kernel(
    const uint8_t* __restrict__ weight,   // FP8 E4M3 [N, K] row-major / 行主序
    const half*    __restrict__ act_x,    // FP16 activation [K] / FP16 激活向量
    const half*    __restrict__ scales,   // FP16 [N, K/GS] per-group scales / 逐组缩放
    half*          __restrict__ out,      // FP16 output [N] / 输出向量
    int K,                                // inner dimension / 内维度
    int num_groups)                       // K / GS = groups per row / 每行组数
{
    // ── 1. One block → one output element n / 每个 block 处理一个输出 n ───────
    const int n   = blockIdx.x;
    const int tid = threadIdx.x;

    // Row pointers for this output element
    // 当前输出元素对应的行指针
    const uint8_t* w_row = weight + (size_t)n * K;
    const half*    s_row = scales + (size_t)n * num_groups;

    // ── 2. Vectorized inner loop: 4 elements per iteration ───────────────────
    //       向量化内循环：每次处理 4 个元素
    constexpr int VEC = 4;
    const int k_start = tid * VEC;             // first element for this thread
    const int stride  = BLOCK_DIM * VEC;       // stride across K dimension

    float sum = 0.0f;

    for (int kv = k_start; kv < K; kv += stride) {
        // Read 4 FP8 weight bytes as a single uint32_t (coalesced 4-byte load).
        // 以 uint32_t 一次读取 4 个 FP8 字节（合并 4 字节加载）。
        uint32_t w_packed = *reinterpret_cast<const uint32_t*>(w_row + kv);

        // Read 4 FP16 activation values as float2 (coalesced 8-byte load).
        // 以 float2 一次读取 4 个 FP16 激活值（合并 8 字节加载）。
        float2 ax_packed = *reinterpret_cast<const float2*>(act_x + kv);
        const half* ax = reinterpret_cast<const half*>(&ax_packed);

        // Per-group scale: all 4 elements share the same scale (GS % VEC == 0).
        // 逐组 scale：4 个元素共享同一 scale（GS 能被 VEC 整除）。
        float s = __half2float(s_row[kv / 128]);   // GS = 128, hardcoded for speed

        // Unpack 4 FP8 → float, dequant, fused multiply-add.
        // 解包 4 个 FP8 → float，反量化，融合乘加。
        const uint8_t* w_bytes = reinterpret_cast<const uint8_t*>(&w_packed);

        #pragma unroll
        for (int v = 0; v < VEC; ++v) {
            // Reinterpret uint8_t as __nv_fp8_e4m3 and convert to float.
            // 将 uint8_t 重解释为 __nv_fp8_e4m3 并转换为 float。
            float w_val = static_cast<float>(
                *reinterpret_cast<const __nv_fp8_e4m3*>(&w_bytes[v]));
            float x_val = __half2float(ax[v]);
            sum += (w_val * s) * x_val;           // dequant + dot in registers
        }
    }

    // ── 3. Warp-level reduction via shuffle / Warp 级 shuffle 规约 ───────────
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    // ── 4. Block-level reduction via shared memory / Block 级共享内存规约 ─────
    constexpr int NUM_WARPS = BLOCK_DIM / 32;
    __shared__ float warp_sums[NUM_WARPS];

    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    if (lane_id == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    // Thread 0 performs final serial reduction (NUM_WARPS is small, e.g. 8).
    // 线程 0 做最终串行规约（NUM_WARPS 很小，如 8）。
    if (tid == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int i = 0; i < NUM_WARPS; ++i)
            total += warp_sums[i];
        out[n] = __float2half(total);
    }
}

// ============================================================================
// launch_w8a16_gemv — Host launcher for the fused W8A16 GEMV kernel.
// launch_w8a16_gemv — 融合 W8A16 GEMV 的主机端启动器。
//
// Replaces the two-step (launch_dequantize + launch_fp16x16_gemm) path
// for the M=1 decode phase, cutting HBM traffic by ~5×.
// 替换两步路径（launch_dequantize + launch_fp16x16_gemm），
// 在 M=1 解码阶段将 HBM 流量减少约 5 倍。
//
// Parameters match the existing View-based API:
//   act_x    — FP16 [1, K]  (row vector, K = last dim)
//   weight   — FP8  [N, K]  (row-major, same layout as before)
//   scales   — FP16 [N, K/GS]
//   output   — FP16 [1, N]
//   GS       — group size (must be 128)
// ============================================================================
void launch_w8a16_gemv(
    const View& act_x,      // FP16 [1, K]
    const View& weight,      // FP8  [N, K]
    const View& scales,      // FP16 [N, K/GS]
    View&       output,      // FP16 [1, N]
    int         group_size,  // 128
    cudaStream_t stream)
{
    const int N = weight.dims[0];
    const int K = weight.dims[1];
    const int num_groups = K / group_size;

    constexpr int BLOCK = 256;

    w8a16_gemv_kernel<BLOCK><<<N, BLOCK, 0, stream>>>(
        static_cast<const uint8_t*>(weight.data_ptr),
        static_cast<const half*>(act_x.data_ptr),
        static_cast<const half*>(scales.data_ptr),
        static_cast<half*>(output.data_ptr),
        K, num_groups);
    CUDA_CHECK_LAST();
}

// [EN] Universal Linear GEMM launcher (also used for QKV fused projection).
// [CN] 通用 Linear/GEMM 启动器 (同时用于融合 QKV 投影)。
// [Bug/Imperfection: Hardcodes FP8 weight + FP16 activation path; does not yet dispatch on View::dtype.
// 硬编码了 FP8 权重 + FP16 激活的路径；尚未根据 View::dtype 分派。]
void launch_linear_gemm(
    const View& input,    // [Total_Tokens, In_Features]
    const View& weight,   // [Out_Features, In_Features]
    View& output,         // [Total_Tokens, Out_Features]
    cublasHandle_t handle,
    cudaStream_t /*stream*/)
{
    // cuBLAS is column-major; we pass row-major params "flipped" to emulate row-major GEMM.
    int m = calculate_rows(input);          // Total tokens
    int k = input.dims[input.num_dims - 1]; // In features  (e.g. hidden_dim)
    int n = weight.dims[0];                 // Out features (e.g. QKV total dim)

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    CUBLAS_CHECK(cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_T,                  // weight is transposed on the fly
        n, m, k,
        &alpha,
        weight.data_ptr, CUDA_R_8F_E4M3, n,        // FP8 weights
        input.data_ptr,  CUDA_R_16F,    k,         // FP16 activations
        &beta,
        output.data_ptr, CUDA_R_16F,    n,         // FP16 output
        CUBLAS_COMPUTE_32F,                        // FP32 accumulation for accuracy
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));           // force Tensor Core
}

#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include "utils/utils.h"
#include "core/view.h"
#include "ops/kernel.h"

// ============================================================================
// dequant_fp8e4m3_to_fp16_kernel
//
// Converts a packed FP8-E4M3 weight tensor to FP16 by applying per-group
// scales (group_size=128 by default, matching the AtomFlow export format).
//
// 将 FP8-E4M3 权重张量按逐组 scale（默认 group_size=128，与 AtomFlow 导出格式匹配）
// 转换为 FP16 格式。
//
// Memory layout / 内存布局:
//   weight [rows, cols]           — FP8 E4M3 packed bytes
//   scales [rows, cols/group_size] — FP16 per-group scale factors
//   dst    [rows, cols]           — FP16 output
//
// Thread mapping / 线程映射:
//   One thread per output element: idx = row * cols + col.
//   Scale index for element (row, col): row * (cols/group_size) + col/group_size.
//   每个线程处理一个输出元素，scale 下标 = row*(cols/GS) + col/GS。
//
// [Bug/Imperfection: Each thread computes its (row, col) via integer division,
//  which is slow on GPU (idiv ~20 cycles). For production, prefer a 2-D grid
//  launch (blockIdx.x = col tile, blockIdx.y = row) to avoid the division.
//  每个线程通过整数除法计算 (row, col)，GPU 整除约需 20 个周期。
//  生产环境推荐使用二维 grid（blockIdx.x=列分块, blockIdx.y=行）以避免除法。]
// ============================================================================
static __global__ void dequant_fp8e4m3_to_fp16_kernel(
    const __nv_fp8_e4m3* __restrict__ src,    // FP8 weight tensor / FP8 权重张量
    const __half*         __restrict__ scales, // FP16 per-group scales / FP16 逐组缩放因子
    __half*               __restrict__ dst,    // FP16 output / FP16 输出
    int rows, int cols, int group_size)
{
    // ── 1. Flat element index / 展平元素下标 ─────────────────────────────────
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;

    // ── 2. Decode (row, col) to find the correct scale group
    //       解码 (row, col) 以找到对应的 scale 组
    int row = idx / cols;
    int col = idx % cols;
    int scale_idx = row * (cols / group_size) + col / group_size;

    // ── 3. Multiply: FP8 value × FP16 scale → FP16 output
    //       乘法：FP8 值 × FP16 scale → FP16 输出
    //
    //    CUDA's __nv_fp8_e4m3 has an implicit conversion operator to float,
    //    giving correct E4M3 decode (bias=7, no infinities, has NaN=0x7F).
    //    __nv_fp8_e4m3 具有隐式 float 转换运算符，正确解码 E4M3（偏置=7，
    //    无无穷大，NaN=0x7F）。
    float val   = static_cast<float>(src[idx]);
    float scale = __half2float(scales[scale_idx]);
    dst[idx]    = __float2half(val * scale);
}

// ============================================================================
// launch_dequantize_fp8_to_fp16
//
// Host-side launcher for dequant_fp8e4m3_to_fp16_kernel.
// Reshapes the dequant workspace View to match weight dimensions before use.
//
// 主机端启动器，为 dequant_fp8e4m3_to_fp16_kernel 提供调度。
// 在使用前将反量化工作区 View 的形状调整为与权重维度匹配。
// ============================================================================
void launch_dequantize_fp8_to_fp16(
    const View& weight,      // FP8_E4M3 [rows, cols]           权重（FP8）
    const View& scales,      // FP16 [rows, cols/group_size]     逐组 scale
    View&       dst,         // FP16 [rows, cols]                FP16 输出工作区
    int         group_size,  // elements per scale group         每 scale 组的元素数
    cudaStream_t stream)
{
    int rows  = weight.dims[0];
    int cols  = weight.dims[1];
    int total = rows * cols;

    constexpr int BLOCK = 256;
    int grid = (total + BLOCK - 1) / BLOCK;

    dequant_fp8e4m3_to_fp16_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_fp8_e4m3*>(weight.data_ptr),
        static_cast<const __half*>(scales.data_ptr),
        static_cast<__half*>(dst.data_ptr),
        rows, cols, group_size);
    CUDA_CHECK_LAST();
}

#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "utils/utils.h"
#include "core/view.h"
#include "ops/kernel.h"

// [EN] Optimized Attention Kernel utilizing Shared Memory for data reuse and coalesced loads.
// [CN] 利用共享内存实现数据复用和合并加载的优化注意力核函数。
// [Bug/Imperfection: Still does not fully implement FlashAttention's tiling over the sequence dimension, risking Smem overflow for very long sequences. 仍未完全实现FlashAttention在序列维度上的分块，对于超长序列存在Smem溢出的风险。]
template <typename T>
__global__ void tiled_attention_kernel(
    const T* q_base, const T* k_base, const T* v_base, 
    T* out, 
    int seq_len, int head_dim, 
    int qkv_stride) 
{
    // [EN] Cooperative fetching: A Block of threads loads a Tile of K into Shared Memory.
    // [CN] 协作预取：一个线程块将K的一个分块加载到共享内存中。
    // [Bug/Imperfection: Hardcoded tile size (TILE_SIZE). If sequence length is not a multiple of TILE_SIZE, boundary checks are required but missing here. 硬编码的分块大小(TILE_SIZE)。如果序列长度不是TILE_SIZE的倍数，需要边界检查，但此处缺失。]
    constexpr int TILE_SIZE = 128;
    __shared__ T k_smem[TILE_SIZE][128]; // 假设 head_dim = 128

    // 此时，行映射逻辑改变。假设一个 Warp (32 线程) 负责处理 1 个 Token。
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int row = blockIdx.x * (blockDim.x / 32) + warp_id; 

    if (row >= seq_len) return;

    // ... (在外部循环中遍历 K 的所有 Tile) ...
    for(int t_idx = 0; t_idx < seq_len; t_idx += TILE_SIZE) {
        
        // 1. 全体线程协作，把 TILE_SIZE 个 K 从 HBM 搬到 k_smem (实现合并访存)
        // 每个线程搬运几个元素，而不是一个人搬运全部
        // ... (搬运代码) ...
        
        __syncthreads(); // 物理屏障：等大家把这块 K 都搬完

        // 2. 数据复用：现在所有的 Warp 都可以去读极速的 k_smem 了
        // 而且每个 Warp 内的 32 个线程，可以并行去算 Q * K^T 的点积（每个人算 4 个维度的乘加）
        // ... (Warp 级规约求和) ...

        __syncthreads(); // 等大家都算完，准备搬下一块 K
    }
}

// Explicit instantiation for half precision
// half 精度的显式实例化
template __global__ void tiled_attention_kernel<half>(
    const half*, const half*, const half*, half*, int, int, int);

// Launcher: wraps the grid/block configuration
// 启动器：封装网格/线程块配置
// [Bug/Imperfection: Grid maps one block per query token (blockIdx.x = token row).
//  For seq_len=1 (decode) this launches only 1 block, leaving most SMs idle.
//  In production, launch 1 warp per head using a 2D grid (token × head).
//  网格将每个 block 映射到一个 query token。seq_len=1 时只启动 1 个 block，
//  大多数 SM 空闲。生产环境应使用 2D 网格（token × head）每 head 启动一个 warp。]
template <typename T>
void launch_tiled_attention_kernel(
    const T* q_base, const T* k_base, const T* v_base,
    T* out,
    int seq_len, int head_dim, int qkv_stride,
    cudaStream_t stream)
{
    constexpr int BLOCK = 128;
    int grid = (seq_len + (BLOCK / 32) - 1) / (BLOCK / 32);
    tiled_attention_kernel<T><<<grid, BLOCK, 0, stream>>>(
        q_base, k_base, v_base, out, seq_len, head_dim, qkv_stride);
    CUDA_CHECK_LAST();
}

// Explicit instantiation of launcher
template void launch_tiled_attention_kernel<half>(
    const half*, const half*, const half*, half*, int, int, int, cudaStream_t);
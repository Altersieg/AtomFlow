#include <cuda_fp16.h>

// [EN] Naive Element-wise vector addition for Residual Connection.
// [CN] 用于残差连接的朴素逐元素向量加法。
// [Bug/Imperfection: scalar memory accesses (half) severely underutilize the 256-byte memory transaction window. Memory bandwidth efficiency is less than 50%.
// 标量内存访问 (half) 严重未充分利用 256 字节的内存事务窗口。内存带宽效率低于 50%。]
__global__ void residual_add_kernel_naive(const half* x_original, const half* x_sublayer_out, half* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // 物理动作：发两次独立的 16-bit 读请求，发一次 16-bit 写请求
    float orig = __half2float(x_original[idx]);
    float sub  = __half2float(x_sublayer_out[idx]);
    
    out[idx] = __float2half(orig + sub);
}
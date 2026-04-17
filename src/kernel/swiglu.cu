#include <math.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "view.h"
#include "kernel.h"

// [EN] Element-wise SwiGLU activation: out = (gate * sigmoid(gate)) * up
// [CN] 逐元素 SwiGLU 激活：out = (gate * sigmoid(gate)) * up
// [Bug/Imperfection: Using float for internal compute is safe, but memory bandwidth is wasted if not loading as float4.
// 内部计算使用 float 是安全的，但如果不作为 float4 加载，内存带宽会被浪费。]
__global__ void swiglu_kernel(const half* gate, const half* up, half* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float g = __half2float(gate[idx]);
    float u = __half2float(up[idx]);
    
    // [EN] SiLU computation. __expf is fast but loses slight precision.
    // [CN] SiLU 计算。__expf 很快，但会丢失轻微的精度。
    // [Bug/Imperfection: The sigmoid function here doesn't account for extreme overflow/underflow, though rare in normalized networks.
    // 此处的 sigmoid 函数没有考虑极端的上溢/下溢，尽管在归一化网络中这很罕见。]
    float silu = g / (1.0f + __expf(-g));
    
    out[idx] = __float2half(silu * u);
}
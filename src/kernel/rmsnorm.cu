#include <cuda_fp16.h>
#include <cuda_fp8.h> 
#include <cub/block/block_reduce.cuh>
#include <cublas_v2.h>
#include "view.h"
#include "kernel.h"

template <typename T, int BLOCK_DIM>
__global__ void rms_norm_kernel(const T* in, const T* wei, T* out, int cols, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const T* row_in = in + row * cols;
    T* row_out = out + row * cols;

    float sum_sq = 0.0f;

    // --- 编译期分支：计算平方和 ---
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
        // FP8 path: __nv_fp8_e4m3 exposes an implicit conversion operator to float.
        // (CUDA 12 no longer ships __fp8_to_half_raw / __half_to_fp8_rn.)
        for (int i = tid; i < cols; i += blockDim.x) {
            float val = static_cast<float>(row_in[i]);
            sum_sq += val * val;
        }
    } else {
        // FP16 的标准路径
        for (int i = tid; i < cols; i += blockDim.x) {
            float val = __half2float(row_in[i]);
            sum_sq += val * val;
        }
    }

    // --- CUB Reduce 核心逻辑 ---
    // 1. 定义 CUB 的 BlockReduce 类型
    using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
    
    // 2. 申请 CUB 所需的共享内存空间
    __shared__ typename BlockReduce::TempStorage temp_storage;

    // 3. 执行块内规约求和 (所有线程的 sum_sq 加在一起)
    float total_sum_sq = BlockReduce(temp_storage).Sum(sum_sq);

    // 4. 计算最终的方差 (只有 thread 0 会得到正确的 total_sum_sq)
    __shared__ float s_variance;
    if (tid == 0) {
        // rsqrtf 是快速平方根倒数硬件指令： 1 / sqrt(x)
        s_variance = rsqrtf(total_sum_sq / cols + eps);
    }
    
    // 5. 屏障同步：确保所有线程在进行下一步乘法前，都能读到算好的 s_variance
    __syncthreads();

    // --- 编译期分支：写回结果 ---
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
        // FP8 writeback: construct __nv_fp8_e4m3 directly from float.
        for (int i = tid; i < cols; i += blockDim.x) {
            float val = static_cast<float>(row_in[i]);
            float w   = static_cast<float>(wei[i]);
            row_out[i] = __nv_fp8_e4m3(val * s_variance * w);
        }
    } else {
        // FP16 路径
        for (int i = tid; i < cols; i += blockDim.x) {
            float val = __half2float(row_in[i]);
            row_out[i] = __float2half(val * s_variance * __half2float(wei[i]));
        }
    }
}

void launch_rms_norm(const View& input, const View& weight, View& output, float eps, cudaStream_t stream) {
    
    int32_t rows = calculate_rows(input);
    int32_t cols = input.dims[input.num_dims - 1];
    
    constexpr int THREAD_PER_BLOCK = 128;

    switch (input.dtype) {
        case DataType::FP16: {
            const half* in_ptr  = static_cast<const half*>(input.data_ptr);
            const half* wei_ptr = static_cast<const half*>(weight.data_ptr);
            half*       out_ptr = static_cast<half*>(output.data_ptr);
            rms_norm_kernel<half, THREAD_PER_BLOCK><<<rows, THREAD_PER_BLOCK, 0, stream>>>(
                in_ptr, wei_ptr, out_ptr, cols, eps);
            break;
        }
        case DataType::FP8_E4M3: {
            const __nv_fp8_e4m3* in_ptr  = static_cast<const __nv_fp8_e4m3*>(input.data_ptr);
            const __nv_fp8_e4m3* wei_ptr = static_cast<const __nv_fp8_e4m3*>(weight.data_ptr);
            __nv_fp8_e4m3*       out_ptr = static_cast<__nv_fp8_e4m3*>(output.data_ptr);
            rms_norm_kernel<__nv_fp8_e4m3, THREAD_PER_BLOCK><<<rows, THREAD_PER_BLOCK, 0, stream>>>(
                in_ptr, wei_ptr, out_ptr, cols, eps);
            break;
        }
        default:
            throw std::runtime_error("Unsupported data type");
    }
}
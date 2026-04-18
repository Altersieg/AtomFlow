#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "utils/utils.h"
#include "core/view.h"
#include "ops/kernel.h"

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

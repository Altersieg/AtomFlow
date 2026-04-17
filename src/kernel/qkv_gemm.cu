

void lauch_qkv_gemm(
    const View& input,    // [Total_Tokens, Hidden_Dim]
    const View& weight,   // [QKV_Total_Dim, Hidden_Dim] -> 预先拼接好的权重
    View& output,         // [Total_Tokens, QKV_Total_Dim]
    cublasHandle_t handle
){
    // 1. 物理参数提取 (cuBLAS 默认是列优先，我们按行优先传参需要反过来想)
    int m = calculate_rows(input);       // Total Tokens
    int k = input.dims[input.num_dims-1]; // Hidden Dim (4096)
    int n = weight.dims[0];               // QKV Total Dim (4096 + 1024 + 1024)

    // 2. 这里的“暴力”体现在 cublasGemmEx
    // 它支持混合精度（FP8 权重, FP16 激活, FP32 计算）
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasGemmEx(handle, 
        CUBLAS_OP_N, CUBLAS_OP_T, // 重点：这里用 OP_T 让 cuBLAS 在读取权重时自动转置
        n, m, k, 
        &alpha, 
        weight.data_ptr(), CUDA_R_8F_E4M3, n,  // 权重：FP8
        input.data_ptr(),  CUDA_R_16F, k,      // 输入：FP16
        &beta, 
        output.data_ptr(), CUDA_R_16F, n,      // 输出：FP16
        CUBA_COMPUTE_32F,                      // 内部用 FP32 累加保证精度
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);        // 强制调用 Tensor Core
} 
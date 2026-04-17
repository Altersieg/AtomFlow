#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <iostream>

// [EN] Naive forward pass for a single Transformer layer. Pure linear execution, no abstractions.
// [CN] 单个 Transformer 层的朴素前向传播。纯线性执行，没有抽象。
// [Bug/Imperfection: Passing a dozen raw pointers is extremely error-prone and destroys cache locality if not managed properly. 
// 传递十几个裸指针极易出错，如果管理不当，会破坏缓存局部性。]
void layer_forward_naive(
    cublasHandle_t handle,
    int seq_len, int hidden_dim,
    half* d_X,              // Input/Output Tensor: [seq_len, hidden_dim]
    half* d_norm_weight,    // Weights ...
    half* d_qkv_weight,
    half* d_o_weight,
    // --- 巨大的中间临时显存 (今晚不管浪费，先跑通) ---
    half* d_norm_out,       
    half* d_qkv_out,
    half* d_attn_out
) {
    // 1. RMSNorm (你的自定义核函数)
    // [EN] Launch RMSNorm kernel. Block size is fixed for MVP.
    // [CN] 启动 RMSNorm 核函数。MVP 阶段线程块大小固定。
    // [Bug/Imperfection: Hardcoded grid/block dimensions will crash or underutilize the GPU if hidden_dim changes. 
    // 如果 hidden_dim 改变，硬编码的网格/线程块维度将导致崩溃或 GPU 利用率不足。]
    int threads = 256;
    int blocks = seq_len;
    rmsnorm_kernel<<<blocks, threads>>>(d_X, d_norm_weight, d_norm_out, hidden_dim);

    // 2. QKV 投影 (直接用 cuBLAS 砸)
    // [EN] cuBLAS HGEMM for QKV projection. Note that cuBLAS assumes Column-Major order!
    // [CN] 用于 QKV 投影的 cuBLAS HGEMM。注意 cuBLAS 假设是以列为主序的！
    // [Bug/Imperfection: Calling cuBLAS HGEMM creates massive synchronous overhead compared to fused kernels. Transpose logic might be wrong if not careful with cublasOperation_t.
    // 与融合核函数相比，调用 cuBLAS HGEMM 会产生巨大的同步开销。如果不小心处理 cublasOperation_t，转置逻辑可能会错。]
    const half alpha = __float2half(1.0f);
    const half beta = __float2half(0.0f);
    cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                /*m*/ 5120, /*n*/ seq_len, /*k*/ hidden_dim,
                &alpha, 
                d_qkv_weight, hidden_dim, // 假设权重已在 CPU 预先转置
                d_norm_out, hidden_dim,
                &beta, 
                d_qkv_out, 5120);

    // 3. RoPE (自定义核函数，直接原地修改 d_qkv_out)
    rope_kernel<<<...>>>(d_qkv_out, seq_len);

    // 4. Attention (你刚才写的 Tiled Attention)
    tiled_attention_kernel<<<...>>>(d_qkv_out, d_attn_out, seq_len);

    // 5. O_proj (再次调用 cuBLAS)
    cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, /*...*/, d_attn_out, d_X_new);

    // 6. 第一次残差相加 (将结果直接加回原数组 d_X，实现状态更新)
    residual_add_kernel_naive<<<...>>>(d_X, d_X_new, d_X, seq_len * hidden_dim);

    // ... FFN 的逻辑同理，直接往下复制粘贴 ...
}
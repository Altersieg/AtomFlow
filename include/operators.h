
#pragma once

#include "view.h"

int32_t calculate_rows(const View& view) {
    if (view.num_dims < 2) return 1;
    int32_t rows = 1;
    for (int i = 0; i < view.num_dims - 1; ++i) { 
        rows *= view.dims[i]; //row = B * S (Batch Size * Sequence Length)
    }
    return rows;
}

void launch_rms_norm(
    const View& input,   // [Batch, Seq, Hidden]
    const View& weight,  // [Hidden]
    View& output,        // [Batch, Seq, Hidden]
    float eps,           
    cudaStream_t stream  
); 

void launch_rope(
    const View& q_input,   // [1, Seq, h * q_num]
    const View& k_input,  // [1, Seq, h * kv_num]
    const float* cos_cache,
    const float* sin_cache,
    int seq_len, // later on : flattern rope for mutiple users.
    int kv_head_num,
    int q_head_num,
    int head_dim,
    cudaStream_t stream  
); 

void lauch_qkv_gemm(
    const View& input,    // [Total_Tokens, Hidden_Dim]
    const View& weight,   // [QKV_Total_Dim, Hidden_Dim] -> 预先拼接好的权重
    View& output,         // [Total_Tokens, QKV_Total_Dim]
    cublasHandle_t handle
);

#pragma once

#include "view.h"

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
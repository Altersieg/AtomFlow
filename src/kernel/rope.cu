#include "operators.h"

template<T>
__global__ void rope_kernel( T* q_ptr,   // [1, Seq, h * q_num]
    T* k_ptr,  // [1, Seq, h * kv_num]
    const float* cos_cache,
    const float* sin_cache,
    int seq_len, // later on : flattern rope for mutiple users.
    int kv_head_num,
    int q_head_num,
    int head_dim,
){
    int pos = blockIdx.x;// token num
    int d = threadIdx.x;// pair num
    int cache_id = pos * ( head_dim / 2 ) + d;
    float cos_val = cos_cache[cache_id];    
    float sin_val = sin_cache[cache_id];

    for(int h = 0; h < q_head_num; h++) {
        T* q_current_ptr = q_ptr + pos * q_head_num * head_dim + h * head_dim;
        //if FP16 (half) or FP8，cast to float for high accuracy calculation whith cos_val 
        float x = (float)q_current_ptr[d];
        float y = (float)q_current_ptr[d + head_dim / 2];

        q_current_ptr[d] = (x * cos_val - y * sin_val);
        q_current_ptr[d + head_dim / 2] = (x * sin_val + y * cos_val);
    }

        for(int h = 0; h < kv_head_num; h++) {
        T* k_current_ptr = k_ptr + pos * q_head_num * head_dim + h * head_dim;
        //if FP16 (half) or FP8，cast to float for high accuracy calculation whith cos_val 
        float x = (float)k_current_ptr[d];
        float y = (float)k_current_ptr[d + head_dim / 2];

        q_current_ptr[k] = (x * cos_val - y * sin_val);
        q_current_ptr[k + head_dim / 2] = (x * sin_val + y * cos_val);
    }
}


void launch_rope(    
    const View& q_input,  
    const View& k_input,  
    const float* cos_cache,
    const float* sin_cache,
    int seq_len, 
    int kv_head_num,
    int q_head_num,
    int head_dim,
    cudaStream_t stream) 
{
    int thread_per_block = head_dim / 2; 
    int blocks_per_grid = seq_len; 

    switch (q_input.dtype) {
        case DataType::FP16: {
            half* q_ptr = static_cast<half*>(q_input.data_ptr()); 
            half* k_ptr = static_cast<half*>(k_input.data_ptr());
            rope_kernel<half><<<blocks_per_grid, thread_per_block, 0, stream>>>(
                q_ptr, k_ptr, cos_cache, sin_cache, seq_len, kv_head_num, q_head_num, head_dim);
            break;
        }
        case DataType::FP8_E4M3: {
            __nv_fp8_e4m3* q_ptr = static_cast<__nv_fp8_e4m3*>(q_input.data_ptr());
            __nv_fp8_e4m3* k_ptr = static_cast<__nv_fp8_e4m3*>(k_input.data_ptr());
            rope_kernel<__nv_fp8_e4m3><<<blocks_per_grid, thread_per_block, 0, stream>>>(
                q_ptr, k_ptr, cos_cache, sin_cache, seq_len, kv_head_num, q_head_num, head_dim); 
            break;
        }
        default:
            throw std::runtime_error("Unsupported data type in RoPE");
    }
}
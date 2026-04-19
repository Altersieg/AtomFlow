#pragma once

#include "core/view.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>

// [EN] Helper to calculate the flat row dimension for batched sequence processing.
// [CN] 辅助函数：计算用于批处理序列的展平的行维度。
// [Bug/Imperfection: If num_dims is deeply nested, the loop is fine, but it does not account for memory alignment or stride gaps in non-contiguous Views.
// 如果 num_dims 嵌套很深，循环没问题，但它没有考虑非连续 View 中的内存对齐或跨步间隙。]
inline int32_t calculate_rows(const View& view) { 
    if (view.num_dims < 2) return 1;
    int32_t rows = 1;
    for (int i = 0; i < view.num_dims - 1; ++i) { 
        rows *= view.dims[i]; 
    }
    return rows;
}

// ----------------------------------------------------------------------------
// 1. 归一化与位置编码 (Normalization & RoPE)
// ----------------------------------------------------------------------------

void launch_rms_norm(
    const View& input,   
    const View& weight,  
    View& output,        
    float eps,           
    cudaStream_t stream  
); 

void launch_rope(
    const View& q_input,  
    const View& k_input,  
    const float* cos_cache,
    const float* sin_cache,
    int seq_len, 
    int kv_head_num,
    int q_head_num,
    int head_dim,
    cudaStream_t stream  
); 

// ----------------------------------------------------------------------------
// 2. 统一矩阵乘法 (The Universal GEMM / Linear Layer)
// 覆盖: QKV_proj, O_proj, Gate_proj, Up_proj, Down_proj, lm_head
// ----------------------------------------------------------------------------

// [EN] Universal launcher for all Linear layers. Calls cuBLAS underneath.
// [CN] 适用于所有线性层的通用启动器。底层调用 cuBLAS。
// [Bug/Imperfection: Missing cublasOperation_t parameters. Hardcoding Transpose/No-Transpose logic inside the implementation assumes all weights are stored in the same layout, which may not match the .safetensors format.
// 缺失 cublasOperation_t 参数。在实现内部硬编码转置/非转置逻辑假设所有权重都以相同的布局存储，这可能与 .safetensors 格式不匹配。]
void launch_linear_gemm(
    const View& input,    // [Total_Tokens, In_Features]
    const View& weight,   // [Out_Features, In_Features]
    View& output,         // [Total_Tokens, Out_Features]
    cublasHandle_t handle,
    cudaStream_t stream
);

// ----------------------------------------------------------------------------
// 3. 注意力机制 (Attention)
// ----------------------------------------------------------------------------

// [EN] Tiled Attention Kernel launcher. Added template declaration for C++ compliance.
// [CN] Tiled Attention 核函数启动器。添加了模板声明以符合 C++ 规范。
// [Bug/Imperfection: Exposing raw pointers here breaks the View abstraction layer. It tightly couples the caller to physical memory addresses instead of logical tensors.
// 在这里暴露裸指针破坏了 View 抽象层。它将调用者与物理内存地址紧密耦合，而不是逻辑张量。]
// [EN] GQA attention kernel. For seq_kv=1 (decode): softmax of one score = 1.0,
//      so out[h] = V[kv_head] (trivially exact). General seq_kv > 1 not yet
//      implemented (outputs zero); extend when KV-cache is added.
// [CN] GQA 注意力核函数。seq_kv=1（解码）时：单值 softmax = 1.0，
//      故 out[h] = V[kv_head]（精确无误）。seq_kv > 1 通用路径尚未实现
//      （输出零）；添加 KV 缓存后再扩展。
template <typename T>
void launch_tiled_attention_kernel(
    const T* q_base,       // [seq_q, n_q_heads  * head_dim]
    const T* k_base,       // [seq_kv, n_kv_heads * head_dim]
    const T* v_base,       // [seq_kv, n_kv_heads * head_dim]
    T* out,                // [seq_q, n_q_heads  * head_dim]
    int seq_q,             // number of query tokens (= 1 for decode)
    int seq_kv,            // number of key-value tokens (= seq_q for self-attn)
    int head_dim,          // head dimension (= 128 for Llama 3.2 3B)
    int n_q_heads,         // number of query heads (= 24)
    int n_kv_heads,        // number of key-value heads for GQA (= 8)
    cudaStream_t stream
);

// ----------------------------------------------------------------------------
// 4. 残差与激活函数 (Residuals & Activations)
// ----------------------------------------------------------------------------

// [EN] Element-wise SwiGLU: out = (gate * sigmoid(gate)) * up
// [CN] 逐元素 SwiGLU 激活：out = (gate * sigmoid(gate)) * up
// [Bug/Imperfection: Requires 3 memory round-trips. Fusing this with the preceding Gate and Up GEMM epilogues would save massive memory bandwidth.
// 需要 3 次内存往返。将其与之前的 Gate 和 Up GEMM 收尾操作(epilogue)融合将节省大量的内存带宽。]
void launch_swiglu(
    const View& gate_input,
    const View& up_input,
    View& output,
    cudaStream_t stream
);

// [EN] In-place Residual Addition: inout_tensor += sublayer_tensor
// [CN] 原地残差相加：inout_tensor += sublayer_tensor
// [Bug/Imperfection: Operating entirely in-place makes debugging difficult, as the original state is irreversibly destroyed after execution.
// 完全在原地操作会使调试变得困难，因为执行后原始状态被不可逆转地破坏了。]
void launch_residual_add(
    View& inout_tensor,       // 作为输入，同时也是输出，直接覆写
    const View& sublayer_tensor, 
    cudaStream_t stream
);

// ----------------------------------------------------------------------------
// 4b. FP8 Dequantization + FP16×FP16 GEMM
// ----------------------------------------------------------------------------

// [EN] Dequantize a packed FP8-E4M3 weight tensor to FP16 using per-group
//      FP16 scales (group_size elements share one scale).
// [CN] 使用逐组 FP16 scale（group_size 个元素共享一个 scale）
//      将 FP8-E4M3 权重张量反量化为 FP16。
void launch_dequantize_fp8_to_fp16(
    const View& weight,      // FP8_E4M3 [rows, cols]
    const View& scales,      // FP16 [rows, cols/group_size]
    View&       dst,         // FP16 [rows, cols]  pre-allocated workspace
    int         group_size,
    cudaStream_t stream
);

// [EN] Fused W8A16 GEMV — dequantize FP8 weights and compute dot-product in one
//      kernel, with ZERO intermediate global-memory writes. For M=1 decode only.
// [CN] 融合 W8A16 GEMV — 在单个 kernel 内完成 FP8 反量化和内积计算，
//      零中间全局显存写回。仅用于 M=1 解码阶段。
void launch_w8a16_gemv(
    const View& act_x,       // FP16 [1, K]
    const View& weight,      // FP8  [N, K]
    const View& scales,      // FP16 [N, K/GS]
    View&       output,      // FP16 [1, N]
    int         group_size,  // 128
    cudaStream_t stream
);

// [EN] FP16×FP16 → FP16 GEMM with FP32 accumulation.
//      Same matrix layout as launch_linear_gemm; use after dequantizing weights.
// [CN] FP16×FP16 → FP16 GEMM（FP32 累加）。
//      矩阵布局与 launch_linear_gemm 相同；在反量化权重后使用。
void launch_fp16x16_gemm(
    const View& input,       // [Total_Tokens, In_Features]
    const View& weight,      // [Out_Features, In_Features]  FP16 (dequantized)
    View& output,            // [Total_Tokens, Out_Features]
    cublasHandle_t handle,
    cudaStream_t stream
);

// ----------------------------------------------------------------------------
// 5. 采样机制 (Sampling)
// ----------------------------------------------------------------------------

// [EN] ArgMax for final logits to find the next token ID.
// [CN] 对最终 logits 执行 ArgMax 以寻找下一个 token ID。
// [Bug/Imperfection: Greedy search only. Lacks parameters for Top-K, Top-P, or temperature scaling, strictly limiting the model's generation diversity.
// 仅支持贪婪搜索。缺乏 Top-K、Top-P 或温度缩放的参数，严格限制了模型生成的连贯多样性。]
void launch_argmax(
    const View& logits, // [1, vocab_size]
    int* out_token_id,  // Device pointer for the single integer result
    cudaStream_t stream
);
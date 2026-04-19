#pragma once
// ============================================================================
// model_state.h — GPU-resident data structures for model weights & activations
// model_state.h — GPU 端模型权重与激活的数据结构
//
// Extracted from main.cu to decouple data layout from execution logic.
// 从 main.cu 中提取，使数据布局与执行逻辑解耦。
// ============================================================================

#include "core/view.h"

// ============================================================================
// Per-layer GPU weight Views (all pointers into the weight memory pool)
// 每层 GPU 权重 View（所有指针均指向权重内存池内部）
// ============================================================================
struct LayerWeights {
    View input_norm;      // FP32 [D]
    View post_norm;       // FP32 [D]
    View qkv;             // FP8  [QKV_OUT, D]
    View qkv_scales;      // FP16 [QKV_OUT, D/GS]  per-group dequant scales
    View o_proj;          // FP8  [D, D]
    View o_proj_scales;   // FP16 [D, D/GS]
    View gate_proj;       // FP8  [FFN, D]
    View gate_scales;     // FP16 [FFN, D/GS]
    View up_proj;         // FP8  [FFN, D]
    View up_scales;       // FP16 [FFN, D/GS]
    View down_proj;       // FP8  [D, FFN]
    View down_scales;     // FP16 [D, FFN/GS]
};

// ============================================================================
// GPU activation buffer layout (single arena, NO residual copy buffer)
// GPU 激活缓冲区布局（单一 arena，无残差拷贝缓冲区）
//
// Residual strategy: x IS the running residual throughout all 28 layers.
//   rms_norm(x, w, x_norm) reads x → writes x_norm, x remains untouched.
//   launch_residual_add(x, sublayer_out) does x += sublayer_out in-place.
//   Zero cudaMemcpy calls inside the layer loop.
//
// 残差策略：x 贯穿 28 层，始终作为流动残差。
//   rms_norm 读 x → 写 x_norm，x 保持不变。
//   launch_residual_add(x, sublayer_out) 原地执行 x += sublayer_out。
//   层循环内部零 cudaMemcpy 调用。
// ============================================================================
struct ActBuffers {
    View x;             // FP16 [1, D]        hidden state AND running residual base
    View x_norm;        // FP16 [1, D]        rms_norm scratch output
    View qkv_out;       // FP16 [1, QKV_OUT]
    View attn_out;      // FP16 [1, D]        attention output before o_proj
    View gate_out;      // FP16 [1, FFN]      gate_proj; reused for swiglu output
    View up_out;        // FP16 [1, FFN]      up_proj
    View ffn_out;       // FP16 [1, D]        o_proj output; reused for down_proj output
    View logits;        // FP32 [1, V]
    View dequant_ws;    // FP16 [FFN, D]      workspace for FP8→FP16 dequantized weights
    int*   d_token_id;  // INT32 [1]
};

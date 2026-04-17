#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <iostream>
#include "view.h"
#include "kernel.h"

// [EN] Stub for a single-layer Transformer forward pass.
//      The body is currently disabled because the launch_* helpers are still
//      being wired up. Once every kernel launcher in kernel.h has a definition
//      this stub will be filled in.
// [CN] 单层 Transformer 前向传播的占位实现。
//      当前函数体被关闭，因为 kernel.h 中的 launch_* 辅助函数还在对接。
//      等所有 launcher 定义齐整后，再把这里补上。
// [Bug/Imperfection: Using a stub hides the scheduling complexity until the
//      rest of the kernel surface is stable. No cross-layer fusion is attempted.
// 使用 stub 在其他 kernel 接口稳定之前隐藏了调度复杂度，也没有做跨层融合。]
void layer_forward_naive(
    cublasHandle_t /*handle*/,
    int /*seq_len*/, int /*hidden_dim*/,
    View& /*x_inout*/,           // [seq_len, hidden_dim]
    const View& /*norm_weight*/,
    const View& /*qkv_weight*/,
    const View& /*o_weight*/,
    View& /*norm_out*/,
    View& /*qkv_out*/,
    View& /*attn_out*/,
    cudaStream_t /*stream*/)
{
    // TODO: wire the full pipeline once launch_rms_norm, launch_linear_gemm,
    //       launch_rope, launch_tiled_attention_kernel, launch_swiglu and
    //       launch_residual_add are all compiling and tested in isolation.
}

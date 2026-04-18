#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include "utils/utils.h"
#include "core/view.h"
#include "ops/kernel.h"
#include "memory/weight_loader.h"
#include "utils/profiler.h"

static constexpr const char* DEFAULT_WEIGHTS = "models/llama3_2_atomflow.bin";

// ============================================================================
// Small device-side helpers that avoid CPU↔GPU round-trips
// 避免 CPU↔GPU 往返的小型设备端辅助 kernel
// ============================================================================

// Embed lookup: read row `token_id` from FP32 embed table, write FP16 to dst.
// 嵌入查找：从 FP32 嵌入表中读取 token_id 对应行，写入 FP16 dst。
static __global__ void k_embed_lookup(const float* table, int token_id,
                                      half* dst, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D) dst[i] = __float2half(table[token_id * D + i]);
}

// FP16 → FP32 cast for lm_head input (needs float path into cublasSgemm).
// FP16 转 FP32，供 lm_head 使用（cublasSgemm 需要 float 输入）。
static __global__ void k_cast_fp16_to_fp32(const half* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

// ============================================================================
// Per-layer GPU weight Views (all pointers into d_weight_pool)
// 每层 GPU 权重 View（所有指针均指向 d_weight_pool 内部）
// ============================================================================
struct LayerWeights {
    View input_norm;    // FP32 [D]
    View post_norm;     // FP32 [D]
    View qkv;           // FP8  [QKV_OUT, D]
    View o_proj;        // FP8  [D, D]
    View gate_proj;     // FP8  [FFN, D]
    View up_proj;       // FP8  [FFN, D]
    View down_proj;     // FP8  [D, FFN]
    // [Bug/Imperfection: Per-group FP16 scales are loaded from file but NOT
    //  passed to launch_linear_gemm. The cuBLAS call uses alpha=1.0, meaning
    //  all FP8 values are treated as having scale=1. Quantization error will
    //  be severe without proper dequantization in the GEMM epilogue.
    //  每组 FP16 Scale 已从文件加载但未传入 launch_linear_gemm。
    //  cuBLAS 调用使用 alpha=1.0，意味着所有 FP8 值均被视为 scale=1。
    //  若不在 GEMM epilogue 中做正确的反量化，量化误差将极为严重。]
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
    View x;           // FP16 [1, D]   hidden state AND running residual base
    View x_norm;      // FP16 [1, D]   rms_norm scratch output
    View qkv_out;     // FP16 [1, QKV_OUT]
    View attn_out;    // FP16 [1, D]   attention output before o_proj
    View gate_out;    // FP16 [1, FFN] gate_proj; reused for swiglu output
    View up_out;      // FP16 [1, FFN] up_proj
    View ffn_out;     // FP16 [1, D]   o_proj output; reused for down_proj output
    View logits;      // FP32 [1, V]
    float* x_norm_fp32; // FP32 [D]    cast of x_norm before FP32 lm_head GEMM
    int*   d_token_id;  // INT32 [1]
};

// ============================================================================
// RoPE precomputation (CPU → GPU)
// RoPE 预计算（CPU → GPU）
// ============================================================================
static void build_rope_cache(float* d_cos, float* d_sin,
                              int max_seq, int head_dim, float base,
                              cudaStream_t stream) {
    // cos[s, i] = cos(s / base^(2i/head_dim))
    // RoPE reference: Su et al. (2021) "RoFormer"
    std::vector<float> h_cos(max_seq * head_dim);
    std::vector<float> h_sin(max_seq * head_dim);
    for (int s = 0; s < max_seq; ++s) {
        for (int i = 0; i < head_dim / 2; ++i) {
            float theta = s / std::pow(base, 2.0f * i / head_dim);
            h_cos[s * head_dim + 2 * i]     = std::cos(theta);
            h_cos[s * head_dim + 2 * i + 1] = std::cos(theta);
            h_sin[s * head_dim + 2 * i]     = std::sin(theta);
            h_sin[s * head_dim + 2 * i + 1] = std::sin(theta);
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(d_cos, h_cos.data(),
                               max_seq * head_dim * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_sin, h_sin.data(),
                               max_seq * head_dim * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char* argv[]) {

    const std::string weights_path = (argc > 1) ? argv[1] : DEFAULT_WEIGHTS;

    // =========================================================================
    // 1. mmap weights + parse header
    //    内存映射权重文件并解析头部
    // =========================================================================
    WeightLoader loader(weights_path);
    const AtomHeader& hdr = loader.header;

    const int D       = hdr.dim;          // 3072
    const int FFN     = hdr.hidden_dim;   // 8192
    const int V       = hdr.vocab_size;   // 128256
    const int NL      = hdr.n_layers;     // 28
    const int GS      = hdr.group_size;   // 128
    const int NH      = hdr.n_heads;      // 24
    const int NKV     = hdr.n_kv_heads;   // 8
    const int HD      = D / NH;           // head_dim = 128
    const int Q_DIM   = D;                // 3072
    const int KV_DIM  = NKV * HD;         // 1024  (GQA)
    const int QKV_OUT = Q_DIM + KV_DIM + KV_DIM;  // 5120

    std::printf("──────────────────────────────────────────────\n");
    std::printf("AtomFlow  MVP Decode Step  |  %s\n", weights_path.c_str());
    std::printf("  D=%d  FFN=%d  NL=%d  NH=%d  NKV=%d  HD=%d\n",
                D, FFN, NL, NH, NKV, HD);
    std::printf("  QKV_OUT=%d  V=%d\n", QKV_OUT, V);
    std::printf("  file size: %.2f GiB\n",
                static_cast<double>(loader.file_size()) / (1 << 30));
    std::printf("──────────────────────────────────────────────\n");

    // =========================================================================
    // 2. Compute weight pool size and allocate GPU arena
    //    计算权重池大小并分配 GPU arena
    //
    // Layout (matches export_atomflow.py write order exactly):
    //   embed_tokens [V, D] FP32
    //   For each layer: norm×2(FP32), qkv/o/gate/up/down(FP8), scales(FP16 ignored)
    //   model.norm [D] FP32
    //   lm_head [V, D] FP32
    //
    // [Bug/Imperfection: FP16 scale tensors are consumed from the mmap cursor
    //  to keep the cursor in sync, but we do NOT copy them to the GPU pool.
    //  When per-group dequantization is implemented, these must be added to
    //  the pool and passed to the GEMM epilogue.
    //  为保持 mmap 游标同步，FP16 Scale 张量会从游标中消耗，但不复制到 GPU 池。
    //  实现每组反量化后，必须将其加入池并传给 GEMM epilogue。]
    // =========================================================================

    auto fp8_sz   = [](int r, int c) -> size_t { return (size_t)r * c; };
    auto scale_sz = [&](int r, int c) -> size_t {
        return (size_t)r * (c / GS) * sizeof(uint16_t);
    };
    auto fp32_sz  = [](int n) -> size_t { return (size_t)n * sizeof(float); };

    size_t pool_sz = 0;
    pool_sz += fp32_sz(V) * D;                          // embed_tokens
    for (int i = 0; i < NL; ++i) {
        pool_sz += fp32_sz(D) * 2;                      // 2 × RMSNorm
        pool_sz += fp8_sz(QKV_OUT, D);                  // qkv
        pool_sz += fp8_sz(D, D);                        // o_proj
        pool_sz += fp8_sz(FFN, D);                      // gate
        pool_sz += fp8_sz(FFN, D);                      // up
        pool_sz += fp8_sz(D, FFN);                      // down
    }
    pool_sz += fp32_sz(D);                              // model.norm
    pool_sz += fp32_sz(V) * D;                          // lm_head
    pool_sz += pool_sz / 16;                            // 6% headroom for alignment

    void* d_weight_pool = nullptr;
    CUDA_CHECK(cudaMalloc(&d_weight_pool, pool_sz));
    std::printf("[Weight pool]  GPU alloc %.2f GiB\n",
                (double)pool_sz / (1 << 30));

    // =========================================================================
    // 3. Copy each weight tensor from mmap → GPU pool, build Views
    //    从 mmap 逐张量复制到 GPU 池，同时构建 View
    //
    // Strategy: walk a device cursor (d_cur) in parallel with WeightLoader.
    // 策略：在 GPU 池中维护游标 d_cur，与 WeightLoader 保持同步推进。
    // =========================================================================
    uint8_t* d_cur = static_cast<uint8_t*>(d_weight_pool);

    // Helper lambdas to copy and build a View in one call.
    // 辅助 lambda：单次调用完成复制 + View 构建。
    auto copy_view = [&](const void* h_src, size_t bytes,
                         DataType dtype, std::initializer_list<int> dims) -> View {
        CUDA_CHECK(cudaMemcpy(d_cur, h_src, bytes, cudaMemcpyHostToDevice));
        View v = create_contiguous_view(d_cur, dtype, dims);
        d_cur += bytes;
        return v;
    };
    // Consume scale bytes from loader cursor without uploading to GPU pool.
    // 从加载器游标消耗 scale 字节但不上传到 GPU 池。
    auto skip_scales = [&](int rows, int cols) {
        loader.next<uint16_t>(scale_sz(rows, cols) / sizeof(uint16_t));
    };

    // ---- embed_tokens ----
    View embed_v = copy_view(
        loader.next<float>((size_t)V * D),
        fp32_sz(V) * D, DataType::FP32, {V, D});

    // ---- Per-layer weights ----
    std::vector<LayerWeights> lw(NL);
    for (int i = 0; i < NL; ++i) {
        lw[i].input_norm = copy_view(loader.next<float>(D), fp32_sz(D),
                                      DataType::FP32, {D});
        lw[i].post_norm  = copy_view(loader.next<float>(D), fp32_sz(D),
                                      DataType::FP32, {D});

        lw[i].qkv       = copy_view(loader.next<uint8_t>(fp8_sz(QKV_OUT, D)),
                                     fp8_sz(QKV_OUT, D), DataType::FP8_E4M3,
                                     {QKV_OUT, D});
        skip_scales(QKV_OUT, D);

        lw[i].o_proj    = copy_view(loader.next<uint8_t>(fp8_sz(D, D)),
                                     fp8_sz(D, D), DataType::FP8_E4M3, {D, D});
        skip_scales(D, D);

        lw[i].gate_proj = copy_view(loader.next<uint8_t>(fp8_sz(FFN, D)),
                                     fp8_sz(FFN, D), DataType::FP8_E4M3, {FFN, D});
        skip_scales(FFN, D);

        lw[i].up_proj   = copy_view(loader.next<uint8_t>(fp8_sz(FFN, D)),
                                     fp8_sz(FFN, D), DataType::FP8_E4M3, {FFN, D});
        skip_scales(FFN, D);

        lw[i].down_proj = copy_view(loader.next<uint8_t>(fp8_sz(D, FFN)),
                                     fp8_sz(D, FFN), DataType::FP8_E4M3, {D, FFN});
        skip_scales(D, FFN);

        if (i == 0)
            std::printf("[layer  0 weights loaded]  d_cur offset=%td B\n",
                        d_cur - (uint8_t*)d_weight_pool);
    }

    // ---- Final norm + lm_head ----
    View final_norm_v = copy_view(loader.next<float>(D),
                                   fp32_sz(D), DataType::FP32, {D});

    // [Bug/Imperfection: lm_head is stored as FP32 but launch_linear_gemm's
    //  cuBLAS call hardcodes CUDA_R_8F_E4M3 for the weight operand. The call
    //  will succeed but read FP32 bits as FP8, producing completely wrong logits.
    //  Fix: add a separate launch_linear_gemm_fp32 or convert lm_head to FP8.
    //  lm_head 以 FP32 存储，但 launch_linear_gemm 的 cuBLAS 调用硬编码了
    //  CUDA_R_8F_E4M3 权重类型。调用会成功，但 FP32 位会被解读为 FP8，
    //  产生完全错误的 logits。修复方案：添加单独的 FP32 GEMM launcher
    //  或将 lm_head 转换为 FP8。]
    View lm_head_v = copy_view(loader.next<float>((size_t)V * D),
                                fp32_sz(V) * D, DataType::FP32, {V, D});

    std::printf("[All weights uploaded]  pool used: %.2f GiB\n",
                (double)(d_cur - (uint8_t*)d_weight_pool) / (1 << 30));

    // =========================================================================
    // 4. Allocate activation arena
    //    分配激活 arena
    // =========================================================================
    const size_t SEQ = 1;  // single decode step / 单步解码
    size_t act_sz = 0;
    act_sz += SEQ * D       * sizeof(half)  * 3;  // x, x_norm, attn_out
    act_sz += SEQ * QKV_OUT * sizeof(half);        // qkv_out
    act_sz += SEQ * FFN     * sizeof(half)  * 2;  // gate_out, up_out
    act_sz += SEQ * D       * sizeof(half);        // ffn_out
    act_sz += SEQ * D       * sizeof(float);       // x_norm_fp32 (lm_head input)
    act_sz += SEQ * V       * sizeof(float);       // logits FP32
    act_sz += sizeof(int);                          // token_id
    act_sz += 2 * HD        * sizeof(float);        // rope cos/sin (seq_len=1)

    void* d_act_pool = nullptr;
    CUDA_CHECK(cudaMalloc(&d_act_pool, act_sz));
    CUDA_CHECK(cudaMemset(d_act_pool, 0, act_sz));

    uint8_t* a = static_cast<uint8_t*>(d_act_pool);
    ActBuffers act;

    auto mk = [&](size_t bytes, DataType dt, std::initializer_list<int> dims) -> View {
        View v = create_contiguous_view(a, dt, dims);
        a += bytes;
        return v;
    };

    act.x          = mk(SEQ*D      *sizeof(half),  DataType::FP16, {(int)SEQ, D});
    act.x_norm     = mk(SEQ*D      *sizeof(half),  DataType::FP16, {(int)SEQ, D});
    act.attn_out   = mk(SEQ*D      *sizeof(half),  DataType::FP16, {(int)SEQ, D});
    act.qkv_out    = mk(SEQ*QKV_OUT*sizeof(half),  DataType::FP16, {(int)SEQ, QKV_OUT});
    act.gate_out   = mk(SEQ*FFN    *sizeof(half),  DataType::FP16, {(int)SEQ, FFN});
    act.up_out     = mk(SEQ*FFN    *sizeof(half),  DataType::FP16, {(int)SEQ, FFN});
    act.ffn_out    = mk(SEQ*D      *sizeof(half),  DataType::FP16, {(int)SEQ, D});
    act.x_norm_fp32= reinterpret_cast<float*>(a); a += SEQ*D*sizeof(float);
    act.logits     = mk(SEQ*V      *sizeof(float), DataType::FP32, {(int)SEQ, V});
    act.d_token_id = reinterpret_cast<int*>(a);   a += sizeof(int);
    float* d_cos   = reinterpret_cast<float*>(a); a += HD * sizeof(float);
    float* d_sin   = reinterpret_cast<float*>(a); a += HD * sizeof(float);

    // =========================================================================
    // 5. CUDA context, profiler, RoPE cache
    //    CUDA 上下文、Profiler、RoPE 缓存
    // =========================================================================
    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUBLAS_CHECK(cublasSetStream(cublas, stream));

    EngineProfiler prof;

    // Precompute RoPE sin/cos for seq_pos=0 (decode step, single position)
    // 为 seq_pos=0 预计算 RoPE sin/cos（解码步骤，单个位置）
    build_rope_cache(d_cos, d_sin, /*max_seq=*/1, HD, /*base=*/500000.0f, stream);

    // =========================================================================
    // 6. Mock input: look up embedding for token id=1 (BOS)
    //    模拟输入：查询 token id=1 (BOS) 的 embedding
    //
    // [Bug/Imperfection: A real decoder would receive the token id from argmax
    //  of the previous step and index into embed_tokens on GPU. Here we copy
    //  one row of the CPU-side embedding as a placeholder.
    //  真实解码器应接收上一步 argmax 的 token id 并在 GPU 上索引 embed_tokens。
    //  此处仅复制 CPU 端 embedding 的一行作为占位符。]
    // =========================================================================
    prof.mark_prefill_start();
    {
        // k_embed_lookup: read embed_tokens[BOS=1] (FP32) → x (FP16), pure device.
        // k_embed_lookup：纯设备端读 embed_tokens[BOS=1] FP32 → 写 x FP16，零往返。
        constexpr int BOS_TOKEN_ID = 1;
        k_embed_lookup<<<(D + 255) / 256, 256, 0, stream>>>(
            static_cast<const float*>(embed_v.data_ptr),
            BOS_TOKEN_ID,
            static_cast<half*>(act.x.data_ptr),
            D);
    }
    prof.mark_first_token();   // TTFT boundary (mocked) / TTFT 边界（模拟）

    // =========================================================================
    // 7. Forward loop — 28 Transformer layers  (ZERO cudaMemcpy inside)
    //    前向循环 —— 28 个 Transformer 层（内部零 cudaMemcpy）
    //
    // Residual topology:
    //   act.x is the ONLY residual buffer for all 28 layers.
    //   rms_norm reads x → writes x_norm; x is untouched (serves as residual).
    //   Every sublayer computes its output into a SEPARATE scratch buffer.
    //   launch_residual_add(x, scratch) does x += scratch in-place.
    //   No save-and-restore, no ping-pong buffers, no cudaMemcpy.
    //
    // 残差拓扑：
    //   act.x 是所有 28 层唯一的残差缓冲区。
    //   rms_norm 读 x → 写 x_norm；x 保持不变（充当残差）。
    //   每个子层将输出写入独立的暂存缓冲区。
    //   launch_residual_add(x, scratch) 原地执行 x += scratch。
    //   无保存-恢复，无乒乓缓冲，无 cudaMemcpy。
    //
    // [Bug/Imperfection: NO KV Cache. K and V from previous positions are
    //  discarded each step. Attention over seq_len=1 produces the correct
    //  output ONLY for the first token; all subsequent positions see only
    //  a self-attention of length 1 and will generate incoherent tokens.
    //  无 KV 缓存。每步丢弃历史 K/V。seq_len=1 的注意力仅对第一个 token
    //  正确；后续位置只能看到长度为 1 的自注意力，生成不连贯的 token。]
    // =========================================================================
    prof.mark_decode_start(stream);  // TPOT clock starts / TPOT 计时开始

    for (int layer = 0; layer < NL; ++layer) {
        const LayerWeights& w = lw[layer];
        // Memory traffic constant for RMSNorm (read x, read weight, write x_norm)
        // RMSNorm 的显存流量常量（读 x，读 weight，写 x_norm）
        const size_t bw_norm = SEQ * D * sizeof(half) * 2 + SEQ * D * sizeof(float);

        // ── Step 1: RMSNorm (Attention)  x → x_norm  [x UNCHANGED = residual base]
        //    注意力 RMSNorm：x → x_norm（x 保持不变，充当残差基）
        {
            auto _t = prof.scoped_device("rms_norm_attn", stream);
            launch_rms_norm(act.x, w.input_norm, act.x_norm, 1e-5f, stream);
        }
        prof.annotate_bandwidth("rms_norm_attn", bw_norm);

        // ── Step 2: Fused QKV GEMM  x_norm [1,D] × qkv [QKV_OUT,D]ᵀ → qkv_out [1,QKV_OUT]
        //    融合 QKV GEMM
        {
            auto _t = prof.scoped_device("qkv_gemm", stream);
            launch_linear_gemm(act.x_norm, w.qkv, act.qkv_out, cublas, stream);
        }
        prof.annotate_bandwidth("qkv_gemm",
            SEQ * D * sizeof(half) + (size_t)QKV_OUT * D + SEQ * QKV_OUT * sizeof(half));

        // ── Step 3: QKV pointer split + RoPE
        //    QKV 指针拆分 + RoPE
        //
        // qkv_out is a flat [1, Q_DIM + KV_DIM + KV_DIM] buffer.
        // Q, K, V share the SAME underlying memory in a packed interleaved layout:
        //   q_ptr = qkv_out.data_ptr
        //   k_ptr = q_ptr + Q_DIM   (offset by Q_DIM half elements)
        //   v_ptr = k_ptr + KV_DIM  (offset by another KV_DIM half elements)
        // The Views have contiguous strides for seq_len=1, so this is exact.
        //
        // qkv_out 是展平的 [1, Q_DIM+KV_DIM+KV_DIM] 缓冲区。
        // Q/K/V 共享同一内存，紧密打包：
        //   q_ptr = qkv_out.data_ptr
        //   k_ptr = q_ptr + Q_DIM
        //   v_ptr = k_ptr + KV_DIM
        // seq_len=1 时 View 是连续的，此拆分是精确的。
        //
        // [Bug/Imperfection: For seq_len > 1, rows of Q/K/V are interleaved with
        //  stride=QKV_OUT, not stride=Q_DIM or KV_DIM. These contiguous Views will
        //  silently read garbage for multi-token prefill. Fix: use non-contiguous
        //  Views with stride[0]=QKV_OUT or copy Q/K/V to separate buffers.
        //  seq_len > 1 时 Q/K/V 各行间隔为 QKV_OUT，非 Q_DIM/KV_DIM。
        //  上方的连续 View 在多 token prefill 时会静默读取垃圾数据。
        //  修复：使用 stride[0]=QKV_OUT 的非连续 View 或将 Q/K/V 复制到独立缓冲区。]
        half* const qkv_base = static_cast<half*>(act.qkv_out.data_ptr);
        half* const q_ptr    = qkv_base;
        half* const k_ptr    = qkv_base + Q_DIM;
        half* const v_ptr    = qkv_base + Q_DIM + KV_DIM;
        View q_view = create_contiguous_view(q_ptr, DataType::FP16, {(int)SEQ, Q_DIM});
        View k_view = create_contiguous_view(k_ptr, DataType::FP16, {(int)SEQ, KV_DIM});
        {
            auto _t = prof.scoped_device("rope", stream);
            launch_rope(q_view, k_view, d_cos, d_sin, (int)SEQ, NKV, NH, HD, stream);
        }

        // ── Step 4: Tiled Attention  Q,K,V → attn_out [1,D]
        //    Tiled 注意力
        {
            auto _t = prof.scoped_device("tiled_attn", stream);
            launch_tiled_attention_kernel<half>(
                q_ptr, k_ptr, v_ptr,
                static_cast<half*>(act.attn_out.data_ptr),
                (int)SEQ, HD, QKV_OUT, stream);
        }

        // ── Step 5: O_proj  attn_out [1,D] × o_proj [D,D]ᵀ → ffn_out [1,D]
        //    O 投影（借用 ffn_out 作为临时缓冲区）
        {
            auto _t = prof.scoped_device("o_proj", stream);
            launch_linear_gemm(act.attn_out, w.o_proj, act.ffn_out, cublas, stream);
        }
        prof.annotate_bandwidth("o_proj",
            SEQ * D * sizeof(half) + (size_t)D * D + SEQ * D * sizeof(half));

        // ── Step 6: Residual Add 1  x += ffn_out  (IN-PLACE, no cudaMemcpy)
        //    残差相加 1：x += ffn_out（原地，无 cudaMemcpy）
        {
            auto _t = prof.scoped_device("res_add_attn", stream);
            launch_residual_add(act.x, act.ffn_out, stream);
        }
        // act.x now = x_before_layer + attn_sublayer_output
        // act.x 现在 = 层前 x + 注意力子层输出

        // ── Step 7: RMSNorm (MLP)  x → x_norm  [x is now post-attn residual base]
        //    MLP 前 RMSNorm：x → x_norm（x 现在是注意力后残差基）
        {
            auto _t = prof.scoped_device("rms_norm_mlp", stream);
            launch_rms_norm(act.x, w.post_norm, act.x_norm, 1e-5f, stream);
        }
        prof.annotate_bandwidth("rms_norm_mlp", bw_norm);

        // ── Step 8a: Gate_proj  x_norm [1,D] × gate [FFN,D]ᵀ → gate_out [1,FFN]
        //    Gate 投影
        {
            auto _t = prof.scoped_device("gate_proj", stream);
            launch_linear_gemm(act.x_norm, w.gate_proj, act.gate_out, cublas, stream);
        }
        prof.annotate_bandwidth("gate_proj",
            SEQ * D * sizeof(half) + (size_t)FFN * D + SEQ * FFN * sizeof(half));

        // ── Step 8b: Up_proj  x_norm [1,D] × up [FFN,D]ᵀ → up_out [1,FFN]
        //    Up 投影
        //    [Bug/Imperfection: gate_proj and up_proj share the same input x_norm
        //     and could be fused into one [2×FFN, D] GEMM, halving kernel-launch
        //     overhead and improving SM occupancy at seq=1.
        //     两者共享同一 x_norm，可融合为 [2×FFN, D] GEMM，减半启动开销
        //     并改善 SM 占用率（seq=1 时两者均占用不足）。]
        {
            auto _t = prof.scoped_device("up_proj", stream);
            launch_linear_gemm(act.x_norm, w.up_proj, act.up_out, cublas, stream);
        }
        prof.annotate_bandwidth("up_proj",
            SEQ * D * sizeof(half) + (size_t)FFN * D + SEQ * FFN * sizeof(half));

        // ── Step 9: SwiGLU  gate_out = silu(gate_out) ⊙ up_out  (in-place on gate_out)
        //    SwiGLU 激活（原地覆写 gate_out）
        {
            auto _t = prof.scoped_device("swiglu", stream);
            launch_swiglu(act.gate_out, act.up_out, act.gate_out, stream);
        }
        prof.annotate_bandwidth("swiglu", SEQ * FFN * sizeof(half) * 3);

        // ── Step 10: Down_proj  gate_out [1,FFN] × down [D,FFN]ᵀ → ffn_out [1,D]
        //     Down 投影
        {
            auto _t = prof.scoped_device("down_proj", stream);
            launch_linear_gemm(act.gate_out, w.down_proj, act.ffn_out, cublas, stream);
        }
        prof.annotate_bandwidth("down_proj",
            SEQ * FFN * sizeof(half) + (size_t)D * FFN + SEQ * D * sizeof(half));

        // ── Step 11: Residual Add 2  x += ffn_out  (IN-PLACE, no cudaMemcpy)
        //     残差相加 2：x += ffn_out（原地，无 cudaMemcpy）
        {
            auto _t = prof.scoped_device("res_add_ffn", stream);
            launch_residual_add(act.x, act.ffn_out, stream);
        }
        // act.x now = post-attn residual + ffn_sublayer_output = full layer output
        // act.x 现在 = 注意力后残差 + FFN 子层输出 = 完整层输出
    }

    // =========================================================================
    // 8. Final RMSNorm → lm_head (FP32 path) → ArgMax
    //    最终 RMSNorm → lm_head FP32 路径 → ArgMax
    //
    // lm_head uses a dedicated FP32 GEMM path:
    //   a) k_cast_fp16_to_fp32: x_norm (FP16) → x_norm_fp32 (FP32)  [device kernel]
    //   b) cublasSgemm:  logits = lm_head[V,D] × x_norm_fp32[D,1]  (column-major)
    //
    // lm_head 使用专用 FP32 GEMM 路径：
    //   a) k_cast_fp16_to_fp32：x_norm FP16 → x_norm_fp32 FP32（设备 kernel）
    //   b) cublasSgemm：logits = lm_head[V,D] × x_norm_fp32[D,1]（列主序）
    // =========================================================================
    {
        auto _t = prof.scoped_device("final_norm", stream);
        launch_rms_norm(act.x, final_norm_v, act.x_norm, 1e-5f, stream);
    }
    {
        // Cast x_norm FP16 → FP32 on device (no CPU involvement)
        // 在设备上将 x_norm FP16 转换为 FP32（无 CPU 参与）
        auto _t = prof.scoped_device("lm_head", stream);
        k_cast_fp16_to_fp32<<<(D + 255) / 256, 256, 0, stream>>>(
            static_cast<const half*>(act.x_norm.data_ptr),
            act.x_norm_fp32, D);

        // cublasSgemm: logits[V,1] = lm_head[V,D] × x_norm_fp32[D,1]
        // In cuBLAS column-major:
        //   lm_head [V,D] row-major ≡ A [D,V] col-major (lda=D, CUBLAS_OP_T)
        //   x_norm  [1,D] row-major ≡ B [D,1] col-major (ldb=D, CUBLAS_OP_N)
        //   logits  [1,V] row-major ≡ C [V,1] col-major (ldc=V)
        //
        // [Bug/Imperfection: cublasSgemm has higher overhead than cublasGemmEx
        //  with TC_COMPUTE_32F. For a D=3072, V=128256 GEMV this is memory-bound
        //  (ratio K/N ≈ 24), so compute type matters less than HBM bandwidth.
        //  Consider cublasSgemv for a pure matrix-vector path (no batching needed).
        //  cublasSgemm 开销高于 cublasGemmEx TC_COMPUTE_32F。
        //  D=3072, V=128256 的 GEMV 是内存受限的（K/N ≈ 24），
        //  计算类型影响小于 HBM 带宽。可考虑 cublasSgemv 纯矩阵向量路径。]
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            V, 1, D,
            &alpha,
            static_cast<const float*>(lm_head_v.data_ptr), D,  // A: lm_head[V,D] row-major
            act.x_norm_fp32,                                    // B: x_norm_fp32[D,1]
            D,
            &beta,
            static_cast<float*>(act.logits.data_ptr), V));      // C: logits[V,1]
    }
    prof.mark_decode_end(stream);  // TPOT clock stops / TPOT 计时结束
    {
        auto _t = prof.scoped_device("argmax", stream);
        launch_argmax(act.logits, act.d_token_id, stream);
    }

    // =========================================================================
    // 9. Read result and print report
    //    读取结果并打印报告
    // =========================================================================
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int h_token_id = -1;
    CUDA_CHECK(cudaMemcpy(&h_token_id, act.d_token_id,
                           sizeof(int), cudaMemcpyDeviceToHost));
    std::printf("\n[Output]  next token id = %d\n", h_token_id);

    prof.print_report();

    // =========================================================================
    // 10. Cleanup / 清理
    // =========================================================================
    cudaFree(d_weight_pool);
    cudaFree(d_act_pool);
    cudaStreamDestroy(stream);
    cublasDestroy(cublas);

    return EXIT_SUCCESS;
}

#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "utils/utils.h"
#include "core/view.h"
#include "ops/kernel.h"

// ============================================================================
// gqa_attention_kernel — Correct GQA Softmax Attention
//
// Design: one thread per (output element) = (query_head, head_dim_index).
//   Grid : dim3(n_q_heads * seq_q)    — one block per (token, query head)
//   Block: head_dim                   — one thread per dimension
//
// For seq_kv == 1 (single-token decode step):
//   softmax of exactly one score is always 1.0 regardless of the score value,
//   so the attention output is simply out[h,d] = V[kv_head, d].
//   This path is exact and introduces zero error.
//
// For seq_kv > 1:
//   Uses shared memory for reduction of the Q·K dot product across head_dim
//   threads, then online softmax, then weighted V accumulation.
//   This is not flash-attention (no sequence-dimension tiling), so shared
//   memory usage grows with seq_kv — sufficient for short sequences.
//
// [Bug/Imperfection: seq_kv > 1 path uses per-thread O(seq_kv) shared memory
//  allocated dynamically. For very long sequences this exceeds the 48 KB Smem
//  limit. A proper production kernel would tile over the KV sequence dimension
//  (FlashAttention-style). The current implementation is correct for validation
//  with seq_kv=1 and serves as a reference for future optimization.
//  seq_kv > 1 路径动态分配 O(seq_kv) 共享内存。对于超长序列，这会超过 48 KB
//  上限。生产内核应在 KV 序列维度上分块（FlashAttention 风格）。当前实现
//  在 seq_kv=1 的验证场景下完全正确，可作为后续优化的参考基线。]
// ============================================================================
template <typename T>
__global__ void gqa_attention_kernel(
    const T* __restrict__ q,     // [seq_q,  n_q_heads  * head_dim]
    const T* __restrict__ k,     // [seq_kv, n_kv_heads * head_dim]
    const T* __restrict__ v,     // [seq_kv, n_kv_heads * head_dim]
    T* __restrict__ out,         // [seq_q,  n_q_heads  * head_dim]
    int seq_q, int seq_kv,
    int head_dim, int n_q_heads, int n_kv_heads)
{
    // blockIdx.x = linear index over (seq_q × n_q_heads)
    // threadIdx.x = head dimension index d
    int blk = blockIdx.x;
    int d   = threadIdx.x;

    int q_tok = blk / n_q_heads;
    int h     = blk % n_q_heads;

    if (q_tok >= seq_q || h >= n_q_heads || d >= head_dim) return;

    const int Q_dim  = n_q_heads  * head_dim;
    const int KV_dim = n_kv_heads * head_dim;
    int kv_h = h * n_kv_heads / n_q_heads;   // GQA: share KV heads

    // ── Fast path: seq_kv == 1 ───────────────────────────────────────────
    // softmax([x]) = 1.0 for any x.  attn_out = V[kv_h, d].
    // softmax([x]) = 1.0，无论 x 为何值。attn_out = V[kv_h, d]。
    if (seq_kv == 1) {
        out[q_tok * Q_dim + h * head_dim + d] = v[kv_h * head_dim + d];
        return;
    }

    // ── General path: seq_kv > 1 ────────────────────────────────────────
    // Shared memory layout:
    //   [0 .. head_dim)   — partial dot product reduction buffer
    //   [head_dim .. head_dim + seq_kv) — accumulated softmax scores
    //
    // 共享内存布局：
    //   [0 .. head_dim)           — 点积规约缓冲区
    //   [head_dim .. head_dim + seq_kv) — softmax score 累积
    extern __shared__ float smem[];
    float* s_partial = smem;               // length head_dim
    float* s_scores  = smem + head_dim;    // length seq_kv

    const T* Q_h = q + q_tok * Q_dim + h * head_dim;
    float inv_sqrt_hd = rsqrtf(static_cast<float>(head_dim));

    // ── 1. Compute score for each key token (causal: s <= q_tok) ────────
    // 按因果掩码计算每个 key token 的 attention score（s <= q_tok）
    for (int s = 0; s <= q_tok && s < seq_kv; ++s) {
        const T* K_s = k + s * KV_dim + kv_h * head_dim;

        // Thread d contributes Q[d]*K[s][d] to the dot product
        // 线程 d 贡献点积中的 Q[d]*K[s][d] 项
        s_partial[d] = __half2float(Q_h[d]) * __half2float(K_s[d]);
        __syncthreads();

        // Thread 0 reduces and stores the score
        // 线程 0 规约并存储 score
        if (d == 0) {
            float score = 0.0f;
            for (int i = 0; i < head_dim; ++i) score += s_partial[i];
            s_scores[s] = score * inv_sqrt_hd;
        }
        __syncthreads();
    }

    // ── 2. Online softmax (max + normalize) — computed by thread 0 ──────
    // 在线 softmax（最大值 + 归一化）— 由线程 0 计算
    if (d == 0) {
        int n_keys = min(q_tok + 1, seq_kv);
        float score_max = -1e38f;
        for (int s = 0; s < n_keys; ++s)
            score_max = fmaxf(score_max, s_scores[s]);
        float sum_exp = 0.0f;
        for (int s = 0; s < n_keys; ++s) {
            s_scores[s] = expf(s_scores[s] - score_max);
            sum_exp += s_scores[s];
        }
        // Normalize in-place
        // 原地归一化
        for (int s = 0; s < n_keys; ++s) s_scores[s] /= sum_exp;
        // Zero-fill masked positions so thread d's accumulation is clean
        // 将掩码位置清零，使线程 d 的累加结果干净
        for (int s = n_keys; s < seq_kv; ++s) s_scores[s] = 0.0f;
    }
    __syncthreads();

    // ── 3. Weighted sum of V across key tokens ───────────────────────────
    // 对 key token 的 V 进行加权求和
    float acc = 0.0f;
    int n_keys = min(q_tok + 1, seq_kv);
    for (int s = 0; s < n_keys; ++s) {
        const T* V_s = v + s * KV_dim + kv_h * head_dim;
        acc += s_scores[s] * __half2float(V_s[d]);
    }
    out[q_tok * Q_dim + h * head_dim + d] = __float2half(acc);
}

// ============================================================================
// Launcher
// ============================================================================
template <typename T>
void launch_tiled_attention_kernel(
    const T* q_base, const T* k_base, const T* v_base,
    T* out,
    int seq_q, int seq_kv, int head_dim, int n_q_heads, int n_kv_heads,
    cudaStream_t stream)
{
    int total_blocks = n_q_heads * seq_q;
    int block_size   = head_dim;

    // Shared memory: head_dim floats (reduction) + seq_kv floats (scores)
    // 共享内存：head_dim 个 float（规约）+ seq_kv 个 float（score）
    size_t smem_bytes = (seq_kv == 1)
        ? 0
        : static_cast<size_t>(head_dim + seq_kv) * sizeof(float);

    gqa_attention_kernel<T><<<total_blocks, block_size, smem_bytes, stream>>>(
        q_base, k_base, v_base, out,
        seq_q, seq_kv, head_dim, n_q_heads, n_kv_heads);
    CUDA_CHECK_LAST();
}

// Explicit instantiation of launcher for FP16
// 为 FP16 显式实例化启动器
template void launch_tiled_attention_kernel<half>(
    const half*, const half*, const half*, half*,
    int, int, int, int, int, cudaStream_t);

// ============================================================================
// gqa_cached_attention_kernel — Autoregressive M=1 decode with KV cache
// gqa_cached_attention_kernel — 带 KV 缓存的自回归 M=1 解码
//
// [EN] Design for single-token decode (M=1):
//   Grid : n_q_heads blocks      (24 for Llama 3.2 3B)
//   Block: head_dim threads       (128)
//   Each block handles one query head’s full attention computation.
//
//   Phase 1 — Score computation:
//     For each history position t in [0, seq_len):
//       All HD threads compute Q[h,d]*K[t,kv_h,d] in parallel,
//       then tree-reduce to a single dot product. Thread 0 stores
//       score[t] = dot / sqrt(HD) into shared memory.
//
//   Phase 2 — Softmax (thread 0):
//     Online max + exp + normalize over seq_len scores.
//
//   Phase 3 — Weighted V accumulation:
//     Each thread d accumulates sum_t(score[t] * V[t,kv_h,d]).
//
// [CN] 单 token 解码 (M=1) 设计：
//   Grid : n_q_heads 个 block
//   Block: head_dim 个线程
//   每个 block 处理一个 query head 的完整注意力计算。
//
//   第1阶段 — Score 计算：树形规约得到 Q·K^T / sqrt(HD)。
//   第2阶段 — Softmax：线程 0 计算在线 max + exp + 归一化。
//   第3阶段 — 加权 V 累加：每个线程 d 累加 score[t] * V[t,kv_h,d]。
//
// Shared memory: (head_dim + seq_len) * sizeof(float).
// For MAX_SEQ_LEN=2048, HD=128: (128+2048)*4 = 8.5 KB per block. Fine.
// ============================================================================
__global__ void gqa_cached_attention_kernel(
    const half* __restrict__ q,        // [1, n_q_heads * head_dim]
    const half* __restrict__ k_cache,  // [MAX_SEQ_LEN, n_kv_heads * head_dim]
    const half* __restrict__ v_cache,  // [MAX_SEQ_LEN, n_kv_heads * head_dim]
    half* __restrict__ out,            // [1, n_q_heads * head_dim]
    int seq_len,      // current_pos + 1
    int head_dim,     // 128
    int n_q_heads,    // 24
    int n_kv_heads)   // 8
{
    const int h    = blockIdx.x;    // query head index [0, NH)
    const int d    = threadIdx.x;   // head dim index   [0, HD)
    const int kv_h = h * n_kv_heads / n_q_heads;  // GQA mapping / GQA 映射
    const int KV_stride = n_kv_heads * head_dim;   // row stride in KV cache

    const float inv_sqrt_hd = rsqrtf(static_cast<float>(head_dim));

    // [EN] Load Q[h, d] into register once (reused for every key position).
    // [CN] 将 Q[h, d] 加载到寄存器（每个 key 位置复用）。
    const float q_val = __half2float(q[h * head_dim + d]);

    // Shared memory layout: [head_dim] reduction buffer + [seq_len] scores
    // 共享内存布局：[head_dim] 规约缓冲 + [seq_len] 分数
    extern __shared__ float smem[];
    float* s_reduce = smem;                // [head_dim]
    float* s_scores = smem + head_dim;     // [seq_len]

    // ── Phase 1: Compute attention scores / 计算注意力分数 ─────────────
    for (int t = 0; t < seq_len; ++t) {
        // [EN] Each thread contributes Q[d]*K[t,kv_h,d] to the dot product.
        // [CN] 每个线程贡献点积的 Q[d]*K[t,kv_h,d] 项。
        float k_val = __half2float(k_cache[t * KV_stride + kv_h * head_dim + d]);
        s_reduce[d] = q_val * k_val;
        __syncthreads();

        // [EN] Tree reduction across head_dim threads.
        // [CN] head_dim 个线程的树形规约。
        for (int stride = head_dim >> 1; stride > 0; stride >>= 1) {
            if (d < stride) s_reduce[d] += s_reduce[d + stride];
            __syncthreads();
        }

        // Thread 0 stores the scaled score / 线程 0 存储缩放后的分数
        if (d == 0) s_scores[t] = s_reduce[0] * inv_sqrt_hd;
        __syncthreads();
    }

    // ── Phase 2: Softmax (thread 0 only) / Softmax（仅线程 0） ──────────
    if (d == 0) {
        float max_s = -1e38f;
        for (int t = 0; t < seq_len; ++t)
            max_s = fmaxf(max_s, s_scores[t]);

        float sum_exp = 0.0f;
        for (int t = 0; t < seq_len; ++t) {
            s_scores[t] = expf(s_scores[t] - max_s);
            sum_exp += s_scores[t];
        }

        float inv_sum = 1.0f / sum_exp;
        for (int t = 0; t < seq_len; ++t)
            s_scores[t] *= inv_sum;
    }
    __syncthreads();

    // ── Phase 3: Weighted V accumulation / 加权 V 累加 ────────────────
    float acc = 0.0f;
    for (int t = 0; t < seq_len; ++t) {
        float v_val = __half2float(v_cache[t * KV_stride + kv_h * head_dim + d]);
        acc += s_scores[t] * v_val;
    }

    out[h * head_dim + d] = __float2half(acc);
}

// ============================================================================
// launch_cached_attention — Launcher for autoregressive GQA with KV cache
// launch_cached_attention — 带 KV 缓存的自回归 GQA 启动器
// ============================================================================
void launch_cached_attention(
    const half* q,          // [1, n_q_heads * head_dim]
    const half* k_cache,    // [MAX_SEQ_LEN, n_kv_heads * head_dim]
    const half* v_cache,    // [MAX_SEQ_LEN, n_kv_heads * head_dim]
    half* out,              // [1, n_q_heads * head_dim]
    int seq_len,            // current_pos + 1
    int head_dim,           // 128
    int n_q_heads,          // 24
    int n_kv_heads,         // 8
    cudaStream_t stream)
{
    // [EN] One block per query head, head_dim threads per block.
    // [CN] 每个 query head 一个 block，每个 block head_dim 个线程。
    int blocks = n_q_heads;
    int threads = head_dim;
    size_t smem_bytes = static_cast<size_t>(head_dim + seq_len) * sizeof(float);

    gqa_cached_attention_kernel<<<blocks, threads, smem_bytes, stream>>>(
        q, k_cache, v_cache, out,
        seq_len, head_dim, n_q_heads, n_kv_heads);
    CUDA_CHECK_LAST();
}
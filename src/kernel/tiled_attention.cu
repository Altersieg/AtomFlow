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
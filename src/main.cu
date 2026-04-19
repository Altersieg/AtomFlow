#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include "utils/utils.h"
#include "utils/validator.h"
#include "core/view.h"
#include "ops/kernel.h"
#include "memory/weight_loader.h"
#include "utils/profiler.h"

// ============================================================================
// Benchmark-mode switches / 基准测试模式开关
//
//   ENABLE_VALIDATOR  0 → strip ALL ground-truth validation (chk, fopen/fread,
//                         cudaMemcpy D2H, printf debug) from the forward loop.
//   ENABLE_PROFILER   0 → strip per-kernel cudaEvent record/create overhead.
//
// Set both to 0 to measure the true theoretical GPU TPOT with ZERO host-side
// interference inside the 28-layer forward pass.
//
//   ENABLE_VALIDATOR  0 → 从前向循环中去除所有基准验证（chk、fopen/fread、
//                         cudaMemcpy D2H、printf 调试）。
//   ENABLE_PROFILER   0 → 去除每个 kernel 的 cudaEvent 录制/创建开销。
//
// 两者都设为 0 可测量 28 层前向传播中零主机干扰的真实理论 GPU TPOT。
// ============================================================================
#ifndef ENABLE_VALIDATOR
#define ENABLE_VALIDATOR 0
#endif
#ifndef ENABLE_PROFILER
#define ENABLE_PROFILER  0
#endif

// Warm-up iterations for benchmark mode / 基准测试模式的预热迭代次数
#ifndef BENCHMARK_WARMUP_ITERS
#define BENCHMARK_WARMUP_ITERS 3
#endif

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
// load_fp32_bin_to_fp16_device
//
// Reads a raw FP32 binary file from disk, converts each element to FP16 on the
// CPU, then uploads the converted data to a GPU destination via cudaMemcpy.
//
// 从磁盘读取原始 FP32 二进制文件，在 CPU 端逐元素转换为 FP16，
// 然后通过 cudaMemcpy 同步上传到 GPU 目标地址。
//
// Parameters / 参数:
//   path           — path to the .bin file written by dump_ground_truth.py
//                    dump_ground_truth.py 生成的 .bin 文件路径
//   d_dst          — GPU device pointer to write FP16 data into
//                    用于写入 FP16 数据的 GPU 设备指针
//   expected_numel — expected number of float elements in the file
//                    文件中预期的 float 元素数量
//
// Returns / 返回:
//   Number of elements loaded, or 0 on failure (file missing / size mismatch).
//   已加载的元素数量；文件缺失或大小不匹配时返回 0。
//
// [Bug/Imperfection: CPU-side FP32→FP16 conversion is accurate (IEEE round-to-
//  nearest-even via __float2half) but traverses the data twice: once to convert,
//  once inside cudaMemcpy. An alternative is to upload FP32 and cast on the GPU
//  with k_cast_fp32_to_fp16, saving the CPU loop at the cost of 2× PCIe transfer.
//  For D=3072 both paths are negligible (<1 ms).
//  CPU 端转换精度正确（通过 __float2half 执行 IEEE 最近偶数舍入），
//  但数据被遍历两次：一次转换，一次在 cudaMemcpy 内部。
//  替代方案是先传 FP32，再在 GPU 上用 k_cast_fp32_to_fp16 cast，
//  以 2× PCIe 带宽换省去 CPU 循环。D=3072 时两种路径均可忽略不计（< 1 ms）。]
// ============================================================================
static size_t load_fp32_bin_to_fp16_device(const char* path,
                                            void*       d_dst,
                                            int         expected_numel)
{
    // ── Step 1: Open file / 打开文件 ─────────────────────────────────────────
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return 0;   // file absent — caller decides fallback / 文件不存在，由调用方决定回退策略

    // ── Step 2: Read FP32 elements into host buffer
    //    将 FP32 元素读入主机缓冲区
    //
    //    std::vector guarantees contiguous storage that fread can write directly.
    //    std::vector 保证连续存储，fread 可直接写入。
    std::vector<float> h_fp32(expected_numel);
    size_t n_read = std::fread(h_fp32.data(), sizeof(float), expected_numel, f);
    std::fclose(f);

    if (n_read != static_cast<size_t>(expected_numel)) {
        std::fprintf(stderr,
            "[load_fp32_bin] WARNING: %s — expected %d floats, got %zu\n",
            path, expected_numel, n_read);
        return 0;
    }

    // ── Step 3: Convert FP32 → FP16 element-wise on the CPU
    //    在 CPU 端逐元素将 FP32 转换为 FP16
    //
    //    __float2half() from <cuda_fp16.h> performs IEEE-754 round-to-nearest-
    //    even, matching PyTorch's default fp16 cast behaviour.
    //    <cuda_fp16.h> 的 __float2half() 执行 IEEE-754 最近偶数舍入，
    //    与 PyTorch 默认的 fp16 转换行为一致。
    std::vector<__half> h_fp16(expected_numel);
    for (int i = 0; i < expected_numel; ++i) {
        h_fp16[i] = __float2half(h_fp32[i]);
    }

    // ── Step 4: Synchronous host-to-device copy / 同步主机到设备拷贝
    //
    //    IMPORTANT: cudaMemcpy (synchronous) is required here, NOT
    //    cudaMemcpyAsync.  h_fp16 is a pageable (non-pinned) std::vector.
    //    cudaMemcpyAsync with pageable memory does NOT guarantee that the GPU
    //    has finished reading before the host buffer is destroyed at scope exit.
    //    Using Async would be a use-after-free bug at the end of this function.
    //
    //    重要：此处必须使用 cudaMemcpy（同步），而非 cudaMemcpyAsync。
    //    h_fp16 是可分页（非 pinned）的 std::vector。
    //    cudaMemcpyAsync 搭配可分页内存不保证 GPU 在主机缓冲区被销毁前完成读取。
    //    使用 Async 版本会在函数退出时造成 use-after-free 错误。
    CUDA_CHECK(cudaMemcpy(d_dst,
                          h_fp16.data(),
                          expected_numel * sizeof(__half),
                          cudaMemcpyHostToDevice));

    return static_cast<size_t>(expected_numel);
}

// ============================================================================
// Per-layer GPU weight Views (all pointers into d_weight_pool)
// 每层 GPU 权重 View（所有指针均指向 d_weight_pool 内部）
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
    float* x_norm_fp32; // FP32 [D]           cast of x_norm before FP32 lm_head GEMM
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
    //   For each layer: norm×2(FP32), qkv(FP8)+scales(FP16), o/gate/up/down same pattern
    //   model.norm [D] FP32
    //   lm_head [V, D] FP32
    // =========================================================================

    auto fp8_sz   = [](int r, int c) -> size_t { return (size_t)r * c; };
    auto scale_sz = [&](int r, int c) -> size_t {
        return (size_t)r * (c / GS) * sizeof(uint16_t);
    };
    auto fp32_sz  = [](int n) -> size_t { return (size_t)n * sizeof(float); };

    size_t pool_sz = 0;
    pool_sz += fp32_sz(V) * D;                          // embed_tokens
    for (int i = 0; i < NL; ++i) {
        pool_sz += fp32_sz(D) * 2;                      // 2 × RMSNorm weights
        pool_sz += fp8_sz(QKV_OUT, D);                  // qkv weights
        pool_sz += scale_sz(QKV_OUT, D);                // qkv scales
        pool_sz += fp8_sz(D, D);                        // o_proj weights
        pool_sz += scale_sz(D, D);                      // o_proj scales
        pool_sz += fp8_sz(FFN, D);                      // gate weights
        pool_sz += scale_sz(FFN, D);                    // gate scales
        pool_sz += fp8_sz(FFN, D);                      // up weights
        pool_sz += scale_sz(FFN, D);                    // up scales
        pool_sz += fp8_sz(D, FFN);                      // down weights
        pool_sz += scale_sz(D, FFN);                    // down scales
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
    // Copy FP16 per-group scales from mmap cursor into the GPU weight pool.
    // 将 FP16 逐组 scale 从 mmap 游标复制到 GPU 权重池。
    auto copy_scales = [&](int rows, int cols) -> View {
        size_t bytes = scale_sz(rows, cols);
        return copy_view(loader.next<uint16_t>(bytes / sizeof(uint16_t)),
                         bytes, DataType::FP16, {rows, cols / GS});
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

        lw[i].qkv          = copy_view(loader.next<uint8_t>(fp8_sz(QKV_OUT, D)),
                                         fp8_sz(QKV_OUT, D), DataType::FP8_E4M3, {QKV_OUT, D});
        lw[i].qkv_scales    = copy_scales(QKV_OUT, D);

        lw[i].o_proj        = copy_view(loader.next<uint8_t>(fp8_sz(D, D)),
                                         fp8_sz(D, D), DataType::FP8_E4M3, {D, D});
        lw[i].o_proj_scales = copy_scales(D, D);

        lw[i].gate_proj     = copy_view(loader.next<uint8_t>(fp8_sz(FFN, D)),
                                         fp8_sz(FFN, D), DataType::FP8_E4M3, {FFN, D});
        lw[i].gate_scales   = copy_scales(FFN, D);

        lw[i].up_proj       = copy_view(loader.next<uint8_t>(fp8_sz(FFN, D)),
                                         fp8_sz(FFN, D), DataType::FP8_E4M3, {FFN, D});
        lw[i].up_scales     = copy_scales(FFN, D);

        lw[i].down_proj     = copy_view(loader.next<uint8_t>(fp8_sz(D, FFN)),
                                         fp8_sz(D, FFN), DataType::FP8_E4M3, {D, FFN});
        lw[i].down_scales   = copy_scales(D, FFN);

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
    act_sz += SEQ * D            * sizeof(half)  * 3;  // x, x_norm, attn_out
    act_sz += SEQ * QKV_OUT      * sizeof(half);        // qkv_out
    act_sz += SEQ * FFN          * sizeof(half)  * 2;  // gate_out, up_out
    act_sz += SEQ * D            * sizeof(half);        // ffn_out
    act_sz += (size_t)FFN * D   * sizeof(half);        // dequant_ws (largest FP8 weight → FP16)
    act_sz += SEQ * D            * sizeof(float);       // x_norm_fp32 (lm_head input)
    act_sz += SEQ * V            * sizeof(float);       // logits FP32
    act_sz += sizeof(int);                              // token_id
    act_sz += 2 * HD             * sizeof(float);       // rope cos/sin (seq_len=1)

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
    act.ffn_out    = mk(SEQ*D          *sizeof(half),  DataType::FP16, {(int)SEQ, D});
    // Dequantization workspace: large enough for the biggest FP8 weight (FFN×D).
    // 反量化工作区：大小足以容纳最大的 FP8 权重（FFN×D）。
    act.dequant_ws = mk((size_t)FFN*D  *sizeof(half),  DataType::FP16, {FFN, D});
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
        // Inject the ground-truth last-token embedding into act.x.
        // On success, act.x holds exactly the same FP16 vector that HuggingFace
        // placed at the last sequence position before layer 0.
        // On failure (file not present), fall back to the BOS embed-lookup kernel.
        //
        // 将基准真值最后 token 嵌入注入 act.x。
        // 成功时，act.x 持有与 HuggingFace 在第 0 层之前最后序列位置
        // 相同的 FP16 向量。文件不存在时，回退到 BOS 嵌入查找 kernel。
        const char* embed_gt_path = "ground_truth/gt_input_embeddings.bin";
        size_t n = load_fp32_bin_to_fp16_device(embed_gt_path, act.x.data_ptr, D);
        if (n > 0) {
            std::printf("[Input]  GT embeddings loaded (%zu floats → FP16): %s\n",
                        n, embed_gt_path);
        } else {
            // Fallback: embed BOS token (id=1) directly on the GPU.
            // 回退：在 GPU 上直接嵌入 BOS token (id=1)。
            std::printf("[Input]  GT file absent — BOS token (id=1) fallback\n");
            constexpr int BOS_TOKEN_ID = 1;
            k_embed_lookup<<<(D + 255) / 256, 256, 0, stream>>>(
                static_cast<const float*>(embed_v.data_ptr),
                BOS_TOKEN_ID,
                static_cast<half*>(act.x.data_ptr),
                D);
            CUDA_CHECK_LAST();
        }
    }
    prof.mark_first_token();   // TTFT boundary / TTFT 边界

#if ENABLE_VALIDATOR
    // ── Task 1 sanity check: print first 5 GPU act.x values vs GT file ───────
    // Task 1 健全性检查：打印 GPU act.x 前 5 个元素与 GT 文件对比。
    //
    // Both values should agree to ~3 decimal places (FP16 has ~3.3 significant
    // decimal digits). A large diff means the embedding injection is still wrong.
    // 两者应在小数点后 3 位内吃合（FP16 约 3.3 位十进制有效数字）。
    // 差异较大意味着嵌入注入仍然错误。
    {
        constexpr int N_CHK = 5;
        __half h_x[N_CHK];
        // Synchronous copy: block until GPU has written the embeddings.
        // 同步拷贝：阻塞直到 GPU 完成嵌入写入。
        CUDA_CHECK(cudaMemcpy(h_x, act.x.data_ptr,
                              N_CHK * sizeof(__half), cudaMemcpyDeviceToHost));
        std::FILE* fg = std::fopen("ground_truth/gt_input_embeddings.bin", "rb");
        float h_gt[N_CHK] = {};
        if (fg) { std::fread(h_gt, sizeof(float), N_CHK, fg); std::fclose(fg); }
        std::printf("[act.x check]  First %d elements (GPU FP16 vs GT FP32):\n", N_CHK);
        for (int i = 0; i < N_CHK; ++i) {
            float gpu_val = __half2float(h_x[i]);
            std::printf("  [%d]  GPU=% .6f   GT=% .6f   diff=% .2e\n",
                        i, gpu_val, h_gt[i], h_gt[i] - gpu_val);
        }
    }
#endif // ENABLE_VALIDATOR

#if ENABLE_VALIDATOR
    // =========================================================================
    // Validator setup — skipped silently if ground_truth/ files are absent.
    // 验证器设置——如果 ground_truth/ 文件不存在则静默跳过。
    // =========================================================================
    validate_print_header();
    int v_pass = 0, v_total = 0;

    // chk: sync stream → validate View against GT file → accumulate pass/fail.
    // File-existence is probed with fopen; missing files are silently skipped.
    // chk：同步流 → 验证 View 与基准文件 → 累计通过/失败计数。
    // 用 fopen 检查文件是否存在；缺失文件静默跳过。
    auto chk = [&](const View& v, const char* path, const char* label) {
        std::FILE* probe = std::fopen(path, "rb");
        if (!probe) return;
        std::fclose(probe);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto r = validate_view(v, path, label);
        if (r.numel > 0) { v_pass += r.passed; ++v_total; }
    };
#endif // ENABLE_VALIDATOR

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
    // =====================================================================
    // Macro benchmark: warm-up + chrono-based pure TPOT measurement.
    // 宏观基准：预热 + 基于 chrono 的纯 TPOT 测量。
    //
    // The lambda `run_forward` encapsulates the ENTIRE forward pass
    // (28 layers + final_norm + lm_head + argmax) so it can be called
    // repeatedly for warm-up and then timed.
    //
    // run_forward lambda 封装整个前向传播（28 层 + final_norm + lm_head + argmax），
    // 以便反复调用用于预热和计时。
    // =====================================================================

    // --- Forward pass body (used by both warm-up and timed run) ---
    // --- 前向传播主体（预热和计时共用）---
    // Lambda captures all locals by reference.
    // Lambda 通过引用捕获所有局部变量。
    auto _run_forward_body = [&]() {

#if ENABLE_PROFILER
    prof.mark_decode_start(stream);  // TPOT clock starts / TPOT 计时开始
#endif

    for (int layer = 0; layer < NL; ++layer) {
        const LayerWeights& w = lw[layer];
#if ENABLE_VALIDATOR
        // Probe points: layers 0, 13, 27 for ground-truth comparison.
        // 探测点：第 0、13、27 层与基准真值对比。
        const bool is_probe = (layer == 0 || layer == 13 || layer == 27);
#endif
#if ENABLE_PROFILER
        // Memory traffic constant for RMSNorm (read x, read weight, write x_norm)
        // RMSNorm 的显存流量常量（读 x，读 weight，写 x_norm）
        const size_t bw_norm = SEQ * D * sizeof(half) * 2 + SEQ * D * sizeof(float);
#endif

        // ── Step 1: RMSNorm (Attention)  x → x_norm  [x UNCHANGED = residual base]
        //    注意力 RMSNorm：x → x_norm（x 保持不变，充当残差基）
        {
#if ENABLE_PROFILER
            auto _t = prof.scoped_device("rms_norm_attn", stream);
#endif
            launch_rms_norm(act.x, w.input_norm, act.x_norm, 1e-5f, stream);
        }
#if ENABLE_PROFILER
        prof.annotate_bandwidth("rms_norm_attn", bw_norm);
#endif
#if ENABLE_VALIDATOR
        if (is_probe) {
            char path[64], label[48];
            std::snprintf(path,  sizeof(path),  "ground_truth/gt_layer%d_norm_in.bin", layer);
            std::snprintf(label, sizeof(label), "Layer %d  Norm Out (x_norm)", layer);
            chk(act.x_norm, path, label);
        }
#endif

        // ── Step 2: Fused W8A16 QKV GEMV  x_norm [1,D] × qkv [QKV_OUT,D]ᵀ → qkv_out [1,QKV_OUT]
        //    融合 W8A16 QKV GEMV：FP8 反量化 + 内积在单 kernel 内完成，零中间写回
        {
#if ENABLE_PROFILER
            auto _t = prof.scoped_device("qkv_gemm", stream);
#endif
            launch_w8a16_gemv(act.x_norm, w.qkv, w.qkv_scales,
                              act.qkv_out, GS, stream);
        }
#if ENABLE_PROFILER
        prof.annotate_bandwidth("qkv_gemm",
            SEQ * D * sizeof(half)
            + (size_t)QKV_OUT * D
            + (size_t)QKV_OUT * (D / GS) * sizeof(half)
            + SEQ * QKV_OUT * sizeof(half));
#endif

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
#if ENABLE_PROFILER
            auto _t = prof.scoped_device("rope", stream);
#endif
            launch_rope(q_view, k_view, d_cos, d_sin, (int)SEQ, NKV, NH, HD, stream);
        }

        // ── Step 4: Tiled Attention  Q,K,V → attn_out [1,D]
        //    Tiled 注意力
        {
#if ENABLE_PROFILER
            auto _t = prof.scoped_device("tiled_attn", stream);
#endif
            launch_tiled_attention_kernel<half>(
                q_ptr, k_ptr, v_ptr,
                static_cast<half*>(act.attn_out.data_ptr),
                (int)SEQ, (int)SEQ,   // seq_q, seq_kv (self-attention)
                HD, NH, NKV, stream); // head_dim, n_q_heads, n_kv_heads
        }
        // ── Step 5: Fused W8A16 O_proj  attn_out [1,D] × o_proj [D,D]ᵀ → ffn_out [1,D]
        //    融合 W8A16 O 投影（借用 ffn_out 作为临时缓冲区）
        {
#if ENABLE_PROFILER
            auto _t = prof.scoped_device("o_proj", stream);
#endif
            launch_w8a16_gemv(act.attn_out, w.o_proj, w.o_proj_scales,
                              act.ffn_out, GS, stream);
        }
#if ENABLE_PROFILER
        prof.annotate_bandwidth("o_proj",
            SEQ * D * sizeof(half) + (size_t)D * D
            + (size_t)D * (D / GS) * sizeof(half) + SEQ * D * sizeof(half));
#endif
#if ENABLE_VALIDATOR
        if (is_probe) {
            char path[64], label[48];
            std::snprintf(path,  sizeof(path),  "ground_truth/gt_layer%d_attn_out.bin", layer);
            std::snprintf(label, sizeof(label), "Layer %d  Attn Out (post o_proj)", layer);
            chk(act.ffn_out, path, label);
        }
#endif

        // ── Step 6: Residual Add 1  x += ffn_out  (IN-PLACE, no cudaMemcpy)
        //    残差相加 1：x += ffn_out（原地，无 cudaMemcpy）
        {
#if ENABLE_PROFILER
            auto _t = prof.scoped_device("res_add_attn", stream);
#endif
            launch_residual_add(act.x, act.ffn_out, stream);
        }
        // act.x now = x_before_layer + attn_sublayer_output
        // act.x 现在 = 层前 x + 注意力子层输出

        // ── Step 7: RMSNorm (MLP)  x → x_norm  [x is now post-attn residual base]
        //    MLP 前 RMSNorm：x → x_norm（x 现在是注意力后残差基）
        {
#if ENABLE_PROFILER
            auto _t = prof.scoped_device("rms_norm_mlp", stream);
#endif
            launch_rms_norm(act.x, w.post_norm, act.x_norm, 1e-5f, stream);
        }
#if ENABLE_PROFILER
        prof.annotate_bandwidth("rms_norm_mlp", bw_norm);
#endif

        // ── Step 8a: Fused W8A16 Gate_proj  x_norm [1,D] × gate [FFN,D]ᵀ → gate_out [1,FFN]
        //     融合 W8A16 Gate 投影
        {
#if ENABLE_PROFILER
            auto _t = prof.scoped_device("gate_proj", stream);
#endif
            launch_w8a16_gemv(act.x_norm, w.gate_proj, w.gate_scales,
                              act.gate_out, GS, stream);
        }
#if ENABLE_PROFILER
        prof.annotate_bandwidth("gate_proj",
            SEQ * D * sizeof(half) + (size_t)FFN * D
            + (size_t)FFN * (D / GS) * sizeof(half) + SEQ * FFN * sizeof(half));
#endif

        // ── Step 8b: Up_proj  x_norm [1,D] × up [FFN,D]ᵀ → up_out [1,FFN]
        //    [Bug/Imperfection: gate_proj and up_proj share the same input x_norm
        //     and could be fused into one [2×FFN, D] GEMM, halving kernel-launch
        //     overhead and improving SM occupancy at seq=1.
        //     两者共享同一 x_norm，可融合为 [2×FFN, D] GEMM，减半启动开销
        //     并改善 SM 占用率（seq=1 时两者均占用不足）。]
        {
#if ENABLE_PROFILER
            auto _t = prof.scoped_device("up_proj", stream);
#endif
            launch_w8a16_gemv(act.x_norm, w.up_proj, w.up_scales,
                              act.up_out, GS, stream);
        }
#if ENABLE_PROFILER
        prof.annotate_bandwidth("up_proj",
            SEQ * D * sizeof(half) + (size_t)FFN * D
            + (size_t)FFN * (D / GS) * sizeof(half) + SEQ * FFN * sizeof(half));
#endif

        // ── Step 9: SwiGLU  gate_out = silu(gate_out) ⊙ up_out  (in-place on gate_out)
        //    SwiGLU 激活（原地覆写 gate_out）
        {
#if ENABLE_PROFILER
            auto _t = prof.scoped_device("swiglu", stream);
#endif
            launch_swiglu(act.gate_out, act.up_out, act.gate_out, stream);
        }
#if ENABLE_PROFILER
        prof.annotate_bandwidth("swiglu", SEQ * FFN * sizeof(half) * 3);
#endif
        // ── Step 10: Fused W8A16 Down_proj  gate_out [1,FFN] × down [D,FFN]ᵀ → ffn_out [1,D]
        //     融合 W8A16 Down 投影
        {
#if ENABLE_PROFILER
            auto _t = prof.scoped_device("down_proj", stream);
#endif
            launch_w8a16_gemv(act.gate_out, w.down_proj, w.down_scales,
                              act.ffn_out, GS, stream);
        }
#if ENABLE_PROFILER
        prof.annotate_bandwidth("down_proj",
            SEQ * FFN * sizeof(half) + (size_t)D * FFN
            + (size_t)D * (FFN / GS) * sizeof(half) + SEQ * D * sizeof(half));
#endif
#if ENABLE_VALIDATOR
        if (is_probe) {
            char path[64], label[48];
            std::snprintf(path,  sizeof(path),  "ground_truth/gt_layer%d_mlp_out.bin", layer);
            std::snprintf(label, sizeof(label), "Layer %d  MLP Out (post down_proj)", layer);
            chk(act.ffn_out, path, label);
        }
#endif

        // ── Step 11: Residual Add 2  x += ffn_out  (IN-PLACE, no cudaMemcpy)
        //     残差相加 2：x += ffn_out（原地，无 cudaMemcpy）
        {
#if ENABLE_PROFILER
            auto _t = prof.scoped_device("res_add_ffn", stream);
#endif
            launch_residual_add(act.x, act.ffn_out, stream);
        }
        // act.x now = post-attn residual + ffn_sublayer_output = full layer output
        // act.x 现在 = 注意力后残差 + FFN 子层输出 = 完整层输出
    } // end layer loop / 结束层循环

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
#if ENABLE_PROFILER
        auto _t = prof.scoped_device("final_norm", stream);
#endif
        launch_rms_norm(act.x, final_norm_v, act.x_norm, 1e-5f, stream);
    }
    {
#if ENABLE_PROFILER
        auto _t = prof.scoped_device("lm_head", stream);
#endif
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
#if ENABLE_PROFILER
    prof.mark_decode_end(stream);  // TPOT clock stops / TPOT 计时结束
#endif
    {
#if ENABLE_PROFILER
        auto _t = prof.scoped_device("argmax", stream);
#endif
        launch_argmax(act.logits, act.d_token_id, stream);
    }

    }; // end _run_forward_body lambda / 结束前向传播 lambda

    // =====================================================================
    // Warm-up: run the forward pass BENCHMARK_WARMUP_ITERS times to
    // stabilize GPU clocks, fill caches, and JIT-compile any lazy kernels.
    // 预热：运行前向传播 BENCHMARK_WARMUP_ITERS 次以稳定 GPU 时钟、
    // 填充缓存、并 JIT 编译所有惰性 kernel。
    // =====================================================================
    std::printf("\n[Benchmark]  Warm-up: %d iterations...\n", BENCHMARK_WARMUP_ITERS);
    for (int wi = 0; wi < BENCHMARK_WARMUP_ITERS; ++wi) {
        // Re-inject embeddings each iteration (forward pass mutates act.x).
        // 每次迭代重新注入嵌入（前向传播会修改 act.x）。
        load_fp32_bin_to_fp16_device("ground_truth/gt_input_embeddings.bin",
                                     act.x.data_ptr, D);
        _run_forward_body();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("[Benchmark]  Warm-up done.\n");

    // =====================================================================
    // Timed run: single forward pass with macroscopic chrono timer.
    // 计时运行：使用宏观 chrono 计时器的单次前向传播。
    // =====================================================================
    // Re-inject embeddings for the timed run.
    // 为计时运行重新注入嵌入。
    load_fp32_bin_to_fp16_device("ground_truth/gt_input_embeddings.bin",
                                 act.x.data_ptr, D);
    CUDA_CHECK(cudaDeviceSynchronize());  // drain pipeline before timing

    auto t_start = std::chrono::steady_clock::now();
    _run_forward_body();
    CUDA_CHECK(cudaDeviceSynchronize());  // wait for ALL GPU work to finish
    auto t_end = std::chrono::steady_clock::now();

    double tpot_chrono_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    double tok_per_sec    = 1000.0 / tpot_chrono_ms;

    std::printf("\n");
    std::printf("╔══════════════════════════════════════════════════════╗\n");
    std::printf("║  AtomFlow  \u00b7  Pure GPU Benchmark (chrono)       ║\n");
    std::printf("╠══════════════════════════════════════════════════════╣\n");
    std::printf("║  Warm-up iterations:  %d                          ║\n", BENCHMARK_WARMUP_ITERS);
    std::printf("║  TPOT (chrono):       %8.3f ms                 ║\n", tpot_chrono_ms);
    std::printf("║  Generation speed:    %8.1f tok/s               ║\n", tok_per_sec);
    std::printf("╚══════════════════════════════════════════════════════╝\n");

    // =========================================================================
    // 9. Read result and print report
    //    读取结果并打印报告
    // =========================================================================
    CUDA_CHECK(cudaStreamSynchronize(stream));

#if ENABLE_VALIDATOR
    // Validate logits — sync already done above.
    // 验证 logits——上方的同步已保证 GPU 写完。
    chk(act.logits, "ground_truth/gt_logits.bin", "Logits");
    validate_print_footer(v_pass, v_total);
#endif

    int h_token_id = -1;
    CUDA_CHECK(cudaMemcpy(&h_token_id, act.d_token_id,
                           sizeof(int), cudaMemcpyDeviceToHost));
    std::printf("\n[Output]  next token id = %d\n", h_token_id);

#if ENABLE_PROFILER
    prof.print_report();
#endif

    // =========================================================================
    // 10. Cleanup / 清理
    // =========================================================================
    cudaFree(d_weight_pool);
    cudaFree(d_act_pool);
    cudaStreamDestroy(stream);
    cublasDestroy(cublas);

    return EXIT_SUCCESS;
}

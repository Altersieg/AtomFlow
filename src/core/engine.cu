#include "core/engine.h"
#include "memory/weight_loader.h"
#include "ops/kernel.h"
#include "utils/utils.h"
#if ENABLE_VALIDATOR
#include "utils/validator.h"
#endif

#include <cstdio>
#include <cstring>

// ============================================================================
// Destructor — release ALL GPU resources
// 析构函数 — 释放所有 GPU 资源
// ============================================================================
AtomFlowEngine::~AtomFlowEngine() {
    if (d_weight_pool_) cudaFree(d_weight_pool_);
    if (d_act_pool_)    cudaFree(d_act_pool_);
    if (d_kv_pool_)     cudaFree(d_kv_pool_);
    if (d_cublas_ws_)   cudaFree(d_cublas_ws_);
    if (stream_)        cudaStreamDestroy(stream_);
    if (cublas_)        cublasDestroy(cublas_);
}

// ============================================================================
// initialize — load weights, allocate GPU arenas, build RoPE cache
// initialize — 加载权重、分配 GPU arena、构建 RoPE 缓存
// ============================================================================
void AtomFlowEngine::initialize(const std::string& weights_path) {

    // ── 1. mmap weights + parse header / 内存映射权重文件并解析头部 ──────
    WeightLoader loader(weights_path);
    const AtomHeader& hdr = loader.header;

    D       = hdr.dim;
    FFN     = hdr.hidden_dim;
    V       = hdr.vocab_size;
    NL      = hdr.n_layers;
    GS      = hdr.group_size;
    NH      = hdr.n_heads;
    NKV     = hdr.n_kv_heads;
    HD      = D / NH;
    Q_DIM   = D;
    KV_DIM  = NKV * HD;
    QKV_OUT = Q_DIM + KV_DIM + KV_DIM;

    std::printf("──────────────────────────────────────────────\n");
    std::printf("AtomFlow  MVP Decode Step  |  %s\n", weights_path.c_str());
    std::printf("  D=%d  FFN=%d  NL=%d  NH=%d  NKV=%d  HD=%d\n",
                D, FFN, NL, NH, NKV, HD);
    std::printf("  QKV_OUT=%d  V=%d\n", QKV_OUT, V);
    std::printf("  file size: %.2f GiB\n",
                static_cast<double>(loader.file_size()) / (1 << 30));
    std::printf("──────────────────────────────────────────────\n");

    // ── 2. Compute weight pool size and allocate GPU arena / 计算权重池大小 ──
    auto fp8_sz   = [](int r, int c) -> size_t { return (size_t)r * c; };
    auto scale_sz = [&](int r, int c) -> size_t {
        return (size_t)r * (c / GS) * sizeof(uint16_t);
    };
    auto fp32_sz  = [](int n) -> size_t { return (size_t)n * sizeof(float); };
    auto fp16_sz  = [](int n) -> size_t { return (size_t)n * sizeof(half); };

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
    pool_sz += fp16_sz(V) * D;                          // lm_head (FP16)
    pool_sz += pool_sz / 16;                            // 6% headroom

    CUDA_CHECK(cudaMalloc(&d_weight_pool_, pool_sz));
    std::printf("[Weight pool]  GPU alloc %.2f GiB\n",
                (double)pool_sz / (1 << 30));

    // ── 3. Copy weights from mmap → GPU pool, build Views / 拷贝权重 ────
    uint8_t* d_cur = static_cast<uint8_t*>(d_weight_pool_);

    auto copy_view = [&](const void* h_src, size_t bytes,
                         DataType dtype, std::initializer_list<int> dims) -> View {
        CUDA_CHECK(cudaMemcpy(d_cur, h_src, bytes, cudaMemcpyHostToDevice));
        View v = create_contiguous_view(d_cur, dtype, dims);
        d_cur += bytes;
        return v;
    };
    auto copy_scales = [&](int rows, int cols) -> View {
        size_t bytes = scale_sz(rows, cols);
        return copy_view(loader.next<uint16_t>(bytes / sizeof(uint16_t)),
                         bytes, DataType::FP16, {rows, cols / GS});
    };

    // ---- embed_tokens ----
    embed_v_ = copy_view(
        loader.next<float>((size_t)V * D),
        fp32_sz(V) * D, DataType::FP32, {V, D});

    // ---- Per-layer weights ----
    lw_.resize(NL);
    for (int i = 0; i < NL; ++i) {
        lw_[i].input_norm = copy_view(loader.next<float>(D), fp32_sz(D),
                                      DataType::FP32, {D});
        lw_[i].post_norm  = copy_view(loader.next<float>(D), fp32_sz(D),
                                      DataType::FP32, {D});

        lw_[i].qkv          = copy_view(loader.next<uint8_t>(fp8_sz(QKV_OUT, D)),
                                         fp8_sz(QKV_OUT, D), DataType::FP8_E4M3, {QKV_OUT, D});
        lw_[i].qkv_scales    = copy_scales(QKV_OUT, D);

        lw_[i].o_proj        = copy_view(loader.next<uint8_t>(fp8_sz(D, D)),
                                         fp8_sz(D, D), DataType::FP8_E4M3, {D, D});
        lw_[i].o_proj_scales = copy_scales(D, D);

        lw_[i].gate_proj     = copy_view(loader.next<uint8_t>(fp8_sz(FFN, D)),
                                         fp8_sz(FFN, D), DataType::FP8_E4M3, {FFN, D});
        lw_[i].gate_scales   = copy_scales(FFN, D);

        lw_[i].up_proj       = copy_view(loader.next<uint8_t>(fp8_sz(FFN, D)),
                                         fp8_sz(FFN, D), DataType::FP8_E4M3, {FFN, D});
        lw_[i].up_scales     = copy_scales(FFN, D);

        lw_[i].down_proj     = copy_view(loader.next<uint8_t>(fp8_sz(D, FFN)),
                                         fp8_sz(D, FFN), DataType::FP8_E4M3, {D, FFN});
        lw_[i].down_scales   = copy_scales(D, FFN);

        if (i == 0)
            std::printf("[layer  0 weights loaded]  d_cur offset=%td B\n",
                        d_cur - (uint8_t*)d_weight_pool_);
    }

    // ---- Final norm + lm_head ----
    final_norm_v_ = copy_view(loader.next<float>(D),
                               fp32_sz(D), DataType::FP32, {D});
    // [EN] lm_head stored as FP16 — halves VRAM; cublasGemmEx does FP16-in/FP32-out.
    // [CN] lm_head 以 FP16 存储 — 显存减半；cublasGemmEx 实现 FP16 输入 / FP32 输出。
    lm_head_v_ = copy_view(loader.next<uint16_t>((size_t)V * D),
                            fp16_sz(V) * D, DataType::FP16, {V, D});

    std::printf("[All weights uploaded]  pool used: %.2f GiB\n",
                (double)(d_cur - (uint8_t*)d_weight_pool_) / (1 << 30));

    // ── 4. Allocate activation arena / 分配激活 arena ────────────────────
    size_t act_sz = 0;
    act_sz += SEQ * D            * sizeof(half)  * 3;  // x, x_norm, attn_out
    act_sz += SEQ * QKV_OUT      * sizeof(half);        // qkv_out
    act_sz += SEQ * FFN          * sizeof(half)  * 2;  // gate_out, up_out
    act_sz += SEQ * D            * sizeof(half);        // ffn_out
    act_sz += SEQ * V            * sizeof(float);       // logits FP32
    act_sz += sizeof(int);                              // token_id
    // [EN] RoPE cache: [MAX_SEQ_LEN, HD/2] per cos and sin.
    // [CN] RoPE 缓存：cos 和 sin 各 [MAX_SEQ_LEN, HD/2]。
    act_sz += 2 * (size_t)MAX_SEQ_LEN * (HD / 2) * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_act_pool_, act_sz));
    CUDA_CHECK(cudaMemset(d_act_pool_, 0, act_sz));

    uint8_t* a = static_cast<uint8_t*>(d_act_pool_);

    auto mk = [&](size_t bytes, DataType dt, std::initializer_list<int> dims) -> View {
        View v = create_contiguous_view(a, dt, dims);
        a += bytes;
        return v;
    };

    act_.x          = mk(SEQ*D      *sizeof(half),  DataType::FP16, {(int)SEQ, D});
    act_.x_norm     = mk(SEQ*D      *sizeof(half),  DataType::FP16, {(int)SEQ, D});
    act_.attn_out   = mk(SEQ*D      *sizeof(half),  DataType::FP16, {(int)SEQ, D});
    act_.qkv_out    = mk(SEQ*QKV_OUT*sizeof(half),  DataType::FP16, {(int)SEQ, QKV_OUT});
    act_.gate_out   = mk(SEQ*FFN    *sizeof(half),  DataType::FP16, {(int)SEQ, FFN});
    act_.up_out     = mk(SEQ*FFN    *sizeof(half),  DataType::FP16, {(int)SEQ, FFN});
    act_.ffn_out    = mk(SEQ*D          *sizeof(half),  DataType::FP16, {(int)SEQ, D});
    act_.logits     = mk(SEQ*V      *sizeof(float), DataType::FP32, {(int)SEQ, V});
    act_.d_token_id = reinterpret_cast<int*>(a);   a += sizeof(int);
    d_cos_          = reinterpret_cast<float*>(a); a += (size_t)MAX_SEQ_LEN * (HD / 2) * sizeof(float);
    d_sin_          = reinterpret_cast<float*>(a); a += (size_t)MAX_SEQ_LEN * (HD / 2) * sizeof(float);

    // ── 5. Allocate KV cache pool / 分配 KV 缓存池 ─────────────────
    // [EN] One contiguous allocation for all layers' K and V caches.
    //      Layout: [L0_K | L0_V | L1_K | L1_V | ... | L(NL-1)_K | L(NL-1)_V]
    //      Each slab: MAX_SEQ_LEN * KV_DIM * sizeof(half) bytes.
    // [CN] 一次性连续分配所有层的 K 和 V 缓存。
    //      布局：[L0_K | L0_V | L1_K | L1_V | ... | L(NL-1)_K | L(NL-1)_V]
    //      每块：MAX_SEQ_LEN * KV_DIM * sizeof(half) 字节。
    {
        const size_t slab_bytes = (size_t)MAX_SEQ_LEN * KV_DIM * sizeof(half);
        const size_t kv_pool_sz = (size_t)NL * 2 * slab_bytes;

        CUDA_CHECK(cudaMalloc(&d_kv_pool_, kv_pool_sz));
        CUDA_CHECK(cudaMemset(d_kv_pool_, 0, kv_pool_sz));

        uint8_t* kv_cur = static_cast<uint8_t*>(d_kv_pool_);
        for (int i = 0; i < NL; ++i) {
            act_.k_cache[i] = create_contiguous_view(
                kv_cur, DataType::FP16, {MAX_SEQ_LEN, KV_DIM});
            kv_cur += slab_bytes;

            act_.v_cache[i] = create_contiguous_view(
                kv_cur, DataType::FP16, {MAX_SEQ_LEN, KV_DIM});
            kv_cur += slab_bytes;
        }
        current_pos_ = 0;

        std::printf("[KV cache]  GPU alloc %.2f MiB  (%d layers \u00d7 2 \u00d7 %d \u00d7 %d \u00d7 FP16)\n",
                    (double)kv_pool_sz / (1 << 20), NL, MAX_SEQ_LEN, KV_DIM);
    }

    // ── 6. CUDA context, RoPE cache / CUDA 上下文、RoPE 缓存 ───────────
    CUBLAS_CHECK(cublasCreate(&cublas_));
    CUDA_CHECK(cudaStreamCreate(&stream_));
    CUBLAS_CHECK(cublasSetStream(cublas_, stream_));

    // Pre-allocate a cuBLAS workspace so that cuBLAS never calls cudaMalloc
    // during stream capture (which would be illegal). 4 MiB is sufficient
    // for all GEMM shapes used in Llama 3.2 3B.
    // 预分配 cuBLAS 工作区，使 cuBLAS 在流捕获期间不再调用 cudaMalloc
    // （否则将非法）。4 MiB 足够 Llama 3.2 3B 的所有 GEMM 形状。
    static constexpr size_t CUBLAS_WS_SIZE = 4 * 1024 * 1024;  // 4 MiB
    CUDA_CHECK(cudaMalloc(&d_cublas_ws_, CUBLAS_WS_SIZE));
    CUBLAS_CHECK(cublasSetWorkspace(cublas_, d_cublas_ws_, CUBLAS_WS_SIZE));

    // [EN] Build RoPE cache for all positions up to MAX_SEQ_LEN.
    // [CN] 为所有位置（至 MAX_SEQ_LEN）构建 RoPE 缓存。
    build_rope_cache(d_cos_, d_sin_, MAX_SEQ_LEN, HD, /*base=*/500000.0f, stream_);
}

// ============================================================================
// inject_input — load embeddings into act_.x
// inject_input — 将嵌入加载到 act_.x
// ============================================================================
void AtomFlowEngine::inject_input(const char* gt_embed_path, bool verbose) {
    size_t n = load_fp32_bin_to_fp16_device(gt_embed_path, act_.x.data_ptr, D);
    if (n > 0) {
        if (verbose)
            std::printf("[Input]  GT embeddings loaded (%zu floats → FP16): %s\n",
                        n, gt_embed_path);
    } else {
        if (verbose)
            std::printf("[Input]  GT file absent — BOS token (id=1) fallback\n");
        constexpr int BOS_TOKEN_ID = 1;
        launch_embed_lookup(
            static_cast<const float*>(embed_v_.data_ptr),
            BOS_TOKEN_ID,
            static_cast<half*>(act_.x.data_ptr),
            D, stream_);
    }
}

// ============================================================================
// inject_token — embed lookup on GPU for autoregressive decode
// inject_token — GPU 端嵌入查找，用于自回归解码
// ============================================================================
void AtomFlowEngine::inject_token(int token_id) {
    launch_embed_lookup(
        static_cast<const float*>(embed_v_.data_ptr),
        token_id,
        static_cast<half*>(act_.x.data_ptr),
        D, stream_);
}

// ============================================================================
// forward_pass — 28 Transformer layers + final_norm + lm_head + argmax
// forward_pass — 28 个 Transformer 层 + final_norm + lm_head + argmax
// ============================================================================
void AtomFlowEngine::forward_pass() {

#if ENABLE_PROFILER
    prof_.mark_decode_start(stream_);
#endif

    for (int layer = 0; layer < NL; ++layer) {
        const LayerWeights& w = lw_[layer];
#if ENABLE_VALIDATOR
        const bool is_probe = (layer == 0 || layer == 13 || layer == 27);
#endif
#if ENABLE_PROFILER
        const size_t bw_norm = SEQ * D * sizeof(half) * 2 + SEQ * D * sizeof(float);
#endif

        // ── Step 1: RMSNorm (Attention) / 注意力 RMSNorm ────────────────
        {
#if ENABLE_PROFILER
            auto _t = prof_.scoped_device("rms_norm_attn", stream_);
#endif
            launch_rms_norm(act_.x, w.input_norm, act_.x_norm, 1e-5f, stream_);
        }
#if ENABLE_PROFILER
        prof_.annotate_bandwidth("rms_norm_attn", bw_norm);
#endif
#if ENABLE_VALIDATOR
        if (is_probe) {
            char path[64], label[48];
            std::snprintf(path,  sizeof(path),  "ground_truth/gt_layer%d_norm_in.bin", layer);
            std::snprintf(label, sizeof(label), "Layer %d  Norm Out (x_norm)", layer);
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            auto r = validate_view(act_.x_norm, path, label);
            if (r.numel > 0) { v_pass += r.passed; ++v_total; }
        }
#endif

        // ── Step 2: Fused W8A16 QKV GEMV / 融合 W8A16 QKV GEMV ─────────
        {
#if ENABLE_PROFILER
            auto _t = prof_.scoped_device("qkv_gemm", stream_);
#endif
            launch_w8a16_gemv(act_.x_norm, w.qkv, w.qkv_scales,
                              act_.qkv_out, GS, stream_);
        }
#if ENABLE_PROFILER
        prof_.annotate_bandwidth("qkv_gemm",
            SEQ * D * sizeof(half)
            + (size_t)QKV_OUT * D
            + (size_t)QKV_OUT * (D / GS) * sizeof(half)
            + SEQ * QKV_OUT * sizeof(half));
#endif

        // ── Step 3: QKV split + RoPE / QKV 拆分 + RoPE ───────────────
        half* const qkv_base = static_cast<half*>(act_.qkv_out.data_ptr);
        half* const q_ptr    = qkv_base;
        half* const k_ptr    = qkv_base + Q_DIM;
        View q_view = create_contiguous_view(q_ptr, DataType::FP16, {(int)SEQ, Q_DIM});
        View k_view = create_contiguous_view(k_ptr, DataType::FP16, {(int)SEQ, KV_DIM});
        {
#if ENABLE_PROFILER
            auto _t = prof_.scoped_device("rope", stream_);
#endif
            // [EN] Pass position-offset cos/sin pointers so that
            //      rope_kernel (pos=0, seq=1) reads from current_pos_.
            // [CN] 传入位置偏移后的 cos/sin 指针，使
            //      rope_kernel (pos=0, seq=1) 读取 current_pos_ 的值。
            const int half_hd = HD / 2;
            launch_rope(q_view, k_view,
                        d_cos_ + current_pos_ * half_hd,
                        d_sin_ + current_pos_ * half_hd,
                        (int)SEQ, NKV, NH, HD, stream_);
        }

        // ── Step 3b: Write K,V into KV cache / 将 K,V 写入 KV 缓存 ─────
        {
#if ENABLE_PROFILER
            auto _t = prof_.scoped_device("kv_cache_write", stream_);
#endif
            launch_write_kv_cache(act_.qkv_out,
                                  act_.k_cache[layer], act_.v_cache[layer],
                                  current_pos_, Q_DIM, KV_DIM, stream_);
        }

        // ── Step 4: Cached GQA Attention / 带缓存的 GQA 注意力 ─────────
        {
#if ENABLE_PROFILER
            auto _t = prof_.scoped_device("cached_attn", stream_);
#endif
            // [EN] seq_len = current_pos + 1  (attend over all cached history).
            // [CN] seq_len = current_pos + 1（对所有已缓存历史进行注意力）。
            launch_cached_attention(
                q_ptr,
                static_cast<const half*>(act_.k_cache[layer].data_ptr),
                static_cast<const half*>(act_.v_cache[layer].data_ptr),
                static_cast<half*>(act_.attn_out.data_ptr),
                current_pos_ + 1, HD, NH, NKV, stream_);
        }

        // ── Step 5: Fused W8A16 O_proj / 融合 W8A16 O 投影 ─────────────
        {
#if ENABLE_PROFILER
            auto _t = prof_.scoped_device("o_proj", stream_);
#endif
            launch_w8a16_gemv(act_.attn_out, w.o_proj, w.o_proj_scales,
                              act_.ffn_out, GS, stream_);
        }
#if ENABLE_PROFILER
        prof_.annotate_bandwidth("o_proj",
            SEQ * D * sizeof(half) + (size_t)D * D
            + (size_t)D * (D / GS) * sizeof(half) + SEQ * D * sizeof(half));
#endif
#if ENABLE_VALIDATOR
        if (is_probe) {
            char path[64], label[48];
            std::snprintf(path,  sizeof(path),  "ground_truth/gt_layer%d_attn_out.bin", layer);
            std::snprintf(label, sizeof(label), "Layer %d  Attn Out (post o_proj)", layer);
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            auto r = validate_view(act_.ffn_out, path, label);
            if (r.numel > 0) { v_pass += r.passed; ++v_total; }
        }
#endif

        // ── Step 6: Residual Add 1 / 残差相加 1 ─────────────────────────
        {
#if ENABLE_PROFILER
            auto _t = prof_.scoped_device("res_add_attn", stream_);
#endif
            launch_residual_add(act_.x, act_.ffn_out, stream_);
        }

        // ── Step 7: RMSNorm (MLP) / MLP 前 RMSNorm ─────────────────────
        {
#if ENABLE_PROFILER
            auto _t = prof_.scoped_device("rms_norm_mlp", stream_);
#endif
            launch_rms_norm(act_.x, w.post_norm, act_.x_norm, 1e-5f, stream_);
        }
#if ENABLE_PROFILER
        prof_.annotate_bandwidth("rms_norm_mlp", bw_norm);
#endif

        // ── Step 8a: Fused W8A16 Gate_proj / 融合 W8A16 Gate 投影 ───────
        {
#if ENABLE_PROFILER
            auto _t = prof_.scoped_device("gate_proj", stream_);
#endif
            launch_w8a16_gemv(act_.x_norm, w.gate_proj, w.gate_scales,
                              act_.gate_out, GS, stream_);
        }
#if ENABLE_PROFILER
        prof_.annotate_bandwidth("gate_proj",
            SEQ * D * sizeof(half) + (size_t)FFN * D
            + (size_t)FFN * (D / GS) * sizeof(half) + SEQ * FFN * sizeof(half));
#endif

        // ── Step 8b: Fused W8A16 Up_proj / 融合 W8A16 Up 投影 ──────────
        {
#if ENABLE_PROFILER
            auto _t = prof_.scoped_device("up_proj", stream_);
#endif
            launch_w8a16_gemv(act_.x_norm, w.up_proj, w.up_scales,
                              act_.up_out, GS, stream_);
        }
#if ENABLE_PROFILER
        prof_.annotate_bandwidth("up_proj",
            SEQ * D * sizeof(half) + (size_t)FFN * D
            + (size_t)FFN * (D / GS) * sizeof(half) + SEQ * FFN * sizeof(half));
#endif

        // ── Step 9: SwiGLU / SwiGLU 激活 ────────────────────────────────
        {
#if ENABLE_PROFILER
            auto _t = prof_.scoped_device("swiglu", stream_);
#endif
            launch_swiglu(act_.gate_out, act_.up_out, act_.gate_out, stream_);
        }
#if ENABLE_PROFILER
        prof_.annotate_bandwidth("swiglu", SEQ * FFN * sizeof(half) * 3);
#endif

        // ── Step 10: Fused W8A16 Down_proj / 融合 W8A16 Down 投影 ───────
        {
#if ENABLE_PROFILER
            auto _t = prof_.scoped_device("down_proj", stream_);
#endif
            launch_w8a16_gemv(act_.gate_out, w.down_proj, w.down_scales,
                              act_.ffn_out, GS, stream_);
        }
#if ENABLE_PROFILER
        prof_.annotate_bandwidth("down_proj",
            SEQ * FFN * sizeof(half) + (size_t)D * FFN
            + (size_t)D * (FFN / GS) * sizeof(half) + SEQ * D * sizeof(half));
#endif
#if ENABLE_VALIDATOR
        if (is_probe) {
            char path[64], label[48];
            std::snprintf(path,  sizeof(path),  "ground_truth/gt_layer%d_mlp_out.bin", layer);
            std::snprintf(label, sizeof(label), "Layer %d  MLP Out (post down_proj)", layer);
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            auto r = validate_view(act_.ffn_out, path, label);
            if (r.numel > 0) { v_pass += r.passed; ++v_total; }
        }
#endif

        // ── Step 11: Residual Add 2 / 残差相加 2 ────────────────────────
        {
#if ENABLE_PROFILER
            auto _t = prof_.scoped_device("res_add_ffn", stream_);
#endif
            launch_residual_add(act_.x, act_.ffn_out, stream_);
        }
    } // end layer loop / 结束层循环

    // ── Final RMSNorm + lm_head (FP32) + ArgMax ─────────────────────────
    {
#if ENABLE_PROFILER
        auto _t = prof_.scoped_device("final_norm", stream_);
#endif
        launch_rms_norm(act_.x, final_norm_v_, act_.x_norm, 1e-5f, stream_);
    }
    {
#if ENABLE_PROFILER
        auto _t = prof_.scoped_device("lm_head", stream_);
#endif
        // [EN] cublasGemmEx: FP16 inputs (lm_head + x_norm), FP32 output (logits).
        //      Eliminates the FP16→FP32 cast kernel and halves lm_head VRAM.
        // [CN] cublasGemmEx：FP16 输入（lm_head + x_norm），FP32 输出（logits）。
        //      消除 FP16→FP32 转换 kernel，lm_head 显存减半。
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasGemmEx(cublas_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            V, 1, D,
            &alpha,
            lm_head_v_.data_ptr, CUDA_R_16F, D,
            act_.x_norm.data_ptr, CUDA_R_16F, D,
            &beta,
            act_.logits.data_ptr, CUDA_R_32F, V,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT));
    }
#if ENABLE_PROFILER
    prof_.mark_decode_end(stream_);
#endif
    {
#if ENABLE_PROFILER
        auto _t = prof_.scoped_device("argmax", stream_);
#endif
        launch_argmax(act_.logits, act_.d_token_id, stream_);
    }
}

// ============================================================================
// get_output_token — D2H copy of argmax result
// get_output_token — argmax 结果的 D2H 拷贝
// ============================================================================
int AtomFlowEngine::get_output_token() {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    int h_token_id = -1;
    CUDA_CHECK(cudaMemcpy(&h_token_id, act_.d_token_id,
                           sizeof(int), cudaMemcpyDeviceToHost));
    return h_token_id;
}

// ============================================================================
// Validation helpers (compiled only when ENABLE_VALIDATOR=1)
// 验证辅助函数（仅在 ENABLE_VALIDATOR=1 时编译）
// ============================================================================
#if ENABLE_VALIDATOR
void AtomFlowEngine::validate_logits() {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    std::FILE* probe = std::fopen("ground_truth/gt_logits.bin", "rb");
    if (!probe) return;
    std::fclose(probe);
    auto r = validate_view(act_.logits, "ground_truth/gt_logits.bin", "Logits");
    if (r.numel > 0) { v_pass += r.passed; ++v_total; }
}
#endif

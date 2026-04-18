#include <cuda_runtime.h>
#include <float.h>
#include "utils/utils.h"
#include "core/view.h"
#include "ops/kernel.h"

// [EN] Two-phase parallel ArgMax over a [1, vocab_size] logits vector.
//  Phase 1: Each block reduces its chunk to a (max_val, max_idx) pair in smem.
//  Phase 2: Block 0 reduces the per-block results.
// [CN] 对 [1, vocab_size] logits 向量的两阶段并行 ArgMax。
//  阶段 1：每个 block 将其分块归约为 smem 中的 (max_val, max_idx) 对。
//  阶段 2：block 0 对每个 block 的结果进行归约。
//
// [Bug/Imperfection: Two-pass approach requires a temporary device buffer for
//  per-block results. Here we use statically-sized shared memory sized for
//  BLOCK_SIZE=256. For vocab_size=128256, we need ceil(128256/256)=501 blocks;
//  the second-pass kernel assumes all 501 results fit in one block's smem.
//  If vocab grows beyond 256*256=65536, a third pass or CUB is needed.
//  两阶段方法需要临时设备缓冲区存储每个 block 的结果。对于 vocab=128256，
//  需要 501 个 block；第二阶段 kernel 假设 501 个结果能放入一个 block 的 smem。
//  如果 vocab 超过 65536，需要第三阶段或使用 CUB。]

static __global__ void argmax_phase1(
    const float* logits, int size,
    float* block_vals, int* block_idxs)
{
    extern __shared__ char smem[];
    float* s_vals = reinterpret_cast<float*>(smem);
    int*   s_idxs = reinterpret_cast<int*>(smem + blockDim.x * sizeof(float));

    int tid  = threadIdx.x;
    int gid  = blockIdx.x * blockDim.x + tid;

    float my_val = (gid < size) ? logits[gid] : -FLT_MAX;
    int   my_idx = (gid < size) ? gid         : -1;

    // Thread-local max across strided elements (handles size > gridDim*BLOCK)
    // 线程局部最大值（处理 size > gridDim*BLOCK 的情况）
    for (int i = gid + gridDim.x * blockDim.x; i < size; i += gridDim.x * blockDim.x) {
        if (logits[i] > my_val) { my_val = logits[i]; my_idx = i; }
    }

    s_vals[tid] = my_val;
    s_idxs[tid] = my_idx;
    __syncthreads();

    // Parallel tree reduction within block
    // block 内并行树形归约
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && s_vals[tid + stride] > s_vals[tid]) {
            s_vals[tid] = s_vals[tid + stride];
            s_idxs[tid] = s_idxs[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_vals[blockIdx.x] = s_vals[0];
        block_idxs[blockIdx.x] = s_idxs[0];
    }
}

static __global__ void argmax_phase2(
    const float* block_vals, const int* block_idxs,
    int n_blocks, int* out_token_id)
{
    extern __shared__ char smem[];
    float* s_vals = reinterpret_cast<float*>(smem);
    int*   s_idxs = reinterpret_cast<int*>(smem + blockDim.x * sizeof(float));

    int tid = threadIdx.x;
    s_vals[tid] = (tid < n_blocks) ? block_vals[tid] : -FLT_MAX;
    s_idxs[tid] = (tid < n_blocks) ? block_idxs[tid] : -1;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && s_vals[tid + stride] > s_vals[tid]) {
            s_vals[tid] = s_vals[tid + stride];
            s_idxs[tid] = s_idxs[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) *out_token_id = s_idxs[0];
}

// Persistent scratch buffer — allocated once, reused across calls.
// 持久暂存缓冲区 —— 分配一次，跨调用复用。
static float* g_block_vals = nullptr;
static int*   g_block_idxs = nullptr;
static int    g_n_blocks    = 0;

void launch_argmax(const View& logits, int* out_token_id, cudaStream_t stream) {
    const int   V     = logits.dims[logits.num_dims - 1];  // vocab_size
    constexpr int BLOCK = 256;
    const int   GRID  = (V + BLOCK - 1) / BLOCK;

    // Lazy alloc scratch / 惰性分配暂存空间
    if (g_n_blocks < GRID) {
        if (g_block_vals) CUDA_CHECK(cudaFree(g_block_vals));
        if (g_block_idxs) CUDA_CHECK(cudaFree(g_block_idxs));
        CUDA_CHECK(cudaMalloc(&g_block_vals, GRID * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g_block_idxs, GRID * sizeof(int)));
        g_n_blocks = GRID;
    }

    const size_t smem = BLOCK * (sizeof(float) + sizeof(int));

    argmax_phase1<<<GRID, BLOCK, smem, stream>>>(
        static_cast<const float*>(logits.data_ptr),
        V, g_block_vals, g_block_idxs);
    CUDA_CHECK_LAST();

    // Phase 2 must fit in one block: GRID <= BLOCK (satisfied for vocab<=65536)
    // 阶段 2 必须在一个 block 内完成：GRID <= BLOCK
    const int P2_BLOCK = (GRID <= BLOCK) ? BLOCK : BLOCK;  // always BLOCK
    argmax_phase2<<<1, P2_BLOCK, smem, stream>>>(
        g_block_vals, g_block_idxs, GRID, out_token_id);
    CUDA_CHECK_LAST();
}

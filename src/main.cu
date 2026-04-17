#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include "view.h"
#include "kernel.h"

// [EN] Minimal main skeleton. It performs no real inference yet; its purpose
//      is to anchor the link step so the whole project compiles end-to-end.
//      As kernel launchers come online, wire them in here one by one.
// [CN] 最小 main 骨架。目前不做真实推理，仅用于锚定链接阶段，让整个工程
//      端到端能编译通过。kernel launcher 逐步上线时，在这里一条条接入。
int main() {
    // ---- 1. Dummy model dimensions (Llama 3.2 3B) ----
    constexpr int seq_len    = 16;
    constexpr int hidden_dim = 3072;
    constexpr size_t n_elems = static_cast<size_t>(seq_len) * hidden_dim;
    constexpr size_t n_bytes = n_elems * sizeof(half);

    // ---- 2. Allocate a single chunk of device memory ----
    void* d_ptr = nullptr;
    cudaError_t err = cudaMalloc(&d_ptr, n_bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // ---- 3. Build a contiguous View over it ----
    View x = create_contiguous_view(d_ptr, DataType::FP16, {seq_len, hidden_dim});

    // ---- 4. Demonstrate that the raw pointer is reachable ----
    half* base = static_cast<half*>(x.data_ptr);
    std::printf("AtomFlow skeleton: View[%d, %d] base=%p strides=[%d, %d]\n",
                x.dims[0], x.dims[1], (void*)base, x.strides[0], x.strides[1]);

    // ---- 5. cuBLAS / CUDA stream hooks (ready for the first kernel call) ----
    cublasHandle_t cublas;
    cublasCreate(&cublas);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasSetStream(cublas, stream);

    // TODO: launch_rms_norm(x, ...);
    // TODO: launch_linear_gemm(x, qkv_weight, qkv_out, cublas, stream);
    // TODO: launch_rope(...);
    // TODO: launch_tiled_attention_kernel(...);

    // ---- 6. Cleanup ----
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cublasDestroy(cublas);
    cudaFree(d_ptr);

    return EXIT_SUCCESS;
}

struct AtomContext {
    ModelConfig config;

    MemoryPlanner planner;
    BlockManager block_manager;

    cudaStream_t stream;
    cublasHandle_t cublas_handle;

    AtomContext(const ModelConfig& cfg, void* arena_raw, size_t arena_size) 
        : config(cfg), planner(arena_raw, arena_size) 
    {
        cudaStreamCreate(&stream);
        cublasCreate(&cublas_handle);
        cublasSetStream(cublas_handle, stream);
    }

    ~AtomContext() {
        cublasDestroy(cublas_handle);
        cudaStreamDestroy(stream);
    }
};
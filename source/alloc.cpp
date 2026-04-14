#include "alloc.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>


ArenaAllocator::ArenaAllocator(size_t total_size) {

    capacity_ = align_to_128(total_size);
    offset_ = 0; 

    cudaError_t err = cudaMalloc(&base_ptr_, capacity_);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: Failed to allocate " 
                  << capacity_ / (1024 * 1024) << " MB. "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE); // Fatal error, exit immediately
    }
}

ArenaAllocator::~ArenaAllocator() {
    if(base_ptr_)
        cudaFree(base_ptr_); //RAII
}

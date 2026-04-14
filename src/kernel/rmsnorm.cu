#include "operators.h"


__global__ void rms_norm_kernel() {
    
}

void launch_rms_norm(const View& input, const View& weight, View& output, float eps, cudaStream_t stream) {
    switch (input.dtype) {
        case DataType::FP16:
            rms_norm_kernel<<<1, 1, 0, stream>>>(input, weight, output, eps);
            break;
        case DataType::FP8_E4M3:
            rms_norm_kernel<<<1, 1, 0, stream>>>(input, weight, output, eps);
            break;
        default:
            throw std::runtime_error("Unsupported data type");
    }
}
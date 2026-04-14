#pragma once
#include <cstdint>

enum class DataType {
    FP32,
    FP16,
    BF16,
    FP8_E4M3
};

struct View {
    DataType dtype;
    void* data_ptr;

    int32_t dims[4];// [Batch, Head, Seq, Dim]
    int32_t strides[4];// [Batch, Head, Seq, Dim]
    int8_t num_dims;//real dims that are used

    __device__ __host__ size_t offset(int d0) const { return d0 * strides[0]; }
    __device__ __host__ size_t offset(int d0, int d1) const { return d0 * strides[0] + d1 * strides[1]; }
    __device__ __host__ size_t offset(int d0, int d1, int d2) const { return d0 * strides[0] + d1 * strides[1] + d2 * strides[2]; }
    __device__ __host__ size_t offset(int d0, int d1, int d2, int d3) const { return d0 * strides[0] + d1 * strides[1] + d2 * strides[2] + d3 * strides[3]; }

    template<typename T>
    __device__ __host__ T* const at_offset(size_t off) const { return static_cast<T*>(data_ptr) + off; }  
    template<typename T, typename... Args>
    __device__ __host__ T* at(Args... args) const { return at_offset<T>(offset(args...)); }
};

inline View create_contiguous_view(void* ptr, DataType type, std::initializer_list<int> d) {
    View v;
    v.data_ptr = ptr;
    v.dtype = type;
    v.num_dims = d.size();
    
    int i = 0;
    for (int dim : d) v.dims[i++] = dim;

    // Calculate strides
    v.strides[v.num_dims - 1] = 1;
    for (int j = v.num_dims - 2; j >= 0; --j) {
        v.strides[j] = v.strides[j + 1] * v.dims[j + 1];
    }
    return v;
}
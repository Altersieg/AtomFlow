#include "view.h"
// [EN] A structure representing the logically split Q, K, and V views sharing the same physical memory.
// [CN] 表示逻辑上分离但共享同一块物理内存的Q、K、V视图结构体。
// [Bug/Imperfection: Hardcoded metadata instantiation. Does not dynamically support varying GQA (Grouped-Query Attention) ratios easily. 硬编码的元数据实例化。无法轻易动态支持不同的分组查询注意力(GQA)比例。]
struct QKVViews {
    View q;
    View k;
    View v;
};

// [EN] A utility to slice the fused QKV view into three separate views with correct strides and offsets.
// [CN] 将融合的QKV视图切片为三个独立的视图，并赋予正确的跨步和偏移量的工具函数。
// [Bug/Imperfection: Modifying the view properties manually bypasses safety checks; a proper stride calculation system inside the View class would be more robust. 手动修改视图属性绕过了安全检查；在View类内部实现完善的跨步计算系统会更健壮。]
QKVViews slice_qkv_view(const View& fused_qkv, int q_dim, int k_dim, int v_dim) {
    QKVViews split;
    int total_stride = q_dim + k_dim + v_dim;

    // Q View: 基指针不变，但我们告诉它列维度变了，并且跨步变成了 total_stride
    split.q = fused_qkv;
    split.q.dims[1] = q_dim;
    // 假设你的 View 类里加了一个 stride 属性
    split.q.stride = total_stride; 

    // K View: 物理指针向前偏移 q_dim
    split.k = fused_qkv;
    split.k.data_ptr_ = static_cast<half*>(fused_qkv.data_ptr()) + q_dim;
    split.k.dims[1] = k_dim;
    split.k.stride = total_stride;

    // V View: 物理指针向前偏移 q_dim + k_dim
    split.v = fused_qkv;
    split.v.data_ptr_ = static_cast<half*>(fused_qkv.data_ptr()) + q_dim + k_dim;
    split.v.dims[1] = v_dim;
    split.v.stride = total_stride;

    return split;
}
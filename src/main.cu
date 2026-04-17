


// [EN] Calculate logical base pointers for Q, K, and V from the fused GEMM output.
// [CN] 从融合的GEMM输出中计算Q、K和V的逻辑基指针。
// [Bug/Imperfection: Direct static_cast of data_ptr bypasses type safety checks in the View class. 直接对data_ptr进行static_cast绕过了View类中的类型安全检查。]
half* qkv_base_ptr = static_cast<half*>(qkv_output.data_ptr());

int q_dim = 3072; // 24 heads * 128
int k_dim = 1024; // 8 heads * 128
int v_dim = 1024; // 8 heads * 128
int qkv_stride = q_dim + k_dim + v_dim; // 5120 (Total dimension per token)

// 拿到第 0 个 Token 的绝对起点
half* q_base = qkv_base_ptr;
half* k_base = qkv_base_ptr + q_dim;
half* v_base = qkv_base_ptr + q_dim + k_dim;
# AtomFlow (AF) 引擎

**🌐 语言切换：** [English](./README.md) | **简体中文**

---

一个从零实现、面向 **NVIDIA Ada / Blackwell** GPU 的极简 LLM 推理引擎，
目标是在几千行可读的 C++/CUDA 代码里，把现代推理栈的底层物理机制
（FP8 张量核心、Arena 内存分配、分页 KV 缓存、RadixAttention）**掀开来**给人看。

## 构建与运行

**环境要求**

- CUDA Toolkit ≥ 12.4
- CMake ≥ 3.24
- GCC ≥ 11（C++17）
- GPU：Ada Lovelace（`sm_89`）或 Blackwell（`sm_120`）

**一键跑通**

```bash
git clone https://github.com/altersieg/AtomFlow.git
cd AtomFlow
cmake -S . -B build
cmake --build build -j
./build/atomflow
```

**两档构建模式**

| 模式            | 命令                                                              | 适用场景                               |
| --------------- | ----------------------------------------------------------------- | -------------------------------------- |
| MVP（默认）     | `cmake -S . -B build`                                             | 快速迭代、分离编译友好                  |
| 极致性能        | `cmake -S . -B build -DATOMFLOW_MONOLITHIC_BUILD=ON`              | LTO + 跨文件激进内联                    |

## 项目结构

```
AtomFlow/
├── CMakeLists.txt           # CMake 配置：sm_89，MVP / 单体构建切换
├── include/
│   ├── view.h               # POD 张量描述符（dims, strides, dtype）
│   ├── alloc.h              # ArenaAllocator：一次性 cudaMalloc 大池
│   ├── memory_planner.h     # 栈式激活内存分配器
│   ├── atom_context.h       # 全局运行时上下文（模型 + arena + stream）
│   ├── kernel.h             # Kernel launcher 声明
│   ├── qkvview.h            # 零拷贝 QKV 切片
│   ├── config.h             # ModelConfig
│   ├── block_manager.h      # KV 缓存 block 分配器（进行中）
│   └── operators.h          # 算子级辅助函数
└── src/
    ├── main.cu              # 入口 / 骨架驱动
    └── kernel/
        ├── rmsnorm.cu       # FP16 / FP8 RMSNorm，使用 CUB 块内规约
        ├── qkv_gemm.cu      # FP8 权重 × FP16 激活，走 cuBLAS
        ├── rope.cu          # 旋转位置编码（原地修改）
        ├── tiled_attention.cu  # 共享内存分块注意力（FlashAttn 简化版）
        ├── residual_add.cu  # 逐元素残差相加
        ├── swiglu.cu        # SwiGLU 激活
        └── layer.cu         # 单层 Transformer 前向调度
```

## 路线图

- [x] `ArenaAllocator` + `MemoryPlanner`
- [x] `View` + 零拷贝 QKV 切片
- [x] RMSNorm kernel（FP16 / FP8）
- [x] QKV 融合 GEMM（cuBLAS，FP8 权重）
- [x] RoPE kernel
- [x] CMake 构建系统（MVP + 单体/LTO 两档）
- [ ] 分块注意力 kernel（共享内存分块已完成；Tensor Core `mma` 指令待接）
- [ ] SwiGLU + residual_add 的 launcher 外壳
- [ ] 完整单层前向（`layer_forward_naive`）
- [ ] 端到端 Llama 3.2 3B 推理
- [ ] RadixAttention KV 缓存共享（原子引用计数 + 写时复制）
- [ ] 结构化解码 FSM 融合进 softmax
- [ ] Python 绑定（PyBind11）
- [ ] NCCL 多卡通信

## 设计哲学

- **极简基础设施** —— 基于 POD 的 `View` 系统，零开销张量管理。
- **酷炫算子** —— 激进使用 FP8、向量化内存访问、CUB 原语。
- **极简调度** —— 逻辑 View 直接映射到硬件 grid，无多余抽象层。

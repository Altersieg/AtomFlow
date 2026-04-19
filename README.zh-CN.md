# AtomFlow (AF) 引擎

**🌐 语言切换：** [English](./README.md) | **简体中文**

---

一个从零实现、面向 **NVIDIA Ada / Blackwell** GPU 的极简 LLM 推理引擎，
目标是在几千行可读的 C++/CUDA 代码里，把现代推理栈的底层物理机制
（FP8 量化、Arena 内存分配、融合 GEMV 算子）**掀开来**给人看。

**当前状态：** 端到端 **Llama 3.2 3B** 单 token 解码 **~58 tok/s**
（纯 GPU 计时，W8A16 融合 GEMV，RTX 5060 Ti 16 GB）。

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
./build/atomflow                    # 纯基准测试模式（ENABLE_VALIDATOR=0）
```

**验证模式**（逐层与 HuggingFace 基准真值对比）：

```bash
cmake -S . -B build -DCMAKE_CUDA_FLAGS="-DENABLE_VALIDATOR=1 -DENABLE_PROFILER=1"
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
├── CMakeLists.txt              # CMake 配置：sm_89，MVP / 单体构建切换
├── include/
│   ├── core/
│   │   ├── engine.h            # AtomFlowEngine 类（初始化、前向、清理）
│   │   ├── model_state.h       # LayerWeights 与 ActBuffers POD 结构体
│   │   ├── view.h              # POD 张量描述符（dims, strides, dtype）
│   │   ├── atom_context.h      # 全局运行时上下文
│   │   ├── qkvview.h           # 零拷贝 QKV 切片
│   │   └── config.h            # ModelConfig
│   ├── ops/
│   │   └── kernel.h            # 所有 kernel launcher 声明
│   ├── memory/
│   │   └── weight_loader.h     # 基于 mmap 的零拷贝权重加载
│   └── utils/
│       ├── utils.h             # CUDA_CHECK / CUBLAS_CHECK 宏
│       ├── profiler.h          # EngineProfiler（延迟 cudaEvent 计时）
│       └── validator.h         # 基准真值余弦相似度验证
├── src/
│   ├── main.cu                 # 轻量级 CLI + 基准测试工具（约 100 行）
│   ├── core/
│   │   └── engine.cu           # AtomFlowEngine 实现
│   ├── kernel/
│   │   ├── helpers.cu          # 嵌入查找、FP16↔FP32 转换、RoPE 缓存、文件 I/O
│   │   ├── qkv_gemm.cu         # 融合 W8A16 GEMV 算子 + cuBLAS GEMM 后备
│   │   ├── rmsnorm.cu          # FP16 RMSNorm
│   │   ├── rope.cu             # 旋转位置编码（原地修改）
│   │   ├── tiled_attention.cu  # 共享内存分块注意力
│   │   ├── residual_add.cu     # 逐元素残差相加
│   │   ├── swiglu.cu           # SwiGLU 激活
│   │   ├── argmax.cu           # 贪心 ArgMax 下一 token 采样
│   │   ├── dequant.cu          # FP8→FP16 反量化（遗留，优先使用融合路径）
│   │   └── layer.cu            # 层级辅助函数
│   └── utils/
│       ├── profiler.cpp        # EngineProfiler 实现
│       └── validator.cpp       # 基准真值对比工具
└── tools/
    ├── export_atomflow.py      # 导出 HF Llama 3.2 → AtomFlow .bin（AWQ + FP8）
    └── dump_ground_truth.py    # 导出逐层激活用于验证
```

## 路线图

- [x] `View` + 零拷贝 QKV 切片
- [x] RMSNorm、RoPE、SwiGLU、residual_add 算子
- [x] 分块注意力 kernel（共享内存）
- [x] FP8 权重导出 + AWQ 平滑（`export_atomflow.py`）
- [x] 端到端 Llama 3.2 3B 单 token 解码（28 层 + lm_head）
- [x] **融合 W8A16 GEMV 算子**（FP8 反量化 + 内积一次完成，零中间写回）
- [x] 模块化 `AtomFlowEngine` 类（初始化 / 前向 / 清理）
- [x] 纯 GPU 基准测试模式（`ENABLE_VALIDATOR=0`，`ENABLE_PROFILER=0`）
- [x] 逐层基准真值验证（37/37 PASS）
- [ ] KV 缓存（支持多 token 生成）
- [ ] CUDA Graphs 加速解码循环
- [ ] RadixAttention KV 缓存共享
- [ ] Tensor Core `mma` 注意力路径
- [ ] Python 绑定（PyBind11）
- [ ] NCCL 多卡通信

## 设计哲学

- **极简基础设施** —— 基于 POD 的 `View` 系统，零开销张量管理。
- **酷炫算子** —— 融合 W8A16 GEMV：向量化 FP8 加载、warp shuffle 规约、零中间全局写回。
- **极简调度** —— 逻辑 View 直接映射到硬件 grid，无框架开销。
- **工业级模块化** —— `AtomFlowEngine` 封装全部 GPU 状态；`main.cu` 仅约 100 行薄壳。

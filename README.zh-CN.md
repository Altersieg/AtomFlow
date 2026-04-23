# AtomFlow

**面向消费级 GPU 的裸金属零开销 LLM 推理引擎**

**🌐 语言切换：** [English](./README.md) | **简体中文**

---

## 摘要

AtomFlow 是一个从零开始的 CUDA C++ 推理引擎，在单张 **NVIDIA RTX 5060 Ti**（16 GB GDDR7，448 GB/s，Blackwell 架构）上实现了 **Llama 3.2 3B** 自回归解码 **68.7 tok/s**。

该引擎面向 $M=1$（单 token）解码的物理现实：在算术强度 < 1 时，性能**完全受显存带宽约束**。每一个架构决策——静态 arena、融合 W8A16 GEMV、连续 KV 缓存——都是该约束的直接结果。无动态分配、无间接寻址、无框架。

| 指标 | 数值 |
|------|------|
| 模型 | Llama 3.2 3B · 28 层 · GQA 24Q/8KV · V=128K |
| 硬件 | RTX 5060 Ti · 16 GB GDDR7 · 448 GB/s |
| TPOT (CUDA Graph) | 13.3–15.2 ms → **66–75 tok/s** |
| AR 解码 (KV 缓存) | **65–69 tok/s** |
| 总 VRAM 占用 | 5.5 GiB（16 GB 的 34%） |
| W8A16 GEMV 峰值带宽 | 233 GB/s（理论峰值的 52%） |
| 运行时 cudaMalloc | **0 次** |
| 数值精度 | 37/37 逐层探针 PASS |

### 对标 vLLM 基准测试

在同一张 **RTX 5060 Ti** 单卡环境下，针对 Llama 3.2 3B 的 $M=1$ 解码任务，与业界标杆 **vLLM**（开启全量编译优化：`torch.compile` + CUDA Graph 捕获）进行头对头对比。

| 引擎 | TPOT (ms) | 说明 |
|------|-----------|------|
| **AtomFlow** | **~14.5** | 静态连续 KV、融合 W8A16 GEMV、CUDA Graph |
| vLLM（优化后） | ~16.1 | `torch.compile` + CUDA Graph + PagedAttention |
| **领先幅度** | **−11.5%** | AtomFlow 单步延迟占优 |

**结论解读。** 在单流（$M=1$）、非并发场景下，KV 缓存的连续算术寻址相对于分页间接寻址具有绝对的访存优势：无页表查找、无 TLB 压力、无散列加载。11.5% 的延迟优势即是 PagedAttention 在无并发需求时为"灵活性"所付出的经验代价。PagedAttention 在多请求批处理场景下仍为最优解；AtomFlow 所针对的是设计空间的另一极端。

---

## 架构一：AtomFlow Core — 极致性能区

生产级单流引擎。零抽象开销。每字节显存在编译时可溯源。

### 核心架构决策

- **静态 Arena 分配器**
  - 初始化时 4 次 `cudaMalloc`，推理期间 0 次。
  - 权重池（5.17 GiB）+ 激活 Arena（1.5 MiB）+ KV 缓存池（224 MiB）+ cuBLAS 工作区（4 MiB）。
  - 所有 View 为 POD 描述符，指向预分配 arena——无堆分配、无 `new`、无 `shared_ptr`。

- **IO 感知 Kernel 融合**
  - 融合 W8A16 GEMV：FP8-E4M3 反量化 + FP16 内积在单次 kernel 中完成。
  - 反量化中间值仅存于寄存器——**零全局显存写回**。
  - 激活 arena 从 49.5 MiB（含显式反量化工作区）缩减至 **1.5 MiB**。

- **静态 KV 缓存**
  - 每层连续 `[MAX_SEQ_LEN, KV_DIM]`——纯算术寻址：`cache[t × stride + head × HD + d]`。
  - 无页表、无间接加载、无散列块引起的 TLB 压力。
  - 自研 GQA 注意力 kernel：树形归约 Q·Kᵀ + 共享内存 softmax + 并行 V 累加。

- **CUDA Graph 捕获**
  - 280+ 次 kernel 启动录制为单一 `cudaGraphExec_t`。
  - CPU 分发开销完全消除——实测较 eager 加速 1.04–1.06×。

- **mmap 权重加载**
  - 二进制模型文件（`4.87 GiB`）通过 `mmap` 映射 → 单次批量 `cudaMemcpy` 到设备池。
  - 权重零主机端内存分配。

---

## 架构二：Server Sandbox — 实验区

一个**独立的 CMake 构建目标**（`atomflow-server-test`），用于评估服务器级调度开销。

**目的：**

提供一个无头仿真环境，用于量化以下物理成本：

- **分页 KV 缓存** — 非连续物理块 + 页表间接寻址。
- **块管理器** — 基于空闲列表的分配/回收，支持写时复制引用计数。
- **连续批处理** — 可变序列长度的动态请求调度。

**设计约束：**

沙箱仅与核心引擎共享底层数学 kernel（`src/kernel/*`）。它不链接 `engine.cu` 或 `main.cu`。两者之间无继承关系——它们是物理上独立的编译目标。

```bash
cmake --build build -j --target atomflow-server-test
./build/atomflow-server-test
```

---

## 构建与运行

**环境要求**

- CUDA Toolkit ≥ 12.4
- CMake ≥ 3.24
- GCC ≥ 11（C++17）
- GPU：Blackwell（`sm_120`）

**快速开始**

```bash
git clone https://github.com/altersieg/AtomFlow.git
cd AtomFlow
cmake -S . -B build
cmake --build build -j

./build/atomflow                # 基准测试 + 自回归生成
python tools/decode_tokens.py   # 将 token ID 解码为文本
```

**验证模式**（逐层对比 HuggingFace 基准真值）：

```bash
cmake -S . -B build -DCMAKE_CUDA_FLAGS="-DENABLE_VALIDATOR=1"
cmake --build build -j
./build/atomflow    # 期望输出：37/37 PASS，token 791
```

**构建模式**

| 模式 | 命令 | 说明 |
|------|------|------|
| MVP（默认） | `cmake -S . -B build` | 分离编译，快速迭代 |
| 极致性能 | `cmake -S . -B build -DATOMFLOW_MONOLITHIC_BUILD=ON` | LTO + 跨文件激进内联 |
| 含性能分析 | `-DCMAKE_CUDA_FLAGS="-DENABLE_PROFILER=1"` | 逐 kernel cudaEvent 计时 |

---

## 项目结构

`src/kernel/*` 是核心引擎与服务器沙箱之间的**共享边界**：两个 CMake 目标（`atomflow` 与 `atomflow-server-test`）链接同一套底层数学 kernel，除此之外不共享任何代码。

```
AtomFlow/
├── CMakeLists.txt                  # 双目标构建：atomflow + atomflow-server-test
├── pyproject.toml                  # PEP 621 元数据（setuptools_scm 动态版本）
├── setup.py                        # torch.utils.cpp_extension CUDA 构建钩子
├── run_inference.sh                # 端到端便捷运行脚本
├── include/
│   ├── core/
│   │   ├── engine.h                # AtomFlowEngine（初始化、前向传播、KV 缓存）
│   │   ├── model_state.h           # LayerWeights 与 ActBuffers（POD 结构体）
│   │   ├── view.h                  # POD 张量描述符（dims, strides, dtype）
│   │   ├── qkvview.h               # qkv_out 上的零拷贝 QKV 切片
│   │   ├── atom_context.h          # 全局运行时上下文（handle、stream）
│   │   └── config.h                # 编译期 ModelConfig 常量
│   ├── ops/
│   │   └── kernel.h                # 所有 kernel launcher 声明
│   ├── memory/
│   │   ├── weight_loader.h         # 基于 mmap 的权重加载
│   │   ├── alloc.h                 # Arena bump-allocator 辅助
│   │   ├── memory_planner.h        # 静态 arena 尺寸规划
│   │   └── block_manager.h         # 沙箱块管理器占位
│   └── utils/
│       ├── utils.h                 # CUDA_CHECK / CUBLAS_CHECK 宏
│       ├── profiler.h              # EngineProfiler（延迟 cudaEvent 计时）
│       └── validator.h             # 基准真值验证（余弦相似度）
├── src/
│   ├── main.cu                     # CLI + 基准测试 + 自回归循环 + token 日志
│   ├── core/
│   │   └── engine.cu               # AtomFlowEngine 完整实现
│   ├── kernel/                     # ← 共享边界：两个目标均链接此目录
│   │   ├── helpers.cu              # 嵌入查找、RoPE 缓存、KV 缓存写入
│   │   ├── qkv_gemm.cu             # 融合 W8A16 GEMV + cuBLAS lm_head GEMM
│   │   ├── tiled_attention.cu      # GQA 缓存注意力（M=1 优化）
│   │   ├── rmsnorm.cu              # FP16 RMSNorm
│   │   ├── rope.cu                 # 旋转位置编码（原地修改）
│   │   ├── residual_add.cu         # 逐元素残差相加
│   │   ├── swiglu.cu               # SwiGLU 激活
│   │   ├── argmax.cu               # 贪心 ArgMax 采样
│   │   └── layer.cu                # 层级辅助函数
│   ├── experimental/
│   │   └── server_arch/            # 沙箱源文件（目标：atomflow-server-test）
│   │       ├── server_engine.h     # ServerEngine 类（沙箱）
│   │       ├── server_engine.cu    # atomflow-server-test 入口点
│   │       └── paged_kv_manager.h  # KVBlock、PageTable、BlockManager 占位
│   └── utils/
│       ├── profiler.cpp            # EngineProfiler 实现
│       └── validator.cpp           # 基准真值对比
├── tools/
│   ├── export_atomflow.py          # HF Llama 3.2 → AtomFlow .bin（AWQ + FP8 GS=128）
│   ├── dump_ground_truth.py        # 导出逐层激活用于验证器
│   └── decode_tokens.py            # 离线 token → 文本解码器（本地 tokenizer）
├── models/                         # 本地模型权重与 tokenizer 文件
└── ground_truth/                   # 验证器使用的逐层 FP32 激活转储
```

---

## 路线图

路线图严格限定在”已完成“与“未来3 个月”两个区间内；长远愿望清单已移除，以保持路线图的可信性。

### 已完成

- [x] 静态 arena 分配器（权重 + 激活 + KV 缓存池）
- [x] 融合 W8A16 GEMV（零中间 DRAM 写回，峰值 233 GB/s）
- [x] 静态 KV 缓存 + GQA 缓存注意力（自回归解码）
- [x] CUDA Graph 捕获（280+ kernel → 单次图分发）
- [x] 逐层数值验证（37/37 PASS）
- [x] Token 日志 + 离线 Python 解码器
- [x] 独立服务器沙箱 CMake 目标（`atomflow-server-test`）
- [x] ServerEngine 脚手架 + BlockManager / PageTable 占位

### 未来3 个月

- [ ] **沙箱内的 PagedAttention kernel** — 与静态连续 KV kernel 进行量化延迟对比（**对比本身**即为交付物）
- [ ] **更多前後缀算子融合** — 如 LayerNorm + MatMul、SwiGLU + down-projection；持续评估寄存器压力抵消融合收益的临界点

---

## 设计哲学

- **物理优先** — 每个决策源自硬件约束（448 GB/s 带宽、16 GB 显存、$M=1$ 算术强度 < 1）。例如，由于 $M=1$ 解码受带宽约束，任何会引入额外 DRAM 往返的抽象默认被拒绝。
- **零运行时分配** — 所有内存在初始化时静态分区；推理期间为纯 kernel 分发。
- **融合优先（Fusion-First）** — 默认采用 kernel 融合；当两个 kernel 共享数据时，举证责任在于”保持分离“一方。当前已融合：W8A16 GEMV（FP8 反量化 + FP16 内积仅在寄存器）、自研 GQA 注意力 kernel（Q·Kᵀ + softmax + V 累加仅在共享内存）。持续工作聚焦于”激进融合仍有收益“与”寄存器压力抵消收益“的临界点。
- **测量而非假设** — 内置性能分析器（`ENABLE_PROFILER=1`）和验证器（`ENABLE_VALIDATOR=1`）支撑每一项指标。
- **双架构（建设中）** — 生产引擎优先 $M=1$ 单流解码；并行的服务器沙箱目标正在构建，用于量化分页/批处理相对静态连续布局的开销。最终交付物就是这份对比本身，而非一套生产级服务系统。

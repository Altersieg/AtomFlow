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
- GPU：Ada Lovelace（`sm_89`）或 Blackwell（`sm_120`）

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

```
AtomFlow/
├── CMakeLists.txt                  # 双目标构建：atomflow + atomflow-server-test
├── include/
│   ├── core/
│   │   ├── engine.h                # AtomFlowEngine（初始化、前向传播、KV 缓存）
│   │   ├── model_state.h           # LayerWeights 与 ActBuffers（POD 结构体）
│   │   └── view.h                  # POD 张量描述符（dims, strides, dtype）
│   ├── ops/
│   │   └── kernel.h                # 所有 kernel launcher 声明
│   ├── memory/
│   │   └── weight_loader.h         # 基于 mmap 的权重加载
│   └── utils/
│       ├── utils.h                 # CUDA_CHECK / CUBLAS_CHECK 宏
│       ├── profiler.h              # EngineProfiler（延迟 cudaEvent 计时）
│       └── validator.h             # 基准真值验证（余弦相似度）
├── src/
│   ├── main.cu                     # CLI + 基准测试 + 自回归循环 + token 日志
│   ├── core/
│   │   └── engine.cu               # AtomFlowEngine 完整实现
│   ├── kernel/
│   │   ├── helpers.cu              # 嵌入查找、RoPE 缓存、KV 缓存写入
│   │   ├── qkv_gemm.cu            # 融合 W8A16 GEMV + cuBLAS lm_head GEMM
│   │   ├── tiled_attention.cu     # GQA 缓存注意力（M=1 优化）
│   │   ├── rmsnorm.cu             # FP16 RMSNorm
│   │   ├── rope.cu                # 旋转位置编码（原地修改）
│   │   ├── residual_add.cu        # 逐元素残差相加
│   │   ├── swiglu.cu              # SwiGLU 激活
│   │   └── argmax.cu              # 贪心 ArgMax 采样
│   ├── experimental/
│   │   └── server_arch/
│   │       ├── server_engine.h     # ServerEngine 类（沙箱）
│   │       ├── server_engine.cu    # atomflow-server-test 入口点
│   │       └── paged_kv_manager.h  # KVBlock、PageTable、BlockManager 占位
│   └── utils/
│       ├── profiler.cpp            # EngineProfiler 实现
│       └── validator.cpp           # 基准真值对比
└── tools/
    ├── export_atomflow.py          # HF Llama 3.2 → AtomFlow .bin（AWQ + FP8 GS=128）
    ├── dump_ground_truth.py        # 导出逐层激活用于验证器
    └── decode_tokens.py            # 离线 token → 文本解码器（本地 tokenizer）
```

---

## 路线图

### 核心引擎

- [x] 静态 arena 分配器（权重 + 激活 + KV 缓存池）
- [x] 融合 W8A16 GEMV（零中间 DRAM 写回，峰值 233 GB/s）
- [x] 静态 KV 缓存 + GQA 缓存注意力（自回归解码）
- [x] CUDA Graph 捕获（280+ kernel → 单次图分发）
- [x] 逐层数值验证（37/37 PASS）
- [x] Token 日志 + 离线 Python 解码器
- [ ] **Prefill 阶段优化** — 基于 GEMM 的批量 token 处理以降低 TTFT
- [ ] Tensor Core `mma.sync` 注意力路径（长上下文解码）
- [ ] Top-K / Top-P 采样 + 温度缩放
- [ ] Python 绑定（PyBind11）
- [ ] NCCL 多卡通信

### 服务器沙箱

- [x] 独立 CMake 目标，共享数学 kernel
- [x] ServerEngine 脚手架 + BlockManager/PageTable 占位
- [ ] **PagedAttention kernel** — 对比静态连续 KV 的精确延迟差值
- [ ] 连续批处理调度器 + 多请求队列
- [ ] 量化对比：页表查找开销（ns）vs. 线性寻址

---

## 设计哲学

- **物理优先** — 每个决策源自硬件约束（448 GB/s 带宽、16 GB 显存、$M=1$ 算术强度 < 1）。
- **零运行时分配** — 所有内存在初始化时静态分区；推理期间为纯 kernel 分发。
- **融合一切** — 若两个 kernel 共享数据，则它们必须合为一个 kernel。无中间 DRAM 往返。
- **测量而非假设** — 内置性能分析器（`ENABLE_PROFILER=1`）和验证器（`ENABLE_VALIDATOR=1`）支撑每一项指标。
- **双架构** — 生产引擎针对当前最快路径优化（静态、连续、单流）。沙箱探索未来可能需要的能力（分页、批处理、调度）。

#pragma once
// ============================================================================
// paged_kv_manager.h — PagedAttention KV cache block management (stub)
// paged_kv_manager.h — PagedAttention KV 缓存块管理（占位）
//
// [EN] Defines the data structures for a true paged KV cache:
//      - KVBlock:      fixed-size physical memory slab (one block of tokens).
//      - PageTable:    per-sequence logical→physical block mapping.
//      - BlockManager: free-list allocator that hands out / reclaims KVBlocks.
//
//      These are EMPTY scaffolding. The implementation will be filled in once
//      the server scheduling loop in ServerEngine is functional.
//
// [CN] 定义真正的分页 KV 缓存所需的数据结构：
//      - KVBlock:      固定大小的物理内存块（一个 token 块）。
//      - PageTable:    每序列的逻辑→物理块映射。
//      - BlockManager: 空闲列表分配器，负责分配和回收 KVBlock。
//
//      目前均为空壳。待 ServerEngine 的调度循环可用后再填充实现。
// ============================================================================

#include <cstdint>
#include <vector>
#include <cuda_fp16.h>

namespace atomflow::server {

// ── Configuration constants / 配置常量 ─────────────────────────────────
// [EN] BLOCK_SIZE: number of tokens stored per physical KV block.
//      Typical values: 16 or 32. Smaller blocks reduce internal fragmentation
//      but increase page-table overhead.
// [CN] BLOCK_SIZE: 每个物理 KV 块存储的 token 数。
//      典型值：16 或 32。块越小，内部碎片越少，但页表开销越大。
static constexpr int BLOCK_SIZE = 16;

// ============================================================================
// KVBlock — One physical slab of KV cache memory
// KVBlock — 一个物理 KV 缓存内存块
//
// [EN] Represents BLOCK_SIZE consecutive token positions worth of K and V
//      data for a single layer. The actual device memory is managed by
//      BlockManager; this struct is a lightweight descriptor.
//
// [CN] 表示单层 BLOCK_SIZE 个连续 token 位置的 K 和 V 数据。
//      实际设备内存由 BlockManager 管理；此结构体仅为轻量级描述符。
// ============================================================================
struct KVBlock {
    int      block_id  = -1;       // [EN] Unique ID within the pool / [CN] 池中的唯一 ID
    half*    k_data    = nullptr;  // [EN] Device ptr → [BLOCK_SIZE, kv_dim] / [CN] 设备指针
    half*    v_data    = nullptr;  // [EN] Device ptr → [BLOCK_SIZE, kv_dim] / [CN] 设备指针
    int      ref_count = 0;        // [EN] Copy-on-write reference count / [CN] 写时复制引用计数
};

// ============================================================================
// PageTable — Per-sequence logical-to-physical block mapping
// PageTable — 每序列的逻辑→物理块映射
//
// [EN] For a sequence of length S, the page table maps logical block index
//      b = token_pos / BLOCK_SIZE  to a physical KVBlock.
//      This is the core data structure that enables non-contiguous KV storage
//      and zero-copy context reuse (fork a PageTable, bump ref_counts).
//
// [CN] 对于长度为 S 的序列，页表将逻辑块索引
//      b = token_pos / BLOCK_SIZE 映射到物理 KVBlock。
//      这是实现非连续 KV 存储和零拷贝上下文复用（fork 页表、增加引用计数）
//      的核心数据结构。
// ============================================================================
struct PageTable {
    // [EN] Ordered list of physical block IDs, one per logical block.
    // [CN] 物理块 ID 的有序列表，每个逻辑块一个。
    std::vector<int> block_ids;

    // [EN] Number of tokens actually written in the last block (0..BLOCK_SIZE).
    // [CN] 最后一个块中实际写入的 token 数（0..BLOCK_SIZE）。
    int last_block_fill = 0;
};

// ============================================================================
// BlockManager — Free-list allocator for KVBlocks
// BlockManager — KVBlock 的空闲列表分配器
//
// [EN] Pre-allocates a fixed pool of KVBlocks at init time (one cudaMalloc),
//      then serves alloc/free requests via a simple free-list.
//      No runtime cudaMalloc — deterministic memory, just like the base engine.
//
// [CN] 初始化时预分配固定的 KVBlock 池（一次 cudaMalloc），
//      然后通过简单的空闲列表响应 alloc/free 请求。
//      无运行时 cudaMalloc — 确定性内存，与基础引擎一致。
// ============================================================================
class BlockManager {
public:
    BlockManager() = default;
    ~BlockManager() = default;

    // [EN] Allocate the physical block pool on GPU.
    //      total_blocks: number of blocks to pre-allocate.
    //      kv_dim:       KV head dimension (n_kv_heads * head_dim).
    //      n_layers:     number of transformer layers.
    // [CN] 在 GPU 上分配物理块池。
    //      total_blocks: 预分配的块数。
    //      kv_dim:       KV 头维度（n_kv_heads * head_dim）。
    //      n_layers:     transformer 层数。
    void initialize(int total_blocks, int kv_dim, int n_layers);

    // [EN] Acquire one free block. Returns block_id, or -1 if pool exhausted.
    // [CN] 获取一个空闲块。返回 block_id，如果池耗尽则返回 -1。
    int allocate_block();

    // [EN] Return a block to the free list.
    // [CN] 将一个块归还到空闲列表。
    void free_block(int block_id);

    // [EN] Number of blocks currently available.
    // [CN] 当前可用的块数。
    int num_free_blocks() const;

private:
    std::vector<KVBlock> pool_;          // [EN] All physical blocks / [CN] 所有物理块
    std::vector<int>     free_list_;     // [EN] Stack of available block IDs / [CN] 可用块 ID 栈
    void*                d_block_pool_ = nullptr;  // [EN] Single GPU allocation / [CN] 单次 GPU 分配
};

} // namespace atomflow::server

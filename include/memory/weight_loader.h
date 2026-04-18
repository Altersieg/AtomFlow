#pragma once

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "core/view.h"

// [EN] Memory alignment utility. Forces the pointer forward to the next multiple of `alignment`.
// [CN] 内存对齐工具。强制指针向前推进到 `alignment` 的下一个倍数。
inline size_t align_up(size_t size, size_t alignment = 256) {
    return (size + alignment - 1) & ~(alignment - 1);
}

// [EN] AtomFlow binary header corresponding exactly to the Python struct.pack format.
// [CN] 与 Python struct.pack 格式完全对应的 AtomFlow 二进制文件头。
struct AtomHeader {
    char magic[4];      // "ATOM"
    int  version;       // 1
    int  group_size;    // 128
    int  dim;           // Hidden size
    int  hidden_dim;    // FFN inner size
    int  n_layers;      // 28
    int  n_heads;       // 24
    int  n_kv_heads;    // 8
    int  vocab_size;    // 128256
    int  max_seq_len;   // 2048
};

// [EN] RAII wrapper for zero-copy weight loading via OS-level memory mapping.
// [CN] 通过操作系统级内存映射实现零拷贝权重加载的 RAII 封装。
// [Bug/Imperfection: Currently assumes the entire 6 GB file fits into host RAM
//  if accessed heavily. For Blackwell/Hopper, mapping directly to GPU memory
//  via cuFile (GDS) would bypass the CPU page-cache entirely.
//  目前假设频繁访问时整个 6 GB 文件能装入主机内存。
//  对于 Blackwell/Hopper，通过 cuFile (GDS) 直接映射到 GPU 显存可完全绕过
//  CPU 页面缓存。]
class WeightLoader {
private:
    int      fd_          = -1;
    size_t   file_size_   = 0;
    void*    mapped_data_ = MAP_FAILED;
    uint8_t* cursor_      = nullptr;  // parse cursor / 解析游标

public:
    AtomHeader header;

    explicit WeightLoader(const std::string& filepath) {
        fd_ = open(filepath.c_str(), O_RDONLY);
        if (fd_ == -1)
            throw std::runtime_error("Failed to open weights file: " + filepath);

        struct stat sb;
        if (fstat(fd_, &sb) == -1) {
            close(fd_);
            throw std::runtime_error("Failed to stat weights file.");
        }
        file_size_ = static_cast<size_t>(sb.st_size);

        // [EN] Map the file into virtual address space. MAP_PRIVATE prevents
        //      accidental disk writes on COW faults.
        // [CN] 将文件映射到虚拟地址空间。MAP_PRIVATE 通过 COW 防止意外写磁盘。
        mapped_data_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (mapped_data_ == MAP_FAILED) {
            close(fd_);
            throw std::runtime_error("Failed to mmap weights file.");
        }

        // [EN] Advise the kernel to read-ahead sequentially (prefetcher hint).
        // [CN] 通知内核顺序预读（预取器提示）。
        madvise(mapped_data_, file_size_, MADV_SEQUENTIAL);

        // Parse header and validate magic number.
        // 解析头部并验证魔数。
        cursor_ = static_cast<uint8_t*>(mapped_data_);
        header  = *reinterpret_cast<const AtomHeader*>(cursor_);

        if (header.magic[0] != 'A' || header.magic[1] != 'T' ||
            header.magic[2] != 'O' || header.magic[3] != 'M')
            throw std::runtime_error("Invalid magic number — not an AtomFlow .bin file.");

        cursor_ += 256;  // skip the full 256-byte header / 跳过完整 256 字节头部
    }

    ~WeightLoader() {
        if (mapped_data_ != MAP_FAILED) munmap(mapped_data_, file_size_);
        if (fd_ != -1)                  close(fd_);
    }

    // Non-copyable — owns fd and mmap region.
    // 不可拷贝 —— 拥有 fd 和 mmap 区域。
    WeightLoader(const WeightLoader&)            = delete;
    WeightLoader& operator=(const WeightLoader&) = delete;

    // [EN] Advance the cursor by align_up(size, 256) bytes and return the
    //      pre-advance pointer. Backed by the OS page-cache (zero-copy).
    // [CN] 将游标向前推进 align_up(size, 256) 字节并返回推进前的指针。
    //      由操作系统页面缓存支持（零拷贝）。
    const void* next_block(size_t size_in_bytes) {
        const void* ptr = cursor_;
        cursor_ += align_up(size_in_bytes);

        // Defensive bounds check / 防御性越界检查
        if (cursor_ > static_cast<uint8_t*>(mapped_data_) + file_size_)
            throw std::runtime_error(
                "Weight file EOF reached unexpectedly — alignment mismatch?");
        return ptr;
    }

    // Convenience: return a typed pointer without an extra cast at call sites.
    // 便利函数：返回有类型的指针，调用方无需额外 cast。
    template<typename T>
    const T* next(size_t num_elements) {
        return static_cast<const T*>(next_block(num_elements * sizeof(T)));
    }

    // Current byte offset from the start of the file (for debugging).
    // 距文件起始的当前字节偏移（用于调试）。
    size_t current_offset() const {
        return static_cast<size_t>(cursor_ - static_cast<uint8_t*>(mapped_data_));
    }

    size_t file_size() const { return file_size_; }
};
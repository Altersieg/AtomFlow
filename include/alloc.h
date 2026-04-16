#ifndef _ARENA_ALLOCATOR_H_
#define _ARENA_ALLOCATOR_H_

#include <cstddef> // size_t
#include <stdexcept>

class ArenaAllocator {
    public:

        ArenaAllocator(size_t total_size);

        ~ArenaAllocator();

        ArenaAllocator(const ArenaAllocator&) = delete; // copy is forbidden
        ArenaAllocator& operator=(const ArenaAllocator&) = delete; // assignment is forbidden

        void* allocate(size_t size) {
            size = align_to_128(size);
            if (offset_ + size > capacity_) {
                throw std::runtime_error("ArenaAllocator: out of memory! Check max_seq_len.");
            }
            void* ptr = static_cast<char*>(base_ptr_) + offset_;
            offset_ += size;
            return ptr;
        } 

        void reset_workspace(size_t new_offset) {  //reset the workspace
            offset_ = new_offset;
        }

        // for logging
        size_t get_capacity() const { return capacity_; }
        size_t get_offset() const { return offset_; }

    private:
        static constexpr size_t align_to_128(size_t size) { return (size + 127) & ~127; }
        void* base_ptr_ = nullptr;
        size_t offset_ = 0;
        size_t capacity_ = 0;
};

struct MemoryMap {
    // for datetype modification in main.cpp
    __nv_fp8_e4m3* weights;     
    float* activations;         
    __nv_fp8_e4m3* kv_cache;   
};
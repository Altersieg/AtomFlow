class MemoryPlanner { // for management of activation zone
private:
    char* base_ptr;
    size_t current_offset;
    size_t total_size;

public:

    MemoryPlanner(void* arena, size_t size) : base_ptr(static_cast<char*>(arena)), current_offset(0), total_size(size) {}

    template<typename T>
    T* allocate(size_t num_elements) {
        size_t bytes = num_elements * sizeof(T);
        size_t aligned_bytes = (bytes + 15) & ~15; 

        if (current_offset + aligned_bytes > total_size) {
            throw std::runtime_error("Arena Out Of Memory!"); 
        }
        T* ptr = reinterpret_cast<T*>(base_ptr + current_offset);
        current_offset += aligned_bytes;
        return ptr;
    }

    void reset_to(size_t offset) {
        current_offset = offset;
    }   

    size_t get_offset() const { return current_offset; }
};
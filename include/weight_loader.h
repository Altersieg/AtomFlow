#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

class WeightLoader {
public:
    void* mapped_ptr;
    size_t file_size;

    WeightLoader(const char* filename) {
        int fd = open(filename, O_RDONLY);
        
        // 1. get the size of model
        file_size = lseek(fd, 0, SEEK_END);
        
        // 2. start mmap ：readonly、private
        mapped_ptr = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        
        if (mapped_ptr == MAP_FAILED) {
            throw std::runtime_error("mmap failed");
        }

        // 3. 关键提示：告诉 OS 我们要进行大规模顺序读取，预取数据
        madvise(mapped_ptr, file_size, MADV_SEQUENTIAL);

        close(fd); // 映射建立后，fd 即可关闭
    }

    ~WeightLoader() {
        munmap(mapped_ptr, file_size);
    }
};
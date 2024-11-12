#include <sys/mman.h>
#include <unistd.h>
#include <cstdint>

#include "utils.h"

bool check_executable(const void* addr)
{
    size_t page_size = sysconf(_SC_PAGESIZE);
    void* aligned_addr = (void*)((uintptr_t)(addr) & ~(page_size - 1));

    // Attempt to set the memory as executable to confirm permissions
    if (mprotect(aligned_addr, page_size, PROT_READ | PROT_WRITE | PROT_EXEC) != 0) {
        return false;
    } 

    return true;
}
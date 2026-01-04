#include "LinearAllocator.hpp"

#include <cstdlib>
#include <stdexcept>

// 1. Implementation of the Constructor
LinearAllocator::LinearAllocator(size_t max_size) {
    arena = std::malloc(max_size);
    capacity = max_size;
    offset = 0;
}

// 2. Implementation of the Destructor
LinearAllocator::~LinearAllocator() {
    std::free(arena);

}

// 3. Implementation of alloc_raw
void* LinearAllocator::alloc_raw(size_t size, size_t alignment) {
    uintptr_t base_addr = reinterpret_cast<uintptr_t>(arena);
    uintptr_t current_addr = base_addr + offset;

    size_t padding = (alignment - current_addr % alignment) % alignment;

    if(offset + padding + size > capacity) return nullptr;

    uintptr_t aligned_addr = current_addr + padding;

    offset += padding + size;

    return reinterpret_cast<void*>(aligned_addr);
}

// 4. Implementation of reset
void LinearAllocator::reset() {
    offset = 0;
}

// 5. Implementation of get_usage
size_t LinearAllocator::get_usage(){
    return offset;
}

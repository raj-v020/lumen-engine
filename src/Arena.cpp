#include "Arena.hpp"

#include <cstdlib>
#include <stdexcept>

namespace lumen {
Arena::Arena(size_t max_size) {
    arena = std::malloc(max_size);
    capacity = max_size;
    offset = 0;
}

Arena::~Arena() {
    std::free(arena);

}

void* Arena::alloc_raw(size_t size, size_t alignment) {
    uintptr_t base_addr = reinterpret_cast<uintptr_t>(arena);
    uintptr_t current_addr = base_addr + offset;

    size_t padding = (alignment - current_addr % alignment) % alignment;

    if(offset + padding + size > capacity) return nullptr;

    uintptr_t aligned_addr = current_addr + padding;

    offset += padding + size;

    return reinterpret_cast<void*>(aligned_addr);
}

void Arena::reset() {
    offset = 0;
}

void* Arena::get_data(){
    return static_cast<void*>(arena);
}

size_t Arena::get_usage(){
    return offset;
}
}

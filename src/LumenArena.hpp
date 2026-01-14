#include <lumen/memory/LumenArena.hpp>
#include <iostream>
#include <stdexcept>

namespace lumen {
namespace memory {

LumenArena::LumenArena(size_t capacity) 
: capacity(capacity), offset(0) {

    arena_ptr = std::aligned_alloc(64, capacity);

    if (!arena_ptr) {
        throw std::runtime_error("LumenArena: Failed to pre-allocate memory pool.");
    }
}

LumenArena::~LumenArena() {
    if (arena_ptr) {
        std::free(arena_ptr);
    }
}

void* LumenArena::alloc_raw(size_t size, size_t alignment) {
    uintptr_t base_addr = reinterpret_cast<uintptr_t>(arena_ptr);
    uintptr_t current_addr = base_addr + offset;

    size_t padding = (alignment - (current_addr % alignment)) % alignment;

    if (offset + padding + size > capacity) {
        return nullptr; 
    }

    uintptr_t aligned_addr = current_addr + padding;

    offset += (padding + size);

    return reinterpret_cast<void*>(aligned_addr);
}

void LumenArena::reset() {
    offset = 0;
}

}
}

#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include <lumen/memory/LumenArena.hpp>
#include <iostream>
#include <stdexcept>
#include <cstddef>

namespace lumen {
namespace memory {

LumenArena::LumenArena(size_t capacity) 
    : capacity(capacity), offset(0) {
    
    size_t safety_margin = 1024 * 1024; 
    capacity = (capacity + safety_margin + 63) & ~63;
    arena_ptr = std::aligned_alloc(64, capacity);

    if (!arena_ptr) {
        throw std::runtime_error("LumenArena: Failed to pre-allocate memory pool.");
    }

    const auto& api = Ort::GetApi();
    OrtStatus* status = api.CreateMemoryInfo("Cpu", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault, &m_memory_info);
    if (status != nullptr) {
        throw std::runtime_error("LumenArena: Failed to create OrtMemoryInfo.");
    }

    m_ort_interface.version = ORT_API_VERSION;
    m_ort_interface.Alloc   = LumenArena::OrtAlloc;
    m_ort_interface.Free    = LumenArena::OrtFree;
    m_ort_interface.Info    = LumenArena::OrtInfo;
    
    m_ort_interface.Reserve = nullptr;
    m_ort_interface.GetStats = nullptr;
}

LumenArena::~LumenArena() {
    if (arena_ptr) {
        std::free(arena_ptr);
    }
    if (m_memory_info) {
        Ort::GetApi().ReleaseMemoryInfo(m_memory_info);
    }
}

void* ORT_API_CALL LumenArena::OrtAlloc(OrtAllocator* this_, size_t size) {
    auto* instance = reinterpret_cast<LumenArena*>(
        reinterpret_cast<char*>(this_) - offsetof(LumenArena, m_ort_interface)
    );
    return instance->alloc_raw(size, 64);
}

void ORT_API_CALL LumenArena::OrtFree(OrtAllocator* this_, void* p) {
    // The entire slab is reclaimed by reset() at the end of the inference request.
}

const OrtMemoryInfo* ORT_API_CALL LumenArena::OrtInfo(const OrtAllocator* this_) {
    auto* instance = reinterpret_cast<const LumenArena*>(
        reinterpret_cast<const char*>(this_) - offsetof(LumenArena, m_ort_interface)
    );
    return instance->m_memory_info;
}

void* LumenArena::alloc_raw(size_t size, size_t alignment) {
    size_t effective_alignment = (alignment < 64) ? 64 : alignment;

    uintptr_t base_addr = reinterpret_cast<uintptr_t>(arena_ptr);
    uintptr_t current_addr = base_addr + offset;

    size_t padding = (effective_alignment - (current_addr % effective_alignment)) % effective_alignment;

    if (offset + padding + size > capacity) {
        return nullptr; 
    }

    uintptr_t aligned_addr = current_addr + padding;
    void* ptr = reinterpret_cast<void*>(aligned_addr);
    
    offset += (padding + size);

    return reinterpret_cast<void*>(aligned_addr);
}

void LumenArena::reset() {
    if (!warmed_up) {
        warmup_offset = offset;
        warmed_up = true;
    }
    offset = warmup_offset;
}

}
}

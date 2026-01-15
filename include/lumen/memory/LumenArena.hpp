#pragma once

#include <lumen/interfaces/ILumenAllocator.hpp>
#include <onnxruntime_c_api.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace lumen {
namespace memory {

class LumenArena : public interfaces::ILumenAllocator {
private:
    void* arena_ptr;
    size_t capacity;
    size_t offset;
    size_t warmup_offset = 0;
    bool warmed_up = false;

    OrtAllocator m_ort_interface;
    OrtMemoryInfo* m_memory_info = nullptr;

    static void* ORT_API_CALL OrtAlloc(OrtAllocator* this_, size_t size);
    static void ORT_API_CALL OrtFree(OrtAllocator* this_, void* p);
    static const OrtMemoryInfo* ORT_API_CALL OrtInfo(const OrtAllocator* this_);

    void* alloc_raw(size_t size, size_t alignment);

public:
    explicit LumenArena(size_t capacity);
    ~LumenArena() override;

    LumenArena(const LumenArena&) = delete;
    LumenArena& operator=(const LumenArena&) = delete;

    void* Alloc(size_t size) override {
        return alloc_raw(size, 64);
    }

    void Free(void* p) override {
        // Do nothing. Arena memory is reclaimed all at once via reset().
    }

    size_t GetAllocatedSize() const override {
        return offset;
    }

    const char* GetName() const override {
        return "LumenArena";
    }

    void reset();

    OrtAllocator* get_ort_interface() { return &m_ort_interface; }

    template <typename T>
    T* alloc_typed(size_t count = 1) {
        return static_cast<T*>(alloc_raw(count * sizeof(T), alignof(T)));
    }

    size_t get_capacity() const { return capacity; }
};

}
}

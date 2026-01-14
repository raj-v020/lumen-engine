#pragma once

#include <lumen/interfaces/ILumenAllocator.hpp>
#include <onnxruntime_cxx_api.h>
#include <atomic>

namespace lumen {
namespace memory {

class StandardOrtAllocator : public interfaces::ILumenAllocator {
private:
    Ort::AllocatorWithDefaultOptions default_ort_allocator;
    std::atomic<size_t> current_allocated_bytes{0};

public:
    void* Alloc(size_t size) override {
        void* p = default_ort_allocator.Alloc(size);
        if (p) {
            current_allocated_bytes += size;
        }
        return p;
    }

    void Free(void* p) override {
        default_ort_allocator.Free(p);
    }

    size_t GetAllocatedSize() const override {
        return current_allocated_bytes.load();
    }

    const char* GetName() const override {
        return "StandardOrtAllocator";
    }
};

}
}

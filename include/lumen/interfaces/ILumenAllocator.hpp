#pragma once

#include <cstddef>

namespace lumen {
namespace interfaces {

class ILumenAllocator {
public:
    virtual ~ILumenAllocator() = default;

    virtual void* Alloc(size_t size) = 0;
    virtual void Free(void* p) = 0;

    virtual size_t GetAllocatedSize() const = 0;
    virtual const char* GetName() const = 0;
};

}
}


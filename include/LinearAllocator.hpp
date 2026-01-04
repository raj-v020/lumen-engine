#pragma once
#include <cstddef>
#include <cstdint>

class LinearAllocator{
private:
  void *arena;
  size_t capacity;
  size_t offset;

  void* alloc_raw(size_t size, size_t alignment);
public:
  explicit LinearAllocator(size_t capacity);
  ~LinearAllocator();

  LinearAllocator(const LinearAllocator&) = delete;
  LinearAllocator& operator=(const LinearAllocator&) = delete;

  template <typename T>
  T* alloc(size_t count = 1) {
    size_t bytes = count * sizeof(T);
    size_t align = alignof(T);
    void* ptr = alloc_raw(bytes, align);
    return static_cast<T*>(ptr);
  }
  
  void reset();

  size_t get_usage();
};

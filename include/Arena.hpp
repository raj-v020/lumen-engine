#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>

class Arena{
private:
  void *arena;
  size_t capacity;
  size_t offset;

  void* alloc_raw(size_t size, size_t alignment);
public:
  explicit Arena(size_t capacity);
  ~Arena();

  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;

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

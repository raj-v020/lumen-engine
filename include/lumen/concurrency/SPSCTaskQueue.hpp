#pragma once

#include <atomic>
#include <lumen/interfaces/ITaskQueue.hpp>
#include <optional>
#include <vector>

namespace lumen {
namespace concurrency {

class SPSCTaskQueue : public interfaces::ITaskQueue {
private:
  static constexpr size_t Capacity = 1024;
  static constexpr size_t Mask = Capacity - 1;

  std::vector<core::InferenceTask> buffer;

  alignas(64) std::atomic<size_t> head{0};
  alignas(64) std::atomic<size_t> tail{0};

  std::atomic<bool> stop_flag{false};

public:
  SPSCTaskQueue() : buffer(Capacity) {}

  bool push(core::InferenceTask task) override {
    if (stop_flag.load(std::memory_order_relaxed)) {
      return false;
    }

    size_t t = tail.load(std::memory_order_relaxed);
    size_t h = head.load(std::memory_order_acquire);

    if (((t + 1) & Mask) == (h & Mask)) {
      return false;
    }

    buffer[t & Mask] = std::move(task);

    tail.store(t + 1, std::memory_order_release);
    return true;
  }

  std::optional<core::InferenceTask> pop() override {
    size_t h = head.load(std::memory_order_relaxed);
    size_t t = tail.load(std::memory_order_acquire);

    if (h == t)
      return std::nullopt;

    core::InferenceTask task = std::move(buffer[h & Mask]);
    head.store(h + 1, std::memory_order_release);
    return task;
  }

  size_t pop_batch(std::vector<core::InferenceTask> &batch,
                   size_t max) override {
    size_t count = 0;
    while (count < max) {
      auto task = pop();
      if (!task)
        break;
      batch.push_back(std::move(*task));
      count++;
    }
    return count;
  }

  size_t size() const override {
    size_t t = tail.load(std::memory_order_relaxed);
    size_t h = head.load(std::memory_order_relaxed);
    return t - h;
  }

  bool empty() const override { return size() == 0; }

  void shutdown() override { stop_flag.store(true); }
};

} // namespace concurrency
} // namespace lumen

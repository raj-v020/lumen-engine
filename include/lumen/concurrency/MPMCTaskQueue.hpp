#pragma once
#include <atomic>
#include <lumen/interfaces/ITaskQueue.hpp>
#include <optional>
#include <thread>
#include <vector>

namespace lumen {
namespace concurrency {

class MPMCTaskQueue : public interfaces::ITaskQueue {
private:
  static constexpr size_t Capacity = 1024;
  static constexpr size_t Mask = Capacity - 1;

  struct Cell {
    std::atomic<size_t> sequence;
    core::InferenceTask data;
  };

  std::vector<Cell> buffer;

  alignas(64) std::atomic<size_t> head{0};
  alignas(64) std::atomic<size_t> tail{0};

  std::atomic<bool> stop_flag{false};

public:
  MPMCTaskQueue() : buffer(Capacity) {
    for (size_t i = 0; i < Capacity; ++i) {
      buffer[i].sequence.store(i, std::memory_order_relaxed);
    }
  }

  bool push(core::InferenceTask task) override {
    if (stop_flag.load(std::memory_order_relaxed)) {
      return false;
    }

    size_t t = tail.load(std::memory_order_relaxed);

    while (true) {
      Cell &cell = buffer[t & Mask];
      size_t seq = cell.sequence.load(std::memory_order_acquire);
      intptr_t diff = (intptr_t)seq - (intptr_t)t;

      if (diff == 0) {
        if (tail.compare_exchange_weak(t, t + 1, std::memory_order_relaxed)) {
          cell.data = std::move(task);

          cell.sequence.store(t + 1, std::memory_order_release);
          return true;
        }
      } else if (diff < 0) {
        return false;
      } else {
        t = tail.load(std::memory_order_relaxed);
      }
    }
  }

  std::optional<core::InferenceTask> pop() override {
    size_t h = head.load(std::memory_order_relaxed);

    while (true) {
      Cell &cell = buffer[h & Mask];
      size_t seq = cell.sequence.load(std::memory_order_acquire);
      intptr_t diff = (intptr_t)seq - (intptr_t)(h + 1);

      if (diff == 0) {
        if (head.compare_exchange_weak(h, h + 1, std::memory_order_relaxed)) {
          core::InferenceTask task = std::move(cell.data);

          cell.sequence.store(h + Capacity, std::memory_order_release);
          return task;
        }
      } else if (diff < 0) {
        return std::nullopt;
      } else {
        h = head.load(std::memory_order_relaxed);
      }
    }
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
    return (t > h) ? (t - h) : 0;
  }

  bool empty() const override { return size() == 0; }

  void shutdown() override { stop_flag.store(true); }
};

} // namespace concurrency
} // namespace lumen

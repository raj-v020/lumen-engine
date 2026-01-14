#pragma once

#include <lumen/interfaces/IResultQueue.hpp>
#include <queue>
#include <mutex>
#include <condition_variable>

namespace lumen {
namespace concurrency {

class NaiveResultQueue : public interfaces::IResultQueue {
private:
  std::queue<core::InferenceResult> data;
  mutable std::mutex guard;
  std::condition_variable cv;
  bool stop = false;

public:
  void push(core::InferenceResult result) override {
    std::lock_guard<std::mutex> lock(guard);
    data.push(std::move(result));
    cv.notify_one();
  }

  std::optional<core::InferenceResult> pop() override {
    std::unique_lock<std::mutex> lock(guard);
    cv.wait(lock, [this] { return !data.empty() || stop; });

    if (data.empty()) return std::nullopt;

    core::InferenceResult result = std::move(data.front());
    data.pop();
    return result;
  }

  std::optional<core::InferenceResult> pop_immediate() override {
    std::lock_guard<std::mutex> lock(guard);
    
    if (data.empty()) {
        return std::nullopt; 
    }

    core::InferenceResult result = std::move(data.front());
    data.pop();
    return result;
  }

  size_t size() const override {
    std::lock_guard<std::mutex> lock(guard);
    return data.size();
  }

  bool empty() const override {
    std::lock_guard<std::mutex> lock(guard);
    return data.empty();
  }

  void shutdown() {
    {
      std::lock_guard<std::mutex> lock(guard);
      stop = true;
    }
    cv.notify_all();
  }
};

}
}


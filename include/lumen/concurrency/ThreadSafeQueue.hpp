#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>

namespace lumen {
template <typename T>
class ThreadSafeQueue {
private:
  std::queue<T> data;
  std::mutex guard;
  std::condition_variable cv;
  bool stop = false;

public:
  void push(T item) {
    std::lock_guard<std::mutex> lock(guard);
    data.push(std::move(item));
    cv.notify_one();
  }

  bool try_pop(T& item) {
    std::unique_lock<std::mutex> lock(guard);
    cv.wait(lock, [this] { 
      return !data.empty() || stop; 
    });
    if (data.empty() && stop) return false;
    item = std::move(data.front());
    data.pop();
    return true;
  }
  bool try_pop_immediate(T& item) {
    std::lock_guard<std::mutex> lock(guard);
    if (data.empty()) return false;

    item = std::move(data.front());
    data.pop();
    return true;
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

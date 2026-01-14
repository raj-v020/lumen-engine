#pragma once

#include <vector>
#include <thread>
#include <memory>
#include <atomic>

#include <lumen/interfaces/ITaskQueue.hpp>
#include <lumen/interfaces/IResultQueue.hpp>
#include <lumen/interfaces/ILumenAllocator.hpp>
#include <lumen/core/InferenceEngine.hpp>

namespace lumen {
namespace concurrency {

class ThreadPool {
private:
  std::shared_ptr<interfaces::ITaskQueue> task_queue;
  std::shared_ptr<interfaces::IResultQueue> response_queue;

  core::InferenceEngine& engine;

  std::vector<std::shared_ptr<interfaces::ILumenAllocator>> worker_allocators;

  std::vector<std::thread> workers;
  std::atomic<bool> stop{false}; 

  public:
    ThreadPool(
      std::shared_ptr<interfaces::ITaskQueue> tq, 
      std::shared_ptr<interfaces::IResultQueue> rq, 
      core::InferenceEngine& e,
      size_t thread_count
    );

    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    void worker_loop(int thread_id);
    void shutdown();
  };

}
}

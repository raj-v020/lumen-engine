#pragma once

#include <vector>
#include <thread>
#include <memory>
#include <atomic>
#include <onnxruntime_cxx_api.h>

#include <lumen/interfaces/ITaskQueue.hpp>
#include <lumen/interfaces/IResultQueue.hpp>
#include <lumen/interfaces/ILumenAllocator.hpp>
#include <lumen/core/InferenceEngine.hpp>

namespace lumen {
namespace concurrency {

struct WorkerPod {
    std::shared_ptr<interfaces::ILumenAllocator> allocator;
    std::unique_ptr<core::InferenceEngine> engine;
};

class ThreadPool {
private:
  std::shared_ptr<interfaces::ITaskQueue> task_queue;
  std::shared_ptr<interfaces::IResultQueue> response_queue;
  
  std::vector<std::unique_ptr<WorkerPod>> worker_pods;

  std::vector<std::thread> workers;
  std::atomic<bool> stop{false}; 

  public:
    ThreadPool(
      std::shared_ptr<interfaces::ITaskQueue> tq, 
      std::shared_ptr<interfaces::IResultQueue> rq, 
      const std::string& model_path
    );

    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    void worker_loop(int thread_id);
    void shutdown();
  };

}
}

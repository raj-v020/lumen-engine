#pragma once

#include <vector>
#include <thread>
#include "Arena.hpp"
#include "ThreadSafeQueue.hpp"
#include "InferenceEngine.hpp"
#include "Processor.hpp"
#include "InferenceTask.hpp"
#include "InferenceResult.hpp"


namespace lumen{
class ThreadPool{
private:
  ThreadSafeQueue<InferenceTask>& task_queue;
  ThreadSafeQueue<InferenceResult>& response_queue;
  InferenceEngine& engine;

  std::vector<std::unique_ptr<Arena>> worker_arenas;
  std::vector<std::thread> workers;
  bool stop = false; 

public:
  ThreadPool(ThreadSafeQueue<InferenceTask>& tq, ThreadSafeQueue<InferenceResult>& rq, InferenceEngine& e);
  ~ThreadPool();
  void worker_loop(int thread_id);
};
}

#include <lumen/concurrency/ThreadPool.hpp>
#include <lumen/utils/Timer.hpp>
#include <lumen/memory/StandardOrtAllocator.hpp> 
#include <lumen/memory/LumenArena.hpp>
#include <lumen/interfaces/ITaskQueue.hpp>
#include <lumen/interfaces/IResultQueue.hpp>
#include <lumen/utils/LumenConfigManager.hpp>

#include <iostream>
#include <vector>
#include <thread>
#include <cstring>
#include <onnxruntime_cxx_api.h>

namespace lumen {
namespace concurrency {

ThreadPool::ThreadPool(
    std::shared_ptr<interfaces::ITaskQueue> tq, 
    std::shared_ptr<interfaces::IResultQueue> rq, 
    const std::string& model_path
) : task_queue(tq), response_queue(rq) {

    auto& config = utils::ConfigManager::get();
    size_t thread_count = config.get_thread_count();
    if (thread_count == 0) {
        thread_count = std::thread::hardware_concurrency();
        if (thread_count == 0) thread_count = 2;
    }
    auto alloc_type = config.get_alloc_type();

    worker_pods.reserve(thread_count);
    
    for (size_t i = 0; i < thread_count; i++) {
        auto pod = std::make_unique<WorkerPod>();

        OrtAllocator* ort_interface = nullptr;

        if (alloc_type == utils::AllocatorType::LUMEN_ARENA) {
            auto arena = std::make_shared<memory::LumenArena>(64 * 1024 * 1024);
            pod->allocator = arena;
            ort_interface = arena->get_ort_interface();
        } else {
            pod->allocator = std::make_shared<memory::StandardOrtAllocator>();
        }

        pod->engine = std::make_unique<core::InferenceEngine>(model_path, ort_interface);

        worker_pods.push_back(std::move(pod));

        workers.emplace_back([this, i]() {
            this->worker_loop(static_cast<int>(i)); 
        });
    }

    std::cout << "[LUMEN] ThreadPool initialized with " << thread_count 
              << " isolated pods using " << (alloc_type == utils::AllocatorType::LUMEN_ARENA ? "LumenArena" : "Standard") 
              << " allocators." << std::endl;
}

ThreadPool::~ThreadPool() {
    shutdown();
}

void ThreadPool::shutdown() {
    if (stop) return;
    stop = true;
    if (task_queue) {
        task_queue->shutdown();
    }
    for (std::thread &worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void ThreadPool::worker_loop(int thread_id) {
    auto& pod = *worker_pods[thread_id];
    auto& config = utils::ConfigManager::get();

    size_t batch_size = config.get_batch_size();
    std::vector<core::InferenceTask> batch_buffer;
    batch_buffer.reserve(batch_size);

    while (!stop) {

        batch_buffer.clear();

        size_t count = task_queue->pop_batch(batch_buffer, batch_size);
        if (count == 0) continue;

        for(auto& task : batch_buffer){
            if (task.trace) {
                auto now = std::chrono::steady_clock::now();
                std::chrono::duration<double, std::milli> wait_time = now - task.trace->start_ts;
                task.trace->queue_wait_ms = wait_time.count();
            }

            core::InferenceResult res = pod.engine->run(std::move(task), *pod.allocator);

            if (auto* arena = dynamic_cast<memory::LumenArena*>(pod.allocator.get())) {
                arena->reset(); 
            }

            response_queue->push(std::move(res));
        }
    }
}

}
}

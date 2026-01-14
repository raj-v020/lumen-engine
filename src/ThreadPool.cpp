#include <lumen/concurrency/ThreadPool.hpp>
#include <lumen/telemetry/TelemetryManager.hpp>
#include <lumen/utils/Timer.hpp>
#include <lumen/memory/StandardOrtAllocator.hpp> 
#include <lumen/interfaces/ITaskQueue.hpp>
#include <lumen/interfaces/IResultQueue.hpp>
#include <cstring>

namespace lumen {
namespace concurrency {

ThreadPool::ThreadPool(
    std::shared_ptr<interfaces::ITaskQueue> tq, 
    std::shared_ptr<interfaces::IResultQueue> rq, 
    core::InferenceEngine& e,
    size_t thread_count
) : task_queue(tq), response_queue(rq), engine(e) {

    if (thread_count == 0) {
        thread_count = std::thread::hardware_concurrency();
        if (thread_count == 0) thread_count = 2;
    }

    worker_allocators.reserve(thread_count);
    
    for (size_t i = 0; i < thread_count; i++) {
        worker_allocators.push_back(std::make_shared<memory::StandardOrtAllocator>());

        workers.emplace_back([this, i]() {
            this->worker_loop(static_cast<int>(i)); 
        });
    }
}

ThreadPool::~ThreadPool() {
    if (task_queue) {
        task_queue->shutdown();
    }
    shutdown();
}

void ThreadPool::shutdown() {
    stop = true;
    for (std::thread &worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void ThreadPool::worker_loop(int thread_id) {
    auto& local_allocator = worker_allocators[thread_id];

    while (!stop) {
        auto task_opt = task_queue->pop();
        
        if (!task_opt) continue;

        auto& task = *task_opt;

        if (task.trace) {
            auto now = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::milli> wait_time = now - task.trace->start_ts;
            task.trace->queue_wait_ms = wait_time.count();
        }

        double* e2e_target = task.trace ? &task.trace->total_e2e_ms : nullptr;
        lumen::utils::Timer e2e_timer(e2e_target);

        void* target = local_allocator->Alloc(task.data.size());
        std::memcpy(target, task.data.data(), task.data.size());

        core::InferenceResult res = engine.run(std::move(task), *local_allocator);
        
        response_queue->push(std::move(res));
    }
}

}
}

#include "ThreadPool.hpp"
#include "ThreadSafeQueue.hpp"
#include "Arena.hpp"
#include "InferenceEngine.hpp"
#include "InferenceTask.hpp"
#include "InferenceResult.hpp"
#include "Timer.hpp"
#include "TelemetryManager.hpp"

namespace lumen{
ThreadPool::ThreadPool(ThreadSafeQueue<InferenceTask>& tq, ThreadSafeQueue<InferenceResult>& rq, InferenceEngine& e) : task_queue(tq), response_queue(rq), engine(e){
    unsigned int num_cores = std::thread::hardware_concurrency();
    if (num_cores == 0) num_cores = 2;

    worker_arenas.reserve(num_cores);
    for (size_t i = 0; i < num_cores; i++) {
        worker_arenas.push_back(std::make_unique<Arena>(1024 * 1024 * 20));

        workers.emplace_back([this, i]() {
            this->worker_loop(i); 
        });
    }
}

ThreadPool::~ThreadPool(){
    task_queue.shutdown();

    for (std::thread &worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void ThreadPool::worker_loop(int thread_id){
    InferenceTask task;
    while(task_queue.try_pop(task)){
        // --- 1. TELEMETRY: QUEUE WAIT TIME ---
        if (task.trace) {
            auto now = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::milli> wait_time = now - task.trace->start_ts;
            task.trace->queue_wait_ms = wait_time.count();
        }

        // --- 2. TELEMETRY: TOTAL E2E TIMER ---
        lumen::LumenTimer e2e_timer(task.trace ? &task.trace->total_e2e_ms : static_cast<double*>(nullptr));

        Arena& local_arena = *worker_arenas[thread_id];

        void* target = local_arena.alloc<uint8_t>(task.data.size());
        std::memcpy(target, task.data.data(), task.data.size());

        // --- 3. INFERENCE CALL ---
        std::string res_str = engine.infer(
            local_arena, 
            *(task.pre), 
            *(task.post), 
            task.trace.get()
        );

        InferenceResult res = {
            .client_fd = task.client_fd,
            .response = std::move(res_str)
        };
        response_queue.push(std::move(res));

        // --- 4. TELEMETRY: HANDOFF ---
        if (task.trace) {
            lumen::TelemetryManager::get().capture_trace(std::move(task.trace));
        }

        local_arena.reset();
    }
}
}

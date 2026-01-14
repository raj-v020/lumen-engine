#include "ThreadPool.hpp"
#include "ThreadSafeQueue.hpp"
#include "Arena.hpp"
#include "InferenceEngine.hpp"
#include "InferenceTask.hpp"
#include "InferenceResult.hpp"

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
        Arena& local_arena = *worker_arenas[thread_id];

        void* target = local_arena.alloc<uint8_t>(task.data.size());
        std::memcpy(target, task.data.data(), task.data.size());

        std::string res_str = engine.infer(local_arena, *(task.pre), *(task.post));

        InferenceResult res = {
            .client_fd = task.client_fd,
            .response = std::move(res_str)
        };
        response_queue.push(std::move(res));

        local_arena.reset();
    }
}
}

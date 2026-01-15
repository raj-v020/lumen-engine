#include <lumen/concurrency/ThreadPool.hpp>
#include <lumen/telemetry/TelemetryManager.hpp>
#include <lumen/utils/Timer.hpp>
#include <lumen/memory/StandardOrtAllocator.hpp> 
#include <lumen/memory/LumenArena.hpp>
#include <lumen/interfaces/ITaskQueue.hpp>
#include <lumen/interfaces/IResultQueue.hpp>
#include <lumen/utils/LumenConfigManager.hpp>
#include <cstring>
#include <onnxruntime_cxx_api.h>

namespace lumen {
namespace concurrency {

ThreadPool::ThreadPool(
    std::shared_ptr<interfaces::ITaskQueue> tq, 
    std::shared_ptr<interfaces::IResultQueue> rq, 
    const std::string& model_path,
    size_t thread_count
) : task_queue(tq), response_queue(rq) {

    if (thread_count == 0) {
        thread_count = std::thread::hardware_concurrency();
        if (thread_count == 0) thread_count = 2;
    }
    auto& config = utils::ConfigManager::get();
    auto alloc_type = config.get_alloc_type();

    worker_pods.reserve(thread_count);
    
    for (size_t i = 0; i < thread_count; i++) {
        auto pod = std::make_unique<WorkerPod>();

        std::string env_name = "lumen_worker_" + std::to_string(i);
        pod->env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, env_name.c_str());

        OrtAllocator* ort_interface = nullptr;

        if (alloc_type == utils::AllocatorType::LUMEN_ARENA) {
            auto arena = std::make_shared<memory::LumenArena>(64 * 1024 * 1024);
            pod->allocator = arena;
            ort_interface = arena->get_ort_interface();

            auto& api = Ort::GetApi();
            Ort::ThrowOnError(api.RegisterAllocator(static_cast<OrtEnv*>(*pod->env), ort_interface));
            
        } else {
            pod->allocator = std::make_shared<memory::StandardOrtAllocator>();
        }

        pod->engine = std::make_unique<core::InferenceEngine>(model_path, pod->env.get(), ort_interface);

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
    auto& pod = *worker_pods[thread_id];

    while (!stop) {
        auto task_opt = task_queue->pop();

        if (auto* arena = dynamic_cast<memory::LumenArena*>(pod.allocator.get())) {
            arena->reset(); 
        }

        if (!task_opt) continue;

        auto& task = *task_opt;

        if (task.trace) {
            auto now = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::milli> wait_time = now - task.trace->start_ts;
            task.trace->queue_wait_ms = wait_time.count();
        }

        core::InferenceResult res = pod.engine->run(std::move(task), *pod.allocator);
        
        response_queue->push(std::move(res));
    }
}

}
}

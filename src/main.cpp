#include <iostream>
#include <vector>
#include <csignal>
#include <atomic>
#include <memory>

#include <lumen/network/TCPServer.hpp>
#include <lumen/core/InferenceEngine.hpp>
#include <lumen/concurrency/ThreadPool.hpp>
#include <lumen/concurrency/NaiveTaskQueue.hpp>
#include <lumen/concurrency/NaiveResultQueue.hpp>
#include <lumen/telemetry/TelemetryManager.hpp>
#include <lumen/utils/LumenConfigManager.hpp>

#include <lumen/models/ImageNetProcessor.hpp> 

std::atomic<bool> g_running(true);

void signal_handler(int) {
    g_running = false;
}

int main() {
    std::signal(SIGINT, signal_handler);

    auto& config = lumen::utils::ConfigManager::get();
    config.load_from_file("../config.json");

    try {
        std::shared_ptr<lumen::interfaces::ITaskQueue> task_q;
        if (config.get_queue_type() == lumen::utils::QueueType::NAIVE_MUTEX) {
            task_q = std::make_shared<lumen::concurrency::NaiveTaskQueue>();
        } 
        
        auto response_q = std::make_shared<lumen::concurrency::NaiveResultQueue>();

        lumen::core::InferenceEngine engine(config.get_model_path());

        std::shared_ptr<lumen::interfaces::IPreProcessor> pre;
        std::shared_ptr<lumen::interfaces::IPostProcessor> post;

        if (config.get_processor_type() == lumen::utils::ProcessorType::IMAGENET) {
            pre = std::make_shared<lumen::core::SqueezeNetPreProcessor>();
            std::string labels = config.get_metadata_extra("label_path");
            post = std::make_shared<lumen::core::SqueezeNetPostProcessor>(labels);
        }

        lumen::concurrency::ThreadPool pool(task_q, response_q, engine, 8);
        lumen::network::TCPServer server("8080", task_q, response_q, engine, pre, post);

        std::cout << "\033[1;36m[LUMEN]\033[0m Lab Initialized." << std::endl;
        std::cout << "\033[1;36m[LUMEN]\033[0m Profile: " 
                  << (config.get_alloc_type() == lumen::utils::AllocatorType::STANDARD ? "Standard" : "Arena") 
                  << " | Queue: Naive" << std::endl;

        while (g_running) {
            server.run(); 
        }

    } catch (const std::exception& e) {
        std::cerr << "\033[1;31m[LUMEN CRITICAL ERROR]\033[0m " << e.what() << std::endl;
        return 1;
    }

    std::cout << "[LUMEN] Shutting down Telemetry..." << std::endl;
    lumen::telemetry::TelemetryManager::get().shutdown();

    std::cout << "[LUMEN] Goodbye." << std::endl;
    return 0;
}

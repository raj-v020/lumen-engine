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

#include <lumen/models/ImageNetProcessor.hpp>

std::atomic<bool> g_running(true);

void signal_handler(int) {
    g_running = false;
}

int main() {
    std::signal(SIGINT, signal_handler);

    std::string model_path = "../models/squeezenet1.1-7.onnx";
    std::string labels_path = "../models/labels.txt";

    try {
        auto task_q = std::make_shared<lumen::concurrency::NaiveTaskQueue>();
        auto response_q = std::make_shared<lumen::concurrency::NaiveResultQueue>();

        lumen::core::InferenceEngine engine(model_path);

        auto pre = std::make_shared<lumen::core::SqueezeNetPreProcessor>();
        auto post = std::make_shared<lumen::core::SqueezeNetPostProcessor>(labels_path);

        lumen::concurrency::ThreadPool pool(task_q, response_q, engine, 8);

        lumen::network::TCPServer server("8080", task_q, response_q, engine, pre, post);

        std::cout << "\033[1;36m[LUMEN]\033[0m System Initialized. Entering main loop..." << std::endl;
        std::cout << "\033[1;36m[LUMEN]\033[0m Press Ctrl+C to shutdown gracefully." << std::endl;

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

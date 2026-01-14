#include <iostream>
#include <vector>
#include <csignal>
#include <atomic>

#include "TCPServer.hpp"
#include "InferenceEngine.hpp"
#include "SqueezeNet.hpp"
#include "ResNet.hpp"
#include "ThreadSafeQueue.hpp"
#include "InferenceTask.hpp"
#include "InferenceResult.hpp"
#include "ThreadPool.hpp"
#include "TelemetryManager.hpp"

using namespace lumen;

std::atomic<bool> g_running(true);

void signal_handler(int) {
    g_running = false;
}

int main() {
    std::signal(SIGINT, signal_handler);
    std::string model_path = "../models/squeezenet1.1-7.onnx";
    // std::string model_path = "../models/resnet18-v1-7.onnx";
    std::string labels_path = "../models/labels.txt";

    ThreadSafeQueue<InferenceTask> task_q;
    ThreadSafeQueue<InferenceResult> response_q;

    InferenceEngine engine(model_path);
    auto pre = std::make_shared<SqueezeNetPreProcessor>();
    auto post = std::make_shared<SqueezeNetPostProcessor>(labels_path);
    /*
    auto pre = std::make_shared<ResNetPreProcessor>();
    auto post = std::make_shared<ResNetPostProcessor>(labels_path);
    */
    ThreadPool pool(task_q, response_q, engine);

    TCPServer server("8080", task_q, response_q, engine, pre, post);
    std::cout << "[Lumen] Server listening on port 8080..." << std::endl;
    std::cout << "[Lumen] Telemetry Active. Watch stdout for heartbeat." << std::endl;

    while (g_running) {
        try {
            server.run(); 
            
        } catch (const std::exception& e) {
            std::cerr << "[Lumen Error] " << e.what() << std::endl;
        }
    }

    std::cout << "[Lumen] Shutting down Telemetry..." << std::endl;
    TelemetryManager::get().shutdown();

    return 0;
}

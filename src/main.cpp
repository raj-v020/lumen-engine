#include <iostream>
#include <vector>
#include "TCPServer.hpp"
#include "InferenceEngine.hpp"
#include "SqueezeNet.hpp"
#include "ResNet.hpp"
#include "ThreadSafeQueue.hpp"
#include "InferenceTask.hpp"
#include "InferenceResult.hpp"
#include "ThreadPool.hpp"

using namespace lumen;

int main() {
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

    while (true) {
        try {
            server.run();
        } catch (const std::exception& e) {
            std::cerr << "[Lumen Error] " << e.what() << std::endl;
        }
    }

    return 0;
}

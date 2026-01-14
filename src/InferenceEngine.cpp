#include "InferenceEngine.hpp"

#include <iostream>
#include "Arena.hpp"
#include "IProcessor.hpp"
#include "Timer.hpp"

namespace lumen {

InferenceEngine::InferenceEngine(std::string model_path){
    session = std::make_unique<Ort::Session>(env, model_path.c_str(), sessionOptions);
    Ort::AllocatorWithDefaultOptions allocator;
    
    for (size_t i = 0; i < session->GetInputCount(); i++) {
        // 1. Get Name
        auto name_ptr = session->GetInputNameAllocated(i, allocator);
        input_names.emplace_back(name_ptr.get());
        input_node_names.push_back(input_names.back().c_str());

        // 2. GET SHAPE AUTOMATICALLY
        Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> dims = tensor_info.GetShape();

        // Handle dynamic batch sizes (-1 or 0) by forcing them to 1 for inference
        if(dims[0] <= 0) dims[0] = 1;

        input_node_dims.push_back(dims);
    }

    for (size_t i = 0; i < session->GetOutputCount(); i++) {
        auto name_ptr = session->GetOutputNameAllocated(i, allocator);
        output_names.emplace_back(name_ptr.get());
        output_node_names.push_back(output_names.back().c_str());
    }
}


InferenceEngine::~InferenceEngine(){
    std::cout << "Lumen InferenceEngine: Shutting down and releasing model resources." << std::endl;
}

std::string InferenceEngine::infer(Arena& arena, IPreProcessor& pre, IPostProcessor& post) {
    LumenTimer total_timer("Total Inference Loop");

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    const std::vector<int64_t>& shape = input_node_dims[0];
    size_t input_tensor_size = 1;
    for (auto dim : shape) {
        if(dim > 0) input_tensor_size *= dim;
    }

    std::vector<float> processed_data(input_tensor_size);

    {
        LumenTimer pre_timer("Data Preprocessing");
        pre.transform(arena.get_data(), processed_data.data(), shape[3], shape[2]);
    }

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        processed_data.data(),
        processed_data.size(),
        shape.data(),
        shape.size()
    );

    std::vector<Ort::Value> output_tensors;
    {
        LumenTimer onnx_timer("ONNX Session Run");
        output_tensors = session->Run(
            runOptions,
            input_node_names.data(),
            &input_tensor,
            1,
            output_node_names.data(),
            1
        );
    }

    float* output_data = output_tensors[0].GetTensorMutableData<float>();

    // Note: Some models (like ResNet) might have 1000 classes, 
    // but it's safer to get the count from the output tensor shape
    size_t output_count = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> results(output_data, output_data + output_count);

    return post.handle_results(results);
}
}

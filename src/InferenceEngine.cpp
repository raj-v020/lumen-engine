#include <lumen/core/InferenceEngine.hpp>
#include <lumen/utils/Timer.hpp>
#include <iostream>

namespace lumen {
namespace core {
namespace utils = lumen::utils;

InferenceEngine::InferenceEngine(const std::string& model_path) {
    session = std::make_unique<Ort::Session>(env, model_path.c_str(), sessionOptions);
    Ort::AllocatorWithDefaultOptions allocator;
    
    size_t num_input_nodes = session->GetInputCount();
    input_names.reserve(num_input_nodes);
    input_node_names.reserve(num_input_nodes);

    for (size_t i = 0; i < num_input_nodes; i++) {
        auto name_ptr = session->GetInputNameAllocated(i, allocator);
        input_names.emplace_back(name_ptr.get());
        input_node_names.push_back(input_names.back().c_str());

        Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> dims = tensor_info.GetShape();

        if(dims[0] <= 0) dims[0] = 1;
        input_node_dims.push_back(dims);
    }

    size_t num_output_nodes = session->GetOutputCount();
    output_names.reserve(num_output_nodes);
    output_node_names.reserve(num_output_nodes);

    for (size_t i = 0; i < num_output_nodes; i++) {
        auto name_ptr = session->GetOutputNameAllocated(i, allocator);
        output_names.emplace_back(name_ptr.get());
        output_node_names.push_back(output_names.back().c_str());
    }
}

InferenceEngine::~InferenceEngine() {
    std::cout << "Lumen InferenceEngine: Shutting down and releasing model resources." << std::endl;
}

InferenceResult InferenceEngine::run(InferenceTask task, interfaces::ILumenAllocator& allocator) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    const std::vector<int64_t>& shape = input_node_dims[0];
    
    size_t input_tensor_size = 1;
    for (auto dim : shape) {
        if(dim > 0) input_tensor_size *= dim;
    }

    size_t input_float_count = 1;
    for (auto dim : shape) {
        if (dim > 0) input_float_count *= dim;
    }

    float* input_tensor_ptr = static_cast<float*>(allocator.Alloc(input_float_count * sizeof(float)));

    // --- PHASE 1: PREPROCESSING ---
    {
        utils::Timer pre_timer(task.trace ? &task.trace->preprocess_ms : nullptr);
        task.pre->transform(task.data.data(), input_tensor_ptr, shape[3], shape[2]);
    }

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_ptr,
        input_float_count,
        shape.data(),
        shape.size()
    );

    std::vector<Ort::Value> output_tensors;

    // --- PHASE 2: INFERENCE (ONNX RUN) ---
    {
        utils::Timer onnx_timer(task.trace ? &task.trace->inference_ms : nullptr);
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
    size_t output_count = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> results(output_data, output_data + output_count);

    // --- PHASE 3: POST-PROCESSING ---
    std::string final_result_str;
    {
        utils::Timer post_timer(task.trace ? &task.trace->postprocess_ms : nullptr);
        final_result_str = task.post->handle_results(results);
    }

    return InferenceResult {
        .client_fd = task.client_fd,
        .response = std::move(final_result_str),
        .trace = std::move(task.trace) 
    };
}

}
}

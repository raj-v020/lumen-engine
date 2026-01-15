#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>
#include <lumen/core/InferenceEngine.hpp>
#include <lumen/utils/Timer.hpp>
#include <iostream>

namespace lumen {
namespace core {
namespace utils = lumen::utils;

InferenceEngine::InferenceEngine(
    const std::string& model_path, 
    Ort::Env* local_env,
    OrtAllocator* allocator
) {
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetInterOpNumThreads(1);
    sessionOptions.SetExecutionMode(ORT_SEQUENTIAL);

    if (allocator) {
        sessionOptions.AddConfigEntry("session.use_env_allocators", "1");
        sessionOptions.EnableCpuMemArena();
        sessionOptions.EnableMemPattern();
    }

    session = std::make_unique<Ort::Session>(*local_env, model_path.c_str(), sessionOptions);
    
    Ort::AllocatorWithDefaultOptions default_allocator;
    
    size_t num_input_nodes = session->GetInputCount();
    input_names.reserve(num_input_nodes);
    input_node_names.reserve(num_input_nodes);

    for (size_t i = 0; i < num_input_nodes; i++) {
        auto name_ptr = session->GetInputNameAllocated(i, default_allocator);
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
        auto name_ptr = session->GetOutputNameAllocated(i, default_allocator);
        output_names.emplace_back(name_ptr.get());
        output_node_names.push_back(output_names.back().c_str());

        Ort::TypeInfo type_info = session->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> dims = tensor_info.GetShape();

        if (dims.size() > 0 && dims[0] <= 0) dims[0] = 1;

        output_node_dims.push_back(dims);
    }
}

InferenceEngine::~InferenceEngine() {
    std::cout << "Lumen InferenceEngine: Shutting down and releasing model resources." << std::endl;
}

InferenceResult InferenceEngine::run(InferenceTask task, interfaces::ILumenAllocator& allocator) {
    // 1. Calculate Sizes
    const std::vector<int64_t>& input_shape = input_node_dims[0];
    size_t input_count = 1;
    for (auto dim : input_shape) if (dim > 0) input_count *= dim;

    // 2. Allocate Input Tensor from the Arena
    float* input_ptr = static_cast<float*>(allocator.Alloc(input_count * sizeof(float)));
    // 3. Phase 1: Preprocessing
    {
        utils::Timer pre_timer(task.trace ? &task.trace->preprocess_ms : nullptr);
        task.pre->transform(task.data.data(), input_ptr, input_shape[3], input_shape[2]);
    }

    // 4. Wrap Input in Ort::Value
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_ptr, input_count, input_shape.data(), input_shape.size());

    // 5. PRE-ALLOCATE OUTPUTS
    const std::vector<int64_t>& output_shape = output_node_dims[0];
    size_t output_count = 1;
    for (auto dim : output_shape) if (dim > 0) output_count *= dim;

    float* output_ptr = static_cast<float*>(allocator.Alloc(output_count * sizeof(float)));
    
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
        memory_info, output_ptr, output_count, output_shape.data(), output_shape.size());

    // 6. Phase 2: Inference
    {
        utils::Timer onnx_timer(task.trace ? &task.trace->inference_ms : nullptr);
        
        session->Run(runOptions, 
                    input_node_names.data(), &input_tensor, 1,
                    output_node_names.data(), &output_tensor, 1);
    }

    // 7. Phase 3: Post-processing
    std::string final_result_str;
    {
        utils::Timer post_timer(task.trace ? &task.trace->postprocess_ms : nullptr);
        final_result_str = task.post->handle_results(output_ptr, output_count);
    }

    return InferenceResult {
        .client_fd = task.client_fd,
        .response = std::move(final_result_str),
        .trace = std::move(task.trace) 
    };
}
}
}





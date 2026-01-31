#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>
#include <lumen/core/InferenceEngine.hpp>
#include <lumen/utils/Timer.hpp>
#include <lumen/memory/LumenArena.hpp>
#include <iostream>
#include <cstring>

namespace lumen {
namespace core {
namespace utils = lumen::utils;

InferenceEngine::InferenceEngine(
    const std::string& model_path, 
    OrtAllocator* allocator
) {
    std::string env_name = "LumenEnv_" + std::to_string((uintptr_t)this);
    env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, env_name.c_str());

    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetInterOpNumThreads(1);
    sessionOptions.SetExecutionMode(ORT_SEQUENTIAL);

    if (allocator) {
        auto& api = Ort::GetApi();
        Ort::ThrowOnError(api.RegisterAllocator((OrtEnv*)(*env), allocator));
        sessionOptions.AddConfigEntry("session.use_env_allocators", "1");
        sessionOptions.EnableCpuMemArena();
        sessionOptions.EnableMemPattern();
    }

    session = std::make_unique<Ort::Session>(*env, model_path.c_str(), sessionOptions);
    
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

    if (allocator) {
        auto* arena = lumen::memory::LumenArena::FromOrt(allocator);
        if (arena) {
            migrate_metadata_to_arena(arena);
        }
    }
}

InferenceEngine::~InferenceEngine() {
    std::cout << "Lumen InferenceEngine: Shutting down and releasing model resources." << std::endl;
}

void InferenceEngine::migrate_metadata_to_arena(memory::LumenArena* arena) {
    num_inputs = input_node_dims.size();
    num_outputs = output_node_dims.size();

    // 1. Allocate top-level arrays in Arena
    arena_input_node_names = arena->alloc_typed<const char*>(num_inputs);
    arena_output_node_names = arena->alloc_typed<const char*>(num_outputs);
    arena_input_node_dims = arena->alloc_typed<int64_t*>(num_inputs);
    arena_output_node_dims = arena->alloc_typed<int64_t*>(num_outputs);
    arena_input_dim_counts = arena->alloc_typed<size_t>(num_inputs);
    arena_output_dim_counts = arena->alloc_typed<size_t>(num_outputs);

    // 2. Deep-copy Inputs to Arena
    for (size_t i = 0; i < num_inputs; ++i) {
        size_t name_len = input_names[i].size() + 1;
        char* name_arena = arena->alloc_typed<char>(name_len);
        std::memcpy(name_arena, input_names[i].c_str(), name_len);
        arena_input_node_names[i] = name_arena;

        size_t dim_count = input_node_dims[i].size();
        int64_t* dims_arena = arena->alloc_typed<int64_t>(dim_count);
        std::memcpy(dims_arena, input_node_dims[i].data(), dim_count * sizeof(int64_t));
        arena_input_node_dims[i] = dims_arena;
        arena_input_dim_counts[i] = dim_count;
    }

    // 3. Deep-copy Outputs to Arena
    for (size_t i = 0; i < num_outputs; ++i) {
        size_t name_len = output_names[i].size() + 1;
        char* name_arena = arena->alloc_typed<char>(name_len);
        std::memcpy(name_arena, output_names[i].c_str(), name_len);
        arena_output_node_names[i] = name_arena;

        size_t dim_count = output_node_dims[i].size();
        int64_t* dims_arena = arena->alloc_typed<int64_t>(dim_count);
        std::memcpy(dims_arena, output_node_dims[i].data(), dim_count * sizeof(int64_t));
        arena_output_node_dims[i] = dims_arena;
        arena_output_dim_counts[i] = dim_count;
    }

    // 4. Mark steady state and clear temporary heap vectors
    metadata_migrated = true;
    input_names.clear(); input_names.shrink_to_fit();
    output_names.clear(); output_names.shrink_to_fit();
    input_node_dims.clear(); input_node_dims.shrink_to_fit();
    output_node_dims.clear(); output_node_dims.shrink_to_fit();
}

struct ScopedFree {
    interfaces::ILumenAllocator& allocator;
    void* ptr;
    ScopedFree(interfaces::ILumenAllocator& a, void* p) : allocator(a), ptr(p) {}
    ~ScopedFree() { if (ptr) allocator.Free(ptr); }
};

InferenceResult InferenceEngine::run(InferenceTask task, interfaces::ILumenAllocator& allocator) {
    // A. ONE-TIME HIJACK: Migrate metadata to Arena during Warmup
    const int64_t* input_shape = nullptr;
    size_t input_dim_count = 0;
    const int64_t* output_shape = nullptr;
    size_t output_dim_count = 0;

    if (metadata_migrated) {
        input_shape = arena_input_node_dims[0];
        input_dim_count = arena_input_dim_counts[0];
        output_shape = arena_output_node_dims[0];
        output_dim_count = arena_output_dim_counts[0];
    } else {
        input_shape = input_node_dims[0].data();
        input_dim_count = input_node_dims[0].size();
        output_shape = output_node_dims[0].data();
        output_dim_count = output_node_dims[0].size();
    }

    // 1. Calculate Sizes
    size_t input_count = 1;
    for (size_t i = 0; i < input_dim_count; ++i) if (input_shape[i] > 0) input_count *= input_shape[i];

    // 2. HEAP PROTECTION: Guard against corrupted payloads
    if (task.data.size() < input_count) {
        return InferenceResult { .client_fd = task.client_fd, .response = "Error: Payload size mismatch", .trace = std::move(task.trace) };
    }

    // 3. ALLOCATE FROM ARENA (Zero-Copy)
    float* input_ptr = static_cast<float*>(allocator.Alloc(input_count * sizeof(float)));
    ScopedFree input_guard(allocator, input_ptr);
    // 4. Phase 1: Preprocessing (Using stable arena dimensions)
    {
        utils::Timer pre_timer(task.trace ? &task.trace->preprocess_ms : nullptr);
        task.pre->transform(task.data.data(), input_ptr, input_shape[3], input_shape[2]);
    }

    std::string final_result_str;
    {
        // 5. Wrap Tensors using ARENA-BACKED shape pointers (Total Stability)
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_ptr, input_count, input_shape, input_dim_count);

        size_t output_count = 1;
        for (size_t i = 0; i < output_dim_count; ++i) if (output_shape[i] > 0) output_count *= output_shape[i];

        float* output_ptr = static_cast<float*>(allocator.Alloc(output_count * sizeof(float)));
        ScopedFree output_guard(allocator, output_ptr);

        Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
            memory_info, output_ptr, output_count, output_shape, output_dim_count);

        // 6. Phase 2: Inference
        {
            utils::Timer onnx_timer(task.trace ? &task.trace->inference_ms : nullptr);
            if (metadata_migrated) {
                session->Run(runOptions, arena_input_node_names, &input_tensor, 1, arena_output_node_names, &output_tensor, 1);
            } else {
                session->Run(runOptions, input_node_names.data(), &input_tensor, 1, output_node_names.data(), &output_tensor, 1);
            }
        }

        // 7. Phase 3: Post-processing
        {
            utils::Timer post_timer(task.trace ? &task.trace->postprocess_ms : nullptr);
            final_result_str = task.post->handle_results(output_ptr, output_count);
        }
    }

    return InferenceResult { .client_fd = task.client_fd, .response = std::move(final_result_str), .trace = std::move(task.trace) };
}
}
}

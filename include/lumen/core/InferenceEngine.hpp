#pragma once

#include <onnxruntime_cxx_api.h>
#include <memory>
#include <vector>
#include <string>

#include <lumen/interfaces/ILumenAllocator.hpp>
#include <lumen/core/InferenceTask.hpp>
#include <lumen/core/InferenceResult.hpp>

namespace lumen {
namespace core {

class InferenceEngine {
private:
  Ort::SessionOptions sessionOptions;
  std::unique_ptr<Ort::Session> session;

  std::vector<std::string> input_names;
  std::vector<std::string> output_names;

  std::vector<const char*> input_node_names;
  std::vector<const char*> output_node_names;

  std::vector<std::vector<int64_t>> input_node_dims;
  std::vector<std::vector<int64_t>> output_node_dims;

  Ort::RunOptions runOptions;

public:
  InferenceEngine(const std::string& model_path, Ort::Env* local_env, OrtAllocator* allocator = nullptr);
  ~InferenceEngine();

  InferenceResult run(InferenceTask task, interfaces::ILumenAllocator& allocator);

  InferenceEngine(const InferenceEngine&) = delete;
  InferenceEngine& operator=(const InferenceEngine&) = delete;
};

}
}

#pragma once
#include <onnxruntime_cxx_api.h>
#include "Arena.hpp"
#include "Processor.hpp"

namespace lumen {
class InferenceEngine{
private:
  Ort::Env env;
  Ort::SessionOptions sessionOptions;
  std::unique_ptr<Ort::Session> session;

  std::vector<std::string> input_names;
  std::vector<std::string> output_names;

  std::vector<const char*> input_node_names;
  std::vector<const char*> output_node_names;

  std::vector<std::vector<int64_t>> input_node_dims;

  Ort::RunOptions runOptions;
public:
  InferenceEngine(std::string model_path);
  ~InferenceEngine();
  void infer(lumen::Arena& arena, IPreprocessor& pre, IPostProcessor& post);
};
}

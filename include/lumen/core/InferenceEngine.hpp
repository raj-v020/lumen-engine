#pragma once

#include <memory>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

#include <lumen/core/InferenceResult.hpp>
#include <lumen/core/InferenceTask.hpp>
#include <lumen/interfaces/ILumenAllocator.hpp>
#include <lumen/memory/LumenArena.hpp>

namespace lumen {
namespace core {

class InferenceEngine {
private:
  std::unique_ptr<Ort::Env> env;

  Ort::SessionOptions sessionOptions;
  std::unique_ptr<Ort::Session> session;

  // TEMPORARY HEAP METADATA
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;

  std::vector<const char *> input_node_names;
  std::vector<const char *> output_node_names;

  std::vector<std::vector<int64_t>> input_node_dims;
  std::vector<std::vector<int64_t>> output_node_dims;

  // ARENA-BACKED METADATA
  const char **arena_input_node_names = nullptr;
  const char **arena_output_node_names = nullptr;

  int64_t **arena_input_node_dims = nullptr;
  int64_t **arena_output_node_dims = nullptr;

  size_t *arena_input_dim_counts = nullptr;
  size_t *arena_output_dim_counts = nullptr;

  size_t num_inputs = 0;
  size_t num_outputs = 0;

  bool metadata_migrated = false;

  Ort::RunOptions runOptions;

  void migrate_metadata_to_arena(memory::LumenArena *arena);

public:
  InferenceEngine(const std::string &model_path,
                  OrtAllocator *allocator = nullptr);
  ~InferenceEngine();

  InferenceResult run(InferenceTask task,
                      interfaces::ILumenAllocator &allocator);

  InferenceEngine(const InferenceEngine &) = delete;
  InferenceEngine &operator=(const InferenceEngine &) = delete;
};

} // namespace core
} // namespace lumen

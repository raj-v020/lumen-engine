#pragma once

#include <chrono>
#include <cstdint>
#include <string>

namespace lumen {
namespace core {

struct InferenceTrace {
  uint32_t request_id;

  std::chrono::steady_clock::time_point start_ts;

  double queue_wait_ms = 0.0;
  double preprocess_ms = 0.0;
  double inference_ms  = 0.0;
  double postprocess_ms = 0.0;
  double total_e2e_ms   = 0.0;

  std::string metadata = "";

  explicit InferenceTrace(uint32_t id) 
  : request_id(id), start_ts(std::chrono::steady_clock::now()) {}

  double get_elapsed_ms() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(now - start_ts).count();
  }
};

}
}

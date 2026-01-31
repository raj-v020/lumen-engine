#pragma once

#include <lumen/core/InferenceTask.hpp>
#include <optional>
#include <vector>

namespace lumen {
namespace interfaces {

class ITaskQueue {
public:
  virtual ~ITaskQueue() = default;

  virtual bool push(core::InferenceTask task) = 0;
  virtual std::optional<core::InferenceTask> pop() = 0;

  virtual size_t pop_batch(std::vector<core::InferenceTask> &batch,
                           size_t max_batch_size) = 0;

  virtual size_t size() const = 0;
  virtual bool empty() const = 0;

  virtual void shutdown() = 0;
};

} // namespace interfaces
} // namespace lumen

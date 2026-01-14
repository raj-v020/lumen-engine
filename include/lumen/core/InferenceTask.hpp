#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>

#include <lumen/interfaces/IProcessor.hpp>
#include <lumen/core/InferenceTrace.hpp>

namespace lumen {
namespace core {

struct InferenceTask {
  int client_fd;
  std::vector<uint8_t> data;

  std::shared_ptr<interfaces::IPreProcessor> pre;
  std::shared_ptr<interfaces::IPostProcessor> post;

  std::unique_ptr<InferenceTrace> trace; 

  InferenceTask(int fd, 
                std::vector<uint8_t>&& tensor_data, 
                std::shared_ptr<interfaces::IPreProcessor> pr, 
                std::shared_ptr<interfaces::IPostProcessor> po,
                std::unique_ptr<InferenceTrace> tr = nullptr)
    : client_fd(fd), 
    data(std::move(tensor_data)), 
    pre(std::move(pr)), 
    post(std::move(po)),
    trace(std::move(tr))
  {}

  InferenceTask() : client_fd(-1), pre(nullptr), post(nullptr), trace(nullptr) {}
};

}
}

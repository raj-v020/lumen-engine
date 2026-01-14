#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace lumen {
struct InferenceTask {
    int client_fd;
    std::vector<uint8_t> data;

    std::shared_ptr<IPreProcessor> pre;
    std::shared_ptr<IPostProcessor> post;
    std::unique_ptr<InferenceTrace> trace; 

    InferenceTask(int fd, 
                  std::vector<uint8_t>&& tensor_data, 
                  std::shared_ptr<IPreProcessor> pr, 
                  std::shared_ptr<IPostProcessor> po,
                  std::unique_ptr<InferenceTrace> tr = nullptr)
        : client_fd(fd), 
          data(std::move(tensor_data)), 
          pre(std::move(pr)), 
          post(std::move(po)),
          trace(std::move(tr))
    {}

    InferenceTask() : client_fd(-1), trace(nullptr) {}
};
}


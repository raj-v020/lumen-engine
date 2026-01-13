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
    InferenceTask(int fd, std::vector<uint8_t>&& tensor_data, 
                  std::shared_ptr<IPreProcessor> pr, 
                  std::shared_ptr<IPostProcessor> po)
        : client_fd(fd), data(std::move(tensor_data)), 
          pre(std::move(pr)), post(std::move(po)) {}
    InferenceTask() : client_fd(-1) {}
};
}


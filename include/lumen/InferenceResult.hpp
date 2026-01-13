#pragma once
#include <string>

namespace lumen {
struct InferenceResult {
    int client_fd;
    std::string response;
};
}

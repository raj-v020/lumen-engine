#pragma once

#include <string>
#include <memory>
#include <lumen/core/InferenceTrace.hpp>

namespace lumen {
namespace core {

struct InferenceResult {
    int client_fd;
    std::string response;

    std::unique_ptr<InferenceTrace> trace;
    bool success = true;
    std::string error_message = "";
};

}
}

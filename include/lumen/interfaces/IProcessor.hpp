#pragma once

namespace lumen {
namespace interfaces{
class IPreProcessor {
public:
    virtual ~IPreProcessor() = default;
    virtual void transform(const uint8_t* input, float* output, size_t width, size_t height) = 0;
};

class IPostProcessor {
public:
    virtual ~IPostProcessor() = default;
    virtual std::string handle_results(const float* results_ptr, size_t count) = 0;
};
}
}


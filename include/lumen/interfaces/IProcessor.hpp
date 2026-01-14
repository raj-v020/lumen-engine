#pragma once

namespace lumen {
class IPreProcessor {
public:
    virtual ~IPreProcessor() = default;
    virtual void transform(void* raw_arena, float* tensor_buffer, size_t width, size_t height) = 0;
};

class IPostProcessor {
public:
    virtual ~IPostProcessor() = default;
    virtual std::string handle_results(const std::vector<float>& results) = 0;
};
}


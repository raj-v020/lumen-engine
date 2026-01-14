#pragma once

#include <lumen/core/InferenceResult.hpp>
#include <optional>

namespace lumen {
namespace interfaces {

class IResultQueue {
public:
    virtual ~IResultQueue() = default;

    virtual void push(core::InferenceResult result) = 0;
    virtual std::optional<core::InferenceResult> pop() = 0;
    virtual std::optional<core::InferenceResult> pop_immediate() = 0;

    virtual size_t size() const = 0;
    virtual bool empty() const = 0;
};

}
}


#pragma once

#include <iostream>
#include <chrono>
#include <string_view>

namespace lumen {
namespace utils {

constexpr bool ENABLE_CONSOLE_LOGGING = true; 

class Timer {
private:
    double* m_target;
    std::string_view m_name;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start;

public:
    explicit Timer(double* target) 
        : m_target(target), m_name("") {
        if (m_target) {
            m_start = std::chrono::high_resolution_clock::now();
        }
    }

    explicit Timer(std::string_view name) 
        : m_target(nullptr), m_name(name) {
        if constexpr (ENABLE_CONSOLE_LOGGING) {
            m_start = std::chrono::high_resolution_clock::now();
        }
    }

    ~Timer() {
        if (m_target || (!m_name.empty() && ENABLE_CONSOLE_LOGGING)) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration<double, std::milli>(end - m_start).count();

            // Case 1: Write to telemetry trace memory
            if (m_target) {
                *m_target = duration;
            } 
            // Case 2: Print to console
            else if (!m_name.empty()) {
                std::cout << "[PROFILE] " << m_name << ": " << duration << " ms" << std::endl;
            }
        }
    }
};

} // namespace utils
} // namespace lumen

#pragma once

#include <iostream>
#include <chrono>
#include <string_view>

namespace lumen {

constexpr bool ENABLE_CONSOLE_LOGGING = true; 

class LumenTimer {
private:
    double* m_target;
    std::string_view m_name;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start;

public:
    // CONSTRUCTOR 1: Telemetry Mode (Targeted)
    // Writes the duration directly into the provided memory address (e.g., &trace->preprocess_ms).
    explicit LumenTimer(double* target) 
        : m_target(target){
        if (m_target) {
            m_start = std::chrono::high_resolution_clock::now();
        }
    }

    // CONSTRUCTOR 2: Console Mode (Legacy/Debug)
    explicit LumenTimer(std::string_view name) 
        : m_target(nullptr), m_name(name) {
        if constexpr (ENABLE_CONSOLE_LOGGING) {
            m_start = std::chrono::high_resolution_clock::now();
        }
    }
    ~LumenTimer() {
        // Case 1: Trace Mode - Write to Memory (Silent)
        if (m_target) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration<double, std::milli>(end - m_start).count();
            *m_target = duration;
        }
        // Case 2: Console Mode - Print to Screen (Verbose)
        else if (!m_name.empty()) {
            if constexpr (ENABLE_CONSOLE_LOGGING) {
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration<double, std::milli>(end - m_start).count();
                std::cout << "[PROFILE] " << m_name << ": " << duration << " ms" << std::endl;
            }
        }
    }
};
}

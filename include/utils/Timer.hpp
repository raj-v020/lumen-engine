#pragma once

#include <iostream>
#include <chrono>
#include <string_view>

constexpr bool ENABLE_PROFILING = true;

class LumenTimer {
public:
  explicit LumenTimer(std::string_view name) 
  : m_name(name), m_start(std::chrono::high_resolution_clock::now()) {}

  ~LumenTimer() {
    if constexpr (ENABLE_PROFILING){
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration<double, std::milli>(end - m_start).count();
      std::cout << "[PROFILE] " << m_name << ": " << duration << " ms" << std::endl;
    }
  }

private:
  std::string_view m_name;
  std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
};

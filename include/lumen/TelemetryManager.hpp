#pragma once

#include "InferenceTrace.hpp"
#include <vector>
#include <mutex>
#include <memory>
#include <thread>
#include <atomic>
#include <condition_variable>

namespace lumen {

class TelemetryManager {
private:
    TelemetryManager(); 
    ~TelemetryManager();

    void reporting_loop();

    void process_snapshot(std::vector<std::unique_ptr<InferenceTrace>>& batch);

    std::mutex m_mutex;
    std::vector<std::unique_ptr<InferenceTrace>> m_trace_buffer;

    std::atomic<bool> m_running;
    std::thread m_reporter_thread;

public:
    static TelemetryManager& get() {
        static TelemetryManager instance;
        return instance;
    }

    TelemetryManager(const TelemetryManager&) = delete;
    TelemetryManager& operator=(const TelemetryManager&) = delete;

    void capture_trace(std::unique_ptr<InferenceTrace> trace);

    void shutdown();
};
}

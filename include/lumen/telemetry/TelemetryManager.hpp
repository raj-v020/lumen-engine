#pragma once

#include <lumen/core/InferenceTrace.hpp>
#include <vector>
#include <mutex>
#include <memory>
#include <thread>
#include <atomic>
#include <condition_variable>

namespace lumen {
namespace telemetry {
class TelemetryManager {
private:
    TelemetryManager(); 
    ~TelemetryManager();

    void reporting_loop();

    void process_snapshot(std::vector<std::unique_ptr<core::InferenceTrace>>& batch);

    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::vector<std::unique_ptr<core::InferenceTrace>> m_trace_buffer;

    std::atomic<bool> m_running;
    std::thread m_reporter_thread;

public:
    static TelemetryManager& get() {
        static TelemetryManager instance;
        return instance;
    }

    TelemetryManager(const TelemetryManager&) = delete;
    TelemetryManager& operator=(const TelemetryManager&) = delete;

    void capture_trace(std::unique_ptr<core::InferenceTrace> trace);

    void shutdown();
};

}
}

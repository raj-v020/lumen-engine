#include <lumen/telemetry/TelemetryManager.hpp>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <lumen/utils/LumenConfigManager.hpp>

namespace lumen {
namespace telemetry {

TelemetryManager::TelemetryManager() : m_running(true) {
    m_reporter_thread = std::thread(&TelemetryManager::reporting_loop, this);
}

TelemetryManager::~TelemetryManager() {
    shutdown();
}

void TelemetryManager::shutdown() {
    if (m_running) {
        m_running = false;
        m_cv.notify_all();
        if (m_reporter_thread.joinable()) {
            m_reporter_thread.join();
        }
    }
}

void TelemetryManager::capture_trace(std::unique_ptr<core::InferenceTrace> trace) {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_trace_buffer.push_back(std::move(trace));
    }
    m_cv.notify_one();
}

void TelemetryManager::reporting_loop() {
    while (m_running) {
        std::vector<std::unique_ptr<core::InferenceTrace>> local_batch;
        
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv.wait_for(lock, std::chrono::seconds(1), [this] {
                return !m_trace_buffer.empty() || !m_running;
            });

            if (!m_running && m_trace_buffer.empty()) break;

            local_batch.swap(m_trace_buffer);
        }

        if (!local_batch.empty()) {
            process_snapshot(local_batch);
        }
    }
}

void TelemetryManager::process_snapshot(std::vector<std::unique_ptr<core::InferenceTrace>>& batch) {
    size_t count = batch.size();
    if (count == 0) return;

    double sum_e2e = 0.0, sum_queue = 0.0, sum_pre = 0.0, sum_infer = 0.0, sum_post = 0.0;
    std::vector<double> latencies;
    latencies.reserve(count);

    auto& config = lumen::utils::ConfigManager::get();
    std::string path = config.get_telemetry_csv_path();

    for (const auto& t : batch) {
        if (t) {
            save_to_csv(*t, path);
            latencies.push_back(t->total_e2e_ms);

            sum_e2e   += t->total_e2e_ms;
            sum_queue += t->queue_wait_ms;
            sum_pre   += t->preprocess_ms;
            sum_infer += t->inference_ms;
            sum_post  += t->postprocess_ms;
        }
    }

    size_t p99_idx = static_cast<size_t>(count * 0.99);
    if (p99_idx >= count) p99_idx = count - 1;
    std::nth_element(latencies.begin(), latencies.begin() + p99_idx, latencies.end());
    double p99_val = latencies[p99_idx];

    std::cout << "\033[1;32m[LUMEN TELEMETRY]\033[0m " 
              << "Batch: " << count << " reqs | "
              << "Avg E2E: " << std::fixed << std::setprecision(2) << (sum_e2e / count) << "ms | "
              << "P99: " << p99_val << "ms\n"
              << "  └─ Breakdown: [Queue: " << (sum_queue / count) 
              << "ms, Pre: " << (sum_pre / count) 
              << "ms, Infer: " << (sum_infer / count) 
              << "ms, Post: " << (sum_post / count) << "ms]" 
              << std::endl;
}

void TelemetryManager::save_to_csv(const core::InferenceTrace& trace, const std::string& path) {
    bool exists = std::filesystem::exists(path);
    std::ofstream file(path, std::ios::app);
    
    if (!file.is_open()) return;

    auto& config = lumen::utils::ConfigManager::get();
    std::string q_name = (config.get_queue_type() == utils::QueueType::LOCK_FREE_RING) ? "LockFree" : "Naive";
    std::string a_name = (config.get_alloc_type() == utils::AllocatorType::LUMEN_ARENA) ? "Arena" : "Standard";

    if (!exists || std::filesystem::file_size(path) == 0) {
        file << "timestamp,request_id,queue_type,alloc_type,queue_ms,preprocess_ms,inference_ms,postprocess_ms,total_ms\n";
    }

    double total = trace.queue_wait_ms + trace.preprocess_ms + trace.inference_ms + trace.postprocess_ms;

    file << std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) << ","
         << trace.request_id << ","
         << q_name << ","
         << a_name << ","
         << std::fixed << std::setprecision(4)
         << trace.queue_wait_ms << ","
         << trace.preprocess_ms << ","
         << trace.inference_ms << ","
         << trace.postprocess_ms << ","
         << total << "\n";
}

}
}

#include <iostream>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include "TelemetryManager.hpp"

namespace lumen {

TelemetryManager::TelemetryManager() : m_running(true) {
    m_reporter_thread = std::thread(&TelemetryManager::reporting_loop, this);
}

TelemetryManager::~TelemetryManager() {
    shutdown();
}

void TelemetryManager::shutdown() {
    if (m_running) {
        m_running = false;
        if (m_reporter_thread.joinable()) {
            m_reporter_thread.join();
        }
    }
}

void TelemetryManager::capture_trace(std::unique_ptr<InferenceTrace> trace) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_trace_buffer.push_back(std::move(trace));
}

void TelemetryManager::reporting_loop() {
    while (m_running) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (!m_running) break;

        std::vector<std::unique_ptr<InferenceTrace>> local_batch;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            local_batch.swap(m_trace_buffer);
        }

        if (!local_batch.empty()) {
            process_snapshot(local_batch);
        }
    }
}

void TelemetryManager::process_snapshot(std::vector<std::unique_ptr<InferenceTrace>>& batch) {
    size_t count = batch.size();
    if (count == 0) return;

    double sum_e2e = 0.0;
    double sum_queue = 0.0;
    double sum_pre = 0.0;
    double sum_infer = 0.0;
    double sum_post = 0.0;

    std::vector<double> latencies;
    latencies.reserve(count);

    for (const auto& t : batch) {
        latencies.push_back(t->total_e2e_ms);
        
        sum_e2e   += t->total_e2e_ms;
        sum_queue += t->queue_wait_ms;
        sum_pre   += t->preprocess_ms;
        sum_infer += t->inference_ms;
        sum_post  += t->postprocess_ms;
    }

    double avg_e2e   = sum_e2e / count;
    double avg_queue = sum_queue / count;
    double avg_pre   = sum_pre / count;
    double avg_infer = sum_infer / count;
    double avg_post  = sum_post / count;

    size_t p99_index = static_cast<size_t>(count * 0.99);
    if (p99_index >= count) p99_index = count - 1;
    std::nth_element(latencies.begin(), latencies.begin() + p99_index, latencies.end());
    double p99_latency = latencies[p99_index];

    std::cout << "[TELEMETRY] RPS: " << count 
              << " | Avg: " << std::fixed << std::setprecision(2) << avg_e2e << "ms"
              << " (Q: " << avg_queue 
              << ", Pre: " << avg_pre 
              << ", Inf: " << avg_infer 
              << ", Post: " << avg_post << ")"
              << " | P99: " << p99_latency << "ms" 
              << std::endl;
}
}

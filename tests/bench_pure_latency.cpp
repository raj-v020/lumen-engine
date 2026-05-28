#include <algorithm>
#include <atomic>
#include <cstdint>
#include <iostream>
#include <lumen/concurrency/MPMCTaskQueue.hpp>
#include <lumen/core/InferenceTask.hpp>
#include <thread>
#include <vector>
#include <x86intrin.h>

// Preprocessor guards allow environment overrides via compiler flags (e.g.,
// -DNUM_PRODUCERS=1)
#ifndef NUM_PRODUCERS
static constexpr int NUM_PRODUCERS = 4;
#endif

#ifndef NUM_CONSUMERS
static constexpr int NUM_CONSUMERS = 4;
#endif

#ifndef CPU_HZ
static constexpr uint64_t CPU_HZ =
    3000000000ULL; // Default 3.0 GHz nominal frequency
#endif

static constexpr int TASKS_PER_PRODUCER = 100000;
static constexpr int TOTAL_TASKS = NUM_PRODUCERS * TASKS_PER_PRODUCER;

// Expected CPU nominal frequency for SLA conversion (e.g., 3.0 GHz)
// 75ms @ 3.0 GHz = 0.075 * 3,000,000,000 = 225,000,000 clock cycles
static constexpr uint64_t CYCLE_THRESHOLD = (75 * CPU_HZ) / 1000;

std::atomic<bool> g_running{true};

int main() {
  lumen::concurrency::MPMCTaskQueue q;
  std::atomic<bool> start_signal{false};
  std::atomic<int> successfully_consumed{0};
  std::vector<std::vector<uint64_t>> producer_latencies(NUM_PRODUCERS);
  std::vector<std::vector<uint64_t>> consumer_latencies(NUM_CONSUMERS);

  std::vector<std::thread> producers;
  for (int p = 0; p < NUM_PRODUCERS; p++) {
    producers.emplace_back([&, p]() {
      // PreAllocate Tasks
      std::vector<std::unique_ptr<lumen::core::InferenceTask>> tasks;
      tasks.reserve(TASKS_PER_PRODUCER);
      for (int t = 0; t < TASKS_PER_PRODUCER; t++) {
        std::vector<uint8_t> dummy_bytes(10, 0);
        tasks.push_back(std::make_unique<lumen::core::InferenceTask>(
            t, std::move(dummy_bytes), nullptr, nullptr));
      }

      // Spin Wait for start signal
      while (!start_signal.load()) {
        std::this_thread::yield();
      }

      // Push tasks and calucated latencies
      int i = 0;
      while (i < TASKS_PER_PRODUCER) {
        uint64_t t1 = __rdtsc();
        __builtin_ia32_lfence();

        // Attempt push. If it succeeds, ownership transfers and returns true.
        // If it fails, tasks[i] is left untouched because the move boundary
        // aborted.
        if (q.push(std::move(*tasks[i]))) {
          __builtin_ia32_lfence();
          uint64_t t2 = __rdtsc();

          // Log the actual latency into the producer matrix slot
          producer_latencies[p].push_back(t2 - t1);
          i++; // Only advance to the next pre-allocated task on verified
               // success
        } else {
          // Queue was full, back off instantly and retry the exact same index
          // 'i'
          std::this_thread::yield();
        }
      }
    });

    // Set Core Affinity for Producer: Map sequentially to even logical cores
    // (0, 2, 4, 6)
    int core_id = p * 2;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(producers.back().native_handle(), sizeof(cpu_set_t),
                           &cpuset);
  }

  std::vector<std::thread> consumers;
  for (int c = 0; c < NUM_CONSUMERS; c++) {
    consumers.emplace_back([&, c]() {
      while (!start_signal.load()) {
        std::this_thread::yield();
      }

      while (true) {
        uint64_t t1 = __rdtsc();
        __builtin_ia32_lfence();

        auto tsk = q.pop();

        __builtin_ia32_lfence();
        uint64_t t2 = __rdtsc();

        if (tsk) {
          consumer_latencies[c].push_back(t2 - t1);
          successfully_consumed.fetch_add(1, std::memory_order_release);
        } else {
          if (successfully_consumed.load(std::memory_order_acquire) ==
              TOTAL_TASKS)
            break;
          std::this_thread::yield();
        }
      }
    });

    // Set Core Affinity for Consumer: Map to sibling odd logical cores (1, 3,
    // 5, 7)
    int core_id = (c * 2) + 1;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(consumers.back().native_handle(), sizeof(cpu_set_t),
                           &cpuset);
  }

  start_signal.store(true, std::memory_order_release);

  for (int p = 0; p < NUM_PRODUCERS; p++) {
    producers[p].join();
  }

  for (int c = 0; c < NUM_CONSUMERS; c++) {
    consumers[c].join();
  }

  size_t total_size = 0;
  for (const auto &row : producer_latencies) {
    total_size += row.size();
  }
  for (const auto &row : consumer_latencies) {
    total_size += row.size();
  }

  std::vector<uint64_t> flat;
  flat.reserve(total_size);

  for (const auto &row : producer_latencies) {
    flat.insert(flat.end(), row.begin(), row.end());
  }
  for (const auto &row : consumer_latencies) {
    flat.insert(flat.end(), row.begin(), row.end());
  }

  std::sort(flat.begin(), flat.end());

  size_t p95_idx = static_cast<size_t>(flat.size() * 0.95);
  size_t p99_idx = static_cast<size_t>(flat.size() * 0.99);

  uint64_t p95_cycles = flat[p95_idx];
  uint64_t p99_cycles = flat[p99_idx];

  // Convert cycles back to human-readable milliseconds based on CPU frequency
  double p95_ms = (static_cast<double>(p95_cycles) / CPU_HZ) * 1000.0;
  double p99_ms = (static_cast<double>(p99_cycles) / CPU_HZ) * 1000.0;

  std::cout << "================ LUMEN CYCLES REPORT ================\n";
  std::cout << "P95 Latency: " << p95_cycles << " cycles (" << p95_ms
            << " ms)\n";
  std::cout << "P99 Latency: " << p99_cycles << " cycles (" << p99_ms
            << " ms)\n";

  if (p99_cycles > CYCLE_THRESHOLD) {
    std::cerr << "FAILURE: P99 Tail Latency breached 75ms SLA limit!\n";
    return 1;
  }

  std::cout << "SUCCESS: Performance profile fits safely within SLA limits.\n";
  return 0;
}

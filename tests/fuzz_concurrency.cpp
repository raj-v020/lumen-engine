#include <iostream>
#include <lumen/concurrency/MPMCTaskQueue.hpp>
#include <lumen/core/InferenceTask.hpp>
#include <thread>
#include <vector>

#include <atomic>

static constexpr int NUM_PRODUCERS = 4;
static constexpr int NUM_CONSUMERS = 4;
static constexpr int TASKS_PER_PRODUCER = 100000;
static constexpr int TOTAL_TASKS = NUM_PRODUCERS * TASKS_PER_PRODUCER;

std::atomic<bool> g_running{true};

int main() {
  lumen::concurrency::MPMCTaskQueue q;
  std::atomic<bool> start_signal{false};
  std::vector<std::atomic<int>> global_checksum(TOTAL_TASKS);

  for (auto &checksum : global_checksum) {
    checksum.store(0);
  }

  std::atomic<int> successfully_consumed{0};
  std::atomic<int> production_failures{0};

  std::vector<std::thread> producers;
  for (int p = 0; p < NUM_PRODUCERS; p++) {
    producers.emplace_back([&, p]() {
      while (!start_signal.load()) {
        std::this_thread::yield();
      }
      int start_id = p * TASKS_PER_PRODUCER;
      for (int i = 0; i < TASKS_PER_PRODUCER; i++) {
        int fd = start_id + i;
        std::vector<uint8_t> dummy_bytes(10, 0);

        while (!q.push(lumen::core::InferenceTask(fd, std::move(dummy_bytes),
                                                  nullptr, nullptr))) {
          production_failures.fetch_add(1);
          std::this_thread::yield();
        }
      }
    });
  }

  std::vector<std::thread> consumers;
  for (int c = 0; c < NUM_CONSUMERS; c++) {
    consumers.emplace_back([&, c]() {
      while (!start_signal.load()) {
        std::this_thread::yield();
      }

      while (true) {
        auto tsk = q.pop();
        if (tsk) {
          int fd = tsk->client_fd;
          global_checksum[fd].fetch_add(1, std::memory_order_relaxed);
          successfully_consumed.fetch_add(1, std::memory_order_release);
        } else {
          if (successfully_consumed.load(std::memory_order_acquire) ==
              TOTAL_TASKS)
            break;
          std::this_thread::yield();
        }
      }
    });
  }

  start_signal.store(true);

  for (int p = 0; p < NUM_PRODUCERS; p++) {
    producers[p].join();
  }

  for (int c = 0; c < NUM_CONSUMERS; c++) {
    consumers[c].join();
  }

  bool test_failed = false;
  for (int t = 0; t < TOTAL_TASKS; t++) {
    int flag = global_checksum[t].load();

    if (flag == 0) {
      std::cerr << "Task " << t << " Dropped\n";
      test_failed = true;
    } else if (flag > 1) {
      std::cerr << "RACE CONDITION: At task " << t << "\n";
      test_failed = true;
    }
  }
  if (test_failed) {
    std::cerr
        << "FAILURE: Concurrency verification failed with active data races\n";
    return 1;
  }
  std::cout
      << "SUCCESS: Fuzz test executed successfully with zero data races\n";
  return 0;
}

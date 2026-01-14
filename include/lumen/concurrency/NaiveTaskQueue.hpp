#pragma once

#include <lumen/interfaces/ITaskQueue.hpp>
#include <queue>
#include <mutex>
#include <condition_variable>

namespace lumen {
namespace concurrency {

class NaiveTaskQueue : public interfaces::ITaskQueue {
private:
    std::queue<core::InferenceTask> data;
    mutable std::mutex guard;
    std::condition_variable cv;
    bool stop = false;

public:
    void push(core::InferenceTask task) override {
        std::lock_guard<std::mutex> lock(guard);
        data.push(std::move(task));
        cv.notify_one();
    }

        std::optional<core::InferenceTask> pop() override {
            std::unique_lock<std::mutex> lock(guard);
            cv.wait(lock, [this] { 
                return !data.empty() || stop; 
            });

            if (data.empty()) return std::nullopt;

            core::InferenceTask task = std::move(data.front());
            data.pop();
            return task;
        }

        size_t pop_batch(std::vector<core::InferenceTask>& batch, size_t max_batch_size) override {
            std::lock_guard<std::mutex> lock(guard);
            size_t count = 0;

            while (!data.empty() && count < max_batch_size) {
                batch.push_back(std::move(data.front()));
                data.pop();
                count++;
            }
            return count;
        }

        size_t size() const override {
            std::lock_guard<std::mutex> lock(guard);
            return data.size();
        }

        bool empty() const override {
            std::lock_guard<std::mutex> lock(guard);
            return data.empty();
        }

        void shutdown() override {
            {
                std::lock_guard<std::mutex> lock(guard); 
                stop = true;
            }
            cv.notify_all();
        }
    };

}
}


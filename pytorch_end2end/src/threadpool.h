#pragma once

#include <thread>
#include <mutex>
#include <queue>
#include <functional>
#include <condition_variable>

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads_);

    void add_task(std::function<void()>&& task);

    ~ThreadPool();

private:
    void task_runner();

    bool working;
    size_t num_threads;

    std::vector<std::thread> pool;
    std::queue<std::function<void()>> tasks;
    std::mutex tasks_mutex;
    std::condition_variable tasks_condition;
};

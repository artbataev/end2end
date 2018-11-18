#pragma once

#include <thread>
#include <mutex>
#include <atomic>
#include <queue>
#include <functional>
#include <condition_variable>

class ThreadPool {
public:
    void add_task(const std::function<void()>& task);

    void configure_threads(size_t num_threads_);

    void suspend_work();

    void resume_work();

    ~ThreadPool();

private:
    void task_runner(std::mutex& tasks_mutex,
                     std::condition_variable& condition,
                     std::queue<std::function<void()>>& tasks);

    bool working = false;
    size_t num_threads;

    std::vector<std::thread> pool;
    std::queue<std::function<void()>> tasks;
    std::mutex tasks_mutex;
    std::condition_variable tasks_condition;
};

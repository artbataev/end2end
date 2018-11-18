#include "threadpool.h"

void ThreadPool::add_task(const std::function<void()>& task) {
    {
        std::lock_guard<std::mutex> guard{tasks_mutex};
        tasks.push(task);
    }
    tasks_condition.notify_one();
}

void ThreadPool::configure_threads(size_t num_threads_) {
    suspend_work();
    num_threads = num_threads_;
    resume_work();
}

void ThreadPool::suspend_work() {
    if (working) {
        working = false;
        tasks_condition.notify_all();
        for (auto& t: pool)
            if (t.joinable())
                t.join();
    }
    pool.resize(0);
}

void ThreadPool::resume_work() {
    if (!working) {
        for (int _ = 0; _ < num_threads; _++) {
            pool.emplace_back(std::thread([&] {
                this->task_runner(tasks_mutex, tasks_condition, tasks);
            }));
        }
        working = true;
    }
}

ThreadPool::~ThreadPool() {
   suspend_work();
}

void ThreadPool::task_runner(std::mutex& tasks_mutex,
                             std::condition_variable& condition,
                             std::queue<std::function<void()>>& tasks) {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock{tasks_mutex};
            while (working && tasks.empty())
                condition.wait(lock);
            if (!tasks.empty()) {
                task = tasks.front();
                tasks.pop();
            } else if (!working) {
                return;
            }
        }
        task();
    }
}

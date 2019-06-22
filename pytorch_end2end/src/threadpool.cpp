// Copyright 2019 Vladimir Bataev

#include "threadpool.h"

#include <utility>

ThreadPool::ThreadPool(size_t num_threads_)
    : num_threads{num_threads_}, working{true} {
  pool.reserve(num_threads);
  for (int _ = 0; _ < num_threads; _++) {
    pool.emplace_back(std::thread([&] { task_runner(); }));
  }
}

void ThreadPool::add_task(std::function<void()>&& task) {
  {
    std::lock_guard<std::mutex> guard{tasks_mutex};
    tasks.emplace(task);
  }
  tasks_condition.notify_one();
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> guard{tasks_mutex};
    working = false;
  }
  tasks_condition.notify_all();
  for (auto& t : pool) t.join();
}

void ThreadPool::task_runner() {
  while (true) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock{tasks_mutex};
      tasks_condition.wait(lock, [&] { return !working || !tasks.empty(); });
      if (!working && tasks.empty()) return;
      task = std::move(tasks.front());
      tasks.pop();
    }
    task();
  }
}

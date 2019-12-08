// Copyright 2019 Vladimir Bataev

#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class ThreadPool {
 public:
  explicit ThreadPool(size_t num_threads_);

  void add_task(std::function<void()>&& task);

  ~ThreadPool();

 private:
  void task_runner();

  size_t num_threads;
  bool working;

  std::vector<std::thread> pool;
  std::queue<std::function<void()>> tasks;
  std::mutex tasks_mutex;
  std::condition_variable tasks_condition;
};

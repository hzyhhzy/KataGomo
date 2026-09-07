#ifndef NEURALNET_NNBATCHDISPATCH_H_
#define NEURALNET_NNBATCHDISPATCH_H_

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

// GPU-aware dispatch, following doomoooo/KataGomo_fork's final-migration
// NNBatchingDispatcher (5dfd8cb16). An idle device may start a partial batch;
// a busy device waits for a full batch, without any fixed sleep or timeout.
// Padding, if appropriate for a particular backend, is a separate policy.
//
// The queue retains its own lock. Every external queue mutation and every
// change to rowCap must be followed by notify(), AFTER releasing the queue's
// lock. The mutex handshake in notify() prevents a lost wakeup between our
// observation of the queue and condition.wait(). Only this dispatcher may pop
// from the queue while enabled, INCLUDING workers with preferFullBatch=false.
// Otherwise another consumer could empty the observed queue while we hold
// this mutex and make waitPopUpToN block, preventing producer notifications.
//
// Mode changes and mapping resets require no workers in dispatch/inference:
// either all workers are stopped or all are held at the startup barrier.
// Destruction requires all worker threads to be stopped.
// CUDA's default device (-1) is an alias for device 0, not a separate GPU.
class NNBatchingDispatcher {
 public:
  NNBatchingDispatcher(bool enabled_, const std::vector<int>& gpuIds)
    : enabled(enabled_), gpuIdxByServerThread(normalizeGpuIds(gpuIds)),
      serverThreadHasActiveBatch(gpuIds.size(), false) {}

  NNBatchingDispatcher(const NNBatchingDispatcher&) = delete;
  NNBatchingDispatcher& operator=(const NNBatchingDispatcher&) = delete;

  // The evaluator decides this after every worker reports its eligibility at
  // the startup barrier, before ANY worker enters waitForBatch. This lets an
  // all-generic evaluator keep the original queue path without an extra lock.
  // Notifications from producers may overlap this startup decision, hence the
  // atomic flag. Worker startup/join barriers, not this relaxed flag, provide
  // the synchronization required for a mode change.
  void setEnabled(bool enabled_) {
    std::lock_guard<std::mutex> lock(mutex);
    for(bool active : serverThreadHasActiveBatch) {
      if(active)
        throw std::logic_error("NN batch dispatcher: changed mode with an active batch");
    }
    enabled.store(enabled_, std::memory_order_relaxed);
  }

  template<typename Queue, typename T>
  bool waitForBatch(
    Queue& queue, std::vector<T>& resultBufs, int maxBatchSize,
    const std::atomic<int>& rowCap, int serverThreadIdx, bool preferFullBatch
  ) {
    if(!enabled.load(std::memory_order_relaxed))
      return queue.waitPopUpToN(resultBufs, desiredBatchSize(maxBatchSize, rowCap));

    std::unique_lock<std::mutex> lock(mutex);
    checkLane(serverThreadIdx);
    if(serverThreadHasActiveBatch[serverThreadIdx])
      throw std::logic_error("NN batch dispatcher: lane already owns an active batch");

    while(true) {
      if(queue.isClosed())
        return false;
      const std::size_t queuedRows = queue.size();
      if(queuedRows > 0) {
        const int desired = desiredBatchSize(maxBatchSize, rowCap);
        if(!preferFullBatch || queuedRows >= static_cast<std::size_t>(desired) ||
           deviceIsIdle(serverThreadIdx)) {
          // All consumers use this mutex. A producer may add rows, or close
          // the queue, but no other consumer can steal the observed rows.
          const bool gotAnything = queue.waitPopUpToN(resultBufs, desired);
          if(gotAnything)
            serverThreadHasActiveBatch[serverThreadIdx] = true;
          return gotAnything;
        }
      }
      else if(queue.isReadOnly() || queue.isClosed())
        return false;
      condition.wait(lock);
    }
  }

  // Call exactly once for each successful waitForBatch, after inference and
  // result publication. An RAII guard in the caller should cover all paths.
  void completeBatch(int serverThreadIdx) {
    if(!enabled.load(std::memory_order_relaxed))
      return;
    {
      std::lock_guard<std::mutex> lock(mutex);
      checkLane(serverThreadIdx);
      if(!serverThreadHasActiveBatch[serverThreadIdx])
        throw std::logic_error("NN batch dispatcher: lane has no active batch to complete");
      serverThreadHasActiveBatch[serverThreadIdx] = false;
    }
    condition.notify_all();
  }

  void notify() {
    if(!enabled.load(std::memory_order_relaxed))
      return;
    {
      std::lock_guard<std::mutex> lock(mutex);
    }
    condition.notify_all();
  }

  // No workers may be running or waiting during reconfiguration. Reject
  // unfinished ownership as an additional check, and clear all old state.
  void resetGpuIdxByServerThread(const std::vector<int>& gpuIds) {
    std::vector<int> normalized = normalizeGpuIds(gpuIds);
    std::lock_guard<std::mutex> lock(mutex);
    for(bool active : serverThreadHasActiveBatch) {
      if(active)
        throw std::logic_error("NN batch dispatcher: reconfigured with an active batch");
    }
    gpuIdxByServerThread = std::move(normalized);
    serverThreadHasActiveBatch.assign(gpuIds.size(), false);
  }

 private:
  std::atomic<bool> enabled;
  std::vector<int> gpuIdxByServerThread;
  std::vector<bool> serverThreadHasActiveBatch;
  std::mutex mutex;
  std::condition_variable condition;

  static std::vector<int> normalizeGpuIds(const std::vector<int>& gpuIds) {
    std::vector<int> result = gpuIds;
    for(int& gpu : result) {
      if(gpu < -1)
        throw std::invalid_argument("NN batch dispatcher: invalid GPU index");
      if(gpu == -1)
        gpu = 0;
    }
    return result;
  }

  static int desiredBatchSize(int maxBatchSize, const std::atomic<int>& rowCap) {
    const int cap = rowCap.load(std::memory_order_acquire);
    if(maxBatchSize <= 0 || cap <= 0)
      throw std::invalid_argument("NN batch dispatcher: batch size must be positive");
    return std::min(maxBatchSize, cap);
  }

  void checkLane(int lane) const {
    if(lane < 0 || static_cast<std::size_t>(lane) >= gpuIdxByServerThread.size())
      throw std::out_of_range("NN batch dispatcher: invalid lane index");
  }

  bool deviceIsIdle(int lane) const {
    const int gpu = gpuIdxByServerThread[lane];
    for(std::size_t i = 0; i < serverThreadHasActiveBatch.size(); i++) {
      if(gpuIdxByServerThread[i] == gpu && serverThreadHasActiveBatch[i])
        return false;
    }
    return true;
  }
};

#endif  // NEURALNET_NNBATCHDISPATCH_H_

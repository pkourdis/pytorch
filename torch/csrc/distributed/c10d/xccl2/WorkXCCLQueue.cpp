#ifdef USE_C10D_XCCL

#include <torch/csrc/distributed/c10d/xccl2/WorkXCCL.hpp>

namespace c10d::xccl2 {

namespace {
bool isFailure(WorkXCCL::WorkStatus status) {
  return status == WorkXCCL::WorkStatus::TIMEDOUT ||
      status == WorkXCCL::WorkStatus::ERROR;
}
} // namespace

WorkXCCL::WorkStatus WorkXCCLQueue::garbageCollectLocked(
    std::vector<std::shared_ptr<WorkXCCL::State>>& completed) {
  WorkXCCL::WorkStatus overall_status = WorkXCCL::WorkStatus::COMPLETED;

  // Keep popping completed elements until we hit an in-progress element
  // or the queue is empty
  // Use an iterator to safely remove empty queues while iterating
  auto it = stream_work_queues_.begin();
  while (it != stream_work_queues_.end()) {
    auto& work_queue = it->second;

    while (!work_queue.empty()) {
      auto& work = work_queue.front();

      // Use the checkStatus function to determine the work status
      WorkXCCL::WorkStatus status = work.state->checkStatus();

      if (status == WorkXCCL::WorkStatus::COMPLETED) {
        // Tensor references must be released by a caller thread, not by the
        // watchdog that runs garbageCollect(), so park the inputs here and let
        // the next enqueueWork() clear them. The state holds no tensors and is
        // handed to the caller for its completion hooks.
        completedInputTensors_.push(std::move(work.inputTensors));
        completed.push_back(std::move(work.state));
        work_queue.pop();
        // Continue to the next element in the queue
      } else if (isFailure(status)) {
        // Report the first failure, but keep draining the other streams so
        // their completed work still reaches the completion hooks.
        if (!isFailure(overall_status)) {
          overall_status = status;
        }
        break;
      } else {
        // NOT_STARTED or INPROGRESS - stop processing this queue
        if (overall_status == WorkXCCL::WorkStatus::COMPLETED) {
          overall_status = status;
        }
        break;
      }
    }

    // If the queue is now empty, remove it from the map
    if (work_queue.empty()) {
      it = stream_work_queues_.erase(it);
    } else {
      ++it;
    }
  }

  return overall_status;
}

// Thread-safety: This method is called from the timeout watchdog thread while
// the main thread may be enqueuing work via enqueueWork(). The
// work_queues_mutex_ ensures proper synchronization - both garbageCollect() and
// enqueueWork() acquire the mutex before accessing stream_work_queues_.
WorkXCCL::WorkStatus WorkXCCLQueue::garbageCollect(
    std::vector<std::shared_ptr<WorkXCCL::State>>& completed) {
  std::lock_guard<std::mutex> lock(work_queues_mutex_);
  return garbageCollectLocked(completed);
}

// Must be called on a caller thread, without work_queues_mutex_ held.
void WorkXCCLQueue::clearCompletedInputTensors() {
  std::queue<std::shared_ptr<WorkXCCL::InputTensorShelf>> completed;
  {
    std::lock_guard<std::mutex> lock(work_queues_mutex_);
    completed.swap(completedInputTensors_);
  }
  while (!completed.empty()) {
    completed.front()->clear();
    completed.pop();
  }
}

WorkXCCL::WorkStatus WorkXCCLQueue::finalize(
    std::vector<std::shared_ptr<WorkXCCL::State>>& completed) {
  // Because this function is typically called after the timeout thread has
  // already joined, we might not need to lock here.  But doing the lock anyway,
  // as defensive programming, just in case someone moves the thread join order
  // later.  The cost of the lock itself should be small on modern linux systems
  // (uncontended locks are typically just an atomic operation).
  WorkXCCL::WorkStatus status = WorkXCCL::WorkStatus::COMPLETED;
  {
    std::lock_guard<std::mutex> lock(work_queues_mutex_);

    // Initialize the status to COMPLETED to cover the case where the queue is
    // empty
    while (!stream_work_queues_.empty()) {
      status = garbageCollectLocked(completed);
      if (status == WorkXCCL::WorkStatus::ERROR ||
          status == WorkXCCL::WorkStatus::TIMEDOUT ||
          status == WorkXCCL::WorkStatus::COMPLETED) {
        break;
      }
    }

    // NOTE: finalize MUST return without holding references to any work state
    // or tensor, otherwise it may leak them and cause side effects.
    stream_work_queues_.clear();
  }
  clearCompletedInputTensors();

  return status;
}

bool WorkXCCLQueue::empty() {
  std::lock_guard<std::mutex> lock(work_queues_mutex_);
  return stream_work_queues_.empty();
}

void WorkXCCLQueue::enqueueWork(
    c10::intrusive_ptr<WorkXCCL> work,
    c10::xpu::XPUStream stream) {
  {
    std::lock_guard<std::mutex> lock(work_queues_mutex_);
    // Add work to stream's queue after events have been recorded
    stream_work_queues_[stream].push({work->state_, work->inputTensors_});
  }
  clearCompletedInputTensors();
}

} // namespace c10d::xccl2

#endif // USE_C10D_XCCL

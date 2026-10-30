// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#ifdef USE_C10D_XCCL

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <ATen/xpu/XPUEvent.h>
#include <c10/xpu/XPUStream.h>

#include <torch/csrc/distributed/c10d/Work.hpp>

namespace c10d::xccl2 {

class ProcessGroupXCCL;

// Pool of XPU events shared between a ProcessGroupXCCL and every work it
// creates. A work hands its events back here when destroyed, which can happen
// after the backend is gone: user code may still hold the handle returned by an
// async collective when destroy_process_group() runs. The pool is therefore
// owned jointly instead of being reached through a pointer to the backend.
class XcclEventPool {
 public:
  explicit XcclEventPool(size_t max_size) : max_size_(max_size) {}

  [[nodiscard]] std::unique_ptr<at::xpu::XPUEvent> get();
  void put(std::unique_ptr<at::xpu::XPUEvent> event);
  // Release pooled events at finalize() without waiting for works that are
  // still holding theirs; those are freed when the last work goes away.
  void clear();

 private:
  std::queue<std::unique_ptr<at::xpu::XPUEvent>> pool_;
  std::mutex mutex_;
  const size_t max_size_;
};

// The backend identity a work still needs once it has been issued: the logging
// prefixes and the device whose current stream wait() orders against. Copied at
// construction, for the same lifetime reason as XcclEventPool. The accessors
// exist to satisfy the TC_LOG prefix helpers, which are templated on the comm.
struct XcclCommInfo {
  std::string name;
  int rank{};
  int size{};
  at::Device device{at::kXPU, 0};

  std::string_view getCommName() const {
    return name;
  }
  int getRank() const {
    return rank;
  }
};

// Work object for the XCCL TorchComms backend. Ported from torchcomms'
// TorchWorkXCCL, but rebased onto c10d::Work (upstream subclassed
// torchcomms::TorchWork). Completion is tracked with a pair of XPU events; the
// Future/result handling that the torchcomms BackendWrapper used to provide is
// folded in here (see setOutputs/getFuture). A work keeps no pointer to its
// backend: it can outlive it, so it snapshots the identity it needs and shares
// ownership of the event pool. (Upstream held a shared_ptr to the comm, which
// is incompatible with c10d's intrusive_ptr ownership of the backend.)
//
// Upstream reached the XPU driver through torchcomms' XpuApi indirection
// (eventRecord/eventQuery/streamWaitEvent). That layer is not ported: it exists
// to let torchcomms mock the driver in unit tests, and in-tree we can use
// at::xpu::XPUEvent directly, which wraps the same sycl::event.
class WorkXCCL : public c10d::Work {
 public:
  enum class WorkStatus {
    NOT_STARTED,
    INPROGRESS,
    COMPLETED,
    TIMEDOUT,
    ERROR,
  };

  WorkXCCL(
      ProcessGroupXCCL* comm,
      c10::xpu::XPUStream stream,
      std::chrono::milliseconds timeout_ms,
      const std::vector<at::Tensor>& inputTensors);
  WorkXCCL(
      ProcessGroupXCCL* comm,
      c10::xpu::XPUStream stream,
      std::chrono::milliseconds timeout_ms,
      at::Tensor inputTensor);
  ~WorkXCCL() override;

  WorkXCCL(const WorkXCCL&) = delete;
  WorkXCCL(WorkXCCL&&) = delete;
  WorkXCCL& operator=(const WorkXCCL&) = delete;
  WorkXCCL& operator=(WorkXCCL&&) = delete;

  // c10d::Work overrides.
  bool isCompleted() override;
  bool isSuccess() const override;
  bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;
  void synchronize() override;
  std::vector<at::Tensor> result() override;
  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

  std::chrono::milliseconds getTimeout() const {
    return timeout_ms_;
  }

  WorkStatus status() const {
    return status_.load(std::memory_order_relaxed);
  }

  // Output tensors for result()/getFuture(). Set by the backend after issuing.
  void setOutputs(std::vector<at::Tensor> outputs, bool asyncOp) {
    if (asyncOp) {
      outputs_ = std::move(outputs);
    }
  }
  // A coalesced c10d op issues one work per tensor but must hand back a single
  // handle; the last work adopts the rest so that they outlive the call.
  void setChildren(std::vector<c10::intrusive_ptr<WorkXCCL>> children) {
    children_ = std::move(children);
  }

 protected:
  void recordStart(std::string_view coll_name);
  void recordEnd();

  friend class ProcessGroupXCCL;
  friend class WorkXCCLQueue;

 private:
  void setStatus(WorkStatus status) {
    status_.store(status, std::memory_order_relaxed);
  }
  // Poll the XPU events and advance status; used by the GC queue + watchdog.
  WorkStatus checkStatus();
  void recordFunctionStart(std::string_view coll_name);
  // Make the current stream wait on the work's end event (the c10d "wait"
  // semantics for accelerator work: order subsequent current-stream ops after
  // this collective).
  void synchronizeInternal();

  std::vector<at::Tensor> inputTensors_;
  at::Tensor inputTensor_;
  std::vector<at::Tensor> outputs_;
  std::vector<c10::intrusive_ptr<WorkXCCL>> children_;

  XcclCommInfo comm_info_;
  std::shared_ptr<XcclEventPool> event_pool_;
  std::unique_ptr<at::xpu::XPUEvent> start_event_;
  std::unique_ptr<at::xpu::XPUEvent> end_event_;
  c10::xpu::XPUStream stream_; // not owned by this class

  std::chrono::milliseconds timeout_ms_;
  uint64_t seq_{0};

  std::atomic<WorkStatus> status_{WorkStatus::NOT_STARTED};
  std::optional<std::chrono::steady_clock::time_point> start_completed_time_;
  std::optional<at::RecordFunction> recordFunction_;
  c10::intrusive_ptr<c10::ivalue::Future> future_;

  // Set by the backend for a synchronous barrier: synchronizeInternal() then
  // host-blocks the CPU thread (in addition to the stream-ordered wait) to
  // mirror the stock XPU backend, whose barrier host-blocks. See barrierImpl.
  bool hostBlocking_{false};
};

class WorkXCCLQueue {
 public:
  WorkXCCLQueue() = default;
  ~WorkXCCLQueue() = default;

  WorkXCCL::WorkStatus garbageCollect();
  // Finalize function can only be called from the main thread
  WorkXCCL::WorkStatus finalize();
  void enqueueWork(
      c10::intrusive_ptr<WorkXCCL> work,
      c10::xpu::XPUStream stream);
  bool empty();

 private:
  WorkXCCL::WorkStatus garbageCollectLocked();
  std::unordered_map<
      c10::xpu::XPUStream,
      std::queue<c10::intrusive_ptr<WorkXCCL>>>
      stream_work_queues_;
  // Completed work is parked here so that the tensor references it holds are
  // released by a caller thread rather than by the watchdog that runs
  // garbageCollect(); see WorkXCCLQueue.cpp.
  std::queue<c10::intrusive_ptr<WorkXCCL>> completed_work_queue_;
  std::mutex work_queues_mutex_;
};

} // namespace c10d::xccl2

#endif // USE_C10D_XCCL

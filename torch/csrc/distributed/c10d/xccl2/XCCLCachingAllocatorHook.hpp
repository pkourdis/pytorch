#pragma once

#ifdef USE_C10D_XCCL

#include <mutex>
#include <set>
#include <unordered_map>

#include <c10/core/Device.h>
#include <c10/xpu/XPUCachingAllocator.h>

namespace c10d::xccl2 {

class ProcessGroupXCCL;

// Port of torchcomms' TorchCommXCCLCCA: a process-wide singleton that
// registers XPU caching-allocator segments with every live ProcessGroupXCCL
// via onecclCommRegister. It is only a performance hint; collectives are
// correct without it.
class XCCLCachingAllocatorHook {
 public:
  static XCCLCachingAllocatorHook& getInstance();

  void regDeregMem(const c10::CachingDeviceAllocator::TraceEntry& te);
  void registerComm(ProcessGroupXCCL* comm);
  void deregisterComm(ProcessGroupXCCL* comm);

 private:
  XCCLCachingAllocatorHook();

  // Seeds registeredMemMap_ with segments that already exist when the hook is
  // first constructed, so a comm created later still registers them.
  void registerMemPreHook();
  bool shouldTrackSegment(const c10::MempoolId_t& mempool_id) const;

  struct MemInfo {
    size_t len;
    c10::DeviceIndex device;
  };

  std::mutex mutex_;
  std::unordered_map<void*, MemInfo> registeredMemMap_;
  std::set<ProcessGroupXCCL*> registeredComms_;
  bool register_default_pool_segments_;
};

} // namespace c10d::xccl2

#endif // USE_C10D_XCCL

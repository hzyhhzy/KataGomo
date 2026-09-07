#pragma once
#include <cstdint>
#include <map>
#include <mutex>

// Generated CuTe/FA4 ABI common to all three inspected artifacts:
// Module contains one member named module (cudaLibrary_t), and both generated
// initialization entry points accept a void** argument list. Runtime is injected
// so the state machine can be tested without CUDA headers, a driver, or a GPU.
namespace b11aot {
enum class InitStage { Ready, CurrentDevice, DeviceProperties, Eligibility, Library, Device, Unload };

template<class Module, class Runtime>
class CurrentDeviceLoader {
public:
  using Error = typename Runtime::Error;
  using Entry = void (*)(void**);
  struct Result {
    Error error;
    InitStage stage;
    int device;
    bool ok() const { return error == Runtime::success(); }
  };

  CurrentDeviceLoader(Entry initializeLibrary, Entry initializeDevice)
    : initializeLibrary_(initializeLibrary), initializeDevice_(initializeDevice),
      libraryResult_{Runtime::success(), InitStage::Library, -1} {}
  CurrentDeviceLoader(const CurrentDeviceLoader&) = delete;
  CurrentDeviceLoader& operator=(const CurrentDeviceLoader&) = delete;

  // Call once during each ComputeHandle's setup, before timing/graph capture.
  // No cudaSetDevice, cudaGetDeviceCount, or enumeration of unrelated devices.
  // Errors are cached and returned, not printed, thrown, or discarded.
  Result prepareCurrentDevice() {
    int current = -1;
    Error error = Runtime::getDevice(&current);
    if(error != Runtime::success()) return {error, InitStage::CurrentDevice, current};
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = devices_.find(current);
    if(found != devices_.end()) return found->second;
    typename Runtime::DeviceProperties properties{};
    error = Runtime::getDeviceProperties(&properties, current);
    if(error != Runtime::success())
      return remember({error, InitStage::DeviceProperties, current});
    if(!Runtime::isEligible(properties))
      return remember({Runtime::unsupported(), InitStage::Eligibility, current});

    if(!libraryAttempted_) {
      libraryAttempted_ = true;
      auto* libraryPointer = &module_.module;
      error = Runtime::success();
      void* arguments[] = {&libraryPointer, &error};
      initializeLibrary_(arguments);
      libraryResult_ = {error, InitStage::Library, current};
    }
    if(!libraryResult_.ok())
      return remember({libraryResult_.error, InitStage::Library, current});

    auto* libraryPointer = &module_.module;
    std::int32_t device = current;
    error = Runtime::success();
    void* arguments[] = {&libraryPointer, &device, &error};
    initializeDevice_(arguments);
    return remember({error, error == Runtime::success() ? InitStage::Ready : InitStage::Device, current});
  }

  // Launch code may use this only after a successful prepareCurrentDevice for
  // its current device. Cache the Result in the owning handle; do not put this
  // mutex-bearing setup helper on every attention/FFN launch's hot path.
  Module* module() { return &module_; }

  // Optional explicit engine cleanup: all streams/handles must already be
  // drained/destroyed. Invoke before cudaDeviceReset, never from a static
  // destructor. On unload failure retain state and propagate the failure.
  Result shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    if(module_.module) {
      const Error error = Runtime::unload(module_.module);
      if(error != Runtime::success()) return {error, InitStage::Unload, -1};
    }
    module_ = Module{};
    devices_.clear();
    libraryAttempted_ = false;
    libraryResult_ = {Runtime::success(), InitStage::Library, -1};
    return {Runtime::success(), InitStage::Ready, -1};
  }

private:
  Result remember(Result result) {
    devices_.emplace(result.device, result);
    return result;
  }
  Module module_{};
  Entry initializeLibrary_;
  Entry initializeDevice_;
  std::mutex mutex_;
  bool libraryAttempted_ = false;
  Result libraryResult_;
  std::map<int, Result> devices_;
};
} // namespace b11aot

// CPU-only fake-runtime tests: no CUDA headers, libraries, driver, or GPU calls.
#include "aot_loader_core.h"
#include <cassert>
#include <thread>
#include <vector>

namespace {
struct FakeRuntime {
  using Error = int;
  struct DeviceProperties { bool eligible = false; };
  static thread_local int current;
  static int queryError, propertiesError, libraryError, deviceError, unloadError;
  static int libraryCalls, deviceCalls, unloadCalls;
  static std::vector<int> queried, loaded;
  static int success() { return 0; }
  static int unsupported() { return 801; }
  static int getDevice(int* device) { *device = current; return queryError; }
  static int getDeviceProperties(DeviceProperties* properties, int device) {
    queried.push_back(device); properties->eligible = device == 0 || device == 2; return propertiesError;
  }
  static bool isEligible(const DeviceProperties& properties) { return properties.eligible; }
  static int unload(void*) { ++unloadCalls; return unloadError; }
  static void reset() {
    queryError = propertiesError = libraryError = deviceError = unloadError = 0;
    libraryCalls = deviceCalls = unloadCalls = 0;
    queried.clear(); loaded.clear(); current = 0;
  }
  static void initialize(void** arguments) {
    ++libraryCalls;
    auto library = *static_cast<void***>(arguments[0]);
    if(!libraryError) *library = reinterpret_cast<void*>(0x1000);
    *static_cast<int*>(arguments[1]) = libraryError;
  }
  static void load(void** arguments) {
    ++deviceCalls;
    loaded.push_back(*static_cast<std::int32_t*>(arguments[1]));
    *static_cast<int*>(arguments[2]) = deviceError;
  }
};
thread_local int FakeRuntime::current = 0;
int FakeRuntime::queryError, FakeRuntime::propertiesError, FakeRuntime::libraryError;
int FakeRuntime::deviceError, FakeRuntime::unloadError;
int FakeRuntime::libraryCalls, FakeRuntime::deviceCalls, FakeRuntime::unloadCalls;
std::vector<int> FakeRuntime::queried, FakeRuntime::loaded;

struct ExactModule { void* module; };
struct MaskModule { void* module; };
struct FfnModule { void* module; };
template<class Module> using Loader = b11aot::CurrentDeviceLoader<Module, FakeRuntime>;

template<class Module> void testModule() {
  using Stage = b11aot::InitStage;
  FakeRuntime::reset();
  Loader<Module> loader(FakeRuntime::initialize, FakeRuntime::load);
  FakeRuntime::current = 1;
  const auto unsupported = loader.prepareCurrentDevice();
  assert(unsupported.error == 801 && unsupported.stage == Stage::Eligibility);
  assert(FakeRuntime::libraryCalls == 0 && FakeRuntime::deviceCalls == 0);
  FakeRuntime::current = 0;
  std::vector<std::thread> threads;
  for(int i = 0; i < 8; i++) threads.emplace_back([&] { assert(loader.prepareCurrentDevice().ok()); });
  for(auto& thread : threads) thread.join();
  assert(FakeRuntime::libraryCalls == 1 && FakeRuntime::deviceCalls == 1);
  FakeRuntime::current = 2;
  assert(loader.prepareCurrentDevice().ok());
  assert(FakeRuntime::libraryCalls == 1 && FakeRuntime::deviceCalls == 2);
  assert((FakeRuntime::queried == std::vector<int>{1,0,2}));
  assert((FakeRuntime::loaded == std::vector<int>{0,2}));
  FakeRuntime::unloadError = 5;
  assert(loader.shutdown().error == 5 && loader.module()->module != nullptr);
  FakeRuntime::unloadError = 0;
  assert(loader.shutdown().ok() && loader.module()->module == nullptr);
  assert(loader.prepareCurrentDevice().ok());
  assert(FakeRuntime::libraryCalls == 2 && FakeRuntime::deviceCalls == 3);
  assert(loader.shutdown().ok());
}

void testFailures() {
  using Stage = b11aot::InitStage;
  FakeRuntime::reset();
  Loader<ExactModule> loader(FakeRuntime::initialize, FakeRuntime::load);
  FakeRuntime::queryError = 10;
  assert(loader.prepareCurrentDevice().stage == Stage::CurrentDevice);
  assert(FakeRuntime::queried.empty());
  FakeRuntime::queryError = 0; FakeRuntime::propertiesError = 11;
  const auto propertyFailure = loader.prepareCurrentDevice();
  assert(propertyFailure.error == 11 && propertyFailure.stage == Stage::DeviceProperties);
  assert(FakeRuntime::libraryCalls == 0 && FakeRuntime::deviceCalls == 0);
  assert(loader.shutdown().ok()); FakeRuntime::propertiesError = 0; FakeRuntime::libraryError = 12;
  const auto libraryFailure = loader.prepareCurrentDevice();
  assert(libraryFailure.error == 12 && libraryFailure.stage == Stage::Library);
  assert(FakeRuntime::deviceCalls == 0);
  FakeRuntime::current = 2;
  assert(loader.prepareCurrentDevice().error == 12 && FakeRuntime::libraryCalls == 1);
  assert(loader.shutdown().ok()); FakeRuntime::libraryError = 0; FakeRuntime::deviceError = 13;
  const auto deviceFailure = loader.prepareCurrentDevice();
  assert(deviceFailure.error == 13 && deviceFailure.stage == Stage::Device);
  assert(loader.prepareCurrentDevice().error == 13 && FakeRuntime::deviceCalls == 1);
  assert(loader.shutdown().ok());
}
} // namespace

int main() {
  testModule<ExactModule>(); testModule<MaskModule>(); testModule<FfnModule>(); testFailures();
}

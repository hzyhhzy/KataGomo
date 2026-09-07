#pragma once
#include "aot_loader_core.h"
#include <cuda_runtime_api.h>
#include <cstring>

namespace b11aot {
struct Rtx5090Runtime {
  using Error = cudaError_t;
  using DeviceProperties = cudaDeviceProp;
  static Error success() { return cudaSuccess; }
  static Error unsupported() { return cudaErrorNotSupported; }
  static Error getDevice(int* device) { return cudaGetDevice(device); }
  static Error getDeviceProperties(DeviceProperties* properties, int device) {
    return cudaGetDeviceProperties(properties, device);
  }
  static bool isEligible(const DeviceProperties& properties) {
    return properties.major == 12 && properties.minor == 0 &&
      std::strcmp(properties.name, "NVIDIA GeForce RTX 5090") == 0;
  }
  static Error unload(cudaLibrary_t library) { return cudaLibraryUnload(library); }
};
} // namespace b11aot

// Static replacement for the `_cuda*` bridge symbols that
// libcute_dsl_runtime.so provides for the CUTLASS DSL AOT host objects.
// These are thin wrappers over the CUDA Runtime / Driver APIs so the final
// KataGo binary has no dependency on the Python venv.
#include <cuda.h>
#include <cuda_runtime_api.h>

extern "C" {

cudaError_t _cudaGetDevice(int* device) {
  return cudaGetDevice(device);
}

cudaError_t _cudaDeviceGetAttribute(int* value, int attr, int device) {
  return cudaDeviceGetAttribute(value, (cudaDeviceAttr)attr, device);
}

cudaError_t _cudaLibraryLoadData(
  cudaLibrary_t* library,
  const void* code,
  cudaJitOption* jitOptions,
  void** jitOptionsValues,
  unsigned int numJitOptions,
  cudaLibraryOption* libraryOptions,
  void** libraryOptionValues,
  unsigned int numLibraryOptions
) {
  return cudaLibraryLoadData(
    library, code, jitOptions, jitOptionsValues, numJitOptions,
    libraryOptions, libraryOptionValues, numLibraryOptions
  );
}

cudaError_t _cudaLibraryGetKernel(cudaKernel_t* pKernel, cudaLibrary_t library, const char* name) {
  return cudaLibraryGetKernel(pKernel, library, name);
}

CUresult _cuKernelGetAttribute(int* pi, int attrib, CUkernel kernel, int device) {
  return cuKernelGetAttribute(pi, (CUfunction_attribute)attrib, kernel, device);
}

cudaError_t _cudaFuncSetAttribute(const void* func, cudaFuncAttribute attr, int value) {
  return cudaFuncSetAttribute(func, attr, value);
}

cudaError_t _cudaKernelSetAttributeForDevice(
  cudaKernel_t kernel, cudaFuncAttribute attr, int value, int device
) {
  return cudaKernelSetAttributeForDevice(kernel, attr, value, device);
}

cudaError_t _cudaLaunchKernelEx(cudaLaunchConfig_t* config, const void* func, void** args) {
  return cudaLaunchKernelExC(config, func, args);
}

} // extern "C"

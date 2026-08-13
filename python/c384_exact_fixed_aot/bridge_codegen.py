#!/usr/bin/env python3
"""Render the CUDA bridge for one CuTe-exported exact-shape object.

The generated launch function is deliberately boring: it only validates that
the current device was prepared and calls the already-loaded wrapper. Module
loading is isolated in the explicit per-device prepare symbol, so first-seen
inference never allocates, initializes a module, or takes a mutex.
"""

from __future__ import annotations

from contract import Task, require, IDENTIFIER


def _validate(task: Task) -> None:
    for value in (task.artifact_stem, task.prepare_symbol, task.launch_symbol):
        require(IDENTIFIER.fullmatch(value) is not None,
                f"unsafe generated C identifier: {value!r}")


def _preamble(task: Task) -> str:
    _validate(task)
    stem = task.artifact_stem
    return f'''#include "{stem}.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <mutex>

namespace {{

constexpr int MaxPreparedDevices = 32;
std::array<{stem}_Kernel_Module_t,MaxPreparedDevices> modules{{}};
std::array<std::atomic<bool>,MaxPreparedDevices> prepared{{}};
std::array<std::mutex,MaxPreparedDevices> prepareMutexes;

cudaError_t validateDeviceOrdinal(int deviceOrdinal) {{
  int deviceCount = 0;
  cudaError_t status = cudaGetDeviceCount(&deviceCount);
  if(status != cudaSuccess)
    return status;
  return deviceOrdinal >= 0 && deviceOrdinal < deviceCount &&
      deviceOrdinal < MaxPreparedDevices ? cudaSuccess : cudaErrorInvalidDevice;
}}

}}  // namespace

extern "C" cudaError_t {task.prepare_symbol}(int deviceOrdinal) {{
  cudaError_t status = validateDeviceOrdinal(deviceOrdinal);
  if(status != cudaSuccess)
    return status;
  if(prepared[deviceOrdinal].load(std::memory_order_acquire))
    return cudaSuccess;
  std::lock_guard<std::mutex> lock(prepareMutexes[deviceOrdinal]);
  if(prepared[deviceOrdinal].load(std::memory_order_relaxed))
    return cudaSuccess;
  int previousDevice = -1;
  status = cudaGetDevice(&previousDevice);
  if(status != cudaSuccess)
    return status;
  if(previousDevice != deviceOrdinal) {{
    status = cudaSetDevice(deviceOrdinal);
    if(status != cudaSuccess)
      return status;
  }}
  (void)cudaGetLastError();
  {stem}_Kernel_Module_Load(&modules[deviceOrdinal]);
  status = cudaPeekAtLastError();
  if(previousDevice != deviceOrdinal) {{
    cudaError_t restore = cudaSetDevice(previousDevice);
    if(status == cudaSuccess)
      status = restore;
  }}
  if(status == cudaSuccess)
    prepared[deviceOrdinal].store(true,std::memory_order_release);
  return status;
}}
'''


def render_qkv_rope_bridge(task: Task) -> str:
    require(task.family == "qkv_rope", "QKV bridge received another family")
    stem = task.artifact_stem
    return _preamble(task) + f'''
extern "C" cudaError_t {task.launch_symbol}(
  const half* input,
  const half* packedWeights,
  const half2* cosSin,
  half* packedQkvOutput,
  int tokenRows,
  int deviceOrdinal,
  cudaStream_t stream
) {{
  if(tokenRows != {task.token_rows})
    return cudaErrorInvalidValue;
  if(deviceOrdinal < 0 || deviceOrdinal >= MaxPreparedDevices ||
     !prepared[deviceOrdinal].load(std::memory_order_acquire))
    return cudaErrorNotReady;
  {stem}_Tensor_a_arg_t a = {{const_cast<half*>(input)}};
  {stem}_Tensor_b_arg_t b = {{const_cast<half*>(packedWeights)}};
  {stem}_Tensor_c_arg_t c = {{packedQkvOutput}};
  {stem}_Tensor_table_arg_t table = {{
    const_cast<half*>(reinterpret_cast<const half*>(cosSin))
  }};
  int32_t wrapperStatus = cute_dsl_{stem}_wrapper(
    &modules[deviceOrdinal],&a,&b,&c,&table,stream);
  return wrapperStatus == 0 ? cudaPeekAtLastError() : cudaErrorUnknown;
}}
'''


def render_dual_ffn_bridge(task: Task) -> str:
    require(task.family == "dual_ffn", "dual bridge received another family")
    stem = task.artifact_stem
    return _preamble(task) + f'''
extern "C" cudaError_t {task.launch_symbol}(
  const half* input,
  const half* pairedWeights,
  const half* unusedGateWeights,
  half* output,
  int tokenRows,
  int deviceOrdinal,
  cudaStream_t stream
) {{
  (void)unusedGateWeights;
  if(tokenRows != {task.token_rows})
    return cudaErrorInvalidValue;
  if(deviceOrdinal < 0 || deviceOrdinal >= MaxPreparedDevices ||
     !prepared[deviceOrdinal].load(std::memory_order_acquire))
    return cudaErrorNotReady;
  {stem}_Tensor_a_arg_t a = {{const_cast<half*>(input)}};
  {stem}_Tensor_b_arg_t b = {{const_cast<half*>(pairedWeights)}};
  {stem}_Tensor_c_arg_t c = {{output}};
  int32_t wrapperStatus = cute_dsl_{stem}_wrapper(
    &modules[deviceOrdinal],&a,&b,&c,stream);
  return wrapperStatus == 0 ? cudaPeekAtLastError() : cudaErrorUnknown;
}}
'''

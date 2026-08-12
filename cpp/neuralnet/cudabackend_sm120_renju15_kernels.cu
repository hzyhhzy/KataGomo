#include "cudabackend_sm120_renju15_kernels.h"

#include <cstdint>

namespace Renju15Sm120 {
namespace {

constexpr int Channels = 256;
union Half8Pack {
  uint4 packed;
  half2 values[4];
};

__global__ void rmsNorm256Warp4Vec8Kernel(
  const uint4* __restrict__ input,
  uint4* __restrict__ output,
  const uint4* __restrict__ gamma,
  int totalRows,
  float epsilon
) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * 4 + warp;
  if(row >= totalRows)
    return;

  constexpr int vectorsPerRow = Channels / 8;
  Half8Pack in;
  Half8Pack g;
  Half8Pack out;
  in.packed = input[(std::size_t)row * vectorsPerRow + lane];
  g.packed = gamma[lane];

  float values[8];
  float sumSquares = 0.0f;
#pragma unroll
  for(int element = 0; element < 4; element++) {
    const float2 value = __half22float2(in.values[element]);
    values[2 * element] = value.x;
    values[2 * element + 1] = value.y;
    sumSquares += value.x * value.x + value.y * value.y;
  }
  for(int offset = 16; offset > 0; offset >>= 1)
    sumSquares += __shfl_xor_sync(0xffffffff, sumSquares, offset);
  const float scale = rsqrtf(sumSquares / (float)Channels + epsilon);

#pragma unroll
  for(int element = 0; element < 4; element++) {
    const float2 gammaValues = __half22float2(g.values[element]);
    out.values[element] = __floats2half2_rn(
      values[2 * element] * scale * gammaValues.x,
      values[2 * element + 1] * scale * gammaValues.y);
  }
  output[(std::size_t)row * vectorsPerRow + lane] = out.packed;
}

bool aligned16(const void* pointer) {
  return (reinterpret_cast<std::uintptr_t>(pointer) & 15U) == 0;
}

} // namespace

cudaError_t launchRmsNorm256(
  const half* input,
  half* output,
  const half* gamma,
  int totalRows,
  float epsilon,
  RmsNorm256Tactic tactic,
  cudaStream_t stream
) {
  if(input == nullptr || output == nullptr || gamma == nullptr ||
     totalRows <= 0 || !aligned16(input) || !aligned16(output) ||
     !aligned16(gamma))
    return cudaErrorInvalidValue;
  if(tactic != RmsNorm256Tactic::Warp4Vec8)
    return cudaErrorNotSupported;
  const int blocks = (totalRows + 3) / 4;
  rmsNorm256Warp4Vec8Kernel<<<blocks, 128, 0, stream>>>(
    reinterpret_cast<const uint4*>(input),
    reinterpret_cast<uint4*>(output),
    reinterpret_cast<const uint4*>(gamma),
    totalRows, epsilon);
  return cudaPeekAtLastError();
}

} // namespace Renju15Sm120


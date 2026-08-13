#include "rms_norm.h"

#include <cstdint>

namespace Renju15Sm120 {
namespace {

constexpr int Channels = 256;
constexpr int Channels384 = 384;
union Half8Pack {
  uint4 packed;
  half2 values[4];
};

#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
union Int8Pack {
  uint2 packed;
  int8_t values[8];
};
#endif

union Half4Pack {
  uint2 packed;
  half2 values[2];
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

#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
__global__ void rmsNorm256Warp4Vec8Fp16Int8Kernel(
  const uint4* __restrict__ input,
  uint4* __restrict__ outputFp16,
  uint2* __restrict__ outputInt8,
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
  Int8Pack quantized;
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
#pragma unroll
  for(int element = 0; element < 4; element++) {
    // Deliberately quantize the rounded FP16 value, not the unrounded float.
    const float2 rounded = __half22float2(out.values[element]);
    float x0 = fminf(4.0f,fmaxf(-4.0f,rounded.x));
    float x1 = fminf(4.0f,fmaxf(-4.0f,rounded.y));
    int q0 = __float2int_rn(x0 * (127.0f / 4.0f));
    int q1 = __float2int_rn(x1 * (127.0f / 4.0f));
    q0 = q0 < -127 ? -127 : (q0 > 127 ? 127 : q0);
    q1 = q1 < -127 ? -127 : (q1 > 127 ? 127 : q1);
    quantized.values[2 * element] = static_cast<int8_t>(q0);
    quantized.values[2 * element + 1] = static_cast<int8_t>(q1);
  }
  const std::size_t vector = (std::size_t)row * vectorsPerRow + lane;
  outputFp16[vector] = out.packed;
  outputInt8[vector] = quantized.packed;
}
#endif

__global__ void rmsNorm384Warp4Vec4x3Kernel(
  const uint2* __restrict__ input,
  uint2* __restrict__ output,
  const uint2* __restrict__ gamma,
  int totalRows,
  float epsilon
) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * 4 + warp;
  if(row >= totalRows)
    return;

  constexpr int vectorsPerRow = Channels384 / 4;
  constexpr int rounds = vectorsPerRow / 32;
  Half4Pack in[rounds];
  Half4Pack g[rounds];
  Half4Pack out[rounds];
  float values[rounds][4];
  float sumSquares = 0.0f;
#pragma unroll
  for(int round = 0; round < rounds; round++) {
    const int vector = lane + round * 32;
    in[round].packed = input[(std::size_t)row * vectorsPerRow + vector];
    g[round].packed = gamma[vector];
#pragma unroll
    for(int pair = 0; pair < 2; pair++) {
      const float2 value = __half22float2(in[round].values[pair]);
      values[round][2 * pair] = value.x;
      values[round][2 * pair + 1] = value.y;
      sumSquares += value.x * value.x + value.y * value.y;
    }
  }
  for(int offset = 16; offset > 0; offset >>= 1)
    sumSquares += __shfl_xor_sync(0xffffffff, sumSquares, offset);
  const float scale = rsqrtf(sumSquares / (float)Channels384 + epsilon);

#pragma unroll
  for(int round = 0; round < rounds; round++) {
#pragma unroll
    for(int pair = 0; pair < 2; pair++) {
      const float2 gammaValues = __half22float2(g[round].values[pair]);
      out[round].values[pair] = __floats2half2_rn(
        values[round][2 * pair] * scale * gammaValues.x,
        values[round][2 * pair + 1] * scale * gammaValues.y);
    }
    const int vector = lane + round * 32;
    output[(std::size_t)row * vectorsPerRow + vector] = out[round].packed;
  }
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

#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
cudaError_t launchRmsNorm256Fp16Int8(
  const half* input,
  half* outputFp16,
  int8_t* outputInt8,
  const half* gamma,
  int totalRows,
  float epsilon,
  RmsNorm256Tactic tactic,
  cudaStream_t stream
) {
  if(input == nullptr || outputFp16 == nullptr || outputInt8 == nullptr ||
     gamma == nullptr || totalRows <= 0 || !aligned16(input) ||
     !aligned16(outputFp16) || !aligned16(outputInt8) || !aligned16(gamma))
    return cudaErrorInvalidValue;
  if(tactic != RmsNorm256Tactic::Warp4Vec8)
    return cudaErrorNotSupported;
  const int blocks = (totalRows + 3) / 4;
  rmsNorm256Warp4Vec8Fp16Int8Kernel<<<blocks,128,0,stream>>>(
    reinterpret_cast<const uint4*>(input),
    reinterpret_cast<uint4*>(outputFp16),
    reinterpret_cast<uint2*>(outputInt8),
    reinterpret_cast<const uint4*>(gamma),totalRows,epsilon);
  return cudaPeekAtLastError();
}
#endif

cudaError_t launchRmsNorm384(
  const half* input,
  half* output,
  const half* gamma,
  int totalRows,
  float epsilon,
  RmsNorm384Tactic tactic,
  cudaStream_t stream
) {
  if(input == nullptr || output == nullptr || gamma == nullptr ||
     totalRows <= 0 || !aligned16(input) || !aligned16(output) ||
     !aligned16(gamma))
    return cudaErrorInvalidValue;
  if(tactic != RmsNorm384Tactic::Warp4Vec4x3)
    return cudaErrorNotSupported;
  const int blocks = (totalRows + 3) / 4;
  rmsNorm384Warp4Vec4x3Kernel<<<blocks, 128, 0, stream>>>(
    reinterpret_cast<const uint2*>(input),
    reinterpret_cast<uint2*>(output),
    reinterpret_cast<const uint2*>(gamma),
    totalRows, epsilon);
  return cudaPeekAtLastError();
}

} // namespace Renju15Sm120


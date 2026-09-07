#pragma once
// Include in exactly one CUDA translation unit; host callers include the .h.
#include "cudab11affine.h"
#include <climits>
#include <cmath>
#include <cstdint>

namespace b11affine {
template<int Width> struct Packed;
template<> struct Packed<4> {
  using Storage = uint2;
  union __align__(8) Value { uint2 storage; half2 pairs[2]; };
};
template<> struct Packed<8> {
  using Storage = uint4;
  union __align__(16) Value { uint4 storage; half2 pairs[4]; };
};

__device__ __forceinline__ half silu(half roundedAffine) {
  // Identical expression to official siluh. Do not use h2exp or FP16 division:
  // only affine/mask and final storage are FP16; the SiLU calculation is FP32.
  const float value = __half2float(roundedAffine);
  return __float2half(value / (1.0f + expf(-value)));
}

template<int Channels, int Width, bool Masked>
__global__ __launch_bounds__(256) void flatAffineSilu(
  const typename Packed<Width>::Storage* input,
  typename Packed<Width>::Storage* output,
  const typename Packed<Width>::Storage* __restrict__ scale,
  const typename Packed<Width>::Storage* __restrict__ bias,
  const half* __restrict__ mask,
  int vectorCount
) {
  constexpr int vectorsPerRow = Channels / Width;
  const int vectorIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if(vectorIndex >= vectorCount) return;
  const int row = vectorIndex / vectorsPerRow;
  const int channelVector = vectorIndex - row * vectorsPerRow;
  typename Packed<Width>::Value x, s, b, y;
  x.storage = input[vectorIndex];
  s.storage = scale[channelVector];
  b.storage = bias[channelVector];
  half2 maskPair;
  if(Masked) maskPair = __half2half2(mask[row]);
  #pragma unroll
  for(int pair = 0; pair < Width / 2; pair++) {
    // Packed operations retain per-lane FP16 round-to-nearest semantics of
    // official __hfma followed by __hmul. Do not combine into FP32 FMA.
    half2 affine = __hfma2(x.pairs[pair], s.pairs[pair], b.pairs[pair]);
    if(Masked) affine = __hmul2(affine, maskPair);
    y.pairs[pair] = __halves2half2(silu(__low2half(affine)), silu(__high2half(affine)));
  }
  // Each thread owns the entire vector. No restrict on input/output so the
  // official in-place BatchNorm use is valid as well as disjoint buffers.
  output[vectorIndex] = y.storage;
}

template<int Channels, int Width>
static void launch(const half* input, half* output, const half* scale,
  const half* bias, const half* mask, int vectorCount, int threads, cudaStream_t stream) {
  using Storage = typename Packed<Width>::Storage;
  const int blocks = (vectorCount - 1) / threads + 1;
  if(mask)
    flatAffineSilu<Channels,Width,true><<<blocks,threads,0,stream>>>(
      reinterpret_cast<const Storage*>(input), reinterpret_cast<Storage*>(output),
      reinterpret_cast<const Storage*>(scale), reinterpret_cast<const Storage*>(bias), mask, vectorCount);
  else
    flatAffineSilu<Channels,Width,false><<<blocks,threads,0,stream>>>(
      reinterpret_cast<const Storage*>(input), reinterpret_cast<Storage*>(output),
      reinterpret_cast<const Storage*>(scale), reinterpret_cast<const Storage*>(bias), nullptr, vectorCount);
}
} // namespace b11affine

bool launchB11AffineSiluCandidate(
  const half* input, half* output, const half* scale, const half* bias,
  const half* mask, int totalRows, int channels, int vectorWidth,
  int blockThreads, cudaStream_t stream
) {
  if(totalRows <= 0 || (channels != 384 && channels != 768) ||
     (vectorWidth != 4 && vectorWidth != 8) ||
     (blockThreads != 128 && blockThreads != 256) ||
     !input || !output || !scale || !bias) return false;
  const std::uintptr_t alignmentMask = static_cast<std::uintptr_t>(vectorWidth * sizeof(half) - 1);
  if((reinterpret_cast<std::uintptr_t>(input) | reinterpret_cast<std::uintptr_t>(output) |
      reinterpret_cast<std::uintptr_t>(scale) | reinterpret_cast<std::uintptr_t>(bias)) & alignmentMask)
    return false;
  if(mask && (reinterpret_cast<std::uintptr_t>(mask) & (alignof(half) - 1))) return false;
  const std::int64_t vectors = std::int64_t(totalRows) * (channels / vectorWidth);
  if(vectors > INT_MAX - 255) return false;
  const int count = static_cast<int>(vectors);
  if(channels == 384) {
    if(vectorWidth == 4) b11affine::launch<384,4>(input,output,scale,bias,mask,count,blockThreads,stream);
    else b11affine::launch<384,8>(input,output,scale,bias,mask,count,blockThreads,stream);
  }
  else {
    if(vectorWidth == 4) b11affine::launch<768,4>(input,output,scale,bias,mask,count,blockThreads,stream);
    else b11affine::launch<768,8>(input,output,scale,bias,mask,count,blockThreads,stream);
  }
  return true;
}

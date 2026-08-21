// CUDA wrapper for the shared CUDA/ROCm GPU kernels.
// All kernel code lives in cudaandrocmhelpers.inc, which is shared with the ROCm backend
// (rocmhelpers.hip). See the comment at the top of that file for the macro contract.

#include "../neuralnet/cudahelpers.h"

#include <cmath>
#include <stdexcept>

// Evaluated per device architecture during nvcc's per-arch device compilation passes, so the
// half-precision kernel bodies are compiled exactly for the archs that support them.
#if __CUDA_ARCH__ >= 530
#define KATAGO_GPU_SUPPORTS_FP16
#endif

#define KATAGO_GPU_CUDA 1
#define KATAGO_GPU_SINCOSF __sincosf

// Tensor-core flash attention is CUDA-only (mma.sync/ldmatrix/cp.async PTX). The shared .inc
// provides a fallback stub for builds without it.
#include "../neuralnet/cudaflashmma.cuh"
#define KATAGO_HAS_FLASH_MMA 1

#include "../neuralnet/cudaandrocmhelpers.inc"

// Keep this CUDA-only helper separate from the shared legacy SwiGLU kernels.
// The comparison form deliberately propagates NaN, matching torch.clamp and
// the established clip epilogue; fminf/fmaxf would replace NaN with a bound.
__device__ __forceinline__ float orderedClampSymmetric(float value, float clip) {
  return value > clip ? clip : (value < -clip ? -clip : value);
}

template<int ELTS_PER_THREAD>
__global__ void swiGLUOrderedClippedFP16Kernel(
  const half* linear, const half* gate, half* out, int size, float clip
) {
#ifdef KATAGO_GPU_SUPPORTS_FP16
  const half2* linear2 = reinterpret_cast<const half2*>(linear);
  const half2* gate2 = reinterpret_cast<const half2*>(gate);
  half2* out2 = reinterpret_cast<half2*>(out);
  const int pairCount = size >> 1;
  const int tileStart = blockIdx.x * blockDim.x * ELTS_PER_THREAD;
  const int lane = threadIdx.x;

  #pragma unroll
  for(int d = 0; d < ELTS_PER_THREAD; d++) {
    const int pair = tileStart + d * blockDim.x + lane;
    if(pair < pairCount) {
      const half2 linearValue = linear2[pair];
      const half2 gateValue = gate2[pair];
      const float linear0 = orderedClampSymmetric(
        siluf(__half2float(__low2half(linearValue))),clip);
      const float linear1 = orderedClampSymmetric(
        siluf(__half2float(__high2half(linearValue))),clip);
      const float gate0 = orderedClampSymmetric(
        __half2float(__low2half(gateValue)),clip);
      const float gate1 = orderedClampSymmetric(
        __half2float(__high2half(gateValue)),clip);
      out2[pair] = __halves2half2(
        __float2half_rn(linear0 * gate0),
        __float2half_rn(linear1 * gate1)
      );
    }
  }

  if((size & 1) != 0 && blockIdx.x == 0 && lane == 0) {
    const int last = size - 1;
    const float linearValue = orderedClampSymmetric(
      siluf(__half2float(linear[last])),clip);
    const float gateValue = orderedClampSymmetric(__half2float(gate[last]),clip);
    out[last] = __float2half_rn(linearValue * gateValue);
  }
#else
  (void)linear;
  (void)gate;
  (void)out;
  (void)size;
  (void)clip;
#endif
}

void customCudaSwiGLUOrderedClippedFP16(
  const half* linear, const half* gate, half* out, int size, float clip,
  cudaStream_t stream
) {
  if(size <= 0)
    return;
  if(!(clip > 0.0f) || !std::isfinite(clip))
    throw std::runtime_error(
      "customCudaSwiGLUOrderedClippedFP16: clip must be finite and positive");
  constexpr int ELTS_PER_THREAD = 4;
  constexpr int threads = 256;
  const int pairCount = size >> 1;
  int blocks =
    (pairCount + threads * ELTS_PER_THREAD - 1) /
    (threads * ELTS_PER_THREAD);
  if(blocks < 1)
    blocks = 1;
  swiGLUOrderedClippedFP16Kernel<ELTS_PER_THREAD>
    <<<blocks,threads,0,stream>>>(linear,gate,out,size,clip);
}

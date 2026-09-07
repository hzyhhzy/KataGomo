#ifndef NEURALNET_CUDAB11GEMM_H_
#define NEURALNET_CUDAB11GEMM_H_

#include "cudaincludes.h"

// Dense, contiguous, row-major FP16 matrices:
//   output[M,N] = input[M,K] * weights[K,N] + (betaIsOne ? output[M,N] : 0).
// Weights MUST already be transposed from the model's output-major layout.
// All pointers require 16-byte alignment. Input/weights may not alias output;
// betaIsOne deliberately uses output itself as the residual source.
// Accepted (N,K): (384,384), (384,1152), (384,768), (768,384), (1152,384).
// Model/device/board/batch dispatch belongs to the caller. No allocation or
// synchronization is performed; failures are returned as CUDA error codes.
// Accumulation and epilogue arithmetic use FP16, matching the existing fast
// FP16 CUDA path's precision class, but not promising bitwise cuBLAS identity.
cudaError_t launchB11Gemm(
  const half* input, const half* weights, half* output,
  int M, int N, int K, bool betaIsOne, cudaStream_t stream);

#endif

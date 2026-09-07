#ifndef NEURALNET_CUDAB11QKVROPE_H_
#define NEURALNET_CUDAB11QKVROPE_H_

#include "cudaincludes.h"

namespace CudaB11QkvRope {

// Call on the consuming current device at handle setup, before warmup/capture.
// Configures the kernel's dynamic shared-memory attribute if required. Does not
// allocate memory, launch a kernel, or select/check the model or GPU. This source
// can be compiled with the normal CUDA architecture list; the caller must gate
// execution to the supported B11/RTX5090 profile, including board/batch/layout.
cudaError_t prepareForCurrentDevice();

// FP16 A[M,384] * W[384,1152], row-major, beta=0. W is the existing input-major
// combined QKV weight layout, not output-major model weights. Output remains
// row-major [M,1152]: Q384, K384, V384 per token. Q/K are rotated with learnable
// FP32 RoPE after the GEMM's FP16 epilogue conversion; V is unchanged.
//
// M=B*361 for B1..96. A/W/output must be nonnull, 16-byte aligned, and output
// must not overlap inputs, weights, or freqs. freqs is immutable device FP32
// [192,2], with each pair's x/y frequency, on the same device as the matrices.
// All buffers are caller-owned and must remain valid until stream completion.
// On success the caller must NOT apply a second standalone RoPE to this output.
//
// This function does no allocation, attribute setup, device queries, locking,
// synchronization, or fallback. Call prepareForCurrentDevice first. It returns
// CUDA status (including a possibly pending prior asynchronous error); the caller
// must propagate a launch error, not silently recompute into partially used data.
cudaError_t run(const half* A,const half* W,half* output,int M,
  const float* freqs,cudaStream_t stream);

} // namespace CudaB11QkvRope

#endif


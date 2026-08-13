#ifndef KATAGO_RENJU15_RESIDUAL_GEMM_SM120_H
#define KATAGO_RENJU15_RESIDUAL_GEMM_SM120_H

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstddef>

enum KatagoRenju15ResidualGemmFamily {
  KATAGO_RENJU15_RESIDUAL_GEMM_OUT_PROJ = 1,
  KATAGO_RENJU15_RESIDUAL_GEMM_FFN_DOWN = 2,
  KATAGO_RENJU15_RESIDUAL_GEMM_C384_OUT_PROJ = 3,
  KATAGO_RENJU15_RESIDUAL_GEMM_C384_FFN_DOWN = 4,
};

enum KatagoRenju15ResidualGemmTactic {
  KATAGO_RENJU15_RESIDUAL_GEMM_DISABLED = 0,
  KATAGO_RENJU15_RESIDUAL_GEMM_M128_N128_K32_S3 = 1,
  KATAGO_RENJU15_RESIDUAL_GEMM_C384_M128_N128_K32_S3 = 2,
};

struct KatagoRenju15ResidualGemmDescriptor {
  int tactic;
  const char* tacticId;
  int threadblockM;
  int threadblockN;
  int threadblockK;
  int warpM;
  int warpN;
  int warpK;
  int stages;
  int swizzle;
  int outputChannels;
};

// configuredTokenRows is exact GEMM M for legacy C256 families and maximum M
// for dynamic C384 families. It is independent of weights and model depth.
extern "C" void* katago_renju15_residual_gemm_sm120_create(
  int family,
  int tactic,
  int configuredTokenRows);
extern "C" void katago_renju15_residual_gemm_sm120_destroy(void* opaque);
extern "C" const char* katago_renju15_residual_gemm_sm120_active_marker(
  const void* opaque);

extern "C" bool katago_renju15_residual_gemm_sm120_supports(
  const void* opaque,
  int matBatchSize,
  int inputChannels,
  int outputChannels,
  bool usingFp16,
  bool exactNoMask);

// C is the residual and D in the CUTLASS alpha*A*B+beta*C epilogue, with
// alpha=beta=1. Once supports() returned true, any non-success launch status
// is fatal at the caller; there is no after-enqueue generic retry.
extern "C" cudaError_t katago_renju15_residual_gemm_sm120_launch(
  void* opaque,
  const half* input,
  const half* weights,
  half* residual,
  int matBatchSize,
  cudaStream_t stream);

#endif

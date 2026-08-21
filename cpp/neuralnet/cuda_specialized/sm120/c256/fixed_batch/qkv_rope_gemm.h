#ifndef KATAGO_RENJU15_QKV_ROPE_GEMM_SM120_H
#define KATAGO_RENJU15_QKV_ROPE_GEMM_SM120_H

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstddef>

enum KatagoRenju15QKVRoPEGemmSm120Tactic {
  KATAGO_RENJU15_QKV_ROPE_GEMM_DISABLED = 0,
  KATAGO_RENJU15_QKV_ROPE_GEMM_M128_N128_K32_S3 = 1,
  KATAGO_C256_S361_QKV_ROPE_GEMM_M128_N128_K32_S3 = 2,
};

struct KatagoRenju15QKVRoPEGemmSm120Descriptor {
  int tactic;
  const char* tacticId;
  int tileM;
  int tileN;
  int tileK;
  int warpM;
  int warpN;
  int warpK;
  int stages;
  int swizzle;
  int threads;
  std::size_t dynamicSharedBytes;
  int inputChannels;
  int projectionChannels;
  int fixedBatchSize;
  int sequenceLength;
};

// Creation is exact-batch and SM120-gated. The tactic selects either the
// S225/B36 or manifest-selected S361 fixed-batch shape. The returned handle owns only CUTLASS launch
// state; packed weights and RoPE tables remain owned by the provider.
extern "C" void* katago_renju15_qkv_rope_gemm_sm120_create(
  int tactic,
  int fixedBatchSize
);
extern "C" void katago_renju15_qkv_rope_gemm_sm120_destroy(void* opaque);
extern "C" const char* katago_renju15_qkv_rope_gemm_sm120_active_marker(
  const void* opaque
);

// This is the final launch-time fail-closed gate. precomputedHalf2 means the
// selected exact plan uses a precomputed [S][8*16] half2(cos,sin) table.
extern "C" bool katago_renju15_qkv_rope_gemm_sm120_supports(
  const void* opaque,
  int batchSize,
  int seqLen,
  int inputChannels,
  int projectionChannels,
  int numHeads,
  int numKVHeads,
  int headDim,
  int ropePairs,
  bool usingFp16,
  bool usingNhwc,
  bool requireExactNNLen,
  bool recipeEligibleNoMask,
  bool precomputedHalf2
);

// input is row-major [B*S,256]. packedWeights is row-major
// [3][256,256]. output is planar [Q][K][V], each [B*S,256]. The
// epilogue rotates Q/K and leaves V unchanged.
extern "C" cudaError_t katago_renju15_qkv_rope_gemm_sm120_launch(
  void* opaque,
  const half* input,
  const half* packedWeights,
  const half2* cosSin,
  half* output,
  int batchSize,
  cudaStream_t stream
);

#endif

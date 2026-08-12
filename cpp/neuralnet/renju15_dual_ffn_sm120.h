#ifndef KATAGO_RENJU15_DUAL_FFN_SM120_H_
#define KATAGO_RENJU15_DUAL_FFN_SM120_H_

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstddef>

// Selected C256/F768 SM120 shared-A dual-GEMM+SwiGLU kernel. Selection is made
// by the typed operation recipe, never by a model hash or environment string.

enum KatagoRenju15DualFfnSm120Tactic {
  KATAGO_RENJU15_DUAL_FFN_DISABLED = 0,
  KATAGO_RENJU15_DUAL_FFN_M128_N64_K32_S3_SW4 = 1,
  KATAGO_RENJU15_DUAL_FFN_C384_F1024_M128_N64_K32_S3_SW4 = 2,
};

struct KatagoRenju15DualFfnSm120Descriptor {
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
  int inputChannels;
  int ffnChannels;
};

// fixedTokenRows is M in the GEMM. It deliberately does not encode a board,
// batch, model depth, or model weights.
extern "C" void* katago_renju15_dual_ffn_sm120_create(
  int tactic,
  int fixedTokenRows);
extern "C" void katago_renju15_dual_ffn_sm120_destroy(void* opaque);

extern "C" const char* katago_renju15_dual_ffn_sm120_active_marker(
  const void* opaque);

extern "C" bool katago_renju15_dual_ffn_sm120_request_eligible(
  int tokenRows,
  int inputChannels,
  int ffnChannels,
  bool usingFp16,
  bool usingNhwc,
  bool exactNoMask);

// exactNoMask is a semantic flag from the top-level inference request, not an
// inference from a possibly materialized all-ones mask buffer.
extern "C" bool katago_renju15_dual_ffn_sm120_supports(
  const void* opaque,
  int tokenRows,
  int inputChannels,
  int ffnChannels,
  bool usingFp16,
  bool usingNhwc,
  bool exactNoMask);

// The typed handle supports either A[M,256] x W[256,768] or
// A[M,384] x W[384,1024], followed by
// output = SiLU(A*linearWeights) * (A*gateWeights), all contiguous row-major
// FP16. The handle is stream-local and must not be launched concurrently.
extern "C" cudaError_t katago_renju15_dual_ffn_sm120_launch(
  void* opaque,
  const half* input,
  const half* linearWeights,
  const half* gateWeights,
  half* output,
  cudaStream_t stream);

#endif


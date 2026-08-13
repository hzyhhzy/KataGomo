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
  KATAGO_RENJU15_DUAL_FFN_C384_F1024_CLIP7_M128_N64_K32_S3_SW4 = 3,
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

// configuredTokenRows is exact M for the legacy C256 tactic and maximum M for
// the dynamic C384 tactic. It does not encode model depth or model weights.
extern "C" void* katago_renju15_dual_ffn_sm120_create(
  int tactic,
  int configuredTokenRows);
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

// The typed handle supports either exact-M A[M,256] x W[256,768] or dynamic-M
// A[M,384] x W[384,1024], where 0 < M <= the configured maximum, followed by
// output = SiLU(A*linearWeights) * (A*gateWeights), all contiguous row-major
// FP16. The C384 clip7 tactic instead independently clamps both the SiLU
// result and gate to [-7,7] before multiplying. The handle is stream-local and
// must not be launched concurrently.
extern "C" cudaError_t katago_renju15_dual_ffn_sm120_launch(
  void* opaque,
  const half* input,
  const half* linearWeights,
  const half* gateWeights,
  half* output,
  int tokenRows,
  cudaStream_t stream);

#endif

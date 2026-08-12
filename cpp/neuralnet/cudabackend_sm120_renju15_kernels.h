#ifndef KATAGO_CUDA_BACKEND_SM120_RENJU15_KERNELS_H
#define KATAGO_CUDA_BACKEND_SM120_RENJU15_KERNELS_H

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace Renju15Sm120 {

enum class RmsNorm256Tactic {
  Disabled = 0,
  Warp4Vec8 = 1,
};

enum class RmsNorm384Tactic {
  Disabled = 0,
  Warp4Vec4x3 = 1,
};

#if defined(KATAGO_ENABLE_RENJU15_RMS_SM120) && KATAGO_ENABLE_RENJU15_RMS_SM120
// Transformer RMSNorm for C=256. The winning kernel processes four rows per
// block, one warp per row, with vectorized half8 IO. totalRows is dynamic, so
// model depth, board dimensions, and model weights are not part of its ABI.
cudaError_t launchRmsNorm256(
  const half* input,
  half* output,
  const half* gamma,
  int totalRows,
  float epsilon,
  RmsNorm256Tactic tactic,
  cudaStream_t stream
);

#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
// Experimental quantized companion for the C=256 kernel above. The FP16
// output remains bit-for-bit the same operation consumed by the V projection;
// the second output is symmetric signed INT8 with zero point 0 and a fixed
// clip of [-4,4] (scale 4/127). Quantization is applied after the FP16 rounding
// so the two outputs have an explicit, reproducible relationship.
cudaError_t launchRmsNorm256Fp16Int8(
  const half* input,
  half* outputFp16,
  int8_t* outputInt8,
  const half* gamma,
  int totalRows,
  float epsilon,
  RmsNorm256Tactic tactic,
  cudaStream_t stream
);
#endif

// Transformer RMSNorm for C=384. Each warp owns one row and performs three
// coalesced half4 vector rounds (32 lanes * 4 halves * 3 rounds = 384).
// totalRows is fully dynamic; the production matcher separately qualifies the
// representative batch buckets used by the surrounding GEMM tactics.
cudaError_t launchRmsNorm384(
  const half* input,
  half* output,
  const half* gamma,
  int totalRows,
  float epsilon,
  RmsNorm384Tactic tactic,
  cudaStream_t stream
);

#endif

} // namespace Renju15Sm120

#endif

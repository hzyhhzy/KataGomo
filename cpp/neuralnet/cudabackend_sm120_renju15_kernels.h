#ifndef KATAGO_CUDA_BACKEND_SM120_RENJU15_KERNELS_H
#define KATAGO_CUDA_BACKEND_SM120_RENJU15_KERNELS_H

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstddef>

namespace Renju15Sm120 {

enum class RmsNorm256Tactic {
  Disabled = 0,
  Warp4Vec8 = 1,
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

#endif

} // namespace Renju15Sm120

#endif

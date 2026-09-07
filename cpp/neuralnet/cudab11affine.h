#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

// Candidate only. Caller gates B11/RTX5090/FP16/NHWC/19x19/SiLU/actual batch.
// Returns false without a launch if local layout/size guards fail. On true,
// caller must check cudaPeekAtLastError as for the original CUDA helper.
// Supports input == output, or disjoint input/output; partial overlap is invalid.
bool launchB11AffineSiluCandidate(
  const half* input, half* output, const half* scale, const half* bias,
  const half* mask, int totalRows, int channels, int vectorWidth,
  int blockThreads, cudaStream_t stream);

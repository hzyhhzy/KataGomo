#ifndef NEURALNET_CUDAHELPERS_H_
#define NEURALNET_CUDAHELPERS_H_

#include "../neuralnet/cudaincludes.h"
#include "../neuralnet/activations.h"

#include "../neuralnet/cudaandrocmhelpers.h"

// Canonical v105 positive-clip SwiGLU. Inputs and output are FP16, while SiLU,
// both ordered clamps, and the product are evaluated in FP32 before one final
// round-to-nearest conversion to half.
void customCudaSwiGLUOrderedClippedFP16(
  const half* linear, const half* gate, half* out, int size, float clip,
  cudaStream_t stream
);

// CUDA-only fast path for Q/K RMSNorm on an interleaved combined-QKV
// projection. The normalized values are rounded to FP16 before RoPE, matching
// the ordinary RMSNorm-then-RoPE execution contract. Learned RoPE tables are
// built once per layer in FP32 using the same device sincos implementation as
// the table-free fallback, so inference avoids transcendental work without
// reducing the cosine/sine precision.
enum class CudaFusedQKNormRopeMode {
  None = 0,
  Fixed = 1,
  LearnedTable = 2,
};

bool customCudaFusedQKNormRoPESupportsShape(int qHeadDim);

void customCudaBuildLearnedRopeTableFP32(
  const float* freqs, float* cosSinTable,
  int seqLen, int numKVHeads, int numPairs, int nnXLen,
  cudaStream_t stream
);

void customCudaFusedQKNormRoPEFP16(
  half* qBuf, half* kBuf,
  const half* qGamma, const half* kGamma,
  const half* fixedCosTable, const half* fixedSinTable,
  const float* learnedCosSinTable,
  int tokenRows, int seqLen,
  int numHeads, int numKVHeads, int qHeadDim,
  int qStride, int kvStride, int numPairs,
  float qEpsilon, float kEpsilon,
  CudaFusedQKNormRopeMode ropeMode,
  int multiprocessorCount,
  cudaStream_t stream
);

#endif  // NEURALNET_CUDAHELPERS_H_

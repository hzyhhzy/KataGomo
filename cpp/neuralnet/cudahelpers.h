#ifndef NEURALNET_CUDAHELPERS_H_
#define NEURALNET_CUDAHELPERS_H_

#include "../neuralnet/cudaincludes.h"
#include "../neuralnet/activations.h"

#include "../neuralnet/cudaandrocmhelpers.h"

// SwiGLU with independent symmetric clamps on SiLU(linear) and gate before
// multiplication. The FP16 overload evaluates SiLU, clamps, and product in
// FP32 and rounds only the final result back to half.
void customCudaSwiGLUClipped(
  const float* linear, const float* gate, float* out, int size, float clip,
  cudaStream_t stream
);
void customCudaSwiGLUClipped(
  const half* linear, const half* gate, half* out, int size, float clip,
  cudaStream_t stream
);

#endif  // NEURALNET_CUDAHELPERS_H_

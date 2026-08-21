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

#endif  // NEURALNET_CUDAHELPERS_H_

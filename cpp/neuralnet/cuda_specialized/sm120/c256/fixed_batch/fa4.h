#ifndef KATAGO_RENJU15_FA4_SM120_H
#define KATAGO_RENJU15_FA4_SM120_H

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace Renju15Fa4Sm120 {

enum class Tactic {
  Disabled = 0,
  B36Tm128Tn128S1Both16,
};

struct LaunchResult {
  bool attempted;
  cudaError_t status;
  const char* marker;

  bool launched() const { return attempted && status == cudaSuccess; }
};

const char* tacticName(Tactic tactic);

// The only linked AOT object is the selected B36/S225/H8/D32 winner. A local
// shape or capability mismatch returns attempted=false before enqueue so the
// generic attention path remains safe.
LaunchResult launch(
  Tactic tactic,
  half* q,
  half* k,
  half* v,
  half* output,
  int batch,
  int seq,
  int heads,
  int kvHeads,
  int qHeadDim,
  int vHeadDim,
  bool usingFp16,
  bool usingNhwc,
  const void* mask,
  bool recipeEligibleNoMask,
  int computeMajor,
  int computeMinor,
  cudaStream_t stream
);

} // namespace Renju15Fa4Sm120

#endif

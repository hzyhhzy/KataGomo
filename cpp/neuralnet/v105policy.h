#ifndef NEURALNET_V105POLICY_H_
#define NEURALNET_V105POLICY_H_

#include <cstddef>

struct TrunkDesc;

namespace V105CudaPolicy {

enum class SwiGLUPlan {
  LegacyUnclipped,
  OrderedClippedFP32,
};

struct Decision {
  bool isV105;
  bool hasQKNorm;
  bool hasPositiveSwiGLUClip;

  bool needsQKNClipSemantics() const {
    return isV105 && (hasQKNorm || hasPositiveSwiGLUClip);
  }
};

// Two-plane FFN projection storage. The second plane must remain 4-byte
// aligned for the ordered-clipped FP16 helper's half2 loads, including when
// maxBatch*XY*ffnChannels is odd. All fields are in the max-batch allocation
// domain; actual-batch kernel sizes remain unchanged.
struct ProjectedScratchLayout {
  size_t planeElements;
  size_t planeStrideElements;
  size_t planeStrideBytes;
  size_t totalBytes;
};

// Pure descriptor policy. PTQ ranges deliberately do not participate in the
// FP16 execution plan.
Decision classify(int modelVersion, const TrunkDesc& trunk);

// Q/K normalization requires planar Q/K/V buffers before the per-head norm.
bool shouldUseCombinedQKV(bool useQKNorm, bool otherwiseEligible);

// clip==0 keeps the established helper byte-for-byte; positive clip selects
// the ordered FP32 clamp helper. Invalid clip values fail closed.
SwiGLUPlan selectSwiGLUPlan(float swigluClip);

// Pure checked layout helper, shared by CUDA construction and the CPU wire
// contract. Supports the CUDA int-index boundary exactly and rejects any
// larger logical plane before allocating device memory.
ProjectedScratchLayout makeProjectedScratchLayout(
  size_t maxBatchSize,
  size_t nnXLen,
  size_t nnYLen,
  size_t ffnChannels,
  size_t elementBytes
);

// Canonical native v105 CUDA execution is FP16-only. The trunk argument keeps
// this policy at every existing loader/handle call site without allocations.
void requireCurrentExecution(int modelVersion, const TrunkDesc& trunk, bool useFP16);

}  // namespace V105CudaPolicy

#endif  // NEURALNET_V105POLICY_H_

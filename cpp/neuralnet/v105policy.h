#ifndef NEURALNET_V105POLICY_H_
#define NEURALNET_V105POLICY_H_

struct TrunkDesc;

namespace V105CudaPolicy {

struct Decision {
  bool isV105;
  bool hasQKNorm;
  bool hasPositiveSwiGLUClip;

  bool needsQKNClipSemantics() const {
    return isV105 && (hasQKNorm || hasPositiveSwiGLUClip);
  }
};

// Pure descriptor policy: PTQ ranges do not alter the v102 FP16 fallback, but
// Q/K normalization and positive SwiGLU clipping do and must never be dropped.
Decision classify(int modelVersion, const TrunkDesc& trunk);

// Temporary fail-closed gate. The CUDA semantic execution commit will replace
// this with the real QKN/clip-capable route.
void requireCurrentQKNClipSemantics(int modelVersion, const TrunkDesc& trunk);
void requireCurrentExecution(int modelVersion, const TrunkDesc& trunk, bool useFP16);

}  // namespace V105CudaPolicy

#endif  // NEURALNET_V105POLICY_H_

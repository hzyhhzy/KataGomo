#include "../neuralnet/v105policy.h"

#include <cmath>
#include <limits>
#include <vector>

#include "../core/global.h"
#include "../neuralnet/desc.h"

using namespace std;

namespace V105CudaPolicy {
namespace {

size_t checkedMultiply(size_t a, size_t b, const char* context) {
  if(a != 0 && b > std::numeric_limits<size_t>::max() / a)
    throw StringError(string("v105 CUDA ") + context + " size overflow");
  return a * b;
}

void inspectBlocks(
  const vector<pair<int,unique_ptr_void>>& blocks,
  bool& hasQKNorm,
  bool& hasPositiveSwiGLUClip
) {
  for(const auto& entry: blocks) {
    if(entry.first == TRANSFORMER_ATTENTION_BLOCK_KIND) {
      const TransformerAttentionDesc* attention =
        (const TransformerAttentionDesc*)entry.second.get();
      hasQKNorm = hasQKNorm || attention->useQKNorm;
    }
    else if(entry.first == TRANSFORMER_FFN_BLOCK_KIND) {
      const TransformerFFNDesc* ffn = (const TransformerFFNDesc*)entry.second.get();
      hasPositiveSwiGLUClip = hasPositiveSwiGLUClip || ffn->swigluClip > 0.0f;
    }
    else if(entry.first == NESTED_BOTTLENECK_BLOCK_KIND) {
      const NestedBottleneckResidualBlockDesc* nested =
        (const NestedBottleneckResidualBlockDesc*)entry.second.get();
      inspectBlocks(nested->blocks,hasQKNorm,hasPositiveSwiGLUClip);
    }
  }
}

}  // namespace

Decision classify(int modelVersion, const TrunkDesc& trunk) {
  Decision decision = {modelVersion == 105,false,false};
  if(decision.isV105)
    inspectBlocks(trunk.blocks,decision.hasQKNorm,decision.hasPositiveSwiGLUClip);
  return decision;
}

bool shouldUseCombinedQKV(bool useQKNorm, bool otherwiseEligible) {
  return otherwiseEligible && !useQKNorm;
}

SwiGLUPlan selectSwiGLUPlan(float swigluClip) {
  if(swigluClip == 0.0f)
    return SwiGLUPlan::LegacyUnclipped;
  if(std::isfinite(swigluClip) && swigluClip > 0.0f)
    return SwiGLUPlan::OrderedClippedFP32;
  throw StringError("v105 CUDA SwiGLU clip must be finite and nonnegative");
}

ProjectedScratchLayout makeProjectedScratchLayout(
  size_t maxBatchSize,
  size_t nnXLen,
  size_t nnYLen,
  size_t ffnChannels,
  size_t elementBytes
) {
  if(maxBatchSize == 0 || nnXLen == 0 || nnYLen == 0 || ffnChannels == 0)
    throw StringError("v105 CUDA projected scratch dimensions must be positive");
  if(elementBytes != 2 && elementBytes != 4)
    throw StringError("v105 CUDA projected scratch element width must be 2 or 4 bytes");

  size_t planeElements = checkedMultiply(maxBatchSize,nnXLen,"projected scratch");
  planeElements = checkedMultiply(planeElements,nnYLen,"projected scratch");
  planeElements = checkedMultiply(planeElements,ffnChannels,"projected scratch");
  if(planeElements > (size_t)std::numeric_limits<int>::max())
    throw StringError("v105 CUDA projected scratch exceeds the CUDA 32-bit index limit");

  const size_t alignmentElements = 4 / elementBytes;
  if(planeElements > std::numeric_limits<size_t>::max() - (alignmentElements - 1))
    throw StringError("v105 CUDA projected scratch alignment overflow");
  const size_t planeStrideElements =
    ((planeElements + alignmentElements - 1) / alignmentElements) * alignmentElements;
  const size_t planeStrideBytes =
    checkedMultiply(planeStrideElements,elementBytes,"projected scratch stride");
  const size_t totalBytes = checkedMultiply(planeStrideBytes,2,"projected scratch total");
  if(planeStrideBytes % 4 != 0)
    throw StringError("v105 CUDA projected scratch failed 4-byte plane alignment");

  return {
    planeElements,
    planeStrideElements,
    planeStrideBytes,
    totalBytes,
  };
}

void requireCurrentExecution(int modelVersion, const TrunkDesc& trunk, bool useFP16) {
  (void)trunk;
  if(modelVersion == 105 && !useFP16)
    throw StringError("v105 CUDA execution requires FP16");
}

}  // namespace V105CudaPolicy

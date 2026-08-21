#include "../neuralnet/v105policy.h"

#include <cmath>
#include <vector>

#include "../core/global.h"
#include "../neuralnet/desc.h"

using namespace std;

namespace V105CudaPolicy {
namespace {

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

void requireCurrentExecution(int modelVersion, const TrunkDesc& trunk, bool useFP16) {
  (void)trunk;
  if(modelVersion == 105 && !useFP16)
    throw StringError("v105 CUDA execution requires FP16");
}

}  // namespace V105CudaPolicy

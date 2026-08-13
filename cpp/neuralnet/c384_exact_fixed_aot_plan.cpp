#include "../neuralnet/c384_exact_fixed_aot_plan.h"

#include <cstring>

namespace C384ExactFixedAot {

namespace {

constexpr int kCandidateBatches[] = {28,24,40};

const RegistrySpan& registryFor(Family family, const RegistryView& registry) {
  return family == Family::QkvRope ? registry.qkvRope : registry.dualFfn;
}

PieceSelection selectPiece(
  const RuntimeShape& shape,
  Family family,
  const RegistryView& registry,
  const char* requestedId
) {
  PieceSelection result;
  if(!pieceShapeEligible(shape,family)) {
    result.reason = RejectReason::ShapeMismatch;
    return result;
  }
  const RegistrySpan& familyRegistry = registryFor(family,registry);
  if(!registrySpanWellFormed(family,familyRegistry)) {
    result.reason = RejectReason::InvalidRegistry;
    return result;
  }
  if(requestedId == nullptr || requestedId[0] == '\0') {
    result.reason = RejectReason::NoRequestedTactic;
    return result;
  }
  result.tactic = findExactTactic(
    shape,family,familyRegistry,requestedId);
  result.reason = result.tactic == nullptr ?
    RejectReason::RegistryMiss : RejectReason::None;
  return result;
}

}  // namespace

const int* candidateBatches(std::size_t& count) {
  count = sizeof(kCandidateBatches) / sizeof(kCandidateBatches[0]);
  return kCandidateBatches;
}

int candidateBatchPriority(int batchSize) {
  std::size_t count = 0;
  const int* batches = candidateBatches(count);
  for(std::size_t i = 0; i < count; i++) {
    if(batches[i] == batchSize)
      return static_cast<int>(i);
  }
  return -1;
}

int tokenRowsForBatch(int batchSize) {
  return candidateBatchPriority(batchSize) >= 0 ? batchSize * kSequenceLength : 0;
}

namespace {

bool commonShapeEligible(const RuntimeShape& shape) {
  return modelStructureEligible(shape) &&
    candidateBatchPriority(shape.batchSize) >= 0 &&
    shape.boardX == kBoardX && shape.boardY == kBoardY &&
    shape.sequenceLength == kSequenceLength &&
    shape.enqueuedRows == tokenRowsForBatch(shape.batchSize) &&
    shape.deviceOrdinal >= 0 &&
    shape.channels == kChannels &&
    shape.computeCapability == kComputeCapability &&
    shape.usingFp16 && shape.usingNhwc && shape.exactNoMask;
}

}  // namespace

bool modelStructureEligible(const RuntimeShape& shape) {
  return shape.modelDepth > 0 &&
    shape.attentionBlockCount == shape.modelDepth &&
    shape.ffnBlockCount == shape.modelDepth &&
    shape.alternatingAttentionFfn;
}

bool attentionShapeEligible(const RuntimeShape& shape) {
  return commonShapeEligible(shape) &&
    shape.numHeads == kNumHeads && shape.numKvHeads == kNumKvHeads &&
    shape.qHeadDim == kHeadDim && shape.vHeadDim == kHeadDim &&
    shape.ropePairsTotal == kRopePairsTotal && shape.learnedRope;
}

bool ffnShapeEligible(const RuntimeShape& shape) {
  return commonShapeEligible(shape) &&
    shape.ffnChannels == kFfnChannels && shape.swiglu;
}

bool targetShapeEligible(const RuntimeShape& shape) {
  return shape.modelDepth == 36 &&
    attentionShapeEligible(shape) && ffnShapeEligible(shape);
}

bool pieceShapeEligible(const RuntimeShape& shape, Family family) {
  if(family == Family::QkvRope)
    return attentionShapeEligible(shape);
  if(family == Family::DualFfn)
    return ffnShapeEligible(shape);
  return false;
}

bool preparedPackedFa4Compatible(
  const RuntimeShape& shape,
  const PreparedPackedFa4* preparedPackedFa4
) {
  return attentionShapeEligible(shape) && preparedPackedFa4 != nullptr &&
    preparedPackedFa4->abiVersion == kPackedFa4ProofAbiVersion &&
    preparedPackedFa4->batchSize == shape.batchSize &&
    preparedPackedFa4->sequenceLength == shape.sequenceLength &&
    preparedPackedFa4->numHeads == shape.numHeads &&
    preparedPackedFa4->numKvHeads == shape.numKvHeads &&
    preparedPackedFa4->qHeadDim == shape.qHeadDim &&
    preparedPackedFa4->vHeadDim == shape.vHeadDim &&
    preparedPackedFa4->deviceOrdinal == shape.deviceOrdinal &&
    preparedPackedFa4->acceptsPackedTokenQkv &&
    preparedPackedFa4->id != nullptr && preparedPackedFa4->id[0] != '\0' &&
    preparedPackedFa4->implementationCookie != 0;
}

bool tacticKeyWellFormed(const TacticKey& key) {
  if(key.abiVersion != kRegistryAbiVersion ||
     candidateBatchPriority(key.batchSize) < 0 ||
     key.tokenRows != tokenRowsForBatch(key.batchSize) ||
     key.id == nullptr || key.id[0] == '\0')
    return false;
  if(key.family == Family::QkvRope)
    return key.launchGridSms == 0 && key.packedQkvOutput && !key.pairedFfnWeights;
  if(key.family == Family::DualFfn)
    return (key.launchGridSms == 170 || key.launchGridSms == 340) &&
      !key.packedQkvOutput && key.pairedFfnWeights;
  return false;
}

bool tacticKeyCompatible(const RuntimeShape& shape, const TacticKey& key) {
  return pieceShapeEligible(shape,key.family) && tacticKeyWellFormed(key) &&
    key.batchSize == shape.batchSize &&
    key.tokenRows == shape.batchSize * shape.sequenceLength;
}

bool registrySpanWellFormed(Family family, const RegistrySpan& registry) {
  if(registry.count == 0)
    return true;
  if(registry.entries == nullptr || registry.stride < sizeof(TacticKey))
    return false;
  const unsigned char* entries =
    static_cast<const unsigned char*>(registry.entries);
  for(std::size_t i = 0; i < registry.count; i++) {
    const TacticKey* key = reinterpret_cast<const TacticKey*>(
      entries + i * registry.stride);
    if(key->family != family || !tacticKeyWellFormed(*key))
      return false;
    for(std::size_t j = 0; j < i; j++) {
      const TacticKey* previous = reinterpret_cast<const TacticKey*>(
        entries + j * registry.stride);
      if(previous->batchSize == key->batchSize &&
         std::strcmp(previous->id,key->id) == 0)
        return false;
    }
  }
  return true;
}

const TacticKey* findExactTactic(
  const RuntimeShape& shape,
  Family family,
  const RegistrySpan& registry,
  const char* requestedId
) {
  if(requestedId == nullptr || requestedId[0] == '\0' ||
     !registrySpanWellFormed(family,registry) ||
     registry.entries == nullptr || registry.count == 0 ||
     registry.stride < sizeof(TacticKey))
    return nullptr;
  const unsigned char* entries =
    static_cast<const unsigned char*>(registry.entries);
  for(std::size_t i = 0; i < registry.count; i++) {
    const TacticKey* key = reinterpret_cast<const TacticKey*>(
      entries + i * registry.stride);
    if(key->family == family && tacticKeyCompatible(shape,*key) &&
       std::strcmp(key->id,requestedId) == 0)
      return key;
  }
  return nullptr;
}

Selection select(
  const RuntimeShape& shape,
  const RegistryView& registry,
  const char* requestedQkvRopeId,
  const char* requestedDualFfnId,
  const PreparedPackedFa4* preparedPackedFa4
) {
  Selection result;
  result.targetShape = targetShapeEligible(shape);
  result.qkvRope = selectPiece(
    shape,Family::QkvRope,registry,requestedQkvRopeId);
  result.dualFfn = selectPiece(
    shape,Family::DualFfn,registry,requestedDualFfnId);
  if(result.qkvRope.selected() &&
     !preparedPackedFa4Compatible(shape,preparedPackedFa4)) {
    result.qkvRope.tactic = nullptr;
    result.qkvRope.reason = RejectReason::MissingSameBatchFa4;
  }
  else if(result.qkvRope.selected())
    result.packedFa4 = preparedPackedFa4;
  return result;
}

const char* rejectReasonName(RejectReason reason) {
  switch(reason) {
  case RejectReason::None: return "none";
  case RejectReason::ShapeMismatch: return "shape-mismatch";
  case RejectReason::NoRequestedTactic: return "no-requested-tactic";
  case RejectReason::InvalidRegistry: return "invalid-registry";
  case RejectReason::RegistryMiss: return "registry-miss";
  case RejectReason::MissingSameBatchFa4: return "missing-same-batch-fa4";
  case RejectReason::InvalidImplementation: return "invalid-implementation";
  case RejectReason::PreparationFailed: return "preparation-failed";
  }
  return "unknown";
}

}  // namespace C384ExactFixedAot

#include "ffn_down.h"

#include <cstring>

namespace C384ExactFfnDownAot {

namespace {

bool aligned16(const void* pointer) {
  return pointer != nullptr &&
    (reinterpret_cast<std::uintptr_t>(pointer) & 0x0fU) == 0;
}

}  // namespace

bool targetShapeEligible(const RuntimeShape& shape) {
  return shape.modelDepth > 0 &&
    shape.attentionBlockCount == shape.modelDepth &&
    shape.ffnBlockCount == shape.modelDepth &&
    shape.alternatingAttentionFfn &&
    shape.batchSize == kSelectedBatch &&
    shape.tokenRows == kSelectedTokenRows &&
    shape.boardX == kBoardX && shape.boardY == kBoardY &&
    shape.sequenceLength == kSequenceLength &&
    shape.channels == kChannels && shape.ffnChannels == kInputChannels &&
    shape.deviceOrdinal >= 0 &&
    shape.computeCapability == kComputeCapability &&
    shape.usingFp16 && shape.usingNhwc && shape.exactNoMask && shape.swiglu;
}

bool tacticWellFormed(const Tactic& tactic) {
  return tactic.registryAbiVersion == kRegistryAbiVersion &&
    tactic.nativeAbiVersion == kNativeAbiVersion &&
    tactic.batchSize == kSelectedBatch &&
    tactic.tokenRows == kSelectedTokenRows &&
    tactic.inputChannels == kInputChannels &&
    tactic.outputChannels == kOutputChannels &&
    tactic.id != nullptr && tactic.id[0] != '\0' &&
    std::strcmp(tactic.id,kProductionTacticId) == 0 &&
    tactic.prepare != nullptr && tactic.query != nullptr &&
    tactic.launch != nullptr;
}

bool registryWellFormed(const Tactic* tactics, std::size_t count) {
  if(count == 0)
    return tactics == nullptr;
  if(tactics == nullptr)
    return false;
  for(std::size_t i = 0; i < count; i++) {
    if(!tacticWellFormed(tactics[i]))
      return false;
    for(std::size_t j = 0; j < i; j++) {
      if(std::strcmp(tactics[i].id,tactics[j].id) == 0)
        return false;
    }
  }
  // A production registry is collapsed to exactly one B28 tactic.
  return count == 1;
}

bool queryMatchesTactic(
  const C384ResidualRawDescriptorV1& native,
  const Tactic& tactic
) {
  return tacticWellFormed(tactic) &&
    native.abi_version == C384ResidualAotAbi::kAbiVersion &&
    native.family == static_cast<std::uint32_t>(
      C384ResidualAotAbi::Family::FfnDownResidual) &&
    native.batch == static_cast<std::uint32_t>(tactic.batchSize) &&
    native.m == static_cast<std::uint32_t>(tactic.tokenRows) &&
    native.k == static_cast<std::uint32_t>(tactic.inputChannels) &&
    native.n == static_cast<std::uint32_t>(tactic.outputChannels) &&
    native.tile_m == static_cast<std::uint32_t>(kTileM) &&
    native.tile_n == static_cast<std::uint32_t>(kTileN) &&
    native.tile_k == static_cast<std::uint32_t>(kTileK) &&
    native.stages == static_cast<std::uint32_t>(kStages) &&
    native.atom_m == static_cast<std::uint32_t>(kAtomM) &&
    native.atom_n == static_cast<std::uint32_t>(kAtomN) &&
    native.atom_k == static_cast<std::uint32_t>(kAtomK) &&
    native.epilogue_stages ==
      static_cast<std::uint32_t>(kEpilogueStages) &&
    native.natural_ctas == static_cast<std::uint32_t>(kNaturalCtas) &&
    native.launch_ctas == static_cast<std::uint32_t>(kLaunchCtas) &&
    native.max_active_clusters ==
      static_cast<std::uint32_t>(kMaxActiveClusters) &&
    native.candidate_id != nullptr &&
    std::strcmp(native.candidate_id,tactic.id) == 0;
}

PreparedSelection prepareSelectionFromRegistry(
  const RuntimeShape& shape,
  const char* requestedId,
  const Tactic* tactics,
  std::size_t count
) {
  PreparedSelection result;
  if(!targetShapeEligible(shape)) {
    result.reason = RejectReason::ShapeMismatch;
    return result;
  }
  if(requestedId == nullptr || requestedId[0] == '\0') {
    result.reason = RejectReason::NoRequestedTactic;
    return result;
  }
  if(!registryWellFormed(tactics,count)) {
    result.reason = RejectReason::InvalidRegistry;
    return result;
  }
  const Tactic* selected = nullptr;
  for(std::size_t i = 0; i < count; i++) {
    if(std::strcmp(tactics[i].id,requestedId) == 0) {
      selected = &tactics[i];
      break;
    }
  }
  if(selected == nullptr) {
    result.reason = RejectReason::RegistryMiss;
    return result;
  }
  if(!tacticWellFormed(*selected)) {
    result.reason = RejectReason::InvalidImplementation;
    return result;
  }

  C384ResidualRawDescriptorV1 native{};
  if(selected->query(&native) != cudaSuccess) {
    result.reason = RejectReason::QueryFailed;
    return result;
  }
  if(!queryMatchesTactic(native,*selected)) {
    result.reason = RejectReason::QueryMismatch;
    return result;
  }
  if(selected->prepare(shape.deviceOrdinal) != cudaSuccess) {
    result.reason = RejectReason::PreparationFailed;
    return result;
  }
  result.tactic = selected;
  result.reason = RejectReason::None;
  result.deviceOrdinal = shape.deviceOrdinal;
  return result;
}

bool supports(
  const PreparedSelection& selection,
  const RuntimeShape& actualShape,
  const void* activation,
  const void* rowMajorWeights,
  const void* residualInOut
) {
  if(!selection.selected() || selection.tactic == nullptr ||
     !tacticWellFormed(*selection.tactic) ||
     !targetShapeEligible(actualShape) ||
     selection.deviceOrdinal != actualShape.deviceOrdinal ||
     selection.tactic->batchSize != actualShape.batchSize ||
     selection.tactic->tokenRows != actualShape.tokenRows)
    return false;

  return aligned16(activation) && aligned16(rowMajorWeights) &&
    aligned16(residualInOut);
}

const char* rejectReasonName(RejectReason reason) {
  switch(reason) {
  case RejectReason::None: return "none";
  case RejectReason::ShapeMismatch: return "shape-mismatch";
  case RejectReason::NoRequestedTactic: return "no-requested-tactic";
  case RejectReason::InvalidRegistry: return "invalid-registry";
  case RejectReason::RegistryMiss: return "registry-miss";
  case RejectReason::InvalidImplementation: return "invalid-implementation";
  case RejectReason::QueryFailed: return "query-failed";
  case RejectReason::QueryMismatch: return "query-mismatch";
  case RejectReason::PreparationFailed: return "preparation-failed";
  }
  return "unknown";
}

}  // namespace C384ExactFfnDownAot

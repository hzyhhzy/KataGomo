#include "../neuralnet/c384_exact_ffn_down_aot.h"
#include "../neuralnet/c384_exact_fixed_aot_plan.h"

#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace C384ExactFfnDownAot;

namespace {

constexpr const char* kId = kProductionTacticId;

C384ResidualRawDescriptorV1 descriptor{};
cudaError_t queryStatus = cudaSuccess;
cudaError_t prepareStatus = cudaSuccess;
int preparedDevice = -1;
int launchCount = 0;

void resetDescriptor() {
  descriptor = C384ResidualRawDescriptorV1{};
  descriptor.abi_version = C384ResidualAotAbi::kAbiVersion;
  descriptor.family = static_cast<std::uint32_t>(
    C384ResidualAotAbi::Family::FfnDownResidual);
  descriptor.batch = kSelectedBatch;
  descriptor.m = kSelectedTokenRows;
  descriptor.k = kInputChannels;
  descriptor.n = kOutputChannels;
  descriptor.tile_m = kTileM;
  descriptor.tile_n = kTileN;
  descriptor.tile_k = kTileK;
  descriptor.stages = kStages;
  descriptor.atom_m = kAtomM;
  descriptor.atom_n = kAtomN;
  descriptor.atom_k = kAtomK;
  descriptor.epilogue_stages = kEpilogueStages;
  descriptor.natural_ctas = kNaturalCtas;
  descriptor.launch_ctas = kLaunchCtas;
  descriptor.max_active_clusters = kMaxActiveClusters;
  descriptor.candidate_id = kId;
  queryStatus = cudaSuccess;
  prepareStatus = cudaSuccess;
  preparedDevice = -1;
  launchCount = 0;
}

cudaError_t query(C384ResidualRawDescriptorV1* output) {
  if(queryStatus != cudaSuccess)
    return queryStatus;
  if(output == nullptr)
    return cudaErrorInvalidValue;
  *output = descriptor;
  return cudaSuccess;
}

cudaError_t prepare(int deviceOrdinal) {
  preparedDevice = deviceOrdinal;
  return prepareStatus;
}

cudaError_t launch(
  const half*, const half*, half*, int, int, cudaStream_t
) {
  launchCount++;
  return cudaSuccess;
}

Tactic validTactic() {
  return {
    kRegistryAbiVersion,kNativeAbiVersion,kSelectedBatch,kSelectedTokenRows,
    kInputChannels,kOutputChannels,kId,prepare,query,launch
  };
}

RuntimeShape validShape() {
  RuntimeShape shape;
  shape.modelDepth = 36;
  shape.attentionBlockCount = 36;
  shape.ffnBlockCount = 36;
  shape.alternatingAttentionFfn = true;
  shape.batchSize = kSelectedBatch;
  shape.tokenRows = kSelectedTokenRows;
  shape.boardX = kBoardX;
  shape.boardY = kBoardY;
  shape.sequenceLength = kSequenceLength;
  shape.channels = kChannels;
  shape.ffnChannels = kInputChannels;
  shape.deviceOrdinal = 0;
  shape.computeCapability = kComputeCapability;
  shape.usingFp16 = true;
  shape.usingNhwc = true;
  shape.exactNoMask = true;
  shape.swiglu = true;
  return shape;
}

void require(bool condition, const char* message) {
  if(!condition) {
    std::cerr << "C384_EXACT_FFN_DOWN_CONTRACT_FAILED: " << message
              << std::endl;
    std::exit(1);
  }
}

}  // namespace

int main() {
  require(C384ExactFixedAot::productionBatchEligible(28,6300),
    "shared production plan rejected B28/M6300");
  require(!C384ExactFixedAot::productionBatchEligible(24,5400),
    "shared production plan admitted B24/M5400");
  resetDescriptor();
  const RuntimeShape shape = validShape();
  Tactic tactic = validTactic();
  require(targetShapeEligible(shape),"B28 target shape rejected");
  require(tacticWellFormed(tactic),"typed B28 tactic rejected");
  require(registryWellFormed(&tactic,1),"one-entry production registry rejected");

  RuntimeShape b24 = shape;
  b24.batchSize = 24;
  b24.tokenRows = 24 * kSequenceLength;
  require(!targetShapeEligible(b24),"B24 entered the production AOT route");
  RuntimeShape tail = shape;
  tail.tokenRows--;
  require(!targetShapeEligible(tail),"nonintegral/tail rows entered AOT");
  RuntimeShape wrongLayout = shape;
  wrongLayout.usingNhwc = false;
  require(!targetShapeEligible(wrongLayout),"NCHW entered AOT");
  RuntimeShape masked = shape;
  masked.exactNoMask = false;
  require(!targetShapeEligible(masked),"masked input entered AOT");

  PreparedSelection selected = prepareSelectionFromRegistry(
    shape,kId,&tactic,1);
  require(selected.selected(),"valid tactic did not prepare");
  require(preparedDevice == 0,"prepare did not bind the selected device");
  require(supports(
      selected,shape,reinterpret_cast<void*>(0x1000),
      reinterpret_cast<void*>(0x2000),reinterpret_cast<void*>(0x3000)),
    "aligned exact launch preflight failed");
  require(!supports(
      selected,shape,reinterpret_cast<void*>(0x1002),
      reinterpret_cast<void*>(0x2000),reinterpret_cast<void*>(0x3000)),
    "unaligned activation did not fall back before enqueue");
  PreparedSelection invalidTacticSelection = selected;
  Tactic malformedSelected = tactic;
  malformedSelected.id = "not-the-promoted-id";
  invalidTacticSelection.tactic = &malformedSelected;
  require(!supports(
      invalidTacticSelection,shape,reinterpret_cast<void*>(0x1000),
      reinterpret_cast<void*>(0x2000),reinterpret_cast<void*>(0x3000)),
    "hot-path preflight accepted a malformed published tactic");
  require(!supports(
      selected,b24,reinterpret_cast<void*>(0x1000),
      reinterpret_cast<void*>(0x2000),reinterpret_cast<void*>(0x3000)),
    "B24 reused a B28 prepared selection");
  RuntimeShape wrongDevice = shape;
  wrongDevice.deviceOrdinal = 1;
  require(!supports(
      selected,wrongDevice,reinterpret_cast<void*>(0x1000),
      reinterpret_cast<void*>(0x2000),reinterpret_cast<void*>(0x3000)),
    "prepared selection crossed devices");
  require(launchCount == 0,"support preflight enqueued work");

  require(prepareSelectionFromRegistry(shape,"missing",&tactic,1).reason ==
      RejectReason::RegistryMiss,
    "unknown ID did not fail closed");
  require(prepareSelectionFromRegistry(shape,"",&tactic,1).reason ==
      RejectReason::NoRequestedTactic,
    "empty ID did not select generic fallback");
  require(prepareSelectionFromRegistry(b24,kId,&tactic,1).reason ==
      RejectReason::ShapeMismatch,
    "B24 rejection happened after registry preparation");

  descriptor.family = static_cast<std::uint32_t>(
    C384ResidualAotAbi::Family::OutProjResidual);
  require(prepareSelectionFromRegistry(shape,kId,&tactic,1).reason ==
      RejectReason::QueryMismatch,
    "out-projection descriptor entered the down registry");
  resetDescriptor();
  descriptor.m--;
  require(prepareSelectionFromRegistry(shape,kId,&tactic,1).reason ==
      RejectReason::QueryMismatch,
    "stale exact-M descriptor prepared");
  resetDescriptor();
  descriptor.launch_ctas--;
  require(prepareSelectionFromRegistry(shape,kId,&tactic,1).reason ==
      RejectReason::QueryMismatch,
    "stale launch-grid descriptor prepared");
  resetDescriptor();
  queryStatus = cudaErrorInvalidValue;
  require(prepareSelectionFromRegistry(shape,kId,&tactic,1).reason ==
      RejectReason::QueryFailed,
    "query failure did not fail closed");
  resetDescriptor();
  prepareStatus = cudaErrorInvalidDevice;
  require(prepareSelectionFromRegistry(shape,kId,&tactic,1).reason ==
      RejectReason::PreparationFailed,
    "prepare failure did not fail closed");

  resetDescriptor();
  Tactic b24Tactic = tactic;
  b24Tactic.batchSize = 24;
  b24Tactic.tokenRows = 24 * kSequenceLength;
  require(!tacticWellFormed(b24Tactic),"B24 descriptor accepted by production registry");
  Tactic duplicated[] = {tactic,tactic};
  require(!registryWellFormed(duplicated,2),"multi-entry production registry accepted");
  require(registryWellFormed(nullptr,0),"empty provider is not a valid disabled state");
  require(prepareSelection(shape,kId).reason == RejectReason::RegistryMiss,
    "checked-in empty provider did not fail closed before enqueue");

  std::cout << "C384 exact B28 FFN-down contract PASS" << std::endl;
  return 0;
}

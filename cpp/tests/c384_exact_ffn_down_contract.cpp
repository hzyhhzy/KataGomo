#include "../neuralnet/cuda_specialized/sm120/c384/fixed_batch/ffn_down.h"
#include "../neuralnet/cuda_specialized/sm120/c384/fixed_batch/plan.h"

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
    "shared production plan rejected bs28/M6300");
  require(!C384ExactFixedAot::productionBatchEligible(24,5400),
    "shared production plan admitted bs24/M5400");
  resetDescriptor();
  const RuntimeShape shape = validShape();
  Tactic tactic = validTactic();
  require(targetShapeEligible(shape),"b36 bs28 target shape rejected");
  RuntimeShape b24Model = shape;
  b24Model.modelDepth = 24;
  b24Model.attentionBlockCount = 24;
  b24Model.ffnBlockCount = 24;
  require(targetShapeEligible(b24Model),"b24 bs28 target shape rejected");
  const C384ExactFixedAot::TransactionProgress b24Complete = {
    24,24,24,24,24,24
  };
  const C384ExactFixedAot::TransactionProgress b36Complete = {
    36,36,36,36,36,36
  };
  require(C384ExactFixedAot::transactionProgressComplete(b24Complete),
    "complete b24 transaction rejected");
  require(C384ExactFixedAot::transactionProgressComplete(b36Complete),
    "complete b36 transaction rejected");
  C384ExactFixedAot::TransactionProgress incomplete = b24Complete;
  incomplete.qkvFa4Count--;
  require(!C384ExactFixedAot::transactionProgressComplete(incomplete),
    "incomplete b24 transaction committed");
  incomplete = b24Complete;
  incomplete.modelDepth = 36;
  require(!C384ExactFixedAot::transactionProgressComplete(incomplete),
    "mismatched typed model depth committed");
  RuntimeShape unequalModel = shape;
  unequalModel.attentionBlockCount--;
  require(!targetShapeEligible(unequalModel),
    "unequal attention/FFN counts entered AOT");
  RuntimeShape nonAlternatingModel = shape;
  nonAlternatingModel.alternatingAttentionFfn = false;
  require(!targetShapeEligible(nonAlternatingModel),
    "non-alternating model entered AOT");
  RuntimeShape emptyModel = shape;
  emptyModel.modelDepth = 0;
  emptyModel.attentionBlockCount = 0;
  emptyModel.ffnBlockCount = 0;
  require(!targetShapeEligible(emptyModel),"empty model entered AOT");
  RuntimeShape c256Model = shape;
  c256Model.channels = 256;
  require(!targetShapeEligible(c256Model),"C256 model entered C384 AOT");
  require(tacticWellFormed(tactic),"typed bs28 tactic rejected");
  require(registryWellFormed(&tactic,1),"one-entry production registry rejected");

  RuntimeShape bs24 = shape;
  bs24.batchSize = 24;
  bs24.tokenRows = 24 * kSequenceLength;
  require(!targetShapeEligible(bs24),"bs24 entered the production AOT route");
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
      selected,bs24,reinterpret_cast<void*>(0x1000),
      reinterpret_cast<void*>(0x2000),reinterpret_cast<void*>(0x3000)),
    "bs24 reused a bs28 prepared selection");
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
  require(prepareSelectionFromRegistry(bs24,kId,&tactic,1).reason ==
      RejectReason::ShapeMismatch,
    "bs24 rejection happened after registry preparation");

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
  Tactic bs24Tactic = tactic;
  bs24Tactic.batchSize = 24;
  bs24Tactic.tokenRows = 24 * kSequenceLength;
  require(!tacticWellFormed(bs24Tactic),
    "bs24 descriptor accepted by production registry");
  Tactic duplicated[] = {tactic,tactic};
  require(!registryWellFormed(duplicated,2),"multi-entry production registry accepted");
  require(registryWellFormed(nullptr,0),"empty provider is not a valid disabled state");
  require(prepareSelection(shape,kId).reason == RejectReason::RegistryMiss,
    "checked-in empty provider did not fail closed before enqueue");

  std::cout << "C384 exact B28 FFN-down contract PASS" << std::endl;
  return 0;
}

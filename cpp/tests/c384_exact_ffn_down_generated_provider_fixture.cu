#include "c384_exact_ffn_down_aot.h"
#include "c384_residual_aot_abi.h"

namespace {

cudaError_t fixturePrepare(int deviceOrdinal) {
  return deviceOrdinal == 0 ? cudaSuccess : cudaErrorInvalidDevice;
}

cudaError_t fixtureQuery(C384ResidualRawDescriptorV1* output) {
  if(output == nullptr)
    return cudaErrorInvalidValue;
  *output = C384ResidualRawDescriptorV1{
    C384ResidualAotAbi::kAbiVersion,
    static_cast<std::uint32_t>(
      C384ResidualAotAbi::Family::FfnDownResidual),
    C384ExactFfnDownAot::kSelectedBatch,
    C384ExactFfnDownAot::kSelectedTokenRows,
    C384ExactFfnDownAot::kInputChannels,
    C384ExactFfnDownAot::kOutputChannels,
    C384ExactFfnDownAot::kTileM,
    C384ExactFfnDownAot::kTileN,
    C384ExactFfnDownAot::kTileK,
    C384ExactFfnDownAot::kStages,
    C384ExactFfnDownAot::kAtomM,
    C384ExactFfnDownAot::kAtomN,
    C384ExactFfnDownAot::kAtomK,
    C384ExactFfnDownAot::kEpilogueStages,
    C384ExactFfnDownAot::kNaturalCtas,
    C384ExactFfnDownAot::kLaunchCtas,
    C384ExactFfnDownAot::kMaxActiveClusters,
    C384ExactFfnDownAot::kProductionTacticId
  };
  return cudaSuccess;
}

cudaError_t fixtureLaunch(
  const half* activation,
  const half* weights,
  half* residual,
  int tokenRows,
  int deviceOrdinal,
  cudaStream_t
) {
  return activation != nullptr && weights != nullptr && residual != nullptr &&
    tokenRows == C384ExactFfnDownAot::kSelectedTokenRows &&
    deviceOrdinal == 0 ? cudaSuccess : cudaErrorInvalidValue;
}

const C384ExactFfnDownAot::Tactic fixtureTactic{
  C384ExactFfnDownAot::kRegistryAbiVersion,
  C384ExactFfnDownAot::kNativeAbiVersion,
  C384ExactFfnDownAot::kSelectedBatch,
  C384ExactFfnDownAot::kSelectedTokenRows,
  C384ExactFfnDownAot::kInputChannels,
  C384ExactFfnDownAot::kOutputChannels,
  C384ExactFfnDownAot::kProductionTacticId,
  fixturePrepare,
  fixtureQuery,
  fixtureLaunch
};

}  // namespace

namespace C384ExactFfnDownAot {

const Tactic* generatedTactics(std::size_t& count) {
  count = 1;
  return &fixtureTactic;
}

}  // namespace C384ExactFfnDownAot

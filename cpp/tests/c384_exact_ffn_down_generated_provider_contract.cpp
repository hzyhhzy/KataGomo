#include "c384_exact_ffn_down_aot.h"

#include <iostream>

int main() {
  C384ExactFfnDownAot::RuntimeShape shape;
  shape.modelDepth = 36;
  shape.attentionBlockCount = 36;
  shape.ffnBlockCount = 36;
  shape.alternatingAttentionFfn = true;
  shape.batchSize = C384ExactFfnDownAot::kSelectedBatch;
  shape.tokenRows = C384ExactFfnDownAot::kSelectedTokenRows;
  shape.boardX = C384ExactFfnDownAot::kBoardX;
  shape.boardY = C384ExactFfnDownAot::kBoardY;
  shape.sequenceLength = C384ExactFfnDownAot::kSequenceLength;
  shape.channels = C384ExactFfnDownAot::kChannels;
  shape.ffnChannels = C384ExactFfnDownAot::kInputChannels;
  shape.deviceOrdinal = 0;
  shape.computeCapability = C384ExactFfnDownAot::kComputeCapability;
  shape.usingFp16 = true;
  shape.usingNhwc = true;
  shape.exactNoMask = true;
  shape.swiglu = true;

  const C384ExactFfnDownAot::PreparedSelection selected =
    C384ExactFfnDownAot::prepareSelection(
      shape,C384ExactFfnDownAot::kProductionTacticId);
  alignas(16) unsigned char activationStorage[16]{};
  alignas(16) unsigned char weightStorage[16]{};
  alignas(16) unsigned char residualStorage[16]{};
  const half* activation = reinterpret_cast<const half*>(activationStorage);
  const half* weights = reinterpret_cast<const half*>(weightStorage);
  half* residual = reinterpret_cast<half*>(residualStorage);
  if(!selected.selected() ||
     !C384ExactFfnDownAot::supports(
       selected,shape,activation,weights,residual) ||
     selected.tactic->launch(
       activation,weights,residual,shape.tokenRows,shape.deviceOrdinal,
       nullptr) != cudaSuccess) {
    std::cerr << "C384_EXACT_FFN_DOWN_GENERATED_PROVIDER_CONTRACT_FAILED"
              << std::endl;
    return 1;
  }
  std::cout << "C384 exact B28 generated-provider contract PASS" << std::endl;
  return 0;
}

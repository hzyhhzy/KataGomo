#ifndef KATAGO_C384_EXPERIMENTAL_INT8_POLICY_H_
#define KATAGO_C384_EXPERIMENTAL_INT8_POLICY_H_

#include <cstdint>
#include <cstring>

namespace C384Int8Experiment {

enum class EngineMode : uint32_t {
  Off = 0,
  Conservative = 1,
  Aggressive = 2,
  Invalid = 3,
};

inline EngineMode parseEngineMode(const char* value) noexcept {
  if(value == nullptr || value[0] == '\0' || std::strcmp(value,"off") == 0)
    return EngineMode::Off;
  if(std::strcmp(value,"conservative") == 0)
    return EngineMode::Conservative;
  if(std::strcmp(value,"aggressive") == 0)
    return EngineMode::Aggressive;
  return EngineMode::Invalid;
}

inline const char* engineModeName(EngineMode mode) noexcept {
  switch(mode) {
  case EngineMode::Off: return "off";
  case EngineMode::Conservative: return "conservative";
  case EngineMode::Aggressive: return "aggressive";
  case EngineMode::Invalid: return "invalid";
  }
  return "invalid";
}

struct EngineEligibility {
  int modelVersion = 0;
  int batchSize = 0;
  int boardX = 0;
  int boardY = 0;
  int modelDepth = 0;
  int attentionBlocks = 0;
  int ffnBlocks = 0;
  int channels = 0;
  int heads = 0;
  int headDim = 0;
  int ffnChannels = 0;
  uint32_t computeCapability = 0;
  bool alternatingAttentionFfn = false;
  bool usingFp16 = false;
  bool usingNhwc = false;
  bool exactNoMask = false;
  bool learnedRope = false;
  bool qkNorm = false;
  bool calibratedFfnProduct = false;
};

inline bool engineShapeEligible(const EngineEligibility& shape) noexcept {
  return shape.modelVersion == 105 && shape.batchSize == 28 &&
    shape.boardX == 15 && shape.boardY == 15 &&
    shape.modelDepth == 36 && shape.attentionBlocks == 36 &&
    shape.ffnBlocks == 36 && shape.alternatingAttentionFfn &&
    shape.channels == 384 && shape.heads == 12 && shape.headDim == 32 &&
    shape.ffnChannels == 1024 && shape.computeCapability == 120 &&
    shape.usingFp16 && shape.usingNhwc && shape.exactNoMask &&
    shape.learnedRope && shape.qkNorm && shape.calibratedFfnProduct;
}

}  // namespace C384Int8Experiment

#endif  // KATAGO_C384_EXPERIMENTAL_INT8_POLICY_H_

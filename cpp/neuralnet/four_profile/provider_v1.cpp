#include "provider_v1.h"

#include <cmath>
#include <cstring>

namespace FourProfile {
namespace {

float floatFromBits(uint32_t bits) {
  float value;
  static_assert(sizeof(value) == sizeof(bits),"32-bit float required");
  std::memcpy(&value,&bits,sizeof(value));
  return value;
}

bool finitePositiveBits(uint32_t bits) {
  const float value = floatFromBits(bits);
  return std::isfinite(value) && value > 0.0f;
}

bool optionalRangeValid(bool present, uint32_t bits) {
  return present ? finitePositiveBits(bits) : bits == 0;
}

bool layerScalarsValid(const BlockViewV1& attention, const BlockViewV1& ffn) {
  if(!optionalRangeValid(
       attention.attention.hasInputQuantRange,
       attention.attentionScalars.inputQuantMaxAbsBits
     ) ||
     !optionalRangeValid(
       attention.attention.hasOutputQuantRange,
       attention.attentionScalars.outputQuantMaxAbsBits
     ) ||
     !optionalRangeValid(ffn.ffn.hasInputQuantRange,ffn.ffnScalars.inputQuantMaxAbsBits) ||
     !optionalRangeValid(ffn.ffn.hasProductQuantRange,ffn.ffnScalars.productQuantMaxAbsBits))
    return false;

  if(ffn.ffn.clipClass == ClipClassV1::Zero)
    return ffn.ffnScalars.swigluClipBits == 0;
  if(ffn.ffn.clipClass == ClipClassV1::PositiveFinite)
    return finitePositiveBits(ffn.ffnScalars.swigluClipBits);
  return false;
}

}  // namespace

bool AttentionSpecV1::operator==(const AttentionSpecV1& other) const {
  return channels == other.channels &&
    numHeads == other.numHeads &&
    numKVHeads == other.numKVHeads &&
    qHeadDim == other.qHeadDim &&
    vHeadDim == other.vHeadDim &&
    useRope == other.useRope &&
    learnableRope == other.learnableRope &&
    useQKNorm == other.useQKNorm &&
    hasInputQuantRange == other.hasInputQuantRange &&
    hasOutputQuantRange == other.hasOutputQuantRange;
}

bool FfnSpecV1::operator==(const FfnSpecV1& other) const {
  return channels == other.channels &&
    hiddenChannels == other.hiddenChannels &&
    useSwiGLU == other.useSwiGLU &&
    clipClass == other.clipClass &&
    hasInputQuantRange == other.hasInputQuantRange &&
    hasProductQuantRange == other.hasProductQuantRange;
}

bool RuntimeKeyV1::operator==(const RuntimeKeyV1& other) const {
  return deviceComputeCapability == other.deviceComputeCapability &&
    boardX == other.boardX &&
    boardY == other.boardY &&
    physicalBatchSize == other.physicalBatchSize &&
    sameGpuConcurrency == other.sameGpuConcurrency &&
    exactBoard == other.exactBoard &&
    maskMode == other.maskMode &&
    maskNull == other.maskNull &&
    inputStorage == other.inputStorage &&
    outputStorage == other.outputStorage &&
    requestedExecution == other.requestedExecution &&
    layout == other.layout;
}

bool ProfileKeyV1::operator==(const ProfileKeyV1& other) const {
  return modelVersion == other.modelVersion &&
    runtime == other.runtime &&
    attention == other.attention &&
    ffn == other.ffn;
}

bool RuntimeCallV1::sameIdentity(const RuntimeCallV1& other) const {
  return key == other.key &&
    actualBatchSize == other.actualBatchSize &&
    sequenceSize == other.sequenceSize &&
    transformerBeginBlock == other.transformerBeginBlock &&
    transformerPairCount == other.transformerPairCount &&
    stream == other.stream &&
    trunk == other.trunk &&
    trunkScratch == other.trunkScratch &&
    mask == other.mask &&
    workspace == other.workspace &&
    workspaceBytes == other.workspaceBytes;
}

TransformerSpanV1 analyzeTransformerSpanV1(const std::vector<BlockViewV1>& blocks) {
  TransformerSpanV1 span;
  size_t first = blocks.size();
  size_t last = blocks.size();
  for(size_t i = 0; i < blocks.size(); i++) {
    const BlockKindV1 kind = blocks[i].kind;
    if(kind != BlockKindV1::TransformerAttention && kind != BlockKindV1::TransformerFfn)
      continue;
    if(first == blocks.size())
      first = i;
    last = i;
  }

  if(first == blocks.size()) {
    span.detail = "model has no transformer blocks";
    return span;
  }

  span.hasTransformer = true;
  span.beginBlock = first;
  span.endBlock = last + 1;
  const size_t blockCount = span.endBlock - span.beginBlock;
  if(blockCount == 0 || blockCount % 2 != 0) {
    span.detail = "transformer span does not contain an even number of blocks";
    return span;
  }

  for(size_t offset = 0; offset < blockCount; offset++) {
    const BlockKindV1 expected = offset % 2 == 0 ?
      BlockKindV1::TransformerAttention : BlockKindV1::TransformerFfn;
    if(blocks[first + offset].kind != expected) {
      span.detail = "transformer span must be one contiguous (attention,ffn)^N sequence";
      return span;
    }
  }

  span.valid = true;
  span.pairCount = blockCount / 2;
  span.detail = "valid contiguous transformer span";
  return span;
}

bool deriveProfileKeyV1(
  const ModelViewV1& model,
  const TransformerSpanV1& span,
  const RuntimeKeyV1& runtime,
  ProfileKeyV1& key,
  std::string& detail
) {
  if(!span.valid || span.pairCount == 0 ||
     span.endBlock > model.blocks.size() ||
     span.endBlock - span.beginBlock != span.pairCount * 2) {
    detail = "cannot derive a profile key from an invalid transformer span";
    return false;
  }

  const BlockViewV1& firstAttention = model.blocks[span.beginBlock];
  const BlockViewV1& firstFfn = model.blocks[span.beginBlock + 1];
  key = ProfileKeyV1();
  key.modelVersion = model.modelVersion;
  key.runtime = runtime;
  key.attention = firstAttention.attention;
  key.ffn = firstFfn.ffn;

  for(size_t layer = 0; layer < span.pairCount; layer++) {
    const BlockViewV1& attention = model.blocks[span.beginBlock + layer * 2];
    const BlockViewV1& ffn = model.blocks[span.beginBlock + layer * 2 + 1];
    if(attention.kind != BlockKindV1::TransformerAttention ||
       ffn.kind != BlockKindV1::TransformerFfn) {
      detail = "transformer span kind changed while deriving profile key";
      return false;
    }
    if(!layerScalarsValid(attention,ffn)) {
      detail = "transformer layer contains invalid clip or PTQ range scalars";
      return false;
    }
    if(attention.attention != key.attention) {
      detail = "attention layers in transformer span do not share one profile key";
      return false;
    }
    if(ffn.ffn != key.ffn) {
      detail = "FFN layers in transformer span do not share one profile key";
      return false;
    }
  }

  detail = "homogeneous profile key";
  return true;
}

ProviderOpResultV1 ProviderOpResultV1::success(
  size_t enqueuedOperations,
  uint64_t planGeneration_,
  uint64_t runToken_
) {
  ProviderOpResultV1 result;
  result.ok = true;
  result.enqueued = enqueuedOperations;
  result.planGeneration = planGeneration_;
  result.runToken = runToken_;
  return result;
}

ProviderOpResultV1 ProviderOpResultV1::failure(
  const std::string& detail,
  size_t enqueuedOperations
) {
  ProviderOpResultV1 result;
  result.ok = false;
  result.enqueued = enqueuedOperations;
  result.detail = detail;
  return result;
}

const char* availabilityNameV1(AvailabilityV1 availability) {
  switch(availability) {
  case AvailabilityV1::Available: return "available";
  case AvailabilityV1::Unavailable: return "unavailable";
  case AvailabilityV1::Uncertified: return "uncertified";
  }
  return "invalid";
}

const char* reasonNameV1(ReasonV1 reason) {
  switch(reason) {
  case ReasonV1::Disabled: return "disabled";
  case ReasonV1::Selected: return "selected";
  case ReasonV1::NoTransformer: return "no-transformer";
  case ReasonV1::Unmatched: return "unmatched";
  case ReasonV1::InvalidTransformerSpan: return "invalid-transformer-span";
  case ReasonV1::NonUniformTransformerSpan: return "nonuniform-transformer-span";
  case ReasonV1::ProviderUnavailable: return "provider-unavailable";
  case ReasonV1::ProviderUncertified: return "provider-uncertified";
  case ReasonV1::ProviderCreationFailed: return "provider-creation-failed";
  case ReasonV1::PrepareFailed: return "prepare-failed";
  case ReasonV1::CommitFailed: return "commit-failed";
  case ReasonV1::RuntimePreflightFailed: return "runtime-preflight-failed";
  case ReasonV1::EnqueueFailedBeforeWork: return "enqueue-failed-before-work";
  }
  return "invalid";
}

}  // namespace FourProfile

#include "native_model_view.h"

#include <cmath>
#include <cstring>

namespace FourProfile {
namespace {

uint32_t floatBits(float value) {
  uint32_t bits;
  static_assert(sizeof(bits) == sizeof(value),"32-bit float required");
  std::memcpy(&bits,&value,sizeof(bits));
  return bits;
}

ClipClassV1 clipClass(float clip) {
  if(clip == 0.0f)
    return ClipClassV1::Zero;
  if(std::isfinite(clip) && clip > 0.0f)
    return ClipClassV1::PositiveFinite;
  return ClipClassV1::Invalid;
}

}  // namespace

ModelViewV1 buildNativeModelViewV1(
  const ModelDesc& model,
  const NativeExecutableBlocksV1& executableBlocks
) {
  const auto& descriptors = model.trunk.blocks;
  if(descriptors.size() != executableBlocks.size()) {
    throw FatalErrorV1(
      "four-profile native descriptor/executable block counts differ"
    );
  }

  ModelViewV1 view;
  view.modelVersion = model.version;
  view.blocks.reserve(descriptors.size());
  const bool hasV105Scalars = model.version >= 105;

  for(size_t i = 0; i < descriptors.size(); i++) {
    const int kind = descriptors[i].first;
    if(executableBlocks[i].first != kind) {
      throw FatalErrorV1(
        "four-profile native descriptor/executable block kinds differ at index " +
        std::to_string(i)
      );
    }

    BlockViewV1 block;
    if(kind == TRANSFORMER_ATTENTION_BLOCK_KIND) {
      const auto* desc =
        static_cast<const TransformerAttentionDesc*>(descriptors[i].second.get());
      block.kind = BlockKindV1::TransformerAttention;
      block.attention.channels = desc->preLN.numChannels;
      block.attention.numHeads = desc->numHeads;
      block.attention.numKVHeads = desc->numKVHeads;
      block.attention.qHeadDim = desc->qHeadDim;
      block.attention.vHeadDim = desc->vHeadDim;
      block.attention.useRope = desc->useRope;
      block.attention.learnableRope = desc->learnableRope;
      block.attention.useQKNorm = desc->useQKNorm;
      block.attention.hasInputQuantRange = hasV105Scalars;
      block.attention.hasOutputQuantRange = hasV105Scalars;
      block.attentionScalars.inputQuantMaxAbsBits =
        hasV105Scalars ? floatBits(desc->attentionInputQuantMaxAbs) : 0;
      block.attentionScalars.outputQuantMaxAbsBits =
        hasV105Scalars ? floatBits(desc->attentionOutputQuantMaxAbs) : 0;
      block.officialAttention.descriptor = desc;
      block.officialAttention.executable = executableBlocks[i].second;
    }
    else if(kind == TRANSFORMER_FFN_BLOCK_KIND) {
      const auto* desc =
        static_cast<const TransformerFFNDesc*>(descriptors[i].second.get());
      block.kind = BlockKindV1::TransformerFfn;
      block.ffn.channels = desc->numChannels;
      block.ffn.hiddenChannels = desc->ffnChannels;
      block.ffn.useSwiGLU = desc->useSwiGLU;
      block.ffn.clipClass = clipClass(desc->swigluClip);
      block.ffn.hasInputQuantRange = hasV105Scalars;
      block.ffn.hasProductQuantRange = hasV105Scalars;
      block.ffnScalars.swigluClipBits = floatBits(desc->swigluClip);
      block.ffnScalars.inputQuantMaxAbsBits =
        hasV105Scalars ? floatBits(desc->ffnInputQuantMaxAbs) : 0;
      block.ffnScalars.productQuantMaxAbsBits =
        hasV105Scalars ? floatBits(desc->productQuantMaxAbs) : 0;
      block.officialFfn.descriptor = desc;
      block.officialFfn.executable = executableBlocks[i].second;
    }
    // Ordinary, global-pooling, and nested blocks remain opaque official
    // blocks. In particular, transformers inside a nested block are not
    // flattened into the top-level candidate span.
    view.blocks.push_back(std::move(block));
  }
  return view;
}

}  // namespace FourProfile

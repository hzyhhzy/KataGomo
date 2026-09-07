#ifndef NEURALNET_CUDAB11PROFILE_H_
#define NEURALNET_CUDAB11PROFILE_H_

// Pure dispatch predicates. Deliberately independent of CUDA headers/runtime so
// the profile and synthetic device cases can be tested on CPU-only machines.
#include "desc.h"
#include <climits>
#include <cstddef>
#include <cstring>
#include <vector>

// Structural profile, independent of checkpoint name and weight values. Do not
// inspect weight-vector sizes: callers may retain the descriptor after releasing
// its weights. The model loader remains responsible for validating disk data.
inline bool isOfficialB11C768(const ModelDesc& d) {
  if(d.modelVersion != 17 || d.numInputChannels != 22 || d.numInputGlobalChannels != 19 ||
     d.numInputMetaChannels != 0 || d.trunk.trunkNumChannels != 768 ||
     d.trunk.blocks.size() != 11 || d.trunk.numBlocks != 11)
    return false;
  const auto pointwise = [](const ConvLayerDesc& c) {
    return c.convXSize == 1 && c.convYSize == 1 && c.dilationX == 1 && c.dilationY == 1;
  };
  for(const auto& outer : d.trunk.blocks) {
    if(outer.first != NESTED_BOTTLENECK_BLOCK_KIND || outer.second == nullptr) return false;
    const auto& n = *static_cast<const NestedBottleneckResidualBlockDesc*>(outer.second.get());
    if(n.numBlocks != 6 || n.blocks.size() != 6 ||
       n.preBN.numChannels != 768 || n.postBN.numChannels != 384 ||
       n.preConv.inChannels != 768 || n.preConv.outChannels != 384 ||
       n.postConv.inChannels != 384 || n.postConv.outChannels != 768 ||
       !pointwise(n.preConv) || !pointwise(n.postConv)) return false;
    for(std::size_t i = 0; i < n.blocks.size(); i++) {
      const auto& item = n.blocks[i];
      if(item.second == nullptr) return false;
      if(i % 2 == 0) {
        if(item.first != TRANSFORMER_ATTENTION_BLOCK_KIND) return false;
        const auto& a = *static_cast<const TransformerAttentionDesc*>(item.second.get());
        if(a.numHeads != 12 || a.numKVHeads != 12 || a.qHeadDim != 32 || a.vHeadDim != 32 ||
           !a.useRope || !a.learnableRope || a.ropeNumKVHeads != 12 || a.ropeNumPairs != 16 ||
           a.preLN.numChannels != 384 ||
           a.qProj.inChannels != 384 || a.qProj.outChannels != 384 ||
           a.kProj.inChannels != 384 || a.kProj.outChannels != 384 ||
           a.vProj.inChannels != 384 || a.vProj.outChannels != 384 ||
           a.outProj.inChannels != 384 || a.outProj.outChannels != 384) return false;
      }
      else {
        if(item.first != TRANSFORMER_FFN_BLOCK_KIND) return false;
        const auto& f = *static_cast<const TransformerFFNDesc*>(item.second.get());
        if(f.numChannels != 384 || f.ffnChannels != 1152 || !f.useSwiGLU ||
           f.preLN.numChannels != 384 ||
           f.linear1.inChannels != 384 || f.linear1.outChannels != 1152 ||
           f.linearGate.inChannels != 384 || f.linearGate.outChannels != 1152 ||
           f.linear2.inChannels != 1152 || f.linear2.outChannels != 384) return false;
      }
    }
  }
  return true;
}

inline bool isB11CudaHardwareLayoutEligible(const char* deviceName, int major, int minor,
  bool useFP16, bool useNHWC, int nnXLen, int nnYLen) {
  return deviceName != nullptr && std::strcmp(deviceName,"NVIDIA GeForce RTX 5090") == 0 &&
    major == 12 && minor == 0 && useFP16 && useNHWC && nnXLen == 19 && nnYLen == 19;
}

// The supported set is an explicit caller decision, not an implicit profile
// constant. This permits benchmark screening followed by a narrower final set.
// These predicates do not replace an actualBatch == capacity check when a
// particular launch path requires that invariant.
inline bool isB11SupportedBatch(int batch, int inclusiveMin, int inclusiveMax) {
  return inclusiveMin >= 1 && inclusiveMax >= inclusiveMin &&
    batch >= inclusiveMin && batch <= inclusiveMax;
}

inline bool isB11SupportedBatch(int batch, const int* supported, std::size_t count) {
  if(batch < 1 || supported == nullptr || count == 0) return false;
  for(std::size_t i = 0; i < count; i++)
    if(batch == supported[i]) return true;
  return false;
}

template<std::size_t N>
inline bool isB11SupportedBatch(int batch, const int (&supported)[N]) {
  return isB11SupportedBatch(batch,supported,N);
}

// Kept deliberately narrow: the two strongest measured throughput profiles.
// New batch sizes need numerical and performance validation before adding here.
inline bool isB11OptimizedBatch(int batch) {
  constexpr int supported[] = {13,16};
  return isB11SupportedBatch(batch,supported);
}

// A complete planned mapping, not createComputeContext's deduplicated device
// inventory. Match the CUDA backend's explicit rule that -1 selects GPU 0.
// Zero means unknown/invalid/not used and must never be replaced by a guessed
// stream count. No CUDA calls or inspection of unrelated devices is needed.
inline int getB11ExpectedConcurrentGpuThreads(const std::vector<int>& mapping, int actualDevice) {
  if(actualDevice < 0 || mapping.empty() || mapping.size() > static_cast<std::size_t>(INT_MAX))
    return 0;
  int count = 0;
  for(int requestedDevice : mapping) {
    if(requestedDevice < -1) return 0;
    const int resolvedDevice = requestedDevice == -1 ? 0 : requestedDevice;
    if(resolvedDevice == actualDevice) count++;
  }
  return count;
}

// Only measured per-GPU concurrency profiles enable the automatic cache policy.
// The complete model/device/layout gate and actualBatch == capacity check are
// separate requirements. Deliberately unaffected by research batch expansion.
inline bool isB11AutoL2Eligible(int batch, int expectedConcurrentGpuThreads) {
  return (batch == 13 || batch == 16) && expectedConcurrentGpuThreads == 3;
}

#endif

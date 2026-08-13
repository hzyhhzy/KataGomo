#include "neuralnet/cuda_specialized/sm120/c384/experimental_int8/kernels.h"
#include "neuralnet/cuda_specialized/sm120/c384/experimental_int8/policy.h"
#include "neuralnet/cuda_specialized/sm120/c384/experimental_int8/weights.h"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

using namespace C384Int8Experiment;

int8_t quantizeNoNeg128(float value, float clip, float scale) {
  const float bounded = std::max(-clip,std::min(clip,value));
  const long rounded = std::lrint(bounded / scale);
  const long saturated = std::max(-127L,std::min(127L,rounded));
  return static_cast<int8_t>(saturated);
}

void require(bool condition, const char* message) {
  if(!condition)
    throw std::runtime_error(message);
}

}  // namespace

int main() {
  try {
    static_assert(kBatch == 28,"prototype batch contract changed");
    static_assert(kSequence == 225,"prototype board contract changed");
    static_assert(kTokenRows == 6300,"prototype M contract changed");
    static_assert(kChannels == 384,"prototype C contract changed");
    static_assert(kQkChannels == 768,"prototype QK contract changed");
    static_assert(kQkvChannels == 1152,"prototype QKV contract changed");
    static_assert(kFfnChannels == 1024,"prototype FFN contract changed");
    static_assert(int(ProjectionMode::ConservativeQk) == 1,"mode ABI changed");
    static_assert(int(ProjectionMode::AggressiveQkv) == 2,"mode ABI changed");

    require(std::fesetround(FE_TONEAREST) == 0,"cannot select round-to-nearest-even");
    const std::array<float,13> normValues = {
      -std::numeric_limits<float>::infinity(), -4.1f, -4.0f, -2.0f,
      -0.5f * kNormActivationScale, -0.0f, 0.0f,
      0.5f * kNormActivationScale, 2.0f, 4.0f, 4.1f,
      std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()
    };
    for(const float value : normValues) {
      if(std::isnan(value))
        continue; // GPU finite-input contract rejects NaNs before quantization.
      const int q = int(quantizeNoNeg128(value,kNormActivationClip,kNormActivationScale));
      require(q >= -127 && q <= 127,"norm quantizer emitted -128 or overflow");
    }
    require(int(quantizeNoNeg128(-100.0f,kNormActivationClip,kNormActivationScale)) == -127,
            "negative norm saturation changed");
    require(int(quantizeNoNeg128(100.0f,kNormActivationClip,kNormActivationScale)) == 127,
            "positive norm saturation changed");
    require(int(quantizeNoNeg128(0.5f * kNormActivationScale,
                                 kNormActivationClip,kNormActivationScale)) == 0,
            "ties-to-even quantization changed");

    for(int i = -1000; i <= 1000; i++) {
      const float value = 0.073f * float(i);
      const int q = int(quantizeNoNeg128(value,kClip7ProductClip,kClip7ProductScale));
      require(q >= -127 && q <= 127,"clip7-product quantizer emitted -128 or overflow");
      require(std::isfinite(float(q) * kClip7ProductScale),"product dequant is not finite");
    }
    require(int(quantizeNoNeg128(-49.0f,kClip7ProductClip,kClip7ProductScale)) == -127,
            "negative product endpoint changed");
    require(int(quantizeNoNeg128(49.0f,kClip7ProductClip,kClip7ProductScale)) == 127,
            "positive product endpoint changed");

    require(parseEngineMode(nullptr) == EngineMode::Off,
            "unset C384 INT8 mode must default off");
    require(parseEngineMode("off") == EngineMode::Off,"off mode parsing changed");
    require(parseEngineMode("conservative") == EngineMode::Conservative,
            "conservative mode parsing changed");
    require(parseEngineMode("aggressive") == EngineMode::Aggressive,
            "aggressive mode parsing changed");
    require(parseEngineMode("yes") == EngineMode::Invalid,
            "unknown C384 INT8 mode did not fail closed");
    EngineEligibility eligible;
    eligible.modelVersion = 105;
    eligible.batchSize = 28;
    eligible.boardX = 15;
    eligible.boardY = 15;
    eligible.modelDepth = 36;
    eligible.attentionBlocks = 36;
    eligible.ffnBlocks = 36;
    eligible.channels = 384;
    eligible.heads = 12;
    eligible.headDim = 32;
    eligible.ffnChannels = 1024;
    eligible.computeCapability = 120;
    eligible.alternatingAttentionFfn = true;
    eligible.usingFp16 = true;
    eligible.usingNhwc = true;
    eligible.exactNoMask = true;
    eligible.learnedRope = true;
    eligible.qkNorm = true;
    eligible.swigluClipBits = kClip7Bits;
    require(engineShapeEligible(eligible),"exact v105 C384 engine shape was rejected");
    eligible.modelDepth = 35;
    require(!engineShapeEligible(eligible),"non-b36 engine shape was accepted");
    eligible.modelDepth = 36;
    eligible.qkNorm = false;
    require(!engineShapeEligible(eligible),"non-QKN engine shape was accepted");

    std::vector<float> q(std::size_t(kChannels) * kChannels,0.0f);
    std::vector<float> k(q.size(),0.0f);
    std::vector<float> v(q.size(),0.0f);
    q[std::size_t(3) * kChannels + 5] = 1.0f;
    k[std::size_t(7) * kChannels + 11] = -2.0f;
    v[std::size_t(13) * kChannels + 17] = 4.0f;
    const PackedWeights conservative = packProjection(
      q,k,v,EngineMode::Conservative);
    const PackedWeights aggressive = packProjection(
      q,k,v,EngineMode::Aggressive);
    require(conservative.inputChannels == 384 &&
            conservative.outputChannels == 768 &&
            conservative.values.size() == std::size_t(384) * 768,
            "conservative packed QK shape changed");
    require(aggressive.inputChannels == 384 &&
            aggressive.outputChannels == 1152 &&
            aggressive.values.size() == std::size_t(384) * 1152,
            "aggressive packed QKV shape changed");
    require(conservative.values[std::size_t(5) * 384 + 3] != 0 &&
            conservative.values[std::size_t(384 + 11) * 384 + 7] != 0,
            "QK output-major packing changed");
    require(aggressive.values[std::size_t(2 * 384 + 17) * 384 + 13] == 127,
            "QKV shared-scale V endpoint changed");
    require(std::fabs(conservative.scale - 2.0f / 127.0f) < 1.0e-8f &&
            std::fabs(aggressive.scale - 4.0f / 127.0f) < 1.0e-8f,
            "projection common scale changed");

    std::cout << "KATAGO_C384_INT8_CPU_CONTRACT_PASS"
              << " M=" << kTokenRows
              << " C=" << kChannels
              << " QKV=" << kQkvChannels
              << " F=" << kFfnChannels
              << " norm_scale=" << kNormActivationScale
              << " product_scale=" << kClip7ProductScale
              << " engine_default=off"
              << '\n';
    return 0;
  }
  catch(const std::exception& error) {
    std::cerr << "KATAGO_C384_INT8_CPU_CONTRACT_FAIL: " << error.what() << '\n';
    return 1;
  }
}

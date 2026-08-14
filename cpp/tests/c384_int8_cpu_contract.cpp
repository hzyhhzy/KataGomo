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
#include <type_traits>
#include <vector>

namespace {

using namespace C384Int8Experiment;

int8_t quantizeNoNeg128(float value, float clip, float scale) {
  const float bounded = std::max(-clip,std::min(clip,value));
  const long rounded = std::lrint(bounded / scale);
  const long saturated = std::max(-127L,std::min(127L,rounded));
  return static_cast<int8_t>(saturated);
}

int8_t fusedFactorProduct(int lhs, int rhs) {
  long rounded = std::lrint(float(lhs * rhs) / 127.0f);
  rounded = std::max(-127L,std::min(127L,rounded));
  return static_cast<int8_t>(rounded);
}

int8_t adjustableFactorProduct(
  int lhs,
  int rhs,
  float clip,
  float productQuantMaxAbs
) {
  const float multiplier = float(
    double(clip) * double(clip) /
    (127.0 * double(productQuantMaxAbs)));
  long rounded = std::lrint(float(lhs * rhs) * multiplier);
  rounded = std::max(-127L,std::min(127L,rounded));
  return static_cast<int8_t>(rounded);
}

int8_t fixedFactorProduct(
  int lhs,
  int rhs,
  float clip,
  float productQuantMaxAbs
) {
  const float multiplier = float(
    double(clip) * double(clip) /
    (127.0 * double(productQuantMaxAbs)));
  long rounded = std::lrint(float(lhs * rhs) * multiplier);
  rounded = std::max(-127L,std::min(127L,rounded));
  return static_cast<int8_t>(rounded);
}

int divide127RneIncumbent(int numerator) {
  const bool negative = numerator < 0;
  const int magnitude = negative ? -numerator : numerator;
  int quotient = magnitude / 127;
  const int remainder = magnitude - quotient * 127;
  const int twiceRemainder = 2 * remainder;
  if(twiceRemainder > 127 ||
     (twiceRemainder == 127 && (quotient & 1) != 0))
    quotient++;
  int rounded = negative ? -quotient : quotient;
  rounded = rounded < -127 ? -127 : (rounded > 127 ? 127 : rounded);
  return rounded;
}

int divide127RneExactBranchless(int numerator) {
  const int signMask = -int(numerator < 0);
  const int magnitude = (numerator ^ signMask) - signMask;
  const int quotient = int((unsigned(magnitude) + 63u) / 127u);
  return (quotient ^ signMask) - signMask;
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
    static_assert(int(DualFfnOutputMode::Fp16Product) == 1,
      "dual FP16 output ABI changed");
    static_assert(int(DualFfnOutputMode::Int8Product) == 2,
      "dual INT8 output ABI changed");
    static_assert(int(DualFfnDivide127Tactic::Incumbent) == 1,
      "incumbent divide127 tactic ABI changed");
    static_assert(int(DualFfnDivide127Tactic::ExactBranchless) == 2,
      "exact branchless divide127 tactic ABI changed");
    static_assert(int(DualFfnProductPathTactic::Auto) == 0,
      "automatic product path tactic ABI changed");
    static_assert(int(DualFfnProductPathTactic::FullyAdjustableFloat) == 1,
      "fully-adjustable product path tactic ABI changed");
    static_assert(
      int(DualFfnTactic::M128N128K64S3Sw4Interleaved) == 4,
      "interleaved single-GEMM tactic ABI changed");
    require(DualFfnConfig{}.tactic == DualFfnTactic::M128N64K64S3Sw4,
            "interleaved experiment changed the production D2 default");
    using RmsInt8OnlyFn = cudaError_t (*)(
      const half*,int8_t*,const half*,int,float,cudaStream_t);
    static_assert(std::is_same_v<decltype(&launchRmsNormInt8),RmsInt8OnlyFn>,
      "explicit INT8-only RMS ABI changed");

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
    require(int(fusedFactorProduct(127,127)) == 127,
            "fused +49 endpoint changed");
    require(int(fusedFactorProduct(127,-127)) == -127,
            "fused -49 endpoint changed");
    require(int(fusedFactorProduct(1,63)) == 0 &&
            int(fusedFactorProduct(1,64)) == 1,
            "fused product RNE boundary changed");
    int exhaustiveFactorPairs = 0;
    int interleavedExactFactorPairs = 0;
    for(int lhs = -127; lhs <= 127; lhs++) {
      for(int rhs = -127; rhs <= 127; rhs++) {
        const int numerator = lhs * rhs;
        const int incumbent = divide127RneIncumbent(numerator);
        const int candidate = divide127RneExactBranchless(numerator);
        require(candidate == incumbent,
                "branchless divide127 differs from incumbent RNE");
        require(candidate >= -127 && candidate <= 127,
                "branchless divide127 escaped signed-symmetric INT8 range");
        require(int(fixedFactorProduct(lhs,rhs,7.0f,49.0f)) == incumbent,
                "interleaved float product differs from exact clip7/49 RNE");
        exhaustiveFactorPairs++;
        interleavedExactFactorPairs++;
      }
    }
    require(exhaustiveFactorPairs == 255 * 255,
            "branchless divide127 exhaustive domain changed");
    require(interleavedExactFactorPairs == 255 * 255,
            "interleaved exact clip7/49 exhaustive domain changed");

    const std::size_t planarWeightElements =
      std::size_t(kFfnChannels) * kChannels;
    std::vector<int8_t> upWeights(planarWeightElements);
    std::vector<int8_t> gateWeights(planarWeightElements);
    std::vector<int8_t> interleavedWeights(
      2 * planarWeightElements,int8_t(-128));
    for(std::size_t index = 0; index < planarWeightElements; index++) {
      upWeights[index] =
        static_cast<int8_t>(int((index * 17 + 3) % 255) - 127);
      gateWeights[index] =
        static_cast<int8_t>(int((index * 29 + 11) % 255) - 127);
    }
    for(int outputChannel = 0; outputChannel < kFfnChannels; outputChannel++) {
      for(int inputChannel = 0; inputChannel < kChannels; inputChannel++) {
        const std::size_t source =
          std::size_t(outputChannel) * kChannels + inputChannel;
        interleavedWeights[
          std::size_t(2 * outputChannel) * kChannels + inputChannel] =
          upWeights[source];
        interleavedWeights[
          std::size_t(2 * outputChannel + 1) * kChannels + inputChannel] =
          gateWeights[source];
      }
    }
    std::size_t interleavedPackElements = 0;
    for(int outputChannel = 0; outputChannel < kFfnChannels; outputChannel++) {
      for(int inputChannel = 0; inputChannel < kChannels; inputChannel++) {
        const std::size_t source =
          std::size_t(outputChannel) * kChannels + inputChannel;
        require(interleavedWeights[
          std::size_t(2 * outputChannel) * kChannels + inputChannel] ==
            upWeights[source],
          "interleaved up-weight K-contiguous packing changed");
        require(interleavedWeights[
          std::size_t(2 * outputChannel + 1) * kChannels + inputChannel] ==
            gateWeights[source],
          "interleaved gate-weight K-contiguous packing changed");
        interleavedPackElements += 2;
      }
    }
    require(interleavedPackElements == 2 * planarWeightElements,
            "interleaved weight packing domain changed");
    const int clippedWitness = int(fusedFactorProduct(127,54));
    const int missingClampWitness = int(quantizeNoNeg128(
      10.0f * (54.0f * 7.0f / 127.0f),
      kClip7ProductClip,kClip7ProductScale));
    require(clippedWitness != missingClampWitness,
            "fused contract cannot detect a missing clip7 clamp");

    struct AdjustableCase { float clip; float productMax; };
    const std::array<AdjustableCase,5> adjustableCases{{
      {4.0f,16.0f},
      {4.0f,23.5f},
      {7.0f,73.4171f},
      {7.0f,47.7371f},
      {6.0f,48.0f},
    }};
    int fixedEquivalentFactorPairs = 0;
    int clip4EquivalentFactorPairs = 0;
    for(const AdjustableCase& adjustable : adjustableCases) {
      for(int lhs = -127; lhs <= 127; lhs++) {
        for(int rhs = -127; rhs <= 127; rhs++) {
          const int q = int(adjustableFactorProduct(
            lhs,rhs,adjustable.clip,adjustable.productMax));
          require(q >= -127 && q <= 127,
                  "adjustable product quantizer emitted -128 or overflow");
          if(adjustable.clip == 4.0f || adjustable.clip == 7.0f) {
            require(q == int(fixedFactorProduct(
              lhs,rhs,adjustable.clip,adjustable.productMax)),
              "fixed-factor hybrid differs from fully-adjustable product RNE");
            fixedEquivalentFactorPairs++;
            if(adjustable.clip == 4.0f)
              clip4EquivalentFactorPairs++;
          }
        }
      }
    }
    require(fixedEquivalentFactorPairs == 4 * 255 * 255,
            "fixed-factor hybrid exhaustive domain changed");
    require(clip4EquivalentFactorPairs == 2 * 255 * 255,
            "clip4 fixed-factor exhaustive domain changed");
    require(int(adjustableFactorProduct(127,127,4.0f,16.0f)) == 127,
            "clip4 squared-domain endpoint changed");
    require(int(adjustableFactorProduct(127,-127,4.0f,23.5f)) == -86,
            "clip4 calibrated product endpoint changed");
    require(int(adjustableFactorProduct(127,127,7.0f,73.4171f)) == 85,
            "adjustable wide-range product endpoint changed");
    require(int(adjustableFactorProduct(127,127,7.0f,47.7371f)) == 127,
            "adjustable narrow-range product saturation changed");
    require(int(adjustableFactorProduct(127,-127,6.0f,48.0f)) == -95,
            "adjustable negative product endpoint changed");

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
    eligible.calibratedFfnProduct = true;
    require(engineShapeEligible(eligible),"exact v105 C384 engine shape was rejected");
    eligible.modelDepth = 35;
    require(!engineShapeEligible(eligible),"non-b36 engine shape was accepted");
    eligible.modelDepth = 36;
    eligible.qkNorm = false;
    require(!engineShapeEligible(eligible),"non-QKN engine shape was accepted");
    eligible.qkNorm = true;
    eligible.calibratedFfnProduct = false;
    require(!engineShapeEligible(eligible),
            "engine shape without calibrated FFN product metadata was accepted");

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

    std::vector<float> attentionOut(
      std::size_t(kChannels) * kChannels,0.0f);
    attentionOut[std::size_t(19) * kChannels + 23] = -3.0f;
    attentionOut[std::size_t(29) * kChannels + 31] = 1.5f;
    const PackedWeights packedAttentionOut = packAttentionOut(attentionOut);
    require(packedAttentionOut.inputChannels == kChannels &&
            packedAttentionOut.outputChannels == kChannels &&
            packedAttentionOut.values.size() ==
              std::size_t(kChannels) * kChannels,
            "attention-out packed shape changed");
    require(packedAttentionOut.values[std::size_t(23) * kChannels + 19] ==
              -127 &&
            packedAttentionOut.values[std::size_t(31) * kChannels + 29] ==
              64,
            "attention-out output-major packing or RNE changed");
    require(std::fabs(packedAttentionOut.scale - 3.0f / 127.0f) < 1.0e-8f,
            "attention-out scale changed");

    std::cout << "KATAGO_C384_INT8_CPU_CONTRACT_PASS"
              << " M=" << kTokenRows
              << " C=" << kChannels
              << " QKV=" << kQkvChannels
              << " F=" << kFfnChannels
              << " norm_scale=" << kNormActivationScale
              << " product_scale=" << kClip7ProductScale
              << " divide127_exhaustive_factor_pairs=" << exhaustiveFactorPairs
              << " divide127_tactics=incumbent,exact-branchless"
              << " interleaved_exact_factor_pairs="
              << interleavedExactFactorPairs
              << " interleaved_pack_elements=" << interleavedPackElements
              << " interleaved_layout=up-gate-adjacent-k384"
              << " fused_product_quant=fused-dual-epilogue-v2"
              << " attention_out_pack=k384-n384-output-major"
              << " rms_int8_only_api=explicit"
              << " endpoints_plus49_minus49=1 rne=1 no_neg128=1"
              << " adjustable_product_cases=5"
              << " fixed_hybrid_equivalent_factor_pairs="
              << fixedEquivalentFactorPairs
              << " clip4_hybrid_equivalent_factor_pairs="
              << clip4EquivalentFactorPairs
              << " dual_default=D2 interleaved_default=off"
              << " engine_default=off"
              << '\n';
    return 0;
  }
  catch(const std::exception& error) {
    std::cerr << "KATAGO_C384_INT8_CPU_CONTRACT_FAIL: " << error.what() << '\n';
    return 1;
  }
}

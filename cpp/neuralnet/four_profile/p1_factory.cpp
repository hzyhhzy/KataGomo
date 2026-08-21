#include "p1_factory.h"

#include <memory>

#if defined(KATAGO_ENABLE_P1_SM120_PROVIDER) && KATAGO_ENABLE_P1_SM120_PROVIDER
#include "p1_provider.h"
#endif

namespace FourProfile {
namespace {

constexpr const char* P1_PROFILE_ID = "P1-c256-h8-s225-fp16-b36-s2";

bool matchesP1(const ProfileKeyV1& key) {
  const RuntimeKeyV1& runtime = key.runtime;
  const AttentionSpecV1& attention = key.attention;
  const FfnSpecV1& ffn = key.ffn;
  return key.modelVersion == 102 &&
    runtime.deviceComputeCapability == 120 &&
    runtime.boardX == 15 && runtime.boardY == 15 &&
    runtime.physicalBatchSize == 36 &&
    runtime.sameGpuConcurrency == 2 &&
    runtime.exactBoard &&
    runtime.maskMode == MaskModeV1::None && runtime.maskNull &&
    runtime.inputStorage == StorageTypeV1::Fp16 &&
    runtime.outputStorage == StorageTypeV1::Fp16 &&
    runtime.requestedExecution == RequestedExecutionV1::Fp16 &&
    runtime.layout == TensorLayoutV1::Nhwc &&
    attention.channels == 256 &&
    attention.numHeads == 8 && attention.numKVHeads == 8 &&
    attention.qHeadDim == 32 && attention.vHeadDim == 32 &&
    attention.useRope && attention.learnableRope &&
    !attention.useQKNorm &&
    !attention.hasInputQuantRange && !attention.hasOutputQuantRange &&
    ffn.channels == 256 && ffn.hiddenChannels == 768 &&
    ffn.useSwiGLU && ffn.clipClass == ClipClassV1::Zero &&
    !ffn.hasInputQuantRange && !ffn.hasProductQuantRange;
}

class P1FactoryV1 final : public FactoryV1 {
public:
  const char* factoryId() const noexcept override { return P1_PROFILE_ID; }
  bool matches(const ProfileKeyV1& key) const override { return matchesP1(key); }

  AvailabilityResultV1 availability(const ProfileKeyV1&) const override {
    AvailabilityResultV1 result;
#if defined(KATAGO_ENABLE_P1_SM120_PROVIDER) && KATAGO_ENABLE_P1_SM120_PROVIDER
    result.availability = AvailabilityV1::Available;
    result.detail = "qualified SM120 P1 kernels and SHA-locked B36 FA4 package are linked";
#else
    result.availability = AvailabilityV1::Unavailable;
    result.detail =
      "P1 profile matched, but the qualified SM120/CUTLASS/FA4 provider is not linked";
#endif
    return result;
  }

  std::unique_ptr<ProviderV1> create(const ProfileKeyV1& key) const override {
#if defined(KATAGO_ENABLE_P1_SM120_PROVIDER) && KATAGO_ENABLE_P1_SM120_PROVIDER
    return createP1ProviderV1(key);
#else
    (void)key;
    return nullptr;
#endif
  }
};

}  // namespace

void registerP1FactoryV1(RegistryV1& registry) {
  registry.add(std::make_unique<P1FactoryV1>());
}

}  // namespace FourProfile

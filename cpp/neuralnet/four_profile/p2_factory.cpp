#include "p2_factory.h"
#include "p2_build_config.h"

#include <memory>

#if defined(KATAGO_ENABLE_P2_SM120_PROVIDER) && KATAGO_ENABLE_P2_SM120_PROVIDER
#include "p1_provider.h"
#endif

namespace FourProfile {
namespace {

constexpr const char* P2_PROFILE_ID = KATAGO_P2_PROFILE_ID;

bool matchesP2(const ProfileKeyV1& key) {
  const RuntimeKeyV1& runtime = key.runtime;
  const AttentionSpecV1& attention = key.attention;
  const FfnSpecV1& ffn = key.ffn;
  return key.modelVersion == 102 &&
    runtime.deviceComputeCapability == 120 &&
    runtime.boardX == 19 && runtime.boardY == 19 &&
    runtime.physicalBatchSize == KATAGO_P2_FIXED_BATCH &&
    runtime.sameGpuConcurrency == KATAGO_P2_FIXED_CONCURRENCY && runtime.exactBoard &&
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

class P2FactoryV1 final : public FactoryV1 {
public:
  const char* factoryId() const noexcept override { return P2_PROFILE_ID; }
  bool matches(const ProfileKeyV1& key) const override { return matchesP2(key); }

  AvailabilityResultV1 availability(const ProfileKeyV1&) const override {
    AvailabilityResultV1 result;
#if defined(KATAGO_ENABLE_P2_SM120_PROVIDER) && KATAGO_ENABLE_P2_SM120_PROVIDER
    result.availability = AvailabilityV1::Available;
    result.detail = "SM120 C256/S361 kernels and selected fixed-batch FA4 package are linked";
#else
    result.availability = AvailabilityV1::Unavailable;
    result.detail = "P2 profile matched, but its SM120 provider is not linked";
#endif
    return result;
  }

  std::unique_ptr<ProviderV1> create(const ProfileKeyV1& key) const override {
#if defined(KATAGO_ENABLE_P2_SM120_PROVIDER) && KATAGO_ENABLE_P2_SM120_PROVIDER
    return createP2ProviderV1(key);
#else
    (void)key;
    return nullptr;
#endif
  }
};

}  // namespace

void registerP2FactoryV1(RegistryV1& registry) {
  registry.add(std::make_unique<P2FactoryV1>());
}

}  // namespace FourProfile

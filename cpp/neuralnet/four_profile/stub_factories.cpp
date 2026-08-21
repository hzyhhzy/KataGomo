#include "stub_factories.h"
#include "p1_factory.h"

#include <memory>

namespace FourProfile {
namespace {

bool runtimeMatches(
  const RuntimeKeyV1& runtime,
  int board,
  int batch,
  RequestedExecutionV1 requestedExecution
) {
  return runtime.deviceComputeCapability == 120 &&
    runtime.boardX == board &&
    runtime.boardY == board &&
    runtime.physicalBatchSize == batch &&
    runtime.sameGpuConcurrency == 2 &&
    runtime.exactBoard &&
    runtime.maskMode == MaskModeV1::None &&
    runtime.maskNull &&
    runtime.inputStorage == StorageTypeV1::Fp16 &&
    runtime.outputStorage == StorageTypeV1::Fp16 &&
    runtime.requestedExecution == requestedExecution &&
    runtime.layout == TensorLayoutV1::Nhwc;
}

bool attentionMatches(
  const AttentionSpecV1& attention,
  int channels,
  int heads,
  bool qkn,
  bool quantRanges
) {
  return attention.channels == channels &&
    attention.numHeads == heads &&
    attention.numKVHeads == heads &&
    attention.qHeadDim == 32 &&
    attention.vHeadDim == 32 &&
    attention.useRope &&
    attention.learnableRope &&
    attention.useQKNorm == qkn &&
    attention.hasInputQuantRange == quantRanges &&
    attention.hasOutputQuantRange == quantRanges;
}

bool ffnMatches(
  const FfnSpecV1& ffn,
  int channels,
  int hiddenChannels,
  bool clipped,
  bool quantRanges
) {
  return ffn.channels == channels &&
    ffn.hiddenChannels == hiddenChannels &&
    ffn.useSwiGLU &&
    ffn.clipClass == (clipped ? ClipClassV1::PositiveFinite : ClipClassV1::Zero) &&
    ffn.hasInputQuantRange == quantRanges &&
    ffn.hasProductQuantRange == quantRanges;
}

class StubFactoryBaseV1 : public FactoryV1 {
public:
  StubFactoryBaseV1(const char* id_, AvailabilityV1 availability_)
    : id(id_), stubAvailability(availability_) {}

  const char* factoryId() const noexcept override { return id; }

  AvailabilityResultV1 availability(const ProfileKeyV1&) const override {
    AvailabilityResultV1 result;
    result.availability = stubAvailability;
    result.detail = stubAvailability == AvailabilityV1::Uncertified ?
      "profile factory is registered but its specialized implementation is uncertified" :
      "profile factory is registered but specialized kernels are not linked";
    return result;
  }

  std::unique_ptr<ProviderV1> create(const ProfileKeyV1&) const override {
    return nullptr;
  }

private:
  const char* id;
  AvailabilityV1 stubAvailability;
};

class P2StubFactoryV1 final : public StubFactoryBaseV1 {
public:
  P2StubFactoryV1()
    : StubFactoryBaseV1("P2-c256-h8-s361-fp16-b28-s2",AvailabilityV1::Uncertified) {}

  bool matches(const ProfileKeyV1& key) const override {
    return key.modelVersion == 102 &&
      runtimeMatches(key.runtime,19,28,RequestedExecutionV1::Fp16) &&
      attentionMatches(key.attention,256,8,false,false) &&
      ffnMatches(key.ffn,256,768,false,false);
  }
};

class P3StubFactoryV1 final : public StubFactoryBaseV1 {
public:
  P3StubFactoryV1()
    : StubFactoryBaseV1("P3-c384-h12-s225-qkn-positive-clip-fp16-b28-s2",AvailabilityV1::Unavailable) {}

  bool matches(const ProfileKeyV1& key) const override {
    return key.modelVersion == 105 &&
      runtimeMatches(key.runtime,15,28,RequestedExecutionV1::Fp16) &&
      attentionMatches(key.attention,384,12,true,true) &&
      ffnMatches(key.ffn,384,1024,true,true);
  }
};

class P4StubFactoryV1 final : public StubFactoryBaseV1 {
public:
  P4StubFactoryV1()
    : StubFactoryBaseV1("P4-c384-h12-s225-qkn-positive-clip-int8-b28-s2",AvailabilityV1::Unavailable) {}

  bool matches(const ProfileKeyV1& key) const override {
    return key.modelVersion == 105 &&
      runtimeMatches(key.runtime,15,28,RequestedExecutionV1::Int8) &&
      attentionMatches(key.attention,384,12,true,true) &&
      ffnMatches(key.ffn,384,1024,true,true);
  }
};

}  // namespace

void registerBuiltinStubFactoriesV1(
  RegistryV1& registry,
  std::unique_ptr<FactoryV1> p3Override
) {
  registerP1FactoryV1(registry);
  registry.add(std::make_unique<P2StubFactoryV1>());
  if(p3Override != nullptr)
    registry.add(std::move(p3Override));
  else
    registry.add(std::make_unique<P3StubFactoryV1>());
  registry.add(std::make_unique<P4StubFactoryV1>());
}

}  // namespace FourProfile

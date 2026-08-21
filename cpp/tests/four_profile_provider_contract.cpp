#include "../neuralnet/four_profile/manager.h"
#include "../neuralnet/four_profile/mode.h"
#include "../neuralnet/four_profile/runtime_bridge.h"
#include "../neuralnet/four_profile/stub_factories.h"

#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace FourProfile;

namespace {

constexpr uint64_t FAKE_PLAN_GENERATION = 17;
constexpr uint64_t FAKE_RUN_TOKEN = 23;

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error("four-profile provider contract: " + message);
}

void require(bool condition, const std::string& message) {
  if(!condition)
    fail(message);
}

template<typename Exception, typename Func>
void expectException(Func&& func, const std::string& context) {
  try {
    func();
  }
  catch(const Exception&) {
    return;
  }
  fail(context + " did not throw the required exception type");
}

uint32_t floatBits(float value) {
  uint32_t bits;
  static_assert(sizeof(bits) == sizeof(value),"32-bit float required");
  std::memcpy(&bits,&value,sizeof(bits));
  return bits;
}

AttentionSpecV1 attentionSpec(
  int channels = 64,
  int heads = 2,
  bool qkn = false,
  bool quantRanges = false
) {
  AttentionSpecV1 spec;
  spec.channels = channels;
  spec.numHeads = heads;
  spec.numKVHeads = heads;
  spec.qHeadDim = 32;
  spec.vHeadDim = 32;
  spec.useRope = true;
  spec.learnableRope = true;
  spec.useQKNorm = qkn;
  spec.hasInputQuantRange = quantRanges;
  spec.hasOutputQuantRange = quantRanges;
  return spec;
}

FfnSpecV1 ffnSpec(
  int channels = 64,
  int hidden = 192,
  bool clipped = false,
  bool quantRanges = false
) {
  FfnSpecV1 spec;
  spec.channels = channels;
  spec.hiddenChannels = hidden;
  spec.useSwiGLU = true;
  spec.clipClass = clipped ? ClipClassV1::PositiveFinite : ClipClassV1::Zero;
  spec.hasInputQuantRange = quantRanges;
  spec.hasProductQuantRange = quantRanges;
  return spec;
}

RuntimeKeyV1 runtimeKey(
  int board = 15,
  int batch = 7,
  RequestedExecutionV1 requestedExecution = RequestedExecutionV1::Fp16
) {
  RuntimeKeyV1 runtime;
  runtime.deviceComputeCapability = 120;
  runtime.boardX = board;
  runtime.boardY = board;
  runtime.physicalBatchSize = batch;
  runtime.sameGpuConcurrency = 2;
  runtime.exactBoard = true;
  runtime.maskMode = MaskModeV1::None;
  runtime.maskNull = true;
  runtime.inputStorage = StorageTypeV1::Fp16;
  runtime.outputStorage = StorageTypeV1::Fp16;
  runtime.requestedExecution = requestedExecution;
  runtime.layout = TensorLayoutV1::Nhwc;
  return runtime;
}

BlockViewV1 otherBlock() {
  return BlockViewV1();
}

BlockViewV1 attentionBlock(const AttentionSpecV1& spec) {
  static int descriptorSentinel;
  static int executableSentinel;
  BlockViewV1 block;
  block.kind = BlockKindV1::TransformerAttention;
  block.attention = spec;
  block.attentionScalars.inputQuantMaxAbsBits =
    spec.hasInputQuantRange ? floatBits(3.25f) : 0;
  block.attentionScalars.outputQuantMaxAbsBits =
    spec.hasOutputQuantRange ? floatBits(5.5f) : 0;
  block.officialAttention.descriptor = &descriptorSentinel;
  block.officialAttention.executable = &executableSentinel;
  return block;
}

BlockViewV1 ffnBlock(const FfnSpecV1& spec) {
  static int descriptorSentinel;
  static int executableSentinel;
  BlockViewV1 block;
  block.kind = BlockKindV1::TransformerFfn;
  block.ffn = spec;
  block.ffnScalars.swigluClipBits =
    spec.clipClass == ClipClassV1::PositiveFinite ? floatBits(7.0f) : 0;
  block.ffnScalars.inputQuantMaxAbsBits =
    spec.hasInputQuantRange ? floatBits(2.75f) : 0;
  block.ffnScalars.productQuantMaxAbsBits =
    spec.hasProductQuantRange ? floatBits(23.5f) : 0;
  block.officialFfn.descriptor = &descriptorSentinel;
  block.officialFfn.executable = &executableSentinel;
  return block;
}

ModelViewV1 model(
  int depth,
  const AttentionSpecV1& attention,
  const FfnSpecV1& ffn,
  int prefixOther = 0,
  int suffixOther = 0,
  int modelVersion = 105
) {
  ModelViewV1 result;
  result.modelVersion = modelVersion;
  for(int i = 0; i < prefixOther; i++)
    result.blocks.push_back(otherBlock());
  for(int i = 0; i < depth; i++) {
    result.blocks.push_back(attentionBlock(attention));
    result.blocks.push_back(ffnBlock(ffn));
  }
  for(int i = 0; i < suffixOther; i++)
    result.blocks.push_back(otherBlock());
  return result;
}

ProfileKeyV1 keyFor(const ModelViewV1& modelView, const RuntimeKeyV1& runtime) {
  const TransformerSpanV1 span = analyzeTransformerSpanV1(modelView.blocks);
  require(span.valid,"keyFor received invalid model span");
  ProfileKeyV1 key;
  std::string detail;
  require(deriveProfileKeyV1(modelView,span,runtime,key,detail),"keyFor received nonuniform model");
  return key;
}

RuntimeCallV1 runtimeCall(
  const RuntimeKeyV1& runtime,
  const TransformerSpanV1& span
) {
  static int streamSentinel;
  static int trunkSentinel;
  static int trunkScratchSentinel;
  static int maskSentinel;
  static int workspaceSentinel;
  RuntimeCallV1 call;
  call.key = runtime;
  call.actualBatchSize = runtime.physicalBatchSize;
  call.sequenceSize = runtime.boardX * runtime.boardY;
  call.transformerBeginBlock = span.beginBlock;
  call.transformerPairCount = span.pairCount;
  call.stream = &streamSentinel;
  call.trunk = &trunkSentinel;
  call.trunkScratch = &trunkScratchSentinel;
  call.mask = runtime.maskNull ? nullptr : &maskSentinel;
  call.workspace = &workspaceSentinel;
  call.workspaceBytes = 4096;
  return call;
}

struct FakeStats {
  int creates = 0;
  int attentionPrepareCalls = 0;
  int ffnPrepareCalls = 0;
  int commits = 0;
  int committedAttention = 0;
  int committedFfn = 0;
  int preflights = 0;
  int enqueues = 0;
  uint64_t lastEnqueueToken = 0;
  int livePrepared = 0;
  std::vector<std::string> events;
  std::vector<uint32_t> attentionInputRanges;
  std::vector<uint32_t> attentionOutputRanges;
  std::vector<uint32_t> ffnClips;
  std::vector<uint32_t> ffnInputRanges;
  std::vector<uint32_t> productRanges;
};

struct FakeConfig {
  AvailabilityV1 availability = AvailabilityV1::Available;
  size_t failAttentionLayer = std::numeric_limits<size_t>::max();
  size_t failFfnLayer = std::numeric_limits<size_t>::max();
  bool commitOk = true;
  bool allowSmallerActualBatch = false;
  bool preflightThrows = false;
  bool enqueueThrows = false;
  ProviderOpResultV1 preflightResult =
    ProviderOpResultV1::success(0,FAKE_PLAN_GENERATION,FAKE_RUN_TOKEN);
  ProviderOpResultV1 enqueueResult =
    ProviderOpResultV1::success(1,FAKE_PLAN_GENERATION,FAKE_RUN_TOKEN);
};

class FakePreparedAttention final : public PreparedAttentionV1 {
public:
  explicit FakePreparedAttention(const std::shared_ptr<FakeStats>& stats_) : stats(stats_) {
    stats->livePrepared++;
  }
  ~FakePreparedAttention() override { stats->livePrepared--; }
private:
  std::shared_ptr<FakeStats> stats;
};

class FakePreparedFfn final : public PreparedFfnV1 {
public:
  explicit FakePreparedFfn(const std::shared_ptr<FakeStats>& stats_) : stats(stats_) {
    stats->livePrepared++;
  }
  ~FakePreparedFfn() override { stats->livePrepared--; }
private:
  std::shared_ptr<FakeStats> stats;
};

class FakeProvider final : public ProviderV1 {
public:
  FakeProvider(
    std::string id_,
    const std::shared_ptr<FakeStats>& stats_,
    const std::shared_ptr<FakeConfig>& config_
  ) : id(std::move(id_)), stats(stats_), config(config_) {}

  const char* profileId() const noexcept override { return id.c_str(); }

  PrepareAttentionResultV1 prepareAttention(
    size_t layer,
    const AttentionSpecV1&,
    const AttentionLayerScalarsV1& scalars,
    const OfficialAttentionResourcesV1& official
  ) override {
    require(official.complete(),"fake provider saw incomplete official attention resource");
    stats->attentionPrepareCalls++;
    stats->events.push_back("prepare-attention-" + std::to_string(layer));
    stats->attentionInputRanges.push_back(scalars.inputQuantMaxAbsBits);
    stats->attentionOutputRanges.push_back(scalars.outputQuantMaxAbsBits);
    PrepareAttentionResultV1 result;
    if(layer == config->failAttentionLayer) {
      result.detail = "injected attention prepare failure";
      return result;
    }
    result.prepared = std::make_unique<FakePreparedAttention>(stats);
    return result;
  }

  PrepareFfnResultV1 prepareFfn(
    size_t layer,
    const FfnSpecV1&,
    const FfnLayerScalarsV1& scalars,
    const OfficialFfnResourcesV1& official
  ) override {
    require(official.complete(),"fake provider saw incomplete official FFN resource");
    stats->ffnPrepareCalls++;
    stats->events.push_back("prepare-ffn-" + std::to_string(layer));
    stats->ffnClips.push_back(scalars.swigluClipBits);
    stats->ffnInputRanges.push_back(scalars.inputQuantMaxAbsBits);
    stats->productRanges.push_back(scalars.productQuantMaxAbsBits);
    PrepareFfnResultV1 result;
    if(layer == config->failFfnLayer) {
      result.detail = "injected FFN prepare failure";
      return result;
    }
    result.prepared = std::make_unique<FakePreparedFfn>(stats);
    return result;
  }

  bool commit(PreparedSpanV1&& prepared, std::string& detail) override {
    stats->commits++;
    stats->events.push_back("commit");
    stats->committedAttention = (int)prepared.attention.size();
    stats->committedFfn = (int)prepared.ffn.size();
    if(!config->commitOk) {
      detail = "injected commit failure";
      return false;
    }
    committed = std::move(prepared);
    return true;
  }

  uint64_t committedPlanGeneration() const noexcept override {
    return committed.attention.empty() ? 0 : FAKE_PLAN_GENERATION;
  }

  bool acceptsActualBatchSize(
    int actualBatchSize,
    int physicalBatchSize
  ) const noexcept override {
    return actualBatchSize > 0 &&
      (config->allowSmallerActualBatch ?
        actualBatchSize <= physicalBatchSize :
        actualBatchSize == physicalBatchSize);
  }

  ProviderOpResultV1 preflight(const RuntimeCallV1&) override {
    stats->preflights++;
    stats->events.push_back("preflight");
    if(config->preflightThrows)
      throw std::runtime_error("injected preflight exception");
    return config->preflightResult;
  }

  ProviderOpResultV1 enqueue(const RuntimeCallV1&, uint64_t runToken) override {
    stats->enqueues++;
    stats->lastEnqueueToken = runToken;
    stats->events.push_back("enqueue");
    if(config->enqueueThrows)
      throw std::runtime_error("injected enqueue exception");
    return config->enqueueResult;
  }

private:
  std::string id;
  std::shared_ptr<FakeStats> stats;
  std::shared_ptr<FakeConfig> config;
  PreparedSpanV1 committed;
};

class FakeFactory final : public FactoryV1 {
public:
  FakeFactory(
    std::string id_,
    ProfileKeyV1 key_,
    const std::shared_ptr<FakeStats>& stats_,
    const std::shared_ptr<FakeConfig>& config_
  ) : id(std::move(id_)), key(std::move(key_)), stats(stats_), config(config_) {}

  const char* factoryId() const noexcept override { return id.c_str(); }
  bool matches(const ProfileKeyV1& candidate) const override { return candidate == key; }

  AvailabilityResultV1 availability(const ProfileKeyV1&) const override {
    AvailabilityResultV1 result;
    result.availability = config->availability;
    result.detail = "injected factory availability";
    return result;
  }

  std::unique_ptr<ProviderV1> create(const ProfileKeyV1&) const override {
    stats->creates++;
    return std::make_unique<FakeProvider>(id,stats,config);
  }

private:
  std::string id;
  ProfileKeyV1 key;
  std::shared_ptr<FakeStats> stats;
  std::shared_ptr<FakeConfig> config;
};

void addFakeFactory(
  RegistryV1& registry,
  const std::string& id,
  const ProfileKeyV1& key,
  const std::shared_ptr<FakeStats>& stats,
  const std::shared_ptr<FakeConfig>& config
) {
  registry.add(std::make_unique<FakeFactory>(id,key,stats,config));
}

void testDynamicDepthAndLayerlessKey() {
  const RuntimeKeyV1 runtime = runtimeKey();
  const AttentionSpecV1 attention = attentionSpec();
  const FfnSpecV1 ffn = ffnSpec();
  const ProfileKeyV1 referenceKey = keyFor(model(1,attention,ffn),runtime);

  auto stats = std::make_shared<FakeStats>();
  auto config = std::make_shared<FakeConfig>();
  RegistryV1 registry;
  addFakeFactory(registry,"dynamic-N-provider",referenceKey,stats,config);

  int expectedAttention = 0;
  int expectedFfn = 0;
  for(int depth: {1,24,36}) {
    const ModelViewV1 current = model(depth,attention,ffn,1,2);
    const ProfileKeyV1 currentKey = keyFor(current,runtime);
    require(currentKey == referenceKey,"layer count leaked into profile key");

    std::unique_ptr<ManagerV1> manager = ManagerV1::create(
      registry,current,runtime,ModeV1::Auto
    );
    require(manager->report().route == RouteV1::Specialized,"dynamic-depth provider was not selected");
    require(manager->report().span.pairCount == (size_t)depth,"dynamic span depth was not preserved");
    expectedAttention += depth;
    expectedFfn += depth;
    require(stats->attentionPrepareCalls == expectedAttention,"did not prepare exactly N attention layers");
    require(stats->ffnPrepareCalls == expectedFfn,"did not prepare exactly N FFN layers");
    require(stats->commits == (depth == 1 ? 1 : depth == 24 ? 2 : 3),"commit was not once per full span");
    require(stats->committedAttention == depth && stats->committedFfn == depth,"commit did not receive N/N resources");

    RuntimeCallV1 call = runtimeCall(runtime,manager->report().span);
    require(manager->preflight(call) == RouteV1::Specialized,"successful preflight did not select provider");
    require(manager->enqueue(call) == RouteV1::Specialized,"successful enqueue did not stay specialized");
    require(stats->events[stats->events.size()-2] == "preflight" &&
            stats->events.back() == "enqueue","provider enqueue preceded runtime preflight");
  }
}

void testSameGpuConcurrencyBridge() {
  require(countSameGpuConcurrencyV1({0,0},0) == 2,
    "two CUDA server threads on GPU 0 were not reported as S2");
  require(countSameGpuConcurrencyV1({0,1,0},0) == 2,
    "multi-GPU mapping did not count only the target GPU");
  require(countSameGpuConcurrencyV1({0,1,0},1) == 1,
    "multi-GPU mapping reported the wrong concurrency for GPU 1");
  require(countSameGpuConcurrencyV1({-1,-1},-1) == 2,
    "default GPU indices were not normalized to GPU 0");
  require(countSameGpuConcurrencyV1({1,1},0) == 0,
    "missing current GPU did not fail closed to zero concurrency");
}

void testMixedPositiveClipAndPerLayerScalars() {
  const int depth = 36;
  const AttentionSpecV1 attention = attentionSpec(384,12,true,true);
  const FfnSpecV1 ffn = ffnSpec(384,1024,true,true);
  const RuntimeKeyV1 runtime = runtimeKey(15,28,RequestedExecutionV1::Int8);
  ModelViewV1 modelView = model(depth,attention,ffn,0,0,105);

  std::vector<uint32_t> expectedClips;
  for(int layer = 0; layer < depth; layer++) {
    BlockViewV1& attentionLayer = modelView.blocks[(size_t)layer * 2];
    BlockViewV1& ffnLayer = modelView.blocks[(size_t)layer * 2 + 1];
    attentionLayer.attentionScalars.inputQuantMaxAbsBits = floatBits(2.0f + layer * 0.01f);
    attentionLayer.attentionScalars.outputQuantMaxAbsBits = floatBits(3.0f + layer * 0.01f);
    const float clip = layer % 3 == 0 ? 4.0f : (layer % 3 == 1 ? 7.0f : 5.5f);
    ffnLayer.ffnScalars.swigluClipBits = floatBits(clip);
    ffnLayer.ffnScalars.inputQuantMaxAbsBits = floatBits(4.0f + layer * 0.01f);
    ffnLayer.ffnScalars.productQuantMaxAbsBits = floatBits(20.0f + layer * 0.25f);
    expectedClips.push_back(floatBits(clip));
  }

  const ProfileKeyV1 key = keyFor(modelView,runtime);
  require(key.ffn.clipClass == ClipClassV1::PositiveFinite,
    "positive mixed clips were not reduced to a semantic key class");

  RegistryV1 fakeRegistry;
  auto stats = std::make_shared<FakeStats>();
  auto config = std::make_shared<FakeConfig>();
  addFakeFactory(fakeRegistry,"mixed-positive-provider",key,stats,config);
  std::unique_ptr<ManagerV1> manager = ManagerV1::create(
    fakeRegistry,modelView,runtime,ModeV1::Auto
  );
  require(manager->report().route == RouteV1::Specialized,
    "mixed positive clip/range layers were rejected as a nonuniform key");
  require(stats->ffnClips == expectedClips,
    "per-layer actual clip bits were not passed to provider preparation");
  require(stats->attentionInputRanges.size() == (size_t)depth &&
          stats->attentionOutputRanges.size() == (size_t)depth &&
          stats->ffnInputRanges.size() == (size_t)depth &&
          stats->productRanges.size() == (size_t)depth,
    "provider did not receive all four per-layer PTQ range streams");

  RegistryV1 builtinRegistry;
  registerBuiltinStubFactoriesV1(builtinRegistry);
  manager = ManagerV1::create(builtinRegistry,modelView,runtime,ModeV1::Auto);
  require(manager->report().route == RouteV1::Official &&
          manager->report().reason == ReasonV1::ProviderUnavailable,
    "P4 mixed positive clip profile did not reach its stub factory");
}

void testStrictTransformerSpan() {
  const AttentionSpecV1 attention = attentionSpec();
  const FfnSpecV1 ffn = ffnSpec();

  ModelViewV1 prefixed = model(24,attention,ffn,2,3);
  TransformerSpanV1 span = analyzeTransformerSpanV1(prefixed.blocks);
  require(span.valid && span.beginBlock == 2 && span.endBlock == 50 && span.pairCount == 24,
    "ordinary prefix/suffix was not preserved around strict transformer span");

  ModelViewV1 split = model(2,attention,ffn);
  split.blocks.insert(split.blocks.begin()+2,otherBlock());
  span = analyzeTransformerSpanV1(split.blocks);
  require(span.hasTransformer && !span.valid,"ordinary block inside transformer span was skipped");

  RegistryV1 emptyRegistry;
  std::unique_ptr<ManagerV1> manager = ManagerV1::create(
    emptyRegistry,split,runtimeKey(),ModeV1::Auto
  );
  require(manager->report().route == RouteV1::Official &&
          manager->report().reason == ReasonV1::InvalidTransformerSpan,
    "split transformer span did not stay strictly official");

  manager = ManagerV1::create(emptyRegistry,split,runtimeKey(),ModeV1::Required);
  require(manager->report().route == RouteV1::Official,
    "Required mode reinterpreted unmatched split topology as a provider candidate");

  ModelViewV1 noTransformer;
  noTransformer.modelVersion = 105;
  noTransformer.blocks.push_back(otherBlock());
  manager = ManagerV1::create(emptyRegistry,noTransformer,runtimeKey(),ModeV1::Required);
  require(manager->report().route == RouteV1::Official &&
          manager->report().reason == ReasonV1::NoTransformer,
    "unmatched non-transformer model was not strictly official");
}

void testRegistryOverlapIsFatal() {
  const ModelViewV1 modelView = model(1,attentionSpec(),ffnSpec());
  const RuntimeKeyV1 runtime = runtimeKey();
  const ProfileKeyV1 key = keyFor(modelView,runtime);
  RegistryV1 registry;
  auto stats = std::make_shared<FakeStats>();
  auto config = std::make_shared<FakeConfig>();
  addFakeFactory(registry,"overlap-a",key,stats,config);
  addFakeFactory(registry,"overlap-b",key,stats,config);
  expectException<FatalErrorV1>([&](){
    (void)ManagerV1::create(registry,modelView,runtime,ModeV1::Auto);
  },"overlapping provider factories");
  require(stats->creates == 0,"overlap was detected only after provider creation");
}

void testAtomicPrepareCommit() {
  const int depth = 24;
  const ModelViewV1 modelView = model(depth,attentionSpec(),ffnSpec());
  const RuntimeKeyV1 runtime = runtimeKey();
  const ProfileKeyV1 key = keyFor(modelView,runtime);
  RegistryV1 registry;
  auto stats = std::make_shared<FakeStats>();
  auto config = std::make_shared<FakeConfig>();
  config->failFfnLayer = depth - 1;
  addFakeFactory(registry,"fail-Nth-prepare",key,stats,config);

  std::unique_ptr<ManagerV1> manager = ManagerV1::create(
    registry,modelView,runtime,ModeV1::Auto
  );
  require(manager->report().route == RouteV1::Official &&
          manager->report().reason == ReasonV1::PrepareFailed,
    "Nth prepare failure did not fall back in Auto mode");
  require(stats->attentionPrepareCalls == depth && stats->ffnPrepareCalls == depth,
    "injected Nth prepare failure occurred at wrong layer");
  require(stats->commits == 0,"partial N/N preparation was committed");
  require(stats->livePrepared == 0,"staged resources survived failed atomic preparation");

  expectException<ErrorV1>([&](){
    (void)ManagerV1::create(registry,modelView,runtime,ModeV1::Required);
  },"Required Nth prepare failure");
  require(stats->commits == 0,"Required prepare failure committed a partial provider");
}

void testBuiltinAvailabilityAndRequiredMode() {
  RegistryV1 registry;
  registerBuiltinStubFactoriesV1(registry);
  require(registry.size() == 4,"built-in registry does not contain P1/P2/P3/P4 factories");

  auto requireStreamAgnosticMatch = [&registry](
    const ModelViewV1& modelView,
    const RuntimeKeyV1& baseRuntime,
    const std::string& profile
  ) {
    const int concurrentCounts[] = {1,4};
    for(int concurrentCount: concurrentCounts) {
      RuntimeKeyV1 runtime = baseRuntime;
      runtime.sameGpuConcurrency = concurrentCount;
      std::unique_ptr<ManagerV1> candidate = ManagerV1::create(
        registry,modelView,runtime,ModeV1::Auto
      );
      require(candidate->report().reason == ReasonV1::ProviderUnavailable,
        "same-GPU concurrency incorrectly participated in " + profile + " eligibility");
    }
  };

  const ModelViewV1 p2 = model(
    24,attentionSpec(256,8,false,false),ffnSpec(256,768,false,false),0,0,102
  );
  const RuntimeKeyV1 p2Runtime = runtimeKey(19,28,RequestedExecutionV1::Fp16);
  std::unique_ptr<ManagerV1> manager = ManagerV1::create(
    registry,p2,p2Runtime,ModeV1::Auto
  );
  require(manager->report().route == RouteV1::Official &&
          manager->report().reason == ReasonV1::ProviderUnavailable,
    "P2 stub did not use official Auto fallback");
  requireStreamAgnosticMatch(p2,p2Runtime,"P2");
  expectException<ErrorV1>([&](){
    (void)ManagerV1::create(registry,p2,p2Runtime,ModeV1::Required);
  },"Required P2 availability");

  const ModelViewV1 p1 = model(
    24,attentionSpec(256,8,false,false),ffnSpec(256,768,false,false),0,0,102
  );
  const RuntimeKeyV1 p1Runtime = runtimeKey(15,36,RequestedExecutionV1::Fp16);
  manager = ManagerV1::create(registry,p1,p1Runtime,ModeV1::Auto);
  require(manager->report().route == RouteV1::Official &&
          manager->report().reason == ReasonV1::ProviderUnavailable,
    "P1 stub did not use official Auto fallback");
  expectException<ErrorV1>([&](){
    (void)ManagerV1::create(registry,p1,p1Runtime,ModeV1::Required);
  },"Required P1 availability");

  RuntimeKeyV1 maskedRuntime = p1Runtime;
  maskedRuntime.maskMode = MaskModeV1::Dense;
  maskedRuntime.maskNull = false;
  manager = ManagerV1::create(registry,p1,maskedRuntime,ModeV1::Auto);
  require(manager->report().route == RouteV1::Official &&
          manager->report().reason == ReasonV1::Unmatched,
    "masked runtime matched a no-mask-only profile");

  requireStreamAgnosticMatch(p1,p1Runtime,"P1");

  RuntimeKeyV1 wrongPhysicalBatch = p1Runtime;
  wrongPhysicalBatch.physicalBatchSize = 35;
  manager = ManagerV1::create(registry,p1,wrongPhysicalBatch,ModeV1::Auto);
  require(manager->report().reason == ReasonV1::Unmatched,
    "wrong physical batch matched a fixed-batch profile");

  const ModelViewV1 p4 = model(
    36,attentionSpec(384,12,true,true),ffnSpec(384,1024,true,true),0,0,105
  );
  const RuntimeKeyV1 p4Runtime = runtimeKey(15,28,RequestedExecutionV1::Int8);
  manager = ManagerV1::create(registry,p4,p4Runtime,ModeV1::Auto);
  require(manager->report().reason == ReasonV1::ProviderUnavailable,
    "P4 did not keep external FP16 storage separate from internal INT8 execution");
  requireStreamAgnosticMatch(p4,p4Runtime,"P4");
  RuntimeKeyV1 wrongP4Storage = p4Runtime;
  wrongP4Storage.outputStorage = StorageTypeV1::Fp32;
  manager = ManagerV1::create(registry,p4,wrongP4Storage,ModeV1::Auto);
  require(manager->report().reason == ReasonV1::Unmatched,
    "P4 matched non-FP16 external output storage");

  const ModelViewV1 p3 = model(
    36,attentionSpec(384,12,true,false),ffnSpec(384,1024,true,false),0,0,102
  );
  const RuntimeKeyV1 p3Runtime = runtimeKey(15,28,RequestedExecutionV1::Fp16);
  manager = ManagerV1::create(registry,p3,p3Runtime,ModeV1::Auto);
  require(manager->report().reason == ReasonV1::ProviderUnavailable,
    "P3 exact FP16 profile did not reach its stub factory");
  requireStreamAgnosticMatch(p3,p3Runtime,"P3");

  auto requireP1Unmatched = [&](const ModelViewV1& candidate, const RuntimeKeyV1& candidateRuntime,
                                const std::string& field) {
    std::unique_ptr<ManagerV1> candidateManager = ManagerV1::create(
      registry,candidate,candidateRuntime,ModeV1::Auto
    );
    require(candidateManager->report().reason == ReasonV1::Unmatched,
      "P1 factory did not match " + field + " exactly");
  };
  RuntimeKeyV1 changedRuntime = p1Runtime;
  changedRuntime.physicalBatchSize = 35;
  requireP1Unmatched(p1,changedRuntime,"B");
  changedRuntime = p1Runtime;
  changedRuntime.boardX = 14;
  requireP1Unmatched(p1,changedRuntime,"S");
  requireP1Unmatched(
    model(24,attentionSpec(255,8,false,false),ffnSpec(255,768,false,false),0,0,102),
    p1Runtime,"C"
  );
  requireP1Unmatched(
    model(24,attentionSpec(256,7,false,false),ffnSpec(256,768,false,false),0,0,102),
    p1Runtime,"H"
  );
  AttentionSpecV1 changedDim = attentionSpec(256,8,false,false);
  changedDim.qHeadDim = 16;
  requireP1Unmatched(model(24,changedDim,ffnSpec(256,768,false,false),0,0,102),
    p1Runtime,"D");
  requireP1Unmatched(
    model(24,attentionSpec(256,8,false,false),ffnSpec(256,769,false,false),0,0,102),
    p1Runtime,"F"
  );
}

void testBuiltinP1DynamicDepthAndSemanticExclusions() {
  RegistryV1 registry;
  registerBuiltinStubFactoriesV1(registry);
  const RuntimeKeyV1 runtime = runtimeKey(15,36,RequestedExecutionV1::Fp16);
  const AttentionSpecV1 attention = attentionSpec(256,8,false,false);
  const FfnSpecV1 ffn = ffnSpec(256,768,false,false);

  for(int depth: {1,24,36}) {
    const ModelViewV1 candidate = model(depth,attention,ffn,1,1,102);
    const ProfileKeyV1 key = keyFor(candidate,runtime);
    const ResolutionV1 resolution = registry.resolve(key);
    require(resolution.matched(),
      "P1 dynamic depth " + std::to_string(depth) + " did not match");
    require(std::strcmp(
      resolution.factory->factoryId(),"P1-c256-h8-s225-fp16-b36-s2") == 0,
      "P1 dynamic depth resolved to the wrong factory");
  }

  auto requireNoP1Match = [&](const ModelViewV1& candidate,
                              const RuntimeKeyV1& candidateRuntime,
                              const std::string& exclusion) {
    const ResolutionV1 resolution = registry.resolve(
      keyFor(candidate,candidateRuntime));
    require(!resolution.matched(),
      "P1 matched excluded " + exclusion + " semantics");
  };

  RuntimeKeyV1 b35 = runtime;
  b35.physicalBatchSize = 35;
  requireNoP1Match(model(24,attention,ffn,0,0,102),b35,"B35");
  requireNoP1Match(
    model(24,attentionSpec(256,8,true,false),ffn,0,0,102),runtime,"QKN");
  requireNoP1Match(
    model(24,attention,ffnSpec(256,768,true,false),0,0,102),runtime,"positive clip");
  requireNoP1Match(
    model(24,attention,ffn,0,0,105),runtime,"v105");
}

void testRuntimePreflightAndEnqueueFailurePolicy() {
  const ModelViewV1 modelView = model(1,attentionSpec(),ffnSpec());
  const RuntimeKeyV1 runtime = runtimeKey();
  const ProfileKeyV1 key = keyFor(modelView,runtime);
  RuntimeCallV1 call = runtimeCall(
    runtime,analyzeTransformerSpanV1(modelView.blocks)
  );

  {
    RegistryV1 registry;
    auto stats = std::make_shared<FakeStats>();
    auto config = std::make_shared<FakeConfig>();
    config->preflightResult = ProviderOpResultV1::failure("injected preflight rejection");
    addFakeFactory(registry,"auto-preflight",key,stats,config);
    std::unique_ptr<ManagerV1> manager = ManagerV1::create(
      registry,modelView,runtime,ModeV1::Auto
    );
    require(manager->preflight(call) == RouteV1::Official,
      "Auto preflight rejection did not fall back before enqueue");
    require(stats->preflights == 1 && stats->enqueues == 0,
      "Auto fallback enqueued specialized work before preflight completed");
  }

  {
    RegistryV1 registry;
    auto stats = std::make_shared<FakeStats>();
    auto config = std::make_shared<FakeConfig>();
    config->preflightResult = ProviderOpResultV1::failure("injected Required rejection");
    addFakeFactory(registry,"required-preflight",key,stats,config);
    std::unique_ptr<ManagerV1> manager = ManagerV1::create(
      registry,modelView,runtime,ModeV1::Required
    );
    expectException<ErrorV1>([&](){ (void)manager->preflight(call); },
      "Required runtime preflight rejection");
    require(stats->enqueues == 0,"Required preflight rejection enqueued provider work");
  }

  {
    RegistryV1 registry;
    auto stats = std::make_shared<FakeStats>();
    auto config = std::make_shared<FakeConfig>();
    config->preflightResult =
      ProviderOpResultV1::success(1,FAKE_PLAN_GENERATION,FAKE_RUN_TOKEN);
    addFakeFactory(registry,"bad-preflight-enqueue",key,stats,config);
    std::unique_ptr<ManagerV1> manager = ManagerV1::create(
      registry,modelView,runtime,ModeV1::Auto
    );
    expectException<FatalErrorV1>([&](){ (void)manager->preflight(call); },
      "preflight that enqueued work");
  }

  {
    RegistryV1 registry;
    auto stats = std::make_shared<FakeStats>();
    auto config = std::make_shared<FakeConfig>();
    config->enqueueResult = ProviderOpResultV1::failure("injected post-enqueue failure",1);
    addFakeFactory(registry,"fatal-post-enqueue",key,stats,config);
    std::unique_ptr<ManagerV1> manager = ManagerV1::create(
      registry,modelView,runtime,ModeV1::Auto
    );
    require(manager->preflight(call) == RouteV1::Specialized,"fatal enqueue test preflight failed");
    expectException<FatalErrorV1>([&](){ (void)manager->enqueue(call); },
      "provider failure after enqueue");
    require(stats->enqueues == 1,"post-enqueue fatal path did not call provider exactly once");
  }

  {
    RegistryV1 registry;
    auto stats = std::make_shared<FakeStats>();
    auto config = std::make_shared<FakeConfig>();
    config->enqueueResult = ProviderOpResultV1::failure("rejected before enqueue",0);
    addFakeFactory(registry,"auto-zero-enqueue-failure",key,stats,config);
    std::unique_ptr<ManagerV1> manager = ManagerV1::create(
      registry,modelView,runtime,ModeV1::Auto
    );
    require(manager->preflight(call) == RouteV1::Specialized,"zero-enqueue test preflight failed");
    require(manager->enqueue(call) == RouteV1::Official,
      "Auto zero-enqueue failure did not preserve official fallback");
  }


  {
    RegistryV1 registry;
    auto stats = std::make_shared<FakeStats>();
    auto config = std::make_shared<FakeConfig>();
    config->preflightResult =
      ProviderOpResultV1::success(0,FAKE_PLAN_GENERATION + 1,FAKE_RUN_TOKEN);
    addFakeFactory(registry,"stale-plan-preflight",key,stats,config);
    std::unique_ptr<ManagerV1> manager = ManagerV1::create(
      registry,modelView,runtime,ModeV1::Auto
    );
    expectException<FatalErrorV1>([&](){ (void)manager->preflight(call); },
      "preflight with stale plan generation");
    require(stats->enqueues == 0,"stale preflight generation reached provider enqueue");
  }

  {
    RegistryV1 registry;
    auto stats = std::make_shared<FakeStats>();
    auto config = std::make_shared<FakeConfig>();
    config->enqueueThrows = true;
    addFakeFactory(registry,"enqueue-exception",key,stats,config);
    std::unique_ptr<ManagerV1> manager = ManagerV1::create(
      registry,modelView,runtime,ModeV1::Auto
    );
    require(manager->preflight(call) == RouteV1::Specialized,
      "enqueue-exception test preflight failed");
    expectException<FatalErrorV1>([&](){ (void)manager->enqueue(call); },
      "provider enqueue exception with unknown side effects");
    require(stats->enqueues == 1,"enqueue exception was not observed exactly once");
  }
}

void testPreflightBindsCompleteCallIdentity() {
  const ModelViewV1 modelView = model(1,attentionSpec(),ffnSpec());
  const RuntimeKeyV1 runtime = runtimeKey();
  const ProfileKeyV1 key = keyFor(modelView,runtime);
  int sentinels[12] = {};

  RuntimeCallV1 base = runtimeCall(
    runtime,analyzeTransformerSpanV1(modelView.blocks)
  );
  base.stream = &sentinels[0];
  base.trunk = &sentinels[1];
  base.trunkScratch = &sentinels[2];
  base.mask = nullptr;
  base.workspace = &sentinels[4];
  base.workspaceBytes = 4096;

  for(int mutation = 0; mutation < 11; mutation++) {
    RegistryV1 registry;
    auto stats = std::make_shared<FakeStats>();
    auto config = std::make_shared<FakeConfig>();
    addFakeFactory(registry,"call-identity-" + std::to_string(mutation),key,stats,config);
    std::unique_ptr<ManagerV1> manager = ManagerV1::create(
      registry,modelView,runtime,ModeV1::Auto
    );
    require(manager->preflight(base) == RouteV1::Specialized,
      "call identity test preflight failed");

    RuntimeCallV1 changed = base;
    switch(mutation) {
    case 0: changed.stream = &sentinels[5]; break;
    case 1: changed.trunk = &sentinels[6]; break;
    case 2: changed.trunkScratch = &sentinels[7]; break;
    case 3: changed.mask = &sentinels[8]; break;
    case 4: changed.workspace = &sentinels[9]; break;
    case 5: changed.workspaceBytes++; break;
    case 6: changed.actualBatchSize--; break;
    case 7: changed.sequenceSize--; break;
    case 8: changed.transformerBeginBlock++; break;
    case 9: changed.transformerPairCount++; break;
    case 10: changed.mask = &sentinels[10]; break;
    }
    expectException<FatalErrorV1>([&](){ (void)manager->enqueue(changed); },
      "preflight/enqueue call identity mutation " + std::to_string(mutation));
    require(stats->enqueues == 0,
      "call identity mutation reached provider enqueue");
  }
}

void testCentralPreflightEvidenceRejectsBeforeProvider() {
  const ModelViewV1 modelView = model(1,attentionSpec(),ffnSpec());
  const RuntimeKeyV1 runtime = runtimeKey();
  const ProfileKeyV1 key = keyFor(modelView,runtime);
  const RuntimeCallV1 valid = runtimeCall(
    runtime,analyzeTransformerSpanV1(modelView.blocks)
  );

  for(int mutation = 0; mutation < 10; mutation++) {
    RegistryV1 registry;
    auto stats = std::make_shared<FakeStats>();
    auto config = std::make_shared<FakeConfig>();
    addFakeFactory(registry,"central-evidence-" + std::to_string(mutation),key,stats,config);
    std::unique_ptr<ManagerV1> manager = ManagerV1::create(
      registry,modelView,runtime,ModeV1::Auto
    );
    RuntimeCallV1 invalid = valid;
    switch(mutation) {
    case 0: invalid.actualBatchSize--; break;
    case 1: invalid.sequenceSize--; break;
    case 2: invalid.transformerBeginBlock++; break;
    case 3: invalid.transformerPairCount++; break;
    case 4: invalid.stream = nullptr; break;
    case 5: invalid.trunk = nullptr; break;
    case 6: invalid.trunkScratch = nullptr; break;
    case 7: invalid.workspace = nullptr; break;
    case 8: invalid.workspaceBytes = 0; break;
    case 9: invalid.mask = &stats; break;
    }
    require(manager->preflight(invalid) == RouteV1::Official,
      "invalid central preflight evidence did not Auto-fallback");
    require(stats->preflights == 0 && stats->enqueues == 0,
      "invalid central evidence reached the provider");
  }

  RegistryV1 zeroWorkspaceRegistry;
  auto zeroWorkspaceStats = std::make_shared<FakeStats>();
  auto zeroWorkspaceConfig = std::make_shared<FakeConfig>();
  addFakeFactory(
    zeroWorkspaceRegistry,"zero-workspace",key,zeroWorkspaceStats,zeroWorkspaceConfig
  );
  std::unique_ptr<ManagerV1> zeroWorkspaceManager = ManagerV1::create(
    zeroWorkspaceRegistry,modelView,runtime,ModeV1::Auto
  );
  RuntimeCallV1 zeroWorkspace = valid;
  zeroWorkspace.workspace = nullptr;
  zeroWorkspace.workspaceBytes = 0;
  require(zeroWorkspaceManager->preflight(zeroWorkspace) == RouteV1::Specialized,
    "valid null/zero workspace evidence was rejected centrally");
  require(zeroWorkspaceManager->enqueue(zeroWorkspace) == RouteV1::Specialized,
    "valid null/zero workspace call did not reach provider enqueue");
}

void testGenericProviderMayOptIntoDynamicActualBatch() {
  const ModelViewV1 modelView = model(1,attentionSpec(),ffnSpec());
  const RuntimeKeyV1 runtime = runtimeKey(15,7);
  const ProfileKeyV1 key = keyFor(modelView,runtime);

  RegistryV1 registry;
  auto stats = std::make_shared<FakeStats>();
  auto config = std::make_shared<FakeConfig>();
  config->allowSmallerActualBatch = true;
  addFakeFactory(registry,"dynamic-actual-batch",key,stats,config);
  std::unique_ptr<ManagerV1> manager = ManagerV1::create(
    registry,modelView,runtime,ModeV1::Auto
  );

  RuntimeCallV1 call = runtimeCall(
    runtime,analyzeTransformerSpanV1(modelView.blocks)
  );
  call.actualBatchSize = 3;
  require(call.key.physicalBatchSize == 7,
    "dynamic actual batch changed the immutable physical runtime key");
  require(manager->preflight(call) == RouteV1::Specialized,
    "opted-in smaller actual batch did not reach provider preflight");
  require(manager->enqueue(call) == RouteV1::Specialized,
    "opted-in smaller actual batch did not reach provider enqueue");
  require(stats->preflights == 1 && stats->enqueues == 1,
    "dynamic actual-batch call did not execute exactly once");

  auto invalidStats = std::make_shared<FakeStats>();
  RegistryV1 invalidRegistry;
  addFakeFactory(
    invalidRegistry,"dynamic-actual-batch-invalid",key,invalidStats,config
  );
  manager = ManagerV1::create(
    invalidRegistry,modelView,runtime,ModeV1::Auto
  );
  call.actualBatchSize = 8;
  require(manager->preflight(call) == RouteV1::Official,
    "actual batch larger than the physical maximum did not fall back");
  require(invalidStats->preflights == 0 && invalidStats->enqueues == 0,
    "oversized actual batch reached the provider");
}

void testUnmatchedAndOfficialResourceContracts() {
  const ModelViewV1 modelView = model(1,attentionSpec(),ffnSpec());
  const RuntimeKeyV1 runtime = runtimeKey();
  RegistryV1 emptyRegistry;
  std::unique_ptr<ManagerV1> manager = ManagerV1::create(
    emptyRegistry,modelView,runtime,ModeV1::Required
  );
  require(manager->report().route == RouteV1::Official &&
          manager->report().reason == ReasonV1::Unmatched,
    "unmatched Required model was not strictly official");

  RegistryV1 overlappingRegistry;
  const ProfileKeyV1 key = keyFor(modelView,runtime);
  auto stats = std::make_shared<FakeStats>();
  auto config = std::make_shared<FakeConfig>();
  addFakeFactory(overlappingRegistry,"disabled-overlap-a",key,stats,config);
  addFakeFactory(overlappingRegistry,"disabled-overlap-b",key,stats,config);
  manager = ManagerV1::create(overlappingRegistry,modelView,runtime,ModeV1::Off);
  require(manager->report().route == RouteV1::Official &&
          manager->report().reason == ReasonV1::Disabled &&
          stats->creates == 0,
    "Off mode consulted or created a specialization provider");

  ModelViewV1 incomplete = modelView;
  incomplete.blocks[0].officialAttention.executable = nullptr;
  expectException<FatalErrorV1>([&](){
    (void)ManagerV1::create(emptyRegistry,incomplete,runtime,ModeV1::Auto);
  },"incomplete official resources");
}

void testStrictModeParser() {
  require(parseModeSettingV1(nullptr) == ModeV1::Auto,"unset mode did not default to Auto");
  require(parseModeSettingV1("off") == ModeV1::Off,"off mode did not parse");
  require(parseModeSettingV1("auto") == ModeV1::Auto,"auto mode did not parse");
  require(parseModeSettingV1("required") == ModeV1::Required,"required mode did not parse");
  for(const char* invalid: {"", "AUTO", "Required", "auto ", "0", "optional"}) {
    expectException<ErrorV1>([&](){ (void)parseModeSettingV1(invalid); },
      std::string("invalid mode '") + invalid + "'");
  }
}

}  // namespace

int main() {
  try {
    testDynamicDepthAndLayerlessKey();
    testSameGpuConcurrencyBridge();
    testMixedPositiveClipAndPerLayerScalars();
    testStrictTransformerSpan();
    testRegistryOverlapIsFatal();
    testAtomicPrepareCommit();
    testBuiltinAvailabilityAndRequiredMode();
    testBuiltinP1DynamicDepthAndSemanticExclusions();
    testRuntimePreflightAndEnqueueFailurePolicy();
    testPreflightBindsCompleteCallIdentity();
    testCentralPreflightEvidenceRejectsBeforeProvider();
    testGenericProviderMayOptIntoDynamicActualBatch();
    testUnmatchedAndOfficialResourceContracts();
    testStrictModeParser();
    std::cout << "FOUR_PROFILE_PROVIDER_CONTRACT_PASS" << std::endl;
    return 0;
  }
  catch(const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}

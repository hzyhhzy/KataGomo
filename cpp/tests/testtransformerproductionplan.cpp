#include "../tests/tests.h"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "../core/test.h"
#include "../neuralnet/activations.h"
#include "../neuralnet/architecturedesc.h"
#include "../neuralnet/cudabackend_transformer_winner.h"
#include "../neuralnet/cudaopregistry.h"
#include "../neuralnet/desc.h"

using namespace std;
using namespace NeuralNetArchitecture;
using namespace CudaOpRegistry;

namespace {

// This fixture intentionally stores only one sentinel element per learned
// tensor. Architecture and production-plan identity must be weight-free, and
// this keeps the 24/48-layer CPU contract test small and fast.
static void setConv(
  ConvLayerDesc& desc,
  int inChannels,
  int outChannels,
  int kernel,
  float weight
) {
  desc.name = "weight-independent-conv-" + Global::floatToString(weight);
  desc.convXSize = kernel;
  desc.convYSize = kernel;
  desc.inChannels = inChannels;
  desc.outChannels = outChannels;
  desc.dilationX = 1;
  desc.dilationY = 1;
  desc.weights.assign(1,weight);
}

static void setMatMul(MatMulLayerDesc& desc, int inChannels, int outChannels, float weight) {
  desc.name = "weight-independent-matmul-" + Global::floatToString(weight);
  desc.inChannels = inChannels;
  desc.outChannels = outChannels;
  desc.weights.assign(1,weight);
}

static void setMatBias(MatBiasLayerDesc& desc, int channels, float weight) {
  desc.name = "weight-independent-bias-" + Global::floatToString(weight);
  desc.numChannels = channels;
  desc.weights.assign(1,weight);
}

static void setBatchNorm(
  BatchNormLayerDesc& desc,
  int channels,
  bool hasScale,
  bool hasBias,
  float weight
) {
  desc.name = "weight-independent-bn-" + Global::floatToString(weight);
  desc.numChannels = channels;
  desc.epsilon = 1e-20f;
  desc.hasScale = hasScale;
  desc.hasBias = hasBias;
  desc.mean.assign(1,weight);
  desc.variance.assign(1,1.0f + weight);
  desc.scale.assign(1,hasScale ? 1.0f + weight : 1.0f);
  desc.bias.assign(1,hasBias ? weight : 0.0f);
}

static void setActivation(ActivationLayerDesc& desc, int activation, float weight) {
  desc.name = "weight-independent-activation-" + Global::floatToString(weight);
  desc.activation = activation;
}

static unique_ptr_void makeAttention(
  int channels,
  int heads,
  int headDim,
  float weight
) {
  TransformerAttentionDesc* desc = new TransformerAttentionDesc();
  desc->name = "weight-independent-attention-" + Global::floatToString(weight);
  desc->numHeads = heads;
  desc->numKVHeads = heads;
  desc->qHeadDim = headDim;
  desc->vHeadDim = headDim;
  desc->useRope = true;
  desc->learnableRope = true;
  desc->preLN.name = "weight-independent-rms";
  desc->preLN.numChannels = channels;
  desc->preLN.epsilon = 1e-6f;
  desc->preLN.weight.assign(1,weight);
  setMatMul(desc->qProj,channels,heads * headDim,weight);
  setMatMul(desc->kProj,channels,heads * headDim,weight + 0.01f);
  setMatMul(desc->vProj,channels,heads * headDim,weight + 0.02f);
  setMatMul(desc->outProj,heads * headDim,channels,weight + 0.03f);
  desc->ropeNumKVHeads = heads;
  desc->ropeNumPairs = headDim / 2;
  desc->ropeFreqs.assign(1,weight + 0.04f);
  return make_unique_void(desc);
}

static unique_ptr_void makeFFN(int channels, int ffnChannels, float weight) {
  TransformerFFNDesc* desc = new TransformerFFNDesc();
  desc->name = "weight-independent-ffn-" + Global::floatToString(weight);
  desc->numChannels = channels;
  desc->ffnChannels = ffnChannels;
  desc->useSwiGLU = true;
  desc->preLN.name = "weight-independent-rms";
  desc->preLN.numChannels = channels;
  desc->preLN.epsilon = 1e-6f;
  desc->preLN.weight.assign(1,weight);
  setMatMul(desc->linear1,channels,ffnChannels,weight);
  setMatMul(desc->linearGate,channels,ffnChannels,weight + 0.01f);
  setMatMul(desc->linear2,ffnChannels,channels,weight + 0.02f);
  return make_unique_void(desc);
}

static ModelDesc makeModel(
  int logicalLayers,
  int channels,
  int ffnChannels,
  int heads,
  int headDim,
  float weightSeed,
  const string& artifactTag
) {
  ModelDesc model;
  model.name = "production-plan-fixture-" + artifactTag;
  model.sha256 = "artifact-sha-" + artifactTag;
  model.version = 102;
  model.numInputChannels = 22;
  model.numInputGlobalChannels = 39;
  model.numValueChannels = 3;
  model.numScoreValueChannels = 6;
  model.numOwnershipChannels = 1;

  model.trunk.name = "fixture-trunk-" + artifactTag;
  model.trunk.version = 102;
  model.trunk.numBlocks = logicalLayers * 2;
  model.trunk.trunkNumChannels = channels;
  model.trunk.midNumChannels = channels;
  model.trunk.regularNumChannels = channels;
  model.trunk.dilatedNumChannels = 0;
  model.trunk.gpoolNumChannels = channels;
  setConv(model.trunk.initialConv,22,channels,1,weightSeed);
  setMatMul(model.trunk.initialMatMul,39,channels,weightSeed);
  for(int i = 0; i < logicalLayers; i++) {
    model.trunk.blocks.push_back(make_pair(
      TRANSFORMER_ATTENTION_BLOCK_KIND,
      makeAttention(channels,heads,headDim,weightSeed + i)
    ));
    model.trunk.blocks.push_back(make_pair(
      TRANSFORMER_FFN_BLOCK_KIND,
      makeFFN(channels,ffnChannels,weightSeed + i)
    ));
  }
  setBatchNorm(model.trunk.trunkTipBN,channels,true,true,weightSeed);
  setActivation(model.trunk.trunkTipActivation,ACTIVATION_SILU,weightSeed);

  model.policyHead.name = "fixture-policy-" + artifactTag;
  model.policyHead.version = 102;
  setConv(model.policyHead.p1Conv,channels,48,1,weightSeed);
  setConv(model.policyHead.g1Conv,channels,48,1,weightSeed);
  setBatchNorm(model.policyHead.g1BN,48,false,true,weightSeed);
  setActivation(model.policyHead.g1Activation,ACTIVATION_SILU,weightSeed);
  setMatMul(model.policyHead.gpoolToBiasMul,48 * 3,48,weightSeed);
  setBatchNorm(model.policyHead.p1BN,48,false,true,weightSeed);
  setActivation(model.policyHead.p1Activation,ACTIVATION_SILU,weightSeed);
  setConv(model.policyHead.p2Conv,48,1,1,weightSeed);
  setMatMul(model.policyHead.gpoolToPassMul,48 * 3,1,weightSeed);

  model.valueHead.name = "fixture-value-" + artifactTag;
  model.valueHead.version = 102;
  setConv(model.valueHead.v1Conv,channels,96,1,weightSeed);
  setBatchNorm(model.valueHead.v1BN,96,false,true,weightSeed);
  setActivation(model.valueHead.v1Activation,ACTIVATION_SILU,weightSeed);
  setMatMul(model.valueHead.v2Mul,96 * 3,64,weightSeed);
  setMatBias(model.valueHead.v2Bias,64,weightSeed);
  setActivation(model.valueHead.v2Activation,ACTIVATION_SILU,weightSeed);
  setMatMul(model.valueHead.v3Mul,64,3,weightSeed);
  setMatBias(model.valueHead.v3Bias,3,weightSeed);
  setMatMul(model.valueHead.sv3Mul,64,6,weightSeed);
  setMatBias(model.valueHead.sv3Bias,6,weightSeed);
  setConv(model.valueHead.vOwnershipConv,96,1,1,weightSeed);
  return model;
}

static RuntimeOpContext runtimeContext(int batch, int boardX, int boardY, MaskMode mask) {
  RuntimeOpContext context{};
  context.batchSize = batch;
  context.boardX = boardX;
  context.boardY = boardY;
  context.maskMode = mask;
  context.inputType = NumericType::Float16;
  context.outputType = NumericType::Float16;
  context.computeType = NumericType::Float32;
  context.layout = TensorLayout::BSH;
  context.deviceComputeCapability = 120;
  context.streamCount = 2;
  context.runtimeLibraryFingerprint = CudaTransformerWinner::makeRuntimeLibraryFingerprint(
    13000,13020,130101,91400);
  return context;
}

enum class FixtureTacticKind {
  Generic,
  Renju15AttentionB36S2,
  Renju15FFNB36S2,
};

struct FixtureTacticData {
  FixtureTacticKind kind;
  uint64_t cookie;
};

static bool isWinnerRuntime(const CapabilityKey& key) {
  const bool boardShapeMatches =
    key.kind != ArchitectureOpKind::TransformerAttention ||
    (key.boardX == 15 && key.boardY == 15);
  return key.batchSize == 36 && key.spatialArea == 225 && boardShapeMatches &&
    key.maskMode == MaskMode::None &&
    key.inputType == NumericType::Float16 && key.outputType == NumericType::Float16 &&
    key.deviceComputeCapability == 120 && key.streamCount == 2 &&
    key.runtimeLibraryFingerprint == CudaTransformerWinner::makeRuntimeLibraryFingerprint(
      13000,13020,130101,91400);
}

static SupportClass matchFixtureTactic(const OpRequest& request, const void* userData) {
  const FixtureTacticData* data = (const FixtureTacticData*)userData;
  const CapabilityKey& key = request.key;
  if(data->kind == FixtureTacticKind::Generic)
    return SupportClass::CompatibleOnly;
  if(data->kind == FixtureTacticKind::Renju15AttentionB36S2) {
    const bool geometry =
      key.kind == ArchitectureOpKind::TransformerAttention &&
      key.inChannels == 256 && key.outChannels == 256 &&
      key.numHeads == 8 && key.numKVHeads == 8 &&
      key.qHeadDim == 32 && key.vHeadDim == 32;
    return geometry && isWinnerRuntime(key) ?
      SupportClass::CertifiedFast : SupportClass::Unsupported;
  }
  const bool geometry =
    key.kind == ArchitectureOpKind::TransformerFFN &&
    key.inChannels == 256 && key.outChannels == 256 &&
    key.auxiliaryChannels == 768;
  return geometry && isWinnerRuntime(key) ?
    SupportClass::CertifiedFast : SupportClass::Unsupported;
}

static bool prepareFixtureTactic(const OpRequest&, PreparedOp& prepared, void* userData) {
  const FixtureTacticData* data = (const FixtureTacticData*)userData;
  prepared.workspaceAlignment = data->kind == FixtureTacticKind::Generic ? 64 : 256;
  prepared.workspaceBytes = data->kind == FixtureTacticKind::Generic ? 1024 : 8192;
  prepared.implementationCookie = (uintptr_t)data->cookie;
  return true;
}

static TacticRegistration registration(
  uint64_t variant,
  int priority,
  const string& recipe,
  FixtureTacticData* data
) {
  TacticRegistration result{};
  result.id = TacticId{0x52454E4A553135ULL,variant};
  result.recipe = fingerprintRecipe(recipe);
  result.priority = priority;
  result.match = matchFixtureTactic;
  result.prepare = prepareFixtureTactic;
  result.userData = data;
  return result;
}

struct PlanSnapshot {
  vector<OpRequest> requests;
  vector<PreparedOp> prepared;
  PlanFingerprint fingerprint;
};

static PlanSnapshot preparePlan(
  const ArchitectureDesc& architecture,
  const RuntimeOpContext& runtime,
  const Registry& registry
) {
  PlanSnapshot result;
  result.requests = buildOpRequests(architecture,runtime);
  result.prepared.reserve(result.requests.size());
  for(const OpRequest& request: result.requests) {
    ResolveResult resolved = registry.resolveAtConstruction(request);
    testAssert(resolved.found);
    result.prepared.push_back(resolved.prepared);
  }
  result.fingerprint = fingerprintPreparedPlan(result.requests,result.prepared);
  return result;
}

static void assertPreparedIdentity(const PreparedOp& a, const PreparedOp& b) {
  testAssert(a.tactic == b.tactic);
  testAssert(a.recipe == b.recipe);
  testAssert(a.support == b.support);
  testAssert(a.workspaceAlignment == b.workspaceAlignment);
  testAssert(a.workspaceBytes == b.workspaceBytes);
}

static size_t findRequest(
  const PlanSnapshot& plan,
  ArchitectureOpKind kind,
  int inChannels,
  int outChannels,
  int auxiliaryChannels = 0
) {
  for(size_t i = 0; i < plan.requests.size(); i++) {
    const CapabilityKey& key = plan.requests[i].key;
    if(key.kind == kind && key.inChannels == inChannels &&
       key.outChannels == outChannels && key.auxiliaryChannels == auxiliaryChannels)
      return i;
  }
  testAssert(false);
  return 0;
}

static void assertRuntimeKeyLocalization(
  const ArchitectureDesc& architecture,
  const RuntimeOpContext& a,
  const RuntimeOpContext& b,
  uint32_t changedDependencyMask
) {
  vector<OpRequest> requestA = buildOpRequests(architecture,a);
  vector<OpRequest> requestB = buildOpRequests(architecture,b);
  testAssert(requestA.size() == architecture.operators.size());
  testAssert(requestA.size() == requestB.size());
  for(size_t i = 0; i < requestA.size(); i++) {
    const bool shouldChange =
      (architecture.operators[i].runtimeDependencies & changedDependencyMask) != 0;
    testAssert((requestA[i].key != requestB[i].key) == shouldChange);
  }
}

static void assertOnlyTransformerWinnerFallsBack(
  const ArchitectureDesc& architecture,
  const PlanSnapshot& winner,
  const PlanSnapshot& changed
) {
  testAssert(winner.prepared.size() == changed.prepared.size());
  for(size_t i = 0; i < winner.prepared.size(); i++) {
    const ArchitectureOpKind kind = architecture.operators[i].kind;
    if(kind == ArchitectureOpKind::TransformerAttention ||
       kind == ArchitectureOpKind::TransformerFFN) {
      testAssert(winner.prepared[i].support == SupportClass::CertifiedFast);
      testAssert(changed.prepared[i].support == SupportClass::CompatibleOnly);
      testAssert(winner.prepared[i].tactic != changed.prepared[i].tactic);
    }
    else
      assertPreparedIdentity(winner.prepared[i],changed.prepared[i]);
  }
}

static const char* opKindName(ArchitectureOpKind kind) {
  switch(kind) {
  case ArchitectureOpKind::Conv2D: return "Conv2D";
  case ArchitectureOpKind::BatchNormActivation: return "BatchNormActivation";
  case ArchitectureOpKind::MatMul: return "MatMul";
  case ArchitectureOpKind::MatBias: return "MatBias";
  case ArchitectureOpKind::TransformerAttention: return "TransformerAttention";
  case ArchitectureOpKind::TransformerFFN: return "TransformerFFN";
  }
  return "Unknown";
}

static CudaTransformerWinner::DeviceCapability sm120Device() {
  CudaTransformerWinner::DeviceCapability device{};
  device.computeCapability = 120;
  device.warpSize = 32;
  device.sharedBytesPerBlockOptin = 101376;
  device.cudaRuntimeVersion = 13000;
  device.cudaDriverVersion = 13020;
  device.cublasVersion = 130101;
  device.cudnnVersion = 91400;
  device.specializedSm120KernelsAvailable = true;
  return device;
}

static void assertPreparedOpIdentity(const PreparedOp& a, const PreparedOp& b) {
  testAssert(a.tactic == b.tactic);
  testAssert(a.recipe == b.recipe);
  testAssert(a.support == b.support);
  testAssert(a.workspaceAlignment == b.workspaceAlignment);
  testAssert(a.workspaceBytes == b.workspaceBytes);
  testAssert(a.implementationCookie == b.implementationCookie);
}

static void assertProductionPlanIdentity(
  const CudaTransformerWinner::PreparedPlan& a,
  const CudaTransformerWinner::PreparedPlan& b
) {
  testAssert(a.architecture == b.architecture);
  testAssert(a.fingerprint == b.fingerprint);
  testAssert(a.records.size() == b.records.size());
  for(size_t i = 0; i < a.records.size(); i++) {
    const CudaTransformerWinner::PreparedRecord& recordA = a.records[i];
    const CudaTransformerWinner::PreparedRecord& recordB = b.records[i];
    testAssert(recordA.request.key == recordB.request.key);
    testAssert(recordA.request.architecture == recordB.request.architecture);
    testAssert(recordA.request.topologyIndex == recordB.request.topologyIndex);
    testAssert(recordA.found == recordB.found);
    assertPreparedOpIdentity(recordA.operation,recordB.operation);
  }
}

static void assertExactAttentionRecipe(
  const CudaTransformerWinner::AttentionRecipe& recipe
) {
  using namespace CudaTransformerWinner;
  testAssert(recipe.planarQkv == PlanarQkvTactic::CublasHgemmStridedBatchedSquare);
  testAssert(recipe.rmsNorm == RmsNormTactic::Sm120C256Warp4Vec8);
  testAssert(recipe.rope == RopeTactic::LearnedHalf2);
  testAssert(recipe.qkvRope == QkvRopeTactic::Sm120C256H8D32M128N128K32S3);
  testAssert(recipe.attention == AttentionTactic::Fa4Sm120B36S225Tm128Tn128S1Both16);
  testAssert(recipe.outProjection == ResidualTactic::Sm120M128N128K32S3Sw1);
}

static void assertExactFfnRecipe(const CudaTransformerWinner::FfnRecipe& recipe) {
  using namespace CudaTransformerWinner;
  testAssert(recipe.rmsNorm == RmsNormTactic::Sm120C256Warp4Vec8);
  testAssert(recipe.dualFfn == DualFfnTactic::Sm120C256F768M128N64K32S3Sw4);
  testAssert(recipe.downProjection == ResidualTactic::Sm120M128N128K32S3Sw1);
}

static void assertB32DynamicAttentionRecipe(
  const CudaTransformerWinner::AttentionRecipe& recipe
) {
  using namespace CudaTransformerWinner;
  testAssert(recipe.planarQkv == PlanarQkvTactic::CublasHgemmStridedBatchedSquare);
  testAssert(recipe.rmsNorm == RmsNormTactic::Sm120C256Warp4Vec8);
  testAssert(recipe.rope == RopeTactic::LearnedHalf2);
  testAssert(recipe.qkvRope == QkvRopeTactic::Disabled);
  testAssert(recipe.attention == AttentionTactic::Generic);
  testAssert(recipe.outProjection == ResidualTactic::Sm120M128N128K32S3Sw1);
}

static void assertC256StaticAttentionRecipe(
  const CudaTransformerWinner::AttentionRecipe& recipe
) {
  using namespace CudaTransformerWinner;
  testAssert(recipe.planarQkv == PlanarQkvTactic::CublasHgemmStridedBatchedSquare);
  testAssert(recipe.rmsNorm == RmsNormTactic::Sm120C256Warp4Vec8);
  testAssert(recipe.rope == RopeTactic::LearnedHalf2);
  testAssert(recipe.qkvRope == QkvRopeTactic::Disabled);
  testAssert(recipe.attention == AttentionTactic::Generic);
  testAssert(recipe.outProjection == ResidualTactic::CublasHgemmBetaOne);
}

static void assertC256StaticFfnRecipe(const CudaTransformerWinner::FfnRecipe& recipe) {
  using namespace CudaTransformerWinner;
  testAssert(recipe.rmsNorm == RmsNormTactic::Sm120C256Warp4Vec8);
  testAssert(recipe.dualFfn == DualFfnTactic::Disabled);
  testAssert(recipe.downProjection == ResidualTactic::CublasHgemmBetaOne);
}

static void assertMaskSafeAttentionRecipe(
  const CudaTransformerWinner::AttentionRecipe& recipe
) {
  using namespace CudaTransformerWinner;
  testAssert(recipe.planarQkv == PlanarQkvTactic::CublasHgemmStridedBatchedSquare);
  testAssert(recipe.rmsNorm == RmsNormTactic::GenericHalf);
  testAssert(recipe.rope == RopeTactic::LearnedHalf2);
  testAssert(recipe.qkvRope == QkvRopeTactic::Disabled);
  testAssert(recipe.attention == AttentionTactic::Generic);
  testAssert(recipe.outProjection == ResidualTactic::GenericAdd);
}

static void assertGenericFfnRecipe(const CudaTransformerWinner::FfnRecipe& recipe) {
  using namespace CudaTransformerWinner;
  testAssert(recipe.rmsNorm == RmsNormTactic::GenericHalf);
  testAssert(recipe.dualFfn == DualFfnTactic::Disabled);
  testAssert(recipe.downProjection == ResidualTactic::CublasHgemmBetaOne);
}

static void assertDisabledFfnRecipe(const CudaTransformerWinner::FfnRecipe& recipe) {
  using namespace CudaTransformerWinner;
  testAssert(recipe.rmsNorm == RmsNormTactic::GenericHalf);
  testAssert(recipe.dualFfn == DualFfnTactic::Disabled);
  testAssert(recipe.downProjection == ResidualTactic::GenericAdd);
}

static void assertWideAttentionRecipe(
  const CudaTransformerWinner::AttentionRecipe& recipe
) {
  using namespace CudaTransformerWinner;
  testAssert(recipe.planarQkv == PlanarQkvTactic::CublasHgemmStridedBatchedSquare);
  testAssert(recipe.rmsNorm == RmsNormTactic::GenericHalf);
  testAssert(recipe.rope == RopeTactic::LearnedHalf2);
  testAssert(recipe.qkvRope == QkvRopeTactic::Disabled);
  testAssert(recipe.attention == AttentionTactic::Generic);
  testAssert(recipe.outProjection == ResidualTactic::CublasHgemmBetaOne);
}

template<typename AttentionAssertion, typename FfnAssertion>
static void assertAllTransformerRecipes(
  const ArchitectureDesc& architecture,
  const CudaTransformerWinner::PreparedPlan& plan,
  int expectedLayers,
  bool expectFfnFound,
  AttentionAssertion assertAttention,
  FfnAssertion assertFfn
) {
  testAssert(plan.architecture == architecture.signature);
  testAssert(plan.records.size() == architecture.operators.size());
  int attentionCount = 0;
  int ffnCount = 0;
  for(const ArchitectureOpDesc& op: architecture.operators) {
    const CudaTransformerWinner::PreparedRecord& record = plan.records[op.topologyIndex];
    testAssert(record.request.topologyIndex == op.topologyIndex);
    testAssert(record.request.architecture == architecture.signature);
    if(op.kind == ArchitectureOpKind::TransformerAttention) {
      testAssert(record.found);
      assertAttention(plan.attentionFor(op.topologyIndex));
      attentionCount += 1;
    }
    else if(op.kind == ArchitectureOpKind::TransformerFFN) {
      testAssert(record.found == expectFfnFound);
      assertFfn(plan.ffnFor(op.topologyIndex));
      ffnCount += 1;
    }
  }
  testAssert(attentionCount == expectedLayers);
  testAssert(ffnCount == expectedLayers);
}

static const CudaTransformerWinner::PreparedRecord& firstProductionRecord(
  const CudaTransformerWinner::PreparedPlan& plan,
  ArchitectureOpKind kind
) {
  for(const CudaTransformerWinner::PreparedRecord& record: plan.records) {
    if(record.request.key.kind == kind)
      return record;
  }
  testAssert(false);
  return plan.records[0];
}

}  // namespace

void Tests::runTransformerProductionPlanTests() {
  FixtureTacticData generic{FixtureTacticKind::Generic,1};
  FixtureTacticData attentionWinner{FixtureTacticKind::Renju15AttentionB36S2,2};
  FixtureTacticData ffnWinner{FixtureTacticKind::Renju15FFNB36S2,3};
  Registry registry;
  registry.registerTactic(registration(1,0,"generic-compatible-v1",&generic));
  registry.registerTactic(registration(
    2,100,"renju15-sm120-b36-s2-attention-fa4-tn128-qkv-rope-s3-v1",&attentionWinner
  ));
  registry.registerTactic(registration(
    3,100,"renju15-sm120-b36-s2-ffn-m128-n64-k32-s3-sw4-v1",&ffnWinner
  ));

  const RuntimeOpContext b36 = runtimeContext(36,15,15,MaskMode::None);
  ModelDesc weightsA = makeModel(24,256,768,8,32,0.1f,"weights-a");
  ModelDesc weightsB = makeModel(24,256,768,8,32,0.9f,"weights-b-different-file");
  weightsA.onnxHeader.model_config_sha256 = "export-config-a";
  weightsB.onnxHeader.model_config_sha256 = "export-config-b";
  ArchitectureDesc architectureA = buildArchitectureDesc(weightsA);
  ArchitectureDesc architectureB = buildArchitectureDesc(weightsB);
  testAssert(architectureA.signature == architectureB.signature);
  testAssert(architectureA.canonicalEncoding == architectureB.canonicalEncoding);
  testAssert(getModelProvenance(weightsA).modelName != getModelProvenance(weightsB).modelName);
  testAssert(getModelProvenance(weightsA).artifactSha256 != getModelProvenance(weightsB).artifactSha256);

  PlanSnapshot planA = preparePlan(architectureA,b36,registry);
  PlanSnapshot planB = preparePlan(architectureB,b36,registry);
  testAssert(planA.requests.size() == planB.requests.size());
  for(size_t i = 0; i < planA.requests.size(); i++) {
    testAssert(planA.requests[i].key == planB.requests[i].key);
    assertPreparedIdentity(planA.prepared[i],planB.prepared[i]);
  }
  testAssert(planA.fingerprint == planB.fingerprint);

  // Deeper topology changes the whole-model identity and whole plan, while a
  // repeated local operator keeps the same capability/tactic/recipe.
  ModelDesc layers48 = makeModel(48,256,768,8,32,0.3f,"layers-48");
  ArchitectureDesc architecture48 = buildArchitectureDesc(layers48);
  PlanSnapshot plan48 = preparePlan(architecture48,b36,registry);
  testAssert(architectureA.signature != architecture48.signature);
  testAssert(planA.fingerprint != plan48.fingerprint);
  size_t attention24 = findRequest(planA,ArchitectureOpKind::TransformerAttention,256,256,16);
  size_t attention48 = findRequest(plan48,ArchitectureOpKind::TransformerAttention,256,256,16);
  size_t ffn24 = findRequest(planA,ArchitectureOpKind::TransformerFFN,256,256,768);
  size_t ffn48 = findRequest(plan48,ArchitectureOpKind::TransformerFFN,256,256,768);
  testAssert(planA.requests[attention24].key == plan48.requests[attention48].key);
  testAssert(planA.requests[ffn24].key == plan48.requests[ffn48].key);
  assertPreparedIdentity(planA.prepared[attention24],plan48.prepared[attention48]);
  assertPreparedIdentity(planA.prepared[ffn24],plan48.prepared[ffn48]);

  // Every runtime perturbation must alter only keys that explicitly declared
  // that dependency. Certified B36/S2 transformer tactics then fail closed to
  // the compatible implementation, without disturbing unrelated operators.
  RuntimeOpContext b32 = runtimeContext(32,15,15,MaskMode::None);
  assertRuntimeKeyLocalization(architectureA,b36,b32,OP_RUNTIME_BATCH);
  PlanSnapshot planB32 = preparePlan(architectureA,b32,registry);
  assertOnlyTransformerWinnerFallsBack(architectureA,planA,planB32);

  RuntimeOpContext board19 = runtimeContext(36,19,19,MaskMode::None);
  assertRuntimeKeyLocalization(
    architectureA,b36,board19,OP_RUNTIME_SPATIAL_AREA | OP_RUNTIME_SPATIAL_XY
  );
  PlanSnapshot planBoard19 = preparePlan(architectureA,board19,registry);
  assertOnlyTransformerWinnerFallsBack(architectureA,planA,planBoard19);

  RuntimeOpContext masked = runtimeContext(36,15,15,MaskMode::Dense);
  assertRuntimeKeyLocalization(architectureA,b36,masked,OP_RUNTIME_MASK);
  PlanSnapshot planMasked = preparePlan(architectureA,masked,registry);
  assertOnlyTransformerWinnerFallsBack(architectureA,planA,planMasked);

  // C384/H12/D32/F1024 is shape-compatible with generic CUDA only. The two
  // specialized transformer tactics must reject it, while independent head
  // operators such as value MatMul 64->3 and ownership Conv 96->1 retain their
  // exact capability key and compatible recipe.
  ModelDesc wideModel = makeModel(32,384,1024,12,32,0.5f,"c384-h12-f1024");
  ArchitectureDesc wideArchitecture = buildArchitectureDesc(wideModel);
  PlanSnapshot widePlan = preparePlan(wideArchitecture,b36,registry);
  size_t wideAttention = findRequest(
    widePlan,ArchitectureOpKind::TransformerAttention,384,384,16
  );
  size_t wideFFN = findRequest(
    widePlan,ArchitectureOpKind::TransformerFFN,384,384,1024
  );
  testAssert(widePlan.prepared[wideAttention].support == SupportClass::CompatibleOnly);
  testAssert(widePlan.prepared[wideFFN].support == SupportClass::CompatibleOnly);
  size_t baseV3 = findRequest(planA,ArchitectureOpKind::MatMul,64,3);
  size_t wideV3 = findRequest(widePlan,ArchitectureOpKind::MatMul,64,3);
  size_t baseOwnership = findRequest(planA,ArchitectureOpKind::Conv2D,96,1);
  size_t wideOwnership = findRequest(widePlan,ArchitectureOpKind::Conv2D,96,1);
  testAssert(planA.requests[baseV3].key == widePlan.requests[wideV3].key);
  testAssert(planA.requests[baseOwnership].key == widePlan.requests[wideOwnership].key);
  assertPreparedIdentity(planA.prepared[baseV3],widePlan.prepared[wideV3]);
  assertPreparedIdentity(planA.prepared[baseOwnership],widePlan.prepared[wideOwnership]);

  // Exercise the real immutable production planner, not merely the registry
  // fixture above. G1 requires the complete measured winner recipe on every
  // one of the 24 attention and 24 FFN blocks.
  const CudaTransformerWinner::DeviceCapability device = sm120Device();
  CudaTransformerWinner::PreparedPlan productionG1A =
    CudaTransformerWinner::preparePlan(architectureA,b36,device);
  CudaTransformerWinner::PreparedPlan productionG1B =
    CudaTransformerWinner::preparePlan(architectureB,b36,device);
  assertAllTransformerRecipes(
    architectureA,productionG1A,24,true,assertExactAttentionRecipe,assertExactFfnRecipe
  );
  for(const CudaTransformerWinner::PreparedRecord& record: productionG1A.records) {
    if(record.request.key.kind == ArchitectureOpKind::TransformerAttention) {
      testAssert(record.operation.support == SupportClass::CertifiedFast);
      testAssert(string(CudaTransformerWinner::tacticName(record.operation.tactic)) ==
        "attention-c256-b36-fa4-sm120");
    }
    else if(record.request.key.kind == ArchitectureOpKind::TransformerFFN) {
      testAssert(record.found);
      testAssert(record.operation.support == SupportClass::CompatibleOnly);
      testAssert(string(CudaTransformerWinner::tacticName(record.operation.tactic)) ==
        "ffn-c256-f768-dynamic-sm120");
    }
    else
      testAssert(!record.found);
  }

  // Artifact name, file SHA, export-config SHA, and all tensor values differ;
  // architecture, each local key/recipe/tactic, and the whole plan do not.
  assertProductionPlanIdentity(productionG1A,productionG1B);

  // The operator recipe is independently certified on each owned handle;
  // evaluator S1 versus S2 is an outer throughput topology, not a kernel ABI.
  RuntimeOpContext b36S1 = b36;
  b36S1.streamCount = 1;
  CudaTransformerWinner::PreparedPlan productionG1S1 =
    CudaTransformerWinner::preparePlan(architectureA,b36S1,device);
  assertExactAttentionRecipe(productionG1S1.attentionFor(
    architectureA.operators[attention24].topologyIndex));
  testAssert(productionG1S1.records[attention24].operation.support ==
    SupportClass::CertifiedFast);

  // Runtime-library versions are diagnostic identity, not a tactic gate. If
  // this binary loaded and the exact SM120/resource/shape predicates match,
  // the same embedded kernel is selected across compatible driver/library
  // maintenance versions.
  CudaTransformerWinner::DeviceCapability differentAbi = device;
  differentAbi.cudnnVersion = 91401;
  RuntimeOpContext differentAbiRuntime = b36;
  differentAbiRuntime.runtimeLibraryFingerprint =
    CudaTransformerWinner::makeRuntimeLibraryFingerprint(
      13000,13020,130101,91401);
  CudaTransformerWinner::PreparedPlan productionDifferentAbi =
    CudaTransformerWinner::preparePlan(
      architectureA,differentAbiRuntime,differentAbi);
  const CudaTransformerWinner::AttentionRecipe differentAbiRecipe =
    productionDifferentAbi.attentionFor(
      architectureA.operators[attention24].topologyIndex);
  assertExactAttentionRecipe(differentAbiRecipe);

  CudaTransformerWinner::DeviceCapability unknownAbi = device;
  unknownAbi.cudaRuntimeVersion = 0;
  unknownAbi.cudaDriverVersion = 0;
  unknownAbi.cublasVersion = 0;
  unknownAbi.cudnnVersion = 0;
  RuntimeOpContext fakeFingerprintRuntime = b36;
  fakeFingerprintRuntime.runtimeLibraryFingerprint = 1;
  CudaTransformerWinner::PreparedPlan productionUnknownAbi =
    CudaTransformerWinner::preparePlan(
      architectureA,fakeFingerprintRuntime,unknownAbi);
  assertExactAttentionRecipe(productionUnknownAbi.attentionFor(
    architectureA.operators[attention24].topologyIndex));

  // G5: depth changes whole-model/whole-plan identity. All 48 repeated local
  // blocks still select the same recipes and exact local prepared operations.
  CudaTransformerWinner::PreparedPlan productionG5 =
    CudaTransformerWinner::preparePlan(architecture48,b36,device);
  assertAllTransformerRecipes(
    architecture48,productionG5,48,true,assertExactAttentionRecipe,assertExactFfnRecipe
  );
  testAssert(productionG1A.architecture != productionG5.architecture);
  testAssert(productionG1A.fingerprint != productionG5.fingerprint);
  const CudaTransformerWinner::PreparedRecord& g1Attention = firstProductionRecord(
    productionG1A,ArchitectureOpKind::TransformerAttention
  );
  const CudaTransformerWinner::PreparedRecord& g5Attention = firstProductionRecord(
    productionG5,ArchitectureOpKind::TransformerAttention
  );
  const CudaTransformerWinner::PreparedRecord& g1Ffn = firstProductionRecord(
    productionG1A,ArchitectureOpKind::TransformerFFN
  );
  const CudaTransformerWinner::PreparedRecord& g5Ffn = firstProductionRecord(
    productionG5,ArchitectureOpKind::TransformerFFN
  );
  testAssert(g1Attention.request.key == g5Attention.request.key);
  testAssert(g1Ffn.request.key == g5Ffn.request.key);
  assertPreparedOpIdentity(g1Attention.operation,g5Attention.operation);
  assertPreparedOpIdentity(g1Ffn.operation,g5Ffn.operation);

  // G2: S361 invalidates only fixed/dynamic-M kernels. The C256 RMS, planar
  // QKV, learned half2 RoPE, and cuBLAS beta-one residual pieces remain active.
  CudaTransformerWinner::PreparedPlan productionG2 =
    CudaTransformerWinner::preparePlan(architectureA,board19,device);
  assertAllTransformerRecipes(
    architectureA,productionG2,24,true,
    assertC256StaticAttentionRecipe,assertC256StaticFfnRecipe
  );

  // G3: a dense mask retains only mask-safe planar QKV and learned-half2 RoPE.
  // The no-mask RMS/FA/fused-QKV/beta-one/dual/down tactics all fail closed.
  CudaTransformerWinner::PreparedPlan productionG3 =
    CudaTransformerWinner::preparePlan(architectureA,masked,device);
  assertAllTransformerRecipes(
    architectureA,productionG3,24,false,
    assertMaskSafeAttentionRecipe,assertDisabledFfnRecipe
  );

  // G4: B32*225 is a validated dynamic-M row count. It reuses C256 RMS,
  // learned RoPE, both residual s3 kernels, and dual FFN, but not B36-only FA4
  // or the B36-only fused QKV+RoPE recipe.
  CudaTransformerWinner::PreparedPlan productionG4 =
    CudaTransformerWinner::preparePlan(architectureA,b32,device);
  assertAllTransformerRecipes(
    architectureA,productionG4,24,true,
    assertB32DynamicAttentionRecipe,assertExactFfnRecipe
  );

  // G6: C384/H12/D32/F1024 stays on the generic-safe production path while
  // retaining geometry-general planar QKV, learned half2 RoPE, and beta-one
  // residual GEMMs. No C256/H8/F768-only component may leak into this plan.
  CudaTransformerWinner::PreparedPlan productionG6 =
    CudaTransformerWinner::preparePlan(wideArchitecture,b36,device);
  assertAllTransformerRecipes(
    wideArchitecture,productionG6,32,true,assertWideAttentionRecipe,assertGenericFfnRecipe
  );

  cout << "Production PreparedPlan fingerprints:" << endl;
  cout << "  G1=" << productionG1A.fingerprint.toHex() << endl;
  cout << "  G2-S361=" << productionG2.fingerprint.toHex() << endl;
  cout << "  G3-dense-mask=" << productionG3.fingerprint.toHex() << endl;
  cout << "  G4-B32=" << productionG4.fingerprint.toHex() << endl;
  cout << "  G5-48-layers=" << productionG5.fingerprint.toHex() << endl;
  cout << "  G6-C384-H12-F1024=" << productionG6.fingerprint.toHex() << endl;

  map<ArchitectureOpKind,int> reusableByKind;
  map<ArchitectureOpKind,int> changedByKind;
  multiset<string> baseKeys;
  for(const OpRequest& request: planA.requests) {
    const CapabilityKey& key = request.key;
    baseKeys.insert(
      Global::uint64ToString((uint64_t)key.kind) + ":" +
      Global::intToString(key.inChannels) + ":" +
      Global::intToString(key.outChannels) + ":" +
      Global::intToString(key.auxiliaryChannels) + ":" +
      Global::intToString(key.numHeads) + ":" +
      Global::intToString(key.qHeadDim)
    );
  }
  for(const OpRequest& request: widePlan.requests) {
    const CapabilityKey& key = request.key;
    const string compact =
      Global::uint64ToString((uint64_t)key.kind) + ":" +
      Global::intToString(key.inChannels) + ":" +
      Global::intToString(key.outChannels) + ":" +
      Global::intToString(key.auxiliaryChannels) + ":" +
      Global::intToString(key.numHeads) + ":" +
      Global::intToString(key.qHeadDim);
    auto found = baseKeys.find(compact);
    if(found != baseKeys.end()) {
      reusableByKind[key.kind] += 1;
      baseKeys.erase(found);
    }
    else
      changedByKind[key.kind] += 1;
  }
  cout << "C384/H12/D32/F1024 reuse/fallback matrix:" << endl;
  for(ArchitectureOpKind kind: {
        ArchitectureOpKind::Conv2D,
        ArchitectureOpKind::BatchNormActivation,
        ArchitectureOpKind::MatMul,
        ArchitectureOpKind::MatBias,
        ArchitectureOpKind::TransformerAttention,
        ArchitectureOpKind::TransformerFFN,
      }) {
    cout << "  " << opKindName(kind)
         << " reusable=" << reusableByKind[kind]
         << " changed-or-fallback=" << changedByKind[kind] << endl;
  }
  cout << "Transformer production-plan CPU contract tests passed" << endl;
}

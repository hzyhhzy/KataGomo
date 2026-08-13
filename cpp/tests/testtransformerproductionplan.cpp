#include "../tests/tests.h"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "../core/test.h"
#include "../core/config_parser.h"
#include "../neuralnet/activations.h"
#include "../neuralnet/architecturedesc.h"
#include "../neuralnet/cuda_specialized/sm120/c384/fixed_batch/plan.h"
#include "../neuralnet/cuda_specialized/sm120/c384/fixed_batch/weights.h"
#include "../neuralnet/cudabackend_transformer_winner.h"
#include "../neuralnet/cudaopregistry.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/int8policy.h"

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
  desc->ropeFreqs.assign(
    (size_t)desc->ropeNumKVHeads * desc->ropeNumPairs * 2,
    weight + 0.04f);
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
  model.trunk.regularNumChannels = 192;
  model.trunk.dilatedNumChannels = 64;
  model.trunk.gpoolNumChannels = 64;
  setConv(model.trunk.initialConv,22,channels,3,weightSeed);
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
  setMatMul(model.valueHead.v2Mul,96 * 3,128,weightSeed);
  setMatBias(model.valueHead.v2Bias,128,weightSeed);
  setActivation(model.valueHead.v2Activation,ACTIVATION_SILU,weightSeed);
  setMatMul(model.valueHead.v3Mul,128,3,weightSeed);
  setMatBias(model.valueHead.v3Bias,3,weightSeed);
  setMatMul(model.valueHead.sv3Mul,128,6,weightSeed);
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

static void assertC384DynamicAttentionRecipe(
  const CudaTransformerWinner::AttentionRecipe& recipe
) {
  using namespace CudaTransformerWinner;
  testAssert(recipe.planarQkv == PlanarQkvTactic::CublasHgemmStridedBatchedSquare);
  testAssert(recipe.rmsNorm == RmsNormTactic::Sm120C384Warp4Vec4x3);
  testAssert(recipe.rope == RopeTactic::LearnedHalf2);
  testAssert(recipe.qkvRope == QkvRopeTactic::Disabled);
  testAssert(recipe.attention == AttentionTactic::Generic);
  testAssert(recipe.outProjection == ResidualTactic::CublasHgemmBetaOne);
}

static void assertC384DynamicFfnRecipe(
  const CudaTransformerWinner::FfnRecipe& recipe
) {
  using namespace CudaTransformerWinner;
  testAssert(recipe.rmsNorm == RmsNormTactic::Sm120C384Warp4Vec4x3);
  testAssert(recipe.dualFfn == DualFfnTactic::Sm120C384F1024M128N64K32S3Sw4);
  testAssert(recipe.downProjection == ResidualTactic::CublasHgemmBetaOne);
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

static void refreshProductionPlanFingerprint(
  CudaTransformerWinner::PreparedPlan& plan
) {
  vector<OpRequest> requests;
  vector<PreparedOp> prepared;
  requests.reserve(plan.records.size());
  prepared.reserve(plan.records.size());
  for(const CudaTransformerWinner::PreparedRecord& record: plan.records) {
    requests.push_back(record.request);
    prepared.push_back(record.operation);
  }
  plan.fingerprint = fingerprintPreparedPlan(requests,prepared);
}

static CudaTransformerWinner::C384RuntimePiecePolicy c384TestPiecePolicy(
  uint32_t conservativeMin,
  uint32_t conservativeMax,
  uint32_t twoLaneMin,
  uint32_t twoLaneMax
) {
  CudaTransformerWinner::C384RuntimePiecePolicy policy;
  policy.conservative.minInclusive = conservativeMin;
  policy.conservative.maxInclusive = conservativeMax;
  policy.exactlyTwoSameGpuLanes.minInclusive = twoLaneMin;
  policy.exactlyTwoSameGpuLanes.maxInclusive = twoLaneMax;
  return policy;
}

static void assertC384RuntimeGateContract() {
  using CudaTransformerWinner::C384RuntimeGatePolicy;
  using CudaTransformerWinner::C384RuntimePiece;
  using CudaTransformerWinner::shouldUseC384RuntimePiece;

  C384RuntimeGatePolicy disabled;
  testAssert(!shouldUseC384RuntimePiece(C384RuntimePiece::DualFfn,225,1,disabled));

  // These values are deliberately synthetic CPU-test boundaries, not tuned
  // production thresholds. Final values are supplied only after balanced GPU
  // measurements. The contract under test is actual-row and same-GPU-lane
  // dispatch, independent of the handle's configured maximum batch.
  C384RuntimeGatePolicy policy;
  policy.rowsPerBatch = 225;
  policy.rmsNorm = c384TestPiecePolicy(4,64,2,128);
  policy.dualFfn = c384TestPiecePolicy(1,128,1,128);
  policy.outProjection = c384TestPiecePolicy(16,64,12,64);
  policy.downProjection = c384TestPiecePolicy(24,64,16,64);

  // Invalid/tail row counts fail closed before any launch is enqueued.
  testAssert(!shouldUseC384RuntimePiece(C384RuntimePiece::DualFfn,0,1,policy));
  testAssert(!shouldUseC384RuntimePiece(C384RuntimePiece::DualFfn,-225,1,policy));
  testAssert(!shouldUseC384RuntimePiece(C384RuntimePiece::DualFfn,226,1,policy));
  testAssert(!shouldUseC384RuntimePiece(C384RuntimePiece::DualFfn,225,0,policy));

  // Single-lane boundaries use only actualRows / 225.
  testAssert(!shouldUseC384RuntimePiece(C384RuntimePiece::RmsNorm,3*225,1,policy));
  testAssert(shouldUseC384RuntimePiece(C384RuntimePiece::RmsNorm,4*225,1,policy));
  testAssert(shouldUseC384RuntimePiece(C384RuntimePiece::RmsNorm,64*225,1,policy));
  testAssert(!shouldUseC384RuntimePiece(C384RuntimePiece::RmsNorm,65*225,1,policy));
  testAssert(shouldUseC384RuntimePiece(C384RuntimePiece::DualFfn,225,1,policy));

  // Exactly two same-GPU lanes use independently measured boundaries.
  testAssert(!shouldUseC384RuntimePiece(C384RuntimePiece::RmsNorm,225,2,policy));
  testAssert(shouldUseC384RuntimePiece(C384RuntimePiece::RmsNorm,2*225,2,policy));
  testAssert(!shouldUseC384RuntimePiece(C384RuntimePiece::OutProjection,11*225,2,policy));
  testAssert(shouldUseC384RuntimePiece(C384RuntimePiece::OutProjection,12*225,2,policy));
  testAssert(!shouldUseC384RuntimePiece(C384RuntimePiece::DownProjection,15*225,2,policy));
  testAssert(shouldUseC384RuntimePiece(C384RuntimePiece::DownProjection,16*225,2,policy));

  // Three or more lanes are unmeasured and therefore reuse the conservative
  // range rather than accidentally inheriting the two-lane result.
  testAssert(!shouldUseC384RuntimePiece(C384RuntimePiece::RmsNorm,3*225,3,policy));
  testAssert(shouldUseC384RuntimePiece(C384RuntimePiece::RmsNorm,4*225,3,policy));
  testAssert(!shouldUseC384RuntimePiece(C384RuntimePiece::OutProjection,15*225,4,policy));
  testAssert(shouldUseC384RuntimePiece(C384RuntimePiece::OutProjection,16*225,4,policy));
}

static void assertC384ProductionRuntimeGatePolicy() {
  using CudaTransformerWinner::C384RuntimeBatchRange;
  using CudaTransformerWinner::C384RuntimeGatePolicy;
  using CudaTransformerWinner::C384RuntimePiece;
  using CudaTransformerWinner::c384MeasuredGenericActiveMarker;
  using CudaTransformerWinner::fingerprintRecipeWithC384RuntimeGate;
  using CudaTransformerWinner::productionC384RuntimeGatePolicy;
  using CudaTransformerWinner::shouldUseC384RuntimePiece;

  const C384RuntimeGatePolicy& policy = productionC384RuntimeGatePolicy();
  testAssert(policy.rowsPerBatch == 225);

  // The dual FFN kernel is selected for every valid actual batch, independent
  // of the handle's configured maximum and of measured evaluator topology.
  for(int concurrency: {1,2,3,4}) {
    testAssert(shouldUseC384RuntimePiece(
      C384RuntimePiece::DualFfn,225,concurrency,policy));
    testAssert(shouldUseC384RuntimePiece(
      C384RuntimePiece::DualFfn,128*225,concurrency,policy));
    testAssert(shouldUseC384RuntimePiece(
      C384RuntimePiece::DualFfn,129*225,concurrency,policy));
    testAssert(shouldUseC384RuntimePiece(
      C384RuntimePiece::DualFfn,4660*225,concurrency,policy));
    testAssert(!shouldUseC384RuntimePiece(
      C384RuntimePiece::DualFfn,4661*225,concurrency,policy));
  }

  // Evaluator-local lane counts cannot see a second evaluator sharing this
  // GPU. B1 therefore always uses the topology-safe dual-only policy; RMS is
  // selected from actual B2 upward for every lane topology.
  for(int concurrency: {1,2,3,4}) {
    testAssert(!shouldUseC384RuntimePiece(
      C384RuntimePiece::RmsNorm,225,concurrency,policy));
    testAssert(shouldUseC384RuntimePiece(
      C384RuntimePiece::RmsNorm,2*225,concurrency,policy));
    testAssert(shouldUseC384RuntimePiece(
      C384RuntimePiece::RmsNorm,128*225,concurrency,policy));
  }

  // Residual CUTLASS tactics did not clear the evidence margin, so both pieces
  // deterministically use the generic beta-one path without preparing an
  // unused specialized handle.
  for(int concurrency: {1,2,3}) {
    testAssert(!shouldUseC384RuntimePiece(
      C384RuntimePiece::OutProjection,225,concurrency,policy));
    testAssert(!shouldUseC384RuntimePiece(
      C384RuntimePiece::OutProjection,128*225,concurrency,policy));
    testAssert(!shouldUseC384RuntimePiece(
      C384RuntimePiece::DownProjection,225,concurrency,policy));
    testAssert(!shouldUseC384RuntimePiece(
      C384RuntimePiece::DownProjection,128*225,concurrency,policy));
  }
  testAssert(string(c384MeasuredGenericActiveMarker(
    C384RuntimePiece::OutProjection)) ==
    "KATAGO_C384_MEASURED_GENERIC_ACTIVE piece=out-proj reason=residual-not-beneficial");
  testAssert(string(c384MeasuredGenericActiveMarker(
    C384RuntimePiece::DownProjection)) ==
    "KATAGO_C384_MEASURED_GENERIC_ACTIVE piece=ffn-down reason=residual-not-beneficial");
  testAssert(c384MeasuredGenericActiveMarker(C384RuntimePiece::RmsNorm) == nullptr);
  testAssert(c384MeasuredGenericActiveMarker(C384RuntimePiece::DualFfn) == nullptr);

  // Nonintegral row counts and invalid topology always fail closed before a
  // specialized launch can enqueue work.
  testAssert(!shouldUseC384RuntimePiece(C384RuntimePiece::DualFfn,226,1,policy));
  testAssert(!shouldUseC384RuntimePiece(C384RuntimePiece::RmsNorm,224,2,policy));
  testAssert(!shouldUseC384RuntimePiece(C384RuntimePiece::DualFfn,225,0,policy));

  // Numeric gate fields, including a disabled range, are canonical recipe
  // identity rather than an informal version label in a tactic string.
  const RecipeFingerprint canonical =
    fingerprintRecipeWithC384RuntimeGate("c384-test-tactic",policy);
  const RecipeFingerprint canonicalCopy =
    fingerprintRecipeWithC384RuntimeGate("c384-test-tactic",policy);
  testAssert(canonical == canonicalCopy);
  C384RuntimeGatePolicy changed = policy;
  changed.rmsNorm.conservative.minInclusive += 1;
  testAssert(canonical !=
    fingerprintRecipeWithC384RuntimeGate("c384-test-tactic",changed));
  changed = policy;
  changed.outProjection.conservative = C384RuntimeBatchRange{1,1};
  testAssert(canonical !=
    fingerprintRecipeWithC384RuntimeGate("c384-test-tactic",changed));
}

}  // namespace

void Tests::runTransformerProductionPlanTests() {
  // Fixed-exact C384 AOT search contracts are deliberately independent of
  // the production dynamic-M plan above. The checked-in CUDA registry is
  // empty, while generated search providers use this CPU selector unchanged.
  {
    using namespace C384ExactFixedAot;
    const TacticKey qkvTactics[] = {
      {Family::QkvRope,28,6300,0,"qkv-rope-b28-test",true,false,kRegistryAbiVersion},
      {Family::QkvRope,24,5400,0,"qkv-rope-b24-test",true,false,kRegistryAbiVersion},
    };
    const TacticKey dualTactics[] = {
      {Family::DualFfn,28,6300,170,"dual-b28-grid170-test",false,true,kRegistryAbiVersion},
      {Family::DualFfn,28,6300,340,"dual-b28-grid340-test",false,true,kRegistryAbiVersion},
      {Family::DualFfn,24,5400,170,"dual-b24-grid170-test",false,true,kRegistryAbiVersion},
    };
    const RegistryView registry = {
      {qkvTactics,sizeof(qkvTactics) / sizeof(qkvTactics[0]),sizeof(TacticKey)},
      {dualTactics,sizeof(dualTactics) / sizeof(dualTactics[0]),sizeof(TacticKey)},
    };
    RuntimeShape shape;
    shape.modelDepth = 36;
    shape.attentionBlockCount = 36;
    shape.ffnBlockCount = 36;
    shape.alternatingAttentionFfn = true;
    shape.batchSize = 28;
    shape.enqueuedRows = 6300;
    shape.boardX = 15;
    shape.boardY = 15;
    shape.sequenceLength = 225;
    shape.channels = 384;
    shape.numHeads = 12;
    shape.numKvHeads = 12;
    shape.qHeadDim = 32;
    shape.vHeadDim = 32;
    shape.ffnChannels = 1024;
    shape.ropePairsTotal = 192;
    shape.deviceOrdinal = 0;
    shape.computeCapability = 120;
    shape.usingFp16 = true;
    shape.usingNhwc = true;
    shape.exactNoMask = true;
    shape.learnedRope = true;
    shape.swiglu = true;
    PreparedPackedFa4 fa4;
    fa4.abiVersion = kPackedFa4ProofAbiVersion;
    fa4.batchSize = 28;
    fa4.sequenceLength = 225;
    fa4.numHeads = 12;
    fa4.numKvHeads = 12;
    fa4.qHeadDim = 32;
    fa4.vHeadDim = 32;
    fa4.deviceOrdinal = 0;
    fa4.acceptsPackedTokenQkv = true;
    fa4.id = "fa4-packed-b28-test";
    fa4.implementationCookie = 1;

    std::size_t candidateCount = 0;
    const int* batches = candidateBatches(candidateCount);
    testAssert(candidateCount == 2);
    testAssert(batches[0] == 28 && batches[1] == 24);
    testAssert(candidateBatchPriority(28) == 0);
    testAssert(candidateBatchPriority(24) == 1);
    testAssert(candidateBatchPriority(40) == -1);
    testAssert(candidateBatchPriority(36) == -1);
    // B24 remains searchable diagnostic evidence, but production is locked to
    // exact B28/M6300 and must reject every tail or neighboring batch.
    testAssert(kProductionBatch == 28);
    testAssert(kProductionTokenRows == 6300);
    testAssert(productionBatchEligible(28,6300));
    testAssert(!productionBatchEligible(24,5400));
    testAssert(!productionBatchEligible(28,6299));

    Selection selected = select(
      shape,registry,"qkv-rope-b28-test","dual-b28-grid170-test",&fa4);
    testAssert(selected.targetShape);
    testAssert(selected.qkvRope.selected());
    testAssert(selected.packedFa4 == &fa4);
    testAssert(selected.dualFfn.selected());

    const RegistryView emptyRegistry{};
    selected = select(
      shape,emptyRegistry,"qkv-rope-b28-test","dual-b28-grid170-test",&fa4);
    testAssert(selected.targetShape);
    testAssert(selected.qkvRope.reason == RejectReason::RegistryMiss);
    testAssert(selected.dualFfn.reason == RejectReason::RegistryMiss);

    const TacticKey duplicateQkv[] = {
      {Family::QkvRope,28,6300,0,"duplicate",true,false,kRegistryAbiVersion},
      {Family::QkvRope,28,6300,0,"duplicate",true,false,kRegistryAbiVersion},
    };
    const RegistryView duplicateRegistry = {
      {duplicateQkv,2,sizeof(TacticKey)},
      {dualTactics,sizeof(dualTactics) / sizeof(dualTactics[0]),sizeof(TacticKey)},
    };
    selected = select(
      shape,duplicateRegistry,"duplicate","dual-b28-grid170-test",&fa4);
    testAssert(selected.qkvRope.reason == RejectReason::InvalidRegistry);
    testAssert(selected.dualFfn.selected());

    // A generated record with stale exact-M metadata invalidates its family
    // registry rather than merely becoming an unselectable search candidate.
    const TacticKey staleRowsQkv[] = {
      {Family::QkvRope,28,6299,0,"stale-rows",true,false,kRegistryAbiVersion},
    };
    const RegistryView staleRowsRegistry = {
      {staleRowsQkv,1,sizeof(TacticKey)},
      {dualTactics,sizeof(dualTactics) / sizeof(dualTactics[0]),sizeof(TacticKey)},
    };
    selected = select(
      shape,staleRowsRegistry,"stale-rows","dual-b28-grid170-test",&fa4);
    testAssert(selected.qkvRope.reason == RejectReason::InvalidRegistry);
    testAssert(selected.dualFfn.selected());

    const TacticKey staleAbiQkv[] = {
      {Family::QkvRope,28,6300,0,"stale-abi",true,false,0},
    };
    const RegistryView staleAbiRegistry = {
      {staleAbiQkv,1,sizeof(TacticKey)},
      {dualTactics,sizeof(dualTactics) / sizeof(dualTactics[0]),sizeof(TacticKey)},
    };
    selected = select(
      shape,staleAbiRegistry,"stale-abi","dual-b28-grid170-test",&fa4);
    testAssert(selected.qkvRope.reason == RejectReason::InvalidRegistry);
    testAssert(selected.dualFfn.selected());

    // Packed QKV cannot be consumed by generic SDPA or a different-batch FA4.
    // Dual FFN remains independently reusable.
    fa4.batchSize = 24;
    selected = select(
      shape,registry,"qkv-rope-b28-test","dual-b28-grid340-test",&fa4);
    testAssert(!selected.qkvRope.selected());
    testAssert(selected.packedFa4 == nullptr);
    testAssert(selected.qkvRope.reason == RejectReason::MissingSameBatchFa4);
    testAssert(selected.dualFfn.selected());
    fa4.batchSize = 28;
    fa4.abiVersion = 0;
    selected = select(
      shape,registry,"qkv-rope-b28-test","dual-b28-grid170-test",&fa4);
    testAssert(selected.qkvRope.reason == RejectReason::MissingSameBatchFa4);
    testAssert(selected.dualFfn.selected());
    fa4.abiVersion = kPackedFa4ProofAbiVersion;
    fa4.deviceOrdinal = 1;
    selected = select(
      shape,registry,"qkv-rope-b28-test","dual-b28-grid170-test",&fa4);
    testAssert(selected.qkvRope.reason == RejectReason::MissingSameBatchFa4);
    testAssert(selected.dualFfn.selected());
    fa4.deviceOrdinal = 0;

    selected = select(
      shape,registry,"qkv-rope-b28-stale","dual-missing",&fa4);
    testAssert(selected.qkvRope.reason == RejectReason::RegistryMiss);
    testAssert(selected.dualFfn.reason == RejectReason::RegistryMiss);

    for(const int batch: {24}) {
      shape.batchSize = batch;
      shape.enqueuedRows = batch * 225;
      const char* qkvId = "qkv-rope-b24-test";
      const char* dualId = "dual-b24-grid170-test";
      fa4.batchSize = batch;
      fa4.id = "fa4-packed-b24-test";
      selected = select(shape,registry,qkvId,dualId,&fa4);
      testAssert(selected.qkvRope.selected());
      testAssert(selected.dualFfn.selected());
      testAssert(selected.qkvRope.tactic->tokenRows == batch * 225);
    }

    // B40 was removed from the locked B24/B28 selector and must miss before
    // enqueue. Runtime batch 36 is likewise unrelated to the 36-layer depth.
    shape.batchSize = 40;
    shape.enqueuedRows = 40 * 225;
    selected = select(shape,registry,nullptr,nullptr,nullptr);
    testAssert(!selected.targetShape);
    testAssert(selected.qkvRope.reason == RejectReason::ShapeMismatch);
    testAssert(selected.dualFfn.reason == RejectReason::ShapeMismatch);
    shape.batchSize = 36;
    shape.enqueuedRows = 36 * 225;
    selected = select(shape,registry,nullptr,nullptr,nullptr);
    testAssert(!selected.targetShape);
    testAssert(selected.qkvRope.reason == RejectReason::ShapeMismatch);
    testAssert(selected.dualFfn.reason == RejectReason::ShapeMismatch);
    shape.batchSize = 28;
    shape.enqueuedRows = 6300;
    fa4.batchSize = 28;
    fa4.id = "fa4-packed-b28-test";
    shape.enqueuedRows--;
    testAssert(!targetShapeEligible(shape));
    shape.enqueuedRows = 6300;
    shape.exactNoMask = false;
    testAssert(!targetShapeEligible(shape));
    shape.exactNoMask = true;
    shape.computeCapability = 89;
    testAssert(!targetShapeEligible(shape));
    shape.computeCapability = 120;

    // Model depth is not a local kernel dimension. Both b24 and b36 models
    // reuse the exact bs28 operators when every local shape is unchanged.
    for(const int modelDepth: {24,36}) {
      shape.modelDepth = modelDepth;
      shape.attentionBlockCount = modelDepth;
      shape.ffnBlockCount = modelDepth;
      selected = select(
        shape,registry,"qkv-rope-b28-test","dual-b28-grid170-test",&fa4);
      testAssert(selected.targetShape);
      testAssert(selected.qkvRope.selected());
      testAssert(selected.dualFfn.selected());

      const TransactionProgress complete = {
        modelDepth,modelDepth,modelDepth,modelDepth,modelDepth,modelDepth
      };
      testAssert(transactionProgressComplete(complete));
      TransactionProgress incomplete = complete;
      incomplete.ffnDownCount--;
      testAssert(!transactionProgressComplete(incomplete));
      incomplete = complete;
      incomplete.modelDepth++;
      testAssert(!transactionProgressComplete(incomplete));
    }
    shape.modelDepth = 36;
    shape.attentionBlockCount = 36;
    shape.ffnBlockCount = 36;
    shape.attentionBlockCount = 35;
    testAssert(!modelStructureEligible(shape));
    shape.attentionBlockCount = 36;
    shape.alternatingAttentionFfn = false;
    testAssert(!modelStructureEligible(shape));
    shape.alternatingAttentionFfn = true;
    shape.modelDepth = 0;
    shape.attentionBlockCount = 0;
    shape.ffnBlockCount = 0;
    testAssert(!modelStructureEligible(shape));
    testAssert(!transactionProgressComplete(TransactionProgress{}));
    shape.modelDepth = 36;
    shape.attentionBlockCount = 36;
    shape.ffnBlockCount = 36;

    // The dynamic-depth gate remains C384-specific; the existing C256 route
    // cannot enter either exact family or the whole-model transaction.
    shape.channels = 256;
    selected = select(
      shape,registry,"qkv-rope-b28-test","dual-b28-grid170-test",&fa4);
    testAssert(!selected.targetShape);
    testAssert(selected.qkvRope.reason == RejectReason::ShapeMismatch);
    testAssert(selected.dualFfn.reason == RejectReason::ShapeMismatch);
    shape.channels = 384;

    // Family-local matching preserves reusable kernels on a structurally
    // nearby model: attention changes do not discard a compatible FFN, and
    // FFN changes do not discard a compatible QKV+RoPE operator.
    shape.numHeads = 8;
    shape.numKvHeads = 8;
    shape.ropePairsTotal = 128;
    selected = select(
      shape,registry,"qkv-rope-b28-test","dual-b28-grid170-test",&fa4);
    testAssert(!selected.targetShape);
    testAssert(selected.qkvRope.reason == RejectReason::ShapeMismatch);
    testAssert(selected.dualFfn.selected());
    shape.numHeads = 12;
    shape.numKvHeads = 12;
    shape.ropePairsTotal = 192;
    shape.ffnChannels = 1536;
    selected = select(
      shape,registry,"qkv-rope-b28-test","dual-b28-grid170-test",&fa4);
    testAssert(!selected.targetShape);
    testAssert(selected.qkvRope.selected());
    testAssert(selected.dualFfn.reason == RejectReason::ShapeMismatch);

    // Generated native entry points are versioned independently from the
    // portable registry record and the FA4 proof.
    testAssert(kRegistryAbiVersion == 2);
    testAssert(kQkvRopeNativeAbiVersion == 1);
    testAssert(kDualFfnNativeAbiVersion == 1);
    testAssert(kPackedFa4ProofAbiVersion == 1);

    const size_t qkvElements = (size_t)kChannels * kChannels;
    vector<float> q(qkvElements,0.0f);
    vector<float> k(qkvElements,0.0f);
    vector<float> v(qkvElements,0.0f);
    q[0] = 1.0f;
    k[0] = 2.0f;
    v[0] = 3.0f;
    q[qkvElements - 1] = 4.0f;
    k[qkvElements - 1] = 5.0f;
    v[qkvElements - 1] = 6.0f;
    const vector<float> packedQkv = packQkvWeights(q,k,v);
    testAssert(packedQkv.size() == (size_t)kChannels * kQkvPackedColumns);
    testAssert(packedQkv[packedQkvWeightIndex(0,0,0)] == 1.0f);
    testAssert(packedQkv[packedQkvWeightIndex(0,1,0)] == 2.0f);
    testAssert(packedQkv[packedQkvWeightIndex(0,2,0)] == 3.0f);
    testAssert(packedQkv[packedQkvWeightIndex(383,0,383)] == 4.0f);
    testAssert(packedQkv[packedQkvWeightIndex(383,1,383)] == 5.0f);
    testAssert(packedQkv[packedQkvWeightIndex(383,2,383)] == 6.0f);

    const size_t ffnElements = (size_t)kChannels * kFfnChannels;
    vector<float> up(ffnElements,0.0f);
    vector<float> gate(ffnElements,0.0f);
    for(const int output : {0,63,64,1023}) {
      up[output] = 1000.0f + output;
      gate[output] = 2000.0f + output;
    }
    const vector<float> packedDual = packDualFfnWeights(up,gate);
    testAssert(packedDual.size() ==
      (size_t)kChannels * kDualFfnPackedColumns);
    for(const int output : {0,63,64,1023}) {
      testAssert(packedDual[packedDualFfnWeightIndex(0,false,output)] ==
        1000.0f + output);
      testAssert(packedDual[packedDualFfnWeightIndex(0,true,output)] ==
        2000.0f + output);
    }
    testAssert(packedDualFfnWeightIndex(0,false,63) == 63);
    testAssert(packedDualFfnWeightIndex(0,true,63) == 127);
    testAssert(packedDualFfnWeightIndex(0,false,64) == 128);
    testAssert(packedDualFfnWeightIndex(0,true,64) == 192);
  }

  {
    map<string,string> emptyValues;
    ConfigParser emptyConfig(emptyValues);
    testAssert(NeuralNet::loadCudaUseINT8(emptyConfig,"0"));

    ConfigParser disabledConfig(map<string,string>{{"cudaUseINT8","false"}});
    testAssert(!NeuralNet::loadCudaUseINT8(disabledConfig,"0"));

    ConfigParser ignoredConfig(map<string,string>{{"cudaUseINT8","false"}});
    NeuralNet::ignoreCudaUseINT8(ignoredConfig);
    testAssert(ignoredConfig.unusedKeys().empty());

    ConfigParser perModelConfig(map<string,string>{
      {"cudaUseINT8","false"},
      {"cudaUseINT8-0","true"},
      {"cudaUseINT8-2","false"}
    });
    testAssert(NeuralNet::loadCudaUseINT8(perModelConfig,"0"));
    testAssert(!NeuralNet::loadCudaUseINT8(perModelConfig,"1"));
    testAssert(!NeuralNet::loadCudaUseINT8(perModelConfig,"2"));

    bool rejectedBadConfig = false;
    try {
      ConfigParser invalidConfig(map<string,string>{{"cudaUseINT8","maybe"}});
      (void)NeuralNet::loadCudaUseINT8(invalidConfig,"0");
    }
    catch(const StringError&) {
      rejectedBadConfig = true;
    }
    testAssert(rejectedBadConfig);

    const NeuralNet::CudaInt8Policy defaultPolicy =
      NeuralNet::resolveCudaInt8Policy(true,nullptr);
    testAssert(defaultPolicy.configEnabled);
    testAssert(!defaultPolicy.environmentDisabled);
    testAssert(defaultPolicy.enabled);
    testAssert(NeuralNet::resolveCudaInt8Policy(true,"0").enabled);
    testAssert(!NeuralNet::resolveCudaInt8Policy(true,"1").enabled);
    testAssert(!NeuralNet::resolveCudaInt8Policy(false,nullptr).enabled);

    bool rejectedBadEnvironment = false;
    try {
      (void)NeuralNet::resolveCudaInt8Policy(true,"true");
    }
    catch(const StringError&) {
      rejectedBadEnvironment = true;
    }
    testAssert(rejectedBadEnvironment);
  }

  assertC384RuntimeGateContract();
  assertC384ProductionRuntimeGatePolicy();
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
  const TransformerPairStackTopology pairStackA =
    analyzeTransformerPairStack(architectureA);
  testAssert(pairStackA.eligible);
  testAssert(pairStackA.depth == 24);
  testAssert(pairStackA.attentionCount == 24);
  testAssert(pairStackA.ffnCount == 24);

  // Outer input/head operators are valid, but a non-transformer operation
  // inserted between Attention and FFN must invalidate the whole-model exact
  // transaction rather than being silently filtered out.
  ArchitectureDesc interruptedPairStack = architectureA;
  size_t firstAttention = interruptedPairStack.operators.size();
  for(size_t i = 0; i < interruptedPairStack.operators.size(); i++) {
    if(interruptedPairStack.operators[i].kind ==
       ArchitectureOpKind::TransformerAttention) {
      firstAttention = i;
      break;
    }
  }
  testAssert(firstAttention < interruptedPairStack.operators.size());
  ArchitectureOpDesc insertedConv{};
  insertedConv.kind = ArchitectureOpKind::Conv2D;
  interruptedPairStack.operators.insert(
    interruptedPairStack.operators.begin() + firstAttention + 1,
    insertedConv);
  const TransformerPairStackTopology interruptedTopology =
    analyzeTransformerPairStack(interruptedPairStack);
  testAssert(!interruptedTopology.eligible);
  testAssert(interruptedTopology.depth == 0);
  testAssert(interruptedTopology.attentionCount == 24);
  testAssert(interruptedTopology.ffnCount == 24);
  testAssert(architectureA.signature == architectureB.signature);
  testAssert(architectureA.canonicalEncoding == architectureB.canonicalEncoding);
  // This weight-free fixture mirrors REAL_MODEL_ARCHITECTURE_MANIFEST.json,
  // independently derived from reviewed.bin.gz with SHA-256
  // 40cfa5ab15e23b12d065a2b4611e6b9aad0e02ade724c91657851acc53ffd4c6.
  // Lock both the canonical byte count and digest so field/schema drift fails loudly.
  testAssert(architectureA.canonicalEncoding.size() == 4673);
  testAssert(
    architectureA.signature.toHex() ==
    "ad026614455c0475b31997f1c5452af99d1eb347713f77950671fc5d1a522f24"
  );
  testAssert(
    CudaTransformerWinner::int8LegacyImplicitArchitectureSignature() ==
    architectureA.signature
  );
  ModelDesc explicitV104 = makeModel(
    24,256,768,8,32,0.6f,"explicit-v104-weights");
  explicitV104.version = 104;
  explicitV104.trunk.version = 104;
  explicitV104.policyHead.version = 104;
  explicitV104.valueHead.version = 104;
  const ArchitectureDesc architectureV104 = buildArchitectureDesc(explicitV104);
  testAssert(architectureV104.canonicalEncoding.size() == 4673);
  testAssert(
    architectureV104.signature.toHex() ==
    "bbfa5957d8d87f1225fe4e679be055ff4de4c40c4f8437a19c7be3052c23f49b"
  );
  testAssert(
    CudaTransformerWinner::int8QualifiedArchitectureSignature() ==
    architectureV104.signature
  );
  testAssert(architectureV104.signature != architectureA.signature);
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

  // The generic fixture registry has no C384 entries, so the real 36-layer
  // C384/H12/D32/F1024 target is compatible-only here. Independent head
  // operators such as value MatMul 64->3 and ownership Conv 96->1 retain their
  // exact capability key and compatible recipe.
  ModelDesc wideModel = makeModel(36,384,1024,12,32,0.5f,"b36c384-h12-f1024");
  ArchitectureDesc wideArchitecture = buildArchitectureDesc(wideModel);
  PlanSnapshot widePlan = preparePlan(wideArchitecture,b36,registry);
  size_t wideAttention = findRequest(
    widePlan,ArchitectureOpKind::TransformerAttention,384,384,16
  );
  size_t wideFFN = findRequest(
    widePlan,ArchitectureOpKind::TransformerFFN,384,384,1024
  );
  testAssert(widePlan.requests[wideAttention].key.boardX == 15);
  testAssert(widePlan.requests[wideAttention].key.boardY == 15);
  testAssert(widePlan.requests[wideFFN].key.spatialArea == 225);
  testAssert(widePlan.requests[wideFFN].key.boardX == 0);
  testAssert(widePlan.requests[wideFFN].key.boardY == 0);
  testAssert(widePlan.prepared[wideAttention].support == SupportClass::CompatibleOnly);
  testAssert(widePlan.prepared[wideFFN].support == SupportClass::CompatibleOnly);
  size_t baseV3 = findRequest(planA,ArchitectureOpKind::MatMul,128,3);
  size_t wideV3 = findRequest(widePlan,ArchitectureOpKind::MatMul,128,3);
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
  const CudaTransformerWinner::Int8ExperimentEligibility int8G1 =
    CudaTransformerWinner::evaluateInt8ExperimentEligibility(productionG1A);
  testAssert(int8G1.exactCurrent24LayerModel());
  testAssert(int8G1.architectureSignatureMatches);
  testAssert(!int8G1.explicitV104ArchitectureSignatureMatches);
  testAssert(int8G1.legacyV102ArchitectureSignatureMatches);
  testAssert(int8G1.preparedPlanFingerprintValid);
  testAssert(int8G1.runtimeContractEligible);
  testAssert(int8G1.allTransformerRecordsPrepared);
  testAssert(int8G1.attentionCount == 24);
  testAssert(int8G1.ffnCount == 24);
  testAssert(CudaTransformerWinner::evaluateInt8ExperimentEligibility(
    productionG1B).exactCurrent24LayerModel());

  CudaTransformerWinner::PreparedPlan productionV104 =
    CudaTransformerWinner::preparePlan(architectureV104,b36,device);
  const CudaTransformerWinner::Int8ExperimentEligibility int8V104 =
    CudaTransformerWinner::evaluateInt8ExperimentEligibility(productionV104);
  testAssert(int8V104.exactCurrent24LayerModel());
  testAssert(int8V104.architectureSignatureMatches);
  testAssert(int8V104.explicitV104ArchitectureSignatureMatches);
  testAssert(!int8V104.legacyV102ArchitectureSignatureMatches);

  // Combined INT8+C384 integration contract: same-GPU evaluator concurrency
  // is transported through RuntimeOpContext for launch-time C384 policy only.
  // It must neither alter the weight-free v104 model identity nor disable the
  // qualified C256 INT8 recipe. Each prepared plan retains the requested lane
  // count for diagnostics/dispatch while remaining independently eligible.
  for(uint32_t sameGpuConcurrency: {1u,2u,3u}) {
    RuntimeOpContext concurrentRuntime = b36;
    concurrentRuntime.streamCount = sameGpuConcurrency;
    const CudaTransformerWinner::PreparedPlan concurrentV104 =
      CudaTransformerWinner::preparePlan(
        architectureV104,concurrentRuntime,device);
    testAssert(concurrentV104.runtime.streamCount == sameGpuConcurrency);
    testAssert(concurrentV104.architecture == architectureV104.signature);
    const CudaTransformerWinner::Int8ExperimentEligibility concurrentEligibility =
      CudaTransformerWinner::evaluateInt8ExperimentEligibility(concurrentV104);
    testAssert(concurrentEligibility.exactCurrent24LayerModel());
    testAssert(concurrentEligibility.explicitV104ArchitectureSignatureMatches);
    assertAllTransformerRecipes(
      architectureV104,concurrentV104,24,true,
      assertExactAttentionRecipe,assertExactFfnRecipe
    );
  }

  // Same local shapes and counts but a different semantic block order must
  // not inherit the qualified whole-model INT8 arithmetic recipe.
  ModelDesc reorderedModel = makeModel(
    24,256,768,8,32,0.4f,"reordered-attention-ffn");
  std::swap(reorderedModel.trunk.blocks[1],reorderedModel.trunk.blocks[2]);
  ArchitectureDesc reorderedArchitecture = buildArchitectureDesc(reorderedModel);
  testAssert(reorderedArchitecture.signature != architectureA.signature);
  CudaTransformerWinner::PreparedPlan reorderedPlan =
    CudaTransformerWinner::preparePlan(reorderedArchitecture,b36,device);
  const CudaTransformerWinner::Int8ExperimentEligibility reorderedInt8 =
    CudaTransformerWinner::evaluateInt8ExperimentEligibility(reorderedPlan);
  testAssert(!reorderedInt8.architectureSignatureMatches);
  testAssert(reorderedInt8.preparedPlanFingerprintValid);
  testAssert(reorderedInt8.allTransformerShapesEligible);
  testAssert(reorderedInt8.attentionCount == 24);
  testAssert(reorderedInt8.ffnCount == 24);
  testAssert(!reorderedInt8.exactCurrent24LayerModel());
  for(const CudaTransformerWinner::PreparedRecord& record: productionG1A.records) {
    if(record.request.key.kind == ArchitectureOpKind::TransformerAttention) {
      testAssert(record.request.key.boardX == 15);
      testAssert(record.request.key.boardY == 15);
      testAssert(record.request.key.spatialArea == 225);
    }
    else if(record.request.key.kind == ArchitectureOpKind::TransformerFFN) {
      testAssert(record.request.key.boardX == 0);
      testAssert(record.request.key.boardY == 0);
      testAssert(record.request.key.spatialArea == 225);
    }
  }
  CudaTransformerWinner::PreparedPlan wrongInt8Area = productionG1A;
  for(CudaTransformerWinner::PreparedRecord& record: wrongInt8Area.records) {
    if(record.request.key.kind == ArchitectureOpKind::TransformerFFN) {
      record.request.key.spatialArea = 226;
      break;
    }
  }
  testAssert(!CudaTransformerWinner::evaluateInt8ExperimentEligibility(
    wrongInt8Area).exactCurrent24LayerModel());
  CudaTransformerWinner::PreparedPlan wrongInt8Width = productionG1A;
  for(CudaTransformerWinner::PreparedRecord& record: wrongInt8Width.records) {
    if(record.request.key.kind == ArchitectureOpKind::TransformerAttention) {
      record.request.key.inChannels = 384;
      break;
    }
  }
  testAssert(!CudaTransformerWinner::evaluateInt8ExperimentEligibility(
    wrongInt8Width).exactCurrent24LayerModel());
  CudaTransformerWinner::PreparedPlan wrongInt8Fingerprint = productionG1A;
  wrongInt8Fingerprint.fingerprint.digest[0] ^= 1;
  const CudaTransformerWinner::Int8ExperimentEligibility badFingerprintInt8 =
    CudaTransformerWinner::evaluateInt8ExperimentEligibility(
      wrongInt8Fingerprint);
  testAssert(badFingerprintInt8.architectureSignatureMatches);
  testAssert(!badFingerprintInt8.preparedPlanFingerprintValid);
  testAssert(!badFingerprintInt8.exactCurrent24LayerModel());
  CudaTransformerWinner::PreparedPlan wrongRuntimeBoard = productionG1A;
  wrongRuntimeBoard.runtime.boardX = 9;
  wrongRuntimeBoard.runtime.boardY = 25;
  const CudaTransformerWinner::Int8ExperimentEligibility badRuntimeBoardInt8 =
    CudaTransformerWinner::evaluateInt8ExperimentEligibility(
      wrongRuntimeBoard);
  testAssert(badRuntimeBoardInt8.architectureSignatureMatches);
  testAssert(badRuntimeBoardInt8.preparedPlanFingerprintValid);
  testAssert(!badRuntimeBoardInt8.runtimeContractEligible);
  testAssert(!badRuntimeBoardInt8.exactCurrent24LayerModel());

  // The arithmetic gate is stricter than the weight-free architecture: every
  // transformer record must describe the exact FP16/F32 SM120 contract and a
  // prepared SM120 recipe. The plan fingerprint intentionally does not encode
  // found, so eligibility must reject that state independently.
  CudaTransformerWinner::PreparedPlan missingPreparedRecord = productionG1A;
  for(CudaTransformerWinner::PreparedRecord& record: missingPreparedRecord.records) {
    if(record.request.key.kind == ArchitectureOpKind::TransformerAttention) {
      record.found = false;
      break;
    }
  }
  const CudaTransformerWinner::Int8ExperimentEligibility missingPreparedInt8 =
    CudaTransformerWinner::evaluateInt8ExperimentEligibility(missingPreparedRecord);
  testAssert(missingPreparedInt8.preparedPlanFingerprintValid);
  testAssert(!missingPreparedInt8.allTransformerRecordsPrepared);
  testAssert(!missingPreparedInt8.exactCurrent24LayerModel());

  RuntimeOpContext sm89Runtime = b36;
  sm89Runtime.deviceComputeCapability = 89;
  CudaTransformerWinner::DeviceCapability sm89Device = device;
  sm89Device.computeCapability = 89;
  sm89Device.specializedSm120KernelsAvailable = false;
  const CudaTransformerWinner::Int8ExperimentEligibility sm89Int8 =
    CudaTransformerWinner::evaluateInt8ExperimentEligibility(
      CudaTransformerWinner::preparePlan(architectureA,sm89Runtime,sm89Device));
  testAssert(!sm89Int8.runtimeContractEligible);
  testAssert(!sm89Int8.allTransformerRecordsPrepared);
  testAssert(!sm89Int8.exactCurrent24LayerModel());

  RuntimeOpContext fp32Runtime = b36;
  fp32Runtime.inputType = NumericType::Float32;
  fp32Runtime.outputType = NumericType::Float32;
  const CudaTransformerWinner::Int8ExperimentEligibility fp32Int8 =
    CudaTransformerWinner::evaluateInt8ExperimentEligibility(
      CudaTransformerWinner::preparePlan(architectureA,fp32Runtime,device));
  testAssert(!fp32Int8.runtimeContractEligible);
  testAssert(!fp32Int8.allTransformerRecordsPrepared);
  testAssert(!fp32Int8.exactCurrent24LayerModel());

  RuntimeOpContext nchwRuntime = b36;
  nchwRuntime.layout = TensorLayout::NCHW;
  const CudaTransformerWinner::Int8ExperimentEligibility nchwInt8 =
    CudaTransformerWinner::evaluateInt8ExperimentEligibility(
      CudaTransformerWinner::preparePlan(architectureA,nchwRuntime,device));
  testAssert(!nchwInt8.runtimeContractEligible);
  testAssert(!nchwInt8.allTransformerRecordsPrepared);
  testAssert(!nchwInt8.exactCurrent24LayerModel());

  RuntimeOpContext nhwcRuntime = b36;
  nhwcRuntime.layout = TensorLayout::NHWC;
  testAssert(CudaTransformerWinner::evaluateInt8ExperimentEligibility(
    CudaTransformerWinner::preparePlan(
      architectureA,nhwcRuntime,device)).exactCurrent24LayerModel());

  CudaTransformerWinner::PreparedPlan missingLearnedRope = productionG1A;
  for(CudaTransformerWinner::PreparedRecord& record: missingLearnedRope.records) {
    if(record.request.key.kind == ArchitectureOpKind::TransformerAttention) {
      record.request.key.flags &= ~OP_FLAG_LEARNABLE_ROPE;
      break;
    }
  }
  refreshProductionPlanFingerprint(missingLearnedRope);
  testAssert(!CudaTransformerWinner::evaluateInt8ExperimentEligibility(
    missingLearnedRope).exactCurrent24LayerModel());

  CudaTransformerWinner::PreparedPlan missingSwiGlu = productionG1A;
  for(CudaTransformerWinner::PreparedRecord& record: missingSwiGlu.records) {
    if(record.request.key.kind == ArchitectureOpKind::TransformerFFN) {
      record.request.key.flags &= ~OP_FLAG_USE_SWIGLU;
      break;
    }
  }
  refreshProductionPlanFingerprint(missingSwiGlu);
  testAssert(!CudaTransformerWinner::evaluateInt8ExperimentEligibility(
    missingSwiGlu).exactCurrent24LayerModel());

  CudaTransformerWinner::PreparedPlan wrongRmsContract = productionG1A;
  for(CudaTransformerWinner::PreparedRecord& record: wrongRmsContract.records) {
    if(record.request.key.kind == ArchitectureOpKind::TransformerFFN) {
      record.request.key.semanticScalar0Bits ^= 1;
      break;
    }
  }
  refreshProductionPlanFingerprint(wrongRmsContract);
  testAssert(!CudaTransformerWinner::evaluateInt8ExperimentEligibility(
    wrongRmsContract).exactCurrent24LayerModel());
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
  testAssert(!CudaTransformerWinner::evaluateInt8ExperimentEligibility(
    productionG5).exactCurrent24LayerModel());
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
  testAssert(!CudaTransformerWinner::evaluateInt8ExperimentEligibility(
    productionG2).exactCurrent24LayerModel());
  assertAllTransformerRecipes(
    architectureA,productionG2,24,true,
    assertC256StaticAttentionRecipe,assertC256StaticFfnRecipe
  );

  // G3: a dense mask retains only mask-safe planar QKV and learned-half2 RoPE.
  // The no-mask RMS/FA/fused-QKV/beta-one/dual/down tactics all fail closed.
  CudaTransformerWinner::PreparedPlan productionG3 =
    CudaTransformerWinner::preparePlan(architectureA,masked,device);
  testAssert(!CudaTransformerWinner::evaluateInt8ExperimentEligibility(
    productionG3).exactCurrent24LayerModel());
  assertAllTransformerRecipes(
    architectureA,productionG3,24,false,
    assertMaskSafeAttentionRecipe,assertDisabledFfnRecipe
  );

  // G4: B32*225 is a validated dynamic-M row count. It reuses C256 RMS,
  // learned RoPE, both residual s3 kernels, and dual FFN, but not B36-only FA4
  // or the B36-only fused QKV+RoPE recipe.
  CudaTransformerWinner::PreparedPlan productionG4 =
    CudaTransformerWinner::preparePlan(architectureA,b32,device);
  // Batch is intentionally runtime-dynamic and does not alter architecture
  // qualification; kernel support handles actual-M independently.
  testAssert(CudaTransformerWinner::evaluateInt8ExperimentEligibility(
    productionG4).exactCurrent24LayerModel());
  assertAllTransformerRecipes(
    architectureA,productionG4,24,true,
    assertB32DynamicAttentionRecipe,assertExactFfnRecipe
  );

  // The real b36c384/H12/D32/F1024 target (b36 means 36 transformer layers,
  // not batch size) selects a dynamic-M partial specialization for every
  // representative runtime batch. Attention retains geometry-general planar
  // QKV, learned half2 RoPE, and cuDNN SDPA. C384 RMS and C384/F1024 dual FFN
  // are launch-gated specializations; out/down directly select cuBLAS beta-one
  // and do not prepare never-launched specialized handles.
  CudaTransformerWinner::PreparedPlan productionWide36 =
    CudaTransformerWinner::preparePlan(wideArchitecture,b36,device);
  const CudaTransformerWinner::Int8ExperimentEligibility wideInt8 =
    CudaTransformerWinner::evaluateInt8ExperimentEligibility(productionWide36);
  testAssert(!wideInt8.exactCurrent24LayerModel());
  testAssert(!wideInt8.architectureSignatureMatches);
  testAssert(!wideInt8.explicitV104ArchitectureSignatureMatches);
  testAssert(!wideInt8.legacyV102ArchitectureSignatureMatches);
  assertAllTransformerRecipes(
    wideArchitecture,productionWide36,36,true,
    assertC384DynamicAttentionRecipe,assertC384DynamicFfnRecipe
  );

  // A v104 wire version does not make a structurally different C384 model
  // eligible for the C256 INT8 runtime plan. Independent model-format
  // validation still applies to every v104 model, but the exact
  // C256/H8/F768 architecture gate remains false while all 36 C384 records
  // select their measured FP16 recipes. Therefore CudaHandles cannot enable
  // int8ExperimentPlan or its scratch/embedded-entry consumption paths.
  ModelDesc wideModelV104 =
    makeModel(36,384,1024,12,32,0.7f,"b36c384-h12-f1024-v104");
  wideModelV104.version = 104;
  wideModelV104.trunk.version = 104;
  wideModelV104.policyHead.version = 104;
  wideModelV104.valueHead.version = 104;
  const ArchitectureDesc wideArchitectureV104 = buildArchitectureDesc(wideModelV104);
  const CudaTransformerWinner::PreparedPlan productionWideV104 =
    CudaTransformerWinner::preparePlan(wideArchitectureV104,b36,device);
  const CudaTransformerWinner::Int8ExperimentEligibility wideV104Int8 =
    CudaTransformerWinner::evaluateInt8ExperimentEligibility(productionWideV104);
  testAssert(!wideV104Int8.exactCurrent24LayerModel());
  testAssert(!wideV104Int8.architectureSignatureMatches);
  testAssert(!wideV104Int8.explicitV104ArchitectureSignatureMatches);
  testAssert(!wideV104Int8.legacyV102ArchitectureSignatureMatches);
  assertAllTransformerRecipes(
    wideArchitectureV104,productionWideV104,36,true,
    assertC384DynamicAttentionRecipe,assertC384DynamicFfnRecipe
  );
  for(int batch: {1,8,16,24,32,36,64,128,129,256,512,4660}) {
    const RuntimeOpContext bucket = runtimeContext(batch,15,15,MaskMode::None);
    const CudaTransformerWinner::PreparedPlan bucketPlan =
      CudaTransformerWinner::preparePlan(wideArchitecture,bucket,device);
    assertAllTransformerRecipes(
      wideArchitecture,bucketPlan,36,true,
      assertC384DynamicAttentionRecipe,assertC384DynamicFfnRecipe
    );
    const CudaTransformerWinner::PreparedRecord& attention =
      firstProductionRecord(bucketPlan,ArchitectureOpKind::TransformerAttention);
    const CudaTransformerWinner::PreparedRecord& ffn =
      firstProductionRecord(bucketPlan,ArchitectureOpKind::TransformerFFN);
    testAssert(string(CudaTransformerWinner::tacticName(attention.operation.tactic)) ==
      "attention-c384-h12-dynamic-sm120");
    testAssert(string(CudaTransformerWinner::tacticName(ffn.operation.tactic)) ==
      "ffn-c384-f1024-dynamic-sm120");
  }

  // Layer count is absent from the local tactic gate. The old 32-layer
  // development fixture, the real 36-layer target, and a 48-layer stress
  // fixture have different architecture/whole-plan fingerprints but identical
  // per-attention and per-FFN capability keys and prepared tactics.
  const ModelDesc wideModel32 =
    makeModel(32,384,1024,12,32,0.4f,"c384-h12-f1024-dev32");
  const ModelDesc wideModel48 =
    makeModel(48,384,1024,12,32,0.6f,"c384-h12-f1024-stress48");
  const ArchitectureDesc wideArchitecture32 = buildArchitectureDesc(wideModel32);
  const ArchitectureDesc wideArchitecture48 = buildArchitectureDesc(wideModel48);
  const CudaTransformerWinner::PreparedPlan productionWide32 =
    CudaTransformerWinner::preparePlan(wideArchitecture32,b36,device);
  const CudaTransformerWinner::PreparedPlan productionWide48 =
    CudaTransformerWinner::preparePlan(wideArchitecture48,b36,device);
  assertAllTransformerRecipes(
    wideArchitecture32,productionWide32,32,true,
    assertC384DynamicAttentionRecipe,assertC384DynamicFfnRecipe
  );
  assertAllTransformerRecipes(
    wideArchitecture48,productionWide48,48,true,
    assertC384DynamicAttentionRecipe,assertC384DynamicFfnRecipe
  );
  testAssert(productionWide32.architecture != productionWide36.architecture);
  testAssert(productionWide36.architecture != productionWide48.architecture);
  testAssert(productionWide32.fingerprint != productionWide36.fingerprint);
  testAssert(productionWide36.fingerprint != productionWide48.fingerprint);
  const CudaTransformerWinner::PreparedRecord& wide32Attention =
    firstProductionRecord(productionWide32,ArchitectureOpKind::TransformerAttention);
  const CudaTransformerWinner::PreparedRecord& wide36Attention =
    firstProductionRecord(productionWide36,ArchitectureOpKind::TransformerAttention);
  const CudaTransformerWinner::PreparedRecord& wide48Attention =
    firstProductionRecord(productionWide48,ArchitectureOpKind::TransformerAttention);
  const CudaTransformerWinner::PreparedRecord& wide32Ffn =
    firstProductionRecord(productionWide32,ArchitectureOpKind::TransformerFFN);
  const CudaTransformerWinner::PreparedRecord& wide36Ffn =
    firstProductionRecord(productionWide36,ArchitectureOpKind::TransformerFFN);
  const CudaTransformerWinner::PreparedRecord& wide48Ffn =
    firstProductionRecord(productionWide48,ArchitectureOpKind::TransformerFFN);
  testAssert(wide32Attention.request.key == wide36Attention.request.key);
  testAssert(wide36Attention.request.key == wide48Attention.request.key);
  testAssert(wide32Ffn.request.key == wide36Ffn.request.key);
  testAssert(wide36Ffn.request.key == wide48Ffn.request.key);
  assertPreparedOpIdentity(wide32Attention.operation,wide36Attention.operation);
  assertPreparedOpIdentity(wide36Attention.operation,wide48Attention.operation);
  assertPreparedOpIdentity(wide32Ffn.operation,wide36Ffn.operation);
  assertPreparedOpIdentity(wide36Ffn.operation,wide48Ffn.operation);

  // B129 is beyond the largest measured batch but remains a structurally valid
  // dynamic-M request, so it must reuse the same C384 local tactics. Only the
  // technical 2^20-row implementation bound or a changed board falls back;
  // the safe generic planar/RoPE/beta-one path remains available, and C256
  // exact planning above is unchanged.
  const RuntimeOpContext b129 = runtimeContext(129,15,15,MaskMode::None);
  const CudaTransformerWinner::PreparedPlan productionWideB129 =
    CudaTransformerWinner::preparePlan(wideArchitecture,b129,device);
  assertAllTransformerRecipes(
    wideArchitecture,productionWideB129,36,true,
    assertC384DynamicAttentionRecipe,assertC384DynamicFfnRecipe
  );
  const RuntimeOpContext bOverC384Max = runtimeContext(4661,15,15,MaskMode::None);
  const CudaTransformerWinner::PreparedPlan productionWideOverC384Max =
    CudaTransformerWinner::preparePlan(wideArchitecture,bOverC384Max,device);
  assertAllTransformerRecipes(
    wideArchitecture,productionWideOverC384Max,36,true,
    assertWideAttentionRecipe,assertGenericFfnRecipe
  );
  const CudaTransformerWinner::PreparedPlan productionWideBoard19 =
    CudaTransformerWinner::preparePlan(wideArchitecture,board19,device);
  assertAllTransformerRecipes(
    wideArchitecture,productionWideBoard19,36,true,
    assertWideAttentionRecipe,assertGenericFfnRecipe
  );

  cout << "Production PreparedPlan fingerprints:" << endl;
  cout << "  G1=" << productionG1A.fingerprint.toHex() << endl;
  cout << "  G2-S361=" << productionG2.fingerprint.toHex() << endl;
  cout << "  G3-dense-mask=" << productionG3.fingerprint.toHex() << endl;
  cout << "  G4-B32=" << productionG4.fingerprint.toHex() << endl;
  cout << "  G5-48-layers=" << productionG5.fingerprint.toHex() << endl;
  cout << "  target-36L-C384-H12-F1024=" << productionWide36.fingerprint.toHex() << endl;
  cout << "  target-C384-B129-dynamic-reuse=" << productionWideB129.fingerprint.toHex() << endl;
  cout << "  target-C384-over-max-rows-fallback=" <<
    productionWideOverC384Max.fingerprint.toHex() << endl;

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

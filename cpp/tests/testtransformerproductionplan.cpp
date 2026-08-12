#include "../tests/tests.h"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "../core/test.h"
#include "../neuralnet/activations.h"
#include "../neuralnet/architecturedesc.h"
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
  context.runtimeLibraryFingerprint = 0x13000D130101914ULL;
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
  return key.batchSize == 36 && key.spatialArea == 225 &&
    key.boardX == 15 && key.boardY == 15 && key.maskMode == MaskMode::None &&
    key.inputType == NumericType::Float16 && key.outputType == NumericType::Float16 &&
    key.deviceComputeCapability == 120 && key.streamCount == 2 &&
    key.runtimeLibraryFingerprint == 0x13000D130101914ULL;
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
  size_t attention24 = findRequest(planA,ArchitectureOpKind::TransformerAttention,256,256);
  size_t attention48 = findRequest(plan48,ArchitectureOpKind::TransformerAttention,256,256);
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
  ModelDesc wideModel = makeModel(24,384,1024,12,32,0.5f,"c384-h12-f1024");
  ArchitectureDesc wideArchitecture = buildArchitectureDesc(wideModel);
  PlanSnapshot widePlan = preparePlan(wideArchitecture,b36,registry);
  size_t wideAttention = findRequest(
    widePlan,ArchitectureOpKind::TransformerAttention,384,384
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

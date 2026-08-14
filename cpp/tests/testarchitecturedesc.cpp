#include "../tests/tests.h"

#include <cmath>
#include <limits>
#include <type_traits>

#include "../core/test.h"
#include "../neuralnet/activations.h"
#include "../neuralnet/architecturedesc.h"
#include "../neuralnet/cudaopregistry.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nativeint8quant.h"

using namespace std;
using namespace NeuralNetArchitecture;
using namespace CudaOpRegistry;

namespace {

static void setConv(ConvLayerDesc& desc, int inChannels, int outChannels, int kernel, int dilation, float weight) {
  desc.name = "ignored-conv-name-" + Global::floatToString(weight);
  desc.convXSize = kernel;
  desc.convYSize = kernel;
  desc.inChannels = inChannels;
  desc.outChannels = outChannels;
  desc.dilationX = dilation;
  desc.dilationY = dilation;
  desc.weights.assign((size_t)kernel * kernel * inChannels * outChannels,weight);
}

static void setMatMul(MatMulLayerDesc& desc, int inChannels, int outChannels, float weight) {
  desc.name = "ignored-matmul-name-" + Global::floatToString(weight);
  desc.inChannels = inChannels;
  desc.outChannels = outChannels;
  desc.weights.assign((size_t)inChannels * outChannels,weight);
}

static void setMatBias(MatBiasLayerDesc& desc, int channels, float weight) {
  desc.name = "ignored-bias-name-" + Global::floatToString(weight);
  desc.numChannels = channels;
  desc.weights.assign(channels,weight);
}

static void setBatchNorm(
  BatchNormLayerDesc& desc,
  int channels,
  bool hasScale,
  bool hasBias,
  float weight
) {
  desc.name = "ignored-bn-name-" + Global::floatToString(weight);
  desc.numChannels = channels;
  desc.epsilon = 1e-20f;
  desc.hasScale = hasScale;
  desc.hasBias = hasBias;
  desc.mean.assign(channels,weight);
  desc.variance.assign(channels,1.0f + weight);
  desc.scale.assign(channels,hasScale ? 1.0f + weight : 1.0f);
  desc.bias.assign(channels,hasBias ? weight : 0.0f);
}

static void setActivation(ActivationLayerDesc& desc, int activation, float weight) {
  desc.name = "ignored-activation-name-" + Global::floatToString(weight);
  desc.activation = activation;
}

static void appendTextWeights(ostream& out, size_t count, float value = 0.125f) {
  for(size_t i = 0; i < count; i++)
    out << value << "\n";
}

static void appendTextRMSNorm(
  ostream& out,
  const string& name,
  int channels,
  float epsilon
) {
  out << name << "\n" << channels << "\n" << epsilon << "\n";
  appendTextWeights(out,(size_t)channels);
}

static void appendTextMatMul(
  ostream& out,
  const string& name,
  int inChannels,
  int outChannels
) {
  out << name << "\n" << inChannels << "\n" << outChannels << "\n";
  appendTextWeights(out,(size_t)inChannels * outChannels);
}

static string attentionWireFixture(int modelVersion, bool useQKNorm) {
  ostringstream out;
  out << "attention\n2\n2\n4\n4\n1\n1\n";
  if(modelVersion >= 105)
    out << (useQKNorm ? 1 : 0) << "\n";
  appendTextRMSNorm(out,"attention.norm1",8,1e-6f);
  appendTextMatMul(out,"attention.q",8,8);
  appendTextMatMul(out,"attention.k",8,8);
  appendTextMatMul(out,"attention.v",8,8);
  appendTextMatMul(out,"attention.out",8,8);
  if(useQKNorm) {
    appendTextRMSNorm(out,"attention.q_norm",4,1e-6f);
    appendTextRMSNorm(out,"attention.k_norm",4,2e-6f);
  }
  out << "attention.rope_freqs\n2\n2\n2\n";
  appendTextWeights(out,8);
  return out.str();
}

static string ffnWireFixture(
  int modelVersion,
  float swigluClip,
  float productQuantMaxAbs,
  bool includeProductQuantMaxAbs = true
) {
  ostringstream out;
  out << "ffn\n8\n16\n1\n";
  if(modelVersion >= 105) {
    out << swigluClip << "\n";
    if(includeProductQuantMaxAbs)
      out << productQuantMaxAbs << "\n";
  }
  appendTextRMSNorm(out,"ffn.norm",8,1e-6f);
  appendTextMatMul(out,"ffn.up",8,16);
  appendTextMatMul(out,"ffn.gate",8,16);
  appendTextMatMul(out,"ffn.down",16,8);
  return out.str();
}

static unique_ptr_void makeAttention(int channels, int heads, int headDim, float weight) {
  TransformerAttentionDesc* desc = new TransformerAttentionDesc();
  desc->name = "ignored-attention-name-" + Global::floatToString(weight);
  desc->numHeads = heads;
  desc->numKVHeads = heads;
  desc->qHeadDim = headDim;
  desc->vHeadDim = headDim;
  desc->useRope = true;
  desc->learnableRope = true;
  desc->preLN.name = "ignored-rms-name";
  desc->preLN.numChannels = channels;
  desc->preLN.epsilon = 1e-6f;
  desc->preLN.weight.assign(channels,weight);
  setMatMul(desc->qProj,channels,heads * headDim,weight);
  setMatMul(desc->kProj,channels,heads * headDim,weight + 0.01f);
  setMatMul(desc->vProj,channels,heads * headDim,weight + 0.02f);
  setMatMul(desc->outProj,heads * headDim,channels,weight + 0.03f);
  desc->ropeNumKVHeads = heads;
  desc->ropeNumPairs = headDim / 2;
  desc->ropeFreqs.assign((size_t)heads * (headDim / 2) * 2,weight + 0.04f);
  return make_unique_void(desc);
}

static unique_ptr_void makeFFN(int channels, int ffnChannels, float weight) {
  TransformerFFNDesc* desc = new TransformerFFNDesc();
  desc->name = "ignored-ffn-name-" + Global::floatToString(weight);
  desc->numChannels = channels;
  desc->ffnChannels = ffnChannels;
  desc->useSwiGLU = true;
  desc->preLN.name = "ignored-rms-name";
  desc->preLN.numChannels = channels;
  desc->preLN.epsilon = 1e-6f;
  desc->preLN.weight.assign(channels,weight);
  setMatMul(desc->linear1,channels,ffnChannels,weight);
  setMatMul(desc->linearGate,channels,ffnChannels,weight + 0.01f);
  setMatMul(desc->linear2,ffnChannels,channels,weight + 0.02f);
  return make_unique_void(desc);
}

static ModelDesc makeModel(
  int logicalBlocks,
  int channels,
  int ffnChannels,
  int heads,
  int headDim,
  float weightSeed
) {
  ModelDesc model;
  model.name = "ignored-model-name-" + Global::floatToString(weightSeed);
  model.sha256 = "ignored-artifact-sha-" + Global::floatToString(weightSeed);
  model.version = 102;
  model.numInputChannels = 22;
  model.numInputGlobalChannels = 39;
  model.numValueChannels = 3;
  model.numScoreValueChannels = 6;
  model.numOwnershipChannels = 1;

  model.trunk.name = "ignored-trunk-name";
  model.trunk.version = 102;
  model.trunk.numBlocks = logicalBlocks * 2;
  model.trunk.trunkNumChannels = channels;
  model.trunk.midNumChannels = channels;
  model.trunk.regularNumChannels = channels;
  model.trunk.dilatedNumChannels = 0;
  model.trunk.gpoolNumChannels = channels;
  setConv(model.trunk.initialConv,22,channels,1,1,weightSeed);
  setMatMul(model.trunk.initialMatMul,39,channels,weightSeed);
  for(int i = 0; i < logicalBlocks; i++) {
    model.trunk.blocks.push_back(
      make_pair(TRANSFORMER_ATTENTION_BLOCK_KIND,makeAttention(channels,heads,headDim,weightSeed + i))
    );
    model.trunk.blocks.push_back(
      make_pair(TRANSFORMER_FFN_BLOCK_KIND,makeFFN(channels,ffnChannels,weightSeed + i))
    );
  }
  setBatchNorm(model.trunk.trunkTipBN,channels,true,true,weightSeed);
  setActivation(model.trunk.trunkTipActivation,ACTIVATION_SILU,weightSeed);

  model.policyHead.name = "ignored-policy-name";
  model.policyHead.version = 102;
  setConv(model.policyHead.p1Conv,channels,48,1,1,weightSeed);
  setConv(model.policyHead.g1Conv,channels,48,1,1,weightSeed);
  setBatchNorm(model.policyHead.g1BN,48,false,true,weightSeed);
  setActivation(model.policyHead.g1Activation,ACTIVATION_SILU,weightSeed);
  setMatMul(model.policyHead.gpoolToBiasMul,48 * 3,48,weightSeed);
  setBatchNorm(model.policyHead.p1BN,48,false,true,weightSeed);
  setActivation(model.policyHead.p1Activation,ACTIVATION_SILU,weightSeed);
  setConv(model.policyHead.p2Conv,48,1,1,1,weightSeed);
  setMatMul(model.policyHead.gpoolToPassMul,48 * 3,1,weightSeed);

  model.valueHead.name = "ignored-value-name";
  model.valueHead.version = 102;
  setConv(model.valueHead.v1Conv,channels,96,1,1,weightSeed);
  setBatchNorm(model.valueHead.v1BN,96,false,true,weightSeed);
  setActivation(model.valueHead.v1Activation,ACTIVATION_SILU,weightSeed);
  setMatMul(model.valueHead.v2Mul,96 * 3,64,weightSeed);
  setMatBias(model.valueHead.v2Bias,64,weightSeed);
  setActivation(model.valueHead.v2Activation,ACTIVATION_SILU,weightSeed);
  setMatMul(model.valueHead.v3Mul,64,3,weightSeed);
  setMatBias(model.valueHead.v3Bias,3,weightSeed);
  setMatMul(model.valueHead.sv3Mul,64,6,weightSeed);
  setMatBias(model.valueHead.sv3Bias,6,weightSeed);
  setConv(model.valueHead.vOwnershipConv,96,1,1,1,weightSeed);
  return model;
}

static void enableQKNormAndClip(
  ModelDesc& model,
  float qEpsilon,
  float kEpsilon,
  float swigluClip,
  float productQuantMaxAbs
) {
  model.version = 105;
  model.trunk.version = 105;
  model.policyHead.version = 105;
  model.valueHead.version = 105;
  for(auto& entry: model.trunk.blocks) {
    if(entry.first == TRANSFORMER_ATTENTION_BLOCK_KIND) {
      TransformerAttentionDesc* desc = (TransformerAttentionDesc*)entry.second.get();
      desc->useQKNorm = true;
      desc->qNorm.name = "ignored-q-norm";
      desc->qNorm.numChannels = desc->qHeadDim;
      desc->qNorm.epsilon = qEpsilon;
      desc->qNorm.weight.assign(desc->qHeadDim,1.0f);
      desc->kNorm.name = "ignored-k-norm";
      desc->kNorm.numChannels = desc->qHeadDim;
      desc->kNorm.epsilon = kEpsilon;
      desc->kNorm.weight.assign(desc->qHeadDim,1.0f);
    }
    else if(entry.first == TRANSFORMER_FFN_BLOCK_KIND) {
      TransformerFFNDesc* desc = (TransformerFFNDesc*)entry.second.get();
      desc->swigluClip = swigluClip;
      desc->productQuantMaxAbs = productQuantMaxAbs;
    }
  }
}

static RuntimeOpContext runtimeContext(int batch, int x, int y, MaskMode mask) {
  RuntimeOpContext context{};
  context.batchSize = batch;
  context.boardX = x;
  context.boardY = y;
  context.maskMode = mask;
  context.inputType = NumericType::Float16;
  context.outputType = NumericType::Float16;
  context.computeType = NumericType::Float32;
  context.layout = TensorLayout::BSH;
  context.deviceComputeCapability = 120;
  context.streamCount = 2;
  context.runtimeLibraryFingerprint = 0xC0DA1300ULL;
  return context;
}

static const ArchitectureOpDesc& findOp(const ArchitectureDesc& architecture, ArchitectureOpKind kind) {
  for(const ArchitectureOpDesc& op: architecture.operators) {
    if(op.kind == kind)
      return op;
  }
  testAssert(false);
  return architecture.operators[0];
}

struct TacticTestData {
  CapabilityKey key;
  SupportClass support;
  bool prepareSucceeds;
  uint64_t cookie;
};

static SupportClass matchTestTactic(const OpRequest& request, const void* userData) {
  const TacticTestData* data = (const TacticTestData*)userData;
  return request.key == data->key ? data->support : SupportClass::Unsupported;
}

static bool prepareTestTactic(const OpRequest&, PreparedOp& prepared, void* userData) {
  const TacticTestData* data = (const TacticTestData*)userData;
  if(!data->prepareSucceeds)
    return false;
  prepared.workspaceAlignment = 256;
  prepared.workspaceBytes = 4096;
  prepared.implementationCookie = (uintptr_t)data->cookie;
  return true;
}

static TacticRegistration registration(
  uint64_t variant,
  int priority,
  const string& recipe,
  TacticTestData* data
) {
  TacticRegistration result{};
  result.id = TacticId{0x52454E4A553135ULL,variant};
  result.recipe = fingerprintRecipe(recipe);
  result.priority = priority;
  result.match = matchTestTactic;
  result.prepare = prepareTestTactic;
  result.userData = data;
  return result;
}

}  // namespace

void Tests::runArchitectureDescTests(const string& nativeModelFile) {
  static_assert(NNModelVersion::defaultModelVersion == 102,
    "Explicit INT8 must not change the default exporter/model version");
  testAssert(NNModelVersion::getInputsVersion(104) == 101);
  testAssert(NNModelVersion::getNumSpatialFeatures(104) ==
    NNModelVersion::getNumSpatialFeatures(102));
  testAssert(NNModelVersion::getNumGlobalFeatures(104) ==
    NNModelVersion::getNumGlobalFeatures(102));
  testAssert(NNModelVersion::getInputsVersion(105) == 101);
  testAssert(NNModelVersion::getNumSpatialFeatures(105) == 22);
  testAssert(NNModelVersion::getNumGlobalFeatures(105) == 39);
  static_assert(is_trivially_copyable<OpRequest>::value,"OpRequest should be POD-like");
  static_assert(is_trivially_copyable<CapabilityKey>::value,"CapabilityKey should be POD-like");
  static_assert(is_trivially_copyable<TacticId>::value,"TacticId should be POD-like");
  static_assert(is_trivially_copyable<PreparedOp>::value,"PreparedOp should be POD-like");

  // The version-aware transformer parser must leave every legacy v102 byte in
  // its old position while admitting the new v105 semantic fields.
  {
    istringstream attentionV102In(attentionWireFixture(102,false));
    TransformerAttentionDesc attentionV102(attentionV102In,102,false);
    testAssert(!attentionV102.useQKNorm);
    testAssert(attentionV102.qNorm.numChannels == 0);
    attentionV102In >> ws;
    testAssert(attentionV102In.peek() == EOF);

    istringstream ffnV102In(ffnWireFixture(102,0.0f,0.0f));
    TransformerFFNDesc ffnV102(ffnV102In,102,false);
    testAssert(ffnV102.swigluClip == 0.0f);
    testAssert(ffnV102.productQuantMaxAbs == 0.0f);
    ffnV102In >> ws;
    testAssert(ffnV102In.peek() == EOF);

    istringstream attentionV105In(attentionWireFixture(105,true));
    TransformerAttentionDesc attentionV105(attentionV105In,105,false);
    testAssert(attentionV105.useQKNorm);
    testAssert(attentionV105.qNorm.numChannels == 4);
    testAssert(attentionV105.kNorm.numChannels == 4);
    testAssert(getFloatBits(attentionV105.qNorm.epsilon) == getFloatBits(1e-6f));
    testAssert(getFloatBits(attentionV105.kNorm.epsilon) == getFloatBits(2e-6f));
    attentionV105In >> ws;
    testAssert(attentionV105In.peek() == EOF);

    istringstream ffnV105In(ffnWireFixture(105,7.0f,49.0f));
    TransformerFFNDesc ffnV105(ffnV105In,105,false);
    testAssert(ffnV105.swigluClip == 7.0f);
    testAssert(ffnV105.productQuantMaxAbs == 49.0f);
    ffnV105In >> ws;
    testAssert(ffnV105In.peek() == EOF);

    // productQuantMaxAbs is serialized per FFN in wire order. Verify that a
    // heterogeneous sequence is neither shifted nor accidentally broadcast.
    const float mixedProductRanges[] = {47.7371f,73.4171f,48.0f};
    string mixedWire;
    for(const float range: mixedProductRanges)
      mixedWire += ffnWireFixture(105,7.0f,range);
    istringstream mixedV105In(mixedWire);
    for(const float expectedRange: mixedProductRanges) {
      TransformerFFNDesc mixedFFN(mixedV105In,105,false);
      testAssert(getFloatBits(mixedFFN.productQuantMaxAbs) ==
        getFloatBits(expectedRange));
    }
    mixedV105In >> ws;
    testAssert(mixedV105In.peek() == EOF);

    // A pre-upgrade v105 stream has preLN immediately after swigluClip. The
    // new parser must consume no such stream accidentally: trying to parse the
    // norm name as the mandatory product range fails closed.
    bool rejectedOldV105 = false;
    try {
      istringstream oldV105In(ffnWireFixture(105,7.0f,49.0f,false));
      (void)TransformerFFNDesc(oldV105In,105,false);
    }
    catch(const StringError&) {
      rejectedOldV105 = true;
    }
    testAssert(rejectedOldV105);

    bool rejectedZeroProductRange = false;
    try {
      istringstream zeroProductIn(ffnWireFixture(105,7.0f,0.0f));
      (void)TransformerFFNDesc(zeroProductIn,105,false);
    }
    catch(const StringError&) {
      rejectedZeroProductRange = true;
    }
    testAssert(rejectedZeroProductRange);
  }

  ModelDesc modelA = makeModel(2,256,768,8,32,0.1f);
  ModelDesc modelB = makeModel(2,256,768,8,32,0.9f);
  modelB.onnxHeader.modelName = "ignored-header-model-name";
  modelB.onnxHeader.model_config = "ignored-exporter-config";
  modelB.onnxHeader.model_config_sha256 = "ignored-config-sha";
  modelB.onnxHeader.allmetadata["ignored-key"] = "ignored-value";
  ArchitectureDesc architectureA = buildArchitectureDesc(modelA);
  ArchitectureDesc architectureB = buildArchitectureDesc(modelB);
  testAssert(architectureA.signature == architectureB.signature);
  testAssert(architectureA.canonicalEncoding == architectureB.canonicalEncoding);
  testAssert(getModelProvenance(modelA).artifactSha256 != getModelProvenance(modelB).artifactSha256);
  testAssert(getModelProvenance(modelB).modelConfigSha256 == "ignored-config-sha");
  testAssert(architectureA.outerBatchNormScaleMask == 0x1);
  testAssert(architectureA.outerBatchNormBiasMask == 0xF);
  testAssert(
    architectureA.signature.toHex() ==
    "33ce0bbfc7231d7f211c38a8b5667263c96ef6877710ba0834d96f5104cac38e"
  );

  const uint8_t scaleMasksToTest[] = {0x0,0x1,0x2,0x3,0xF};
  for(uint8_t expectedMask: scaleMasksToTest) {
    modelB.trunk.trunkTipBN.hasScale = (expectedMask & OUTER_NORM_TRUNK_TIP) != 0;
    modelB.policyHead.g1BN.hasScale = (expectedMask & OUTER_NORM_POLICY_G1) != 0;
    modelB.policyHead.p1BN.hasScale = (expectedMask & OUTER_NORM_POLICY_P1) != 0;
    modelB.valueHead.v1BN.hasScale = (expectedMask & OUTER_NORM_VALUE_V1) != 0;
    testAssert(buildArchitectureDesc(modelB).outerBatchNormScaleMask == expectedMask);
  }
  modelB.trunk.trunkTipBN.hasScale = true;
  modelB.policyHead.g1BN.hasScale = false;
  modelB.policyHead.p1BN.hasScale = false;
  modelB.valueHead.v1BN.hasScale = false;
  modelB.valueHead.v1BN.hasBias = false;
  testAssert(buildArchitectureDesc(modelB).outerBatchNormBiasMask == 0x7);
  modelB.valueHead.v1BN.hasBias = true;

  ModelDesc deeperModel = makeModel(4,256,768,8,32,0.3f);
  ArchitectureDesc deeper = buildArchitectureDesc(deeperModel);
  testAssert(architectureA.signature != deeper.signature);
  RuntimeOpContext baseRuntime = runtimeContext(36,15,15,MaskMode::None);
  CapabilityKey ffnA = makeCapabilityKey(findOp(architectureA,ArchitectureOpKind::TransformerFFN),baseRuntime);
  CapabilityKey ffnDeeper = makeCapabilityKey(findOp(deeper,ArchitectureOpKind::TransformerFFN),baseRuntime);
  testAssert(ffnA == ffnDeeper);

  const ArchitectureOpDesc& attentionOp = findOp(architectureA,ArchitectureOpKind::TransformerAttention);
  const ArchitectureOpDesc& ffnOp = findOp(architectureA,ArchitectureOpKind::TransformerFFN);
  const ArchitectureOpDesc& globalMatMulOp = findOp(architectureA,ArchitectureOpKind::MatMul);
  CapabilityKey attention = makeCapabilityKey(attentionOp,baseRuntime);
  CapabilityKey ffn = makeCapabilityKey(ffnOp,baseRuntime);
  CapabilityKey globalMatMul = makeCapabilityKey(globalMatMulOp,baseRuntime);
  testAssert((attention.flags & OP_FLAG_USE_QK_NORM) == 0);
  testAssert(attention.semanticScalar2Bits == 0);
  testAssert(attention.semanticScalar3Bits == 0);
  testAssert((ffn.flags & OP_FLAG_USE_SWIGLU_CLIP) == 0);
  testAssert(ffn.semanticScalar1Bits == 0);
  testAssert(ffn.semanticScalar2Bits == 0);

  ModelDesc qknClipModel = makeModel(2,384,1024,12,32,0.2f);
  enableQKNormAndClip(qknClipModel,1e-6f,2e-6f,7.0f,49.0f);
  ArchitectureDesc qknClipArchitecture = buildArchitectureDesc(qknClipModel);
  testAssert(qknClipArchitecture.signature != architectureA.signature);
  const ArchitectureOpDesc& qknAttentionOp = findOp(
    qknClipArchitecture,ArchitectureOpKind::TransformerAttention
  );
  const ArchitectureOpDesc& clippedFFNOp = findOp(
    qknClipArchitecture,ArchitectureOpKind::TransformerFFN
  );
  CapabilityKey qknAttention = makeCapabilityKey(qknAttentionOp,baseRuntime);
  CapabilityKey clippedFFN = makeCapabilityKey(clippedFFNOp,baseRuntime);
  testAssert((qknAttention.flags & OP_FLAG_USE_QK_NORM) != 0);
  testAssert(qknAttention.semanticScalar2Bits == getFloatBits(1e-6f));
  testAssert(qknAttention.semanticScalar3Bits == getFloatBits(2e-6f));
  testAssert((clippedFFN.flags & OP_FLAG_USE_SWIGLU_CLIP) != 0);
  testAssert(clippedFFN.semanticScalar1Bits == getFloatBits(7.0f));
  testAssert(clippedFFN.semanticScalar2Bits == getFloatBits(49.0f));

  // Gamma tensors are trained weights and deliberately do not affect either
  // identity, while each semantic scalar must invalidate tactic reuse.
  TransformerAttentionDesc* firstQKN = (TransformerAttentionDesc*)
    qknClipModel.trunk.blocks[0].second.get();
  const ArchitectureSignature qknSignature = qknClipArchitecture.signature;
  firstQKN->qNorm.weight[0] = 3.0f;
  testAssert(buildArchitectureDesc(qknClipModel).signature == qknSignature);
  firstQKN->qNorm.epsilon = 4e-6f;
  ArchitectureDesc changedQKNArchitecture = buildArchitectureDesc(qknClipModel);
  testAssert(changedQKNArchitecture.signature != qknSignature);
  testAssert(makeCapabilityKey(
    findOp(changedQKNArchitecture,ArchitectureOpKind::TransformerAttention),
    baseRuntime
  ) != qknAttention);
  firstQKN->qNorm.epsilon = 1e-6f;
  TransformerFFNDesc* firstClippedFFN = (TransformerFFNDesc*)
    qknClipModel.trunk.blocks[1].second.get();
  firstClippedFFN->productQuantMaxAbs = 48.0f;
  ArchitectureDesc changedProductRangeArchitecture = buildArchitectureDesc(qknClipModel);
  testAssert(changedProductRangeArchitecture.signature != qknSignature);
  testAssert(makeCapabilityKey(
    findOp(changedProductRangeArchitecture,ArchitectureOpKind::TransformerFFN),
    baseRuntime
  ) != clippedFFN);
  firstClippedFFN->productQuantMaxAbs = 49.0f;
  firstClippedFFN->swigluClip = 6.0f;
  ArchitectureDesc changedClipArchitecture = buildArchitectureDesc(qknClipModel);
  testAssert(makeCapabilityKey(
    findOp(changedClipArchitecture,ArchitectureOpKind::TransformerFFN),
    baseRuntime
  ) != clippedFFN);

  RuntimeOpContext batch32 = runtimeContext(32,15,15,MaskMode::None);
  testAssert(makeCapabilityKey(ffnOp,batch32) != ffn);
  testAssert(makeCapabilityKey(globalMatMulOp,batch32) != globalMatMul);

  RuntimeOpContext board19 = runtimeContext(36,19,19,MaskMode::None);
  testAssert(makeCapabilityKey(ffnOp,board19) != ffn);
  testAssert(makeCapabilityKey(attentionOp,board19) != attention);
  testAssert(makeCapabilityKey(globalMatMulOp,board19) == globalMatMul);

  RuntimeOpContext sameArea = runtimeContext(36,9,25,MaskMode::None);
  testAssert(makeCapabilityKey(ffnOp,sameArea) == ffn);
  testAssert(makeCapabilityKey(attentionOp,sameArea) != attention);

  RuntimeOpContext masked = runtimeContext(36,15,15,MaskMode::Dense);
  testAssert(makeCapabilityKey(attentionOp,masked) != attention);
  testAssert(makeCapabilityKey(ffnOp,masked) != ffn);
  testAssert(makeCapabilityKey(globalMatMulOp,masked) == globalMatMul);

  ArchitectureOpDesc changed = ffnOp;
  changed.inChannels = 384;
  changed.outChannels = 384;
  testAssert(makeCapabilityKey(changed,baseRuntime) != ffn);
  changed = ffnOp;
  changed.auxiliaryChannels = 1024;
  testAssert(makeCapabilityKey(changed,baseRuntime) != ffn);
  changed = attentionOp;
  changed.numHeads = 12;
  testAssert(makeCapabilityKey(changed,baseRuntime) != attention);
  changed = attentionOp;
  changed.qHeadDim = 64;
  testAssert(makeCapabilityKey(changed,baseRuntime) != attention);

  const ArchitectureOpDesc& convOp = findOp(architectureA,ArchitectureOpKind::Conv2D);
  CapabilityKey conv = makeCapabilityKey(convOp,baseRuntime);
  changed = convOp;
  changed.dilationX = 2;
  changed.dilationY = 2;
  testAssert(makeCapabilityKey(changed,baseRuntime) != conv);

  bool rejectedBadEpsilon = false;
  modelA.trunk.trunkTipBN.epsilon = numeric_limits<float>::quiet_NaN();
  try {
    (void)buildArchitectureDesc(modelA);
  }
  catch(const StringError&) {
    rejectedBadEpsilon = true;
  }
  testAssert(rejectedBadEpsilon);

  bool rejectedBadQKNGeometry = false;
  firstQKN->qNorm.epsilon = 1e-6f;
  firstQKN->qNorm.numChannels = firstQKN->qHeadDim + 1;
  try {
    (void)buildArchitectureDesc(qknClipModel);
  }
  catch(const StringError&) {
    rejectedBadQKNGeometry = true;
  }
  testAssert(rejectedBadQKNGeometry);
  firstQKN->qNorm.numChannels = firstQKN->qHeadDim;

  bool rejectedClipWithoutSwiGLU = false;
  firstClippedFFN->useSwiGLU = false;
  firstClippedFFN->swigluClip = 7.0f;
  try {
    (void)buildArchitectureDesc(qknClipModel);
  }
  catch(const StringError&) {
    rejectedClipWithoutSwiGLU = true;
  }
  testAssert(rejectedClipWithoutSwiGLU);
  firstClippedFFN->useSwiGLU = true;

  bool rejectedMissingV105ProductRange = false;
  firstClippedFFN->productQuantMaxAbs = 0.0f;
  try {
    (void)buildArchitectureDesc(qknClipModel);
  }
  catch(const StringError&) {
    rejectedMissingV105ProductRange = true;
  }
  testAssert(rejectedMissingV105ProductRange);
  firstClippedFFN->productQuantMaxAbs = 49.0f;

  vector<OpRequest> requests = buildOpRequests(architectureB,baseRuntime);
  OpRequest ffnRequest{};
  bool foundFFNRequest = false;
  for(const OpRequest& request: requests) {
    if(request.key.kind == ArchitectureOpKind::TransformerFFN) {
      ffnRequest = request;
      foundFFNRequest = true;
      break;
    }
  }
  testAssert(foundFFNRequest);

  TacticTestData compatible{ffnRequest.key,SupportClass::CompatibleOnly,true,11};
  TacticTestData certified{ffnRequest.key,SupportClass::CertifiedFast,true,22};
  Registry registry;
  registry.registerTactic(registration(1,100,"compatible-generic-v1",&compatible));
  registry.registerTactic(registration(2,1,"certified-renju-v1",&certified));
  ResolveResult resolved = registry.resolveAtConstruction(ffnRequest);
  testAssert(resolved.found);
  testAssert(resolved.prepared.support == SupportClass::CertifiedFast);
  testAssert(resolved.prepared.tactic.variant == 2);
  testAssert(resolved.prepared.implementationCookie == 22);

  bool rejectedDuplicate = false;
  try {
    registry.registerTactic(registration(2,0,"duplicate",&certified));
  }
  catch(const StringError&) {
    rejectedDuplicate = true;
  }
  testAssert(rejectedDuplicate);

  TacticTestData failingCertified{ffnRequest.key,SupportClass::CertifiedFast,false,33};
  Registry fallbackRegistry;
  fallbackRegistry.registerTactic(registration(3,50,"certified-but-prepare-fails",&failingCertified));
  fallbackRegistry.registerTactic(registration(4,10,"compatible-fallback",&compatible));
  ResolveResult fallback = fallbackRegistry.resolveAtConstruction(ffnRequest);
  testAssert(fallback.found);
  testAssert(fallback.prepared.support == SupportClass::CompatibleOnly);
  testAssert(fallback.prepared.tactic.variant == 4);

  vector<OpRequest> oneRequest(1,ffnRequest);
  vector<PreparedOp> onePrepared(1,resolved.prepared);
  PlanFingerprint planA = fingerprintPreparedPlan(oneRequest,onePrepared);
  testAssert(planA.toHex() == "55ef32d613f48d0a81d452f2028edd10a127b2101cd96d537d7c7c9b90a3bfaa");
  onePrepared[0].implementationCookie = 999;
  PlanFingerprint planSame = fingerprintPreparedPlan(oneRequest,onePrepared);
  testAssert(planA == planSame);
  onePrepared[0].workspaceBytes++;
  PlanFingerprint planChanged = fingerprintPreparedPlan(oneRequest,onePrepared);
  testAssert(planA != planChanged);

  // v104 embeds deterministic clip4/per-tensor signed-int8 bytes while
  // retaining and binding every canonical FP32 master. Keep this fixture small
  // but preserve the production 24 attention + 24 SwiGLU FFN topology.
  ModelDesc quantModel = makeModel(24,4,8,1,4,0.125f);
  NativeInt8Quant::Metadata quant = NativeInt8Quant::build(quantModel);
  testAssert(NativeInt8Quant::weightSourceForModel(quantModel) ==
    NativeInt8Quant::WeightSource::LegacyImplicitV102);
  testAssert(string(NativeInt8Quant::weightSourceName(
    NativeInt8Quant::WeightSource::LegacyImplicitV102)) ==
    "legacy-v102-load-time-quant");
  quantModel.nativeInt8Quant = quant;
  bool rejectedMetadataOnLegacy = false;
  try {
    (void)NativeInt8Quant::weightSourceForModel(quantModel);
  }
  catch(const StringError&) {
    rejectedMetadataOnLegacy = true;
  }
  testAssert(rejectedMetadataOnLegacy);
  quantModel.nativeInt8Quant = NativeInt8Quant::Metadata();
  testAssert(quant.present());
  testAssert(quant.entries.size() == NativeInt8Quant::REQUIRED_ENTRY_COUNT);
  testAssert(quant.zeroPoint == 0);
  testAssert(quant.activationClipBits == 0x40800000U);
  testAssert(quant.activationScaleBits == 0x3D010204U);
  for(uint32_t layer = 0; layer < 24; layer++) {
    const size_t base = (size_t)layer * 3;
    testAssert(quant.entries[base].topologyIndex == 2 + layer*2);
    testAssert(quant.entries[base].role == NativeInt8Quant::Role::QK);
    testAssert(quant.entries[base+1].topologyIndex == 3 + layer*2);
    testAssert(quant.entries[base+1].role == NativeInt8Quant::Role::FfnUp);
    testAssert(quant.entries[base+2].topologyIndex == 3 + layer*2);
    testAssert(quant.entries[base+2].role == NativeInt8Quant::Role::FfnGate);
  }
  size_t qkCount = 0;
  size_t upCount = 0;
  size_t gateCount = 0;
  for(const NativeInt8Quant::Entry& entry: quant.entries) {
    if(entry.role == NativeInt8Quant::Role::QK) {
      qkCount++;
      testAssert(entry.layerNames.size() == 2);
      testAssert(entry.outputChannels == 8);
    }
    else if(entry.role == NativeInt8Quant::Role::FfnUp) {
      upCount++;
      testAssert(entry.layerNames.size() == 1);
      testAssert(entry.outputChannels == 8);
    }
    else if(entry.role == NativeInt8Quant::Role::FfnGate) {
      gateCount++;
      testAssert(entry.layerNames.size() == 1);
      testAssert(entry.outputChannels == 8);
    }
    else
      testAssert(false);
    testAssert(entry.inputChannels == 4);
    testAssert(entry.zeroPoint == 0);
    testAssert(entry.layout ==
      NativeInt8Quant::PackedLayout::OutputMajorKContiguous);
    testAssert(entry.activationScaleBits == quant.activationScaleBits);
    testAssert(entry.packedWeights.size() ==
      (size_t)entry.inputChannels * entry.outputChannels);
    for(int8_t value: entry.packedWeights)
      testAssert(value != (int8_t)-128);
  }
  testAssert(qkCount == 24 && upCount == 24 && gateCount == 24);

  const NativeInt8Quant::Entry& selectedQk = NativeInt8Quant::requireEntry(
    quant,2,NativeInt8Quant::Role::QK,
    quant.entries[0].layerNames,4,8);
  testAssert(&selectedQk == &quant.entries[0]);
  testAssert(NativeInt8Quant::weightScale(selectedQk) > 0.0f);
  const auto rejectedBackendLookup = [&](const NativeInt8Quant::Metadata& metadata,
                                         uint32_t topologyIndex,
                                         const vector<string>& names,
                                         uint32_t k,
                                         uint32_t n) {
    try {
      (void)NativeInt8Quant::requireEntry(
        metadata,topologyIndex,NativeInt8Quant::Role::QK,names,k,n);
      return false;
    }
    catch(const StringError&) {
      return true;
    }
  };
  testAssert(rejectedBackendLookup(
    quant,999,quant.entries[0].layerNames,4,8));
  testAssert(rejectedBackendLookup(
    quant,2,vector<string>{"wrong.q","wrong.k"},4,8));
  testAssert(rejectedBackendLookup(
    quant,2,quant.entries[0].layerNames,4,9));
  NativeInt8Quant::Metadata duplicateLookup = quant;
  duplicateLookup.entries.push_back(quant.entries[0]);
  testAssert(rejectedBackendLookup(
    duplicateLookup,2,quant.entries[0].layerNames,4,8));

  ModelDesc explicitSourceModel = makeModel(24,4,8,1,4,0.375f);
  explicitSourceModel.version = 104;
  explicitSourceModel.trunk.version = 104;
  explicitSourceModel.policyHead.version = 104;
  explicitSourceModel.valueHead.version = 104;
  explicitSourceModel.nativeInt8Quant =
    NativeInt8Quant::build(explicitSourceModel);
  testAssert(NativeInt8Quant::weightSourceForModel(explicitSourceModel) ==
    NativeInt8Quant::WeightSource::EmbeddedV104);
  testAssert(string(NativeInt8Quant::weightSourceName(
    NativeInt8Quant::WeightSource::EmbeddedV104)) == "embedded-v104");
  explicitSourceModel.nativeInt8Quant = NativeInt8Quant::Metadata();
  bool rejectedMissingExplicitMetadata = false;
  try {
    (void)NativeInt8Quant::weightSourceForModel(explicitSourceModel);
  }
  catch(const StringError&) {
    rejectedMissingExplicitMetadata = true;
  }
  testAssert(rejectedMissingExplicitMetadata);

  const vector<uint8_t> quantPayload = NativeInt8Quant::encodePayload(quant);
  NativeInt8Quant::Metadata decoded =
    NativeInt8Quant::decodePayload(quantPayload);
  NativeInt8Quant::validate(quantModel,decoded);
  testAssert(NativeInt8Quant::encodePayload(decoded) == quantPayload);

  stringstream trailerStream(ios::in | ios::out | ios::binary);
  NativeInt8Quant::writeTrailer(trailerStream,quant);
  const string canonicalTrailer = trailerStream.str();
  testAssert(canonicalTrailer.find(
    "@KATAGO_QUANT_TRAILER@ 1 ") == 0);
  trailerStream.seekg(0);
  decoded = NativeInt8Quant::readTrailer(trailerStream,quantModel);
  NativeInt8Quant::validate(quantModel,decoded);

  const auto rejectedTrailer = [&](const string& bytes, const ModelDesc& model) {
    stringstream input(bytes,ios::in | ios::binary);
    try {
      (void)NativeInt8Quant::readTrailer(input,model);
      return false;
    }
    catch(const StringError&) {
      return true;
    }
  };

  string corruptOuterSha = canonicalTrailer;
  corruptOuterSha[corruptOuterSha.size()-1] ^= 1;
  testAssert(rejectedTrailer("",quantModel));
  testAssert(rejectedTrailer(corruptOuterSha,quantModel));
  testAssert(rejectedTrailer(canonicalTrailer + "\n",quantModel));
  testAssert(rejectedTrailer(
    canonicalTrailer.substr(0,canonicalTrailer.size()-1),quantModel));
  testAssert(rejectedTrailer(
    "@KATAGO_QUANT_TRAILER@ 2 1 " + string(64,'0') + " @BIN@x",
    quantModel));

  NativeInt8Quant::Metadata badEntry = NativeInt8Quant::build(quantModel);
  badEntry.entries[0].packedWeights[0] ^= 1;
  stringstream badEntryStream(ios::in | ios::out | ios::binary);
  NativeInt8Quant::writeTrailer(badEntryStream,badEntry);
  testAssert(rejectedTrailer(badEntryStream.str(),quantModel));

  NativeInt8Quant::Metadata reordered = NativeInt8Quant::build(quantModel);
  std::swap(reordered.entries[0],reordered.entries[1]);
  stringstream reorderedStream(ios::in | ios::out | ios::binary);
  NativeInt8Quant::writeTrailer(reorderedStream,reordered);
  testAssert(rejectedTrailer(reorderedStream.str(),quantModel));

  // A different trained weight tensor with identical architecture must not be
  // allowed to reuse bytes bound to the original master.
  TransformerAttentionDesc* firstAttention = (TransformerAttentionDesc*)
    quantModel.trunk.blocks[0].second.get();
  firstAttention->qProj.weights[0] += 0.03125f;
  testAssert(rejectedTrailer(canonicalTrailer,quantModel));

  // Optional integration gate for a real native model. V104 loading reaches
  // the strict EOF parser and recomputes all 72 master/packed hashes. The v102
  // branch proves that legacy native artifacts still require no trailer.
  if(!nativeModelFile.empty()) {
    ModelDesc external;
    ModelDesc::loadFromFileMaybeGZipped(nativeModelFile,external,"");
    const string externalArchitecture =
      buildArchitectureDesc(external).signature.toHex();
    if(external.version == (int)NativeInt8Quant::MODEL_VERSION) {
      testAssert(external.nativeInt8Quant.present());
      testAssert(external.nativeInt8Quant.entries.size() ==
        NativeInt8Quant::REQUIRED_ENTRY_COUNT);
      // Byte-equality here is the cross-language exporter/runtime gate. CUDA
      // consumes these validated packed bytes directly and does not rebuild
      // them from FP32 masters for v104.
      const NativeInt8Quant::Metadata runtimeRebuilt =
        NativeInt8Quant::build(external);
      testAssert(NativeInt8Quant::encodePayload(external.nativeInt8Quant) ==
        NativeInt8Quant::encodePayload(runtimeRebuilt));
      testAssert(externalArchitecture ==
        "bbfa5957d8d87f1225fe4e679be055ff4de4c40c4f8437a19c7be3052c23f49b");
      cout << "Loaded and validated native v104 model: " <<
        nativeModelFile << endl;
    }
    else if(external.version == 102) {
      testAssert(!external.nativeInt8Quant.present());
      testAssert(externalArchitecture ==
        "ad026614455c0475b31997f1c5452af99d1eb347713f77950671fc5d1a522f24");
      cout << "Loaded unchanged legacy native v102 model: " <<
        nativeModelFile << endl;
    }
    else {
      testAssert(external.version == 105);
      size_t attentionCount = 0;
      size_t qknCount = 0;
      size_t ffnCount = 0;
      size_t clippedFFNCount = 0;
      for(const auto& entry: external.trunk.blocks) {
        if(entry.first == TRANSFORMER_ATTENTION_BLOCK_KIND) {
          attentionCount++;
          const TransformerAttentionDesc* desc =
            (const TransformerAttentionDesc*)entry.second.get();
          if(desc->useQKNorm) {
            qknCount++;
            testAssert(desc->qNorm.numChannels == desc->qHeadDim);
            testAssert(desc->kNorm.numChannels == desc->qHeadDim);
            testAssert(desc->qNorm.weight.size() == (size_t)desc->qHeadDim);
            testAssert(desc->kNorm.weight.size() == (size_t)desc->qHeadDim);
            testAssert(desc->qNorm.epsilon == 1e-6f);
            testAssert(desc->kNorm.epsilon == 1e-6f);
          }
        }
        else if(entry.first == TRANSFORMER_FFN_BLOCK_KIND) {
          ffnCount++;
          const TransformerFFNDesc* desc =
            (const TransformerFFNDesc*)entry.second.get();
          if(desc->swigluClip > 0.0f) {
            testAssert(desc->swigluClip == 7.0f);
            testAssert(std::isfinite(desc->productQuantMaxAbs));
            testAssert(desc->productQuantMaxAbs > 0.0f);
            clippedFFNCount++;
          }
        }
      }
      testAssert(attentionCount == 36 && qknCount == 36);
      testAssert(ffnCount == 36 && clippedFFNCount == 36);
      cout << "Loaded and validated native v105 QKN/clip model: " <<
        nativeModelFile << endl;
    }
  }

  cout << "Architecture descriptor, CUDA op registry, and native INT8 trailer tests passed" << endl;
}

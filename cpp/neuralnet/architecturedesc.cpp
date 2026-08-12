#include "../neuralnet/architecturedesc.h"

#include <cmath>
#include <cstring>

#include "../core/global.h"
#include "../core/sha2.h"
#include "../neuralnet/desc.h"

using namespace std;

namespace NeuralNetArchitecture {
namespace {

enum class EncodingTag : uint32_t {
  Model = 1,
  Trunk = 2,
  PolicyHead = 3,
  ValueHead = 4,
  Block = 5,
  Conv = 6,
  BatchNorm = 7,
  Activation = 8,
  MatMul = 9,
  MatBias = 10,
  RMSNorm = 11,
  TransformerAttention = 12,
  TransformerFFN = 13,
};

class CanonicalWriter {
 public:
  vector<uint8_t> bytes;

  void u8(uint8_t value) {
    bytes.push_back(value);
  }
  void boolean(bool value) {
    u8(value ? 1 : 0);
  }
  void u32(uint32_t value) {
    for(int i = 0; i < 4; i++)
      bytes.push_back((uint8_t)((value >> (i * 8)) & 0xFFu));
  }
  void i32(int32_t value) {
    u32((uint32_t)value);
  }
  void tag(EncodingTag value) {
    u32((uint32_t)value);
  }
  void raw(const char* value, size_t len) {
    bytes.insert(bytes.end(), value, value + len);
  }
};

static void requirePositive(int value, const char* field) {
  if(value <= 0)
    throw StringError(string("Invalid architecture descriptor: ") + field + " must be positive");
}

static void requireFinitePositive(float value, const char* field) {
  if(!isfinite(value) || value <= 0.0f)
    throw StringError(string("Invalid architecture descriptor: ") + field + " must be finite and positive");
}

static uint32_t activationFlags(int activation) {
  return ((uint32_t)activation << OP_FLAG_ACTIVATION_SHIFT) & OP_FLAG_ACTIVATION_MASK;
}

static void appendOp(ArchitectureDesc& result, ArchitectureOpDesc op) {
  op.topologyIndex = (uint32_t)result.operators.size();
  result.operators.push_back(op);
}

static ArchitectureOpDesc emptyOp(ArchitectureOpKind kind) {
  ArchitectureOpDesc op{};
  op.kind = kind;
  return op;
}

static void encodeConv(
  CanonicalWriter& out,
  const ConvLayerDesc& desc,
  ArchitectureDesc& result,
  bool emitOperator
) {
  requirePositive(desc.convXSize,"convXSize");
  requirePositive(desc.convYSize,"convYSize");
  requirePositive(desc.inChannels,"conv.inChannels");
  requirePositive(desc.outChannels,"conv.outChannels");
  requirePositive(desc.dilationX,"conv.dilationX");
  requirePositive(desc.dilationY,"conv.dilationY");

  out.tag(EncodingTag::Conv);
  out.i32(desc.convXSize);
  out.i32(desc.convYSize);
  out.i32(desc.inChannels);
  out.i32(desc.outChannels);
  // Dilation is explicit even when it is 1. Omitting default-valued dilation
  // here would make future dilated kernels unsafe to reuse.
  out.i32(desc.dilationX);
  out.i32(desc.dilationY);

  if(emitOperator) {
    ArchitectureOpDesc op = emptyOp(ArchitectureOpKind::Conv2D);
    op.runtimeDependencies = OP_RUNTIME_BATCH | OP_RUNTIME_SPATIAL_XY;
    op.inChannels = desc.inChannels;
    op.outChannels = desc.outChannels;
    op.kernelX = desc.convXSize;
    op.kernelY = desc.convYSize;
    op.dilationX = desc.dilationX;
    op.dilationY = desc.dilationY;
    appendOp(result,op);
  }
}

static void encodeBatchNorm(CanonicalWriter& out, const BatchNormLayerDesc& desc) {
  requirePositive(desc.numChannels,"batchNorm.numChannels");
  requireFinitePositive(desc.epsilon,"batchNorm.epsilon");
  out.tag(EncodingTag::BatchNorm);
  out.i32(desc.numChannels);
  out.u32(getFloatBits(desc.epsilon));
  out.boolean(desc.hasScale);
  out.boolean(desc.hasBias);
}

static void encodeActivation(CanonicalWriter& out, const ActivationLayerDesc& desc) {
  out.tag(EncodingTag::Activation);
  out.i32(desc.activation);
}

static void encodeBatchNormActivation(
  CanonicalWriter& out,
  const BatchNormLayerDesc& norm,
  const ActivationLayerDesc& activation,
  ArchitectureDesc& result
) {
  encodeBatchNorm(out,norm);
  encodeActivation(out,activation);

  ArchitectureOpDesc op = emptyOp(ArchitectureOpKind::BatchNormActivation);
  op.runtimeDependencies = OP_RUNTIME_BATCH | OP_RUNTIME_SPATIAL_AREA | OP_RUNTIME_MASK;
  op.flags = activationFlags(activation.activation);
  if(norm.hasScale)
    op.flags |= OP_FLAG_HAS_SCALE;
  if(norm.hasBias)
    op.flags |= OP_FLAG_HAS_BIAS;
  op.inChannels = norm.numChannels;
  op.outChannels = norm.numChannels;
  op.semanticScalar0Bits = getFloatBits(norm.epsilon);
  appendOp(result,op);
}

static void encodeMatMul(
  CanonicalWriter& out,
  const MatMulLayerDesc& desc,
  ArchitectureDesc& result,
  bool emitOperator
) {
  requirePositive(desc.inChannels,"matMul.inChannels");
  requirePositive(desc.outChannels,"matMul.outChannels");
  out.tag(EncodingTag::MatMul);
  out.i32(desc.inChannels);
  out.i32(desc.outChannels);

  if(emitOperator) {
    ArchitectureOpDesc op = emptyOp(ArchitectureOpKind::MatMul);
    op.runtimeDependencies = OP_RUNTIME_BATCH;
    op.inChannels = desc.inChannels;
    op.outChannels = desc.outChannels;
    appendOp(result,op);
  }
}

static void encodeMatBias(
  CanonicalWriter& out,
  const MatBiasLayerDesc& desc,
  ArchitectureDesc& result
) {
  requirePositive(desc.numChannels,"matBias.numChannels");
  out.tag(EncodingTag::MatBias);
  out.i32(desc.numChannels);

  ArchitectureOpDesc op = emptyOp(ArchitectureOpKind::MatBias);
  op.runtimeDependencies = OP_RUNTIME_BATCH;
  op.inChannels = desc.numChannels;
  op.outChannels = desc.numChannels;
  appendOp(result,op);
}

static void encodeRMSNorm(CanonicalWriter& out, const TransformerRMSNormDesc& desc) {
  requirePositive(desc.numChannels,"rmsNorm.numChannels");
  requireFinitePositive(desc.epsilon,"rmsNorm.epsilon");
  out.tag(EncodingTag::RMSNorm);
  out.i32(desc.numChannels);
  out.u32(getFloatBits(desc.epsilon));
}

static void encodeAttention(
  CanonicalWriter& out,
  const TransformerAttentionDesc& desc,
  ArchitectureDesc& result
) {
  requirePositive(desc.numHeads,"attention.numHeads");
  requirePositive(desc.numKVHeads,"attention.numKVHeads");
  requirePositive(desc.qHeadDim,"attention.qHeadDim");
  requirePositive(desc.vHeadDim,"attention.vHeadDim");

  out.tag(EncodingTag::TransformerAttention);
  out.i32(desc.numHeads);
  out.i32(desc.numKVHeads);
  out.i32(desc.qHeadDim);
  out.i32(desc.vHeadDim);
  out.boolean(desc.useRope);
  out.boolean(desc.learnableRope);
  encodeRMSNorm(out,desc.preLN);
  encodeMatMul(out,desc.qProj,result,false);
  encodeMatMul(out,desc.kProj,result,false);
  encodeMatMul(out,desc.vProj,result,false);
  encodeMatMul(out,desc.outProj,result,false);
  if(desc.useRope) {
    if(desc.learnableRope) {
      out.i32(desc.ropeNumKVHeads);
      out.i32(desc.ropeNumPairs);
      out.i32(2);  // Learned 2D frequency tensor's final coordinate dimension.
      out.u32((uint32_t)desc.ropeFreqs.size());
    }
    else {
      requireFinitePositive(desc.ropeTheta,"attention.ropeTheta");
      out.u32(getFloatBits(desc.ropeTheta));
    }
  }

  ArchitectureOpDesc op = emptyOp(ArchitectureOpKind::TransformerAttention);
  op.runtimeDependencies = OP_RUNTIME_BATCH | OP_RUNTIME_MASK |
    (desc.useRope ? OP_RUNTIME_SPATIAL_XY : OP_RUNTIME_SPATIAL_AREA);
  if(desc.useRope)
    op.flags |= OP_FLAG_USE_ROPE;
  if(desc.learnableRope)
    op.flags |= OP_FLAG_LEARNABLE_ROPE;
  op.inChannels = desc.preLN.numChannels;
  op.outChannels = desc.outProj.outChannels;
  op.numHeads = desc.numHeads;
  op.numKVHeads = desc.numKVHeads;
  op.qHeadDim = desc.qHeadDim;
  op.vHeadDim = desc.vHeadDim;
  op.semanticScalar0Bits = getFloatBits(desc.preLN.epsilon);
  op.semanticScalar1Bits = desc.useRope && !desc.learnableRope ? getFloatBits(desc.ropeTheta) : 0;
  appendOp(result,op);
}

static void encodeFFN(
  CanonicalWriter& out,
  const TransformerFFNDesc& desc,
  ArchitectureDesc& result
) {
  requirePositive(desc.numChannels,"ffn.numChannels");
  requirePositive(desc.ffnChannels,"ffn.ffnChannels");
  out.tag(EncodingTag::TransformerFFN);
  out.i32(desc.numChannels);
  out.i32(desc.ffnChannels);
  out.boolean(desc.useSwiGLU);
  encodeRMSNorm(out,desc.preLN);
  encodeMatMul(out,desc.linear1,result,false);
  if(desc.useSwiGLU)
    encodeMatMul(out,desc.linearGate,result,false);
  encodeMatMul(out,desc.linear2,result,false);

  ArchitectureOpDesc op = emptyOp(ArchitectureOpKind::TransformerFFN);
  // This describes the complete pre-norm FFN block, including its masked RMS
  // norm and residual. Finer-grained kernel recipes may still expose mask-free
  // inner GEMMs as substeps, but a whole-block tactic must key on mask mode.
  op.runtimeDependencies = OP_RUNTIME_BATCH | OP_RUNTIME_SPATIAL_AREA | OP_RUNTIME_MASK;
  if(desc.useSwiGLU)
    op.flags |= OP_FLAG_USE_SWIGLU;
  op.inChannels = desc.numChannels;
  op.outChannels = desc.numChannels;
  op.auxiliaryChannels = desc.ffnChannels;
  op.semanticScalar0Bits = getFloatBits(desc.preLN.epsilon);
  appendOp(result,op);
}

static void encodeBlockStack(
  CanonicalWriter& out,
  const vector<pair<int,unique_ptr_void>>& blocks,
  ArchitectureDesc& result
);

static void encodeResidualBlock(
  CanonicalWriter& out,
  const ResidualBlockDesc& desc,
  ArchitectureDesc& result
) {
  encodeBatchNormActivation(out,desc.preBN,desc.preActivation,result);
  encodeConv(out,desc.regularConv,result,true);
  encodeBatchNormActivation(out,desc.midBN,desc.midActivation,result);
  encodeConv(out,desc.finalConv,result,true);
}

static void encodeGPoolBlock(
  CanonicalWriter& out,
  const GlobalPoolingResidualBlockDesc& desc,
  ArchitectureDesc& result
) {
  out.i32(desc.version);
  encodeBatchNormActivation(out,desc.preBN,desc.preActivation,result);
  encodeConv(out,desc.regularConv,result,true);
  encodeConv(out,desc.gpoolConv,result,true);
  encodeBatchNormActivation(out,desc.gpoolBN,desc.gpoolActivation,result);
  encodeMatMul(out,desc.gpoolToBiasMul,result,true);
  encodeBatchNormActivation(out,desc.midBN,desc.midActivation,result);
  encodeConv(out,desc.finalConv,result,true);
}

static void encodeNestedBlock(
  CanonicalWriter& out,
  const NestedBottleneckResidualBlockDesc& desc,
  ArchitectureDesc& result
) {
  out.i32(desc.numBlocks);
  out.u32((uint32_t)desc.blocks.size());
  encodeBatchNormActivation(out,desc.preBN,desc.preActivation,result);
  encodeConv(out,desc.preConv,result,true);
  encodeBlockStack(out,desc.blocks,result);
  encodeBatchNormActivation(out,desc.postBN,desc.postActivation,result);
  encodeConv(out,desc.postConv,result,true);
}

static void encodeBlockStack(
  CanonicalWriter& out,
  const vector<pair<int,unique_ptr_void>>& blocks,
  ArchitectureDesc& result
) {
  for(const auto& entry: blocks) {
    out.tag(EncodingTag::Block);
    out.i32(entry.first);
    if(entry.second.get() == nullptr)
      throw StringError("Invalid architecture descriptor: null block descriptor");
    if(entry.first == ORDINARY_BLOCK_KIND)
      encodeResidualBlock(out,*((const ResidualBlockDesc*)entry.second.get()),result);
    else if(entry.first == GLOBAL_POOLING_BLOCK_KIND)
      encodeGPoolBlock(out,*((const GlobalPoolingResidualBlockDesc*)entry.second.get()),result);
    else if(entry.first == NESTED_BOTTLENECK_BLOCK_KIND)
      encodeNestedBlock(out,*((const NestedBottleneckResidualBlockDesc*)entry.second.get()),result);
    else if(entry.first == TRANSFORMER_ATTENTION_BLOCK_KIND)
      encodeAttention(out,*((const TransformerAttentionDesc*)entry.second.get()),result);
    else if(entry.first == TRANSFORMER_FFN_BLOCK_KIND)
      encodeFFN(out,*((const TransformerFFNDesc*)entry.second.get()),result);
    else
      throw StringError("Invalid architecture descriptor: unknown block kind");
  }
}

static uint8_t outerNormMask(const ModelDesc& model, bool useScale) {
  uint8_t mask = 0;
  if(useScale ? model.trunk.trunkTipBN.hasScale : model.trunk.trunkTipBN.hasBias)
    mask |= OUTER_NORM_TRUNK_TIP;
  if(useScale ? model.policyHead.g1BN.hasScale : model.policyHead.g1BN.hasBias)
    mask |= OUTER_NORM_POLICY_G1;
  if(useScale ? model.policyHead.p1BN.hasScale : model.policyHead.p1BN.hasBias)
    mask |= OUTER_NORM_POLICY_P1;
  if(useScale ? model.valueHead.v1BN.hasScale : model.valueHead.v1BN.hasBias)
    mask |= OUTER_NORM_VALUE_V1;
  return mask;
}

}  // namespace

bool ArchitectureSignature::operator==(const ArchitectureSignature& other) const {
  return schemaVersion == other.schemaVersion && digest == other.digest;
}

bool ArchitectureSignature::operator!=(const ArchitectureSignature& other) const {
  return !(*this == other);
}

string ArchitectureSignature::toHex() const {
  static const char* digits = "0123456789abcdef";
  string result(digest.size() * 2,'0');
  for(size_t i = 0; i < digest.size(); i++) {
    result[i * 2] = digits[digest[i] >> 4];
    result[i * 2 + 1] = digits[digest[i] & 0x0F];
  }
  return result;
}

uint32_t getFloatBits(float value) {
  static_assert(sizeof(float) == sizeof(uint32_t),"32-bit float required");
  uint32_t bits;
  memcpy(&bits,&value,sizeof(bits));
  return bits;
}

ArchitectureDesc buildArchitectureDesc(const ModelDesc& model) {
  if(model.onnxHeader.isOnnx)
    throw StringError("Cannot build a complete architecture signature from an ONNX header-only ModelDesc");

  ArchitectureDesc result{};
  result.signature.schemaVersion = CANONICAL_ARCHITECTURE_SCHEMA_VERSION;
  result.outerBatchNormScaleMask = outerNormMask(model,true);
  result.outerBatchNormBiasMask = outerNormMask(model,false);

  CanonicalWriter out;
  static const char magic[] = "KataGoCanonicalArchitecture";
  out.raw(magic,sizeof(magic) - 1);
  out.u32(CANONICAL_ARCHITECTURE_SCHEMA_VERSION);
  out.tag(EncodingTag::Model);
  out.i32(model.version);
  out.i32(model.numInputChannels);
  out.i32(model.numInputGlobalChannels);
  out.i32(model.numValueChannels);
  out.i32(model.numScoreValueChannels);
  out.i32(model.numOwnershipChannels);
  out.u8(result.outerBatchNormScaleMask);
  out.u8(result.outerBatchNormBiasMask);

  const TrunkDesc& trunk = model.trunk;
  out.tag(EncodingTag::Trunk);
  out.i32(trunk.version);
  out.i32(trunk.numBlocks);
  out.u32((uint32_t)trunk.blocks.size());
  out.i32(trunk.trunkNumChannels);
  out.i32(trunk.midNumChannels);
  out.i32(trunk.regularNumChannels);
  out.i32(trunk.dilatedNumChannels);
  out.i32(trunk.gpoolNumChannels);
  encodeConv(out,trunk.initialConv,result,true);
  encodeMatMul(out,trunk.initialMatMul,result,true);
  encodeBlockStack(out,trunk.blocks,result);
  encodeBatchNormActivation(out,trunk.trunkTipBN,trunk.trunkTipActivation,result);

  const PolicyHeadDesc& policy = model.policyHead;
  out.tag(EncodingTag::PolicyHead);
  out.i32(policy.version);
  encodeConv(out,policy.p1Conv,result,true);
  encodeConv(out,policy.g1Conv,result,true);
  encodeBatchNormActivation(out,policy.g1BN,policy.g1Activation,result);
  encodeMatMul(out,policy.gpoolToBiasMul,result,true);
  encodeBatchNormActivation(out,policy.p1BN,policy.p1Activation,result);
  encodeConv(out,policy.p2Conv,result,true);
  encodeMatMul(out,policy.gpoolToPassMul,result,true);

  const ValueHeadDesc& value = model.valueHead;
  out.tag(EncodingTag::ValueHead);
  out.i32(value.version);
  encodeConv(out,value.v1Conv,result,true);
  encodeBatchNormActivation(out,value.v1BN,value.v1Activation,result);
  encodeMatMul(out,value.v2Mul,result,true);
  encodeMatBias(out,value.v2Bias,result);
  encodeActivation(out,value.v2Activation);
  encodeMatMul(out,value.v3Mul,result,true);
  encodeMatBias(out,value.v3Bias,result);
  encodeMatMul(out,value.sv3Mul,result,true);
  encodeMatBias(out,value.sv3Bias,result);
  encodeConv(out,value.vOwnershipConv,result,true);

  result.canonicalEncoding = std::move(out.bytes);
  SHA2::get256(
    result.canonicalEncoding.data(),
    result.canonicalEncoding.size(),
    result.signature.digest.data()
  );
  return result;
}

ArchitectureSignature getArchitectureSignature(const ModelDesc& model) {
  return buildArchitectureDesc(model).signature;
}

ModelProvenance getModelProvenance(const ModelDesc& model) {
  ModelProvenance result;
  result.modelName = model.name;
  result.artifactSha256 = model.sha256;
  result.modelConfigSha256 = model.onnxHeader.model_config_sha256;
  result.sourceWasOnnx = model.onnxHeader.isOnnx;
  return result;
}

}  // namespace NeuralNetArchitecture

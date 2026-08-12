#include "../neuralnet/cudabackend_transformer_winner.h"

#include <algorithm>
#include <array>
#include <iterator>

using namespace CudaOpRegistry;
using namespace NeuralNetArchitecture;

namespace CudaTransformerWinner {
namespace {

constexpr uint64_t ATTENTION_FAMILY = 0x4B41544154544E31ULL; // KATATTN1
constexpr uint64_t FFN_FAMILY = 0x4B415446464E3031ULL;       // KATFFN01
constexpr uint32_t RMS_EPSILON_1E6_BITS = 0x358637BDu;
constexpr uint32_t C384_ROWS_PER_BATCH = 225;
constexpr uint32_t C384_MAX_TOKEN_ROWS = 1u << 20;
constexpr uint32_t C384_MAX_BATCH = C384_MAX_TOKEN_ROWS / C384_ROWS_PER_BATCH;

enum AttentionVariant : uint64_t {
  ATTENTION_SQUARE_GENERIC = 1,
  ATTENTION_SQUARE_LEARNED_ROPE = 2,
  ATTENTION_SQUARE_MASK_SAFE = 3,
  ATTENTION_SQUARE_LEARNED_ROPE_MASK_SAFE = 4,
  ATTENTION_C256_SM120 = 5,
  ATTENTION_C256_DYNAMIC_SM120 = 6,
  ATTENTION_C256_B36_FA4_SM120 = 7,
  ATTENTION_C384_DYNAMIC_SM120 = 8,
};

enum FfnVariant : uint64_t {
  FFN_GENERIC = 1,
  FFN_C256_F768_SM120 = 2,
  FFN_C256_F768_DYNAMIC_SM120 = 3,
  FFN_C384_F1024_DYNAMIC_SM120 = 4,
};

struct RegistrationContext {
  DeviceCapability device;
};

bool isHalfNhwcNoMask(const CapabilityKey& key) {
  return key.inputType == NumericType::Float16 &&
    key.outputType == NumericType::Float16 &&
    key.computeType == NumericType::Float32 &&
    (key.layout == TensorLayout::NHWC || key.layout == TensorLayout::BSH) &&
    key.maskMode == MaskMode::None && key.batchSize > 0 &&
    key.spatialArea > 0;
}

bool isSm120(const OpRequest& request, const RegistrationContext& context) {
  return request.key.deviceComputeCapability == 120 &&
    context.device.computeCapability == 120 && context.device.warpSize == 32 &&
    context.device.specializedSm120KernelsAvailable;
}

bool isSquareMha(const CapabilityKey& key) {
  return key.inChannels > 0 && key.inChannels == key.outChannels &&
    key.numHeads > 0 && key.numHeads == key.numKVHeads &&
    key.qHeadDim > 0 && key.qHeadDim == key.vHeadDim &&
    key.numHeads * key.qHeadDim == key.inChannels;
}

bool hasLearnedRope(const CapabilityKey& key) {
  return (key.flags & OP_FLAG_USE_ROPE) != 0 &&
    (key.flags & OP_FLAG_LEARNABLE_ROPE) != 0;
}

bool isC256Attention(const CapabilityKey& key) {
  return isSquareMha(key) && key.inChannels == 256 && key.numHeads == 8 &&
    key.numKVHeads == 8 && key.qHeadDim == 32 && key.vHeadDim == 32 &&
    key.auxiliaryChannels == 16 &&
    hasLearnedRope(key);
}

bool isC256F768(const CapabilityKey& key) {
  return key.kind == ArchitectureOpKind::TransformerFFN &&
    key.inChannels == 256 && key.outChannels == 256 &&
    key.auxiliaryChannels == 768 && (key.flags & OP_FLAG_USE_SWIGLU) != 0;
}

bool isC384Attention(const CapabilityKey& key) {
  return isSquareMha(key) && key.inChannels == 384 && key.numHeads == 12 &&
    key.numKVHeads == 12 && key.qHeadDim == 32 && key.vHeadDim == 32 &&
    key.auxiliaryChannels == 16 && hasLearnedRope(key);
}

bool isC384F1024(const CapabilityKey& key) {
  return key.kind == ArchitectureOpKind::TransformerFFN &&
    key.inChannels == 384 && key.outChannels == 384 &&
    key.auxiliaryChannels == 1024 && (key.flags & OP_FLAG_USE_SWIGLU) != 0;
}

bool validatedDynamicRows(const CapabilityKey& key) {
  const int64_t rows = (int64_t)key.batchSize * (int64_t)key.spatialArea;
  constexpr int rowsMeasured[] = {7200,8100,9000,14400,28800};
  if(rows <= 0 || rows > 0x7FFFFFFF)
    return false;
  return std::find(
    std::begin(rowsMeasured),std::end(rowsMeasured),(int)rows
  ) != std::end(rowsMeasured);
}

bool validatedC384Rows(const CapabilityKey& key) {
  const int64_t rows = (int64_t)key.batchSize * (int64_t)key.spatialArea;
  if(key.spatialArea != (int)C384_ROWS_PER_BATCH)
    return false;
  // All staged C384 kernels accept dynamic M and retain a construction-time
  // handle for the requested maximum batch. The implementation limit is a
  // token-row resource bound, not the largest sampled benchmark batch.
  return key.batchSize >= 1 && rows > 0 &&
    rows <= (int64_t)C384_MAX_TOKEN_ROWS;
}

bool validatedC384AttentionRuntime(const CapabilityKey& key) {
  // Attention declares OP_RUNTIME_SPATIAL_XY as well as area because learned
  // RoPE depends on the board axes, so require the exact staged geometry.
  return key.boardX == 15 && key.boardY == 15 && validatedC384Rows(key);
}

bool validatedC384FfnRuntime(const CapabilityKey& key) {
  // FFN declares only OP_RUNTIME_SPATIAL_AREA. Its canonical capability key
  // intentionally zeros boardX/boardY, so do not manufacture an axis
  // dependency that the operator does not have.
  return validatedC384Rows(key);
}

SupportClass matchAttentionSquare(const OpRequest& request, const void*) {
  const CapabilityKey& key = request.key;
  return key.kind == ArchitectureOpKind::TransformerAttention &&
    isHalfNhwcNoMask(key) && isSquareMha(key) ?
    SupportClass::CompatibleOnly : SupportClass::Unsupported;
}

SupportClass matchAttentionSquareLearnedRope(const OpRequest& request, const void*) {
  const CapabilityKey& key = request.key;
  return key.kind == ArchitectureOpKind::TransformerAttention &&
    isHalfNhwcNoMask(key) && isSquareMha(key) && hasLearnedRope(key) ?
    SupportClass::CompatibleOnly : SupportClass::Unsupported;
}

SupportClass matchAttentionSquareMaskSafe(const OpRequest& request, const void*) {
  const CapabilityKey& key = request.key;
  const bool fp16 = key.inputType == NumericType::Float16 &&
    key.outputType == NumericType::Float16 && key.computeType == NumericType::Float32;
  const bool layout = key.layout == TensorLayout::NHWC || key.layout == TensorLayout::BSH;
  return key.kind == ArchitectureOpKind::TransformerAttention && fp16 && layout &&
    key.maskMode == MaskMode::Dense && key.batchSize > 0 && key.spatialArea > 0 &&
    isSquareMha(key) ? SupportClass::CompatibleOnly : SupportClass::Unsupported;
}

SupportClass matchAttentionSquareLearnedRopeMaskSafe(const OpRequest& request, const void* userData) {
  return matchAttentionSquareMaskSafe(request,userData) != SupportClass::Unsupported &&
    hasLearnedRope(request.key) ? SupportClass::CompatibleOnly : SupportClass::Unsupported;
}

SupportClass matchAttentionC256(const OpRequest& request, const void* userData) {
  const RegistrationContext& context = *(const RegistrationContext*)userData;
  return isHalfNhwcNoMask(request.key) && isC256Attention(request.key) &&
    isSm120(request,context) ? SupportClass::CompatibleOnly : SupportClass::Unsupported;
}

SupportClass matchAttentionDynamic(const OpRequest& request, const void* userData) {
  const RegistrationContext& context = *(const RegistrationContext*)userData;
  return isHalfNhwcNoMask(request.key) && isC256Attention(request.key) &&
    isSm120(request,context) && validatedDynamicRows(request.key) ?
    SupportClass::CompatibleOnly : SupportClass::Unsupported;
}

SupportClass matchAttentionExact(const OpRequest& request, const void* userData) {
  const RegistrationContext& context = *(const RegistrationContext*)userData;
  const CapabilityKey& key = request.key;
  // Runtime-library versions remain part of the diagnostic plan fingerprint,
  // but do not gate this embedded SM120 kernel. Successfully loading this
  // CUDA binary and creating its CUDA/cuBLAS/cuDNN handles is the runtime ABI
  // check; operator safety is determined by the exact hardware/resource/shape
  // predicates below.
  if(!isHalfNhwcNoMask(key) || !isC256Attention(key) || !isSm120(request,context) ||
     context.device.sharedBytesPerBlockOptin < 101376 || key.batchSize != 36 ||
     key.boardX != 15 || key.boardY != 15 || key.spatialArea != 225)
    return SupportClass::Unsupported;
  // Stream count is evaluator topology, not an operator-kernel property. S1
  // replay and each leg of S2 execute this same certified recipe on one owned
  // handle stream; the outer benchmark separately verifies distinct streams.
  return SupportClass::CertifiedFast;
}

SupportClass matchAttentionC384Dynamic(const OpRequest& request, const void* userData) {
  const RegistrationContext& context = *(const RegistrationContext*)userData;
  return isHalfNhwcNoMask(request.key) && isC384Attention(request.key) &&
    isSm120(request,context) && validatedC384AttentionRuntime(request.key) ?
    SupportClass::CompatibleOnly : SupportClass::Unsupported;
}

SupportClass matchFfnGeneric(const OpRequest& request, const void*) {
  const CapabilityKey& key = request.key;
  return key.kind == ArchitectureOpKind::TransformerFFN &&
    isHalfNhwcNoMask(key) && key.inChannels > 0 &&
    key.inChannels == key.outChannels && key.auxiliaryChannels > 0 &&
    (key.flags & OP_FLAG_USE_SWIGLU) != 0 ?
    SupportClass::CompatibleOnly : SupportClass::Unsupported;
}

SupportClass matchFfnC256(const OpRequest& request, const void* userData) {
  const RegistrationContext& context = *(const RegistrationContext*)userData;
  return isHalfNhwcNoMask(request.key) && isC256F768(request.key) &&
    isSm120(request,context) ? SupportClass::CompatibleOnly : SupportClass::Unsupported;
}

SupportClass matchFfnDynamic(const OpRequest& request, const void* userData) {
  const RegistrationContext& context = *(const RegistrationContext*)userData;
  return isHalfNhwcNoMask(request.key) && isC256F768(request.key) &&
    isSm120(request,context) && validatedDynamicRows(request.key) ?
    SupportClass::CompatibleOnly : SupportClass::Unsupported;
}

SupportClass matchFfnC384Dynamic(const OpRequest& request, const void* userData) {
  const RegistrationContext& context = *(const RegistrationContext*)userData;
  return isHalfNhwcNoMask(request.key) && isC384F1024(request.key) &&
    isSm120(request,context) && validatedC384FfnRuntime(request.key) ?
    SupportClass::CompatibleOnly : SupportClass::Unsupported;
}

bool prepareNoAllocation(const OpRequest&, PreparedOp& prepared, void*) {
  prepared.workspaceAlignment = 0;
  prepared.workspaceBytes = 0;
  prepared.implementationCookie = 0;
  return true;
}

void registerTactic(
  Registry& registry,
  uint64_t family,
  uint64_t variant,
  int32_t priority,
  const RecipeFingerprint& recipe,
  MatchTacticFn match,
  RegistrationContext* context
) {
  TacticRegistration registration{};
  registration.id = TacticId{family,variant};
  registration.recipe = recipe;
  registration.priority = priority;
  registration.match = match;
  registration.prepare = prepareNoAllocation;
  registration.userData = context;
  registry.registerTactic(registration);
}

void registerTactic(
  Registry& registry,
  uint64_t family,
  uint64_t variant,
  int32_t priority,
  const char* recipe,
  MatchTacticFn match,
  RegistrationContext* context
) {
  registerTactic(
    registry,family,variant,priority,fingerprintRecipe(recipe),match,context);
}

AttentionRecipe attentionRecipe(const PreparedOp* operation) {
  AttentionRecipe recipe;
  if(operation == nullptr || operation->tactic.family != ATTENTION_FAMILY)
    return recipe;
  switch(operation->tactic.variant) {
  case ATTENTION_C384_DYNAMIC_SM120:
    recipe.rmsNorm = RmsNormTactic::Sm120C384Warp4Vec4x3;
    recipe.outProjection = ResidualTactic::CublasHgemmBetaOne;
    recipe.rope = RopeTactic::LearnedHalf2;
    recipe.planarQkv = PlanarQkvTactic::CublasHgemmStridedBatchedSquare;
    break;
  case ATTENTION_C256_B36_FA4_SM120:
    recipe.rope = RopeTactic::LearnedHalf2;
    recipe.qkvRope = QkvRopeTactic::Sm120C256H8D32M128N128K32S3;
    recipe.attention = AttentionTactic::Fa4Sm120B36S225Tm128Tn128S1Both16;
    [[fallthrough]];
  case ATTENTION_C256_DYNAMIC_SM120:
    recipe.outProjection = ResidualTactic::Sm120M128N128K32S3Sw1;
    [[fallthrough]];
  case ATTENTION_C256_SM120:
    recipe.rmsNorm = RmsNormTactic::Sm120C256Warp4Vec8;
    [[fallthrough]];
  case ATTENTION_SQUARE_LEARNED_ROPE:
    recipe.rope = RopeTactic::LearnedHalf2;
    [[fallthrough]];
  case ATTENTION_SQUARE_GENERIC:
    recipe.planarQkv = PlanarQkvTactic::CublasHgemmStridedBatchedSquare;
    if(recipe.outProjection == ResidualTactic::GenericAdd)
      recipe.outProjection = ResidualTactic::CublasHgemmBetaOne;
    break;
  case ATTENTION_SQUARE_LEARNED_ROPE_MASK_SAFE:
    recipe.rope = RopeTactic::LearnedHalf2;
    [[fallthrough]];
  case ATTENTION_SQUARE_MASK_SAFE:
    recipe.planarQkv = PlanarQkvTactic::CublasHgemmStridedBatchedSquare;
    break;
  default:
    break;
  }
  return recipe;
}

FfnRecipe ffnRecipe(const PreparedOp* operation) {
  FfnRecipe recipe;
  if(operation == nullptr || operation->tactic.family != FFN_FAMILY)
    return recipe;
  switch(operation->tactic.variant) {
  case FFN_C384_F1024_DYNAMIC_SM120:
    recipe.rmsNorm = RmsNormTactic::Sm120C384Warp4Vec4x3;
    recipe.dualFfn = DualFfnTactic::Sm120C384F1024M128N64K32S3Sw4;
    recipe.downProjection = ResidualTactic::CublasHgemmBetaOne;
    break;
  case FFN_C256_F768_DYNAMIC_SM120:
    recipe.dualFfn = DualFfnTactic::Sm120C256F768M128N64K32S3Sw4;
    recipe.downProjection = ResidualTactic::Sm120M128N128K32S3Sw1;
    [[fallthrough]];
  case FFN_C256_F768_SM120:
    recipe.rmsNorm = RmsNormTactic::Sm120C256Warp4Vec8;
    [[fallthrough]];
  case FFN_GENERIC:
    if(recipe.downProjection == ResidualTactic::GenericAdd)
      recipe.downProjection = ResidualTactic::CublasHgemmBetaOne;
    break;
  default:
    break;
  }
  return recipe;
}

const PreparedRecord* findRecord(
  const PreparedPlan& plan,
  ArchitectureOpKind kind,
  uint32_t topologyIndex
) {
  for(const PreparedRecord& record: plan.records) {
    if(record.request.topologyIndex == topologyIndex && record.request.key.kind == kind)
      return &record;
  }
  return nullptr;
}

const PreparedRecord* findRecord(
  const PreparedPlan& plan,
  ArchitectureOpKind kind,
  const CapabilityKey& key
) {
  for(const PreparedRecord& record: plan.records) {
    if(record.request.key.kind == kind && record.request.key == key)
      return &record;
  }
  return nullptr;
}

}  // namespace

bool AttentionRecipe::hasPreparedOptimization() const {
  return planarQkv != PlanarQkvTactic::Disabled ||
    rmsNorm != RmsNormTactic::GenericHalf || rope != RopeTactic::Generic ||
    qkvRope != QkvRopeTactic::Disabled || attention != AttentionTactic::Generic ||
    outProjection != ResidualTactic::GenericAdd;
}

bool FfnRecipe::hasPreparedOptimization() const {
  return rmsNorm != RmsNormTactic::GenericHalf ||
    dualFfn != DualFfnTactic::Disabled ||
    downProjection != ResidualTactic::GenericAdd;
}

uint64_t makeRuntimeLibraryFingerprint(
  int cudaRuntimeVersion,
  int cudaDriverVersion,
  int cublasVersion,
  std::size_t cudnnVersion
) {
  if(cudaRuntimeVersion <= 0 || cudaDriverVersion <= 0 ||
     cublasVersion <= 0 || cudnnVersion == 0)
    return 0;
  uint64_t hash = 1469598103934665603ULL;
  const uint64_t values[] = {
    (uint64_t)(uint32_t)cudaRuntimeVersion,
    (uint64_t)(uint32_t)cudaDriverVersion,
    (uint64_t)(uint32_t)cublasVersion,
    (uint64_t)cudnnVersion,
  };
  for(uint64_t value: values) {
    for(int byte = 0; byte < 8; byte++) {
      hash ^= (value >> (byte * 8)) & 0xFFu;
      hash *= 1099511628211ULL;
    }
  }
  return hash == 0 ? 1 : hash;
}

const C384RuntimeGatePolicy& productionC384RuntimeGatePolicy() {
  // Balanced measurements through B128 select dual FFN for every valid actual
  // batch, and its dynamic-M handle remains structurally valid up to the
  // configured maximum (bounded by C384_MAX_TOKEN_ROWS). RMS is positive from
  // B2 upward. B1 also measured positive in a single evaluator, but evaluator-
  // local lane counts cannot see other evaluator instances sharing the GPU;
  // B1 therefore uses dual-only for every topology. Standalone out/down
  // residual tactics never cleared the required +2% margin, so their ranges
  // remain disabled and cuBLAS beta=1 is used.
  static const C384RuntimeGatePolicy policy = []() {
    C384RuntimeGatePolicy value;
    value.rowsPerBatch = C384_ROWS_PER_BATCH;
    value.rmsNorm.conservative = C384RuntimeBatchRange{2,C384_MAX_BATCH};
    value.rmsNorm.exactlyTwoSameGpuLanes = C384RuntimeBatchRange{2,C384_MAX_BATCH};
    value.dualFfn.conservative = C384RuntimeBatchRange{1,C384_MAX_BATCH};
    value.dualFfn.exactlyTwoSameGpuLanes = C384RuntimeBatchRange{1,C384_MAX_BATCH};
    return value;
  }();
  return policy;
}

RecipeFingerprint fingerprintRecipeWithC384RuntimeGate(
  const std::string& stableTacticEncoding,
  const C384RuntimeGatePolicy& policy
) {
  std::vector<uint8_t> bytes;
  const char domain[] = "katago-c384-runtime-gate-recipe";
  bytes.insert(bytes.end(),domain,domain + sizeof(domain) - 1);
  const auto appendU32 = [&](uint32_t value) {
    for(int byte = 0; byte < 4; byte++)
      bytes.push_back((uint8_t)((value >> (byte * 8)) & 0xFFu));
  };
  const auto appendRange = [&](const C384RuntimeBatchRange& range) {
    appendU32(range.minInclusive);
    appendU32(range.maxInclusive);
  };
  const auto appendPiece = [&](const C384RuntimePiecePolicy& piece) {
    appendRange(piece.conservative);
    appendRange(piece.exactlyTwoSameGpuLanes);
  };

  appendU32(1);  // Canonical C384 gate encoding schema.
  appendU32((uint32_t)stableTacticEncoding.size());
  bytes.insert(
    bytes.end(),stableTacticEncoding.begin(),stableTacticEncoding.end());
  appendU32(policy.rowsPerBatch);
  appendPiece(policy.rmsNorm);
  appendPiece(policy.dualFfn);
  appendPiece(policy.outProjection);
  appendPiece(policy.downProjection);
  return fingerprintRecipe(bytes.data(),bytes.size());
}

bool shouldUseC384RuntimePiece(
  C384RuntimePiece piece,
  int actualRows,
  int sameGpuEvaluatorConcurrency,
  const C384RuntimeGatePolicy& policy
) {
  if(actualRows <= 0 || sameGpuEvaluatorConcurrency <= 0 ||
     policy.rowsPerBatch == 0 ||
     (uint32_t)actualRows % policy.rowsPerBatch != 0)
    return false;

  const C384RuntimePiecePolicy* piecePolicy = nullptr;
  switch(piece) {
  case C384RuntimePiece::RmsNorm:
    piecePolicy = &policy.rmsNorm;
    break;
  case C384RuntimePiece::DualFfn:
    piecePolicy = &policy.dualFfn;
    break;
  case C384RuntimePiece::OutProjection:
    piecePolicy = &policy.outProjection;
    break;
  case C384RuntimePiece::DownProjection:
    piecePolicy = &policy.downProjection;
    break;
  default:
    return false;
  }

  const C384RuntimeBatchRange& range = sameGpuEvaluatorConcurrency == 2 ?
    piecePolicy->exactlyTwoSameGpuLanes : piecePolicy->conservative;
  if(range.minInclusive == 0 || range.maxInclusive < range.minInclusive)
    return false;
  const uint32_t actualBatch = (uint32_t)actualRows / policy.rowsPerBatch;
  return actualBatch >= range.minInclusive &&
    actualBatch <= range.maxInclusive;
}

AttentionRecipe PreparedPlan::attentionFor(uint32_t topologyIndex) const {
  const PreparedRecord* record = findRecord(
    *this,ArchitectureOpKind::TransformerAttention,topologyIndex);
  return attentionRecipe(record != nullptr && record->found ? &record->operation : nullptr);
}

FfnRecipe PreparedPlan::ffnFor(uint32_t topologyIndex) const {
  const PreparedRecord* record = findRecord(
    *this,ArchitectureOpKind::TransformerFFN,topologyIndex);
  return ffnRecipe(record != nullptr && record->found ? &record->operation : nullptr);
}

AttentionRecipe PreparedPlan::attentionFor(const CapabilityKey& key) const {
  const PreparedRecord* record = findRecord(
    *this,ArchitectureOpKind::TransformerAttention,key);
  return attentionRecipe(record != nullptr && record->found ? &record->operation : nullptr);
}

FfnRecipe PreparedPlan::ffnFor(const CapabilityKey& key) const {
  const PreparedRecord* record = findRecord(
    *this,ArchitectureOpKind::TransformerFFN,key);
  return ffnRecipe(record != nullptr && record->found ? &record->operation : nullptr);
}

const ArchitectureSignature& int8QualifiedArchitectureSignature() {
  // Native v104: same reviewed model architecture plus the explicit-format
  // version fields that bind mandatory embedded quantization metadata.
  static const ArchitectureSignature signature{
    CANONICAL_ARCHITECTURE_SCHEMA_VERSION,
    std::array<uint8_t,32>{
      0xbb,0xfa,0x59,0x57,0xd8,0xd8,0x7f,0x12,
      0x25,0xfe,0x4e,0x67,0x9b,0xe0,0x55,0xff,
      0x4d,0xe4,0xc4,0x0c,0x4f,0x84,0x37,0xa1,
      0x9c,0x7b,0xe3,0x05,0x2c,0x23,0xf4,0x9b,
    }
  };
  return signature;
}

const ArchitectureSignature& int8LegacyImplicitArchitectureSignature() {
  // Native v102 compatibility identity. It is intentionally separate from
  // the default v104 contract and never aliases the v104 signature.
  static const ArchitectureSignature signature{
    CANONICAL_ARCHITECTURE_SCHEMA_VERSION,
    std::array<uint8_t,32>{
      0xad,0x02,0x66,0x14,0x45,0x5c,0x04,0x75,
      0xb3,0x19,0x97,0xf1,0xc5,0x45,0x2a,0xf9,
      0x9d,0x1e,0xb3,0x47,0x71,0x3f,0x77,0x95,
      0x06,0x71,0xfc,0x5d,0x1a,0x52,0x2f,0x24,
    }
  };
  return signature;
}

Int8ExperimentEligibility evaluateInt8ExperimentEligibility(const PreparedPlan& plan) {
  Int8ExperimentEligibility result;
  result.explicitV104ArchitectureSignatureMatches =
    plan.architecture == int8QualifiedArchitectureSignature();
  result.legacyV102ArchitectureSignatureMatches =
    plan.architecture == int8LegacyImplicitArchitectureSignature();
  result.architectureSignatureMatches =
    result.explicitV104ArchitectureSignatureMatches ||
    result.legacyV102ArchitectureSignatureMatches;
  std::vector<OpRequest> requests;
  std::vector<PreparedOp> prepared;
  requests.reserve(plan.records.size());
  prepared.reserve(plan.records.size());
  const bool runtimeLayoutEligible =
    plan.runtime.layout == TensorLayout::NHWC ||
    plan.runtime.layout == TensorLayout::BSH;
  result.runtimeContractEligible =
    plan.runtime.boardX == 15 && plan.runtime.boardY == 15 &&
    plan.runtime.maskMode == MaskMode::None &&
    plan.runtime.inputType == NumericType::Float16 &&
    plan.runtime.outputType == NumericType::Float16 &&
    plan.runtime.computeType == NumericType::Float32 && runtimeLayoutEligible &&
    plan.runtime.deviceComputeCapability == 120 && plan.runtime.batchSize > 0;
  result.allTransformerShapesEligible = true;
  result.allTransformerRecordsPrepared = true;
  for(const PreparedRecord& record: plan.records) {
    requests.push_back(record.request);
    prepared.push_back(record.operation);
    if(record.request.architecture != plan.architecture) {
      result.architectureSignatureMatches = false;
      result.explicitV104ArchitectureSignatureMatches = false;
      result.legacyV102ArchitectureSignatureMatches = false;
    }
    const CapabilityKey& key = record.request.key;
    if(key.kind == ArchitectureOpKind::TransformerAttention) {
      const AttentionRecipe recipe = plan.attentionFor(record.request.topologyIndex);
      const bool layoutEligible = key.layout == TensorLayout::NHWC ||
        key.layout == TensorLayout::BSH;
      const bool eligible = key.schemaVersion == CAPABILITY_KEY_SCHEMA_VERSION &&
        key.inputType == NumericType::Float16 &&
        key.outputType == NumericType::Float16 &&
        key.computeType == NumericType::Float32 && layoutEligible &&
        key.deviceComputeCapability == 120 && key.batchSize > 0 &&
        key.boardX == 15 && key.boardY == 15 &&
        key.spatialArea == 225 && key.inChannels == 256 &&
        key.outChannels == 256 && key.numHeads == 8 &&
        key.numKVHeads == 8 && key.qHeadDim == 32 && key.vHeadDim == 32 &&
        key.auxiliaryChannels == 16 && key.maskMode == MaskMode::None &&
        key.semanticScalar0Bits == RMS_EPSILON_1E6_BITS &&
        (key.flags & OP_FLAG_USE_ROPE) != 0 &&
        (key.flags & OP_FLAG_LEARNABLE_ROPE) != 0;
      const bool preparedForInt8 = record.found &&
        record.operation.support != SupportClass::Unsupported &&
        recipe.rmsNorm == RmsNormTactic::Sm120C256Warp4Vec8 &&
        recipe.rope == RopeTactic::LearnedHalf2;
      result.allTransformerShapesEligible =
        result.allTransformerShapesEligible && eligible;
      result.allTransformerRecordsPrepared =
        result.allTransformerRecordsPrepared && preparedForInt8;
      if(eligible)
        result.attentionCount++;
    }
    else if(key.kind == ArchitectureOpKind::TransformerFFN) {
      const FfnRecipe recipe = plan.ffnFor(record.request.topologyIndex);
      const bool layoutEligible = key.layout == TensorLayout::NHWC ||
        key.layout == TensorLayout::BSH;
      // FFN keys depend on spatial area, not the independent board axes.
      const bool eligible = key.schemaVersion == CAPABILITY_KEY_SCHEMA_VERSION &&
        key.inputType == NumericType::Float16 &&
        key.outputType == NumericType::Float16 &&
        key.computeType == NumericType::Float32 && layoutEligible &&
        key.deviceComputeCapability == 120 && key.batchSize > 0 &&
        key.spatialArea == 225 && key.boardX == 0 && key.boardY == 0 &&
        key.inChannels == 256 && key.outChannels == 256 &&
        key.auxiliaryChannels == 768 && key.maskMode == MaskMode::None &&
        key.semanticScalar0Bits == RMS_EPSILON_1E6_BITS &&
        (key.flags & OP_FLAG_USE_SWIGLU) != 0;
      const bool preparedForInt8 = record.found &&
        record.operation.support != SupportClass::Unsupported &&
        recipe.rmsNorm == RmsNormTactic::Sm120C256Warp4Vec8;
      result.allTransformerShapesEligible =
        result.allTransformerShapesEligible && eligible;
      result.allTransformerRecordsPrepared =
        result.allTransformerRecordsPrepared && preparedForInt8;
      if(eligible)
        result.ffnCount++;
    }
  }
  result.preparedPlanFingerprintValid =
    fingerprintPreparedPlan(requests,prepared) == plan.fingerprint;
  return result;
}

PreparedPlan preparePlan(
  const ArchitectureDesc& architecture,
  const RuntimeOpContext& runtime,
  const DeviceCapability& device
) {
  RegistrationContext context{device};
  Registry registry;
  registerTactic(registry,ATTENTION_FAMILY,ATTENTION_SQUARE_GENERIC,10,
    "attention:v1;planar=cublas-hgemm-strided-square;rope=generic;out=cublas-beta1",
    matchAttentionSquare,&context);
  registerTactic(registry,ATTENTION_FAMILY,ATTENTION_SQUARE_LEARNED_ROPE,11,
    "attention:v1;planar=cublas-hgemm-strided-square;rope=learned-half2;out=cublas-beta1",
    matchAttentionSquareLearnedRope,&context);
  registerTactic(registry,ATTENTION_FAMILY,ATTENTION_SQUARE_MASK_SAFE,10,
    "attention:v1;mask=dense;planar=cublas-hgemm-strided-square;rope=generic;residual=generic-masked",
    matchAttentionSquareMaskSafe,&context);
  registerTactic(registry,ATTENTION_FAMILY,ATTENTION_SQUARE_LEARNED_ROPE_MASK_SAFE,11,
    "attention:v1;mask=dense;planar=cublas-hgemm-strided-square;rope=learned-half2;residual=generic-masked",
    matchAttentionSquareLearnedRopeMaskSafe,&context);
  registerTactic(registry,ATTENTION_FAMILY,ATTENTION_C256_SM120,20,
    "attention:v1;planar=cublas-hgemm-strided-c256;rms=sm120-warp4vec8;rope=learned-half2;out=cublas-beta1",
    matchAttentionC256,&context);
  registerTactic(registry,ATTENTION_FAMILY,ATTENTION_C256_DYNAMIC_SM120,30,
    "attention:v1;planar=cublas-hgemm-strided-c256;rms=sm120-warp4vec8;rope=learned-half2;out=sm120-m128n128k32s3sw1",
    matchAttentionDynamic,&context);
  registerTactic(registry,ATTENTION_FAMILY,ATTENTION_C256_B36_FA4_SM120,40,
    "attention:v1;planar=cublas-hgemm-strided-c256;rms=sm120-warp4vec8;qkv-rope=sm120-m128n128k32s3;fa4=b36-s225-tm128-tn128-s1-both16;out=sm120-m128n128k32s3sw1",
    matchAttentionExact,&context);
  registerTactic(registry,ATTENTION_FAMILY,ATTENTION_C384_DYNAMIC_SM120,35,
    fingerprintRecipeWithC384RuntimeGate(
      "attention:v3;c384-h12-d32;dynamic-M=B*225;max-M=1048576;s225;planar=cublas-hgemm-strided-square;rms=sm120-warp4vec4x3;rope=learned-half2;out=cublas-beta1",
      productionC384RuntimeGatePolicy()),
    matchAttentionC384Dynamic,&context);
  registerTactic(registry,FFN_FAMILY,FFN_GENERIC,10,
    "ffn:v1;rms=generic-half;down=cublas-beta1",matchFfnGeneric,&context);
  registerTactic(registry,FFN_FAMILY,FFN_C256_F768_SM120,20,
    "ffn:v1;rms=sm120-warp4vec8;down=cublas-beta1",matchFfnC256,&context);
  registerTactic(registry,FFN_FAMILY,FFN_C256_F768_DYNAMIC_SM120,30,
    "ffn:v1;rms=sm120-warp4vec8;dual=sm120-m128n64k32s3sw4;down=sm120-m128n128k32s3sw1",
    matchFfnDynamic,&context);
  registerTactic(registry,FFN_FAMILY,FFN_C384_F1024_DYNAMIC_SM120,35,
    fingerprintRecipeWithC384RuntimeGate(
      "ffn:v3;c384-f1024;dynamic-M=B*225;max-M=1048576;s225;rms=sm120-warp4vec4x3;dual=sm120-c384-f1024-m128n64k32s3sw4;down=cublas-beta1",
      productionC384RuntimeGatePolicy()),
    matchFfnC384Dynamic,&context);

  PreparedPlan plan;
  plan.architecture = architecture.signature;
  plan.runtime = runtime;
  const std::vector<OpRequest> requests = buildOpRequests(architecture,runtime);
  plan.records.reserve(requests.size());
  std::vector<PreparedOp> fingerprintOps;
  fingerprintOps.reserve(requests.size());
  for(const OpRequest& request: requests) {
    const ResolveResult resolved = registry.resolveAtConstruction(request);
    PreparedRecord record;
    record.request = request;
    record.found = resolved.found;
    if(resolved.found)
      record.operation = resolved.prepared;
    plan.records.push_back(record);
    fingerprintOps.push_back(record.operation);
  }
  plan.fingerprint = fingerprintPreparedPlan(requests,fingerprintOps);
  return plan;
}

const char* tacticName(const TacticId& tactic) {
  if(tactic.family == ATTENTION_FAMILY) {
    switch(tactic.variant) {
    case ATTENTION_SQUARE_GENERIC: return "attention-square-generic";
    case ATTENTION_SQUARE_LEARNED_ROPE: return "attention-square-learned-rope";
    case ATTENTION_SQUARE_MASK_SAFE: return "attention-square-mask-safe";
    case ATTENTION_SQUARE_LEARNED_ROPE_MASK_SAFE: return "attention-square-learned-rope-mask-safe";
    case ATTENTION_C256_SM120: return "attention-c256-sm120";
    case ATTENTION_C256_DYNAMIC_SM120: return "attention-c256-dynamic-sm120";
    case ATTENTION_C256_B36_FA4_SM120: return "attention-c256-b36-fa4-sm120";
    case ATTENTION_C384_DYNAMIC_SM120: return "attention-c384-h12-dynamic-sm120";
    default: return "attention-unknown";
    }
  }
  if(tactic.family == FFN_FAMILY) {
    switch(tactic.variant) {
    case FFN_GENERIC: return "ffn-generic";
    case FFN_C256_F768_SM120: return "ffn-c256-f768-sm120";
    case FFN_C256_F768_DYNAMIC_SM120: return "ffn-c256-f768-dynamic-sm120";
    case FFN_C384_F1024_DYNAMIC_SM120: return "ffn-c384-f1024-dynamic-sm120";
    default: return "ffn-unknown";
    }
  }
  return "unprepared";
}

}  // namespace CudaTransformerWinner

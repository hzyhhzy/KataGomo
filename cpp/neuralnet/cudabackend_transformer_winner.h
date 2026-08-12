#ifndef NEURALNET_CUDABACKEND_TRANSFORMER_WINNER_H_
#define NEURALNET_CUDABACKEND_TRANSFORMER_WINNER_H_

#include "../neuralnet/architecturedesc.h"
#include "../neuralnet/cudaopregistry.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace CudaTransformerWinner {

// Runtime device facts used only while the immutable plan is prepared. Model
// names, model bytes, and artifact hashes deliberately do not appear here.
struct DeviceCapability {
  uint32_t computeCapability = 0;
  uint32_t warpSize = 0;
  std::size_t sharedBytesPerBlockOptin = 0;
  int cudaRuntimeVersion = 0;
  int cudaDriverVersion = 0;
  int cublasVersion = 0;
  std::size_t cudnnVersion = 0;
  bool specializedSm120KernelsAvailable = false;
};

enum class PlanarQkvTactic : uint32_t {
  Disabled = 0,
  CublasHgemmStridedBatchedSquare = 1,
};

enum class RmsNormTactic : uint32_t {
  GenericHalf = 0,
  Sm120C256Warp4Vec8 = 1,
  Sm120C384Warp4Vec4x3 = 2,
};

enum class RopeTactic : uint32_t {
  Generic = 0,
  LearnedHalf2 = 1,
};

enum class QkvRopeTactic : uint32_t {
  Disabled = 0,
  Sm120C256H8D32M128N128K32S3 = 1,
};

enum class AttentionTactic : uint32_t {
  Generic = 0,
  Fa4Sm120B36S225Tm128Tn128S1Both16 = 1,
};

enum class DualFfnTactic : uint32_t {
  Disabled = 0,
  Sm120C256F768M128N64K32S3Sw4 = 1,
  Sm120C384F1024M128N64K32S3Sw4 = 2,
};

enum class ResidualTactic : uint32_t {
  GenericAdd = 0,
  CublasHgemmBetaOne = 1,
  Sm120M128N128K32S3Sw1 = 2,
  Sm120C384M128N128K32S3Sw1 = 3,
};

// Per-launch C384 dispatch is deliberately separate from construction-time
// recipe selection. Eligible pieces prepare their specialized state and keep
// a generic path; pieces with a zero range may select generic directly and
// avoid unused state. This pure policy decides by the actual launch row count.
enum class C384RuntimePiece : uint32_t {
  RmsNorm = 0,
  DualFfn = 1,
  OutProjection = 2,
  DownProjection = 3,
};

struct C384RuntimeBatchRange {
  uint32_t minInclusive = 0;
  uint32_t maxInclusive = 0;
};

struct C384RuntimePiecePolicy {
  // Used for one evaluator lane and conservatively for unmeasured topologies
  // such as three or more same-GPU lanes.
  C384RuntimeBatchRange conservative;
  // Used only when exactly two evaluator lanes share this physical GPU.
  C384RuntimeBatchRange exactlyTwoSameGpuLanes;
};

struct C384RuntimeGatePolicy {
  // The measured geometry used to turn actualRows into actual batch. Zero
  // disables the entire policy. Nonintegral batches fail closed to generic.
  uint32_t rowsPerBatch = 0;
  C384RuntimePiecePolicy rmsNorm;
  C384RuntimePiecePolicy dualFfn;
  C384RuntimePiecePolicy outProjection;
  C384RuntimePiecePolicy downProjection;
};

// Evidence-locked production policy for C384/H12/D32/F1024 on 15x15 SM120.
// The returned object is immutable and process-lifetime stable.
const C384RuntimeGatePolicy& productionC384RuntimeGatePolicy();

// Hash a stable tactic encoding together with every numeric gate field in a
// fixed little-endian order. This prevents a plan fingerprint from claiming a
// stale runtime policy when only a threshold or disabled range changed.
CudaOpRegistry::RecipeFingerprint fingerprintRecipeWithC384RuntimeGate(
  const std::string& stableTacticEncoding,
  const C384RuntimeGatePolicy& policy
);

bool shouldUseC384RuntimePiece(
  C384RuntimePiece piece,
  int actualRows,
  int sameGpuEvaluatorConcurrency,
  const C384RuntimeGatePolicy& policy
);

// Exact once-per-handle telemetry for pieces whose measured production recipe
// deliberately selects generic, rather than falling back at launch time.
const char* c384MeasuredGenericActiveMarker(C384RuntimePiece piece);

struct AttentionRecipe {
  PlanarQkvTactic planarQkv = PlanarQkvTactic::Disabled;
  RmsNormTactic rmsNorm = RmsNormTactic::GenericHalf;
  RopeTactic rope = RopeTactic::Generic;
  QkvRopeTactic qkvRope = QkvRopeTactic::Disabled;
  AttentionTactic attention = AttentionTactic::Generic;
  ResidualTactic outProjection = ResidualTactic::GenericAdd;

  bool hasPreparedOptimization() const;
};

struct FfnRecipe {
  RmsNormTactic rmsNorm = RmsNormTactic::GenericHalf;
  DualFfnTactic dualFfn = DualFfnTactic::Disabled;
  ResidualTactic downProjection = ResidualTactic::GenericAdd;

  bool hasPreparedOptimization() const;
};

// A one-to-one request/decision record. found=false is the explicit generic
// fallback for that operator. Recipes are queried by topology while testing,
// or by local CapabilityKey while block objects are constructed.
struct PreparedRecord {
  CudaOpRegistry::OpRequest request{};
  bool found = false;
  CudaOpRegistry::PreparedOp operation{};
};

struct PreparedPlan {
  NeuralNetArchitecture::ArchitectureSignature architecture{};
  CudaOpRegistry::RuntimeOpContext runtime{};
  std::vector<PreparedRecord> records;
  CudaOpRegistry::PlanFingerprint fingerprint{};

  AttentionRecipe attentionFor(uint32_t topologyIndex) const;
  FfnRecipe ffnFor(uint32_t topologyIndex) const;
  AttentionRecipe attentionFor(const CudaOpRegistry::CapabilityKey& key) const;
  FfnRecipe ffnFor(const CudaOpRegistry::CapabilityKey& key) const;
};

// Construction-time shape contract for the experimental C256 INT8 path.
// Attention declares independent X/Y runtime dependence (learned RoPE), while
// FFN deliberately declares area-only dependence and therefore has boardX/Y=0
// in its CapabilityKey.
struct Int8ExperimentEligibility {
  bool architectureSignatureMatches = false;
  bool explicitV104ArchitectureSignatureMatches = false;
  bool legacyV102ArchitectureSignatureMatches = false;
  bool preparedPlanFingerprintValid = false;
  bool runtimeContractEligible = false;
  bool allTransformerShapesEligible = false;
  bool allTransformerRecordsPrepared = false;
  int attentionCount = 0;
  int ffnCount = 0;

  bool exactCurrent24LayerModel() const {
    return architectureSignatureMatches && preparedPlanFingerprintValid &&
      runtimeContractEligible && allTransformerShapesEligible &&
      allTransformerRecordsPrepared && attentionCount == 24 && ffnCount == 24;
  }
};

// Weight-free whole-model identities qualified by checked-in canonical CPU
// fixtures. V104 is the production embedded-quantization format. V102 remains
// a clearly identified compatibility path that quantizes FP32 masters at load.
const NeuralNetArchitecture::ArchitectureSignature&
int8QualifiedArchitectureSignature();
const NeuralNetArchitecture::ArchitectureSignature&
int8LegacyImplicitArchitectureSignature();

Int8ExperimentEligibility evaluateInt8ExperimentEligibility(const PreparedPlan& plan);

// Stable construction-time ABI identity. Zero means that at least one
// required runtime version was unavailable. Exact certification separately
// locks every measured version; this hash prevents a caller from substituting
// an arbitrary nonzero value for that measured tuple.
uint64_t makeRuntimeLibraryFingerprint(
  int cudaRuntimeVersion,
  int cudaDriverVersion,
  int cublasVersion,
  std::size_t cudnnVersion
);

PreparedPlan preparePlan(
  const NeuralNetArchitecture::ArchitectureDesc& architecture,
  const CudaOpRegistry::RuntimeOpContext& runtime,
  const DeviceCapability& device
);

const char* tacticName(const CudaOpRegistry::TacticId& tactic);

}  // namespace CudaTransformerWinner

#endif  // NEURALNET_CUDABACKEND_TRANSFORMER_WINNER_H_

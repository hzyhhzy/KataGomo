#ifndef NEURALNET_CUDABACKEND_TRANSFORMER_WINNER_H_
#define NEURALNET_CUDABACKEND_TRANSFORMER_WINNER_H_

#include "../neuralnet/architecturedesc.h"
#include "../neuralnet/cudaopregistry.h"

#include <cstddef>
#include <cstdint>
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
};

enum class ResidualTactic : uint32_t {
  GenericAdd = 0,
  CublasHgemmBetaOne = 1,
  Sm120M128N128K32S3Sw1 = 2,
};

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

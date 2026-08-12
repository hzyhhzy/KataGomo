#ifndef NEURALNET_CUDAOPREGISTRY_H_
#define NEURALNET_CUDAOPREGISTRY_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include "../neuralnet/architecturedesc.h"

struct ModelDesc;

namespace CudaOpRegistry {

constexpr uint32_t CAPABILITY_KEY_SCHEMA_VERSION = 1;
constexpr uint32_t RECIPE_FINGERPRINT_SCHEMA_VERSION = 1;
constexpr uint32_t PLAN_FINGERPRINT_SCHEMA_VERSION = 1;

enum class NumericType : uint32_t {
  Unspecified = 0,
  Float32 = 1,
  Float16 = 2,
  BFloat16 = 3,
  Int8 = 4,
};

enum class TensorLayout : uint32_t {
  Unspecified = 0,
  NCHW = 1,
  NHWC = 2,
  BSH = 3,
};

enum class MaskMode : uint32_t {
  None = 0,
  Dense = 1,
};

enum class SupportClass : uint32_t {
  Unsupported = 0,
  CompatibleOnly = 1,
  CertifiedFast = 2,
};

struct RuntimeOpContext {
  int32_t batchSize;
  int32_t boardX;
  int32_t boardY;
  MaskMode maskMode;
  NumericType inputType;
  NumericType outputType;
  NumericType computeType;
  TensorLayout layout;
  uint32_t deviceComputeCapability;
  uint32_t streamCount;
  // Hash/version chosen by the backend for CUDA, cuBLAS, and cuDNN ABI
  // assumptions. Zero means that a candidate may not claim exact ABI proof.
  uint64_t runtimeLibraryFingerprint;
};

// Pure per-op compatibility identity. It deliberately excludes topology index,
// model weights, model name, artifact SHA, and whole-model architecture digest.
// Unused fields are canonicalized to zero by makeCapabilityKey().
struct CapabilityKey {
  uint32_t schemaVersion;
  NeuralNetArchitecture::ArchitectureOpKind kind;
  uint32_t flags;
  NumericType inputType;
  NumericType outputType;
  NumericType computeType;
  TensorLayout layout;
  MaskMode maskMode;

  int32_t batchSize;
  int32_t spatialArea;
  int32_t boardX;
  int32_t boardY;
  int32_t inChannels;
  int32_t outChannels;
  int32_t auxiliaryChannels;
  int32_t kernelX;
  int32_t kernelY;
  int32_t dilationX;
  int32_t dilationY;
  int32_t numHeads;
  int32_t numKVHeads;
  int32_t qHeadDim;
  int32_t vHeadDim;

  uint32_t semanticScalar0Bits;
  uint32_t semanticScalar1Bits;
  uint32_t deviceComputeCapability;
  uint32_t streamCount;
  uint64_t runtimeLibraryFingerprint;

  bool operator==(const CapabilityKey& other) const;
  bool operator!=(const CapabilityKey& other) const;
};

struct TacticId {
  uint64_t family;
  uint64_t variant;

  bool operator==(const TacticId& other) const;
  bool operator!=(const TacticId& other) const;
  bool operator<(const TacticId& other) const;
};

struct RecipeFingerprint {
  uint32_t schemaVersion;
  std::array<uint8_t,32> digest;

  bool operator==(const RecipeFingerprint& other) const;
  bool operator!=(const RecipeFingerprint& other) const;
  std::string toHex() const;
};

struct PlanFingerprint {
  uint32_t schemaVersion;
  std::array<uint8_t,32> digest;

  bool operator==(const PlanFingerprint& other) const;
  bool operator!=(const PlanFingerprint& other) const;
  std::string toHex() const;
};

struct OpRequest {
  CapabilityKey key;
  NeuralNetArchitecture::ArchitectureSignature architecture;
  uint32_t topologyIndex;
  uint32_t reserved;
};

// PreparedOp is an immutable construction-time decision record. The opaque
// cookie may point to backend-owned prepared state, but it is never included in
// a recipe or plan fingerprint.
struct PreparedOp {
  TacticId tactic;
  RecipeFingerprint recipe;
  SupportClass support;
  uint32_t workspaceAlignment;
  uint64_t workspaceBytes;
  uintptr_t implementationCookie;
};

struct ResolveResult {
  bool found;
  PreparedOp prepared;
};

using MatchTacticFn = SupportClass (*)(const OpRequest& request, const void* userData);
using PrepareTacticFn = bool (*)(const OpRequest& request, PreparedOp& prepared, void* userData);

struct TacticRegistration {
  TacticId id;
  RecipeFingerprint recipe;
  int32_t priority;
  MatchTacticFn match;
  PrepareTacticFn prepare;
  void* userData;
};

class Registry {
 public:
  void registerTactic(const TacticRegistration& registration);

  // Resolves and prepares exactly once while the model/backend is constructed.
  // Inference hot paths consume PreparedOp and do not query the registry.
  ResolveResult resolveAtConstruction(const OpRequest& request) const;
  size_t size() const;

 private:
  std::vector<TacticRegistration> tactics;
};

CapabilityKey makeCapabilityKey(
  const NeuralNetArchitecture::ArchitectureOpDesc& op,
  const RuntimeOpContext& runtime
);
std::vector<OpRequest> buildOpRequests(
  const NeuralNetArchitecture::ArchitectureDesc& architecture,
  const RuntimeOpContext& runtime
);
std::vector<OpRequest> buildOpRequests(const ModelDesc& model, const RuntimeOpContext& runtime);

RecipeFingerprint fingerprintRecipe(const uint8_t* bytes, size_t len);
RecipeFingerprint fingerprintRecipe(const std::string& stableRecipeEncoding);
PlanFingerprint fingerprintPreparedPlan(
  const std::vector<OpRequest>& requests,
  const std::vector<PreparedOp>& prepared
);

static_assert(std::is_trivially_copyable<CapabilityKey>::value,"CapabilityKey must remain POD-like");
static_assert(std::is_standard_layout<CapabilityKey>::value,"CapabilityKey must remain POD-like");
static_assert(std::is_trivially_copyable<TacticId>::value,"TacticId must remain POD-like");
static_assert(std::is_standard_layout<TacticId>::value,"TacticId must remain POD-like");
static_assert(std::is_trivially_copyable<OpRequest>::value,"OpRequest must remain POD-like");
static_assert(std::is_standard_layout<OpRequest>::value,"OpRequest must remain POD-like");
static_assert(std::is_trivially_copyable<PreparedOp>::value,"PreparedOp must remain POD-like");
static_assert(std::is_standard_layout<PreparedOp>::value,"PreparedOp must remain POD-like");

}  // namespace CudaOpRegistry

#endif  // NEURALNET_CUDAOPREGISTRY_H_

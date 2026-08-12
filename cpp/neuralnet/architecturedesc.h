#ifndef NEURALNET_ARCHITECTUREDESC_H_
#define NEURALNET_ARCHITECTUREDESC_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct ModelDesc;
struct TransformerAttentionDesc;
struct TransformerFFNDesc;

namespace NeuralNetArchitecture {

// Bump this whenever the canonical byte encoding changes. Signatures with
// different schema versions must never be compared as if they were equal.
constexpr uint32_t CANONICAL_ARCHITECTURE_SCHEMA_VERSION = 1;

struct ArchitectureSignature {
  uint32_t schemaVersion;
  std::array<uint8_t,32> digest;

  bool operator==(const ArchitectureSignature& other) const;
  bool operator!=(const ArchitectureSignature& other) const;
  std::string toHex() const;
};

// Artifact identity is deliberately not part of ArchitectureSignature. This
// lets independently trained models with the same descriptor geometry share
// prepared tactics, while callers may still retain provenance for diagnostics.
struct ModelProvenance {
  std::string modelName;
  std::string artifactSha256;
  std::string modelConfigSha256;
  bool sourceWasOnnx;
};

enum class ArchitectureOpKind : uint32_t {
  Conv2D = 1,
  BatchNormActivation = 2,
  MatMul = 3,
  MatBias = 4,
  TransformerAttention = 5,
  TransformerFFN = 6,
};

enum ArchitectureOpFlag : uint32_t {
  OP_FLAG_NONE = 0,
  OP_FLAG_HAS_SCALE = 1u << 0,
  OP_FLAG_HAS_BIAS = 1u << 1,
  OP_FLAG_USE_ROPE = 1u << 2,
  OP_FLAG_LEARNABLE_ROPE = 1u << 3,
  OP_FLAG_USE_SWIGLU = 1u << 4,
  OP_FLAG_ACTIVATION_SHIFT = 16,
  OP_FLAG_ACTIVATION_MASK = 0xFFu << OP_FLAG_ACTIVATION_SHIFT,
};

enum ArchitectureRuntimeDependency : uint32_t {
  OP_RUNTIME_NONE = 0,
  OP_RUNTIME_BATCH = 1u << 0,
  // The implementation depends only on the number of spatial tokens.
  OP_RUNTIME_SPATIAL_AREA = 1u << 1,
  // The implementation depends on X and Y independently (for example RoPE).
  OP_RUNTIME_SPATIAL_XY = 1u << 2,
  OP_RUNTIME_MASK = 1u << 3,
};

// Backend-neutral, weight-free operator geometry. topologyIndex is for mapping
// a resolution back to the model graph and is intentionally excluded from a
// per-op CapabilityKey, so repeated layers and deeper models can reuse tactics.
struct ArchitectureOpDesc {
  uint32_t topologyIndex;
  ArchitectureOpKind kind;
  uint32_t flags;
  uint32_t runtimeDependencies;

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

  // Exact IEEE-754 bit patterns for semantic scalars such as epsilon and a
  // fixed RoPE theta. Learned tensors are weights and are never included.
  uint32_t semanticScalar0Bits;
  uint32_t semanticScalar1Bits;
};

enum OuterNormBit : uint8_t {
  OUTER_NORM_TRUNK_TIP = 1u << 0,
  OUTER_NORM_POLICY_G1 = 1u << 1,
  OUTER_NORM_POLICY_P1 = 1u << 2,
  OUTER_NORM_VALUE_V1 = 1u << 3,
};

struct ArchitectureDesc {
  ArchitectureSignature signature;
  std::vector<uint8_t> canonicalEncoding;
  std::vector<ArchitectureOpDesc> operators;
  uint8_t outerBatchNormScaleMask;
  uint8_t outerBatchNormBiasMask;
};

// Requires a native ModelDesc with complete typed descriptors. The current
// ONNX header-only ModelDesc does not expose enough graph structure and is
// rejected rather than assigned a misleading signature.
ArchitectureDesc buildArchitectureDesc(const ModelDesc& model);
ArchitectureSignature getArchitectureSignature(const ModelDesc& model);
ModelProvenance getModelProvenance(const ModelDesc& model);

// Weight-free local operator descriptions. Besides keeping architecture
// encoding and backend construction in lockstep, these helpers let a backend
// attach a prepared tactic to one concrete block without consulting a global
// model signature in the inference hot path.
ArchitectureOpDesc describeTransformerAttentionOp(const TransformerAttentionDesc& desc);
ArchitectureOpDesc describeTransformerFFNOp(const TransformerFFNDesc& desc);

uint32_t getFloatBits(float value);

}  // namespace NeuralNetArchitecture

#endif  // NEURALNET_ARCHITECTUREDESC_H_

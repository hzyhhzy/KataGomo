#ifndef KATAGO_C384_EXPERIMENTAL_INT8_KERNELS_H_
#define KATAGO_C384_EXPERIMENTAL_INT8_KERNELS_H_

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace C384Int8Experiment {

constexpr int kBatch = 28;
constexpr int kSequence = 225;
constexpr int kTokenRows = kBatch * kSequence;
constexpr int kChannels = 384;
constexpr int kQkChannels = 2 * kChannels;
constexpr int kQkvChannels = 3 * kChannels;
constexpr int kFfnChannels = 1024;
constexpr float kRmsEpsilon = 1.0e-6f;
constexpr float kNormActivationClip = 4.0f;
constexpr float kNormActivationScale = kNormActivationClip / 127.0f;
constexpr float kClip7ProductClip = 49.0f;
constexpr float kClip7ProductScale = kClip7ProductClip / 127.0f;

enum class ProjectionMode : uint32_t {
  ConservativeQk = 1,
  AggressiveQkv = 2,
};

enum class ProjectionTactic : uint32_t {
  M128N128K64S2Sw1 = 1,
  M128N128K64S3Sw1 = 2,
  M128N128K64S3Sw2 = 3,
};

enum class DualFfnTactic : uint32_t {
  M128N64K64S3Sw1 = 1,
  M128N64K64S3Sw4 = 2,
  M128N64K64S4Sw1 = 3,
};

enum class DownTactic : uint32_t {
  M128N128K64S2Sw1 = 1,
  M128N128K64S3Sw1 = 2,
  M128N128K64S3Sw2 = 3,
};

// Mirrors the production C384 Warp4Vec4x3 FP16 RMSNorm arithmetic and emits
// a second row-major signed-INT8 tensor. Quantization occurs after the FP16
// rounding boundary using clip4, round-to-nearest-even, zero point 0, and a
// saturated range of [-127,127].
cudaError_t launchRmsNormFp16Int8(
  const half* input,
  half* outputFp16,
  int8_t* outputInt8,
  const half* gamma,
  int tokenRows,
  float epsilon,
  cudaStream_t stream
);

struct ProjectionConfig {
  ProjectionMode mode = ProjectionMode::AggressiveQkv;
  ProjectionTactic tactic = ProjectionTactic::M128N128K64S3Sw2;
  int maxTokenRows = 0;
  // Output-major K-contiguous signed-INT8 weights. Conservative mode owns
  // [384,768], aggressive mode owns [384,1152].
  const int8_t* packedWeights = nullptr;
  float weightScale = 0.0f;
};

void* createProjection(const ProjectionConfig& config);
void destroyProjection(void* opaque) noexcept;
bool projectionSupports(
  const void* opaque,
  ProjectionMode mode,
  int tokenRows,
  int inputChannels,
  int outputRowStride
) noexcept;

// Writes raw packed token rows. Conservative QK writes columns [0,768) with
// row stride 1152, leaving V [768,1152) untouched for an FP16 V producer.
// Aggressive QKV writes all [0,1152). The resulting FP16 ABI can feed the
// existing C384 QKNorm+RoPE in-place postprocess and packed FA4 consumer.
cudaError_t launchProjection(
  void* opaque,
  int tokenRows,
  const int8_t* activation,
  half* rawPackedQkv,
  int outputRowStride,
  cudaStream_t stream
);

// Conservative engine bridge: copies planar FP16 V [M,384] into the V slice
// of raw packed [M][Q384|K384|V384]. Q/K are not touched.
cudaError_t launchPackPlanarV(
  const half* planarV,
  half* rawPackedQkv,
  int tokenRows,
  cudaStream_t stream
);

struct DualFfnConfig {
  DualFfnTactic tactic = DualFfnTactic::M128N64K64S3Sw4;
  int maxTokenRows = 0;
  const int8_t* packedUpWeights = nullptr;
  const int8_t* packedGateWeights = nullptr;
  float upWeightScale = 0.0f;
  float gateWeightScale = 0.0f;
};

void* createDualFfn(const DualFfnConfig& config);
void destroyDualFfn(void* opaque) noexcept;
bool dualFfnSupports(const void* opaque, int tokenRows) noexcept;

// Shared-A INT8 up/gate GEMM. Each projection dequantizes to FP16 before the
// final FP32 clip7 epilogue computes clamp(SiLU(up),+-7)*clamp(gate,+-7).
cudaError_t launchDualFfnHalf(
  void* opaque,
  int tokenRows,
  const int8_t* activation,
  half* productFp16,
  cudaStream_t stream
);

// First prototype conversion for the aggressive down path. It is deliberately
// a separate callable boundary so timing can expose its cost. The production
// follow-up may replace this call with an INT8-emitting dual epilogue without
// changing the down-GEMM ABI.
cudaError_t launchQuantizeClip7Product(
  const half* productFp16,
  int8_t* productInt8,
  int tokenRows,
  cudaStream_t stream
);

struct DownConfig {
  DownTactic tactic = DownTactic::M128N128K64S3Sw2;
  int maxTokenRows = 0;
  // Output-major K-contiguous signed-INT8 [1024,384].
  const int8_t* packedWeights = nullptr;
  float weightScale = 0.0f;
};

void* createDown(const DownConfig& config);
void destroyDown(void* opaque) noexcept;
bool downSupports(const void* opaque, int tokenRows) noexcept;

// Computes half(alpha * S8[M,1024] * S8[1024,384] + residual), with
// alpha=(49/127)*weightScale and beta=1 in the CUTLASS epilogue.
cudaError_t launchDownResidual(
  void* opaque,
  int tokenRows,
  const int8_t* productInt8,
  const half* residual,
  half* output,
  cudaStream_t stream
);

const char* projectionTacticName(ProjectionTactic tactic) noexcept;
const char* dualFfnTacticName(DualFfnTactic tactic) noexcept;
const char* downTacticName(DownTactic tactic) noexcept;

}  // namespace C384Int8Experiment

#endif  // KATAGO_C384_EXPERIMENTAL_INT8_KERNELS_H_

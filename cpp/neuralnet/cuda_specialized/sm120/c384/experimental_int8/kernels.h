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
constexpr int kHeads = 12;
constexpr int kHeadDim = 32;
constexpr int kRopePairsPerHead = kHeadDim / 2;
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
  // Experimental single-GEMM form. Up/gate output channels are interleaved
  // in one K384xN2048 B tensor and paired by the canonical epilogue thread
  // map. The production/default tactic remains M128N64K64S3Sw4.
  M128N128K64S3Sw4Interleaved = 4,
};

// The conservative engine keeps the existing FP16 down-projection ABI. The
// aggressive engine instead asks the dual GEMM epilogue to quantize its final
// per-layer clipped product directly into the serialized calibrated INT8
// domain. Keeping this choice on the prepared handle makes an accidental
// half/INT8 pointer mismatch fail closed; clip7/productMax49 remains the exact
// integer fast path.
enum class DualFfnOutputMode : uint32_t {
  Fp16Product = 1,
  Int8Product = 2,
};

// The production/default path retains the incumbent RNE implementation. The
// exact branchless alternative is an independent prepared-handle tactic, so a
// single CUDA binary can exercise both real dual-GEMM epilogues. It is only
// admissible for the exact clip=7/productMax=49 INT8 D2 specialization.
enum class DualFfnDivide127Tactic : uint32_t {
  Incumbent = 1,
  ExactBranchless = 2,
};

// Selects how clip7 with a non-square calibrated product domain is evaluated.
// Auto retains the optimized fixed-factor candidate. FullyAdjustableFloat is
// the reference implementation and may be selected for an engine-level A/B.
enum class DualFfnProductPathTactic : uint32_t {
  Auto = 0,
  FullyAdjustableFloat = 1,
};

enum class DownTactic : uint32_t {
  M128N128K64S2Sw1 = 1,
  M128N128K64S3Sw1 = 2,
  M128N128K64S3Sw2 = 3,
};

// Kept distinct from DownTactic even though the initial candidates share the
// same CUTLASS tile family. Attention out is K=384,N=384 and consumes clip4
// activations; FFN down is K=1024,N=384 and consumes the per-layer calibrated
// product domain (with clip7/productMax49 as its exact fast-path case).
enum class AttentionOutTactic : uint32_t {
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

// Aggressive-only fused producer. The INT8 QKV mainloop is identical to the
// selected projection tactic. Its epilogue reduces each token/head D32 Q/K
// vector in FP32, applies gamma, rounds the normalized tensor to FP16, then
// applies learned RoPE. Only the final packed Q/K values reach global memory;
// V is byte-for-byte identical to launchProjection and the [Q|K|V] ABI is
// unchanged. This exact contract intentionally fails closed outside B28/S225,
// H12/D32, epsilon=1e-6.
bool projectionQknormRopeSupports(
  const void* opaque,
  int tokenRows,
  int inputChannels,
  int outputRowStride,
  float qEpsilon,
  float kEpsilon
) noexcept;

const char* projectionQknormRopeMarker() noexcept;

cudaError_t launchProjectionQknormRope(
  void* opaque,
  int tokenRows,
  const int8_t* activation,
  half* packedQkv,
  int outputRowStride,
  const half* qGamma,
  const half* kGamma,
  const half2* learnedRopeCosSin,
  float qEpsilon,
  float kEpsilon,
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
  DualFfnOutputMode outputMode = DualFfnOutputMode::Fp16Product;
  DualFfnDivide127Tactic divide127Tactic =
    DualFfnDivide127Tactic::Incumbent;
  DualFfnProductPathTactic productPathTactic =
    DualFfnProductPathTactic::Auto;
  int maxTokenRows = 0;
  const int8_t* packedUpWeights = nullptr;
  const int8_t* packedGateWeights = nullptr;
  float upWeightScale = 0.0f;
  float gateWeightScale = 0.0f;
  // Serialized v105 FFN semantics. The aggressive path quantizes each
  // clipped factor with swigluClip/127, then requantizes their product with
  // productQuantMaxAbs/127. Keeping both values on the prepared handle makes
  // per-layer calibration immutable and stream-safe.
  float swigluClip = 7.0f;
  float productQuantMaxAbs = 49.0f;
};

void* createDualFfn(const DualFfnConfig& config);
void destroyDualFfn(void* opaque) noexcept;
bool dualFfnSupports(
  const void* opaque,
  DualFfnOutputMode outputMode,
  int tokenRows
) noexcept;

// Shared-A INT8 up/gate GEMM. Each projection dequantizes to FP16 before the
// final FP32 clip7 epilogue computes clamp(SiLU(up),+-7)*clamp(gate,+-7).
cudaError_t launchDualFfnHalf(
  void* opaque,
  int tokenRows,
  const int8_t* activation,
  half* productFp16,
  cudaStream_t stream
);

// Aggressive shared-A dual epilogue. Up and gate are dequantized in FP32,
// transformed as clamp(SiLU(up),+-clip) and clamp(gate,+-clip), quantized in
// registers to symmetric signed INT8 factors, multiplied, then requantized
// directly to productQuantMaxAbs/127. Zero point is 0, conversion is
// round-to-nearest-even, and saturation is [-127,127] (never -128). No FP16
// product tensor is materialized. The clip=7/productMax=49 contract retains
// a dedicated exact integer fast path.
cudaError_t launchDualFfnInt8(
  void* opaque,
  int tokenRows,
  const int8_t* activation,
  int8_t* productInt8,
  cudaStream_t stream
);

// Reports the immutable product requantization path selected at preparation.
// This is evidence/diagnostics only and is never consulted by dispatch.
const char* dualFfnProductQuantPath(const void* opaque) noexcept;

// True only when the dual producer and down consumer were prepared from the
// exact same serialized per-layer productQuantMaxAbs value. Engine wiring
// checks this before publishing the all-or-nothing INT8 transaction.
bool dualFfnDownProductQuantizationMatches(
  const void* dualOpaque,
  const void* downOpaque
) noexcept;

// Runtime resource evidence for the isolated interleaved candidate. Returns
// false for every incumbent dual-GEMM tactic, so production code cannot use
// this diagnostic as a dispatch decision.
struct InterleavedDualFfnKernelResources {
  int registersPerThread = 0;
  int staticSharedBytes = 0;
  int dynamicSharedBytes = 0;
  int threadsPerBlock = 0;
};

bool interleavedDualFfnKernelResources(
  const void* opaque,
  InterleavedDualFfnKernelResources& resources
) noexcept;

// Aggressive-engine RMSNorm. It preserves the same FP16 rounding boundary,
// clip4, RNE, and [-127,127] quantization contract as
// launchRmsNormFp16Int8, but does not materialize an unused FP16 tensor.
cudaError_t launchRmsNormInt8(
  const half* input,
  int8_t* outputInt8,
  const half* gamma,
  int tokenRows,
  float epsilon,
  cudaStream_t stream
);

// Legacy component-control conversion retained only for microbench comparison.
// The aggressive engine path must use launchDualFfnInt8 and never call this.
cudaError_t launchQuantizeClip7Product(
  const half* productFp16,
  int8_t* productInt8,
  int tokenRows,
  cudaStream_t stream
);

// Quantizes the FP16 packed FA4 output [M,384] using clip4, RNE, zero point
// 0, and saturation [-127,127]. The output may reuse the attention RMS INT8
// scratch only after QKV projection has consumed it.
cudaError_t launchQuantizeAttentionOutput(
  const half* attentionFp16,
  int8_t* attentionInt8,
  int tokenRows,
  cudaStream_t stream
);

struct DownConfig {
  DownTactic tactic = DownTactic::M128N128K64S3Sw2;
  int maxTokenRows = 0;
  // Output-major K-contiguous signed-INT8 [1024,384].
  const int8_t* packedWeights = nullptr;
  float weightScale = 0.0f;
  // Real max represented by signed INT8 input value 127. This must match the
  // paired dual-FFN handle from the same serialized v105 block.
  float productQuantMaxAbs = 49.0f;
};

void* createDown(const DownConfig& config);
void destroyDown(void* opaque) noexcept;
bool downSupports(const void* opaque, int tokenRows) noexcept;

// Computes half(alpha * S8[M,1024] * S8[1024,384] + residual), with
// alpha=(productQuantMaxAbs/127)*weightScale and beta=1 in the CUTLASS
// epilogue.
cudaError_t launchDownResidual(
  void* opaque,
  int tokenRows,
  const int8_t* productInt8,
  const half* residual,
  half* output,
  cudaStream_t stream
);

struct AttentionOutConfig {
  AttentionOutTactic tactic = AttentionOutTactic::M128N128K64S3Sw2;
  int maxTokenRows = 0;
  // Output-major K-contiguous signed-INT8 [384,384].
  const int8_t* packedWeights = nullptr;
  float weightScale = 0.0f;
};

void* createAttentionOut(const AttentionOutConfig& config);
void destroyAttentionOut(void* opaque) noexcept;
bool attentionOutSupports(const void* opaque, int tokenRows) noexcept;

// Computes half(alpha * S8[M,384] * S8[384,384] + residual), with
// alpha=(4/127)*weightScale and beta=1 in the CUTLASS epilogue.
cudaError_t launchAttentionOutResidual(
  void* opaque,
  int tokenRows,
  const int8_t* attentionInt8,
  const half* residual,
  half* output,
  cudaStream_t stream
);

const char* projectionTacticName(ProjectionTactic tactic) noexcept;
const char* dualFfnTacticName(DualFfnTactic tactic) noexcept;
const char* downTacticName(DownTactic tactic) noexcept;
const char* attentionOutTacticName(AttentionOutTactic tactic) noexcept;

}  // namespace C384Int8Experiment

#endif  // KATAGO_C384_EXPERIMENTAL_INT8_KERNELS_H_

#ifndef KATAGO_C384_EXACT_FIXED_AOT_PLAN_H_
#define KATAGO_C384_EXACT_FIXED_AOT_PLAN_H_

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace C384ExactFixedAot {

constexpr int kSequenceLength = 225;
constexpr int kBoardX = 15;
constexpr int kBoardY = 15;
constexpr int kChannels = 384;
constexpr int kNumHeads = 12;
constexpr int kNumKvHeads = 12;
constexpr int kHeadDim = 32;
constexpr int kFfnChannels = 1024;
constexpr int kRopePairsTotal = 192;
constexpr uint32_t kComputeCapability = 120;
// Search tooling may still compare historical B24 assets, but the production
// engine route is deliberately fixed to B28 by user direction.
constexpr int kProductionBatch = 28;
constexpr int kProductionTokenRows = kProductionBatch * kSequenceLength;
constexpr uint32_t kRegistryAbiVersion = 2;
constexpr uint32_t kPackedFa4ProofAbiVersion = 1;
constexpr uint32_t kQkvRopeNativeAbiVersion = 1;
constexpr uint32_t kDualFfnNativeAbiVersion = 1;
constexpr uint32_t kSwiGluClip7Bits = 0x40E00000u;

// Search priority, not a range. Every generated kernel remains exact-M.
// Model depth is intentionally absent: it is not runtime batch, and a
// per-operator winner may be reused by another depth with the same shapes.
const int* candidateBatches(std::size_t& count);
int candidateBatchPriority(int batchSize);
int tokenRowsForBatch(int batchSize);
bool productionBatchEligible(int batchSize, int tokenRows);

enum class Family : uint32_t {
  QkvRope = 1,
  DualFfn = 2,
};

struct RuntimeShape {
  // Whole-model transaction metadata. Per-operator matching intentionally
  // remains depth-independent so compatible b24 and b36 models reuse the
  // same exact-bs28 kernel objects.
  int modelDepth = 0;
  int attentionBlockCount = 0;
  int ffnBlockCount = 0;
  bool alternatingAttentionFfn = false;
  int batchSize = 0;
  int enqueuedRows = 0;
  int boardX = 0;
  int boardY = 0;
  int sequenceLength = 0;
  int channels = 0;
  int numHeads = 0;
  int numKvHeads = 0;
  int qHeadDim = 0;
  int vHeadDim = 0;
  int ffnChannels = 0;
  int ropePairsTotal = 0;
  int deviceOrdinal = -1;
  uint32_t computeCapability = 0;
  bool usingFp16 = false;
  bool usingNhwc = false;
  bool exactNoMask = false;
  bool learnedRope = false;
  bool swiglu = false;
  // QKN is completed by the engine's typed post-projection operation. QKV
  // tactics remain reusable only when they explicitly honor the supplied
  // runtime RoPE table (an identity table produces raw packed QKV).
  bool qkNorm = false;
  // Exact IEEE-754 bits. Zero means unclipped SwiGLU.
  uint32_t swigluClipBits = 0;
};

// This is deliberately weight-free metadata. The first member of every CUDA
// tactic descriptor is a TacticKey, allowing the CPU selector to inspect a
// generated registry without depending on CUDA headers or launch functions.
struct TacticKey {
  Family family = Family::QkvRope;
  int batchSize = 0;
  int tokenRows = 0;
  // Used by the dual-FFN search (170 or 340). QKV+RoPE records use zero.
  int launchGridSms = 0;
  const char* id = nullptr;
  bool packedQkvOutput = false;
  bool pairedFfnWeights = false;
  // Zero is invalid. Generated providers must spell out the ABI version so
  // recompiling an old initializer cannot silently claim a newer ABI.
  uint32_t abiVersion = 0;
  // QKV ABI v1 is table-driven, including an all-identity table. Keep the
  // default true so already promoted v1 registry initializers remain source
  // compatible when rebuilt with this extended key. Ignored for dual FFN.
  bool runtimeRopeTableDriven = true;
  // Zero is standard SwiGLU. A dual-FFN tactic for a clipped model must state
  // the exact clip bits and therefore cannot be selected for the old model.
  uint32_t swigluClipBits = 0;
};

static_assert(std::is_standard_layout<TacticKey>::value,
  "TacticKey registry ABI must remain standard-layout");
static_assert(std::is_trivially_copyable<TacticKey>::value,
  "TacticKey registry ABI must remain trivially copyable");

struct RegistrySpan {
  const void* entries = nullptr;
  std::size_t count = 0;
  std::size_t stride = 0;
};

struct RegistryView {
  RegistrySpan qkvRope;
  RegistrySpan dualFfn;
};

// Immutable proof emitted by the separate FA4 preparation layer. The cookie
// is an opaque pointer-sized identity for a fully prepared packed-input FA4
// implementation; it is never fingerprinted and is never dereferenced by the
// portable selector.
struct PreparedPackedFa4 {
  uint32_t abiVersion = 0;
  int batchSize = 0;
  int sequenceLength = 0;
  int numHeads = 0;
  int numKvHeads = 0;
  int qHeadDim = 0;
  int vHeadDim = 0;
  int deviceOrdinal = -1;
  bool acceptsPackedTokenQkv = false;
  const char* id = nullptr;
  uintptr_t implementationCookie = 0;
};

enum class RejectReason : uint32_t {
  None = 0,
  ShapeMismatch,
  NoRequestedTactic,
  InvalidRegistry,
  RegistryMiss,
  MissingSameBatchFa4,
  InvalidImplementation,
  PreparationFailed,
};

struct PieceSelection {
  const TacticKey* tactic = nullptr;
  RejectReason reason = RejectReason::ShapeMismatch;

  bool selected() const { return tactic != nullptr && reason == RejectReason::None; }
};

struct Selection {
  bool targetShape = false;
  PieceSelection qkvRope;
  const PreparedPackedFa4* packedFa4 = nullptr;
  PieceSelection dualFfn;
};

// Construction and first-use accounting for the all-or-nothing exact route.
// Attention and FFN blocks have different prepared objects, so both expected
// counts are retained even though an eligible architecture requires them to
// be equal. The same predicate is used for prepared and active progress.
struct TransactionProgress {
  int modelDepth = 0;
  int attentionBlockCount = 0;
  int ffnBlockCount = 0;
  int qkvFa4Count = 0;
  int dualFfnCount = 0;
  int ffnDownCount = 0;
};

bool modelStructureEligible(const RuntimeShape& shape);
bool attentionShapeEligible(const RuntimeShape& shape);
bool ffnShapeEligible(const RuntimeShape& shape);
bool targetShapeEligible(const RuntimeShape& shape);
bool transactionProgressComplete(const TransactionProgress& progress);
bool pieceShapeEligible(const RuntimeShape& shape, Family family);
bool preparedPackedFa4Compatible(
  const RuntimeShape& shape,
  const PreparedPackedFa4* preparedPackedFa4
);
bool tacticKeyWellFormed(const TacticKey& key);
bool tacticKeyCompatible(const RuntimeShape& shape, const TacticKey& key);
bool registrySpanWellFormed(Family family, const RegistrySpan& registry);

const TacticKey* findExactTactic(
  const RuntimeShape& shape,
  Family family,
  const RegistrySpan& registry,
  const char* requestedId
);

// Packed QKV and typed FA4 proof are returned atomically. QKV+RoPE is rejected
// before enqueue unless the same-batch packed-input FA4 dependency exists.
// Dual FFN is independent and may still be selected.
Selection select(
  const RuntimeShape& shape,
  const RegistryView& registry,
  const char* requestedQkvRopeId,
  const char* requestedDualFfnId,
  const PreparedPackedFa4* preparedPackedFa4
);

const char* rejectReasonName(RejectReason reason);

}  // namespace C384ExactFixedAot

#endif  // KATAGO_C384_EXACT_FIXED_AOT_PLAN_H_

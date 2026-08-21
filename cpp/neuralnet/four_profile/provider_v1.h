#ifndef NEURALNET_FOUR_PROFILE_PROVIDER_V1_H_
#define NEURALNET_FOUR_PROFILE_PROVIDER_V1_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace FourProfile {

enum class BlockKindV1 : uint32_t {
  Other = 0,
  TransformerAttention = 1,
  TransformerFfn = 2,
};

enum class StorageTypeV1 : uint32_t {
  Fp32 = 0,
  Fp16 = 1,
};

// P4 still consumes and produces FP16 storage. RequestedExecutionV1 describes
// the provider's internal execution contract and is intentionally independent
// from input/output storage.
enum class RequestedExecutionV1 : uint32_t {
  Fp16 = 0,
  Int8 = 1,
};

enum class TensorLayoutV1 : uint32_t {
  Nchw = 0,
  Nhwc = 1,
};

enum class MaskModeV1 : uint32_t {
  None = 0,
  Dense = 1,
};

enum class ClipClassV1 : uint32_t {
  Zero = 0,
  PositiveFinite = 1,
  Invalid = 2,
};

enum class AvailabilityV1 : uint32_t {
  Available = 0,
  Unavailable = 1,
  Uncertified = 2,
};

enum class ModeV1 : uint32_t {
  Off = 0,
  Auto = 1,
  Required = 2,
};

enum class RouteV1 : uint32_t {
  Official = 0,
  Specialized = 1,
};

enum class ReasonV1 : uint32_t {
  Disabled = 0,
  Selected = 1,
  NoTransformer = 2,
  Unmatched = 3,
  InvalidTransformerSpan = 4,
  NonUniformTransformerSpan = 5,
  ProviderUnavailable = 6,
  ProviderUncertified = 7,
  ProviderCreationFailed = 8,
  PrepareFailed = 9,
  CommitFailed = 10,
  RuntimePreflightFailed = 11,
  EnqueueFailedBeforeWork = 12,
};

struct AttentionSpecV1 {
  int channels = 0;
  int numHeads = 0;
  int numKVHeads = 0;
  int qHeadDim = 0;
  int vHeadDim = 0;
  bool useRope = false;
  bool learnableRope = false;
  bool useQKNorm = false;
  bool hasInputQuantRange = false;
  bool hasOutputQuantRange = false;

  bool operator==(const AttentionSpecV1& other) const;
  bool operator!=(const AttentionSpecV1& other) const { return !(*this == other); }
};

struct FfnSpecV1 {
  int channels = 0;
  int hiddenChannels = 0;
  bool useSwiGLU = false;
  ClipClassV1 clipClass = ClipClassV1::Zero;
  bool hasInputQuantRange = false;
  bool hasProductQuantRange = false;

  bool operator==(const FfnSpecV1& other) const;
  bool operator!=(const FfnSpecV1& other) const { return !(*this == other); }
};

struct RuntimeKeyV1 {
  int deviceComputeCapability = 0;
  int boardX = 0;
  int boardY = 0;
  int physicalBatchSize = 0;
  int sameGpuConcurrency = 0;
  bool exactBoard = false;
  MaskModeV1 maskMode = MaskModeV1::Dense;
  bool maskNull = false;
  StorageTypeV1 inputStorage = StorageTypeV1::Fp32;
  StorageTypeV1 outputStorage = StorageTypeV1::Fp32;
  RequestedExecutionV1 requestedExecution = RequestedExecutionV1::Fp16;
  TensorLayoutV1 layout = TensorLayoutV1::Nchw;

  bool operator==(const RuntimeKeyV1& other) const;
  bool operator!=(const RuntimeKeyV1& other) const { return !(*this == other); }
};

// pairCount is deliberately absent. A profile key describes one homogeneous
// attention/FFN pair plus its runtime contract, while TransformerSpanV1 carries
// the actual dynamic N/N depth.
struct ProfileKeyV1 {
  int modelVersion = -1;
  RuntimeKeyV1 runtime;
  AttentionSpecV1 attention;
  FfnSpecV1 ffn;

  bool operator==(const ProfileKeyV1& other) const;
  bool operator!=(const ProfileKeyV1& other) const { return !(*this == other); }
};

struct OfficialAttentionResourcesV1 {
  const void* descriptor = nullptr;
  const void* executable = nullptr;

  bool complete() const { return descriptor != nullptr && executable != nullptr; }
};

struct OfficialFfnResourcesV1 {
  const void* descriptor = nullptr;
  const void* executable = nullptr;

  bool complete() const { return descriptor != nullptr && executable != nullptr; }
};

struct AttentionLayerScalarsV1 {
  uint32_t inputQuantMaxAbsBits = 0;
  uint32_t outputQuantMaxAbsBits = 0;
};

struct FfnLayerScalarsV1 {
  uint32_t swigluClipBits = 0;
  uint32_t inputQuantMaxAbsBits = 0;
  uint32_t productQuantMaxAbsBits = 0;
};

struct BlockViewV1 {
  BlockKindV1 kind = BlockKindV1::Other;
  AttentionSpecV1 attention;
  FfnSpecV1 ffn;
  AttentionLayerScalarsV1 attentionScalars;
  FfnLayerScalarsV1 ffnScalars;
  OfficialAttentionResourcesV1 officialAttention;
  OfficialFfnResourcesV1 officialFfn;
};

struct ModelViewV1 {
  int modelVersion = -1;
  std::vector<BlockViewV1> blocks;
};

struct TransformerSpanV1 {
  bool hasTransformer = false;
  bool valid = false;
  size_t beginBlock = 0;
  size_t endBlock = 0;
  size_t pairCount = 0;
  std::string detail;
};

TransformerSpanV1 analyzeTransformerSpanV1(const std::vector<BlockViewV1>& blocks);

// Returns false only when layers within an otherwise-valid (A,F)^N span do
// not have one homogeneous profile key. The actual N remains in span.
bool deriveProfileKeyV1(
  const ModelViewV1& model,
  const TransformerSpanV1& span,
  const RuntimeKeyV1& runtime,
  ProfileKeyV1& key,
  std::string& detail
);

struct AvailabilityResultV1 {
  AvailabilityV1 availability = AvailabilityV1::Unavailable;
  std::string detail;
};

class PreparedAttentionV1 {
public:
  virtual ~PreparedAttentionV1() = default;
};

class PreparedFfnV1 {
public:
  virtual ~PreparedFfnV1() = default;
};

struct PrepareAttentionResultV1 {
  std::unique_ptr<PreparedAttentionV1> prepared;
  std::string detail;
};

struct PrepareFfnResultV1 {
  std::unique_ptr<PreparedFfnV1> prepared;
  std::string detail;
};

struct PreparedSpanV1 {
  std::vector<std::unique_ptr<PreparedAttentionV1>> attention;
  std::vector<std::unique_ptr<PreparedFfnV1>> ffn;
};

struct RuntimeCallV1 {
  RuntimeKeyV1 key;
  int actualBatchSize = 0;
  int sequenceSize = 0;
  size_t transformerBeginBlock = 0;
  size_t transformerPairCount = 0;
  void* stream = nullptr;
  void* trunk = nullptr;
  void* trunkScratch = nullptr;
  void* mask = nullptr;
  void* workspace = nullptr;
  size_t workspaceBytes = 0;

  // The successful zero-enqueue preflight and its one subsequent enqueue must
  // refer to exactly the same buffers, stream, sizes, and runtime key.
  bool sameIdentity(const RuntimeCallV1& other) const;
};

struct ProviderOpResultV1 {
  bool ok = false;
  size_t enqueued = 0;
  uint64_t planGeneration = 0;
  uint64_t runToken = 0;
  std::string detail;

  static ProviderOpResultV1 success(
    size_t enqueuedOperations,
    uint64_t planGeneration,
    uint64_t runToken
  );
  static ProviderOpResultV1 failure(const std::string& detail, size_t enqueuedOperations = 0);
};

class ProviderV1 {
public:
  virtual ~ProviderV1() = default;

  // A real per-ComputeHandle provider owns any cuBLAS/cuDNN/vendor handles it
  // needs and binds them to RuntimeCallV1::stream for each run. The opaque
  // official executable pointers are immutable lifetime anchors only: a
  // provider must not borrow or mutate the backend's private CudaHandles.
  // This isolation is intentional; typed provider adapters may be added later
  // without exposing private official backend state through V1.

  virtual const char* profileId() const noexcept = 0;

  // Preparation may upload resources, but every return or exception must
  // leave all staged asynchronous activity complete enough that destroying
  // the returned resource/provider is safe. This is what makes Auto fallback
  // after a failed Nth prepare deterministic and leak-free.
  virtual PrepareAttentionResultV1 prepareAttention(
    size_t layer,
    const AttentionSpecV1& spec,
    const AttentionLayerScalarsV1& scalars,
    const OfficialAttentionResourcesV1& official
  ) = 0;

  virtual PrepareFfnResultV1 prepareFfn(
    size_t layer,
    const FfnSpecV1& spec,
    const FfnLayerScalarsV1& scalars,
    const OfficialFfnResourcesV1& official
  ) = 0;

  // commit is invoked exactly once, and only after all N attention and all N
  // FFN preparations have succeeded. It may finalize or upload plan resources
  // but must not enqueue inference work. A false result or exception must
  // leave the provider and all moved staged resources safely destructible and
  // uncommitted so the manager can retain the complete official route.
  virtual bool commit(PreparedSpanV1&& prepared, std::string& detail) = 0;
  virtual uint64_t committedPlanGeneration() const noexcept = 0;

  // preflight must enqueue nothing. The enqueued field lets the manager turn
  // a provider contract violation into a fatal error instead of double-running
  // the official implementation.
  virtual ProviderOpResultV1 preflight(const RuntimeCallV1& call) = 0;
  virtual ProviderOpResultV1 enqueue(const RuntimeCallV1& call, uint64_t runToken) = 0;
};

class FactoryV1 {
public:
  virtual ~FactoryV1() = default;

  virtual const char* factoryId() const noexcept = 0;
  virtual bool matches(const ProfileKeyV1& key) const = 0;
  virtual AvailabilityResultV1 availability(const ProfileKeyV1& key) const = 0;
  virtual std::unique_ptr<ProviderV1> create(const ProfileKeyV1& key) const = 0;
};

class ErrorV1 : public std::runtime_error {
public:
  explicit ErrorV1(const std::string& message) : std::runtime_error(message) {}
};

// FatalErrorV1 marks cases where retrying the official route could duplicate
// already-enqueued work, or where registry ambiguity makes execution unsafe.
class FatalErrorV1 : public std::runtime_error {
public:
  explicit FatalErrorV1(const std::string& message) : std::runtime_error(message) {}
};

const char* availabilityNameV1(AvailabilityV1 availability);
const char* reasonNameV1(ReasonV1 reason);

}  // namespace FourProfile

#endif  // NEURALNET_FOUR_PROFILE_PROVIDER_V1_H_

#ifndef KATAGO_C384_EXACT_FFN_DOWN_AOT_H_
#define KATAGO_C384_EXACT_FFN_DOWN_AOT_H_

#include "../neuralnet/c384_residual_aot_abi.h"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace C384ExactFfnDownAot {

constexpr std::uint32_t kRegistryAbiVersion = 1;
constexpr std::uint32_t kNativeAbiVersion = 1;
constexpr int kSelectedBatch = 28;
constexpr int kSequenceLength = 225;
constexpr int kSelectedTokenRows = kSelectedBatch * kSequenceLength;
constexpr int kBoardX = 15;
constexpr int kBoardY = 15;
constexpr int kChannels = 384;
constexpr int kInputChannels = 1024;
constexpr int kOutputChannels = 384;
constexpr std::uint32_t kComputeCapability = 120;
constexpr int kTileM = 128;
constexpr int kTileN = 128;
constexpr int kTileK = 32;
constexpr int kStages = 3;
constexpr int kAtomM = 2;
constexpr int kAtomN = 2;
constexpr int kAtomK = 1;
constexpr int kEpilogueStages = 4;
constexpr int kNaturalCtas = 150;
constexpr int kLaunchCtas = 150;
constexpr int kMaxActiveClusters = 0;
constexpr const char* kProductionTacticId =
  "c384-s225-residual-ffn-down-m6300-k1024-n384-"
  "m128n128k32s3-natural-b28-abi1";

using PrepareFn = cudaError_t (*)(int deviceOrdinal);
using QueryFn = cudaError_t (*)(C384ResidualRawDescriptorV1* output);
using LaunchFn = cudaError_t (*)(
  const half* activation,
  const half* rowMajorWeights,
  half* residualInOut,
  int tokenRows,
  int deviceOrdinal,
  cudaStream_t stream
);

struct Tactic {
  std::uint32_t registryAbiVersion = 0;
  std::uint32_t nativeAbiVersion = 0;
  int batchSize = 0;
  int tokenRows = 0;
  int inputChannels = 0;
  int outputChannels = 0;
  const char* id = nullptr;
  PrepareFn prepare = nullptr;
  QueryFn query = nullptr;
  LaunchFn launch = nullptr;
};

struct RuntimeShape {
  int modelDepth = 0;
  int attentionBlockCount = 0;
  int ffnBlockCount = 0;
  bool alternatingAttentionFfn = false;
  int batchSize = 0;
  int tokenRows = 0;
  int boardX = 0;
  int boardY = 0;
  int sequenceLength = 0;
  int channels = 0;
  int ffnChannels = 0;
  int deviceOrdinal = -1;
  std::uint32_t computeCapability = 0;
  bool usingFp16 = false;
  bool usingNhwc = false;
  bool exactNoMask = false;
  bool swiglu = false;
};

enum class RejectReason : std::uint32_t {
  None = 0,
  ShapeMismatch,
  NoRequestedTactic,
  InvalidRegistry,
  RegistryMiss,
  InvalidImplementation,
  QueryFailed,
  QueryMismatch,
  PreparationFailed,
};

struct PreparedSelection {
  const Tactic* tactic = nullptr;
  RejectReason reason = RejectReason::ShapeMismatch;
  int deviceOrdinal = -1;

  bool selected() const {
    return tactic != nullptr && reason == RejectReason::None &&
      deviceOrdinal >= 0;
  }
};

// A normal build links an empty provider. A generated production package
// replaces only this symbol with one hash-bound B28 descriptor. FFN-down is
// deliberately not part of the QKV/dual registry.
const Tactic* generatedTactics(std::size_t& count);

bool targetShapeEligible(const RuntimeShape& shape);
bool tacticWellFormed(const Tactic& tactic);
bool registryWellFormed(const Tactic* tactics, std::size_t count);
bool queryMatchesTactic(
  const C384ResidualRawDescriptorV1& native,
  const Tactic& tactic
);

// The explicit-registry form is the CPU-testable contract. It performs query
// and eager preparation before publishing a tactic pointer.
PreparedSelection prepareSelectionFromRegistry(
  const RuntimeShape& shape,
  const char* requestedId,
  const Tactic* tactics,
  std::size_t count
);

// Production wrapper supplied by c384_exact_ffn_down_aot_registry.cu.
PreparedSelection prepareSelection(
  const RuntimeShape& shape,
  const char* requestedId
);

// Complete hot-path preflight. False means use the already-prepared cuBLAS
// beta-one implementation before any exact FFN work is enqueued.
bool supports(
  const PreparedSelection& selection,
  const RuntimeShape& actualShape,
  const void* activation,
  const void* rowMajorWeights,
  const void* residualInOut
);

const char* rejectReasonName(RejectReason reason);

}  // namespace C384ExactFfnDownAot

#endif  // KATAGO_C384_EXACT_FFN_DOWN_AOT_H_

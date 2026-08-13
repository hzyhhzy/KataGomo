#ifndef KATAGO_C384_H12_FA4_SM120_H_
#define KATAGO_C384_H12_FA4_SM120_H_

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace C384H12Fa4Sm120 {

constexpr uint32_t kRegistryAbiVersion = 2;
constexpr uint32_t kPreparedProofAbiVersion = 2;
constexpr int kSequenceLength = 225;
constexpr int kHeads = 12;
constexpr int kHeadDim = 32;

enum class ArtifactMode : uint32_t {
  Disabled = 0,
  BatchSearch = 1,
  Production = 2,
};

enum class InputLayout : uint32_t {
  PlanarQkv = 1,
  PackedTokenQkv = 2,
};

enum class Accumulation : uint32_t {
  Fp32 = 1,
  QkFp16 = 2,
  PvFp16 = 3,
  BothFp16 = 4,
};

using IntAccessor = int (*)();
using IdAccessor = const char* (*)();
using PrepareFn = cudaError_t (*)(int deviceOrdinal);
using LaunchFn = cudaError_t (*)(
  void* q,
  void* k,
  void* v,
  void* output,
  int batch,
  int sequence,
  int heads,
  int headDim,
  float softmaxScale,
  uint32_t inputLayout,
  int deviceOrdinal,
  cudaStream_t stream
);

// Every generated AOT bridge keeps its own native symbol prefix. Search
// artifacts therefore coexist in one binary without ELF symbol rewriting.
// Accessors are checked before the first enqueue to catch stale headers or a
// package whose metadata and linked object disagree.
struct Candidate {
  uint32_t abiVersion = 0;
  int batch = 0;
  int sequence = 0;
  int heads = 0;
  int headDim = 0;
  int tileM = 0;
  int tileN = 0;
  int numStages = 0;
  int numWarps = 0;
  InputLayout inputLayout = InputLayout::PlanarQkv;
  Accumulation accumulation = Accumulation::Fp32;
  const char* id = nullptr;
  IntAccessor compiledBatch = nullptr;
  IntAccessor compiledSequence = nullptr;
  IntAccessor compiledHeads = nullptr;
  IntAccessor compiledHeadDim = nullptr;
  IntAccessor compiledTileM = nullptr;
  IntAccessor compiledTileN = nullptr;
  IntAccessor compiledNumStages = nullptr;
  IntAccessor compiledNumWarps = nullptr;
  IntAccessor compiledInputLayout = nullptr;
  IntAccessor compiledAccumulation = nullptr;
  IdAccessor compiledId = nullptr;
  PrepareFn prepare = nullptr;
  LaunchFn launch = nullptr;
};

// Narrow immutable capability handed to the exact QKV+RoPE workstream. A
// packed producer may select only when this proof was prepared for the same
// physical batch, device and explicit packed-token layout.
struct PreparedProof {
  uint32_t abiVersion = 0;
  int batch = 0;
  int sequence = 0;
  int heads = 0;
  int headDim = 0;
  int deviceOrdinal = -1;
  InputLayout inputLayout = InputLayout::PlanarQkv;
  const char* id = nullptr;
  LaunchFn launch = nullptr;
  uintptr_t implementationCookie = 0;
};

struct RegistryView {
  ArtifactMode mode = ArtifactMode::Disabled;
  const Candidate* candidates = nullptr;
  std::size_t count = 0;
};

struct LaunchResult {
  bool attempted = false;
  cudaError_t status = cudaSuccess;
  const char* marker = nullptr;

  bool launched() const { return attempted && status == cudaSuccess; }
};

// A checked-in empty provider is used by normal builds. The package generator
// supplies this function in search/production builds.
RegistryView generatedRegistry();

bool candidateWellFormed(const Candidate& candidate);
bool registryWellFormed(const RegistryView& registry);
const Candidate* findExactCandidate(const RegistryView& registry, int batch);
bool supportsExactBatch(int batch);
cudaError_t prepareProofForExactBatch(
  int batch,
  int deviceOrdinal,
  InputLayout requiredLayout,
  PreparedProof& proof
);
bool proofCompatible(
  const PreparedProof& proof,
  int batch,
  int deviceOrdinal,
  InputLayout requiredLayout
);

// This hook replaces only the no-mask SDPA operation. All pre-enqueue
// mismatches return attempted=false for cuDNN fallback. Once the exact
// candidate is owned, bad pointers, ABI drift, or launch failure are surfaced
// and another attention implementation must not be enqueued.
LaunchResult launch(
  half* q,
  half* k,
  half* v,
  half* output,
  int batch,
  int sequence,
  int heads,
  int kvHeads,
  int qHeadDim,
  int vHeadDim,
  bool usingFp16,
  bool usingNhwc,
  InputLayout inputLayout,
  const void* mask,
  bool recipeEligibleC384,
  const PreparedProof* preparedProof,
  int deviceOrdinal,
  int computeMajor,
  int computeMinor,
  cudaStream_t stream
);

}  // namespace C384H12Fa4Sm120

#endif  // KATAGO_C384_H12_FA4_SM120_H_

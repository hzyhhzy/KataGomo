#ifndef KATAGO_C384_QKNORM_ROPE_SM120_H_
#define KATAGO_C384_QKNORM_ROPE_SM120_H_

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace C384QKNormRopeSm120 {

// This ABI consumes the raw packed output of the C384 exact QKV GEMM:
//   [token][Q384 | K384 | V384]
// Q and K are normalized independently per H12/D32 head, rounded to FP16,
// and then transformed by the learned-RoPE table. V remains byte-for-byte
// unchanged, preserving the existing packed-input FA4 contract.
constexpr uint32_t kAbiVersion = 1;
constexpr int kBatch = 28;
constexpr int kSequence = 225;
constexpr int kHeads = 12;
constexpr int kHeadDim = 32;
constexpr int kChannels = kHeads * kHeadDim;
constexpr int kPackedChannels = 3 * kChannels;
constexpr int kRopePairsPerHead = kHeadDim / 2;
constexpr int kRopePairsTotal = kHeads * kRopePairsPerHead;
constexpr int kThreads = 256;
constexpr int kGridBlocks = 340;
constexpr float kRmsEpsilon = 1.0e-6f;

enum class InputSemantic : uint32_t {
  RawPackedQkv = 1,
};

struct LaunchParams {
  uint32_t abiVersion = 0;
  int batch = 0;
  int sequence = 0;
  int heads = 0;
  int kvHeads = 0;
  int headDim = 0;
  int tokenRows = 0;
  int deviceOrdinal = -1;
  uint32_t computeCapability = 0;
  bool usingFp16 = false;
  bool usingNhwc = false;
  bool learnedRope = false;
  bool qkNorm = false;
  InputSemantic inputSemantic = InputSemantic::RawPackedQkv;
  float qEpsilon = 0.0f;
  float kEpsilon = 0.0f;
};

bool supports(const LaunchParams& params) noexcept;
const char* marker() noexcept;

// qGamma/kGamma are D32 vectors. learnedRopeCosSin is packed as
// [sequence][heads][headDim/2] half2(cos,sin). No allocation, initialization,
// or synchronization occurs here. Shape/semantic mismatches return
// cudaErrorNotSupported before enqueue; malformed pointers return
// cudaErrorInvalidValue.
cudaError_t launchInPlace(
  const LaunchParams& params,
  half* rawPackedQkv,
  const half* qGamma,
  const half* kGamma,
  const half2* learnedRopeCosSin,
  cudaStream_t stream
);

}  // namespace C384QKNormRopeSm120

#endif  // KATAGO_C384_QKNORM_ROPE_SM120_H_

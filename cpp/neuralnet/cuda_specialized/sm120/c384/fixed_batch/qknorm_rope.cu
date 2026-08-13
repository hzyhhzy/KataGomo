#include "qknorm_rope.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace C384QKNormRopeSm120 {

namespace {

constexpr int kPlanesToTransform = 2;
constexpr int kSubgroupWidth = 16;
constexpr int kSubgroupsPerWarp = 2;
constexpr int kWarpsPerBlock = kThreads / 32;
constexpr int kSubgroupsPerBlock = kWarpsPerBlock * kSubgroupsPerWarp;

static_assert(kHeadDim == 2 * kSubgroupWidth,
  "one 16-lane subgroup must cover one D32 head using half2");
static_assert(kThreads % 32 == 0,"QKNorm+RoPE CTA must contain whole warps");
static_assert(kPackedChannels % 2 == 0,"packed QKV rows must be half2 aligned");

__global__ __launch_bounds__(kThreads)
void qknormLearnedRopeHalf2Kernel(
  half* packedQkv,
  const half* qGamma,
  const half* kGamma,
  const half2* ropeCosSin,
  int tokenRows,
  float qEpsilon,
  float kEpsilon
) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int subgroupLane = lane & (kSubgroupWidth - 1);
  const int subgroupInWarp = lane / kSubgroupWidth;
  const int warpInBlock = static_cast<int>(threadIdx.x) / 32;
  const int subgroupInBlock =
    warpInBlock * kSubgroupsPerWarp + subgroupInWarp;
  const int totalHeadVectors = tokenRows * kPlanesToTransform * kHeads;

  for(int vectorIndex =
        static_cast<int>(blockIdx.x) * kSubgroupsPerBlock + subgroupInBlock;
      vectorIndex < totalHeadVectors;
      vectorIndex += static_cast<int>(gridDim.x) * kSubgroupsPerBlock) {
    const int token = vectorIndex / (kPlanesToTransform * kHeads);
    const int vectorInToken = vectorIndex % (kPlanesToTransform * kHeads);
    const int plane = vectorInToken / kHeads;
    const int head = vectorInToken % kHeads;
    const int channelOffset = plane * kChannels + head * kHeadDim;
    half2* const headData = reinterpret_cast<half2*>(
      packedQkv + static_cast<size_t>(token) * kPackedChannels + channelOffset);

    const half2 rawHalf2 = headData[subgroupLane];
    const float2 raw = __half22float2(rawHalf2);
    float sumSquares = raw.x * raw.x + raw.y * raw.y;

    // XOR shuffles with offsets below 16 reduce independently in each half
    // warp. The full mask keeps both 16-lane subgroups converged.
    sumSquares += __shfl_xor_sync(0xffffffffu,sumSquares,8);
    sumSquares += __shfl_xor_sync(0xffffffffu,sumSquares,4);
    sumSquares += __shfl_xor_sync(0xffffffffu,sumSquares,2);
    sumSquares += __shfl_xor_sync(0xffffffffu,sumSquares,1);

    const float epsilon = plane == 0 ? qEpsilon : kEpsilon;
    const float invRms = rsqrtf(sumSquares / static_cast<float>(kHeadDim) + epsilon);
    const half2* const gamma = reinterpret_cast<const half2*>(
      plane == 0 ? qGamma : kGamma);
    const float2 gammaFloat = __half22float2(gamma[subgroupLane]);

    // PyTorch Q/K RMSNorm produces an FP16 tensor before learned RoPE under
    // the target mixed-precision inference contract. Preserve that semantic
    // rounding boundary instead of carrying FP32 normalized values into RoPE.
    const half2 normalizedHalf2 = __floats2half2_rn(
      raw.x * invRms * gammaFloat.x,
      raw.y * invRms * gammaFloat.y);
    const float2 normalized = __half22float2(normalizedHalf2);

    const int xy = token % kSequence;
    const half2 ropeHalf2 = ropeCosSin[
      (static_cast<size_t>(xy) * kHeads + head) * kRopePairsPerHead +
      subgroupLane];
    const float2 rope = __half22float2(ropeHalf2);
    headData[subgroupLane] = __floats2half2_rn(
      normalized.x * rope.x - normalized.y * rope.y,
      normalized.x * rope.y + normalized.y * rope.x);
  }
}

}  // namespace

bool supports(const LaunchParams& params) noexcept {
  return params.abiVersion == kAbiVersion &&
    params.batch == kBatch && params.sequence == kSequence &&
    params.heads == kHeads && params.kvHeads == kHeads &&
    params.headDim == kHeadDim &&
    params.tokenRows == params.batch * params.sequence &&
    params.deviceOrdinal >= 0 && params.computeCapability == 120 &&
    params.usingFp16 && params.usingNhwc && params.learnedRope &&
    params.qkNorm && params.inputSemantic == InputSemantic::RawPackedQkv &&
    params.qEpsilon == kRmsEpsilon && params.kEpsilon == kRmsEpsilon;
}

const char* marker() noexcept {
  return "c384-h12-d32-raw-packed-qknorm-rope-half2-g340-v1";
}

cudaError_t launchInPlace(
  const LaunchParams& params,
  half* rawPackedQkv,
  const half* qGamma,
  const half* kGamma,
  const half2* learnedRopeCosSin,
  cudaStream_t stream
) {
  return launchInPlaceForGridQualification(
    params,rawPackedQkv,qGamma,kGamma,learnedRopeCosSin,kGridBlocks,stream);
}

cudaError_t launchInPlaceForGridQualification(
  const LaunchParams& params,
  half* rawPackedQkv,
  const half* qGamma,
  const half* kGamma,
  const half2* learnedRopeCosSin,
  int gridBlocks,
  cudaStream_t stream
) {
  if(!supports(params))
    return cudaErrorNotSupported;
  if(rawPackedQkv == nullptr || qGamma == nullptr || kGamma == nullptr ||
     learnedRopeCosSin == nullptr)
    return cudaErrorInvalidValue;
  if(gridBlocks <= 0 || gridBlocks > 4096)
    return cudaErrorInvalidConfiguration;
  qknormLearnedRopeHalf2Kernel<<<gridBlocks,kThreads,0,stream>>>(
    rawPackedQkv,qGamma,kGamma,learnedRopeCosSin,params.tokenRows,
    params.qEpsilon,params.kEpsilon);
  return cudaPeekAtLastError();
}

}  // namespace C384QKNormRopeSm120

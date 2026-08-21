// CUDA wrapper for the shared CUDA/ROCm GPU kernels.
// All kernel code lives in cudaandrocmhelpers.inc, which is shared with the ROCm backend
// (rocmhelpers.hip). See the comment at the top of that file for the macro contract.

#include "../neuralnet/cudahelpers.h"

#include <cmath>
#include <stdexcept>

// Evaluated per device architecture during nvcc's per-arch device compilation passes, so the
// half-precision kernel bodies are compiled exactly for the archs that support them.
#if __CUDA_ARCH__ >= 530
#define KATAGO_GPU_SUPPORTS_FP16
#endif

#define KATAGO_GPU_CUDA 1
#define KATAGO_GPU_SINCOSF __sincosf

// Tensor-core flash attention is CUDA-only (mma.sync/ldmatrix/cp.async PTX). The shared .inc
// provides a fallback stub for builds without it.
#include "../neuralnet/cudaflashmma.cuh"
#define KATAGO_HAS_FLASH_MMA 1

#include "../neuralnet/cudaandrocmhelpers.inc"

// Keep this CUDA-only helper separate from the shared legacy SwiGLU kernels.
// The comparison form deliberately propagates NaN, matching torch.clamp and
// the established clip epilogue; fminf/fmaxf would replace NaN with a bound.
__device__ __forceinline__ float orderedClampSymmetric(float value, float clip) {
  return value > clip ? clip : (value < -clip ? -clip : value);
}

template<int ELTS_PER_THREAD>
__global__ void swiGLUOrderedClippedFP16Kernel(
  const half* linear, const half* gate, half* out, int size, float clip
) {
#ifdef KATAGO_GPU_SUPPORTS_FP16
  const half2* linear2 = reinterpret_cast<const half2*>(linear);
  const half2* gate2 = reinterpret_cast<const half2*>(gate);
  half2* out2 = reinterpret_cast<half2*>(out);
  const int pairCount = size >> 1;
  const int tileStart = blockIdx.x * blockDim.x * ELTS_PER_THREAD;
  const int lane = threadIdx.x;

  #pragma unroll
  for(int d = 0; d < ELTS_PER_THREAD; d++) {
    const int pair = tileStart + d * blockDim.x + lane;
    if(pair < pairCount) {
      const half2 linearValue = linear2[pair];
      const half2 gateValue = gate2[pair];
      const float linear0 = orderedClampSymmetric(
        siluf(__half2float(__low2half(linearValue))),clip);
      const float linear1 = orderedClampSymmetric(
        siluf(__half2float(__high2half(linearValue))),clip);
      const float gate0 = orderedClampSymmetric(
        __half2float(__low2half(gateValue)),clip);
      const float gate1 = orderedClampSymmetric(
        __half2float(__high2half(gateValue)),clip);
      out2[pair] = __halves2half2(
        __float2half_rn(linear0 * gate0),
        __float2half_rn(linear1 * gate1)
      );
    }
  }

  if((size & 1) != 0 && blockIdx.x == 0 && lane == 0) {
    const int last = size - 1;
    const float linearValue = orderedClampSymmetric(
      siluf(__half2float(linear[last])),clip);
    const float gateValue = orderedClampSymmetric(__half2float(gate[last]),clip);
    out[last] = __float2half_rn(linearValue * gateValue);
  }
#else
  (void)linear;
  (void)gate;
  (void)out;
  (void)size;
  (void)clip;
#endif
}

void customCudaSwiGLUOrderedClippedFP16(
  const half* linear, const half* gate, half* out, int size, float clip,
  cudaStream_t stream
) {
  if(size <= 0)
    return;
  if(!(clip > 0.0f) || !std::isfinite(clip))
    throw std::runtime_error(
      "customCudaSwiGLUOrderedClippedFP16: clip must be finite and positive");
  constexpr int ELTS_PER_THREAD = 4;
  constexpr int threads = 256;
  const int pairCount = size >> 1;
  int blocks =
    (pairCount + threads * ELTS_PER_THREAD - 1) /
    (threads * ELTS_PER_THREAD);
  if(blocks < 1)
    blocks = 1;
  swiGLUOrderedClippedFP16Kernel<ELTS_PER_THREAD>
    <<<blocks,threads,0,stream>>>(linear,gate,out,size,clip);
}

namespace {

__global__ void buildLearnedRopeTableFP32Kernel(
  const float* freqs,
  float2* cosSinTable,
  int totalPairs,
  int numKVHeads,
  int numPairs,
  int nnXLen
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if(index >= totalPairs)
    return;

  const int pairsPerPosition = numKVHeads * numPairs;
  const int xy = index / pairsPerPosition;
  const int headPair = index - xy * pairsPerPosition;
  const int kvHead = headPair / numPairs;
  const int pair = headPair - kvHead * numPairs;
  const int x = xy % nnXLen;
  const int y = xy / nnXLen;
  const float freqX = freqs[(kvHead * numPairs + pair) * 2];
  const float freqY = freqs[(kvHead * numPairs + pair) * 2 + 1];
  const float angle = static_cast<float>(x) * freqX +
    static_cast<float>(y) * freqY;
  float sinValue;
  float cosValue;
  __sincosf(angle,&sinValue,&cosValue);
  cosSinTable[index] = make_float2(cosValue,sinValue);
}

template<int HEAD_DIM, CudaFusedQKNormRopeMode ROPE_MODE>
__global__ __launch_bounds__(256)
void fusedQKNormRoPEFP16Kernel(
  half* qBuf,
  half* kBuf,
  const half2* qGamma,
  const half2* kGamma,
  const half* fixedCosTable,
  const half* fixedSinTable,
  const float2* learnedCosSinTable,
  int tokenRows,
  int seqLen,
  int numHeads,
  int numKVHeads,
  int qStride,
  int kvStride,
  float qEpsilon,
  float kEpsilon
) {
#ifdef KATAGO_GPU_SUPPORTS_FP16
  constexpr int PAIRS_PER_HEAD = HEAD_DIM / 2;
  constexpr int SUBGROUP_WIDTH = PAIRS_PER_HEAD;
  static_assert(SUBGROUP_WIDTH == 16 || SUBGROUP_WIDTH == 32,
    "fused QKNorm supports D32 and D64");

  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int subgroupLane = lane & (SUBGROUP_WIDTH - 1);
  const int subgroupInBlock = static_cast<int>(threadIdx.x) / SUBGROUP_WIDTH;
  const int subgroupsPerBlock = static_cast<int>(blockDim.x) / SUBGROUP_WIDTH;
  const int vectorsPerToken = numHeads + numKVHeads;
  const int totalVectors = tokenRows * vectorsPerToken;

  for(int vectorIndex = static_cast<int>(blockIdx.x) * subgroupsPerBlock +
        subgroupInBlock;
      vectorIndex < totalVectors;
      vectorIndex += static_cast<int>(gridDim.x) * subgroupsPerBlock) {
    const int token = vectorIndex / vectorsPerToken;
    const int vectorInToken = vectorIndex - token * vectorsPerToken;
    const bool isQuery = vectorInToken < numHeads;
    const int head = isQuery ? vectorInToken : vectorInToken - numHeads;
    const int stride = isQuery ? qStride : kvStride;
    half* const base = isQuery ? qBuf : kBuf;
    half2* const headData = reinterpret_cast<half2*>(
      base + static_cast<size_t>(token) * stride + head * HEAD_DIM);

    const half2 rawHalf2 = headData[subgroupLane];
    const float2 raw = __half22float2(rawHalf2);
    float sumSquares = raw.x * raw.x + raw.y * raw.y;
    #pragma unroll
    for(int offset = SUBGROUP_WIDTH / 2; offset > 0; offset >>= 1)
      sumSquares += __shfl_xor_sync(0xffffffffu,sumSquares,offset);

    const float epsilon = isQuery ? qEpsilon : kEpsilon;
    const float invRms = rsqrtf(
      sumSquares / static_cast<float>(HEAD_DIM) + epsilon);
    const half2 gammaHalf2 =
      (isQuery ? qGamma : kGamma)[subgroupLane];
    const float2 gamma = __half22float2(gammaHalf2);

    // Keep the semantic FP16 boundary between RMSNorm and RoPE.
    const half2 normalizedHalf2 = __floats2half2_rn(
      raw.x * invRms * gamma.x,
      raw.y * invRms * gamma.y);

    if(ROPE_MODE == CudaFusedQKNormRopeMode::None) {
      headData[subgroupLane] = normalizedHalf2;
    }
    else {
      const float2 normalized = __half22float2(normalizedHalf2);
      const int xy = token % seqLen;
      float cosValue;
      float sinValue;
      if(ROPE_MODE == CudaFusedQKNormRopeMode::Fixed) {
        const int tableIndex = subgroupLane * seqLen + xy;
        cosValue = __half2float(fixedCosTable[tableIndex]);
        sinValue = __half2float(fixedSinTable[tableIndex]);
      }
      else {
        const int ropeHead = isQuery ?
          head * numKVHeads / numHeads : head;
        const size_t tableIndex =
          (static_cast<size_t>(xy) * numKVHeads + ropeHead) *
            PAIRS_PER_HEAD + subgroupLane;
        const float2 rope = learnedCosSinTable[tableIndex];
        cosValue = rope.x;
        sinValue = rope.y;
      }
      headData[subgroupLane] = __floats2half2_rn(
        normalized.x * cosValue - normalized.y * sinValue,
        normalized.x * sinValue + normalized.y * cosValue);
    }
  }
#else
  (void)qBuf; (void)kBuf; (void)qGamma; (void)kGamma;
  (void)fixedCosTable; (void)fixedSinTable; (void)learnedCosSinTable;
  (void)tokenRows; (void)seqLen; (void)numHeads; (void)numKVHeads;
  (void)qStride; (void)kvStride; (void)qEpsilon; (void)kEpsilon;
#endif
}

template<int HEAD_DIM>
void launchFusedQKNormRoPEFP16ForHeadDim(
  half* qBuf,
  half* kBuf,
  const half* qGamma,
  const half* kGamma,
  const half* fixedCosTable,
  const half* fixedSinTable,
  const float* learnedCosSinTable,
  int tokenRows,
  int seqLen,
  int numHeads,
  int numKVHeads,
  int qStride,
  int kvStride,
  float qEpsilon,
  float kEpsilon,
  CudaFusedQKNormRopeMode ropeMode,
  int multiprocessorCount,
  cudaStream_t stream
) {
  constexpr int THREADS = 256;
  constexpr int SUBGROUPS_PER_BLOCK = THREADS / (HEAD_DIM / 2);
  const int totalVectors = tokenRows * (numHeads + numKVHeads);
  const int requiredBlocks =
    (totalVectors + SUBGROUPS_PER_BLOCK - 1) / SUBGROUPS_PER_BLOCK;
  const int residentGrid = multiprocessorCount > 0 ?
    multiprocessorCount * 2 : 256;
  const int blocks = requiredBlocks < residentGrid ?
    requiredBlocks : residentGrid;
  if(blocks <= 0)
    return;

  const half2* const qGamma2 = reinterpret_cast<const half2*>(qGamma);
  const half2* const kGamma2 = reinterpret_cast<const half2*>(kGamma);
  const float2* const learnedTable2 =
    reinterpret_cast<const float2*>(learnedCosSinTable);
  switch(ropeMode) {
  case CudaFusedQKNormRopeMode::None:
    fusedQKNormRoPEFP16Kernel<HEAD_DIM,CudaFusedQKNormRopeMode::None>
      <<<blocks,THREADS,0,stream>>>(
        qBuf,kBuf,qGamma2,kGamma2,fixedCosTable,fixedSinTable,learnedTable2,
        tokenRows,seqLen,numHeads,numKVHeads,qStride,kvStride,
        qEpsilon,kEpsilon);
    break;
  case CudaFusedQKNormRopeMode::Fixed:
    fusedQKNormRoPEFP16Kernel<HEAD_DIM,CudaFusedQKNormRopeMode::Fixed>
      <<<blocks,THREADS,0,stream>>>(
        qBuf,kBuf,qGamma2,kGamma2,fixedCosTable,fixedSinTable,learnedTable2,
        tokenRows,seqLen,numHeads,numKVHeads,qStride,kvStride,
        qEpsilon,kEpsilon);
    break;
  case CudaFusedQKNormRopeMode::LearnedTable:
    fusedQKNormRoPEFP16Kernel<HEAD_DIM,CudaFusedQKNormRopeMode::LearnedTable>
      <<<blocks,THREADS,0,stream>>>(
        qBuf,kBuf,qGamma2,kGamma2,fixedCosTable,fixedSinTable,learnedTable2,
        tokenRows,seqLen,numHeads,numKVHeads,qStride,kvStride,
        qEpsilon,kEpsilon);
    break;
  default:
    throw std::runtime_error("customCudaFusedQKNormRoPEFP16: invalid RoPE mode");
  }
}

}  // namespace

bool customCudaFusedQKNormRoPESupportsShape(int qHeadDim) {
  return qHeadDim == 32 || qHeadDim == 64;
}

void customCudaBuildLearnedRopeTableFP32(
  const float* freqs,
  float* cosSinTable,
  int seqLen,
  int numKVHeads,
  int numPairs,
  int nnXLen,
  cudaStream_t stream
) {
  if(freqs == nullptr || cosSinTable == nullptr)
    throw std::runtime_error(
      "customCudaBuildLearnedRopeTableFP32: null buffer");
  if(seqLen <= 0 || numKVHeads <= 0 || numPairs <= 0 || nnXLen <= 0)
    throw std::runtime_error(
      "customCudaBuildLearnedRopeTableFP32: invalid shape");
  const long long totalPairs64 =
    static_cast<long long>(seqLen) * numKVHeads * numPairs;
  if(totalPairs64 > 2147483647LL)
    throw std::runtime_error(
      "customCudaBuildLearnedRopeTableFP32: table exceeds int indexing");
  constexpr int THREADS = 256;
  const int totalPairs = static_cast<int>(totalPairs64);
  const int blocks = (totalPairs + THREADS - 1) / THREADS;
  buildLearnedRopeTableFP32Kernel<<<blocks,THREADS,0,stream>>>(
    freqs,reinterpret_cast<float2*>(cosSinTable),totalPairs,
    numKVHeads,numPairs,nnXLen);
}

void customCudaFusedQKNormRoPEFP16(
  half* qBuf,
  half* kBuf,
  const half* qGamma,
  const half* kGamma,
  const half* fixedCosTable,
  const half* fixedSinTable,
  const float* learnedCosSinTable,
  int tokenRows,
  int seqLen,
  int numHeads,
  int numKVHeads,
  int qHeadDim,
  int qStride,
  int kvStride,
  int numPairs,
  float qEpsilon,
  float kEpsilon,
  CudaFusedQKNormRopeMode ropeMode,
  int multiprocessorCount,
  cudaStream_t stream
) {
  if(qBuf == nullptr || kBuf == nullptr || qGamma == nullptr || kGamma == nullptr)
    throw std::runtime_error("customCudaFusedQKNormRoPEFP16: null Q/K buffer");
  if(tokenRows <= 0 || seqLen <= 0 || numHeads <= 0 || numKVHeads <= 0 ||
     qStride <= 0 || kvStride <= 0 ||
     (ropeMode != CudaFusedQKNormRopeMode::None &&
      numPairs != qHeadDim / 2) ||
     !customCudaFusedQKNormRoPESupportsShape(qHeadDim))
    throw std::runtime_error("customCudaFusedQKNormRoPEFP16: invalid shape");
  if(ropeMode == CudaFusedQKNormRopeMode::Fixed &&
     (fixedCosTable == nullptr || fixedSinTable == nullptr))
    throw std::runtime_error(
      "customCudaFusedQKNormRoPEFP16: missing fixed RoPE table");
  if(ropeMode == CudaFusedQKNormRopeMode::LearnedTable &&
     learnedCosSinTable == nullptr)
    throw std::runtime_error(
      "customCudaFusedQKNormRoPEFP16: missing learned RoPE table");
  const long long totalVectors64 =
    static_cast<long long>(tokenRows) * (numHeads + numKVHeads);
  if(totalVectors64 > 2147483647LL)
    throw std::runtime_error(
      "customCudaFusedQKNormRoPEFP16: vector count exceeds int indexing");

  if(qHeadDim == 32)
    launchFusedQKNormRoPEFP16ForHeadDim<32>(
      qBuf,kBuf,qGamma,kGamma,fixedCosTable,fixedSinTable,
      learnedCosSinTable,tokenRows,seqLen,numHeads,numKVHeads,qStride,kvStride,
      qEpsilon,kEpsilon,ropeMode,multiprocessorCount,stream);
  else
    launchFusedQKNormRoPEFP16ForHeadDim<64>(
      qBuf,kBuf,qGamma,kGamma,fixedCosTable,fixedSinTable,
      learnedCosSinTable,tokenRows,seqLen,numHeads,numKVHeads,qStride,kvStride,
      qEpsilon,kEpsilon,ropeMode,multiprocessorCount,stream);
}

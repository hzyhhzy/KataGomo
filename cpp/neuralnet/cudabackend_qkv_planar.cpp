#ifdef USE_CUDA_BACKEND

#include "../neuralnet/cudabackend_qkv_planar.h"
#include "../neuralnet/cudautils.h"
#if defined(KATAGO_ENABLE_RENJU15_QKV_ROPE_GEMM_SM120) && KATAGO_ENABLE_RENJU15_QKV_ROPE_GEMM_SM120
#include "../neuralnet/renju15_qkv_rope_gemm_sm120.h"
#endif

#include "../core/using.h"

namespace CudaQKVPlanar {

Projection::Projection(
  const vector<float>& qWeights,
  const vector<float>& kWeights,
  const vector<float>& vWeights,
  int inputChannels_,
  int projectionChannels_,
  const CudaTransformerWinner::AttentionRecipe& recipe,
  int fixedBatchSize
) :
  inputChannels(inputChannels_),
  projectionChannels(projectionChannels_),
  packedWeights(nullptr)
#if defined(KATAGO_ENABLE_RENJU15_QKV_ROPE_GEMM_SM120) && KATAGO_ENABLE_RENJU15_QKV_ROPE_GEMM_SM120
  , qkvRopeGemmHandle(nullptr)
#endif
{
  if(recipe.planarQkv !=
     CudaTransformerWinner::PlanarQkvTactic::CublasHgemmStridedBatchedSquare ||
     inputChannels <= 0 || projectionChannels != inputChannels)
    throw StringError("QKV planar projection constructed with incompatible recipe");
  const size_t matrixElements =
    (size_t)inputChannels * (size_t)projectionChannels;
  if(qWeights.size() != matrixElements || kWeights.size() != matrixElements ||
     vWeights.size() != matrixElements)
    throw StringError("Selected planar QKV received incompatible projection weights");

  vector<float> packed;
  packed.reserve(NUM_PROJECTIONS * matrixElements);
  packed.insert(packed.end(),qWeights.begin(),qWeights.end());
  packed.insert(packed.end(),kWeights.begin(),kWeights.end());
  packed.insert(packed.end(),vWeights.begin(),vWeights.end());
  CudaUtils::mallocAndCopyToDevice(
    "selectedQKVPlanar:packedWeights",packed,packedWeights,true);
#if defined(KATAGO_ENABLE_RENJU15_QKV_ROPE_GEMM_SM120) && KATAGO_ENABLE_RENJU15_QKV_ROPE_GEMM_SM120
  if(recipe.qkvRope ==
       CudaTransformerWinner::QkvRopeTactic::Sm120C256H8D32M128N128K32S3 &&
     inputChannels == 256 && projectionChannels == 256 && fixedBatchSize == 36) {
    qkvRopeGemmHandle = katago_renju15_qkv_rope_gemm_sm120_create(
      KATAGO_RENJU15_QKV_ROPE_GEMM_M128_N128_K32_S3,fixedBatchSize);
  }
#else
  (void)fixedBatchSize;
#endif
}

Projection::~Projection() {
#if defined(KATAGO_ENABLE_RENJU15_QKV_ROPE_GEMM_SM120) && KATAGO_ENABLE_RENJU15_QKV_ROPE_GEMM_SM120
  katago_renju15_qkv_rope_gemm_sm120_destroy(qkvRopeGemmHandle);
#endif
  if(packedWeights != nullptr)
    cudaFree(packedWeights);
}

cublasStatus_t Projection::apply(
  cublasHandle_t handle,
  const half* alpha,
  const half* input,
  const half* beta,
  half* output,
  int tokenCount
) const {
  if(handle == nullptr || input == nullptr || output == nullptr ||
     tokenCount <= 0 || inputChannels <= 0 || projectionChannels <= 0)
    return CUBLAS_STATUS_INVALID_VALUE;
  const long long weightStride =
    (long long)inputChannels * (long long)projectionChannels;
  const long long outputStride =
    (long long)projectionChannels * (long long)tokenCount;
  return cublasHgemmStridedBatched(
    handle,CUBLAS_OP_N,CUBLAS_OP_N,
    projectionChannels,tokenCount,inputChannels,
    alpha,
    (const half*)packedWeights,projectionChannels,weightStride,
    input,inputChannels,0,
    beta,
    output,projectionChannels,outputStride,
    NUM_PROJECTIONS);
}

#if defined(KATAGO_ENABLE_RENJU15_QKV_ROPE_GEMM_SM120) && KATAGO_ENABLE_RENJU15_QKV_ROPE_GEMM_SM120
bool Projection::supportsFusedQKVRoPE(
  int batchSize,
  int seqLen,
  int numHeads,
  int numKVHeads,
  int headDim,
  int ropePairs,
  bool usingFP16,
  bool usingNHWC,
  bool exactNoMask,
  bool precomputedHalf2,
  const half2* cosSin
) const {
  return packedWeights != nullptr && cosSin != nullptr &&
    katago_renju15_qkv_rope_gemm_sm120_supports(
      qkvRopeGemmHandle,batchSize,seqLen,inputChannels,projectionChannels,
      numHeads,numKVHeads,headDim,ropePairs,usingFP16,usingNHWC,
      exactNoMask,exactNoMask,precomputedHalf2);
}

cudaError_t Projection::applyFusedQKVRoPE(
  const half* input,
  const half2* cosSin,
  half* output,
  int batchSize,
  cudaStream_t stream
) const {
  return katago_renju15_qkv_rope_gemm_sm120_launch(
    qkvRopeGemmHandle,input,(const half*)packedWeights,cosSin,output,
    batchSize,stream);
}

const char* Projection::fusedQKVRoPEMarker() const {
  return katago_renju15_qkv_rope_gemm_sm120_active_marker(qkvRopeGemmHandle);
}
#endif

}  // namespace CudaQKVPlanar

#endif  // USE_CUDA_BACKEND

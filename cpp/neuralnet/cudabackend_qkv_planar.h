#ifndef NEURALNET_CUDABACKEND_QKV_PLANAR_H_
#define NEURALNET_CUDABACKEND_QKV_PLANAR_H_

#include "../neuralnet/cudabackend_transformer_winner.h"
#include "../neuralnet/cudaincludes.h"

#include <vector>

namespace CudaQKVPlanar {

constexpr int NUM_PROJECTIONS = 3;

// Packs three same-shape Q/K/V matrices once at block construction. The
// generic path is valid for any square MHA width (including C256 and C384);
// the optional fused SM120 epilogue remains gated to its exact local shape.
class Projection {
 public:
  Projection(
    const std::vector<float>& qWeights,
    const std::vector<float>& kWeights,
    const std::vector<float>& vWeights,
    int inputChannels,
    int projectionChannels,
    const CudaTransformerWinner::AttentionRecipe& recipe,
    int fixedBatchSize
  );
  ~Projection();

  Projection() = delete;
  Projection(const Projection&) = delete;
  Projection& operator=(const Projection&) = delete;

  cublasStatus_t apply(
    cublasHandle_t handle,
    const half* alpha,
    const half* input,
    const half* beta,
    half* output,
    int tokenCount
  ) const;

#if defined(KATAGO_ENABLE_RENJU15_QKV_ROPE_GEMM_SM120) && KATAGO_ENABLE_RENJU15_QKV_ROPE_GEMM_SM120
  bool supportsFusedQKVRoPE(
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
  ) const;

  cudaError_t applyFusedQKVRoPE(
    const half* input,
    const half2* cosSin,
    half* output,
    int batchSize,
    cudaStream_t stream
  ) const;

  const char* fusedQKVRoPEMarker() const;
  bool hasFusedQKVRoPEState() const;
#endif

 private:
  int inputChannels;
  int projectionChannels;
  void* packedWeights;
#if defined(KATAGO_ENABLE_RENJU15_QKV_ROPE_GEMM_SM120) && KATAGO_ENABLE_RENJU15_QKV_ROPE_GEMM_SM120
  void* qkvRopeGemmHandle;
#endif
};

}  // namespace CudaQKVPlanar

#endif  // NEURALNET_CUDABACKEND_QKV_PLANAR_H_

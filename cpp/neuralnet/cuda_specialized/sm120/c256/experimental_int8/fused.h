#ifndef KATAGO_RENJU15_INT8_FUSED_SM120_H_
#define KATAGO_RENJU15_INT8_FUSED_SM120_H_

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

// Experimental native INT8 path for C=256 SM120 transformer operations.
// Activation quantization is signed symmetric clip4 (scale 4/127, zp=0).
// Packed weights are signed symmetric per matrix (zp=0), laid out as CUTLASS
// ColumnMajor [K,N], and remain owned by the caller for the handle lifetime.

extern "C" void* katago_renju15_int8_qk_sm120_create(
  int maxTokenRows,
  const int8_t* packedQkWeights,
  float qkWeightScale);

extern "C" void katago_renju15_int8_qk_sm120_destroy(void* opaque);

extern "C" bool katago_renju15_int8_qk_sm120_supports(
  const void* opaque,
  int actualTokenRows,
  int inputChannels,
  int qChannels,
  int kChannels,
  bool usingFp16,
  bool usingNhwc,
  bool exactNoMask);

// Produces row-major [actualTokenRows,512] in temporary contiguous Q|K form.
extern "C" cudaError_t katago_renju15_int8_qk_sm120_launch(
  void* opaque,
  int actualTokenRows,
  const int8_t* activation,
  half* qkContiguous,
  cudaStream_t stream);

// Splits the temporary Q|K rows to the existing compact planar consumers and
// applies the learned RoPE in the same launch. This exact specialization is
// C256/H8/KVH8/D32 with 16 pairs per head.
extern "C" cudaError_t katago_renju15_int8_qk_split_rope_sm120_launch(
  const half* qkContiguous,
  half* qPlanar,
  half* kPlanar,
  const half2* ropeCosSin,
  int batchSize,
  int sequenceLength,
  cudaStream_t stream);

extern "C" void* katago_renju15_int8_dual_ffn_sm120_create(
  int maxTokenRows,
  const int8_t* packedUpWeights,
  const int8_t* packedGateWeights,
  float upWeightScale,
  float gateWeightScale);

extern "C" void katago_renju15_int8_dual_ffn_sm120_destroy(void* opaque);

extern "C" bool katago_renju15_int8_dual_ffn_sm120_supports(
  const void* opaque,
  int actualTokenRows,
  int inputChannels,
  int ffnChannels,
  bool usingFp16,
  bool usingNhwc,
  bool exactNoMask);

// Shared-A dual INT8 GEMM, scalar dequantization for each matrix, and SwiGLU;
// only the final row-major FP16 [M,768] output is materialized.
extern "C" cudaError_t katago_renju15_int8_dual_ffn_sm120_launch(
  void* opaque,
  int actualTokenRows,
  const int8_t* activation,
  half* output,
  cudaStream_t stream);

extern "C" const char* katago_renju15_int8_qk_sm120_marker();
extern "C" const char* katago_renju15_int8_dual_ffn_sm120_marker();

#endif

#pragma once
#include <cuda_runtime_api.h>
extern "C" cudaError_t b11_fa4_mask_m128_n96_s1_prepare();
extern "C" cudaError_t b11_fa4_mask_m128_n96_s1_launch(
  void* q,void* k,void* v,void* out,void* mask,int batch,float scale,bool packedQKV,cudaStream_t stream);

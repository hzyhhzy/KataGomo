#ifndef NEURALNET_CUDAB11_H_
#define NEURALNET_CUDAB11_H_
#include "cudab11profile.h"
#include "cudaincludes.h"
#include "cudab11affine.h"

void launchB11RmsNorm(half* trunk,const half* delta,half* output,const half* gamma,
  const half* mask,int rows,float epsilon,cudaStream_t stream);
void launchB11Rope(half* q,half* k,const float* freqs,int batch,int stride,cudaStream_t stream);
#ifdef USE_B11_FA4
#include "b11_fa4_exact_m128_n96_s1_api.h"
#endif
#ifdef USE_B11_MASKED_FA4
#include "b11_fa4_mask_m128_n96_s1_api.h"
#endif
#endif

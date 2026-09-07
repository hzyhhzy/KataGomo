// Optional schema-2 adapter. No device queries, setup, locks or allocation at launch.
#include "b11_fa4_exact_m128_n96_s1_api.h"
#include "b11_fa4_mask_m128_n96_s1_api.h"
#include "static_fa4_profile.h"
#include "b11_fa4_static_b13_exact_m128_n96_s1_api.h"
#include "b11_fa4_static_b13_mask_m128_n96_s1_api.h"
#include "b11_fa4_static_b16_exact_m128_n96_s1_api.h"
#include "b11_fa4_static_b16_mask_m128_n96_s1_api.h"

extern "C" cudaError_t b11_fa4_exact_m128_n96_s1_prepare() {
  const auto status=b11_fa4_static_b13_exact_m128_n96_s1_prepare();
  return status==cudaSuccess?b11_fa4_static_b16_exact_m128_n96_s1_prepare():status;
}
extern "C" cudaError_t b11_fa4_mask_m128_n96_s1_prepare() {
  const auto status=b11_fa4_static_b13_mask_m128_n96_s1_prepare();
  return status==cudaSuccess?b11_fa4_static_b16_mask_m128_n96_s1_prepare():status;
}
extern "C" cudaError_t b11_fa4_exact_m128_n96_s1_launch(
  void* q,void* k,void* v,void* out,void* mask,int batch,float scale,bool packed,cudaStream_t stream) {
  if(!b11aot::supportsStaticFa4(batch,packed)) return cudaErrorInvalidValue;
  if(batch==13) return b11_fa4_static_b13_exact_m128_n96_s1_launch(q,k,v,out,mask,batch,scale,packed,stream);
  return b11_fa4_static_b16_exact_m128_n96_s1_launch(q,k,v,out,mask,batch,scale,packed,stream);
}
extern "C" cudaError_t b11_fa4_mask_m128_n96_s1_launch(
  void* q,void* k,void* v,void* out,void* mask,int batch,float scale,bool packed,cudaStream_t stream) {
  if(!b11aot::supportsStaticFa4(batch,packed)) return cudaErrorInvalidValue;
  if(batch==13) return b11_fa4_static_b13_mask_m128_n96_s1_launch(q,k,v,out,mask,batch,scale,packed,stream);
  return b11_fa4_static_b16_mask_m128_n96_s1_launch(q,k,v,out,mask,batch,scale,packed,stream);
}

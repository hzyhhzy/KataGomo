#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Reuse the same RoPE phase across the batch. Rotation arithmetic stays FP32,
// as in the generic kernel; only the load/store is vectorized into half2.
__global__ void b11BatchSharedRope(half2* q,half2* k,const float* freqs,int batch,int stride2) {
  const int xy=blockIdx.x,hp=threadIdx.x;
  const float angle=float(xy%19)*freqs[2*hp]+float(xy/19)*freqs[2*hp+1];
  float s,c;__sincosf(angle,&s,&c);
  for(int n=0;n<batch;n++) {
    const size_t offset=(size_t)(n*361+xy)*stride2+hp;
    float2 a=__half22float2(q[offset]);
    float2 b=__half22float2(k[offset]);
    q[offset]=__floats2half2_rn(a.x*c-a.y*s,a.x*s+a.y*c);
    k[offset]=__floats2half2_rn(b.x*c-b.y*s,b.x*s+b.y*c);
  }
}
void launchB11Rope(half* q,half* k,const float* freqs,int batch,int stride,cudaStream_t stream) {
  b11BatchSharedRope<<<361,192,0,stream>>>((half2*)q,(half2*)k,freqs,batch,stride/2);
}

// Four independent 384-channel rows per CTA. Unlike the generic norm this
// needs no shared-memory reduction or block-wide barriers.
template<bool Residual, bool Masked>
__global__ void b11RmsNorm384(half* trunk,const half* delta,half* output,
  const half* gamma,const half* mask,int rows,float epsilon) {
  const int row=blockIdx.x*4+(threadIdx.x>>5),lane=threadIdx.x&31;
  if(row>=rows)return;
  const float m=Masked?__half2float(mask[row]):1.0f;
  half* x=trunk+(size_t)row*384;
  const half* d=Residual?delta+(size_t)row*384:nullptr;
  half* y=output+(size_t)row*384;
  uint4 xm=reinterpret_cast<const uint4*>(x)[lane];
  uint2 xt=reinterpret_cast<const uint2*>(x+256)[lane];
  uint4 gm=reinterpret_cast<const uint4*>(gamma)[lane];
  uint2 gt=reinterpret_cast<const uint2*>(gamma+256)[lane];
  uint4 dm;uint2 dt;
  if(Residual){dm=reinterpret_cast<const uint4*>(d)[lane];dt=reinterpret_cast<const uint2*>(d+256)[lane];}
  float values[12],sum=0.0f;
  #pragma unroll
  for(int i=0;i<12;i++){
    half& v=i<8?reinterpret_cast<half*>(&xm)[i]:reinterpret_cast<half*>(&xt)[i-8];
    if(Residual){
      const half dv=i<8?reinterpret_cast<const half*>(&dm)[i]:reinterpret_cast<const half*>(&dt)[i-8];
      v=__float2half(__half2float(v)+__half2float(dv)*m);
    }
    values[i]=__half2float(v)*m;
  }
  // Pair the squares before accumulation, as in the generic half2 path.
  #pragma unroll
  for(int i=0;i<6;i++)sum+=values[2*i]*values[2*i]+values[2*i+1]*values[2*i+1];
  if(Residual){reinterpret_cast<uint4*>(x)[lane]=xm;reinterpret_cast<uint2*>(x+256)[lane]=xt;}
  #pragma unroll
  for(int offset=16;offset>0;offset>>=1)sum+=__shfl_xor_sync(0xffffffff,sum,offset);
  const float inv=rsqrtf(sum/384.0f+epsilon);
  uint4 om;uint2 ot;
  #pragma unroll
  for(int i=0;i<12;i++){
    const half g=i<8?reinterpret_cast<const half*>(&gm)[i]:reinterpret_cast<const half*>(&gt)[i-8];
    half& o=i<8?reinterpret_cast<half*>(&om)[i]:reinterpret_cast<half*>(&ot)[i-8];
    o=__float2half(values[i]*inv*__half2float(g)*m);
  }
  reinterpret_cast<uint4*>(y)[lane]=om;reinterpret_cast<uint2*>(y+256)[lane]=ot;
}

void launchB11RmsNorm(half* trunk,const half* delta,half* output,const half* gamma,
  const half* mask,int rows,float epsilon,cudaStream_t stream){
  const int blocks=(rows+3)/4;
  if(delta){
    if(mask)b11RmsNorm384<true,true><<<blocks,128,0,stream>>>(trunk,delta,output,gamma,mask,rows,epsilon);
    else b11RmsNorm384<true,false><<<blocks,128,0,stream>>>(trunk,delta,output,gamma,mask,rows,epsilon);
  }else{
    if(mask)b11RmsNorm384<false,true><<<blocks,128,0,stream>>>(trunk,delta,output,gamma,mask,rows,epsilon);
    else b11RmsNorm384<false,false><<<blocks,128,0,stream>>>(trunk,delta,output,gamma,mask,rows,epsilon);
  }
}

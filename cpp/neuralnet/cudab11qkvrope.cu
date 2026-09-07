/*
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// B11 QKV projection with FP32 learnable RoPE in the output store. The base
// CUTLASS GEMM/MMA and FP16 epilogue are unchanged; only the final store iterator
// rotates adjacent Q/K pairs. Its fragment coordinates follow the base iterator.
#include "cudab11qkvrope.h"
#include <cstdint>
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"

namespace {
using Element=cutlass::half_t;
using OutputOp=cutlass::epilogue::thread::LinearCombination<Element,8,Element,Element>;
using DeviceGemm=cutlass::gemm::device::Gemm<
  Element,cutlass::layout::RowMajor,Element,cutlass::layout::RowMajor,
  Element,cutlass::layout::RowMajor,Element,
  cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<128,128,32>,
  cutlass::gemm::GemmShape<64,64,32>,
  cutlass::gemm::GemmShape<16,8,16>,OutputOp,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,3,8,8,false>;
using StockKernel=DeviceGemm::GemmKernel;
using StockEpilogue=StockKernel::Epilogue;
using StockIterator=StockEpilogue::OutputTileIterator;
using Swizzle=DeviceGemm::ThreadblockSwizzle;

class RopeStoreIterator : public StockIterator {
  using Base=StockIterator;
  const float* freqs_;
public:
  using ThreadMap=typename Base::ThreadMap;
  using Fragment=typename Base::Fragment;
  using Layout=typename Base::Layout;
  using TensorCoord=typename Base::TensorCoord;
  static_assert(Base::kElementsPerAccess==8,"RoPE requires eight adjacent halfs per output access");
  static_assert(ThreadMap::Delta::kColumn % 8 == 0,"RoPE pairs stay inside each vector store");

  using ParentParams=typename StockIterator::Params;
  struct Params : ParentParams {
    const float* freqs=nullptr;
    CUTLASS_HOST_DEVICE Params():ParentParams() {}
    CUTLASS_HOST_DEVICE Params(Layout const& layout):ParentParams(layout) {}
  };

  CUTLASS_DEVICE
  RopeStoreIterator(Params const& params,Element* pointer,TensorCoord extent,
    int thread,TensorCoord offset=TensorCoord(),int const* indices=nullptr)
    :Base(params,pointer,extent,thread,offset,indices),freqs_(params.freqs) {}

  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const& fragment,int64_t byteOffset) const {
    // Normal GEMM calls store(), i.e. byteOffset=0. Do not silently pretend that
    // an arbitrary pointer offset retains the same logical token coordinates.
    assert(byteOffset==0);
    Fragment rotated=fragment;
    CUTLASS_PRAGMA_UNROLL
    for(int cluster=0;cluster<ThreadMap::Iterations::kCluster;cluster++) {
      CUTLASS_PRAGMA_UNROLL
      for(int group=0;group<ThreadMap::Iterations::kGroup;group++) {
        CUTLASS_PRAGMA_UNROLL
        for(int row=0;row<ThreadMap::Iterations::kRow;row++) {
          const int fragRow=row+ThreadMap::Iterations::kRow*(group+ThreadMap::Iterations::kGroup*cluster);
          const int token=this->thread_start_row()+row*ThreadMap::Delta::kRow+
            group*ThreadMap::Delta::kGroup+cluster*ThreadMap::Delta::kCluster;
          CUTLASS_PRAGMA_UNROLL
          for(int col=0;col<ThreadMap::Iterations::kColumn;col++) {
            const int channel=this->thread_start_column()+col*ThreadMap::Delta::kColumn;
            // Q/K end on both 128-column CTA and 8-element access boundaries.
            // Tail-M lanes skip frequency reads; base iterator owns store predicates.
            if(token<this->extent_row() && channel<768) {
              const int position=token%361;
              const int firstPair=(channel>=384?channel-384:channel)/2;
              const int first=(fragRow*ThreadMap::Iterations::kColumn+col)*8;
              CUTLASS_PRAGMA_UNROLL
              for(int pair=0;pair<4;pair++) {
                float c,s;
                const int hp=firstPair+pair;
                const float angle=float(position%19)*freqs_[2*hp]+
                  float(position/19)*freqs_[2*hp+1];
                __sincosf(angle,&s,&c);
                const float x=float(fragment[first+2*pair]);
                const float y=float(fragment[first+2*pair+1]);
                // Original FP32 expressions, original nearest FP16 storage.
                // GEMM's half output is already rounded before this wrapper.
                rotated[first+2*pair]=Element(__float2half_rn(x*c-y*s));
                rotated[first+2*pair+1]=Element(__float2half_rn(x*s+y*c));
              }
            }
          }
        }
      }
    }
    // Inherits the original vector-store instruction, predicates, advancement,
    // alignment, and token-major output layout. No extra shared memory/barrier.
    Base::store_with_byte_offset(rotated,byteOffset);
  }
  CUTLASS_DEVICE void store(Fragment const& fragment) const {store_with_byte_offset(fragment,0);}
};

using RopeEpilogue=cutlass::epilogue::threadblock::Epilogue<
  typename StockEpilogue::Shape,typename StockEpilogue::WarpMmaOperator,
  StockEpilogue::kPartitionsK,RopeStoreIterator,
  typename StockEpilogue::AccumulatorFragmentIterator,
  typename StockEpilogue::WarpTileIterator,typename StockEpilogue::SharedLoadIterator,
  typename StockEpilogue::OutputOp,typename StockEpilogue::Padding,
  StockEpilogue::kFragmentsPerIteration>;

using RopeKernel=cutlass::gemm::kernel::Gemm<
  typename StockKernel::Mma,RopeEpilogue,Swizzle,false>;

RopeKernel::Params makeParams(const half* A,const half* W,half* output,int M) {
  const cutlass::gemm::GemmCoord problem(M,1152,384);
  const auto tiled=Swizzle().get_tiled_shape(problem,{128,128,32},1);
  auto* a=const_cast<Element*>(reinterpret_cast<const Element*>(A));
  auto* b=const_cast<Element*>(reinterpret_cast<const Element*>(W));
  auto* d=reinterpret_cast<Element*>(output);
  return RopeKernel::Params(problem,tiled,{a,384},{b,1152},{d,1152},{d,1152},
    OutputOp::Params(Element(1.0f),Element(0.0f)),nullptr,nullptr,nullptr,nullptr);
}
} // namespace

namespace CudaB11QkvRope {

cudaError_t prepareForCurrentDevice() {
  constexpr int sharedBytes=sizeof(RopeKernel::SharedStorage);
  if(sharedBytes<=48*1024)return cudaSuccess;
  return cudaFuncSetAttribute(cutlass::Kernel<RopeKernel>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,sharedBytes);
}

cudaError_t run(const half* A,const half* W,half* output,int M,
  const float* freqs,cudaStream_t stream) {
  // Device/model/layout dispatch and buffer ownership belong to the caller.
  // Retain the validated fixed-19x19 shape range and cheap pointer checks.
  if(M<361||M>96*361||M%361!=0||!A||!W||!output||!freqs||A==output||W==output)
    return cudaErrorInvalidValue;
  if((reinterpret_cast<std::uintptr_t>(A)|reinterpret_cast<std::uintptr_t>(W)|
      reinterpret_cast<std::uintptr_t>(output))&15)return cudaErrorInvalidValue;
  auto params=makeParams(A,W,output,M);
  params.params_D.freqs=freqs;
  const dim3 grid=Swizzle().get_grid_shape(params.grid_tiled_shape);
  cutlass::Kernel<RopeKernel><<<grid,RopeKernel::kThreadCount,
    sizeof(RopeKernel::SharedStorage),stream>>>(params);
  return cudaPeekAtLastError();
}

} // namespace CudaB11QkvRope


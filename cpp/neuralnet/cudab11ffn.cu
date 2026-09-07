/*
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

// The local output functor reproduces the TANH branch of CUTLASS's
// Sigmoid<Array<half_t,8>> and LeftSiLUAndMul, without changing any vendor type
// definition or an otherwise identical template's meaning across translation
// units. Original DualMma/DualEpilogue/tile/layout/wrapper are retained.
#ifdef CUTLASS_USE_TANH_FOR_SIGMOID
#error "Compile this B11 specialization without CUTLASS_USE_TANH_FOR_SIGMOID"
#endif
#include "cudab11ffn.h"
#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include "cutlass/cutlass.h"
#include "cutlass/constants.h"
#include "cutlass/fast_math.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "device/dual_gemm.h"

namespace {
using ElementT=cutlass::half_t;

class B11TanhSwiGLU {
public:
  using ElementOutput=ElementT;
  using ElementAccumulator=ElementT;
  using ElementCompute=ElementT;
  static constexpr int kCount=8;
  static constexpr auto kRound=cutlass::FloatRoundStyle::round_to_nearest;
  using FragmentOutput=cutlass::Array<ElementOutput,kCount>;
  using FragmentAccumulator=cutlass::Array<ElementAccumulator,kCount>;
  using ComputeFragment=cutlass::Array<ElementCompute,kCount>;
  struct Params {};
  CUTLASS_HOST_DEVICE B11TanhSwiGLU(Params const&) {}
  CUTLASS_HOST_DEVICE bool is_source_needed() const {return true;}
  CUTLASS_HOST_DEVICE void set_k_partition(int,int) {assert(false);}

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const& lhs,FragmentAccumulator const& rhs) const {
    cutlass::NumericArrayConverter<ElementCompute,ElementAccumulator,kCount,kRound> toCompute;
    cutlass::NumericArrayConverter<ElementOutput,ElementCompute,kCount,kRound> toOutput;
    const ComputeFragment up=toCompute(lhs),gate=toCompute(rhs);
    cutlass::multiplies<ComputeFragment> multiply;
    cutlass::multiply_add<ComputeFragment> fma;
    cutlass::fast_tanh_op<ComputeFragment> tanh;
    const ElementCompute half=cutlass::constants::half<ElementCompute>();
    // Exact ARRAY math order of the vendored macro path:
    // hmul2(up,.5) -> tanh.approx.f16x2 -> hfma2(tanh,.5,.5)
    // -> hmul2(up,sigmoid) -> hmul2(silu,gate). Do not collapse the half
    // boundaries or replace the FMA with separate multiply and addition.
    const ComputeFragment sigmoid=fma(tanh(multiply(up,half)),half,half);
    const ComputeFragment silu=multiply(up,sigmoid);
    return toOutput(multiply(silu,gate));
  }

  // Concept-completeness only: the selected DualEpilogue uses the array path.
  // Keep the scalar macro's own expression, which is not an explicit array FMA.
  CUTLASS_HOST_DEVICE
  ElementOutput operator()(ElementAccumulator const& lhs,ElementAccumulator const& rhs) const {
    const ElementCompute up(lhs),gate(rhs),half(0.5f);
    const ElementCompute sigmoid=cutlass::fast_tanh(up*half)*half+half;
    const ElementCompute silu=up*sigmoid;
    return ElementOutput(silu*gate);
  }
};

using Output01=cutlass::epilogue::thread::LinearCombination<
  ElementT,8,ElementT,ElementT,cutlass::epilogue::thread::ScaleType::Nothing>;
using DualGemm=cutlass::gemm::device::DualGemm<
  ElementT,cutlass::layout::RowMajor,
  ElementT,cutlass::layout::ColumnMajor,cutlass::layout::ColumnMajor,
  ElementT,cutlass::layout::RowMajor,ElementT,
  cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<128,64,32>,cutlass::gemm::GemmShape<64,32,32>,
  cutlass::gemm::GemmShape<16,8,16>,Output01,Output01,B11TanhSwiGLU,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,3,false,false,false>;

DualGemm::Arguments makeArgs(const half* A,const half* w1,const half* wGate,
  half* output,int M,int N,int K) {
  return DualGemm::Arguments(cutlass::gemm::DualGemmMode::kGemm,{M,N,K},
    {reinterpret_cast<const ElementT*>(A),K},
    {reinterpret_cast<const ElementT*>(w1),K},
    cutlass::TensorRef<const ElementT,cutlass::layout::RowMajor>(),
    cutlass::TensorRef<ElementT,cutlass::layout::RowMajor>(),
    {reinterpret_cast<const ElementT*>(wGate),K},
    cutlass::TensorRef<const ElementT,cutlass::layout::RowMajor>(),
    cutlass::TensorRef<ElementT,cutlass::layout::RowMajor>(),
    {reinterpret_cast<ElementT*>(output),N},
    Output01::Params(),Output01::Params(),B11TanhSwiGLU::Params(),1);
}
} // namespace

namespace CudaB11FFN {
bool supportsShape(int N,int K) {
  if(N!=1152||K!=384)return false;
  const auto* dummy=reinterpret_cast<const half*>(std::uintptr_t(256));
  const auto args=makeArgs(dummy,dummy,dummy,const_cast<half*>(dummy),128,N,K);
  return DualGemm::can_implement(args)==cutlass::Status::kSuccess&&
    DualGemm::get_workspace_size(args)==0;
}
bool supportedOnCurrentDevice() {
  // Setup-only real-kernel probe. Do not replace the original generic FFN probe:
  // these independent template instantiations must each demonstrate execution.
  int device=0;
  cudaDeviceProp prop;
  if(cudaGetDevice(&device)!=cudaSuccess ||
     cudaGetDeviceProperties(&prop,device)!=cudaSuccess ||
     prop.major!=12 || prop.minor!=0 ||
     std::strcmp(prop.name,"NVIDIA GeForce RTX 5090")!=0)
    return false;
  constexpr int M=16,N=64,K=64;
  constexpr size_t numIn=size_t(M)*K+2*size_t(N)*K;
  constexpr size_t numOut=size_t(M)*N;
  half* buffer=nullptr;
  if(cudaMalloc(&buffer,(numIn+numOut)*sizeof(half))!=cudaSuccess)return false;
  half* A=buffer;
  half* w1=A+size_t(M)*K;
  half* wGate=w1+size_t(N)*K;
  half* output=wGate+size_t(N)*K;
  bool ok=cudaMemset(buffer,0,numIn*sizeof(half))==cudaSuccess;
  ok=ok&&cudaMemset(output,0xFF,numOut*sizeof(half))==cudaSuccess;
  if(ok) {
    const auto args=makeArgs(A,w1,wGate,output,M,N,K);
    ok=DualGemm::can_implement(args)==cutlass::Status::kSuccess &&
       DualGemm::get_workspace_size(args)==0;
    if(ok) {
      DualGemm operation;
      ok=operation.initialize(args,nullptr,nullptr)==cutlass::Status::kSuccess;
      ok=ok&&operation.run(nullptr)==cutlass::Status::kSuccess;
    }
  }
  half hostOutput[numOut];
  ok=ok&&cudaMemcpy(hostOutput,output,numOut*sizeof(half),cudaMemcpyDeviceToHost)==cudaSuccess;
  if(ok) {
    for(size_t i=0;i<numOut;i++) {
      if(__half2float(hostOutput[i])!=0.0f) {ok=false;break;}
    }
  }
  // As with the generic probe, a recoverable launch failure must not leak to a
  // later unrelated operation. Unexpected sticky CUDA errors cannot be hidden.
  (void)cudaGetLastError();
  const auto cleanup=cudaFree(buffer);
  return ok&&cleanup==cudaSuccess;
}
void runSwiGLU(const half* A,const half* w1,const half* wGate,half* output,
  int M,int N,int K,cudaStream_t stream) {
  if(!A||!w1||!wGate||!output||(M!=13*361&&M!=16*361)||N!=1152||K!=384||A==output||w1==output||wGate==output||
    ((std::uintptr_t(A)|std::uintptr_t(w1)|std::uintptr_t(wGate)|std::uintptr_t(output))&15))
    throw std::runtime_error("CudaB11FFN: invalid dimensions, alias, or unaligned operand");
  // Deliberately preserve the public production initialize/run wrapper. This
  // specialization changes only the epilogue arithmetic, not Params or launch cache.
  const auto args=makeArgs(A,w1,wGate,output,M,N,K);
  DualGemm operation;
  auto status=operation.initialize(args,nullptr,stream);
  if(status==cutlass::Status::kSuccess)status=operation.run(stream);
  if(status!=cutlass::Status::kSuccess)
    throw std::runtime_error(std::string("CUTLASS local tanh FFN failed (or prior CUDA error pending): ")+
      cutlass::cutlassGetStatusString(status)+", M="+std::to_string(M)+" N="+std::to_string(N)+" K="+std::to_string(K));
}
} // namespace CudaB11FFN

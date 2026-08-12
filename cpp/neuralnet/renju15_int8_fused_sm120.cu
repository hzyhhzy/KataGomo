#include "renju15_int8_fused_sm120.h"

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "device/dual_gemm.h"
#include "thread/left_silu_and_mul.h"

#include <cmath>
#include <cstdint>
#include <new>

namespace {

constexpr int InputChannels = 256;
constexpr int QChannels = 256;
constexpr int KChannels = 256;
constexpr int QkChannels = QChannels + KChannels;
constexpr int FfnChannels = 768;
constexpr int MaxTokenRows = 1 << 20;
constexpr float ActivationScale = 4.0f / 127.0f;

using Int8 = int8_t;
using Accum = int32_t;
using Output = cutlass::half_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;
using DequantToHalf = cutlass::epilogue::thread::LinearCombination<
  Output,8,Accum,float,cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;

using QkGemm = cutlass::gemm::device::Gemm<
  Int8,LayoutA,Int8,LayoutB,Output,LayoutOutput,
  Accum,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<128,128,64>,
  cutlass::gemm::GemmShape<64,64,64>,
  cutlass::gemm::GemmShape<16,8,32>,
  DequantToHalf,Swizzle,3,16,16,false,
  cutlass::arch::OpMultiplyAddSaturate>;

using SwiGLU = cutlass::epilogue::thread::LeftSiLUAndMul<
  Output,8,Output,float>;
using DualFfnGemm = cutlass::gemm::device::DualGemm<
  Int8,LayoutA,
  Int8,LayoutB,LayoutB,
  Output,LayoutOutput,
  Accum,
  cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<128,64,64>,
  cutlass::gemm::GemmShape<64,32,64>,
  cutlass::gemm::GemmShape<16,8,32>,
  DequantToHalf,DequantToHalf,SwiGLU,
  Swizzle,3,false,false,false,16,16,
  cutlass::arch::OpMultiplyAddSaturate>;

static_assert(sizeof(typename QkGemm::GemmKernel::SharedStorage) <= 101376,
              "INT8 QK exceeds RTX 5090 opt-in shared memory");
static_assert(sizeof(typename DualFfnGemm::DualGemmKernel::SharedStorage) <= 101376,
              "INT8 dual FFN exceeds RTX 5090 opt-in shared memory");

bool aligned16(const void* pointer) {
  return (reinterpret_cast<std::uintptr_t>(pointer) & 15U) == 0;
}

bool isSm120Compatible() {
  int device = -1;
  cudaDeviceProp prop{};
  if(cudaGetDevice(&device) != cudaSuccess ||
     cudaGetDeviceProperties(&prop,device) != cudaSuccess)
    return false;
  return prop.major == 12 && prop.minor == 0 &&
    prop.sharedMemPerBlockOptin >= 101376;
}

template<typename Kernel>
cudaError_t setDynamicSharedAttribute(bool& alreadySet) {
  if(alreadySet)
    return cudaSuccess;
  constexpr int sharedBytes = int(sizeof(typename Kernel::SharedStorage));
  if(sharedBytes >= 48 * 1024) {
    cudaError_t status = cudaFuncSetAttribute(
      cutlass::Kernel<Kernel>,cudaFuncAttributeMaxDynamicSharedMemorySize,
      sharedBytes);
    if(status != cudaSuccess)
      return status;
  }
  alreadySet = true;
  return cudaSuccess;
}

struct QkHandle {
  using Kernel = typename QkGemm::GemmKernel;
  using Params = typename Kernel::Params;

  int maxRows;
  const Int8* packedWeights;
  float alpha;
};

struct DualFfnHandle {
  using Kernel = typename DualFfnGemm::DualGemmKernel;
  using Params = typename Kernel::Params;

  int maxRows;
  const Int8* packedUpWeights;
  const Int8* packedGateWeights;
  float upAlpha;
  float gateAlpha;
};

typename QkGemm::Arguments makeQkBoundaryArguments(
  int rows,
  const Int8* packedWeights,
  float alpha
) {
  using TensorRefA = typename QkGemm::TensorRefA;
  using TensorRefB = typename QkGemm::TensorRefB;
  using TensorRefC = typename QkGemm::TensorRefC;
  using TensorRefD = typename QkGemm::TensorRefD;
  return typename QkGemm::Arguments(
    {rows,QkChannels,InputChannels},
    TensorRefA(packedWeights,LayoutA(InputChannels)),
    TensorRefB(packedWeights,LayoutB(InputChannels)),
    TensorRefC(nullptr,LayoutOutput(QkChannels)),
    TensorRefD(
      reinterpret_cast<Output*>(const_cast<Int8*>(packedWeights)),
      LayoutOutput(QkChannels)),
    typename DequantToHalf::Params(alpha,0.0f));
}

typename DualFfnGemm::Arguments makeDualBoundaryArguments(
  int rows,
  const Int8* packedUpWeights,
  const Int8* packedGateWeights,
  float upAlpha,
  float gateAlpha
) {
  using Layout = cutlass::layout::RowMajor;
  typename DualFfnGemm::TensorRefC nullC;
  typename DualFfnGemm::TensorRefD nullD;
  return typename DualFfnGemm::Arguments{
    cutlass::gemm::DualGemmMode::kGemm,
    {rows,FfnChannels,InputChannels},
    {packedUpWeights,LayoutA(InputChannels)},
    {packedUpWeights,LayoutB(InputChannels)},
    nullC,nullD,
    {packedGateWeights,LayoutB(InputChannels)},
    nullC,nullD,
    {reinterpret_cast<Output*>(const_cast<Int8*>(packedUpWeights)),
     Layout(FfnChannels)},
    {upAlpha,0.0f},{gateAlpha,0.0f},{},1,
  };
}

__global__ void splitQkAndApplyLearnedRopeKernel(
  const half2* __restrict__ qk,
  half2* __restrict__ q,
  half2* __restrict__ k,
  const half2* __restrict__ cosSin,
  int sequenceLength
) {
  constexpr int Heads = 8;
  constexpr int PairsPerHead = 16;
  constexpr int PairsPerProjection = Heads * PairsPerHead;
  constexpr int PairsPerQkRow = 2 * PairsPerProjection;

  const int row = blockIdx.x;
  const int hp = threadIdx.x;
  if(hp >= PairsPerProjection)
    return;
  const int xy = row % sequenceLength;
  // Q and K both have H=8 here, so each head maps to the same KV head and
  // consumes the same [xy,head,pair] learned table entry as the incumbent
  // fused QKV+RoPE output iterator.
  const float2 cs = __half22float2(
    cosSin[(std::size_t)xy * PairsPerProjection + hp]);
  const std::size_t source =
    (std::size_t)row * PairsPerQkRow + hp;
  const std::size_t destination =
    (std::size_t)row * PairsPerProjection + hp;
  const float2 qv = __half22float2(qk[source]);
  const float2 kv = __half22float2(qk[source + PairsPerProjection]);
  q[destination] = __floats2half2_rn(
    qv.x * cs.x - qv.y * cs.y,
    qv.x * cs.y + qv.y * cs.x);
  k[destination] = __floats2half2_rn(
    kv.x * cs.x - kv.y * cs.y,
    kv.x * cs.y + kv.y * cs.x);
}

} // namespace

extern "C" void* katago_renju15_int8_qk_sm120_create(
  int maxTokenRows,
  const int8_t* packedQkWeights,
  float qkWeightScale
) {
  if(maxTokenRows <= 0 || maxTokenRows > MaxTokenRows ||
     packedQkWeights == nullptr || !aligned16(packedQkWeights) ||
     !(qkWeightScale > 0.0f) || !std::isfinite(qkWeightScale) ||
     !isSm120Compatible())
    return nullptr;
  bool attributeSet = false;
  if(setDynamicSharedAttribute<QkHandle::Kernel>(attributeSet) !=
       cudaSuccess)
    return nullptr;
  const float alpha = ActivationScale * qkWeightScale;
  if(QkGemm::can_implement(makeQkBoundaryArguments(
       1,packedQkWeights,alpha)) != cutlass::Status::kSuccess ||
     QkGemm::can_implement(makeQkBoundaryArguments(
       maxTokenRows,packedQkWeights,alpha)) != cutlass::Status::kSuccess)
    return nullptr;
  return new(std::nothrow) QkHandle{
    maxTokenRows,packedQkWeights,alpha};
}

extern "C" void katago_renju15_int8_qk_sm120_destroy(void* opaque) {
  delete static_cast<QkHandle*>(opaque);
}

extern "C" bool katago_renju15_int8_qk_sm120_supports(
  const void* opaque,
  int actualTokenRows,
  int inputChannels,
  int qChannels,
  int kChannels,
  bool usingFp16,
  bool usingNhwc,
  bool exactNoMask
) {
  const QkHandle* handle = static_cast<const QkHandle*>(opaque);
  return handle != nullptr && actualTokenRows > 0 &&
    actualTokenRows <= handle->maxRows && inputChannels == InputChannels &&
    qChannels == QChannels && kChannels == KChannels && usingFp16 &&
    usingNhwc && exactNoMask;
}

extern "C" cudaError_t katago_renju15_int8_qk_sm120_launch(
  void* opaque,
  int actualTokenRows,
  const int8_t* activation,
  half* qkContiguous,
  cudaStream_t stream
) {
  QkHandle* handle = static_cast<QkHandle*>(opaque);
  if(handle == nullptr || actualTokenRows <= 0 ||
     actualTokenRows > handle->maxRows || activation == nullptr ||
     qkContiguous == nullptr || !aligned16(activation) ||
     !aligned16(qkContiguous))
    return cudaErrorInvalidValue;

  using Kernel = QkHandle::Kernel;
  cutlass::gemm::GemmCoord problem(
    actualTokenRows,QkChannels,InputChannels);
  cutlass::gemm::GemmCoord tiled = Swizzle::get_tiled_shape(
    problem,{QkGemm::ThreadblockShape::kM,QkGemm::ThreadblockShape::kN,
             QkGemm::ThreadblockShape::kK},1);
  typename Kernel::Mma::IteratorA::TensorRef a(
    const_cast<Int8*>(activation),LayoutA(InputChannels));
  typename Kernel::Mma::IteratorB::TensorRef b(
    const_cast<Int8*>(handle->packedWeights),LayoutB(InputChannels));
  typename Kernel::Epilogue::OutputTileIterator::TensorRef c(
    nullptr,LayoutOutput(QkChannels));
  typename Kernel::Epilogue::OutputTileIterator::TensorRef d(
    reinterpret_cast<Output*>(qkContiguous),LayoutOutput(QkChannels));
  typename QkHandle::Params params(
    problem,tiled,a,b,c,d,
    typename DequantToHalf::Params(handle->alpha,0.0f),nullptr);
  const dim3 grid = Swizzle::get_grid_shape(tiled);
  constexpr int threads = Kernel::kThreadCount;
  constexpr int sharedBytes = int(sizeof(typename Kernel::SharedStorage));
  cutlass::Kernel<Kernel><<<grid,dim3(threads,1,1),sharedBytes,stream>>>(params);
  return cudaPeekAtLastError();
}

extern "C" cudaError_t katago_renju15_int8_qk_split_rope_sm120_launch(
  const half* qkContiguous,
  half* qPlanar,
  half* kPlanar,
  const half2* ropeCosSin,
  int batchSize,
  int sequenceLength,
  cudaStream_t stream
) {
  if(qkContiguous == nullptr || qPlanar == nullptr || kPlanar == nullptr ||
     ropeCosSin == nullptr || batchSize <= 0 || sequenceLength != 225 ||
     batchSize > MaxTokenRows / sequenceLength ||
     !aligned16(qkContiguous) || !aligned16(qPlanar) ||
     !aligned16(kPlanar) || !aligned16(ropeCosSin))
    return cudaErrorInvalidValue;
  const int rows = batchSize * sequenceLength;
  splitQkAndApplyLearnedRopeKernel<<<rows,128,0,stream>>>(
    reinterpret_cast<const half2*>(qkContiguous),
    reinterpret_cast<half2*>(qPlanar),reinterpret_cast<half2*>(kPlanar),
    ropeCosSin,sequenceLength);
  return cudaPeekAtLastError();
}

extern "C" void* katago_renju15_int8_dual_ffn_sm120_create(
  int maxTokenRows,
  const int8_t* packedUpWeights,
  const int8_t* packedGateWeights,
  float upWeightScale,
  float gateWeightScale
) {
  if(maxTokenRows <= 0 || maxTokenRows > MaxTokenRows ||
     packedUpWeights == nullptr || packedGateWeights == nullptr ||
     !aligned16(packedUpWeights) || !aligned16(packedGateWeights) ||
     !(upWeightScale > 0.0f) || !(gateWeightScale > 0.0f) ||
     !std::isfinite(upWeightScale) || !std::isfinite(gateWeightScale) ||
     !isSm120Compatible())
    return nullptr;
  bool attributeSet = false;
  if(setDynamicSharedAttribute<DualFfnHandle::Kernel>(
       attributeSet) != cudaSuccess)
    return nullptr;
  const float upAlpha = ActivationScale * upWeightScale;
  const float gateAlpha = ActivationScale * gateWeightScale;
  if(DualFfnGemm::can_implement(makeDualBoundaryArguments(
       1,packedUpWeights,packedGateWeights,upAlpha,gateAlpha)) !=
       cutlass::Status::kSuccess ||
     DualFfnGemm::can_implement(makeDualBoundaryArguments(
       maxTokenRows,packedUpWeights,packedGateWeights,upAlpha,gateAlpha)) !=
       cutlass::Status::kSuccess)
    return nullptr;
  return new(std::nothrow) DualFfnHandle{
    maxTokenRows,packedUpWeights,packedGateWeights,
    upAlpha,gateAlpha};
}

extern "C" void katago_renju15_int8_dual_ffn_sm120_destroy(void* opaque) {
  delete static_cast<DualFfnHandle*>(opaque);
}

extern "C" bool katago_renju15_int8_dual_ffn_sm120_supports(
  const void* opaque,
  int actualTokenRows,
  int inputChannels,
  int ffnChannels,
  bool usingFp16,
  bool usingNhwc,
  bool exactNoMask
) {
  const DualFfnHandle* handle = static_cast<const DualFfnHandle*>(opaque);
  return handle != nullptr && actualTokenRows > 0 &&
    actualTokenRows <= handle->maxRows && inputChannels == InputChannels &&
    ffnChannels == FfnChannels && usingFp16 && usingNhwc && exactNoMask;
}

extern "C" cudaError_t katago_renju15_int8_dual_ffn_sm120_launch(
  void* opaque,
  int actualTokenRows,
  const int8_t* activation,
  half* output,
  cudaStream_t stream
) {
  DualFfnHandle* handle = static_cast<DualFfnHandle*>(opaque);
  if(handle == nullptr || actualTokenRows <= 0 ||
     actualTokenRows > handle->maxRows || activation == nullptr ||
     output == nullptr || !aligned16(activation) || !aligned16(output))
    return cudaErrorInvalidValue;

  using Kernel = DualFfnHandle::Kernel;
  cutlass::gemm::GemmCoord problem(
    actualTokenRows,FfnChannels,InputChannels);
  cutlass::gemm::GemmCoord tiled = Swizzle::get_tiled_shape(
    problem,{DualFfnGemm::ThreadblockShape::kM,
             DualFfnGemm::ThreadblockShape::kN,
             DualFfnGemm::ThreadblockShape::kK},1);
  typename Kernel::DualMma::IteratorA::TensorRef a(
    const_cast<Int8*>(activation),LayoutA(InputChannels));
  typename Kernel::DualMma::IteratorB0::TensorRef b0(
    const_cast<Int8*>(handle->packedUpWeights),LayoutB(InputChannels));
  typename Kernel::DualMma::IteratorB1::TensorRef b1(
    const_cast<Int8*>(handle->packedGateWeights),LayoutB(InputChannels));
  typename Kernel::Epilogue0::OutputTileIterator::TensorRef nullTensor(
    nullptr,LayoutOutput(FfnChannels));
  typename Kernel::Epilogue1::OutputTileIterator::TensorRef d2(
    reinterpret_cast<Output*>(output),LayoutOutput(FfnChannels));
  typename DualFfnHandle::Params params(
    cutlass::gemm::DualGemmMode::kGemm,problem,tiled,
    a,b0,nullTensor,nullTensor,b1,nullTensor,nullTensor,d2,
    typename DequantToHalf::Params(handle->upAlpha,0.0f),
    typename DequantToHalf::Params(handle->gateAlpha,0.0f),
    typename SwiGLU::Params(),nullptr);
  const dim3 grid = Swizzle::get_grid_shape(tiled);
  constexpr int threads = Kernel::kThreadCount;
  constexpr int sharedBytes = int(sizeof(typename Kernel::SharedStorage));
  cutlass::Kernel<Kernel><<<grid,dim3(threads,1,1),sharedBytes,stream>>>(params);
  return cudaPeekAtLastError();
}

extern "C" const char* katago_renju15_int8_qk_sm120_marker() {
  return "int8-pt-clip4-qk-m128n128k64s3-split-rope";
}

extern "C" const char* katago_renju15_int8_dual_ffn_sm120_marker() {
  return "int8-pt-clip4-dual-ffn-m128n64k64s3";
}

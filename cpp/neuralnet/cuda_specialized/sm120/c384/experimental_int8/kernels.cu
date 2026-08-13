#include "kernels.h"

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "device/dual_gemm.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <new>

namespace C384Int8Experiment {
namespace {

constexpr int kMaxTokenRows = 1 << 20;
constexpr int kThreadsPerRmsBlock = 128;
constexpr int kThreadsPerQuantBlock = 256;

using Int8 = int8_t;
using Accum = int32_t;
using Output = cutlass::half_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;
using DequantToHalf = cutlass::epilogue::thread::LinearCombination<
  Output,8,Accum,float,cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;
using DequantResidualToHalf = cutlass::epilogue::thread::LinearCombination<
  Output,8,Accum,float,cutlass::epilogue::thread::ScaleType::Default>;

template<int Stages, int Swizzle>
using ProjectionGemm = cutlass::gemm::device::Gemm<
  Int8,LayoutA,Int8,LayoutB,Output,LayoutOutput,
  Accum,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<128,128,64>,
  cutlass::gemm::GemmShape<64,64,64>,
  cutlass::gemm::GemmShape<16,8,32>,
  DequantToHalf,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swizzle>,
  Stages,16,16,false,cutlass::arch::OpMultiplyAddSaturate>;

template <
  typename ElementOutput_, int Count,
  typename ElementAccumulator_ = ElementOutput_,
  typename ElementCompute_ = ElementOutput_,
  cutlass::FloatRoundStyle Round = cutlass::FloatRoundStyle::round_to_nearest
>
class LeftClip7SiLUAndMul {
public:
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  static int const kCount = Count;
  using FragmentOutput = cutlass::Array<ElementOutput,kCount>;
  using FragmentAccumulator = cutlass::Array<ElementAccumulator,kCount>;
  using ComputeFragment = cutlass::Array<ElementCompute,kCount>;
  struct Params {};

  CUTLASS_HOST_DEVICE
  LeftClip7SiLUAndMul(Params const&) {}

  CUTLASS_HOST_DEVICE bool is_source_needed() const { return true; }
  CUTLASS_HOST_DEVICE void set_k_partition(int, int) { assert(false); }

  CUTLASS_HOST_DEVICE
  static ElementCompute clipped(ElementCompute value) {
    const ElementCompute limit = ElementCompute(7.0f);
    return value < -limit ? -limit : (value > limit ? limit : value);
  }

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const& lhs,
    FragmentAccumulator const& rhs
  ) const {
    cutlass::NumericArrayConverter<
      ElementCompute,ElementAccumulator,kCount,Round> toCompute;
    cutlass::NumericArrayConverter<
      ElementOutput,ElementCompute,kCount,Round> toOutput;
    ComputeFragment left = toCompute(lhs);
    ComputeFragment right = toCompute(rhs);
    cutlass::epilogue::thread::SiLu<ComputeFragment> silu;
    left = silu(left);
    CUTLASS_PRAGMA_UNROLL
    for(int i = 0; i < kCount; i++)
      left[i] = clipped(left[i]) * clipped(right[i]);
    return toOutput(left);
  }

  CUTLASS_HOST_DEVICE
  ElementOutput operator()(
    ElementAccumulator const& lhs,
    ElementAccumulator const& rhs
  ) const {
    const ElementCompute left(lhs);
    const ElementCompute right(rhs);
    cutlass::epilogue::thread::SiLu<ElementCompute> silu;
    return ElementOutput(clipped(silu(left)) * clipped(right));
  }
};

using Clip7SwiGLU = LeftClip7SiLUAndMul<Output,8,Output,float>;

// The CUTLASS dual-GEMM example uses one element type for its two internal
// epilogue fragments and final D2 fragment. For the direct INT8 form, keep all
// three fragments INT8: epilogue 0/1 dequantize their INT32 accumulators in
// FP32, apply the clip7 factor transforms, and quantize those factors in
// registers. D0/D1 are never stored. The final functor multiplies the two
// register factors and emits the scale-49/127 product directly to global INT8.
template<bool ApplySiLU, int Count>
class Clip7FactorToInt8 {
public:
  using ElementOutput = Int8;
  using ElementSource = Int8;
  using ElementAccumulator = Accum;
  using ElementCompute = float;
  static int const kCount = Count;
  using FragmentOutput = cutlass::Array<ElementOutput,kCount>;
  using FragmentSource = cutlass::Array<ElementSource,kCount>;
  using FragmentAccumulator = cutlass::Array<ElementAccumulator,kCount>;
  using FragmentCompute = cutlass::Array<ElementCompute,kCount>;

  struct Params {
    float alpha;
    CUTLASS_HOST_DEVICE explicit Params(float value = 1.0f) : alpha(value) {}
  };

private:
  float alpha_;

public:
  CUTLASS_HOST_DEVICE explicit Clip7FactorToInt8(Params const& params) :
    alpha_(params.alpha) {}

  CUTLASS_HOST_DEVICE bool is_source_needed() const { return false; }
  CUTLASS_HOST_DEVICE void set_k_partition(int, int) { assert(false); }

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const& accumulator,
    FragmentSource const&
  ) const {
    cutlass::NumericArrayConverter<
      ElementCompute,ElementAccumulator,kCount,
      cutlass::FloatRoundStyle::round_to_nearest> toCompute;
    cutlass::NumericArrayConverter<
      ElementOutput,ElementCompute,kCount,
      cutlass::FloatRoundStyle::round_to_nearest> toOutput;
    FragmentCompute values = toCompute(accumulator);
    CUTLASS_PRAGMA_UNROLL
    for(int i = 0; i < kCount; i++)
      values[i] *= alpha_;
    if(ApplySiLU) {
      cutlass::epilogue::thread::SiLu<FragmentCompute> silu;
      values = silu(values);
    }
    CUTLASS_PRAGMA_UNROLL
    for(int i = 0; i < kCount; i++) {
      const float clipped = values[i] < -7.0f ? -7.0f :
        (values[i] > 7.0f ? 7.0f : values[i]);
      // The explicit clamp before the CUTLASS RNE converter excludes -128.
      values[i] = clipped * (127.0f / 7.0f);
    }
    return toOutput(values);
  }

  CUTLASS_HOST_DEVICE
  ElementOutput operator()(
    ElementAccumulator const& accumulator,
    ElementSource const&
  ) const {
    float value = float(accumulator) * alpha_;
    if(ApplySiLU) {
      cutlass::epilogue::thread::SiLu<float> silu;
      value = silu(value);
    }
    value = value < -7.0f ? -7.0f : (value > 7.0f ? 7.0f : value);
    cutlass::NumericConverter<
      ElementOutput,float,cutlass::FloatRoundStyle::round_to_nearest> convert;
    return convert(value * (127.0f / 7.0f));
  }
};

template<int Count>
class QuantizedClip7FactorMul {
public:
  using ElementOutput = Int8;
  using ElementAccumulator = Int8;
  using ElementCompute = int;
  static int const kCount = Count;
  using FragmentOutput = cutlass::Array<ElementOutput,kCount>;
  using FragmentAccumulator = cutlass::Array<ElementAccumulator,kCount>;
  struct Params {};

  CUTLASS_HOST_DEVICE explicit QuantizedClip7FactorMul(Params const&) {}
  CUTLASS_HOST_DEVICE bool is_source_needed() const { return true; }
  CUTLASS_HOST_DEVICE void set_k_partition(int, int) { assert(false); }

  CUTLASS_HOST_DEVICE
  static int divide127Rne(int numerator) {
    const bool negative = numerator < 0;
    const int magnitude = negative ? -numerator : numerator;
    int quotient = magnitude / 127;
    const int remainder = magnitude - quotient * 127;
    const int twiceRemainder = 2 * remainder;
    if(twiceRemainder > 127 ||
       (twiceRemainder == 127 && (quotient & 1) != 0))
      quotient++;
    int rounded = negative ? -quotient : quotient;
    rounded = rounded < -127 ? -127 : (rounded > 127 ? 127 : rounded);
    return rounded;
  }

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const& lhs,
    FragmentAccumulator const& rhs
  ) const {
    FragmentOutput output;
    CUTLASS_PRAGMA_UNROLL
    for(int i = 0; i < kCount; i++)
      output[i] = static_cast<Int8>(
        divide127Rne(int(lhs[i]) * int(rhs[i])));
    return output;
  }

  CUTLASS_HOST_DEVICE
  ElementOutput operator()(
    ElementAccumulator const& lhs,
    ElementAccumulator const& rhs
  ) const {
    return static_cast<Int8>(divide127Rne(int(lhs) * int(rhs)));
  }
};

using Clip7UpFactorToInt8 = Clip7FactorToInt8<true,8>;
using Clip7GateFactorToInt8 = Clip7FactorToInt8<false,8>;
using Clip7ProductToInt8 = QuantizedClip7FactorMul<8>;

template<int Stages, int Swizzle>
using DualFfnGemm = cutlass::gemm::device::DualGemm<
  Int8,LayoutA,
  Int8,LayoutB,LayoutB,
  Output,LayoutOutput,
  Accum,
  cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<128,64,64>,
  cutlass::gemm::GemmShape<64,32,64>,
  cutlass::gemm::GemmShape<16,8,32>,
  DequantToHalf,DequantToHalf,Clip7SwiGLU,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swizzle>,
  Stages,false,false,false,16,16,
  cutlass::arch::OpMultiplyAddSaturate>;

template<int Stages, int Swizzle>
using DualFfnInt8Gemm = cutlass::gemm::device::DualGemm<
  Int8,LayoutA,
  Int8,LayoutB,LayoutB,
  Int8,LayoutOutput,
  Accum,
  cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<128,64,64>,
  cutlass::gemm::GemmShape<64,32,64>,
  cutlass::gemm::GemmShape<16,8,32>,
  Clip7UpFactorToInt8,Clip7GateFactorToInt8,Clip7ProductToInt8,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swizzle>,
  Stages,false,false,false,16,16,
  cutlass::arch::OpMultiplyAddSaturate>;

template<int Stages, int Swizzle>
using ResidualInt8Gemm = cutlass::gemm::device::Gemm<
  Int8,LayoutA,Int8,LayoutB,Output,LayoutOutput,
  Accum,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<128,128,64>,
  cutlass::gemm::GemmShape<64,64,64>,
  cutlass::gemm::GemmShape<16,8,32>,
  DequantResidualToHalf,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swizzle>,
  Stages,16,16,false,cutlass::arch::OpMultiplyAddSaturate>;

using ProjectionS2Sw1 = ProjectionGemm<2,1>;
using ProjectionS3Sw1 = ProjectionGemm<3,1>;
using ProjectionS3Sw2 = ProjectionGemm<3,2>;
using DualS3Sw1 = DualFfnGemm<3,1>;
using DualS3Sw4 = DualFfnGemm<3,4>;
using DualS4Sw1 = DualFfnGemm<4,1>;
using DualInt8S3Sw1 = DualFfnInt8Gemm<3,1>;
using DualInt8S3Sw4 = DualFfnInt8Gemm<3,4>;
using DualInt8S4Sw1 = DualFfnInt8Gemm<4,1>;
using ResidualS2Sw1 = ResidualInt8Gemm<2,1>;
using ResidualS3Sw1 = ResidualInt8Gemm<3,1>;
using ResidualS3Sw2 = ResidualInt8Gemm<3,2>;

static_assert(sizeof(typename ProjectionS3Sw1::GemmKernel::SharedStorage) <= 101376,
  "C384 INT8 projection exceeds RTX 5090 opt-in shared memory");
static_assert(sizeof(typename ProjectionS2Sw1::GemmKernel::SharedStorage) <= 101376 &&
              sizeof(typename ProjectionS3Sw2::GemmKernel::SharedStorage) <= 101376,
  "a C384 INT8 projection candidate exceeds RTX 5090 opt-in shared memory");
static_assert(sizeof(typename DualS3Sw1::DualGemmKernel::SharedStorage) <= 101376,
  "C384 INT8 dual FFN exceeds RTX 5090 opt-in shared memory");
static_assert(sizeof(typename DualS3Sw4::DualGemmKernel::SharedStorage) <= 101376 &&
              sizeof(typename DualS4Sw1::DualGemmKernel::SharedStorage) <= 101376,
  "a C384 INT8 dual-FFN candidate exceeds RTX 5090 opt-in shared memory");
static_assert(sizeof(typename DualInt8S3Sw1::DualGemmKernel::SharedStorage) <= 101376 &&
              sizeof(typename DualInt8S3Sw4::DualGemmKernel::SharedStorage) <= 101376 &&
              sizeof(typename DualInt8S4Sw1::DualGemmKernel::SharedStorage) <= 101376,
  "a fused-output C384 INT8 dual-FFN exceeds RTX 5090 opt-in shared memory");
static_assert(sizeof(typename ResidualS3Sw1::GemmKernel::SharedStorage) <= 101376,
  "C384 INT8 residual projection exceeds RTX 5090 opt-in shared memory");
static_assert(sizeof(typename ResidualS2Sw1::GemmKernel::SharedStorage) <= 101376 &&
              sizeof(typename ResidualS3Sw2::GemmKernel::SharedStorage) <= 101376,
  "a C384 INT8 residual projection candidate exceeds RTX 5090 opt-in shared memory");

union Half4Pack {
  uint2 packed;
  half2 values[2];
};

union Int8x4Pack {
  uint32_t packed;
  int8_t values[4];
};

bool aligned16(const void* pointer) {
  return (reinterpret_cast<std::uintptr_t>(pointer) & 15U) == 0;
}

bool finitePositive(float value) {
  return value > 0.0f && std::isfinite(value);
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
cudaError_t setDynamicSharedAttribute() {
  constexpr int sharedBytes = int(sizeof(typename Kernel::SharedStorage));
  if(sharedBytes < 48 * 1024)
    return cudaSuccess;
  return cudaFuncSetAttribute(
    cutlass::Kernel<Kernel>,cudaFuncAttributeMaxDynamicSharedMemorySize,
    sharedBytes);
}

struct ProjectionHandle {
  ProjectionConfig config;
  int outputChannels;
  float alpha;
};

struct DualFfnHandle {
  DualFfnConfig config;
  float upAlpha;
  float gateAlpha;
};

struct DownHandle {
  DownConfig config;
  float alpha;
};

struct AttentionOutHandle {
  AttentionOutConfig config;
  float alpha;
};

template<typename Gemm>
typename Gemm::Arguments makeProjectionArguments(
  int rows,
  int outputChannels,
  int outputStride,
  const Int8* activation,
  const Int8* packedWeights,
  Output* output,
  float alpha
) {
  using TensorRefA = typename Gemm::TensorRefA;
  using TensorRefB = typename Gemm::TensorRefB;
  using TensorRefC = typename Gemm::TensorRefC;
  using TensorRefD = typename Gemm::TensorRefD;
  return typename Gemm::Arguments(
    {rows,outputChannels,kChannels},
    TensorRefA(const_cast<Int8*>(activation),LayoutA(kChannels)),
    TensorRefB(const_cast<Int8*>(packedWeights),LayoutB(kChannels)),
    TensorRefC(nullptr,LayoutOutput(outputStride)),
    TensorRefD(output,LayoutOutput(outputStride)),
    typename DequantToHalf::Params(alpha,0.0f));
}

template<typename Gemm>
cudaError_t prepareProjectionTyped(const ProjectionHandle& handle) {
  cudaError_t status = setDynamicSharedAttribute<typename Gemm::GemmKernel>();
  if(status != cudaSuccess)
    return status;
  Output* fakeOutput = reinterpret_cast<Output*>(
    const_cast<Int8*>(handle.config.packedWeights));
  const auto first = makeProjectionArguments<Gemm>(
    1,handle.outputChannels,kQkvChannels,handle.config.packedWeights,
    handle.config.packedWeights,fakeOutput,handle.alpha);
  const auto last = makeProjectionArguments<Gemm>(
    handle.config.maxTokenRows,handle.outputChannels,kQkvChannels,
    handle.config.packedWeights,handle.config.packedWeights,
    fakeOutput,handle.alpha);
  return Gemm::can_implement(first) == cutlass::Status::kSuccess &&
         Gemm::can_implement(last) == cutlass::Status::kSuccess ?
    cudaSuccess : cudaErrorNotSupported;
}

template<typename Gemm>
cudaError_t launchProjectionTyped(
  const ProjectionHandle& handle,
  int rows,
  const Int8* activation,
  Output* output,
  int outputStride,
  cudaStream_t stream
) {
  using Kernel = typename Gemm::GemmKernel;
  using Swizzle = typename Gemm::ThreadblockSwizzle;
  const cutlass::gemm::GemmCoord problem(rows,handle.outputChannels,kChannels);
  const cutlass::gemm::GemmCoord tiled = Swizzle::get_tiled_shape(
    problem,{Gemm::ThreadblockShape::kM,Gemm::ThreadblockShape::kN,
             Gemm::ThreadblockShape::kK},1);
  typename Kernel::Mma::IteratorA::TensorRef a(
    const_cast<Int8*>(activation),LayoutA(kChannels));
  typename Kernel::Mma::IteratorB::TensorRef b(
    const_cast<Int8*>(handle.config.packedWeights),LayoutB(kChannels));
  typename Kernel::Epilogue::OutputTileIterator::TensorRef c(
    nullptr,LayoutOutput(outputStride));
  typename Kernel::Epilogue::OutputTileIterator::TensorRef d(
    output,LayoutOutput(outputStride));
  typename Kernel::Params params(
    problem,tiled,a,b,c,d,
    typename DequantToHalf::Params(handle.alpha,0.0f),nullptr);
  constexpr int threads = Kernel::kThreadCount;
  constexpr int sharedBytes = int(sizeof(typename Kernel::SharedStorage));
  cutlass::Kernel<Kernel><<<
    Swizzle::get_grid_shape(tiled),dim3(threads,1,1),sharedBytes,stream>>>(params);
  return cudaPeekAtLastError();
}

template<typename Gemm>
typename Gemm::Arguments makeDualArguments(
  int rows,
  const Int8* activation,
  const Int8* upWeights,
  const Int8* gateWeights,
  typename Gemm::ElementC* output,
  float upAlpha,
  float gateAlpha
) {
  typename Gemm::TensorRefC nullC;
  typename Gemm::TensorRefD nullD;
  return typename Gemm::Arguments{
    cutlass::gemm::DualGemmMode::kGemm,
    {rows,kFfnChannels,kChannels},
    {const_cast<Int8*>(activation),LayoutA(kChannels)},
    {const_cast<Int8*>(upWeights),LayoutB(kChannels)},
    nullC,nullD,
    {const_cast<Int8*>(gateWeights),LayoutB(kChannels)},
    nullC,nullD,
    {output,LayoutOutput(kFfnChannels)},
    typename Gemm::EpilogueOutputOp0::Params(upAlpha),
    typename Gemm::EpilogueOutputOp1::Params(gateAlpha),
    typename Gemm::EpilogueOutputOp2::Params(),1,
  };
}

template<typename Gemm>
cudaError_t prepareDualTyped(const DualFfnHandle& handle) {
  cudaError_t status = setDynamicSharedAttribute<
    typename Gemm::DualGemmKernel>();
  if(status != cudaSuccess)
    return status;
  typename Gemm::ElementC* fakeOutput =
    reinterpret_cast<typename Gemm::ElementC*>(
    const_cast<Int8*>(handle.config.packedUpWeights));
  const auto first = makeDualArguments<Gemm>(
    1,handle.config.packedUpWeights,handle.config.packedUpWeights,
    handle.config.packedGateWeights,fakeOutput,handle.upAlpha,handle.gateAlpha);
  const auto last = makeDualArguments<Gemm>(
    handle.config.maxTokenRows,handle.config.packedUpWeights,
    handle.config.packedUpWeights,handle.config.packedGateWeights,
    fakeOutput,handle.upAlpha,handle.gateAlpha);
  return Gemm::can_implement(first) == cutlass::Status::kSuccess &&
         Gemm::can_implement(last) == cutlass::Status::kSuccess ?
    cudaSuccess : cudaErrorNotSupported;
}

template<typename Gemm>
cudaError_t launchDualTyped(
  const DualFfnHandle& handle,
  int rows,
  const Int8* activation,
  typename Gemm::ElementC* output,
  cudaStream_t stream
) {
  using Kernel = typename Gemm::DualGemmKernel;
  using Swizzle = typename Gemm::ThreadblockSwizzle;
  const cutlass::gemm::GemmCoord problem(rows,kFfnChannels,kChannels);
  const cutlass::gemm::GemmCoord tiled = Swizzle::get_tiled_shape(
    problem,{Gemm::ThreadblockShape::kM,Gemm::ThreadblockShape::kN,
             Gemm::ThreadblockShape::kK},1);
  typename Kernel::DualMma::IteratorA::TensorRef a(
    const_cast<Int8*>(activation),LayoutA(kChannels));
  typename Kernel::DualMma::IteratorB0::TensorRef b0(
    const_cast<Int8*>(handle.config.packedUpWeights),LayoutB(kChannels));
  typename Kernel::DualMma::IteratorB1::TensorRef b1(
    const_cast<Int8*>(handle.config.packedGateWeights),LayoutB(kChannels));
  typename Kernel::Epilogue0::OutputTileIterator::TensorRef nullTensor(
    nullptr,LayoutOutput(kFfnChannels));
  typename Kernel::Epilogue1::OutputTileIterator::TensorRef d2(
    output,LayoutOutput(kFfnChannels));
  typename Kernel::Params params(
    cutlass::gemm::DualGemmMode::kGemm,problem,tiled,
    a,b0,nullTensor,nullTensor,b1,nullTensor,nullTensor,d2,
    typename Gemm::EpilogueOutputOp0::Params(handle.upAlpha),
    typename Gemm::EpilogueOutputOp1::Params(handle.gateAlpha),
    typename Gemm::EpilogueOutputOp2::Params(),nullptr);
  constexpr int threads = Kernel::kThreadCount;
  constexpr int sharedBytes = int(sizeof(typename Kernel::SharedStorage));
  cutlass::Kernel<Kernel><<<
    Swizzle::get_grid_shape(tiled),dim3(threads,1,1),sharedBytes,stream>>>(params);
  return cudaPeekAtLastError();
}

template<typename Gemm>
typename Gemm::Arguments makeResidualArguments(
  int rows,
  int innerChannels,
  int outputChannels,
  const Int8* activation,
  const Int8* packedWeights,
  const Output* residual,
  Output* output,
  float alpha
) {
  return typename Gemm::Arguments(
    {rows,outputChannels,innerChannels},
    typename Gemm::TensorRefA(
      const_cast<Int8*>(activation),LayoutA(innerChannels)),
    typename Gemm::TensorRefB(
      const_cast<Int8*>(packedWeights),LayoutB(innerChannels)),
    typename Gemm::TensorRefC(
      const_cast<Output*>(residual),LayoutOutput(outputChannels)),
    typename Gemm::TensorRefD(output,LayoutOutput(outputChannels)),
    typename DequantResidualToHalf::Params(alpha,1.0f));
}

template<typename Gemm>
cudaError_t prepareResidualTyped(
  int maxTokenRows,
  int innerChannels,
  int outputChannels,
  const Int8* packedWeights,
  float alpha
) {
  cudaError_t status = setDynamicSharedAttribute<typename Gemm::GemmKernel>();
  if(status != cudaSuccess)
    return status;
  Output* fakeOutput = reinterpret_cast<Output*>(
    const_cast<Int8*>(packedWeights));
  const auto first = makeResidualArguments<Gemm>(
    1,innerChannels,outputChannels,packedWeights,packedWeights,
    fakeOutput,fakeOutput,alpha);
  const auto last = makeResidualArguments<Gemm>(
    maxTokenRows,innerChannels,outputChannels,packedWeights,packedWeights,
    fakeOutput,fakeOutput,alpha);
  return Gemm::can_implement(first) == cutlass::Status::kSuccess &&
         Gemm::can_implement(last) == cutlass::Status::kSuccess ?
    cudaSuccess : cudaErrorNotSupported;
}

template<typename Gemm>
cudaError_t launchResidualTyped(
  int rows,
  int innerChannels,
  int outputChannels,
  const Int8* packedWeights,
  float alpha,
  const Int8* activation,
  const Output* residual,
  Output* output,
  cudaStream_t stream
) {
  using Kernel = typename Gemm::GemmKernel;
  using Swizzle = typename Gemm::ThreadblockSwizzle;
  const cutlass::gemm::GemmCoord problem(rows,outputChannels,innerChannels);
  const cutlass::gemm::GemmCoord tiled = Swizzle::get_tiled_shape(
    problem,{Gemm::ThreadblockShape::kM,Gemm::ThreadblockShape::kN,
             Gemm::ThreadblockShape::kK},1);
  typename Kernel::Mma::IteratorA::TensorRef a(
    const_cast<Int8*>(activation),LayoutA(innerChannels));
  typename Kernel::Mma::IteratorB::TensorRef b(
    const_cast<Int8*>(packedWeights),LayoutB(innerChannels));
  typename Kernel::Epilogue::OutputTileIterator::TensorRef c(
    const_cast<Output*>(residual),LayoutOutput(outputChannels));
  typename Kernel::Epilogue::OutputTileIterator::TensorRef d(
    output,LayoutOutput(outputChannels));
  typename Kernel::Params params(
    problem,tiled,a,b,c,d,
    typename DequantResidualToHalf::Params(alpha,1.0f),nullptr);
  constexpr int threads = Kernel::kThreadCount;
  constexpr int sharedBytes = int(sizeof(typename Kernel::SharedStorage));
  cutlass::Kernel<Kernel><<<
    Swizzle::get_grid_shape(tiled),dim3(threads,1,1),sharedBytes,stream>>>(params);
  return cudaPeekAtLastError();
}

template<bool EmitFp16>
__global__ void rmsNorm384Int8Kernel(
  const uint2* __restrict__ input,
  uint2* __restrict__ outputFp16,
  uint32_t* __restrict__ outputInt8,
  const uint2* __restrict__ gamma,
  int tokenRows,
  float epsilon
) {
  const int warp = int(threadIdx.x) >> 5;
  const int lane = int(threadIdx.x) & 31;
  const int row = int(blockIdx.x) * 4 + warp;
  if(row >= tokenRows)
    return;
  constexpr int vectorsPerRow = kChannels / 4;
  constexpr int rounds = vectorsPerRow / 32;
  Half4Pack in[rounds];
  Half4Pack g[rounds];
  Half4Pack out[rounds];
  Int8x4Pack quantized[rounds];
  float values[rounds][4];
  float sumSquares = 0.0f;
  #pragma unroll
  for(int round = 0; round < rounds; round++) {
    const int vector = lane + round * 32;
    in[round].packed = input[std::size_t(row) * vectorsPerRow + vector];
    g[round].packed = gamma[vector];
    #pragma unroll
    for(int pair = 0; pair < 2; pair++) {
      const float2 value = __half22float2(in[round].values[pair]);
      values[round][2 * pair] = value.x;
      values[round][2 * pair + 1] = value.y;
      sumSquares += value.x * value.x + value.y * value.y;
    }
  }
  for(int offset = 16; offset > 0; offset >>= 1)
    sumSquares += __shfl_xor_sync(0xffffffffu,sumSquares,offset);
  const float scale = rsqrtf(sumSquares / float(kChannels) + epsilon);
  #pragma unroll
  for(int round = 0; round < rounds; round++) {
    #pragma unroll
    for(int pair = 0; pair < 2; pair++) {
      const float2 gammaValues = __half22float2(g[round].values[pair]);
      out[round].values[pair] = __floats2half2_rn(
        values[round][2 * pair] * scale * gammaValues.x,
        values[round][2 * pair + 1] * scale * gammaValues.y);
      const float2 rounded = __half22float2(out[round].values[pair]);
      const float x0 = fminf(4.0f,fmaxf(-4.0f,rounded.x));
      const float x1 = fminf(4.0f,fmaxf(-4.0f,rounded.y));
      int q0 = __float2int_rn(x0 * (127.0f / 4.0f));
      int q1 = __float2int_rn(x1 * (127.0f / 4.0f));
      q0 = q0 < -127 ? -127 : (q0 > 127 ? 127 : q0);
      q1 = q1 < -127 ? -127 : (q1 > 127 ? 127 : q1);
      quantized[round].values[2 * pair] = static_cast<int8_t>(q0);
      quantized[round].values[2 * pair + 1] = static_cast<int8_t>(q1);
    }
    const int vector = lane + round * 32;
    const std::size_t index = std::size_t(row) * vectorsPerRow + vector;
    if constexpr(EmitFp16)
      outputFp16[index] = out[round].packed;
    outputInt8[index] = quantized[round].packed;
  }
}

__global__ void quantizeClip7ProductKernel(
  const uint2* __restrict__ input,
  uint32_t* __restrict__ output,
  std::size_t vectors
) {
  for(std::size_t vector = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
      vector < vectors; vector += std::size_t(gridDim.x) * blockDim.x) {
    Half4Pack in;
    Int8x4Pack out;
    in.packed = input[vector];
    #pragma unroll
    for(int pair = 0; pair < 2; pair++) {
      const float2 value = __half22float2(in.values[pair]);
      const float x0 = fminf(49.0f,fmaxf(-49.0f,value.x));
      const float x1 = fminf(49.0f,fmaxf(-49.0f,value.y));
      int q0 = __float2int_rn(x0 * (127.0f / 49.0f));
      int q1 = __float2int_rn(x1 * (127.0f / 49.0f));
      q0 = q0 < -127 ? -127 : (q0 > 127 ? 127 : q0);
      q1 = q1 < -127 ? -127 : (q1 > 127 ? 127 : q1);
      out.values[2 * pair] = static_cast<int8_t>(q0);
      out.values[2 * pair + 1] = static_cast<int8_t>(q1);
    }
    output[vector] = out.packed;
  }
}

__global__ void quantizeAttentionOutputKernel(
  const uint2* __restrict__ input,
  uint32_t* __restrict__ output,
  std::size_t vectors
) {
  for(std::size_t vector = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
      vector < vectors; vector += std::size_t(gridDim.x) * blockDim.x) {
    Half4Pack in;
    Int8x4Pack out;
    in.packed = input[vector];
    #pragma unroll
    for(int pair = 0; pair < 2; pair++) {
      const float2 value = __half22float2(in.values[pair]);
      const float x0 = fminf(kNormActivationClip,
        fmaxf(-kNormActivationClip,value.x));
      const float x1 = fminf(kNormActivationClip,
        fmaxf(-kNormActivationClip,value.y));
      int q0 = __float2int_rn(x0 * (127.0f / kNormActivationClip));
      int q1 = __float2int_rn(x1 * (127.0f / kNormActivationClip));
      q0 = q0 < -127 ? -127 : (q0 > 127 ? 127 : q0);
      q1 = q1 < -127 ? -127 : (q1 > 127 ? 127 : q1);
      out.values[2 * pair] = static_cast<int8_t>(q0);
      out.values[2 * pair + 1] = static_cast<int8_t>(q1);
    }
    output[vector] = out.packed;
  }
}

__global__ void packPlanarVKernel(
  const uint2* __restrict__ planar,
  uint2* __restrict__ packed,
  int tokenRows
) {
  constexpr int vectorsPerPlane = kChannels / 4;
  constexpr int vectorsPerPackedRow = kQkvChannels / 4;
  const std::size_t total = std::size_t(tokenRows) * vectorsPerPlane;
  for(std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
      index < total; index += std::size_t(gridDim.x) * blockDim.x) {
    const std::size_t row = index / vectorsPerPlane;
    const std::size_t vector = index - row * vectorsPerPlane;
    packed[row * vectorsPerPackedRow + 2 * vectorsPerPlane + vector] =
      planar[index];
  }
}

}  // namespace

cudaError_t launchRmsNormFp16Int8(
  const half* input,
  half* outputFp16,
  int8_t* outputInt8,
  const half* gamma,
  int tokenRows,
  float epsilon,
  cudaStream_t stream
) {
  if(input == nullptr || outputFp16 == nullptr || outputInt8 == nullptr ||
     gamma == nullptr || tokenRows <= 0 || tokenRows > kMaxTokenRows ||
     !finitePositive(epsilon) || !aligned16(input) ||
     !aligned16(outputFp16) || !aligned16(outputInt8) || !aligned16(gamma))
    return cudaErrorInvalidValue;
  rmsNorm384Int8Kernel<true><<<
    (tokenRows + 3) / 4,kThreadsPerRmsBlock,0,stream>>>(
      reinterpret_cast<const uint2*>(input),
      reinterpret_cast<uint2*>(outputFp16),
      reinterpret_cast<uint32_t*>(outputInt8),
      reinterpret_cast<const uint2*>(gamma),tokenRows,epsilon);
  return cudaPeekAtLastError();
}

cudaError_t launchRmsNormInt8(
  const half* input,
  int8_t* outputInt8,
  const half* gamma,
  int tokenRows,
  float epsilon,
  cudaStream_t stream
) {
  if(input == nullptr || outputInt8 == nullptr || gamma == nullptr ||
     tokenRows <= 0 || tokenRows > kMaxTokenRows ||
     !finitePositive(epsilon) || !aligned16(input) ||
     !aligned16(outputInt8) || !aligned16(gamma))
    return cudaErrorInvalidValue;
  rmsNorm384Int8Kernel<false><<<
    (tokenRows + 3) / 4,kThreadsPerRmsBlock,0,stream>>>(
      reinterpret_cast<const uint2*>(input),nullptr,
      reinterpret_cast<uint32_t*>(outputInt8),
      reinterpret_cast<const uint2*>(gamma),tokenRows,epsilon);
  return cudaPeekAtLastError();
}

void* createProjection(const ProjectionConfig& config) {
  const int outputChannels = config.mode == ProjectionMode::ConservativeQk ?
    kQkChannels : (config.mode == ProjectionMode::AggressiveQkv ?
      kQkvChannels : 0);
  if(outputChannels == 0 || config.maxTokenRows <= 0 ||
     config.maxTokenRows > kMaxTokenRows || config.packedWeights == nullptr ||
     !aligned16(config.packedWeights) || !finitePositive(config.weightScale) ||
     !isSm120Compatible())
    return nullptr;
  ProjectionHandle handle{config,outputChannels,
    kNormActivationScale * config.weightScale};
  cudaError_t status = cudaErrorNotSupported;
  switch(config.tactic) {
  case ProjectionTactic::M128N128K64S2Sw1:
    status = prepareProjectionTyped<ProjectionS2Sw1>(handle); break;
  case ProjectionTactic::M128N128K64S3Sw1:
    status = prepareProjectionTyped<ProjectionS3Sw1>(handle); break;
  case ProjectionTactic::M128N128K64S3Sw2:
    status = prepareProjectionTyped<ProjectionS3Sw2>(handle); break;
  }
  if(status != cudaSuccess)
    return nullptr;
  return new(std::nothrow) ProjectionHandle(handle);
}

void destroyProjection(void* opaque) noexcept {
  delete static_cast<ProjectionHandle*>(opaque);
}

bool projectionSupports(
  const void* opaque,
  ProjectionMode mode,
  int tokenRows,
  int inputChannels,
  int outputRowStride
) noexcept {
  const ProjectionHandle* handle = static_cast<const ProjectionHandle*>(opaque);
  return handle != nullptr && handle->config.mode == mode && tokenRows > 0 &&
    tokenRows <= handle->config.maxTokenRows && inputChannels == kChannels &&
    outputRowStride == kQkvChannels;
}

cudaError_t launchProjection(
  void* opaque,
  int tokenRows,
  const int8_t* activation,
  half* rawPackedQkv,
  int outputRowStride,
  cudaStream_t stream
) {
  ProjectionHandle* handle = static_cast<ProjectionHandle*>(opaque);
  if(handle == nullptr || activation == nullptr || rawPackedQkv == nullptr ||
     !projectionSupports(handle,handle->config.mode,tokenRows,kChannels,
       outputRowStride) || !aligned16(activation) || !aligned16(rawPackedQkv))
    return cudaErrorInvalidValue;
  Output* output = reinterpret_cast<Output*>(rawPackedQkv);
  switch(handle->config.tactic) {
  case ProjectionTactic::M128N128K64S2Sw1:
    return launchProjectionTyped<ProjectionS2Sw1>(
      *handle,tokenRows,activation,output,outputRowStride,stream);
  case ProjectionTactic::M128N128K64S3Sw1:
    return launchProjectionTyped<ProjectionS3Sw1>(
      *handle,tokenRows,activation,output,outputRowStride,stream);
  case ProjectionTactic::M128N128K64S3Sw2:
    return launchProjectionTyped<ProjectionS3Sw2>(
      *handle,tokenRows,activation,output,outputRowStride,stream);
  }
  return cudaErrorNotSupported;
}

void* createDualFfn(const DualFfnConfig& config) {
  if(config.maxTokenRows <= 0 || config.maxTokenRows > kMaxTokenRows ||
     config.packedUpWeights == nullptr || config.packedGateWeights == nullptr ||
     !aligned16(config.packedUpWeights) || !aligned16(config.packedGateWeights) ||
     !finitePositive(config.upWeightScale) ||
     !finitePositive(config.gateWeightScale) ||
     (config.outputMode != DualFfnOutputMode::Fp16Product &&
      config.outputMode != DualFfnOutputMode::Int8Product) ||
     !isSm120Compatible())
    return nullptr;
  DualFfnHandle handle{config,
    kNormActivationScale * config.upWeightScale,
    kNormActivationScale * config.gateWeightScale};
  cudaError_t status = cudaErrorNotSupported;
  if(config.outputMode == DualFfnOutputMode::Fp16Product) {
    switch(config.tactic) {
    case DualFfnTactic::M128N64K64S3Sw1:
      status = prepareDualTyped<DualS3Sw1>(handle); break;
    case DualFfnTactic::M128N64K64S3Sw4:
      status = prepareDualTyped<DualS3Sw4>(handle); break;
    case DualFfnTactic::M128N64K64S4Sw1:
      status = prepareDualTyped<DualS4Sw1>(handle); break;
    }
  }
  else {
    switch(config.tactic) {
    case DualFfnTactic::M128N64K64S3Sw1:
      status = prepareDualTyped<DualInt8S3Sw1>(handle); break;
    case DualFfnTactic::M128N64K64S3Sw4:
      status = prepareDualTyped<DualInt8S3Sw4>(handle); break;
    case DualFfnTactic::M128N64K64S4Sw1:
      status = prepareDualTyped<DualInt8S4Sw1>(handle); break;
    }
  }
  if(status != cudaSuccess)
    return nullptr;
  return new(std::nothrow) DualFfnHandle(handle);
}

void destroyDualFfn(void* opaque) noexcept {
  delete static_cast<DualFfnHandle*>(opaque);
}

bool dualFfnSupports(
  const void* opaque,
  DualFfnOutputMode outputMode,
  int tokenRows
) noexcept {
  const DualFfnHandle* handle = static_cast<const DualFfnHandle*>(opaque);
  return handle != nullptr && handle->config.outputMode == outputMode &&
    tokenRows > 0 &&
    tokenRows <= handle->config.maxTokenRows;
}

cudaError_t launchDualFfnHalf(
  void* opaque,
  int tokenRows,
  const int8_t* activation,
  half* productFp16,
  cudaStream_t stream
) {
  DualFfnHandle* handle = static_cast<DualFfnHandle*>(opaque);
  if(handle == nullptr || activation == nullptr || productFp16 == nullptr ||
     !dualFfnSupports(handle,DualFfnOutputMode::Fp16Product,tokenRows) ||
     !aligned16(activation) ||
     !aligned16(productFp16))
    return cudaErrorInvalidValue;
  Output* output = reinterpret_cast<Output*>(productFp16);
  switch(handle->config.tactic) {
  case DualFfnTactic::M128N64K64S3Sw1:
    return launchDualTyped<DualS3Sw1>(
      *handle,tokenRows,activation,output,stream);
  case DualFfnTactic::M128N64K64S3Sw4:
    return launchDualTyped<DualS3Sw4>(
      *handle,tokenRows,activation,output,stream);
  case DualFfnTactic::M128N64K64S4Sw1:
    return launchDualTyped<DualS4Sw1>(
      *handle,tokenRows,activation,output,stream);
  }
  return cudaErrorNotSupported;
}

cudaError_t launchDualFfnInt8(
  void* opaque,
  int tokenRows,
  const int8_t* activation,
  int8_t* productInt8,
  cudaStream_t stream
) {
  DualFfnHandle* handle = static_cast<DualFfnHandle*>(opaque);
  if(handle == nullptr || activation == nullptr || productInt8 == nullptr ||
     !dualFfnSupports(handle,DualFfnOutputMode::Int8Product,tokenRows) ||
     !aligned16(activation) || !aligned16(productInt8))
    return cudaErrorInvalidValue;
  switch(handle->config.tactic) {
  case DualFfnTactic::M128N64K64S3Sw1:
    return launchDualTyped<DualInt8S3Sw1>(
      *handle,tokenRows,activation,productInt8,stream);
  case DualFfnTactic::M128N64K64S3Sw4:
    return launchDualTyped<DualInt8S3Sw4>(
      *handle,tokenRows,activation,productInt8,stream);
  case DualFfnTactic::M128N64K64S4Sw1:
    return launchDualTyped<DualInt8S4Sw1>(
      *handle,tokenRows,activation,productInt8,stream);
  }
  return cudaErrorNotSupported;
}

cudaError_t launchQuantizeClip7Product(
  const half* productFp16,
  int8_t* productInt8,
  int tokenRows,
  cudaStream_t stream
) {
  if(productFp16 == nullptr || productInt8 == nullptr || tokenRows <= 0 ||
     tokenRows > kMaxTokenRows || !aligned16(productFp16) ||
     !aligned16(productInt8))
    return cudaErrorInvalidValue;
  const std::size_t vectors =
    std::size_t(tokenRows) * kFfnChannels / 4;
  const int blocks = int((vectors + kThreadsPerQuantBlock - 1) /
    kThreadsPerQuantBlock);
  quantizeClip7ProductKernel<<<
    blocks,kThreadsPerQuantBlock,0,stream>>>(
      reinterpret_cast<const uint2*>(productFp16),
      reinterpret_cast<uint32_t*>(productInt8),vectors);
  return cudaPeekAtLastError();
}

cudaError_t launchQuantizeAttentionOutput(
  const half* attentionFp16,
  int8_t* attentionInt8,
  int tokenRows,
  cudaStream_t stream
) {
  if(attentionFp16 == nullptr || attentionInt8 == nullptr || tokenRows <= 0 ||
     tokenRows > kMaxTokenRows || !aligned16(attentionFp16) ||
     !aligned16(attentionInt8))
    return cudaErrorInvalidValue;
  const std::size_t vectors = std::size_t(tokenRows) * kChannels / 4;
  // Keep the lightweight conversion from flooding the scheduler ahead of the
  // following GEMM. The kernel is grid-stride, so this cap preserves the full
  // tensor including the final partial traversal.
  constexpr int kAttentionQuantGridCap = 340;
  const int fullBlocks = int((vectors + kThreadsPerQuantBlock - 1) /
    kThreadsPerQuantBlock);
  const int blocks = fullBlocks < kAttentionQuantGridCap ?
    fullBlocks : kAttentionQuantGridCap;
  quantizeAttentionOutputKernel<<<
    blocks,kThreadsPerQuantBlock,0,stream>>>(
      reinterpret_cast<const uint2*>(attentionFp16),
      reinterpret_cast<uint32_t*>(attentionInt8),vectors);
  return cudaPeekAtLastError();
}

cudaError_t launchPackPlanarV(
  const half* planarV,
  half* rawPackedQkv,
  int tokenRows,
  cudaStream_t stream
) {
  if(planarV == nullptr || rawPackedQkv == nullptr || tokenRows <= 0 ||
     tokenRows > kMaxTokenRows || !aligned16(planarV) ||
     !aligned16(rawPackedQkv))
    return cudaErrorInvalidValue;
  constexpr int vectorsPerPlane = kChannels / 4;
  const std::size_t vectors = std::size_t(tokenRows) * vectorsPerPlane;
  const int blocks = int((vectors + kThreadsPerQuantBlock - 1) /
    kThreadsPerQuantBlock);
  packPlanarVKernel<<<blocks,kThreadsPerQuantBlock,0,stream>>>(
    reinterpret_cast<const uint2*>(planarV),
    reinterpret_cast<uint2*>(rawPackedQkv),tokenRows);
  return cudaPeekAtLastError();
}

void* createDown(const DownConfig& config) {
  if(config.maxTokenRows <= 0 || config.maxTokenRows > kMaxTokenRows ||
     config.packedWeights == nullptr || !aligned16(config.packedWeights) ||
     !finitePositive(config.weightScale) || !isSm120Compatible())
    return nullptr;
  DownHandle handle{config,kClip7ProductScale * config.weightScale};
  cudaError_t status = cudaErrorNotSupported;
  switch(config.tactic) {
  case DownTactic::M128N128K64S2Sw1:
    status = prepareResidualTyped<ResidualS2Sw1>(
      config.maxTokenRows,kFfnChannels,kChannels,
      config.packedWeights,handle.alpha); break;
  case DownTactic::M128N128K64S3Sw1:
    status = prepareResidualTyped<ResidualS3Sw1>(
      config.maxTokenRows,kFfnChannels,kChannels,
      config.packedWeights,handle.alpha); break;
  case DownTactic::M128N128K64S3Sw2:
    status = prepareResidualTyped<ResidualS3Sw2>(
      config.maxTokenRows,kFfnChannels,kChannels,
      config.packedWeights,handle.alpha); break;
  }
  if(status != cudaSuccess)
    return nullptr;
  return new(std::nothrow) DownHandle(handle);
}

void destroyDown(void* opaque) noexcept {
  delete static_cast<DownHandle*>(opaque);
}

bool downSupports(const void* opaque, int tokenRows) noexcept {
  const DownHandle* handle = static_cast<const DownHandle*>(opaque);
  return handle != nullptr && tokenRows > 0 &&
    tokenRows <= handle->config.maxTokenRows;
}

cudaError_t launchDownResidual(
  void* opaque,
  int tokenRows,
  const int8_t* productInt8,
  const half* residual,
  half* output,
  cudaStream_t stream
) {
  DownHandle* handle = static_cast<DownHandle*>(opaque);
  if(handle == nullptr || productInt8 == nullptr || residual == nullptr ||
     output == nullptr || !downSupports(handle,tokenRows) ||
     !aligned16(productInt8) || !aligned16(residual) || !aligned16(output))
    return cudaErrorInvalidValue;
  const Output* residualOutput = reinterpret_cast<const Output*>(residual);
  Output* destination = reinterpret_cast<Output*>(output);
  switch(handle->config.tactic) {
  case DownTactic::M128N128K64S2Sw1:
    return launchResidualTyped<ResidualS2Sw1>(
      tokenRows,kFfnChannels,kChannels,handle->config.packedWeights,
      handle->alpha,productInt8,residualOutput,destination,stream);
  case DownTactic::M128N128K64S3Sw1:
    return launchResidualTyped<ResidualS3Sw1>(
      tokenRows,kFfnChannels,kChannels,handle->config.packedWeights,
      handle->alpha,productInt8,residualOutput,destination,stream);
  case DownTactic::M128N128K64S3Sw2:
    return launchResidualTyped<ResidualS3Sw2>(
      tokenRows,kFfnChannels,kChannels,handle->config.packedWeights,
      handle->alpha,productInt8,residualOutput,destination,stream);
  }
  return cudaErrorNotSupported;
}

void* createAttentionOut(const AttentionOutConfig& config) {
  if(config.maxTokenRows <= 0 || config.maxTokenRows > kMaxTokenRows ||
     config.packedWeights == nullptr || !aligned16(config.packedWeights) ||
     !finitePositive(config.weightScale) || !isSm120Compatible())
    return nullptr;
  AttentionOutHandle handle{
    config,kNormActivationScale * config.weightScale};
  cudaError_t status = cudaErrorNotSupported;
  switch(config.tactic) {
  case AttentionOutTactic::M128N128K64S2Sw1:
    status = prepareResidualTyped<ResidualS2Sw1>(
      config.maxTokenRows,kChannels,kChannels,
      config.packedWeights,handle.alpha); break;
  case AttentionOutTactic::M128N128K64S3Sw1:
    status = prepareResidualTyped<ResidualS3Sw1>(
      config.maxTokenRows,kChannels,kChannels,
      config.packedWeights,handle.alpha); break;
  case AttentionOutTactic::M128N128K64S3Sw2:
    status = prepareResidualTyped<ResidualS3Sw2>(
      config.maxTokenRows,kChannels,kChannels,
      config.packedWeights,handle.alpha); break;
  }
  if(status != cudaSuccess)
    return nullptr;
  return new(std::nothrow) AttentionOutHandle(handle);
}

void destroyAttentionOut(void* opaque) noexcept {
  delete static_cast<AttentionOutHandle*>(opaque);
}

bool attentionOutSupports(const void* opaque, int tokenRows) noexcept {
  const AttentionOutHandle* handle =
    static_cast<const AttentionOutHandle*>(opaque);
  return handle != nullptr && tokenRows > 0 &&
    tokenRows <= handle->config.maxTokenRows;
}

cudaError_t launchAttentionOutResidual(
  void* opaque,
  int tokenRows,
  const int8_t* attentionInt8,
  const half* residual,
  half* output,
  cudaStream_t stream
) {
  AttentionOutHandle* handle = static_cast<AttentionOutHandle*>(opaque);
  if(handle == nullptr || attentionInt8 == nullptr || residual == nullptr ||
     output == nullptr || !attentionOutSupports(handle,tokenRows) ||
     !aligned16(attentionInt8) || !aligned16(residual) || !aligned16(output))
    return cudaErrorInvalidValue;
  const Output* residualOutput = reinterpret_cast<const Output*>(residual);
  Output* destination = reinterpret_cast<Output*>(output);
  switch(handle->config.tactic) {
  case AttentionOutTactic::M128N128K64S2Sw1:
    return launchResidualTyped<ResidualS2Sw1>(
      tokenRows,kChannels,kChannels,handle->config.packedWeights,
      handle->alpha,attentionInt8,residualOutput,destination,stream);
  case AttentionOutTactic::M128N128K64S3Sw1:
    return launchResidualTyped<ResidualS3Sw1>(
      tokenRows,kChannels,kChannels,handle->config.packedWeights,
      handle->alpha,attentionInt8,residualOutput,destination,stream);
  case AttentionOutTactic::M128N128K64S3Sw2:
    return launchResidualTyped<ResidualS3Sw2>(
      tokenRows,kChannels,kChannels,handle->config.packedWeights,
      handle->alpha,attentionInt8,residualOutput,destination,stream);
  }
  return cudaErrorNotSupported;
}

const char* projectionTacticName(ProjectionTactic tactic) noexcept {
  switch(tactic) {
  case ProjectionTactic::M128N128K64S2Sw1:
    return "int8-m128n128k64-s2-sw1";
  case ProjectionTactic::M128N128K64S3Sw1:
    return "int8-m128n128k64-s3-sw1";
  case ProjectionTactic::M128N128K64S3Sw2:
    return "int8-m128n128k64-s3-sw2";
  }
  return "invalid";
}

const char* dualFfnTacticName(DualFfnTactic tactic) noexcept {
  switch(tactic) {
  case DualFfnTactic::M128N64K64S3Sw1:
    return "int8-dual-clip7-m128n64k64-s3-sw1";
  case DualFfnTactic::M128N64K64S3Sw4:
    return "int8-dual-clip7-m128n64k64-s3-sw4";
  case DualFfnTactic::M128N64K64S4Sw1:
    return "int8-dual-clip7-m128n64k64-s4-sw1";
  }
  return "invalid";
}

const char* downTacticName(DownTactic tactic) noexcept {
  switch(tactic) {
  case DownTactic::M128N128K64S2Sw1:
    return "int8-down-beta1-m128n128k64-s2-sw1";
  case DownTactic::M128N128K64S3Sw1:
    return "int8-down-beta1-m128n128k64-s3-sw1";
  case DownTactic::M128N128K64S3Sw2:
    return "int8-down-beta1-m128n128k64-s3-sw2";
  }
  return "invalid";
}

const char* attentionOutTacticName(AttentionOutTactic tactic) noexcept {
  switch(tactic) {
  case AttentionOutTactic::M128N128K64S2Sw1:
    return "int8-attention-out-k384-beta1-m128n128k64-s2-sw1";
  case AttentionOutTactic::M128N128K64S3Sw1:
    return "int8-attention-out-k384-beta1-m128n128k64-s3-sw1";
  case AttentionOutTactic::M128N128K64S3Sw2:
    return "int8-attention-out-k384-beta1-m128n128k64-s3-sw2";
  }
  return "invalid";
}

}  // namespace C384Int8Experiment

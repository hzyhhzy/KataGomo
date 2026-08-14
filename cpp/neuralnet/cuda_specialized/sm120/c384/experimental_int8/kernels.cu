#include "kernels.h"

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm.h"
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

// The P3 output thread map is particularly useful for D32 QK normalization:
// each warp covers four output rows over its two epilogue row iterations;
// each row uses 16 lanes, every lane owns 8 adjacent half values, and thus
// each four-lane subgroup owns exactly one head.
// N tiles are 128-wide and all Q/K/V plane boundaries are multiples of 128,
// so no CTA or subgroup straddles a plane boundary.  This iterator consumes
// the already-dequantized FP16 epilogue fragment, performs the D32 FP32
// reduction in registers, preserves the FP16 post-gamma rounding boundary,
// applies learned RoPE, and stores the final packed Q/K values once.  V takes
// the unmodified Base path and is therefore bit-identical to ProjectionGemm.
template<typename BaseIterator>
class QknormRopeOutputTileIterator : public BaseIterator {
public:
  using IteratorBase = BaseIterator;
  using Base = BaseIterator;
  using ThreadMap = typename Base::ThreadMap;
  using Element = typename Base::Element;
  using Layout = typename Base::Layout;
  using TensorRef = typename Base::TensorRef;
  using ConstTensorRef = typename Base::ConstTensorRef;
  using TensorCoord = typename Base::TensorCoord;
  using LongIndex = typename Base::LongIndex;
  using Fragment = typename Base::Fragment;
  using AccessType = typename Base::AccessType;
  using Mask = typename Base::Mask;

  static int const kElementsPerAccess = Base::kElementsPerAccess;
  static int const kIterations = Base::kIterations;

  static_assert(kElementsPerAccess == 8,
    "C384 fused QKNorm requires the qualified eight-half output access");
  static_assert(kHeadDim == 4 * kElementsPerAccess,
    "four adjacent epilogue lanes must cover one complete D32 head");
  static_assert(ThreadMap::kThreads == 128,
    "C384 fused QKNorm requires the qualified 128-thread epilogue map");
  static_assert(ThreadMap::Iterations::kColumn == 1 &&
                ThreadMap::Iterations::kRow == 2 &&
                ThreadMap::Iterations::kGroup == 1 &&
                ThreadMap::Iterations::kCluster == 1,
    "C384 fused QKNorm epilogue iteration geometry changed");
  static_assert(ThreadMap::Delta::kColumn == 128 &&
                ThreadMap::Delta::kRow == 2,
    "C384 fused QKNorm epilogue access deltas changed");
  static_assert(ThreadMap::Shape::kColumn == 128,
    "C384 fused QKNorm requires a 128-wide output tile");
  static_assert((kChannels % 128) == 0,
    "Q/K/V boundaries must align with the 128-wide projection N tile");

  struct Params : public IteratorBase::Params {
    const half* qGamma;
    const half* kGamma;
    const half2* ropeCosSin;
    int totalRows;
    float qEpsilon;
    float kEpsilon;

    CUTLASS_HOST_DEVICE
    Params() : IteratorBase::Params(), qGamma(nullptr), kGamma(nullptr),
      ropeCosSin(nullptr), totalRows(0), qEpsilon(0.0f), kEpsilon(0.0f) {}

    CUTLASS_HOST_DEVICE
    explicit Params(Layout const& layout) : IteratorBase::Params(layout),
      qGamma(nullptr), kGamma(nullptr), ropeCosSin(nullptr), totalRows(0),
      qEpsilon(0.0f), kEpsilon(0.0f) {}
  };

private:
  const half* qGamma_;
  const half* kGamma_;
  const half2* ropeCosSin_;
  int totalRows_;
  float qEpsilon_;
  float kEpsilon_;

public:
  CUTLASS_DEVICE
  QknormRopeOutputTileIterator(
    Params const& params,
    Element* pointer,
    TensorCoord extent,
    int threadIdx,
    TensorCoord threadblockOffset = TensorCoord(),
    int const* indices = nullptr
  ) : Base(params,pointer,extent,threadIdx,threadblockOffset,indices),
      qGamma_(params.qGamma), kGamma_(params.kGamma),
      ropeCosSin_(params.ropeCosSin), totalRows_(params.totalRows),
      qEpsilon_(params.qEpsilon), kEpsilon_(params.kEpsilon) {}

  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const& fragment, int64_t byteOffset) const {
    Fragment transformed = fragment;
    AccessType* accesses = reinterpret_cast<AccessType*>(&transformed);
    const int startRow = Base::thread_start_row();
    const int startColumn = Base::thread_start_column();

    CUTLASS_PRAGMA_UNROLL
    for(int cluster = 0; cluster < ThreadMap::Iterations::kCluster; cluster++) {
      CUTLASS_PRAGMA_UNROLL
      for(int group = 0; group < ThreadMap::Iterations::kGroup; group++) {
        CUTLASS_PRAGMA_UNROLL
        for(int row = 0; row < ThreadMap::Iterations::kRow; row++) {
          const int fragmentRow = row + ThreadMap::Iterations::kRow *
            (group + ThreadMap::Iterations::kGroup * cluster);
          const int rowOffset = row * ThreadMap::Delta::kRow +
            group * ThreadMap::Delta::kGroup +
            cluster * ThreadMap::Delta::kCluster;
          const int outputRow = startRow + rowOffset;

          CUTLASS_PRAGMA_UNROLL
          for(int column = 0; column < ThreadMap::Iterations::kColumn; column++) {
            const int outputColumn =
              startColumn + column * ThreadMap::Delta::kColumn;
            AccessType& access = accesses[
              fragmentRow * ThreadMap::Iterations::kColumn + column];

            // Every 16-lane row subgroup is wholly in Q/K or V because the
            // 128-wide N tile and all plane boundaries are aligned. Keep the
            // subgroup converged across shuffles, including tail rows.
            if(outputColumn < kQkChannels) {
              float sumSquares = 0.0f;
              CUTLASS_PRAGMA_UNROLL
              for(int element = 0; element < kElementsPerAccess; element++) {
                const float value = static_cast<float>(access[element]);
                sumSquares += value * value;
              }
              sumSquares += __shfl_xor_sync(
                0xffffffffu,sumSquares,2,kHeadDim / kElementsPerAccess);
              sumSquares += __shfl_xor_sync(
                0xffffffffu,sumSquares,1,kHeadDim / kElementsPerAccess);

              // Predication only affects whole 16-lane row groups.  Do the
              // reduction above unconditionally, then avoid an out-of-range
              // RoPE table access for the final partial M tile.
              if(outputRow < totalRows_) {
                const int plane = outputColumn / kChannels;
                const int channelInPlane = outputColumn - plane * kChannels;
                const int head = channelInPlane / kHeadDim;
                const int dimension = channelInPlane - head * kHeadDim;
                const half2* gamma = reinterpret_cast<const half2*>(
                  plane == 0 ? qGamma_ : kGamma_);
                const float epsilon = plane == 0 ? qEpsilon_ : kEpsilon_;
                const float invRms = rsqrtf(
                  sumSquares / static_cast<float>(kHeadDim) + epsilon);
                const int xy = outputRow % kSequence;

                CUTLASS_PRAGMA_UNROLL
                for(int element = 0; element < kElementsPerAccess; element += 2) {
                  const float2 gammaValues = __half22float2(
                    gamma[(dimension + element) / 2]);
                  const half2 normalizedHalf = __floats2half2_rn(
                    static_cast<float>(access[element]) * invRms * gammaValues.x,
                    static_cast<float>(access[element + 1]) * invRms * gammaValues.y);
                  const float2 normalized = __half22float2(normalizedHalf);
                  const half2 ropeHalf = ropeCosSin_[
                    (static_cast<std::size_t>(xy) * kChannels +
                     head * kHeadDim + dimension + element) / 2];
                  const float2 rope = __half22float2(ropeHalf);
                  access[element] = Element(
                    normalized.x * rope.x - normalized.y * rope.y);
                  access[element + 1] = Element(
                    normalized.x * rope.y + normalized.y * rope.x);
                }
              }
            }
          }
        }
      }
    }
    Base::store_with_byte_offset(transformed,byteOffset);
  }

  CUTLASS_DEVICE
  void store(Fragment const& fragment) const {
    store_with_byte_offset(fragment,0);
  }
};

template<int Stages, int SwizzleFactor>
struct FusedQknProjectionBundle {
  using DeviceGemm = ProjectionGemm<Stages,SwizzleFactor>;
  using DefaultKernel = typename DeviceGemm::GemmKernel;
  using Mma = typename DefaultKernel::Mma;
  using DefaultEpilogue = typename DefaultKernel::Epilogue;
  static_assert(DefaultEpilogue::kPartitionsK == 1,
    "QKNorm may only run after the final non-split-K projection reduction");
  using DefaultIterator = typename DefaultEpilogue::OutputTileIterator;
  using OutputIterator = QknormRopeOutputTileIterator<DefaultIterator>;
  using Epilogue = cutlass::epilogue::threadblock::Epilogue<
    typename DefaultEpilogue::Shape,
    typename DefaultEpilogue::WarpMmaOperator,
    DefaultEpilogue::kPartitionsK,
    OutputIterator,
    typename DefaultEpilogue::AccumulatorFragmentIterator,
    typename DefaultEpilogue::WarpTileIterator,
    typename DefaultEpilogue::SharedLoadIterator,
    typename DefaultEpilogue::OutputOp,
    typename DefaultEpilogue::Padding,
    DefaultEpilogue::Base::kFragmentsPerIteration>;
  using Swizzle = typename DeviceGemm::ThreadblockSwizzle;
  using Kernel = cutlass::gemm::kernel::Gemm<Mma,Epilogue,Swizzle,false>;
};

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

template<int Count, bool ExactBranchless>
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
    if constexpr(ExactBranchless) {
      // Factors are in [-127,127], hence |numerator| <= 127*127. Because 127
      // is odd, division can never land exactly halfway between integers:
      // RNE is floor((|x|+63)/127), and its quotient is already <= 127.
      const int signMask = -int(numerator < 0);
      const int magnitude = (numerator ^ signMask) - signMask;
      const int quotient = int((unsigned(magnitude) + 63u) / 127u);
      return (quotient ^ signMask) - signMask;
    }
    else {
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
using Clip7ProductToInt8Incumbent = QuantizedClip7FactorMul<8,false>;
using Clip7ProductToInt8ExactBranchless = QuantizedClip7FactorMul<8,true>;

// General v105 per-layer path. The two projection epilogues preserve the
// current FP32 SiLU/clamp boundary, but take the factor clip and its INT8
// multiplier from the immutable prepared handle. The final epilogue then
// requantizes the integer factor product to the independently calibrated
// productQuantMaxAbs domain.
template<bool ApplySiLU, int Count>
class AdjustableFactorToInt8 {
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
    float clip;
    float quantMultiplier;
    CUTLASS_HOST_DEVICE Params(
      float alphaValue = 1.0f,
      float clipValue = 7.0f,
      float quantMultiplierValue = 127.0f / 7.0f
    ) :
      alpha(alphaValue),
      clip(clipValue),
      quantMultiplier(quantMultiplierValue) {}
  };

private:
  float alpha_;
  float clip_;
  float quantMultiplier_;

public:
  CUTLASS_HOST_DEVICE explicit AdjustableFactorToInt8(Params const& params) :
    alpha_(params.alpha),
    clip_(params.clip),
    quantMultiplier_(params.quantMultiplier) {}

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
      const float clipped = values[i] < -clip_ ? -clip_ :
        (values[i] > clip_ ? clip_ : values[i]);
      const float scaled = clipped * quantMultiplier_;
      values[i] = scaled < -127.0f ? -127.0f :
        (scaled > 127.0f ? 127.0f : scaled);
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
    value = value < -clip_ ? -clip_ : (value > clip_ ? clip_ : value);
    const float scaled = value * quantMultiplier_;
    value = scaled < -127.0f ? -127.0f :
      (scaled > 127.0f ? 127.0f : scaled);
    cutlass::NumericConverter<
      ElementOutput,float,cutlass::FloatRoundStyle::round_to_nearest> convert;
    return convert(value);
  }
};

template<int Count>
class AdjustableFactorMul {
public:
  using ElementOutput = Int8;
  using ElementAccumulator = Int8;
  static int const kCount = Count;
  using FragmentOutput = cutlass::Array<ElementOutput,kCount>;
  using FragmentAccumulator = cutlass::Array<ElementAccumulator,kCount>;

  struct Params {
    float multiplier;
    CUTLASS_HOST_DEVICE explicit Params(float value = 1.0f / 127.0f) :
      multiplier(value) {}
  };

private:
  float multiplier_;

  CUTLASS_HOST_DEVICE int requantize(int product) const {
    float scaled = float(product) * multiplier_;
    scaled = scaled < -127.0f ? -127.0f :
      (scaled > 127.0f ? 127.0f : scaled);
    cutlass::NumericConverter<
      Int8,float,cutlass::FloatRoundStyle::round_to_nearest> convert;
    return int(convert(scaled));
  }

public:
  CUTLASS_HOST_DEVICE explicit AdjustableFactorMul(Params const& params) :
    multiplier_(params.multiplier) {}
  CUTLASS_HOST_DEVICE bool is_source_needed() const { return true; }
  CUTLASS_HOST_DEVICE void set_k_partition(int, int) { assert(false); }

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const& lhs,
    FragmentAccumulator const& rhs
  ) const {
    FragmentOutput output;
    CUTLASS_PRAGMA_UNROLL
    for(int i = 0; i < kCount; i++)
      output[i] = static_cast<Int8>(
        requantize(int(lhs[i]) * int(rhs[i])));
    return output;
  }

  CUTLASS_HOST_DEVICE
  ElementOutput operator()(
    ElementAccumulator const& lhs,
    ElementAccumulator const& rhs
  ) const {
    return static_cast<Int8>(requantize(int(lhs) * int(rhs)));
  }
};

using AdjustableUpFactorToInt8 = AdjustableFactorToInt8<true,8>;
using AdjustableGateFactorToInt8 = AdjustableFactorToInt8<false,8>;
using AdjustableProductToInt8 = AdjustableFactorMul<8>;

// Structural single-GEMM experiment. B is K384xN2048 with columns packed as
// [up(c),gate(c)]. A normal N128 CUTLASS epilogue presents each thread with a
// canonical contiguous 16-column INT32 fragment, so all eight up/gate pairs
// can be transformed and compressed without warp exchange or a second
// accumulator source. The first eight output bytes are the Mx1024 product;
// bytes 8..15 are intentionally ignored by PairCompressOutputTileIterator.
template<int Count>
class InterleavedClip7Product {
public:
  using ElementOutput = Int8;
  using ElementAccumulator = Accum;
  using ElementCompute = float;
  static int const kCount = Count;
  static_assert(Count == 16,
    "interleaved dual requires one 16-column INT8 epilogue access");

  using FragmentOutput = cutlass::Array<ElementOutput,Count>;
  using FragmentAccumulator = cutlass::Array<ElementAccumulator,Count>;
  using FragmentSource = cutlass::Array<ElementOutput,Count>;

  struct Params {
    float upAlpha;
    float gateAlpha;
    float productQuantMultiplier;

    CUTLASS_HOST_DEVICE
    Params(
      float upAlphaValue = 1.0f,
      float gateAlphaValue = 1.0f,
      float productQuantMultiplierValue = 1.0f / 127.0f
    ) :
      upAlpha(upAlphaValue),
      gateAlpha(gateAlphaValue),
      productQuantMultiplier(productQuantMultiplierValue) {}
  };

private:
  float upAlpha_;
  float gateAlpha_;
  float productQuantMultiplier_;

  CUTLASS_HOST_DEVICE
  static float clip7(float value) {
    return value < -7.0f ? -7.0f : (value > 7.0f ? 7.0f : value);
  }

  CUTLASS_HOST_DEVICE
  static Int8 quantizeFactor(float value) {
    value = clip7(value) * (127.0f / 7.0f);
    value = value < -127.0f ? -127.0f :
      (value > 127.0f ? 127.0f : value);
    cutlass::NumericConverter<
      Int8,float,cutlass::FloatRoundStyle::round_to_nearest> convert;
    return convert(value);
  }

public:
  CUTLASS_HOST_DEVICE
  explicit InterleavedClip7Product(Params const& params) :
    upAlpha_(params.upAlpha),
    gateAlpha_(params.gateAlpha),
    productQuantMultiplier_(params.productQuantMultiplier) {}

  CUTLASS_HOST_DEVICE bool is_source_needed() const { return false; }
  CUTLASS_HOST_DEVICE void set_k_partition(int, int) { assert(false); }

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const& accumulator) const {
    FragmentOutput output;
    output.clear();
    cutlass::epilogue::thread::SiLu<float> silu;
    cutlass::NumericConverter<
      Int8,float,cutlass::FloatRoundStyle::round_to_nearest> convert;
    CUTLASS_PRAGMA_UNROLL
    for(int pair = 0; pair < Count / 2; pair++) {
      const Int8 up = quantizeFactor(
        silu(float(accumulator[2 * pair]) * upAlpha_));
      const Int8 gate = quantizeFactor(
        float(accumulator[2 * pair + 1]) * gateAlpha_);
      float product = float(int(up) * int(gate)) * productQuantMultiplier_;
      product = product < -127.0f ? -127.0f :
        (product > 127.0f ? 127.0f : product);
      output[pair] = convert(product);
    }
    return output;
  }

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const& accumulator,
    FragmentSource const&
  ) const {
    return (*this)(accumulator);
  }
};

using InterleavedProductOutputOp = InterleavedClip7Product<16>;

template<typename BaseIterator>
class PairCompressOutputTileIterator : public BaseIterator {
public:
  using Base = BaseIterator;
  using ThreadMap = typename Base::ThreadMap;
  using Element = typename Base::Element;
  using Layout = typename Base::Layout;
  using TensorRef = typename Base::TensorRef;
  using ConstTensorRef = typename Base::ConstTensorRef;
  using TensorCoord = typename Base::TensorCoord;
  using LongIndex = typename Base::LongIndex;
  using Fragment = typename Base::Fragment;
  using AccessType = typename Base::AccessType;
  using Mask = typename Base::Mask;

  static int const kElementsPerAccess = Base::kElementsPerAccess;
  static int const kIterations = Base::kIterations;
  static_assert(kElementsPerAccess == 16,
    "interleaved dual requires 16 adjacent canonical accumulator columns");
  static_assert((ThreadMap::Delta::kColumn % 2) == 0,
    "every epilogue access must begin on an up/gate pair boundary");

  using BaseParams = typename Base::Params;
  struct Params : public BaseParams {
    LongIndex compressedStride;

    CUTLASS_HOST_DEVICE
    Params() : BaseParams(), compressedStride(0) {}

    CUTLASS_HOST_DEVICE
    explicit Params(Layout const& layout) :
      BaseParams(layout), compressedStride(layout.stride(0)) {}
  };

private:
  Element* output_;
  LongIndex compressedStride_;
  int extentRows_;
  int extentColumns_;

public:
  CUTLASS_DEVICE
  PairCompressOutputTileIterator(
    Params const& params,
    Element* pointer,
    TensorCoord extent,
    int threadIdx,
    TensorCoord threadblockOffset = TensorCoord(),
    int const* indices = nullptr
  ) :
    Base(params,pointer,extent,threadIdx,threadblockOffset,indices),
    output_(pointer),
    compressedStride_(params.compressedStride),
    extentRows_(extent.row()),
    extentColumns_(extent.column()) {}

  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const& fragment, int64_t byteOffset) const {
    using CompressedAccess = cutlass::AlignedArray<Element,8>;
    AccessType const* accesses = reinterpret_cast<AccessType const*>(&fragment);
    CUTLASS_PRAGMA_UNROLL
    for(int cluster = 0; cluster < ThreadMap::Iterations::kCluster; cluster++) {
      CUTLASS_PRAGMA_UNROLL
      for(int group = 0; group < ThreadMap::Iterations::kGroup; group++) {
        CUTLASS_PRAGMA_UNROLL
        for(int row = 0; row < ThreadMap::Iterations::kRow; row++) {
          const int fragmentRow = row + ThreadMap::Iterations::kRow *
            (group + ThreadMap::Iterations::kGroup * cluster);
          const int outputRow = Base::thread_start_row() +
            row * ThreadMap::Delta::kRow +
            group * ThreadMap::Delta::kGroup +
            cluster * ThreadMap::Delta::kCluster;
          CUTLASS_PRAGMA_UNROLL
          for(int column = 0; column < ThreadMap::Iterations::kColumn; column++) {
            const int interleavedColumn = Base::thread_start_column() +
              column * ThreadMap::Delta::kColumn;
            const bool guard = output_ != nullptr &&
              outputRow < extentRows_ &&
              interleavedColumn + kElementsPerAccess <= extentColumns_;
            CompressedAccess compressed;
            CUTLASS_PRAGMA_UNROLL
            for(int element = 0; element < 8; element++)
              compressed[element] = accesses[
                fragmentRow * ThreadMap::Iterations::kColumn + column][element];
            Element* destination = output_ +
              LongIndex(outputRow) * compressedStride_ + interleavedColumn / 2;
            cutlass::arch::global_store<CompressedAccess,sizeof(CompressedAccess)>(
              compressed,
              reinterpret_cast<void*>(
                reinterpret_cast<uint8_t*>(destination) + byteOffset),
              guard);
          }
        }
      }
    }
  }

  CUTLASS_DEVICE
  void store(Fragment const& fragment) const {
    store_with_byte_offset(fragment,0);
  }
};

template<int SwizzleFactor>
struct InterleavedDualFfnBundle {
  using DeviceGemm = cutlass::gemm::device::Gemm<
    Int8,LayoutA,
    Int8,LayoutB,
    Int8,LayoutOutput,
    Accum,
    cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128,128,64>,
    cutlass::gemm::GemmShape<64,64,64>,
    cutlass::gemm::GemmShape<16,8,32>,
    InterleavedProductOutputOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleFactor>,
    3,16,16,false,cutlass::arch::OpMultiplyAddSaturate>;
  using DefaultKernel = typename DeviceGemm::GemmKernel;
  using Mma = typename DefaultKernel::Mma;
  using DefaultEpilogue = typename DefaultKernel::Epilogue;
  using DefaultIterator = typename DefaultEpilogue::OutputTileIterator;
  using OutputIterator = PairCompressOutputTileIterator<DefaultIterator>;
  using Epilogue = cutlass::epilogue::threadblock::Epilogue<
    typename DefaultEpilogue::Shape,
    typename DefaultEpilogue::WarpMmaOperator,
    DefaultEpilogue::kPartitionsK,
    OutputIterator,
    typename DefaultEpilogue::AccumulatorFragmentIterator,
    typename DefaultEpilogue::WarpTileIterator,
    typename DefaultEpilogue::SharedLoadIterator,
    typename DefaultEpilogue::OutputOp,
    typename DefaultEpilogue::Padding,
    DefaultEpilogue::Base::kFragmentsPerIteration>;
  using Swizzle = typename DeviceGemm::ThreadblockSwizzle;
  using Kernel = cutlass::gemm::kernel::Gemm<Mma,Epilogue,Swizzle,false>;

  static_assert(OutputIterator::ThreadMap::kThreads == 128,
    "interleaved dual must retain the incumbent 128-thread CTA");
  static_assert(OutputIterator::ThreadMap::Iterations::kColumn == 1,
    "one thread must own each complete 16-column pair group");
  static_assert(OutputIterator::ThreadMap::Shape::kColumn == 128 &&
                OutputIterator::ThreadMap::Delta::kColumn == 128,
    "interleaved dual canonical N128 epilogue map changed");
};

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

template<
  int Stages, int Swizzle,
  typename ProductOutputOp = Clip7ProductToInt8Incumbent
>
using DualFfnInt8Gemm = cutlass::gemm::device::DualGemm<
  Int8,LayoutA,
  Int8,LayoutB,LayoutB,
  Int8,LayoutOutput,
  Accum,
  cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<128,64,64>,
  cutlass::gemm::GemmShape<64,32,64>,
  cutlass::gemm::GemmShape<16,8,32>,
  Clip7UpFactorToInt8,Clip7GateFactorToInt8,ProductOutputOp,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swizzle>,
  Stages,false,false,false,16,16,
  cutlass::arch::OpMultiplyAddSaturate>;

template<int Stages, int Swizzle>
using AdjustableDualFfnInt8Gemm = cutlass::gemm::device::DualGemm<
  Int8,LayoutA,
  Int8,LayoutB,LayoutB,
  Int8,LayoutOutput,
  Accum,
  cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<128,64,64>,
  cutlass::gemm::GemmShape<64,32,64>,
  cutlass::gemm::GemmShape<16,8,32>,
  AdjustableUpFactorToInt8,AdjustableGateFactorToInt8,
  AdjustableProductToInt8,
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
using FusedQknProjectionP3 = FusedQknProjectionBundle<3,2>;
using DualS3Sw1 = DualFfnGemm<3,1>;
using DualS3Sw4 = DualFfnGemm<3,4>;
using DualS4Sw1 = DualFfnGemm<4,1>;
using DualInt8S3Sw1 = DualFfnInt8Gemm<3,1>;
using DualInt8S3Sw4 = DualFfnInt8Gemm<3,4>;
using DualInt8S3Sw4ExactBranchless = DualFfnInt8Gemm<
  3,4,Clip7ProductToInt8ExactBranchless>;
using DualInt8S4Sw1 = DualFfnInt8Gemm<4,1>;
using Clip7AdjustableProductDualInt8S3Sw1 = DualFfnInt8Gemm<
  3,1,AdjustableProductToInt8>;
using Clip7AdjustableProductDualInt8S3Sw4 = DualFfnInt8Gemm<
  3,4,AdjustableProductToInt8>;
using Clip7AdjustableProductDualInt8S4Sw1 = DualFfnInt8Gemm<
  4,1,AdjustableProductToInt8>;
using AdjustableDualInt8S3Sw1 = AdjustableDualFfnInt8Gemm<3,1>;
using AdjustableDualInt8S3Sw4 = AdjustableDualFfnInt8Gemm<3,4>;
using AdjustableDualInt8S4Sw1 = AdjustableDualFfnInt8Gemm<4,1>;
using InterleavedDualInt8S3Sw4 = InterleavedDualFfnBundle<4>;
using ResidualS2Sw1 = ResidualInt8Gemm<2,1>;
using ResidualS3Sw1 = ResidualInt8Gemm<3,1>;
using ResidualS3Sw2 = ResidualInt8Gemm<3,2>;

static_assert(sizeof(typename ProjectionS3Sw1::GemmKernel::SharedStorage) <= 101376,
  "C384 INT8 projection exceeds RTX 5090 opt-in shared memory");
static_assert(sizeof(typename ProjectionS2Sw1::GemmKernel::SharedStorage) <= 101376 &&
              sizeof(typename ProjectionS3Sw2::GemmKernel::SharedStorage) <= 101376,
  "a C384 INT8 projection candidate exceeds RTX 5090 opt-in shared memory");
static_assert(
  sizeof(typename FusedQknProjectionP3::Kernel::SharedStorage) <= 101376,
  "the fused C384 INT8 P3 QKV+QKNorm+RoPE kernel exceeds RTX 5090 opt-in shared memory");
static_assert(sizeof(typename DualS3Sw1::DualGemmKernel::SharedStorage) <= 101376,
  "C384 INT8 dual FFN exceeds RTX 5090 opt-in shared memory");
static_assert(sizeof(typename DualS3Sw4::DualGemmKernel::SharedStorage) <= 101376 &&
              sizeof(typename DualS4Sw1::DualGemmKernel::SharedStorage) <= 101376,
  "a C384 INT8 dual-FFN candidate exceeds RTX 5090 opt-in shared memory");
static_assert(sizeof(typename DualInt8S3Sw1::DualGemmKernel::SharedStorage) <= 101376 &&
              sizeof(typename DualInt8S3Sw4::DualGemmKernel::SharedStorage) <= 101376 &&
              sizeof(typename DualInt8S3Sw4ExactBranchless::DualGemmKernel::SharedStorage) <= 101376 &&
              sizeof(typename DualInt8S4Sw1::DualGemmKernel::SharedStorage) <= 101376,
  "a fused-output C384 INT8 dual-FFN exceeds RTX 5090 opt-in shared memory");
static_assert(
  sizeof(typename Clip7AdjustableProductDualInt8S3Sw1::DualGemmKernel::SharedStorage) <= 101376 &&
  sizeof(typename Clip7AdjustableProductDualInt8S3Sw4::DualGemmKernel::SharedStorage) <= 101376 &&
  sizeof(typename Clip7AdjustableProductDualInt8S4Sw1::DualGemmKernel::SharedStorage) <= 101376,
  "a clip7 adjustable-product C384 INT8 dual-FFN exceeds RTX 5090 opt-in shared memory");
static_assert(
  sizeof(typename AdjustableDualInt8S3Sw1::DualGemmKernel::SharedStorage) <= 101376 &&
  sizeof(typename AdjustableDualInt8S3Sw4::DualGemmKernel::SharedStorage) <= 101376 &&
  sizeof(typename AdjustableDualInt8S4Sw1::DualGemmKernel::SharedStorage) <= 101376,
  "an adjustable-scale C384 INT8 dual-FFN exceeds RTX 5090 opt-in shared memory");
static_assert(
  sizeof(typename InterleavedDualInt8S3Sw4::Kernel::SharedStorage) <= 101376,
  "the interleaved single-GEMM C384 INT8 FFN exceeds RTX 5090 opt-in shared memory");
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
  bool fusedQknormRopePrepared;
};

enum class ProductQuantPath : uint32_t {
  Fp16 = 0,
  Clip7SquaredExact = 1,
  Clip7AdjustableProductFloat = 2,
  AdjustableFloat = 3,
};

struct DualFfnHandle {
  DualFfnConfig config;
  float upAlpha;
  float gateAlpha;
  float factorQuantMultiplier;
  float productQuantMultiplier;
  ProductQuantPath productQuantPath;
  Int8* packedInterleavedWeights;
};

struct DownHandle {
  DownConfig config;
  float alpha;
};

struct AttentionOutHandle {
  AttentionOutConfig config;
  float alpha;
};

template<typename OutputOp>
struct DualEpilogueParamBuilder {
  static typename OutputOp::Params factor(
    float alpha,
    float,
    float
  ) {
    return typename OutputOp::Params(alpha);
  }
  static typename OutputOp::Params product(float) {
    return typename OutputOp::Params();
  }
};

template<bool ApplySiLU, int Count>
struct DualEpilogueParamBuilder<AdjustableFactorToInt8<ApplySiLU,Count>> {
  using OutputOp = AdjustableFactorToInt8<ApplySiLU,Count>;
  static typename OutputOp::Params factor(
    float alpha,
    float clip,
    float quantMultiplier
  ) {
    return typename OutputOp::Params(alpha,clip,quantMultiplier);
  }
  static typename OutputOp::Params product(float) {
    return typename OutputOp::Params();
  }
};

template<int Count>
struct DualEpilogueParamBuilder<AdjustableFactorMul<Count>> {
  using OutputOp = AdjustableFactorMul<Count>;
  static typename OutputOp::Params factor(float, float, float) {
    return typename OutputOp::Params();
  }
  static typename OutputOp::Params product(float multiplier) {
    return typename OutputOp::Params(multiplier);
  }
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

template<typename Bundle>
cudaError_t prepareFusedQknProjectionTyped(const ProjectionHandle& handle) {
  using Kernel = typename Bundle::Kernel;
  cudaError_t status = setDynamicSharedAttribute<Kernel>();
  if(status != cudaSuccess)
    return status;
  typename Kernel::Mma::IteratorA::TensorRef a(
    const_cast<Int8*>(handle.config.packedWeights),LayoutA(kChannels));
  typename Kernel::Mma::IteratorB::TensorRef b(
    const_cast<Int8*>(handle.config.packedWeights),LayoutB(kChannels));
  typename Bundle::OutputIterator::TensorRef c(
    nullptr,LayoutOutput(kQkvChannels));
  typename Bundle::OutputIterator::TensorRef d(
    reinterpret_cast<Output*>(
      const_cast<Int8*>(handle.config.packedWeights)),
    LayoutOutput(kQkvChannels));
  const auto supportsRows = [&](int rows) {
    return Kernel::can_implement(
      {rows,kQkvChannels,kChannels},a,b,c,d) == cutlass::Status::kSuccess;
  };
  return supportsRows(1) && supportsRows(handle.config.maxTokenRows) ?
    cudaSuccess : cudaErrorNotSupported;
}

template<typename Bundle>
cudaError_t launchFusedQknProjectionTyped(
  const ProjectionHandle& handle,
  int rows,
  const Int8* activation,
  Output* output,
  int outputStride,
  const half* qGamma,
  const half* kGamma,
  const half2* ropeCosSin,
  float qEpsilon,
  float kEpsilon,
  cudaStream_t stream
) {
  using Kernel = typename Bundle::Kernel;
  using Swizzle = typename Bundle::Swizzle;
  using Iterator = typename Bundle::OutputIterator;
  const cutlass::gemm::GemmCoord problem(rows,kQkvChannels,kChannels);
  const cutlass::gemm::GemmCoord tiled = Swizzle::get_tiled_shape(
    problem,{Bundle::DeviceGemm::ThreadblockShape::kM,
             Bundle::DeviceGemm::ThreadblockShape::kN,
             Bundle::DeviceGemm::ThreadblockShape::kK},1);
  typename Kernel::Mma::IteratorA::TensorRef a(
    const_cast<Int8*>(activation),LayoutA(kChannels));
  typename Kernel::Mma::IteratorB::TensorRef b(
    const_cast<Int8*>(handle.config.packedWeights),LayoutB(kChannels));
  typename Iterator::TensorRef c(nullptr,LayoutOutput(outputStride));
  typename Iterator::TensorRef d(output,LayoutOutput(outputStride));
  typename Kernel::Params params(
    problem,tiled,a,b,c,d,
    typename DequantToHalf::Params(handle.alpha,0.0f),nullptr);
  params.params_D.qGamma = qGamma;
  params.params_D.kGamma = kGamma;
  params.params_D.ropeCosSin = ropeCosSin;
  params.params_D.totalRows = rows;
  params.params_D.qEpsilon = qEpsilon;
  params.params_D.kEpsilon = kEpsilon;
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
  float gateAlpha,
  float factorClip,
  float factorQuantMultiplier,
  float productQuantMultiplier
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
    DualEpilogueParamBuilder<typename Gemm::EpilogueOutputOp0>::factor(
      upAlpha,factorClip,factorQuantMultiplier),
    DualEpilogueParamBuilder<typename Gemm::EpilogueOutputOp1>::factor(
      gateAlpha,factorClip,factorQuantMultiplier),
    DualEpilogueParamBuilder<typename Gemm::EpilogueOutputOp2>::product(
      productQuantMultiplier),1,
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
    handle.config.packedGateWeights,fakeOutput,handle.upAlpha,handle.gateAlpha,
    handle.config.swigluClip,handle.factorQuantMultiplier,
    handle.productQuantMultiplier);
  const auto last = makeDualArguments<Gemm>(
    handle.config.maxTokenRows,handle.config.packedUpWeights,
    handle.config.packedUpWeights,handle.config.packedGateWeights,
    fakeOutput,handle.upAlpha,handle.gateAlpha,handle.config.swigluClip,
    handle.factorQuantMultiplier,handle.productQuantMultiplier);
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
    DualEpilogueParamBuilder<typename Gemm::EpilogueOutputOp0>::factor(
      handle.upAlpha,handle.config.swigluClip,
      handle.factorQuantMultiplier),
    DualEpilogueParamBuilder<typename Gemm::EpilogueOutputOp1>::factor(
      handle.gateAlpha,handle.config.swigluClip,
      handle.factorQuantMultiplier),
    DualEpilogueParamBuilder<typename Gemm::EpilogueOutputOp2>::product(
      handle.productQuantMultiplier),nullptr);
  constexpr int threads = Kernel::kThreadCount;
  constexpr int sharedBytes = int(sizeof(typename Kernel::SharedStorage));
  cutlass::Kernel<Kernel><<<
    Swizzle::get_grid_shape(tiled),dim3(threads,1,1),sharedBytes,stream>>>(params);
  return cudaPeekAtLastError();
}

template<typename Bundle>
cudaError_t prepareInterleavedDualTyped(const DualFfnHandle& handle) {
  using Kernel = typename Bundle::Kernel;
  cudaError_t status = setDynamicSharedAttribute<Kernel>();
  if(status != cudaSuccess)
    return status;
  typename Kernel::Mma::IteratorA::TensorRef a(
    handle.packedInterleavedWeights,LayoutA(kChannels));
  typename Kernel::Mma::IteratorB::TensorRef b(
    handle.packedInterleavedWeights,LayoutB(kChannels));
  typename Bundle::OutputIterator::TensorRef c(
    nullptr,LayoutOutput(kFfnChannels));
  typename Bundle::OutputIterator::TensorRef d(
    handle.packedInterleavedWeights,LayoutOutput(kFfnChannels));
  const auto supportsRows = [&](int rows) {
    return Kernel::can_implement(
      {rows,2 * kFfnChannels,kChannels},a,b,c,d) ==
      cutlass::Status::kSuccess;
  };
  return supportsRows(1) && supportsRows(handle.config.maxTokenRows) ?
    cudaSuccess : cudaErrorNotSupported;
}

template<typename Bundle>
cudaError_t launchInterleavedDualTyped(
  const DualFfnHandle& handle,
  int rows,
  const Int8* activation,
  Int8* output,
  cudaStream_t stream
) {
  using Kernel = typename Bundle::Kernel;
  using Swizzle = typename Bundle::Swizzle;
  using Iterator = typename Bundle::OutputIterator;
  const cutlass::gemm::GemmCoord problem(
    rows,2 * kFfnChannels,kChannels);
  const cutlass::gemm::GemmCoord tiled = Swizzle::get_tiled_shape(
    problem,{Bundle::DeviceGemm::ThreadblockShape::kM,
             Bundle::DeviceGemm::ThreadblockShape::kN,
             Bundle::DeviceGemm::ThreadblockShape::kK},1);
  typename Kernel::Mma::IteratorA::TensorRef a(
    const_cast<Int8*>(activation),LayoutA(kChannels));
  typename Kernel::Mma::IteratorB::TensorRef b(
    handle.packedInterleavedWeights,LayoutB(kChannels));
  typename Iterator::TensorRef c(nullptr,LayoutOutput(kFfnChannels));
  typename Iterator::TensorRef d(output,LayoutOutput(kFfnChannels));
  typename Kernel::Params params(
    problem,tiled,a,b,c,d,
    typename InterleavedProductOutputOp::Params(
      handle.upAlpha,handle.gateAlpha,handle.productQuantMultiplier),
    nullptr);
  constexpr int threads = Kernel::kThreadCount;
  constexpr int sharedBytes = int(sizeof(typename Kernel::SharedStorage));
  cutlass::Kernel<Kernel><<<
    Swizzle::get_grid_shape(tiled),dim3(threads,1,1),sharedBytes,stream>>>(params);
  return cudaPeekAtLastError();
}

__global__ void packInterleavedDualWeightsKernel(
  const uint4* __restrict__ up,
  const uint4* __restrict__ gate,
  uint4* __restrict__ interleaved
) {
  constexpr int vectorsPerColumn = kChannels * int(sizeof(Int8)) /
    int(sizeof(uint4));
  constexpr int totalVectors = kFfnChannels * vectorsPerColumn;
  for(int index = int(blockIdx.x) * blockDim.x + threadIdx.x;
      index < totalVectors;
      index += int(gridDim.x) * blockDim.x) {
    const int column = index / vectorsPerColumn;
    const int vector = index - column * vectorsPerColumn;
    interleaved[(2 * column) * vectorsPerColumn + vector] = up[index];
    interleaved[(2 * column + 1) * vectorsPerColumn + vector] = gate[index];
  }
}

cudaError_t packInterleavedDualWeights(DualFfnHandle& handle) {
  constexpr std::size_t bytes =
    std::size_t(2) * kFfnChannels * kChannels * sizeof(Int8);
  void* allocation = nullptr;
  cudaError_t status = cudaMalloc(&allocation,bytes);
  if(status != cudaSuccess)
    return status;
  handle.packedInterleavedWeights = static_cast<Int8*>(allocation);
  constexpr int threads = 256;
  constexpr int vectors = kFfnChannels * kChannels * int(sizeof(Int8)) /
    int(sizeof(uint4));
  packInterleavedDualWeightsKernel<<<(vectors + threads - 1) / threads,threads>>>(
    reinterpret_cast<const uint4*>(handle.config.packedUpWeights),
    reinterpret_cast<const uint4*>(handle.config.packedGateWeights),
    reinterpret_cast<uint4*>(handle.packedInterleavedWeights));
  status = cudaPeekAtLastError();
  if(status == cudaSuccess)
    status = cudaDeviceSynchronize();
  if(status != cudaSuccess) {
    (void)cudaFree(handle.packedInterleavedWeights);
    handle.packedInterleavedWeights = nullptr;
  }
  return status;
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
    kNormActivationScale * config.weightScale,false};
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
  if(config.mode == ProjectionMode::AggressiveQkv) {
    status = config.tactic == ProjectionTactic::M128N128K64S3Sw2 ?
      prepareFusedQknProjectionTyped<FusedQknProjectionP3>(handle) :
      cudaErrorNotSupported;
    handle.fusedQknormRopePrepared = status == cudaSuccess;
    if(status != cudaSuccess) {
      (void)cudaGetLastError();
      return nullptr;
    }
  }
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

bool projectionQknormRopeSupports(
  const void* opaque,
  int tokenRows,
  int inputChannels,
  int outputRowStride,
  float qEpsilon,
  float kEpsilon
) noexcept {
  const ProjectionHandle* handle = static_cast<const ProjectionHandle*>(opaque);
  return handle != nullptr && handle->fusedQknormRopePrepared &&
    handle->config.mode == ProjectionMode::AggressiveQkv &&
    tokenRows == kTokenRows && inputChannels == kChannels &&
    outputRowStride == kQkvChannels && qEpsilon == kRmsEpsilon &&
    kEpsilon == kRmsEpsilon;
}

const char* projectionQknormRopeMarker() noexcept {
  return "c384-int8-qkv-qknorm-rope-epilogue-h12-d32-half-boundary-v1";
}

cudaError_t launchProjectionQknormRope(
  void* opaque,
  int tokenRows,
  const int8_t* activation,
  half* packedQkv,
  int outputRowStride,
  const half* qGamma,
  const half* kGamma,
  const half2* learnedRopeCosSin,
  float qEpsilon,
  float kEpsilon,
  cudaStream_t stream
) {
  ProjectionHandle* handle = static_cast<ProjectionHandle*>(opaque);
  if(handle == nullptr || activation == nullptr || packedQkv == nullptr ||
     qGamma == nullptr || kGamma == nullptr || learnedRopeCosSin == nullptr ||
     !projectionQknormRopeSupports(
       handle,tokenRows,kChannels,outputRowStride,qEpsilon,kEpsilon) ||
     !aligned16(activation) || !aligned16(packedQkv) ||
     !aligned16(qGamma) || !aligned16(kGamma) ||
     !aligned16(learnedRopeCosSin))
    return cudaErrorInvalidValue;
  Output* output = reinterpret_cast<Output*>(packedQkv);
  if(handle->config.tactic == ProjectionTactic::M128N128K64S3Sw2)
    return launchFusedQknProjectionTyped<FusedQknProjectionP3>(
      *handle,tokenRows,activation,output,outputRowStride,qGamma,kGamma,
      learnedRopeCosSin,qEpsilon,kEpsilon,stream);
  return cudaErrorNotSupported;
}

void* createDualFfn(const DualFfnConfig& config) {
  const bool autoDivide =
    config.divide127Tactic == DualFfnDivide127Tactic::Auto;
  const bool incumbentDivide =
    config.divide127Tactic == DualFfnDivide127Tactic::Incumbent;
  const bool exactBranchlessDivide =
    config.divide127Tactic == DualFfnDivide127Tactic::ExactBranchless;
  const bool autoProductPath =
    config.productPathTactic == DualFfnProductPathTactic::Auto;
  const bool forceFullyAdjustable = config.productPathTactic ==
    DualFfnProductPathTactic::FullyAdjustableFloat;
  const bool interleavedTactic = config.tactic ==
    DualFfnTactic::M128N128K64S3Sw4Interleaved;
  const bool knownTactic = interleavedTactic ||
    config.tactic == DualFfnTactic::M128N64K64S3Sw1 ||
    config.tactic == DualFfnTactic::M128N64K64S3Sw4 ||
    config.tactic == DualFfnTactic::M128N64K64S4Sw1;
  if(config.maxTokenRows <= 0 || config.maxTokenRows > kMaxTokenRows ||
     config.packedUpWeights == nullptr || config.packedGateWeights == nullptr ||
     !aligned16(config.packedUpWeights) || !aligned16(config.packedGateWeights) ||
     !finitePositive(config.upWeightScale) ||
     !finitePositive(config.gateWeightScale) ||
     !finitePositive(config.swigluClip) ||
     !finitePositive(config.productQuantMaxAbs) ||
     (config.outputMode != DualFfnOutputMode::Fp16Product &&
      config.outputMode != DualFfnOutputMode::Int8Product) ||
     (!autoDivide && !incumbentDivide && !exactBranchlessDivide) ||
     (!autoProductPath && !forceFullyAdjustable) ||
     !knownTactic ||
     (interleavedTactic &&
      (config.outputMode != DualFfnOutputMode::Int8Product ||
       !incumbentDivide || !autoProductPath || config.swigluClip != 7.0f)) ||
     (forceFullyAdjustable &&
      (config.outputMode != DualFfnOutputMode::Int8Product ||
       exactBranchlessDivide)) ||
     (exactBranchlessDivide &&
      (config.outputMode != DualFfnOutputMode::Int8Product ||
       config.tactic != DualFfnTactic::M128N64K64S3Sw4 ||
       config.swigluClip != 7.0f ||
       config.productQuantMaxAbs != 49.0f ||
       !autoProductPath)) ||
     (config.outputMode == DualFfnOutputMode::Fp16Product &&
      (config.swigluClip != 7.0f || !autoProductPath)) ||
     !isSm120Compatible())
    return nullptr;
  const bool exactBranchlessEligible =
    config.outputMode == DualFfnOutputMode::Int8Product &&
    config.tactic == DualFfnTactic::M128N64K64S3Sw4 &&
    config.swigluClip == 7.0f &&
    config.productQuantMaxAbs == 49.0f &&
    autoProductPath;
  DualFfnConfig resolvedConfig = config;
  if(autoDivide)
    resolvedConfig.divide127Tactic = exactBranchlessEligible ?
      DualFfnDivide127Tactic::ExactBranchless :
      DualFfnDivide127Tactic::Incumbent;
  const bool useExactBranchless = resolvedConfig.divide127Tactic ==
    DualFfnDivide127Tactic::ExactBranchless;
  const ProductQuantPath productQuantPath =
    config.outputMode == DualFfnOutputMode::Fp16Product ?
      ProductQuantPath::Fp16 :
    forceFullyAdjustable ? ProductQuantPath::AdjustableFloat :
    (config.swigluClip == 7.0f && config.productQuantMaxAbs == 49.0f ?
      ProductQuantPath::Clip7SquaredExact :
     config.swigluClip == 7.0f ?
      ProductQuantPath::Clip7AdjustableProductFloat :
      ProductQuantPath::AdjustableFloat);
  const float factorQuantMultiplier = 127.0f / config.swigluClip;
  const double productQuantMultiplierDouble =
    double(config.swigluClip) * double(config.swigluClip) /
    (127.0 * double(config.productQuantMaxAbs));
  const float productQuantMultiplier = float(productQuantMultiplierDouble);
  if(!finitePositive(factorQuantMultiplier) ||
     !finitePositive(productQuantMultiplier))
    return nullptr;
  DualFfnHandle handle{resolvedConfig,
    kNormActivationScale * config.upWeightScale,
    kNormActivationScale * config.gateWeightScale,
    factorQuantMultiplier,productQuantMultiplier,productQuantPath,nullptr};
  cudaError_t status = cudaErrorNotSupported;
  if(interleavedTactic) {
    status = packInterleavedDualWeights(handle);
    if(status == cudaSuccess)
      status = prepareInterleavedDualTyped<InterleavedDualInt8S3Sw4>(handle);
  }
  else if(config.outputMode == DualFfnOutputMode::Fp16Product) {
    switch(config.tactic) {
    case DualFfnTactic::M128N64K64S3Sw1:
      status = prepareDualTyped<DualS3Sw1>(handle); break;
    case DualFfnTactic::M128N64K64S3Sw4:
      status = prepareDualTyped<DualS3Sw4>(handle); break;
    case DualFfnTactic::M128N64K64S4Sw1:
      status = prepareDualTyped<DualS4Sw1>(handle); break;
    }
  }
  else if(productQuantPath == ProductQuantPath::Clip7SquaredExact) {
    switch(config.tactic) {
    case DualFfnTactic::M128N64K64S3Sw1:
      status = prepareDualTyped<DualInt8S3Sw1>(handle); break;
    case DualFfnTactic::M128N64K64S3Sw4:
      status = useExactBranchless ?
        prepareDualTyped<DualInt8S3Sw4ExactBranchless>(handle) :
        prepareDualTyped<DualInt8S3Sw4>(handle);
      break;
    case DualFfnTactic::M128N64K64S4Sw1:
      status = prepareDualTyped<DualInt8S4Sw1>(handle); break;
    }
  }
  else if(productQuantPath ==
          ProductQuantPath::Clip7AdjustableProductFloat) {
    switch(config.tactic) {
    case DualFfnTactic::M128N64K64S3Sw1:
      status = prepareDualTyped<Clip7AdjustableProductDualInt8S3Sw1>(
        handle); break;
    case DualFfnTactic::M128N64K64S3Sw4:
      status = prepareDualTyped<Clip7AdjustableProductDualInt8S3Sw4>(
        handle); break;
    case DualFfnTactic::M128N64K64S4Sw1:
      status = prepareDualTyped<Clip7AdjustableProductDualInt8S4Sw1>(
        handle); break;
    }
  }
  else {
    switch(config.tactic) {
    case DualFfnTactic::M128N64K64S3Sw1:
      status = prepareDualTyped<AdjustableDualInt8S3Sw1>(handle); break;
    case DualFfnTactic::M128N64K64S3Sw4:
      status = prepareDualTyped<AdjustableDualInt8S3Sw4>(handle); break;
    case DualFfnTactic::M128N64K64S4Sw1:
      status = prepareDualTyped<AdjustableDualInt8S4Sw1>(handle); break;
    }
  }
  if(status != cudaSuccess) {
    if(handle.packedInterleavedWeights != nullptr)
      (void)cudaFree(handle.packedInterleavedWeights);
    return nullptr;
  }
  DualFfnHandle* result = new(std::nothrow) DualFfnHandle(handle);
  if(result == nullptr && handle.packedInterleavedWeights != nullptr)
    (void)cudaFree(handle.packedInterleavedWeights);
  return result;
}

void destroyDualFfn(void* opaque) noexcept {
  DualFfnHandle* handle = static_cast<DualFfnHandle*>(opaque);
  if(handle != nullptr && handle->packedInterleavedWeights != nullptr)
    (void)cudaFree(handle->packedInterleavedWeights);
  delete handle;
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
  case DualFfnTactic::M128N128K64S3Sw4Interleaved:
    return cudaErrorNotSupported;
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
  if(handle->config.tactic ==
     DualFfnTactic::M128N128K64S3Sw4Interleaved) {
    if(handle->packedInterleavedWeights == nullptr ||
       handle->config.swigluClip != 7.0f ||
       handle->config.divide127Tactic != DualFfnDivide127Tactic::Incumbent ||
       handle->config.productPathTactic != DualFfnProductPathTactic::Auto)
      return cudaErrorNotSupported;
    return launchInterleavedDualTyped<InterleavedDualInt8S3Sw4>(
      *handle,tokenRows,activation,productInt8,stream);
  }
  if(handle->productQuantPath ==
     ProductQuantPath::Clip7AdjustableProductFloat) {
    switch(handle->config.tactic) {
    case DualFfnTactic::M128N64K64S3Sw1:
      return launchDualTyped<Clip7AdjustableProductDualInt8S3Sw1>(
        *handle,tokenRows,activation,productInt8,stream);
    case DualFfnTactic::M128N64K64S3Sw4:
      return launchDualTyped<Clip7AdjustableProductDualInt8S3Sw4>(
        *handle,tokenRows,activation,productInt8,stream);
    case DualFfnTactic::M128N64K64S4Sw1:
      return launchDualTyped<Clip7AdjustableProductDualInt8S4Sw1>(
        *handle,tokenRows,activation,productInt8,stream);
    }
    return cudaErrorNotSupported;
  }
  if(handle->productQuantPath == ProductQuantPath::AdjustableFloat) {
    switch(handle->config.tactic) {
    case DualFfnTactic::M128N64K64S3Sw1:
      return launchDualTyped<AdjustableDualInt8S3Sw1>(
        *handle,tokenRows,activation,productInt8,stream);
    case DualFfnTactic::M128N64K64S3Sw4:
      return launchDualTyped<AdjustableDualInt8S3Sw4>(
        *handle,tokenRows,activation,productInt8,stream);
    case DualFfnTactic::M128N64K64S4Sw1:
      return launchDualTyped<AdjustableDualInt8S4Sw1>(
        *handle,tokenRows,activation,productInt8,stream);
    }
    return cudaErrorNotSupported;
  }
  if(handle->productQuantPath != ProductQuantPath::Clip7SquaredExact)
    return cudaErrorNotSupported;
  switch(handle->config.tactic) {
  case DualFfnTactic::M128N64K64S3Sw1:
    return launchDualTyped<DualInt8S3Sw1>(
      *handle,tokenRows,activation,productInt8,stream);
  case DualFfnTactic::M128N64K64S3Sw4:
    if(handle->config.divide127Tactic ==
       DualFfnDivide127Tactic::ExactBranchless)
      return launchDualTyped<DualInt8S3Sw4ExactBranchless>(
        *handle,tokenRows,activation,productInt8,stream);
    return launchDualTyped<DualInt8S3Sw4>(
      *handle,tokenRows,activation,productInt8,stream);
  case DualFfnTactic::M128N64K64S4Sw1:
    return launchDualTyped<DualInt8S4Sw1>(
      *handle,tokenRows,activation,productInt8,stream);
  }
  return cudaErrorNotSupported;
}

const char* dualFfnProductQuantPath(const void* opaque) noexcept {
  const DualFfnHandle* handle = static_cast<const DualFfnHandle*>(opaque);
  if(handle == nullptr)
    return "invalid";
  switch(handle->productQuantPath) {
  case ProductQuantPath::Fp16:
    return "fp16";
  case ProductQuantPath::Clip7SquaredExact:
    return "clip-squared-exact-int-v1";
  case ProductQuantPath::Clip7AdjustableProductFloat:
    return "clip7-fixed-factor-v105-product-float-rne-v1";
  case ProductQuantPath::AdjustableFloat:
    return "v105-per-layer-float-rne-v1";
  }
  return "invalid";
}

const char* dualFfnDivide127Path(const void* opaque) noexcept {
  const DualFfnHandle* handle = static_cast<const DualFfnHandle*>(opaque);
  if(handle == nullptr)
    return "invalid";
  switch(handle->config.divide127Tactic) {
  case DualFfnDivide127Tactic::Auto:
    return "invalid-unresolved-auto";
  case DualFfnDivide127Tactic::Incumbent:
    return "incumbent";
  case DualFfnDivide127Tactic::ExactBranchless:
    return "exact-branchless";
  }
  return "invalid";
}

bool dualFfnDownProductQuantizationMatches(
  const void* dualOpaque,
  const void* downOpaque
) noexcept {
  const DualFfnHandle* dual = static_cast<const DualFfnHandle*>(dualOpaque);
  const DownHandle* down = static_cast<const DownHandle*>(downOpaque);
  return dual != nullptr && down != nullptr &&
    dual->config.outputMode == DualFfnOutputMode::Int8Product &&
    dual->config.productQuantMaxAbs == down->config.productQuantMaxAbs;
}

bool interleavedDualFfnKernelResources(
  const void* opaque,
  InterleavedDualFfnKernelResources& resources
) noexcept {
  resources = InterleavedDualFfnKernelResources{};
  const DualFfnHandle* handle = static_cast<const DualFfnHandle*>(opaque);
  if(handle == nullptr || handle->config.tactic !=
       DualFfnTactic::M128N128K64S3Sw4Interleaved ||
     handle->packedInterleavedWeights == nullptr)
    return false;
  using Kernel = typename InterleavedDualInt8S3Sw4::Kernel;
  cudaFuncAttributes attributes{};
  if(cudaFuncGetAttributes(&attributes,cutlass::Kernel<Kernel>) != cudaSuccess)
    return false;
  resources.registersPerThread = attributes.numRegs;
  resources.staticSharedBytes = int(attributes.sharedSizeBytes);
  resources.dynamicSharedBytes = int(sizeof(typename Kernel::SharedStorage));
  resources.threadsPerBlock = Kernel::kThreadCount;
  return true;
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
     !finitePositive(config.weightScale) ||
     !finitePositive(config.productQuantMaxAbs) || !isSm120Compatible())
    return nullptr;
  const float alpha =
    (config.productQuantMaxAbs / 127.0f) * config.weightScale;
  if(!finitePositive(alpha))
    return nullptr;
  DownHandle handle{config,alpha};
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
  case DualFfnTactic::M128N128K64S3Sw4Interleaved:
    return "int8-interleaved-clip7-m128n128k64-s3-sw4";
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

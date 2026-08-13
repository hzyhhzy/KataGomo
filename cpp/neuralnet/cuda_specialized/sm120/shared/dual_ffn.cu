#include "dual_ffn.h"

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "device/dual_gemm.h"
#include "thread/left_silu_and_mul.h"

#include <cstdint>
#include <memory>
#include <new>
#include <type_traits>

namespace {

constexpr int InputChannels = 256;
constexpr int FfnChannels = 768;
constexpr int C384InputChannels = 384;
constexpr int C384FfnChannels = 1024;
constexpr int MaxTokenRows = 1 << 20;

constexpr bool tokenRowsSupported(
  bool dynamicTokens, int configuredTokens, int actualTokens) {
  return configuredTokens > 0 && actualTokens > 0 &&
    (dynamicTokens ? actualTokens <= configuredTokens :
                     actualTokens == configuredTokens);
}
static_assert(tokenRowsSupported(true,8100,1),"dynamic M accepts one tail row");
static_assert(tokenRowsSupported(true,8100,226),"dynamic M accepts non-board tail");
static_assert(tokenRowsSupported(true,8100,8099),"dynamic M accepts max-minus-one");
static_assert(!tokenRowsSupported(true,8100,8101),"dynamic M rejects above max");
static_assert(tokenRowsSupported(false,8100,8100),"exact M remains exact");
static_assert(!tokenRowsSupported(false,8100,8099),"exact M rejects partial rows");

using Element = cutlass::half_t;
using ProjectionOutput = cutlass::epilogue::thread::LinearCombination<
  Element, 8, Element, float, cutlass::epilogue::thread::ScaleType::Nothing>;
using SwiGLU = cutlass::epilogue::thread::LeftSiLUAndMul<
  Element, 8, Element, float>;

// CUTLASS' stock dual-GEMM epilogue computes SiLU(lhs)*rhs. The QKNorm+clip7
// model uses the same two GEMMs but clips the two operands independently
// before multiplication. Keeping this operation in the register epilogue
// avoids materializing either projection or adding a conversion launch.
template <
  typename ElementOutput_, int Count,
  typename ElementAccumulator_ = ElementOutput_,
  typename ElementCompute_ = ElementOutput_,
  cutlass::FloatRoundStyle Round = cutlass::FloatRoundStyle::round_to_nearest
>
class LeftClippedSiLUAndMul {
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
  LeftClippedSiLUAndMul(Params const&) {}

  CUTLASS_HOST_DEVICE bool is_source_needed() const { return true; }

  CUTLASS_HOST_DEVICE
  void set_k_partition(int, int) { assert(false); }

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

using ClippedSwiGLU = LeftClippedSiLUAndMul<Element,8,Element,float>;
using InstructionShape = cutlass::gemm::GemmShape<16,8,16>;

template<
  int ThreadblockM, int ThreadblockN, int ThreadblockK,
  int WarpM, int WarpN, int WarpK,
  int Stages, int Swizzle,
  typename FinalEpilogue = SwiGLU>
using DualGemm = cutlass::gemm::device::DualGemm<
  Element, cutlass::layout::RowMajor,
  Element, cutlass::layout::RowMajor,
  cutlass::layout::RowMajor,
  Element, cutlass::layout::RowMajor,
  Element,
  cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<ThreadblockM,ThreadblockN,ThreadblockK>,
  cutlass::gemm::GemmShape<WarpM,WarpN,WarpK>,
  InstructionShape,
  ProjectionOutput, ProjectionOutput, FinalEpilogue,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swizzle>,
  Stages, false, false, false, 8, 8>;

using Winner = DualGemm<128,64,32,64,32,32,3,4>;
using Clip7Winner = DualGemm<
  128,64,32,64,32,32,3,4,ClippedSwiGLU>;

constexpr KatagoRenju15DualFfnSm120Descriptor Descriptors[] = {
  {
    KATAGO_RENJU15_DUAL_FFN_M128_N64_K32_S3_SW4,
    "dual-ffn-c256-f768-m128-n64-k32-s3-sw4",
    128,64,32,64,32,32,3,4,InputChannels,FfnChannels,
  },
  {
    KATAGO_RENJU15_DUAL_FFN_C384_F1024_M128_N64_K32_S3_SW4,
    "dual-ffn-c384-f1024-m128-n64-k32-s3-sw4",
    128,64,32,64,32,32,3,4,C384InputChannels,C384FfnChannels,
  },
  {
    KATAGO_RENJU15_DUAL_FFN_C384_F1024_CLIP7_M128_N64_K32_S3_SW4,
    "dual-ffn-c384-f1024-clip7-m128-n64-k32-s3-sw4",
    128,64,32,64,32,32,3,4,C384InputChannels,C384FfnChannels,
  },
};

template<typename Gemm>
typename Gemm::Arguments makeArguments(
  const half* input,
  const half* linearWeights,
  const half* gateWeights,
  half* output,
  int inputChannels,
  int ffnChannels,
  int tokens) {
  using Layout = cutlass::layout::RowMajor;
  typename Gemm::TensorRefC nullC;
  typename Gemm::TensorRefD nullD;
  return {
    cutlass::gemm::DualGemmMode::kGemm,
    {tokens, ffnChannels, inputChannels},
    {reinterpret_cast<const Element*>(input), Layout(inputChannels)},
    {reinterpret_cast<const Element*>(linearWeights), Layout(ffnChannels)},
    nullC, nullD,
    {reinterpret_cast<const Element*>(gateWeights), Layout(ffnChannels)},
    nullC, nullD,
    {reinterpret_cast<Element*>(output), Layout(ffnChannels)},
    {1.0f, 0.0f}, {1.0f, 0.0f}, {}, 1,
  };
}

struct StateBase {
  virtual ~StateBase() = default;
  virtual cudaError_t preparationStatus(int, int, int) { return cudaSuccess; }
  virtual cudaError_t launch(
    const half* input,
    const half* linearWeights,
    const half* gateWeights,
    half* output,
    int inputChannels,
    int ffnChannels,
    int tokens,
    cudaStream_t stream) = 0;
};

cudaError_t cutlassStatus(cutlass::Status status) {
  return status == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorUnknown;
}

template<typename Gemm>
struct State final : StateBase {
  Gemm op;
  bool initialized = false;

  cudaError_t launch(
    const half* input,
    const half* linearWeights,
    const half* gateWeights,
    half* output,
    int inputChannels,
    int ffnChannels,
    int tokens,
    cudaStream_t stream) override {
    typename Gemm::Arguments args = makeArguments<Gemm>(
      input, linearWeights, gateWeights, output,
      inputChannels, ffnChannels, tokens);
    cutlass::Status status;
    if(!initialized) {
      status = op.can_implement(args);
      if(status != cutlass::Status::kSuccess)
        return cutlassStatus(status);
      status = op.initialize(args, nullptr, stream);
      if(status != cutlass::Status::kSuccess)
        return cutlassStatus(status);
      initialized = true;
    }
    else {
      status = op.update(args, nullptr);
      if(status != cutlass::Status::kSuccess)
        return cutlassStatus(status);
    }
    status = op.run(stream);
    if(status != cutlass::Status::kSuccess)
      return cutlassStatus(status);
    return cudaPeekAtLastError();
  }
};

// DualGemm::update has the same pointer-only contract as CUTLASS Gemm::update.
// Rebuild the kernel Params and grid on the stack for every actual M. This is
// allocation-free, has no first-seen-M cold path, and supports arbitrary tail
// rows. C256 keeps using the original single exact-M state.
template<typename Gemm>
struct DynamicState final : StateBase {
  using Kernel = typename Gemm::DualGemmKernel;

  DynamicState(): attributeStatus(cudaSuccess) {}

  cudaError_t preparationStatus(
    int inputChannels, int ffnChannels, int maximumTokens) override {
    static_assert(std::is_trivially_copyable<typename Kernel::Params>::value,
      "dynamic dual GEMM launch passes kernel Params by value");
    // Configure the function in the current device context for each prepared
    // handle; a process-global template static would miss later GPUs.
    const int sharedBytes = int(sizeof(typename Kernel::SharedStorage));
    attributeStatus = sharedBytes < (48 << 10) ? cudaSuccess :
      cudaFuncSetAttribute(
        cutlass::Kernel<Kernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        sharedBytes);
    if(attributeStatus != cudaSuccess)
      return attributeStatus;

    // K/N and pointer-alignment qualification is invariant across actual M.
    // Check both M boundaries at construction, never on the launch hot path.
    const half* alignedInput = reinterpret_cast<const half*>(0x1000);
    const half* alignedLinear = reinterpret_cast<const half*>(0x2000);
    const half* alignedGate = reinterpret_cast<const half*>(0x3000);
    half* alignedOutput = reinterpret_cast<half*>(0x4000);
    const int boundaryTokens[2] = {1,maximumTokens};
    for(int tokens: boundaryTokens) {
      typename Gemm::Arguments args = makeArguments<Gemm>(
        alignedInput,alignedLinear,alignedGate,alignedOutput,
        inputChannels,ffnChannels,tokens);
      if(Gemm::can_implement(args) != cutlass::Status::kSuccess) {
        attributeStatus = cudaErrorInvalidValue;
        break;
      }
    }
    return attributeStatus;
  }

  cudaError_t launch(
    const half* input,
    const half* linearWeights,
    const half* gateWeights,
    half* output,
    int inputChannels,
    int ffnChannels,
    int tokens,
    cudaStream_t stream) override {
    if(attributeStatus != cudaSuccess)
      return attributeStatus;
    typename Gemm::Arguments args = makeArguments<Gemm>(
      input, linearWeights, gateWeights, output,
      inputChannels, ffnChannels, tokens);

    using ThreadblockShape = typename Gemm::ThreadblockShape;
    typename Gemm::ThreadblockSwizzle swizzle;
    const cutlass::gemm::GemmCoord gridShape = swizzle.get_tiled_shape(
      args.problem_size,
      {ThreadblockShape::kM,ThreadblockShape::kN,ThreadblockShape::kK},
      args.mode == cutlass::gemm::DualGemmMode::kBatched ?
        args.batch_count : args.split_k_slices);
    typename Kernel::Params params{
      args.mode,
      args.problem_size,
      gridShape,
      args.ref_A0.non_const_ref(),
      args.ref_B0.non_const_ref(),
      args.ref_C0.non_const_ref(),
      args.ref_D0,
      args.ref_B1.non_const_ref(),
      args.ref_C1.non_const_ref(),
      args.ref_D1,
      args.ref_D2,
      args.epilogue0,
      args.epilogue1,
      args.epilogue2,
      nullptr,
      args.batch_stride_A,
      args.batch_stride_B0,
      args.batch_stride_B1,
      args.batch_stride_C,
      args.batch_stride_D,
    };
    const dim3 grid = swizzle.get_grid_shape(gridShape);
    const dim3 block(Kernel::kThreadCount,1,1);
    const int sharedBytes = int(sizeof(typename Kernel::SharedStorage));
    cutlass::Kernel<Kernel><<<grid,block,sharedBytes,stream>>>(params);
    return cudaPeekAtLastError();
  }

private:
  cudaError_t attributeStatus;
};

struct Handle {
  int tactic;
  int configuredTokens;
  int inputChannels;
  int ffnChannels;
  bool dynamicTokens;
  const KatagoRenju15DualFfnSm120Descriptor* descriptor;
  std::unique_ptr<StateBase> state;
};

const KatagoRenju15DualFfnSm120Descriptor* descriptorForTactic(int tactic) {
  for(const auto& value : Descriptors) {
    if(value.tactic == tactic)
      return &value;
  }
  return nullptr;
}

std::unique_ptr<StateBase> stateForTactic(int tactic) {
  switch(tactic) {
  case KATAGO_RENJU15_DUAL_FFN_M128_N64_K32_S3_SW4:
    return std::unique_ptr<StateBase>(new State<Winner>());
  case KATAGO_RENJU15_DUAL_FFN_C384_F1024_M128_N64_K32_S3_SW4:
    return std::unique_ptr<StateBase>(new DynamicState<Winner>());
  case KATAGO_RENJU15_DUAL_FFN_C384_F1024_CLIP7_M128_N64_K32_S3_SW4:
    return std::unique_ptr<StateBase>(new DynamicState<Clip7Winner>());
  default:
    return nullptr;
  }
}

bool isSm120Compatible() {
  int device = -1;
  cudaDeviceProp prop{};
  if(cudaGetDevice(&device) != cudaSuccess ||
     cudaGetDeviceProperties(&prop, device) != cudaSuccess)
    return false;
  return prop.major == 12 && prop.minor == 0;
}

bool aligned16(const void* ptr) {
  return (reinterpret_cast<std::uintptr_t>(ptr) & 15U) == 0;
}

} // namespace

extern "C" void* katago_renju15_dual_ffn_sm120_create(
  int tactic,
  int configuredTokenRows) {
  if(configuredTokenRows <= 0 || configuredTokenRows > MaxTokenRows ||
     !isSm120Compatible())
    return nullptr;
  const auto* descriptor = descriptorForTactic(tactic);
  if(descriptor == nullptr)
    return nullptr;
  std::unique_ptr<StateBase> state = stateForTactic(tactic);
  if(state == nullptr || state->preparationStatus(
       descriptor->inputChannels,descriptor->ffnChannels,
       configuredTokenRows) != cudaSuccess)
    return nullptr;
  Handle* handle = new(std::nothrow) Handle{
    tactic,
    configuredTokenRows,
    descriptor->inputChannels,
    descriptor->ffnChannels,
    descriptor->inputChannels == C384InputChannels &&
      descriptor->ffnChannels == C384FfnChannels,
    descriptor,
    std::move(state),
  };
  return handle;
}

extern "C" void katago_renju15_dual_ffn_sm120_destroy(void* opaque) {
  delete static_cast<Handle*>(opaque);
}

extern "C" const char* katago_renju15_dual_ffn_sm120_active_marker(
  const void* opaque) {
  const Handle* handle = static_cast<const Handle*>(opaque);
  return handle == nullptr ? nullptr : handle->descriptor->tacticId;
}

extern "C" bool katago_renju15_dual_ffn_sm120_request_eligible(
  int tokenRows,
  int inputChannels,
  int ffnChannels,
  bool usingFp16,
  bool usingNhwc,
  bool exactNoMask) {
  const bool geometry =
    (inputChannels == InputChannels && ffnChannels == FfnChannels) ||
    (inputChannels == C384InputChannels && ffnChannels == C384FfnChannels);
  return tokenRows > 0 && tokenRows <= MaxTokenRows && geometry &&
    usingFp16 && usingNhwc && exactNoMask;
}

extern "C" bool katago_renju15_dual_ffn_sm120_supports(
  const void* opaque,
  int tokenRows,
  int inputChannels,
  int ffnChannels,
  bool usingFp16,
  bool usingNhwc,
  bool exactNoMask) {
  const Handle* handle = static_cast<const Handle*>(opaque);
  return handle != nullptr &&
    tokenRowsSupported(
      handle->dynamicTokens,handle->configuredTokens,tokenRows) &&
    inputChannels == handle->inputChannels &&
    ffnChannels == handle->ffnChannels &&
    katago_renju15_dual_ffn_sm120_request_eligible(
      tokenRows, inputChannels, ffnChannels,
      usingFp16, usingNhwc, exactNoMask);
}

extern "C" cudaError_t katago_renju15_dual_ffn_sm120_launch(
  void* opaque,
  const half* input,
  const half* linearWeights,
  const half* gateWeights,
  half* output,
  int tokenRows,
  cudaStream_t stream) {
  Handle* handle = static_cast<Handle*>(opaque);
  const bool rowsSupported = handle != nullptr &&
    tokenRowsSupported(
      handle->dynamicTokens,handle->configuredTokens,tokenRows);
  if(handle == nullptr || input == nullptr || linearWeights == nullptr ||
     gateWeights == nullptr || output == nullptr || !rowsSupported ||
     !aligned16(input) || !aligned16(linearWeights) ||
     !aligned16(gateWeights) || !aligned16(output))
    return cudaErrorInvalidValue;
  return handle->state->launch(
    input, linearWeights, gateWeights, output,
    handle->inputChannels, handle->ffnChannels,
    tokenRows, stream);
}


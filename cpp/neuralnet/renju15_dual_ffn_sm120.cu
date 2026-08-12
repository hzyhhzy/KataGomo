#include "renju15_dual_ffn_sm120.h"

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "device/dual_gemm.h"
#include "thread/left_silu_and_mul.h"

#include <cstdint>
#include <memory>
#include <new>

namespace {

constexpr int InputChannels = 256;
constexpr int FfnChannels = 768;
constexpr int C384InputChannels = 384;
constexpr int C384FfnChannels = 1024;
constexpr int MaxTokenRows = 1 << 20;

using Element = cutlass::half_t;
using ProjectionOutput = cutlass::epilogue::thread::LinearCombination<
  Element, 8, Element, float, cutlass::epilogue::thread::ScaleType::Nothing>;
using SwiGLU = cutlass::epilogue::thread::LeftSiLUAndMul<
  Element, 8, Element, float>;
using InstructionShape = cutlass::gemm::GemmShape<16,8,16>;

template<
  int ThreadblockM, int ThreadblockN, int ThreadblockK,
  int WarpM, int WarpN, int WarpK,
  int Stages, int Swizzle>
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
  ProjectionOutput, ProjectionOutput, SwiGLU,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swizzle>,
  Stages, false, false, false, 8, 8>;

using Winner = DualGemm<128,64,32,64,32,32,3,4>;

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

struct Handle {
  int tactic;
  int fixedTokens;
  int inputChannels;
  int ffnChannels;
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
  case KATAGO_RENJU15_DUAL_FFN_C384_F1024_M128_N64_K32_S3_SW4:
    return std::unique_ptr<StateBase>(new State<Winner>());
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
  int fixedTokenRows) {
  if(fixedTokenRows <= 0 || fixedTokenRows > MaxTokenRows ||
     !isSm120Compatible())
    return nullptr;
  const auto* descriptor = descriptorForTactic(tactic);
  if(descriptor == nullptr)
    return nullptr;
  std::unique_ptr<StateBase> state = stateForTactic(tactic);
  if(state == nullptr)
    return nullptr;
  Handle* handle = new(std::nothrow) Handle{
    tactic,
    fixedTokenRows,
    descriptor->inputChannels,
    descriptor->ffnChannels,
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
    tokenRows == handle->fixedTokens &&
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
  cudaStream_t stream) {
  Handle* handle = static_cast<Handle*>(opaque);
  if(handle == nullptr || input == nullptr || linearWeights == nullptr ||
     gateWeights == nullptr || output == nullptr ||
     !aligned16(input) || !aligned16(linearWeights) ||
     !aligned16(gateWeights) || !aligned16(output))
    return cudaErrorInvalidValue;
  return handle->state->launch(
    input, linearWeights, gateWeights, output,
    handle->inputChannels, handle->ffnChannels,
    handle->fixedTokens, stream);
}


#include "renju15_residual_gemm_sm120.h"

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"

#include <cstdint>
#include <memory>
#include <new>

namespace {

constexpr int OutputChannels = 256;
constexpr int OutProjInputChannels = 256;
constexpr int FfnDownInputChannels = 768;
constexpr int MaxTokenRows = 1 << 20;

using Element = cutlass::half_t;
using InstructionShape = cutlass::gemm::GemmShape<16,8,16>;
using Epilogue = cutlass::epilogue::thread::LinearCombination<
  Element,
  128 / cutlass::sizeof_bits<Element>::value,
  Element,
  Element>;

template<
  int ThreadblockM, int ThreadblockN, int ThreadblockK,
  int WarpM, int WarpN, int WarpK,
  int Stages, int Swizzle>
using ResidualGemm = cutlass::gemm::device::Gemm<
  Element, cutlass::layout::RowMajor,
  Element, cutlass::layout::RowMajor,
  Element, cutlass::layout::RowMajor,
  Element,
  cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<ThreadblockM,ThreadblockN,ThreadblockK>,
  cutlass::gemm::GemmShape<WarpM,WarpN,WarpK>,
  InstructionShape,
  Epilogue,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swizzle>,
  Stages,
  8,
  8,
  false>;

using Winner = ResidualGemm<128,128,32,64,64,32,3,1>;

constexpr KatagoRenju15ResidualGemmDescriptor Descriptors[] = {
  {KATAGO_RENJU15_RESIDUAL_GEMM_M128_N128_K32_S3,
   "m128-n128-k32-s3-sw1",128,128,32,64,64,32,3,1,
   OutputChannels},
};

template<typename Gemm>
typename Gemm::Arguments makeArguments(
  const half* input,
  const half* weights,
  half* residual,
  int inputChannels,
  int fixedTokens) {
  using Layout = cutlass::layout::RowMajor;
  const Element one = Element(1.0f);
  return typename Gemm::Arguments(
    {fixedTokens, OutputChannels, inputChannels},
    {reinterpret_cast<const Element*>(input), Layout(inputChannels)},
    {reinterpret_cast<const Element*>(weights), Layout(OutputChannels)},
    {reinterpret_cast<const Element*>(residual), Layout(OutputChannels)},
    {reinterpret_cast<Element*>(residual), Layout(OutputChannels)},
    {one, one},
    1);
}

struct StateBase {
  virtual ~StateBase() = default;
  virtual cudaError_t launch(
    const half* input,
    const half* weights,
    half* residual,
    int inputChannels,
    int fixedTokens,
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
    const half* weights,
    half* residual,
    int inputChannels,
    int fixedTokens,
    cudaStream_t stream) override {
    typename Gemm::Arguments args = makeArguments<Gemm>(
      input, weights, residual, inputChannels, fixedTokens);
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
  int family;
  int tactic;
  int inputChannels;
  int fixedTokens;
  const KatagoRenju15ResidualGemmDescriptor* descriptor;
  std::unique_ptr<StateBase> state;
};

const KatagoRenju15ResidualGemmDescriptor* descriptorForTactic(int tactic) {
  for(const auto& value : Descriptors) {
    if(value.tactic == tactic)
      return &value;
  }
  return nullptr;
}

std::unique_ptr<StateBase> stateForTactic(int tactic) {
  switch(tactic) {
  case KATAGO_RENJU15_RESIDUAL_GEMM_M128_N128_K32_S3:
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

int inputChannelsForFamily(int family) {
  if(family == KATAGO_RENJU15_RESIDUAL_GEMM_OUT_PROJ)
    return OutProjInputChannels;
  if(family == KATAGO_RENJU15_RESIDUAL_GEMM_FFN_DOWN)
    return FfnDownInputChannels;
  return 0;
}

} // namespace

extern "C" void* katago_renju15_residual_gemm_sm120_create(
  int family,
  int tactic,
  int fixedTokenRows) {
  if(fixedTokenRows <= 0 || fixedTokenRows > MaxTokenRows ||
     !isSm120Compatible())
    return nullptr;
  const int inputChannels = inputChannelsForFamily(family);
  const auto* descriptor = descriptorForTactic(tactic);
  if(inputChannels == 0 || descriptor == nullptr)
    return nullptr;
  std::unique_ptr<StateBase> state = stateForTactic(tactic);
  if(state == nullptr)
    return nullptr;
  return new(std::nothrow) Handle{
    family,tactic,inputChannels,fixedTokenRows,
    descriptor,std::move(state)};
}

extern "C" void katago_renju15_residual_gemm_sm120_destroy(void* opaque) {
  delete static_cast<Handle*>(opaque);
}

extern "C" const char* katago_renju15_residual_gemm_sm120_active_marker(
  const void* opaque) {
  const Handle* handle = static_cast<const Handle*>(opaque);
  return handle == nullptr ? nullptr : handle->descriptor->tacticId;
}

extern "C" bool katago_renju15_residual_gemm_sm120_supports(
  const void* opaque,
  int matBatchSize,
  int inputChannels,
  int outputChannels,
  bool usingFp16,
  bool exactNoMask) {
  const Handle* handle = static_cast<const Handle*>(opaque);
  return handle != nullptr && matBatchSize == handle->fixedTokens &&
    inputChannels == handle->inputChannels &&
    outputChannels == OutputChannels && usingFp16 && exactNoMask;
}

extern "C" cudaError_t katago_renju15_residual_gemm_sm120_launch(
  void* opaque,
  const half* input,
  const half* weights,
  half* residual,
  cudaStream_t stream) {
  Handle* handle = static_cast<Handle*>(opaque);
  if(handle == nullptr || input == nullptr || weights == nullptr ||
     residual == nullptr || !aligned16(input) || !aligned16(weights) ||
     !aligned16(residual))
    return cudaErrorInvalidValue;
  return handle->state->launch(
    input, weights, residual, handle->inputChannels,
    handle->fixedTokens, stream);
}


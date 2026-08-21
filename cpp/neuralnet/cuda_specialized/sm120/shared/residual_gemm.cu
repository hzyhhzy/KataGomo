#include "residual_gemm.h"
#include "p2_residual_search_config.h"

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"

#include <cstdint>
#include <memory>
#include <new>
#include <type_traits>

namespace {

constexpr int OutputChannels = 256;
constexpr int OutProjInputChannels = 256;
constexpr int FfnDownInputChannels = 768;
constexpr int C384OutputChannels = 384;
constexpr int C384OutProjInputChannels = 384;
constexpr int C384FfnDownInputChannels = 1024;
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

using Winner = ResidualGemm<
  KATAGO_P2_RESIDUAL_SEARCH_THREADBLOCK_M,
  KATAGO_P2_RESIDUAL_SEARCH_THREADBLOCK_N,
  KATAGO_P2_RESIDUAL_SEARCH_THREADBLOCK_K,
  KATAGO_P2_RESIDUAL_SEARCH_WARP_M,
  KATAGO_P2_RESIDUAL_SEARCH_WARP_N,
  KATAGO_P2_RESIDUAL_SEARCH_THREADBLOCK_K,
  KATAGO_P2_RESIDUAL_SEARCH_STAGES,
  KATAGO_P2_RESIDUAL_SEARCH_SWIZZLE>;

using DownWinner = ResidualGemm<
  KATAGO_P2_DOWN_SEARCH_THREADBLOCK_M,
  KATAGO_P2_DOWN_SEARCH_THREADBLOCK_N,32,
  KATAGO_P2_DOWN_SEARCH_WARP_M,
  KATAGO_P2_DOWN_SEARCH_WARP_N,32,
  KATAGO_P2_DOWN_SEARCH_STAGES,
  KATAGO_P2_DOWN_SEARCH_SWIZZLE>;

constexpr KatagoRenju15ResidualGemmDescriptor Descriptors[] = {
  {KATAGO_RENJU15_RESIDUAL_GEMM_M128_N128_K32_S3,
   "m128-n128-k-search-s3-sw1",128,128,
   KATAGO_P2_RESIDUAL_SEARCH_THREADBLOCK_K,64,64,
   KATAGO_P2_RESIDUAL_SEARCH_THREADBLOCK_K,3,1,
   OutputChannels},
  {KATAGO_RENJU15_RESIDUAL_GEMM_C384_M128_N128_K32_S3,
   "c384-m128-n128-k-search-s3-sw1",128,128,
   KATAGO_P2_RESIDUAL_SEARCH_THREADBLOCK_K,64,64,
   KATAGO_P2_RESIDUAL_SEARCH_THREADBLOCK_K,3,1,
   C384OutputChannels},
};

template<typename Gemm>
typename Gemm::Arguments makeArguments(
  const half* input,
  const half* weights,
  half* residual,
  int inputChannels,
  int outputChannels,
  int fixedTokens) {
  using Layout = cutlass::layout::RowMajor;
  const Element one = Element(1.0f);
  return typename Gemm::Arguments(
    {fixedTokens, outputChannels, inputChannels},
    {reinterpret_cast<const Element*>(input), Layout(inputChannels)},
    {reinterpret_cast<const Element*>(weights), Layout(outputChannels)},
    {reinterpret_cast<const Element*>(residual), Layout(outputChannels)},
    {reinterpret_cast<Element*>(residual), Layout(outputChannels)},
    {one, one},
    1);
}

struct StateBase {
  virtual ~StateBase() = default;
  virtual cudaError_t preparationStatus(int, int, int) { return cudaSuccess; }
  virtual cudaError_t launch(
    const half* input,
    const half* weights,
    half* residual,
    int inputChannels,
    int outputChannels,
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
    int outputChannels,
    int fixedTokens,
    cudaStream_t stream) override {
    typename Gemm::Arguments args = makeArguments<Gemm>(
      input, weights, residual, inputChannels, outputChannels, fixedTokens);
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

// CUTLASS 3.x Gemm::update intentionally refreshes pointers and epilogue
// arguments only; it does not refresh problem_size or grid_tiled_shape. Build
// the trivially-copyable kernel Params on the stack for each actual M instead.
// This has no heap allocation or lazy initialization and accepts tail row
// counts that are not a multiple of the board area. The exact C256 state above
// remains byte-for-logic unchanged.
template<typename Gemm>
struct DynamicState final : StateBase {
  using Kernel = typename Gemm::GemmKernel;

  DynamicState(): attributeStatus(cudaSuccess) {}

  cudaError_t preparationStatus(
    int inputChannels, int outputChannels, int maximumTokens) override {
    static_assert(std::is_trivially_copyable<typename Kernel::Params>::value,
      "dynamic GEMM launch passes kernel Params by value");
    // This runs once per prepared handle in the handle's current CUDA device
    // context. A process-global template static would miss later GPUs.
    const int sharedBytes = int(sizeof(typename Kernel::SharedStorage));
    attributeStatus = sharedBytes < (48 << 10) ? cudaSuccess :
      cudaFuncSetAttribute(
        cutlass::Kernel<Kernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        sharedBytes);
    if(attributeStatus != cudaSuccess)
      return attributeStatus;

    // K/N and pointer-alignment qualification is invariant across actual M.
    // Check both M boundaries at handle construction, never on the hot path.
    const half* alignedInput = reinterpret_cast<const half*>(0x1000);
    const half* alignedWeights = reinterpret_cast<const half*>(0x2000);
    half* alignedResidual = reinterpret_cast<half*>(0x3000);
    const int boundaryTokens[2] = {1,maximumTokens};
    for(int tokens: boundaryTokens) {
      typename Gemm::Arguments args = makeArguments<Gemm>(
        alignedInput,alignedWeights,alignedResidual,
        inputChannels,outputChannels,tokens);
      if(Gemm::can_implement(args) != cutlass::Status::kSuccess) {
        attributeStatus = cudaErrorInvalidValue;
        break;
      }
    }
    return attributeStatus;
  }

  cudaError_t launch(
    const half* input,
    const half* weights,
    half* residual,
    int inputChannels,
    int outputChannels,
    int tokens,
    cudaStream_t stream) override {
    if(attributeStatus != cudaSuccess)
      return attributeStatus;
    typename Gemm::Arguments args = makeArguments<Gemm>(
      input, weights, residual, inputChannels, outputChannels, tokens);

    using ThreadblockShape = typename Gemm::ThreadblockShape;
    typename Gemm::ThreadblockSwizzle swizzle;
    const cutlass::gemm::GemmCoord gridShape = swizzle.get_tiled_shape(
      args.problem_size,
      {ThreadblockShape::kM,ThreadblockShape::kN,ThreadblockShape::kK},
      args.split_k_slices);
    typename Kernel::Params params{
      args.problem_size,
      gridShape,
      args.ref_A.non_const_ref(),
      args.ref_B.non_const_ref(),
      args.ref_C.non_const_ref(),
      args.ref_D,
      args.epilogue,
      nullptr,
      args.gather_A_indices,
      args.gather_B_indices,
      args.scatter_D_indices,
    };
    const dim3 grid = swizzle.get_grid_shape(gridShape);
    const dim3 block(Kernel::kThreadCount,1,1);
    const int sharedBytes = int(sizeof(typename Kernel::SharedStorage));
    cutlass::arch::synclog_setup();
    cutlass::Kernel<Kernel><<<grid,block,sharedBytes,stream>>>(params);
    return cudaPeekAtLastError();
  }

private:
  cudaError_t attributeStatus;
};

struct Handle {
  int family;
  int tactic;
  int inputChannels;
  int outputChannels;
  int configuredTokens;
  bool dynamicTokens;
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

std::unique_ptr<StateBase> stateForTactic(int family, int tactic) {
  switch(tactic) {
  case KATAGO_RENJU15_RESIDUAL_GEMM_M128_N128_K32_S3:
    if(family == KATAGO_RENJU15_RESIDUAL_GEMM_FFN_DOWN)
      return std::unique_ptr<StateBase>(new State<DownWinner>());
    return std::unique_ptr<StateBase>(new State<Winner>());
  case KATAGO_RENJU15_RESIDUAL_GEMM_C384_M128_N128_K32_S3:
    return std::unique_ptr<StateBase>(new DynamicState<Winner>());
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
  if(family == KATAGO_RENJU15_RESIDUAL_GEMM_C384_OUT_PROJ)
    return C384OutProjInputChannels;
  if(family == KATAGO_RENJU15_RESIDUAL_GEMM_C384_FFN_DOWN)
    return C384FfnDownInputChannels;
  return 0;
}

int outputChannelsForFamily(int family) {
  if(family == KATAGO_RENJU15_RESIDUAL_GEMM_OUT_PROJ ||
     family == KATAGO_RENJU15_RESIDUAL_GEMM_FFN_DOWN)
    return OutputChannels;
  if(family == KATAGO_RENJU15_RESIDUAL_GEMM_C384_OUT_PROJ ||
     family == KATAGO_RENJU15_RESIDUAL_GEMM_C384_FFN_DOWN)
    return C384OutputChannels;
  return 0;
}

bool dynamicTokensForFamily(int family) {
  return family == KATAGO_RENJU15_RESIDUAL_GEMM_C384_OUT_PROJ ||
    family == KATAGO_RENJU15_RESIDUAL_GEMM_C384_FFN_DOWN;
}

} // namespace

extern "C" void* katago_renju15_residual_gemm_sm120_create(
  int family,
  int tactic,
  int configuredTokenRows) {
  if(configuredTokenRows <= 0 || configuredTokenRows > MaxTokenRows ||
     !isSm120Compatible())
    return nullptr;
  const int inputChannels = inputChannelsForFamily(family);
  const int outputChannels = outputChannelsForFamily(family);
  const auto* descriptor = descriptorForTactic(tactic);
  if(inputChannels == 0 || outputChannels == 0 || descriptor == nullptr ||
     descriptor->outputChannels != outputChannels)
    return nullptr;
  std::unique_ptr<StateBase> state = stateForTactic(family,tactic);
  if(state == nullptr || state->preparationStatus(
       inputChannels,outputChannels,configuredTokenRows) != cudaSuccess)
    return nullptr;
  return new(std::nothrow) Handle{
    family,tactic,inputChannels,outputChannels,configuredTokenRows,
    dynamicTokensForFamily(family),
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
  const bool rowsSupported = handle != nullptr &&
    tokenRowsSupported(
      handle->dynamicTokens,handle->configuredTokens,matBatchSize);
  return rowsSupported &&
    inputChannels == handle->inputChannels &&
    outputChannels == handle->outputChannels && usingFp16 && exactNoMask;
}

extern "C" cudaError_t katago_renju15_residual_gemm_sm120_launch(
  void* opaque,
  const half* input,
  const half* weights,
  half* residual,
  int matBatchSize,
  cudaStream_t stream) {
  Handle* handle = static_cast<Handle*>(opaque);
  const bool rowsSupported = handle != nullptr &&
    tokenRowsSupported(
      handle->dynamicTokens,handle->configuredTokens,matBatchSize);
  if(handle == nullptr || input == nullptr || weights == nullptr ||
     residual == nullptr || !rowsSupported ||
     !aligned16(input) || !aligned16(weights) ||
     !aligned16(residual))
    return cudaErrorInvalidValue;
  return handle->state->launch(
    input, weights, residual, handle->inputChannels,
    handle->outputChannels, matBatchSize, stream);
}


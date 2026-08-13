/***************************************************************************************************
 * Fixed S225/C256/H8/D32 FP16 batched QKV GEMM with learned 2-D RoPE in the epilogue.
 *
 * The output contract is the existing planar FA4 contract. CUTLASS batch z=0/1/2 is Q/K/V;
 * only z<2 is rotated. The implementation is adapted from KataGomo's retained SM89
 * cudabackend_sm89_qkv_rope_gemm.cu, with target-specific dimensions,
 * resource checks, and half2 table consumption.
 **************************************************************************************************/

#include "qkv_rope_gemm.h"

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_batched.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

#include <cstdint>
#include <memory>
#include <new>

namespace {

constexpr int SequenceLength = 225;
constexpr int Channels = 256;
constexpr int Heads = 8;
constexpr int HeadDim = 32;
constexpr int RopePairs = 16;
constexpr int GemmBatch = 3;
constexpr int Batch36 = 36;
constexpr std::size_t RequiredSharedBytesPerBlockOptin = 101376ULL;

using Element = cutlass::half_t;
using Layout = cutlass::layout::RowMajor;
using OutputOp = cutlass::epilogue::thread::LinearCombination<
  Element, 8, Element, float, cutlass::epilogue::thread::ScaleType::Nothing>;
using InstructionShape = cutlass::gemm::GemmShape<16,8,16>;
using Swizzle = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

template<typename BaseIterator>
class RoPEOutputTileIterator : public BaseIterator {
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

  static_assert((kElementsPerAccess % 2) == 0,
    "QKV+RoPE epilogue accesses must preserve adjacent rotary pairs");

  struct Params : public Base::Params {
    const half2* cosSinTable;
    int totalRows;

    CUTLASS_HOST_DEVICE
    Params() : Base::Params(), cosSinTable(nullptr), totalRows(0) {}

    CUTLASS_HOST_DEVICE
    explicit Params(Layout const& layout)
      : Base::Params(layout), cosSinTable(nullptr), totalRows(0) {}
  };

 private:
  const half2* cosSinTable;
  int totalRows;

 public:
  CUTLASS_DEVICE
  RoPEOutputTileIterator(
    Params const& params,
    Element* pointer,
    TensorCoord extent,
    int threadIdx,
    TensorCoord threadblockOffset = TensorCoord(),
    int const* indices = nullptr
  ) : Base(params, pointer, extent, threadIdx, threadblockOffset, indices),
      cosSinTable(params.cosSinTable), totalRows(params.totalRows) {}

  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const& fragment, int64_t byteOffset) const {
    Fragment transformed = fragment;
    // Batched GEMM z=0/1/2 is Q/K/V. V must remain bit-identical to the plain
    // GEMM epilogue. A null table is never accepted by the host launch gate.
    if(cosSinTable != nullptr && int(blockIdx.z) < 2) {
      AccessType* accesses = reinterpret_cast<AccessType*>(&transformed);
      int startRow = Base::thread_start_row();
      int startColumn = Base::thread_start_column();

      CUTLASS_PRAGMA_UNROLL
      for(int cluster = 0; cluster < ThreadMap::Iterations::kCluster; cluster++) {
        CUTLASS_PRAGMA_UNROLL
        for(int group = 0; group < ThreadMap::Iterations::kGroup; group++) {
          CUTLASS_PRAGMA_UNROLL
          for(int row = 0; row < ThreadMap::Iterations::kRow; row++) {
            int fragmentRow = row + ThreadMap::Iterations::kRow *
              (group + ThreadMap::Iterations::kGroup * cluster);
            int rowOffset = row * ThreadMap::Delta::kRow +
              group * ThreadMap::Delta::kGroup +
              cluster * ThreadMap::Delta::kCluster;
            int outputRow = startRow + rowOffset;

            CUTLASS_PRAGMA_UNROLL
            for(int column = 0; column < ThreadMap::Iterations::kColumn; column++) {
              int outputColumn = startColumn + column * ThreadMap::Delta::kColumn;
              if(outputRow < totalRows && outputColumn < Channels) {
                AccessType& access = accesses[
                  fragmentRow * ThreadMap::Iterations::kColumn + column];
                int xy = outputRow % SequenceLength;

                CUTLASS_PRAGMA_UNROLL
                for(int element = 0; element < kElementsPerAccess; element += 2) {
                  int channel = outputColumn + element;
                  int hp = channel / 2;
                  float2 cs = __half22float2(
                    cosSinTable[(size_t)xy * (Channels / 2) + hp]);
                  float v0 = static_cast<float>(access[element]);
                  float v1 = static_cast<float>(access[element + 1]);
                  access[element] = Element(v0 * cs.x - v1 * cs.y);
                  access[element + 1] = Element(v0 * cs.y + v1 * cs.x);
                }
              }
            }
          }
        }
      }
    }
    Base::store_with_byte_offset(transformed, byteOffset);
  }

  CUTLASS_DEVICE
  void store(Fragment const& fragment) const {
    store_with_byte_offset(fragment, 0);
  }
};

template<
  int ThreadblockM, int ThreadblockN, int ThreadblockK,
  int WarpM, int WarpN, int WarpK, int Stages>
struct KernelBundle {
  using ThreadblockShape = cutlass::gemm::GemmShape<
    ThreadblockM,ThreadblockN,ThreadblockK>;
  using WarpShape = cutlass::gemm::GemmShape<WarpM,WarpN,WarpK>;
  using DeviceGemm = cutlass::gemm::device::GemmBatched<
    Element, Layout,
    Element, Layout,
    Element, Layout,
    Element,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    OutputOp,
    Swizzle,
    Stages,
    8,
    8>;
  using DefaultKernel = typename DeviceGemm::DefaultGemmKernel;
  using Mma = typename DefaultKernel::Mma;
  using DefaultEpilogue = typename DefaultKernel::Epilogue;
  using DefaultIterator = typename DefaultEpilogue::OutputTileIterator;
  using RopeIterator = RoPEOutputTileIterator<DefaultIterator>;
  using RopeEpilogue = cutlass::epilogue::threadblock::Epilogue<
    typename DefaultEpilogue::Shape,
    typename DefaultEpilogue::WarpMmaOperator,
    DefaultEpilogue::kPartitionsK,
    RopeIterator,
    typename DefaultEpilogue::AccumulatorFragmentIterator,
    typename DefaultEpilogue::WarpTileIterator,
    typename DefaultEpilogue::SharedLoadIterator,
    typename DefaultEpilogue::OutputOp,
    typename DefaultEpilogue::Padding,
    DefaultEpilogue::Base::kFragmentsPerIteration>;
  using Kernel = cutlass::gemm::kernel::GemmBatched<Mma,RopeEpilogue,Swizzle>;
};

using Winner = KernelBundle<128,128,32,64,64,32,3>;

static_assert(sizeof(typename Winner::Kernel::SharedStorage) <=
    RequiredSharedBytesPerBlockOptin,
  "QKV+RoPE winner exceeds the SM120 opt-in shared-memory contract");
static_assert(Winner::Kernel::kThreadCount <= 1024,
  "QKV+RoPE winner exceeds CUDA threads-per-block limit");

constexpr KatagoRenju15QKVRoPEGemmSm120Descriptor Descriptors[] = {
  {KATAGO_RENJU15_QKV_ROPE_GEMM_M128_N128_K32_S3,
   "qkv-rope-m128-n128-k32-s3-sw1",128,128,32,64,64,32,3,1,
   Winner::Kernel::kThreadCount,sizeof(typename Winner::Kernel::SharedStorage),
   Channels,Channels,SequenceLength},
};

struct StateBase {
  virtual ~StateBase() = default;
  virtual cudaError_t launch(
    const half* input,
    const half* packedWeights,
    const half2* cosSin,
    half* output,
    int tokens,
    cudaStream_t stream) = 0;
};

template<typename Bundle>
struct State final : StateBase {
  using Kernel = typename Bundle::Kernel;
  using Mma = typename Bundle::Mma;
  using Iterator = typename Bundle::RopeIterator;

  typename Kernel::Params params;
  bool configured = false;

  cudaError_t configure(
    const half* input,
    const half* packedWeights,
    const half2* cosSin,
    half* output,
    int tokens) {
    const int smemSize = int(sizeof(typename Kernel::SharedStorage));
    if(smemSize >= 48 * 1024) {
      cudaError_t status = cudaFuncSetAttribute(
        cutlass::Kernel<Kernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smemSize);
      if(status != cudaSuccess)
        return status;
    }

    cutlass::gemm::GemmCoord problem(tokens, Channels, Channels);
    cutlass::gemm::GemmCoord gridShape = Swizzle::get_tiled_shape(
      problem,
      {Bundle::ThreadblockShape::kM,
       Bundle::ThreadblockShape::kN,
       Bundle::ThreadblockShape::kK},
      GemmBatch);
    Layout matrixLayout(Channels);
    typename Mma::IteratorA::TensorRef inputRef(
      reinterpret_cast<Element*>(const_cast<half*>(input)), matrixLayout);
    typename Mma::IteratorB::TensorRef weightRef(
      reinterpret_cast<Element*>(const_cast<half*>(packedWeights)), matrixLayout);
    typename Iterator::TensorRef outputRef(
      reinterpret_cast<Element*>(output), matrixLayout);

    params = typename Kernel::Params(
      problem,
      gridShape,
      inputRef,
      0,
      weightRef,
      (int64_t)Channels * Channels,
      outputRef,
      (int64_t)tokens * Channels,
      outputRef,
      (int64_t)tokens * Channels,
      typename OutputOp::Params(1.0f,0.0f),
      GemmBatch);
    params.params_D.cosSinTable = cosSin;
    params.params_D.totalRows = tokens;
    configured = true;
    return cudaSuccess;
  }

  cudaError_t launch(
    const half* input,
    const half* packedWeights,
    const half2* cosSin,
    half* output,
    int tokens,
    cudaStream_t stream) override {
    if(!configured) {
      cudaError_t status = configure(
        input,packedWeights,cosSin,output,tokens);
      if(status != cudaSuccess)
        return status;
    }
    else {
      params.ref_A.reset(
        reinterpret_cast<Element*>(const_cast<half*>(input)));
      params.ref_B.reset(
        reinterpret_cast<Element*>(const_cast<half*>(packedWeights)));
      params.ref_C.reset(reinterpret_cast<Element*>(output));
      params.ref_D.reset(reinterpret_cast<Element*>(output));
      params.params_D.cosSinTable = cosSin;
    }

    dim3 grid = Swizzle::get_grid_shape(params.grid_tiled_shape);
    dim3 block(Kernel::kThreadCount,1,1);
    const int smemSize = int(sizeof(typename Kernel::SharedStorage));
    cutlass::Kernel<Kernel><<<grid,block,smemSize,stream>>>(params);
    return cudaPeekAtLastError();
  }
};

struct Handle {
  int tactic;
  int fixedBatchSize;
  int fixedTokens;
  const KatagoRenju15QKVRoPEGemmSm120Descriptor* descriptor;
  std::unique_ptr<StateBase> state;
};

const KatagoRenju15QKVRoPEGemmSm120Descriptor* descriptorForTactic(int tactic) {
  for(const auto& descriptor : Descriptors) {
    if(descriptor.tactic == tactic)
      return &descriptor;
  }
  return nullptr;
}

std::unique_ptr<StateBase> stateForTactic(int tactic) {
  switch(tactic) {
  case KATAGO_RENJU15_QKV_ROPE_GEMM_M128_N128_K32_S3:
    return std::unique_ptr<StateBase>(new State<Winner>());
  default:
    return nullptr;
  }
}

bool isSm120Compatible() {
  int device = -1;
  cudaDeviceProp prop{};
  if(cudaGetDevice(&device) != cudaSuccess ||
     cudaGetDeviceProperties(&prop,device) != cudaSuccess)
    return false;
  return prop.major == 12 && prop.minor == 0 && prop.warpSize == 32 &&
    (std::size_t)prop.sharedMemPerBlockOptin >=
      RequiredSharedBytesPerBlockOptin;
}

bool aligned16(const void* ptr) {
  return (reinterpret_cast<std::uintptr_t>(ptr) & 15U) == 0;
}

} // namespace

extern "C" void* katago_renju15_qkv_rope_gemm_sm120_create(
  int tactic,
  int fixedBatchSize) {
  if(fixedBatchSize != Batch36 || !isSm120Compatible())
    return nullptr;
  const auto* descriptor = descriptorForTactic(tactic);
  std::unique_ptr<StateBase> state = stateForTactic(tactic);
  if(descriptor == nullptr || state == nullptr)
    return nullptr;
  return new(std::nothrow) Handle{
    tactic,fixedBatchSize,fixedBatchSize * SequenceLength,
    descriptor,std::move(state)};
}

extern "C" void katago_renju15_qkv_rope_gemm_sm120_destroy(void* opaque) {
  delete static_cast<Handle*>(opaque);
}

extern "C" const char* katago_renju15_qkv_rope_gemm_sm120_active_marker(
  const void* opaque) {
  const Handle* handle = static_cast<const Handle*>(opaque);
  return handle == nullptr ? nullptr : handle->descriptor->tacticId;
}

extern "C" bool katago_renju15_qkv_rope_gemm_sm120_supports(
  const void* opaque,
  int batchSize,
  int seqLen,
  int inputChannels,
  int projectionChannels,
  int numHeads,
  int numKVHeads,
  int headDim,
  int ropePairs,
  bool usingFp16,
  bool usingNhwc,
  bool requireExactNNLen,
  bool recipeEligibleNoMask,
  bool precomputedHalf2) {
  const Handle* handle = static_cast<const Handle*>(opaque);
  return handle != nullptr &&
    batchSize == handle->fixedBatchSize &&
    seqLen == SequenceLength &&
    inputChannels == Channels && projectionChannels == Channels &&
    numHeads == Heads && numKVHeads == Heads &&
    headDim == HeadDim && ropePairs == RopePairs &&
    usingFp16 && usingNhwc && requireExactNNLen &&
    recipeEligibleNoMask && precomputedHalf2;
}

extern "C" cudaError_t katago_renju15_qkv_rope_gemm_sm120_launch(
  void* opaque,
  const half* input,
  const half* packedWeights,
  const half2* cosSin,
  half* output,
  int batchSize,
  cudaStream_t stream) {
  Handle* handle = static_cast<Handle*>(opaque);
  if(handle == nullptr || input == nullptr || packedWeights == nullptr ||
     cosSin == nullptr || output == nullptr || stream == nullptr ||
     batchSize != handle->fixedBatchSize ||
     !aligned16(input) || !aligned16(packedWeights) ||
     !aligned16(cosSin) || !aligned16(output))
    return cudaErrorInvalidValue;
  return handle->state->launch(
    input,packedWeights,cosSin,output,handle->fixedTokens,stream);
}


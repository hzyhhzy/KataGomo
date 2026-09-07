#include "cudab11gemm.h"

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"

#include <array>
#include <cstdint>
#include <mutex>
#include <vector>

namespace {

using Element = cutlass::half_t;
using Epilogue = cutlass::epilogue::thread::LinearCombination<
  Element, 8, Element, Element>;

using Gemm = cutlass::gemm::device::Gemm<
  Element, cutlass::layout::RowMajor,
  Element, cutlass::layout::RowMajor,
  Element, cutlass::layout::RowMajor,
  Element,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<128,128,32>,
  cutlass::gemm::GemmShape<64,64,32>,
  cutlass::gemm::GemmShape<16,8,16>,
  Epilogue,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
  3, 8, 8, false>;

inline bool isSupportedShape(int M, int N, int K) {
  return M > 0 && (
    (N == 384 && (K == 384 || K == 768 || K == 1152)) ||
    ((N == 768 || N == 1152) && K == 384));
}

inline bool isAligned(const void* p) {
  return p != nullptr && (reinterpret_cast<std::uintptr_t>(p) & 15) == 0;
}

// CUDA function attributes belong to a device, not to a model or stream.
// Remember each successful opt-in once per kernel/device, with a thread-local
// fast path. Kernels needing <=48 KiB do not require opt-in in the first place.
template<typename Kernel>
cudaError_t prepareSharedMemory(int device) {
  constexpr int sharedBytes = sizeof(typename Kernel::SharedStorage);
  if(sharedBytes <= 48 * 1024)
    return cudaSuccess;

  thread_local std::vector<int> knownDevices;
  for(int known : knownDevices)
    if(known == device) return cudaSuccess;

  static std::mutex mutex;
  static std::vector<int> preparedDevices;
  std::lock_guard<std::mutex> lock(mutex);
  for(int prepared : preparedDevices) {
    if(prepared == device) {
      knownDevices.push_back(device);
      return cudaSuccess;
    }
  }
  const cudaError_t status = cudaFuncSetAttribute(
    cutlass::Kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedBytes);
  if(status == cudaSuccess) {
    preparedDevices.push_back(device);
    knownDevices.push_back(device);
  }
  return status;
}

template<typename GemmType>
struct LaunchState {
  using Kernel = typename GemmType::GemmKernel;
  typename Kernel::Params params;
  dim3 grid;
  int device = -1;
  int M = 0;
  int N = 0;
  int K = 0;

  bool matches(int device_, int M_, int N_, int K_) const {
    return device == device_ && M == M_ && N == N_ && K == K_;
  }
};

template<typename GemmType>
struct LaunchCache {
  // Bounded host-only cache. Entries contain no owned CUDA resources and their
  // pointer fields are always refreshed before use, even if an allocator has
  // recycled an old model's addresses. This is safe across sequential models,
  // streams, and devices on one thread. Asynchronous launches copy Params.
  std::array<LaunchState<GemmType>,24> entries;
  unsigned next = 0;
};

cudaError_t launch(
  const half* input, const half* weights, half* output,
  int M, int N, int K, bool betaIsOne, cudaStream_t stream
) {
  if(!isSupportedShape(M,N,K) || !isAligned(input) || !isAligned(weights) ||
     !isAligned(output) || input == output || weights == output)
    return cudaErrorInvalidValue;

  using GemmType = Gemm;
  using Kernel = typename GemmType::GemmKernel;
  using Swizzle = typename GemmType::ThreadblockSwizzle;

  int device;
  cudaError_t status = cudaGetDevice(&device);
  if(status != cudaSuccess) return status;
  status = prepareSharedMemory<Kernel>(device);
  if(status != cudaSuccess) return status;

  auto* a = reinterpret_cast<const Element*>(input);
  auto* b = reinterpret_cast<const Element*>(weights);
  auto* d = reinterpret_cast<Element*>(output);
  const typename Epilogue::Params epilogue(
    Element(1.0f), Element(betaIsOne ? 1.0f : 0.0f));

  thread_local LaunchCache<GemmType> cache;
  LaunchState<GemmType>* state = nullptr;
  for(auto& candidate : cache.entries) {
    if(candidate.matches(device,M,N,K)) {
      state = &candidate;
      break;
    }
  }

  if(state == nullptr) {
    const typename GemmType::Arguments args(
      cutlass::gemm::GemmCoord(M,N,K),
      {a,K}, {b,N}, {d,N}, {d,N}, epilogue, 1);
    if(GemmType::can_implement(args) != cutlass::Status::kSuccess)
      return cudaErrorInvalidValue;

    state = &cache.entries[cache.next];
    cache.next = (cache.next + 1) % cache.entries.size();
    Swizzle swizzle;
    const cutlass::gemm::GemmCoord tiledShape = swizzle.get_tiled_shape(
      args.problem_size,
      {GemmType::ThreadblockShape::kM, GemmType::ThreadblockShape::kN,
       GemmType::ThreadblockShape::kK}, 1);
    state->params = typename Kernel::Params{
      args.problem_size, tiledShape,
      args.ref_A.non_const_ref(), args.ref_B.non_const_ref(),
      args.ref_C.non_const_ref(), args.ref_D, args.epilogue, nullptr,
      args.gather_A_indices, args.gather_B_indices, args.scatter_D_indices};
    state->grid = swizzle.get_grid_shape(tiledShape);
    state->device = device;
    state->M = M;
    state->N = N;
    state->K = K;
  }
  else {
    state->params.ref_A.reset(const_cast<Element*>(a));
    state->params.ref_B.reset(const_cast<Element*>(b));
    state->params.ref_C.reset(d);
    state->params.ref_D.reset(d);
    state->params.output_op = epilogue;
  }

  constexpr int sharedBytes = sizeof(typename Kernel::SharedStorage);
  const dim3 block(Kernel::kThreadCount,1,1);
  cutlass::Kernel<Kernel><<<state->grid,block,sharedBytes,stream>>>(state->params);
  return cudaPeekAtLastError();
}

} // namespace

cudaError_t launchB11Gemm(
  const half* input, const half* weights, half* output,
  int M, int N, int K, bool betaIsOne, cudaStream_t stream
) {
  return launch(input,weights,output,M,N,K,betaIsOne,stream);
}

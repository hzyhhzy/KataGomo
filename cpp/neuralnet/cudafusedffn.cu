// CUTLASS dual-GEMM + SwiGLU epilogue implementation of CudaFusedFFN (see cudafusedffn.h).
// Built on the DualGemm extension from CUTLASS's example 45 (vendored under
// external/cutlass/examples/45_dual_gemm, included via CMake): both GEMMs share the A operand
// tile pipeline, and the epilogue combines their accumulators as SiLU(D0) * D1 so the
// intermediates never touch global memory.
//
// A single tile configuration (threadblock 128x64x32, warp 64x32x32, 3 stages, FP16
// accumulation) is instantiated: it measured fastest on every architecture tested
// (sm_86 A5000, sm_89 4090, sm_90 H100, sm_120 RTX PRO 6000), 1.4-1.9x the unfused
// cublasHgemm x2 + SwiGLU kernel sequence at KataGo's FFN shapes.

#include "../neuralnet/cudafusedffn.h"
#include "p2_ffn_search_config.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "device/dual_gemm.h"
#include "thread/left_silu_and_mul.h"

namespace {

using ElementT = cutlass::half_t;
static constexpr int kAlignment = 128 / cutlass::sizeof_bits<ElementT>::value;

using EpilogueOutputOp01 = cutlass::epilogue::thread::LinearCombination<
  ElementT, kAlignment, ElementT, ElementT,
  cutlass::epilogue::thread::ScaleType::Nothing>;
// Deliberately use the upstream fast FP16 epilogue. This is not bit-equivalent
// to KataGo's FP32 SwiGLU helper, so it is qualified as an explicit numerical
// recipe by the standalone tensor contract plus the full 8,192-position
// policy/value loss replay. Do not silently describe this as an exact path.
using FinalEpilogueComputeT = ElementT;
using EpilogueOutputOp2 = cutlass::epilogue::thread::LeftSiLUAndMul<
  ElementT, kAlignment, ElementT, FinalEpilogueComputeT>;

using DualGemm = cutlass::gemm::device::DualGemm<
  ElementT, cutlass::layout::RowMajor,
  ElementT, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor,
  ElementT, cutlass::layout::RowMajor,
  ElementT,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<
    KATAGO_P2_FFN_SEARCH_THREADBLOCK_M,
    KATAGO_P2_FFN_SEARCH_THREADBLOCK_N,32>,
  cutlass::gemm::GemmShape<
    KATAGO_P2_FFN_SEARCH_WARP_M,KATAGO_P2_FFN_SEARCH_WARP_N,32>,
  cutlass::gemm::GemmShape<16, 8, 16>,
  EpilogueOutputOp01, EpilogueOutputOp01, EpilogueOutputOp2,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<
    KATAGO_P2_FFN_SEARCH_SWIZZLE>,
  KATAGO_P2_FFN_SEARCH_STAGES,
  false,  // kStoreD0
  false,  // kStoreD1
  false   // kSplitKSerial
>;

// Keep the stock upstream no-clip DualGemm above byte-for-byte equivalent to
// the Stage 1 recipe. Runtime clipping is a separate epilogue and one separate
// DualGemm instantiation, so clip parameters can never alter the certified
// no-clip kernel's type, resources, or generated code.
template <
  typename ElementOutput_,
  int Count,
  typename ElementAccumulator_ = ElementOutput_,
  typename ElementCompute_ = float,
  cutlass::FloatRoundStyle Round =
    cutlass::FloatRoundStyle::round_to_nearest
>
class LeftRuntimeClippedSiLUAndMul {
 public:
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  static int const kCount = Count;
  using FragmentOutput = cutlass::Array<ElementOutput,kCount>;
  using FragmentAccumulator = cutlass::Array<ElementAccumulator,kCount>;
  using ComputeFragment = cutlass::Array<ElementCompute,kCount>;

  struct Params {
    float clip;

    CUTLASS_HOST_DEVICE
    explicit Params(float clip_ = 0.0f) : clip(clip_) {}
  };

 private:
  ElementCompute clip_;

  CUTLASS_HOST_DEVICE
  ElementCompute clampSymmetric(ElementCompute value) const {
    // Match customCudaClippedSwiGLU's ordered comparisons exactly. In
    // particular, a NaN value propagates rather than being replaced by a
    // bound, while valid clip Params are guaranteed finite and positive.
    return value > clip_ ? clip_ : (value < -clip_ ? -clip_ : value);
  }

 public:
  CUTLASS_HOST_DEVICE
  LeftRuntimeClippedSiLUAndMul(Params const& params) :
    clip_(ElementCompute(params.clip)) {}

  CUTLASS_HOST_DEVICE
  bool is_source_needed() const { return true; }

  CUTLASS_HOST_DEVICE
  void set_k_partition(int, int) { assert(false); }

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const& lhs,
    FragmentAccumulator const& rhs
  ) const {
    cutlass::NumericArrayConverter<
      ElementCompute,ElementAccumulator,kCount,Round> accumulatorToCompute;
    cutlass::NumericArrayConverter<
      ElementOutput,ElementCompute,kCount,Round> computeToOutput;

    ComputeFragment convertedLhs = accumulatorToCompute(lhs);
    ComputeFragment convertedRhs = accumulatorToCompute(rhs);
    cutlass::epilogue::thread::SiLu<ComputeFragment> silu;
    ComputeFragment siluLhs = silu(convertedLhs);
    ComputeFragment result;
    CUTLASS_PRAGMA_UNROLL
    for(int i = 0; i < kCount; i++)
      result[i] = clampSymmetric(siluLhs[i]) *
        clampSymmetric(convertedRhs[i]);
    return computeToOutput(result);
  }

  CUTLASS_HOST_DEVICE
  ElementOutput operator()(
    ElementAccumulator const& lhs,
    ElementAccumulator const& rhs
  ) const {
    const ElementCompute convertedLhs(lhs);
    const ElementCompute convertedRhs(rhs);
    cutlass::epilogue::thread::SiLu<ElementCompute> silu;
    cutlass::NumericConverter<ElementOutput,ElementCompute,Round>
      computeToOutput;
    return computeToOutput(
      clampSymmetric(silu(convertedLhs)) * clampSymmetric(convertedRhs));
  }
};

using ClippedEpilogueOutputOp2 = LeftRuntimeClippedSiLUAndMul<
  ElementT,kAlignment,ElementT,float>;
static_assert(
  sizeof(ClippedEpilogueOutputOp2::Params) == sizeof(float),
  "runtime clip Params must carry exactly one float value, never a pointer");

using ClippedDualGemm = cutlass::gemm::device::DualGemm<
  ElementT, cutlass::layout::RowMajor,
  ElementT, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor,
  ElementT, cutlass::layout::RowMajor,
  ElementT,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<128, 64, 32>,
  cutlass::gemm::GemmShape<64, 32, 32>,
  cutlass::gemm::GemmShape<16, 8, 16>,
  EpilogueOutputOp01, EpilogueOutputOp01, ClippedEpilogueOutputOp2,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
  3,
  false,
  false,
  false
>;

DualGemm::Arguments makeArgs(
  const half* A, const half* w1, const half* wGate, half* out, int M, int N, int K
) {
  return DualGemm::Arguments(
    cutlass::gemm::DualGemmMode::kGemm,
    {M, N, K},
    {(ElementT const*)A, K},
    {(ElementT const*)w1, K},
    cutlass::TensorRef<ElementT const, cutlass::layout::RowMajor>(),
    cutlass::TensorRef<ElementT, cutlass::layout::RowMajor>(),
    {(ElementT const*)wGate, K},
    cutlass::TensorRef<ElementT const, cutlass::layout::RowMajor>(),
    cutlass::TensorRef<ElementT, cutlass::layout::RowMajor>(),
    {(ElementT*)out, N},
    EpilogueOutputOp01::Params(),
    EpilogueOutputOp01::Params(),
    EpilogueOutputOp2::Params(),
    1);
}

ClippedDualGemm::Arguments makeClippedArgs(
  const half* A, const half* w1, const half* wGate, half* out,
  int M, int N, int K, float clip
) {
  return ClippedDualGemm::Arguments(
    cutlass::gemm::DualGemmMode::kGemm,
    {M, N, K},
    {(ElementT const*)A, K},
    {(ElementT const*)w1, K},
    cutlass::TensorRef<ElementT const, cutlass::layout::RowMajor>(),
    cutlass::TensorRef<ElementT, cutlass::layout::RowMajor>(),
    {(ElementT const*)wGate, K},
    cutlass::TensorRef<ElementT const, cutlass::layout::RowMajor>(),
    cutlass::TensorRef<ElementT, cutlass::layout::RowMajor>(),
    {(ElementT*)out, N},
    EpilogueOutputOp01::Params(),
    EpilogueOutputOp01::Params(),
    ClippedEpilogueOutputOp2::Params(clip),
    1);
}

bool argsAreImplementable(const DualGemm::Arguments& args) {
  if(DualGemm::can_implement(args) != cutlass::Status::kSuccess)
    return false;
  // With split-K off no workspace is required. A nonzero requirement would mean the
  // configuration changed, so treat it as unsupported rather than allocating here.
  if(DualGemm::get_workspace_size(args) != 0)
    return false;
  return true;
}

bool clippedArgsAreImplementable(const ClippedDualGemm::Arguments& args) {
  if(ClippedDualGemm::can_implement(args) != cutlass::Status::kSuccess)
    return false;
  if(ClippedDualGemm::get_workspace_size(args) != 0)
    return false;
  return true;
}

bool validProblem(int M, int N, int K) {
  if(M <= 0 || N <= 0 || K <= 0)
    return false;
  // CUTLASS stores all three dimensions and leading dimensions as int. Keep
  // the total element counts representable as well so later backend buffer
  // arithmetic cannot accept a problem that this ABI cannot describe safely.
  const int64_t activationElements = (int64_t)M * K;
  const int64_t weightElements = (int64_t)N * K;
  const int64_t outputElements = (int64_t)M * N;
  return activationElements < INT32_MAX && weightElements < INT32_MAX &&
    outputElements < INT32_MAX;
}

bool validClip(float clip) {
  return clip == 0.0f || (clip > 0.0f && std::isfinite(clip));
}

const half* alignedConstSentinel() {
  return reinterpret_cast<const half*>(uintptr_t(256));
}

half* alignedMutableSentinel() {
  return reinterpret_cast<half*>(uintptr_t(256));
}

}  // namespace

namespace CudaFusedFFN {

bool supportedOnCurrentDevice() {
  // Never launch on a pre-sm_80 device: unlike KataGo's own arch-guarded kernels, CUTLASS's
  // below-arch fallback path is not an empty stub but a device-side trap
  // (CUTLASS_NOT_IMPLEMENTED), which would leave a sticky, unclearable error on the context.
  {
    int device = 0;
    int major = 0;
    if(cudaGetDevice(&device) != cudaSuccess)
      return false;
    if(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess)
      return false;
    if(major < 8)
      return false;
  }
  // Run a real tiny dual GEMM rather than a trivial probe kernel: this exercises the actual
  // kernel launch, including its 48KB+ dynamic shared memory opt-in, which a lighter probe
  // would not. With all-zero inputs the epilogue writes SiLU(0)*0 == 0, so pre-filling the
  // output with a nonzero pattern and checking it became zero verifies the kernel body truly
  // executed (a launch of an empty JIT stub would still report success).
  constexpr int M = 16;
  constexpr int N = 64;
  constexpr int K = 64;
  constexpr size_t numIn = (size_t)M * K + 2 * (size_t)N * K;
  constexpr size_t numOut = (size_t)M * N;
  half* buf = nullptr;
  if(cudaMalloc(&buf, (numIn + numOut) * sizeof(half)) != cudaSuccess)
    return false;
  half* A = buf;
  half* w1 = A + (size_t)M * K;
  half* wGate = w1 + (size_t)N * K;
  half* out = wGate + (size_t)N * K;

  bool ok = cudaMemset(buf, 0, numIn * sizeof(half)) == cudaSuccess;
  ok = ok && cudaMemset(out, 0xFF, numOut * sizeof(half)) == cudaSuccess;
  if(ok) {
    DualGemm::Arguments args = makeArgs(A, w1, wGate, out, M, N, K);
    ok = argsAreImplementable(args);
    if(ok) {
      DualGemm op;
      ok = op.initialize(args, nullptr, nullptr) == cutlass::Status::kSuccess;
      ok = ok && op.run(nullptr) == cutlass::Status::kSuccess;
    }
  }
  half hostOut[numOut];
  ok = ok && cudaMemcpy(hostOut, out, numOut * sizeof(half), cudaMemcpyDeviceToHost) == cudaSuccess;
  if(ok) {
    for(size_t i = 0; i < numOut; i++) {
      if(__half2float(hostOut[i]) != 0.0f) {
        ok = false;
        break;
      }
    }
  }
  // Clear any recoverable (non-sticky) launch error so an unsupported probe cannot leak error
  // state into later, unrelated CUDA calls. A device-side trap would be sticky and unclearable,
  // but the compute capability check above prevents the known trap path.
  (void)cudaGetLastError();
  (void)cudaFree(buf);
  return ok;
}

bool canImplement(
  const half* A, const half* w1, const half* wGate, half* out,
  int M, int N, int K, float clip
) {
  if(!validProblem(M,N,K) || !validClip(clip) || A == nullptr || w1 == nullptr ||
     wGate == nullptr || out == nullptr)
    return false;
  if((uintptr_t(A) | uintptr_t(w1) | uintptr_t(wGate) | uintptr_t(out)) & 15)
    return false;
  if(clip == 0.0f)
    return argsAreImplementable(makeArgs(A,w1,wGate,out,M,N,K));
  return clippedArgsAreImplementable(
    makeClippedArgs(A,w1,wGate,out,M,N,K,clip));
}

bool supportsProblem(int M, int N, int K, float clip) {
  const half* sentinel = alignedConstSentinel();
  return canImplement(
    sentinel,sentinel,sentinel,alignedMutableSentinel(),M,N,K,clip);
}

bool supportsPreparedWeights(
  const half* w1, const half* wGate, int M, int N, int K, float clip
) {
  return canImplement(
    alignedConstSentinel(),w1,wGate,alignedMutableSentinel(),M,N,K,clip);
}

void runSwiGLU(
  const half* A, const half* w1, const half* wGate, half* out,
  int M, int N, int K, float clip, cudaStream_t stream
) {
  if(!canImplement(A,w1,wGate,out,M,N,K,clip))
    throw std::runtime_error(
      "CudaFusedFFN::runSwiGLU: CUTLASS can_implement rejected the actual problem");
  cutlass::Status status = cutlass::Status::kErrorInternal;
  if(clip == 0.0f) {
    DualGemm::Arguments args = makeArgs(A, w1, wGate, out, M, N, K);
    DualGemm op;
    status = op.initialize(args, nullptr, stream);
    // run() repeats a cheap cudaFuncSetAttribute driver call on every invocation. Hoisting it
    // would require replicating the vendored kernel-params construction outside DualGemm, which
    // is not worth the coupling for ~1us of host time per launch.
    if(status == cutlass::Status::kSuccess)
      status = op.run(stream);
  }
  else {
    ClippedDualGemm::Arguments args =
      makeClippedArgs(A,w1,wGate,out,M,N,K,clip);
    ClippedDualGemm op;
    status = op.initialize(args,nullptr,stream);
    if(status == cutlass::Status::kSuccess)
      status = op.run(stream);
  }
  if(status != cutlass::Status::kSuccess) {
    // The caller checked shape support at model load, so this is a genuine failure (a CUDA
    // error or an unexpected argument rejection), never a condition to silently fall back on.
    // Note run() ends with cudaGetLastError, so kErrorInternal may also reflect a pending
    // async error from an earlier kernel on this stream rather than this GEMM itself.
    throw std::runtime_error(
      std::string("CUTLASS fused FFN kernel failed (or a prior CUDA error was pending): ") +
      cutlass::cutlassGetStatusString(status) +
      ", M=" + std::to_string(M) + " N=" + std::to_string(N) +
      " K=" + std::to_string(K) + " clip=" + std::to_string(clip));
  }
}

}  // namespace CudaFusedFFN

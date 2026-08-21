// Optional CUTLASS-based fused transformer FFN for the CUDA backend: one kernel computing
// SiLU(A @ W1) * (A @ Wgate) without writing the two intermediate GEMM outputs to global
// memory. Implemented in cudafusedffn.cu, which is compiled only for the optional upstream
// fused-FFN build (CMake defines KATAGO_ENABLE_OFFICIAL_CUDA_FUSED_FFN), so call sites must be
// guarded on that define.

#ifndef NEURALNET_CUDAFUSEDFFN_H_
#define NEURALNET_CUDAFUSEDFFN_H_

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace CudaFusedFFN {
  // Whether the fused FFN kernel actually works on the current device: runs a tiny dual GEMM
  // and verifies it executed, guarding against PTX JIT-compiling an unsupported architecture
  // to an empty stub as well as any failure to launch with its large dynamic shared memory
  // requirement. Synchronous and slightly costly, so call once per handle creation.
  bool supportedOnCurrentDevice();

  // Construction-time problem contract. This calls the real CUTLASS
  // can_implement() entry point with aligned sentinel operands, including the
  // configured maximum M, so model loading never guesses support from a list
  // of known channel sizes.
  // clip == 0 selects the unchanged upstream no-clip epilogue. A finite
  // positive clip selects the runtime-clipped FP32 epilogue. Every other
  // value is outside this optional kernel family's semantic contract.
  bool supportsProblem(int M, int N, int K, float clip);

  // Construction-time packed-weight contract. This calls can_implement() with
  // the actual two device weight pointers and aligned sentinel activation and
  // output pointers. It is intentionally separate from supportsProblem(): a
  // per-layer allocation can satisfy the shape contract but still violate an
  // operand-alignment contract.
  bool supportsPreparedWeights(
    const half* w1, const half* wGate, int M, int N, int K, float clip
  );

  // Launch-time contract using the actual M and every actual device pointer.
  // Auto dispatch may safely fall back only while this returns false, before
  // the fused kernel has been enqueued. A failure after launch is an execution
  // error and must not be hidden by replaying a generic kernel.
  bool canImplement(
    const half* A, const half* w1, const half* wGate, half* out,
    int M, int N, int K, float clip
  );

  // With clip == 0, out = SiLU(A @ W1) * (A @ Wgate), with FP16 io/GEMM
  // accumulation and the explicitly qualified fast FP16 SiLU/product upstream
  // epilogue. This is an approximate FP16 recipe, not a bit-exact alias of the
  // generic FP32 SwiGLU helper. With finite clip > 0, the separate runtime-
  // clipped epilogue computes in FP32, in the same order as the generic helper:
  // clamp(SiLU(A @ W1), +/-clip) * clamp(A @ Wgate, +/-clip), followed by one
  // round-to-nearest-even conversion to FP16. A is [M, K] row-major (NHWC
  // tokens), w1/wGate are packed
  // out-major ([N, K] row-major), out is [M, N] row-major. All pointers must be
  // 16-byte aligned.
  // The caller must have verified supportedOnCurrentDevice(),
  // supportsProblem(), supportsPreparedWeights(), and canImplement() at their
  // respective lifecycle points. Any failure here is therefore a genuine
  // execution error and throws std::runtime_error.
  void runSwiGLU(
    const half* A, const half* w1, const half* wGate, half* out,
    int M, int N, int K, float clip, cudaStream_t stream
  );
}

#endif  // NEURALNET_CUDAFUSEDFFN_H_

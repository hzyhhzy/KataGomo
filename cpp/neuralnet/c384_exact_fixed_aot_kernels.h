#ifndef KATAGO_C384_EXACT_FIXED_AOT_KERNELS_H_
#define KATAGO_C384_EXACT_FIXED_AOT_KERNELS_H_

#include "../neuralnet/c384_exact_fixed_aot_plan.h"
#include "../neuralnet/cudaincludes.h"

#include <cstddef>

namespace C384ExactFixedAot {

// input: [B*225,384], packedWeights: row-major [384,1152] with stride
// (1152,1), not three planar [384,384] matrices. cosSin is
// [225,192] half2(cos,sin), where 192 = Hkv*(D/2). output is packed by
// token as [B*225,1152] = [Q384,K384,V384] and therefore may only feed the
// exact same-batch FA4 consumer qualified by the CPU selector.
using QkvRopeLaunchFn = cudaError_t (*)(
  const half* input,
  const half* packedWeights,
  const half2* cosSin,
  half* packedQkvOutput,
  cudaStream_t stream
);

// The generated dual kernel uses the old exact-AOT scanner ABI. For paired
// records, pairedWeights stores 64-column up/gate chunks and
// unusedGateWeights must be ignored. output is [B*225,1024].
using DualFfnLaunchFn = cudaError_t (*)(
  const half* input,
  const half* pairedWeights,
  const half* unusedGateWeights,
  half* output,
  cudaStream_t stream
);

// Must eagerly load/configure the generated module for the current CUDA
// device during block construction. It must be idempotent per device and must
// not enqueue inference work. A null hook is an invalid generated descriptor.
using EagerPrepareFn = cudaError_t (*)(int deviceOrdinal);

struct QkvRopeTactic {
  TacticKey key;
  EagerPrepareFn eagerPrepare;
  QkvRopeLaunchFn launch;
};

struct DualFfnTactic {
  TacticKey key;
  EagerPrepareFn eagerPrepare;
  DualFfnLaunchFn launch;
};

static_assert(offsetof(QkvRopeTactic,key) == 0,
  "TacticKey must remain the first QKV descriptor member");
static_assert(offsetof(DualFfnTactic,key) == 0,
  "TacticKey must remain the first dual-FFN descriptor member");

// A normal build links the checked-in empty provider. A search build replaces
// only that provider with generated exact-B entries via CMake. Lookup is
// construction-time only; PreparedCudaSelection is stored per block and no
// registry scan, allocation, or kernel initialization occurs on the hot path.
const QkvRopeTactic* generatedQkvRopeTactics(std::size_t& count);
const DualFfnTactic* generatedDualFfnTactics(std::size_t& count);

struct PreparedCudaSelection {
  const QkvRopeTactic* qkvRope = nullptr;
  const PreparedPackedFa4* packedFa4 = nullptr;
  const DualFfnTactic* dualFfn = nullptr;
  RejectReason qkvRopeReason = RejectReason::ShapeMismatch;
  RejectReason dualFfnReason = RejectReason::ShapeMismatch;
};

PreparedCudaSelection prepareCudaSelection(
  const RuntimeShape& shape,
  const char* requestedQkvRopeId,
  const char* requestedDualFfnId,
  const PreparedPackedFa4* preparedPackedFa4
);

}  // namespace C384ExactFixedAot

#endif  // KATAGO_C384_EXACT_FIXED_AOT_KERNELS_H_

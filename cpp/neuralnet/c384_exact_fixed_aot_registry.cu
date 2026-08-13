#include "../neuralnet/c384_exact_fixed_aot_kernels.h"

#include <type_traits>

namespace C384ExactFixedAot {

static_assert(std::is_standard_layout<QkvRopeTactic>::value,
  "QKV descriptor must remain standard-layout");
static_assert(std::is_standard_layout<DualFfnTactic>::value,
  "dual-FFN descriptor must remain standard-layout");

PreparedCudaSelection prepareCudaSelection(
  const RuntimeShape& shape,
  const char* requestedQkvRopeId,
  const char* requestedDualFfnId,
  const PreparedPackedFa4* preparedPackedFa4
) {
  std::size_t qkvCount = 0;
  std::size_t dualCount = 0;
  const QkvRopeTactic* qkv = generatedQkvRopeTactics(qkvCount);
  const DualFfnTactic* dual = generatedDualFfnTactics(dualCount);
  const RegistryView registry = {
    {qkv,qkvCount,sizeof(QkvRopeTactic)},
    {dual,dualCount,sizeof(DualFfnTactic)},
  };
  const Selection selected = select(
    shape,registry,requestedQkvRopeId,requestedDualFfnId,preparedPackedFa4);

  PreparedCudaSelection result;
  result.qkvRopeReason = selected.qkvRope.reason;
  result.dualFfnReason = selected.dualFfn.reason;
  if(selected.qkvRope.selected()) {
    result.qkvRope = reinterpret_cast<const QkvRopeTactic*>(selected.qkvRope.tactic);
    if(result.qkvRope->nativeAbiVersion != kQkvRopeNativeAbiVersion ||
       result.qkvRope->eagerPrepare == nullptr || result.qkvRope->launch == nullptr) {
      result.qkvRope = nullptr;
      result.qkvRopeReason = RejectReason::InvalidImplementation;
    }
    else if(result.qkvRope->eagerPrepare(shape.deviceOrdinal) != cudaSuccess) {
      result.qkvRope = nullptr;
      result.qkvRopeReason = RejectReason::PreparationFailed;
    }
    else
      result.packedFa4 = selected.packedFa4;
  }
  if(selected.dualFfn.selected()) {
    result.dualFfn = reinterpret_cast<const DualFfnTactic*>(selected.dualFfn.tactic);
    if(result.dualFfn->nativeAbiVersion != kDualFfnNativeAbiVersion ||
       result.dualFfn->eagerPrepare == nullptr || result.dualFfn->launch == nullptr) {
      result.dualFfn = nullptr;
      result.dualFfnReason = RejectReason::InvalidImplementation;
    }
    else if(result.dualFfn->eagerPrepare(shape.deviceOrdinal) != cudaSuccess) {
      result.dualFfn = nullptr;
      result.dualFfnReason = RejectReason::PreparationFailed;
    }
  }
  return result;
}

}  // namespace C384ExactFixedAot

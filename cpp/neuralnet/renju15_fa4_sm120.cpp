#include "renju15_fa4_sm120.h"

#include "renju15_fa4_b36_tn128.h"

#include <cmath>
#include <mutex>

namespace Renju15Fa4Sm120 {
namespace {

r15fa4b36n128_Kernel_Module_t module = {};
std::once_flag moduleOnce;

cudaError_t invoke(
  half* q,
  half* k,
  half* v,
  half* output,
  float scale,
  cudaStream_t stream
) {
  std::call_once(moduleOnce, []() {
    r15fa4b36n128_Kernel_Module_Load(&module);
  });
  r15fa4b36n128_Tensor_mQ_t tensorQ = {(void*)q};
  r15fa4b36n128_Tensor_mK_t tensorK = {(void*)k};
  r15fa4b36n128_Tensor_mV_t tensorV = {(void*)v};
  r15fa4b36n128_Tensor_mO_t tensorOutput = {(void*)output};
  const int32_t status = cute_dsl_r15fa4b36n128_wrapper(
    &module, &tensorQ, &tensorK, &tensorV, &tensorOutput, scale, stream);
  return status == 0 ? cudaPeekAtLastError() : cudaErrorUnknown;
}

} // namespace

const char* tacticName(Tactic tactic) {
  switch(tactic) {
  case Tactic::B36Tm128Tn128S1Both16:
    return "tm128-tn128-s1-both16-b36-s225-h8-d32";
  case Tactic::Disabled:
    return "disabled";
  default:
    return "invalid";
  }
}

LaunchResult launch(
  Tactic tactic,
  half* q,
  half* k,
  half* v,
  half* output,
  int batch,
  int seq,
  int heads,
  int kvHeads,
  int qHeadDim,
  int vHeadDim,
  bool usingFp16,
  bool usingNhwc,
  const void* mask,
  bool recipeEligibleNoMask,
  int computeMajor,
  int computeMinor,
  cudaStream_t stream
) {
  if(tactic != Tactic::B36Tm128Tn128S1Both16 ||
     !recipeEligibleNoMask || computeMajor != 12 || computeMinor != 0 ||
     !usingFp16 || !usingNhwc || mask != nullptr || batch != 36 ||
     seq != 225 || heads != 8 || kvHeads != 8 ||
     qHeadDim != 32 || vHeadDim != 32)
    return {false, cudaSuccess, nullptr};
  if(q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
     stream == nullptr)
    return {true, cudaErrorInvalidValue, tacticName(tactic)};
  const float scale = 1.0f / std::sqrt(32.0f);
  return {true, invoke(q, k, v, output, scale, stream), tacticName(tactic)};
}

} // namespace Renju15Fa4Sm120


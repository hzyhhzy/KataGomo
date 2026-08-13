#ifndef KATAGO_C384_EXACT_FIXED_AOT_WEIGHTS_H_
#define KATAGO_C384_EXACT_FIXED_AOT_WEIGHTS_H_

#include "plan.h"

#include <cstddef>
#include <vector>

namespace C384ExactFixedAot {

constexpr int kQkvPlanes = 3;
constexpr int kDualFfnPairColumns = 64;
constexpr int kQkvPackedColumns = kQkvPlanes * kChannels;
constexpr int kDualFfnPackedColumns = 2 * kFfnChannels;

static_assert(kChannels % 16 == 0,"C384 exact weights require 16-column alignment");
static_assert(kFfnChannels % kDualFfnPairColumns == 0,
  "F1024 exact weights require complete 64-column up/gate pairs");

// Source matrices use the MatMulLayer host layout: row-major [input,output].
// Q/K/V are interleaved per input row as Q384,K384,V384, yielding
// row-major [384,1152].
std::size_t packedQkvWeightIndex(int inputChannel, int plane, int outputChannel);
std::vector<float> packQkvWeights(
  const std::vector<float>& q,
  const std::vector<float>& k,
  const std::vector<float>& v
);

// Source up/gate matrices are each row-major [384,1024]. The packed matrix is
// row-major [384,2048], with every 64 output columns stored as up64,gate64.
std::size_t packedDualFfnWeightIndex(
  int inputChannel,
  bool gate,
  int outputChannel
);
std::vector<float> packDualFfnWeights(
  const std::vector<float>& up,
  const std::vector<float>& gate
);

}  // namespace C384ExactFixedAot

#endif  // KATAGO_C384_EXACT_FIXED_AOT_WEIGHTS_H_

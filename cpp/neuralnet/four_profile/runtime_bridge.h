#ifndef NEURALNET_FOUR_PROFILE_RUNTIME_BRIDGE_H_
#define NEURALNET_FOUR_PROFILE_RUNTIME_BRIDGE_H_

#include <vector>

namespace FourProfile {

inline int normalizeGpuIndexV1(int gpuIdx) noexcept {
  return gpuIdx < 0 ? 0 : gpuIdx;
}

inline int countSameGpuConcurrencyV1(
  const std::vector<int>& gpuIdxByServerThread,
  int gpuIdxForThisThread
) noexcept {
  const int normalizedGpuIdx = normalizeGpuIndexV1(gpuIdxForThisThread);
  int sameGpuConcurrency = 0;
  for(int configuredGpuIdx: gpuIdxByServerThread) {
    if(normalizeGpuIndexV1(configuredGpuIdx) == normalizedGpuIdx)
      sameGpuConcurrency++;
  }
  return sameGpuConcurrency;
}

}  // namespace FourProfile

#endif  // NEURALNET_FOUR_PROFILE_RUNTIME_BRIDGE_H_

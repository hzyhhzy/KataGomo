#ifndef NEURALNET_FOUR_PROFILE_P4_PROVIDER_H_
#define NEURALNET_FOUR_PROFILE_P4_PROVIDER_H_

#include "provider_v1.h"

#include <memory>

namespace FourProfile {

// C384/H12 v105 INT8 provider. The transformer span depth is intentionally
// dynamic; only the per-pair shape and the exact B28/S225 runtime are fixed.
std::unique_ptr<FactoryV1> makeP4FactoryV1();

}  // namespace FourProfile

#endif  // NEURALNET_FOUR_PROFILE_P4_PROVIDER_H_

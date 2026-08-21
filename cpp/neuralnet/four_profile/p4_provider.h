#ifndef NEURALNET_FOUR_PROFILE_P4_PROVIDER_H_
#define NEURALNET_FOUR_PROFILE_P4_PROVIDER_H_

#include "provider_v1.h"

#include <memory>

namespace FourProfile {

// C384/H12 v105 INT8 providers. Transformer depth is dynamic. P4 retains the
// exact B28/S2 FA4 route; the generic factory covers other positive batches
// and board sizes using official MMA attention and the same quantized
// projections.
std::unique_ptr<FactoryV1> makeP4FactoryV1();
std::unique_ptr<FactoryV1> makeGenericC384Int8FactoryV1();

}  // namespace FourProfile

#endif  // NEURALNET_FOUR_PROFILE_P4_PROVIDER_H_

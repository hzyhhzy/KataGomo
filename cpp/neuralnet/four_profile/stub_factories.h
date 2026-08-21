#ifndef NEURALNET_FOUR_PROFILE_STUB_FACTORIES_H_
#define NEURALNET_FOUR_PROFILE_STUB_FACTORIES_H_

#include "registry.h"

namespace FourProfile {

// All four named factories are registered so matching and overlap semantics
// stay stable. P1 becomes real only when its optional SM120 package is linked;
// otherwise it remains unavailable. P2 is explicitly uncertified and P3/P4
// remain unavailable.
void registerBuiltinStubFactoriesV1(RegistryV1& registry);

}  // namespace FourProfile

#endif  // NEURALNET_FOUR_PROFILE_STUB_FACTORIES_H_

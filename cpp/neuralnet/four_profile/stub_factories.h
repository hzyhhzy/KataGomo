#ifndef NEURALNET_FOUR_PROFILE_STUB_FACTORIES_H_
#define NEURALNET_FOUR_PROFILE_STUB_FACTORIES_H_

#include "registry.h"

namespace FourProfile {

// First infrastructure commit: all four named factories are registered so
// matching and overlap semantics are stable before any exclusive kernel moves.
// P2 is explicitly uncertified; the other stubs are unavailable.
void registerBuiltinStubFactoriesV1(RegistryV1& registry);

}  // namespace FourProfile

#endif  // NEURALNET_FOUR_PROFILE_STUB_FACTORIES_H_

#ifndef NEURALNET_FOUR_PROFILE_STUB_FACTORIES_H_
#define NEURALNET_FOUR_PROFILE_STUB_FACTORIES_H_

#include "registry.h"

#include <memory>

namespace FourProfile {

// All four named factories are registered so matching and overlap semantics
// stay stable. P1 and P3 become real when their optional SM120 packages are
// linked; P2 remains uncertified and P4 remains unavailable.
void registerBuiltinStubFactoriesV1(
  RegistryV1& registry,
  std::unique_ptr<FactoryV1> p3Override = nullptr
);

}  // namespace FourProfile

#endif  // NEURALNET_FOUR_PROFILE_STUB_FACTORIES_H_

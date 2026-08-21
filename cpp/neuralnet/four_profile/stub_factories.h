#ifndef NEURALNET_FOUR_PROFILE_STUB_FACTORIES_H_
#define NEURALNET_FOUR_PROFILE_STUB_FACTORIES_H_

#include "registry.h"

#include <memory>

namespace FourProfile {

// All four named factories are registered so matching and overlap semantics
// stay stable. Optional linked implementations replace their corresponding
// unavailable stubs without changing registry order.
void registerBuiltinStubFactoriesV1(
  RegistryV1& registry,
  std::unique_ptr<FactoryV1> p3Override = nullptr,
  std::unique_ptr<FactoryV1> p4Override = nullptr
);

}  // namespace FourProfile

#endif  // NEURALNET_FOUR_PROFILE_STUB_FACTORIES_H_

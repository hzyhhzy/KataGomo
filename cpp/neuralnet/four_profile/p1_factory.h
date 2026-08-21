#ifndef NEURALNET_FOUR_PROFILE_P1_FACTORY_H_
#define NEURALNET_FOUR_PROFILE_P1_FACTORY_H_

#include "registry.h"

namespace FourProfile {

// Register exactly one P1 factory. CUDA builds with the qualified SM120
// package link a real provider; every other build keeps the same matcher and
// factory id but reports Unavailable, preserving strict official fallback.
void registerP1FactoryV1(RegistryV1& registry);

}  // namespace FourProfile

#endif  // NEURALNET_FOUR_PROFILE_P1_FACTORY_H_

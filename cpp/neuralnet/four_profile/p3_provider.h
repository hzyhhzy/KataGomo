#ifndef NEURALNET_FOUR_PROFILE_P3_PROVIDER_H_
#define NEURALNET_FOUR_PROFILE_P3_PROVIDER_H_

#include "provider_v1.h"

#include <memory>

namespace FourProfile {

// Built only by the CUDA backend when the pinned CUTLASS source is present.
// If the three immutable external packages are not linked, this factory still
// matches P3 but reports Unavailable, preserving strict official fallback.
std::unique_ptr<FactoryV1> makeP3FactoryV1();

}  // namespace FourProfile

#endif

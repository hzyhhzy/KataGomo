#ifndef NEURALNET_FOUR_PROFILE_P1_PROVIDER_H_
#define NEURALNET_FOUR_PROFILE_P1_PROVIDER_H_

#include "provider_v1.h"

#include <memory>

namespace FourProfile {

std::unique_ptr<ProviderV1> createP1ProviderV1(const ProfileKeyV1& key);

}  // namespace FourProfile

#endif  // NEURALNET_FOUR_PROFILE_P1_PROVIDER_H_

#ifndef NEURALNET_FOUR_PROFILE_P1_PROVIDER_H_
#define NEURALNET_FOUR_PROFILE_P1_PROVIDER_H_

#include "provider_v1.h"

#include <memory>

namespace FourProfile {

std::unique_ptr<ProviderV1> createP1ProviderV1(const ProfileKeyV1& key);

#if defined(KATAGO_ENABLE_P2_SM120_PROVIDER) && KATAGO_ENABLE_P2_SM120_PROVIDER
// P1 and P2 share the C256 provider implementation and differ only in their
// exact runtime shape, QKV tactic, and FA4 launcher.
std::unique_ptr<ProviderV1> createP2ProviderV1(const ProfileKeyV1& key);
#endif

}  // namespace FourProfile

#endif  // NEURALNET_FOUR_PROFILE_P1_PROVIDER_H_

#ifndef NEURALNET_FOUR_PROFILE_MODE_H_
#define NEURALNET_FOUR_PROFILE_MODE_H_

#include "provider_v1.h"

namespace FourProfile {

// nullptr means the documented default, Auto. Non-null settings are strict:
// only the exact lowercase strings off, auto, and required are accepted.
ModeV1 parseModeSettingV1(const char* setting);
const char* modeNameV1(ModeV1 mode);

}  // namespace FourProfile

#endif  // NEURALNET_FOUR_PROFILE_MODE_H_

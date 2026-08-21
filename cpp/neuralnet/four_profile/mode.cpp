#include "mode.h"

#include <cstring>

namespace FourProfile {

ModeV1 parseModeSettingV1(const char* setting) {
  if(setting == nullptr || std::strcmp(setting,"auto") == 0)
    return ModeV1::Auto;
  if(std::strcmp(setting,"off") == 0)
    return ModeV1::Off;
  if(std::strcmp(setting,"required") == 0)
    return ModeV1::Required;
  throw ErrorV1(
    "invalid KATAGO_FOUR_PROFILE_MODE; expected exact off|auto|required"
  );
}

const char* modeNameV1(ModeV1 mode) {
  switch(mode) {
  case ModeV1::Off: return "off";
  case ModeV1::Auto: return "auto";
  case ModeV1::Required: return "required";
  }
  throw FatalErrorV1("invalid four-profile mode enum");
}

}  // namespace FourProfile

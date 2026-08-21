#ifndef NEURALNET_FOUR_PROFILE_NATIVE_MODEL_VIEW_H_
#define NEURALNET_FOUR_PROFILE_NATIVE_MODEL_VIEW_H_

#include "provider_v1.h"
#include "../desc.h"

#include <utility>
#include <vector>

namespace FourProfile {

// Executable objects are backend-owned and remain available as the complete
// official route. The kind is repeated so descriptor/executable traversal
// cannot silently desynchronize.
using NativeExecutableBlocksV1 = std::vector<std::pair<int,const void*>>;

ModelViewV1 buildNativeModelViewV1(
  const ModelDesc& model,
  const NativeExecutableBlocksV1& executableBlocks
);

}  // namespace FourProfile

#endif  // NEURALNET_FOUR_PROFILE_NATIVE_MODEL_VIEW_H_

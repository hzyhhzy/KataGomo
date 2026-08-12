#ifndef NEURALNET_INT8POLICY_H_
#define NEURALNET_INT8POLICY_H_

#include <string>

class ConfigParser;

namespace NeuralNet {

struct CudaInt8Policy {
  bool configEnabled;
  bool environmentDisabled;
  bool enabled;
};

// Global cudaUseINT8 defaults to true. cudaUseINT8-<modelIndex>, when
// present, overrides it for that model while both values are still parsed.
bool loadCudaUseINT8(ConfigParser& cfg, const std::string& modelIndex);

// Backends other than CUDA deliberately ignore cudaUseINT8 while marking it
// consumed, matching the existing behavior for backend-specific config keys.
void ignoreCudaUseINT8(ConfigParser& cfg);

// KATAGO_DISABLE_INT8 is a one-way emergency kill switch. Only nullptr, "0",
// and "1" are accepted so a misspelled deployment override fails closed.
CudaInt8Policy resolveCudaInt8Policy(
  bool configEnabled,
  const char* disableEnvironmentValue
);

}  // namespace NeuralNet

#endif  // NEURALNET_INT8POLICY_H_

#include "../neuralnet/int8policy.h"

#include "../core/config_parser.h"
#include "../core/global.h"

#include <cstring>

using namespace std;

bool NeuralNet::loadCudaUseINT8(
  ConfigParser& cfg,
  const string& modelIndex
) {
  bool enabled = true;
  if(cfg.contains("cudaUseINT8"))
    enabled = cfg.getBool("cudaUseINT8");

  const string modelKey = "cudaUseINT8-" + modelIndex;
  if(cfg.contains(modelKey))
    enabled = cfg.getBool(modelKey);
  return enabled;
}

void NeuralNet::ignoreCudaUseINT8(ConfigParser& cfg) {
  cfg.markAllKeysUsedWithPrefix("cudaUseINT8");
}

NeuralNet::CudaInt8Policy NeuralNet::resolveCudaInt8Policy(
  bool configEnabled,
  const char* disableEnvironmentValue
) {
  bool environmentDisabled = false;
  if(disableEnvironmentValue == nullptr ||
     std::strcmp(disableEnvironmentValue,"0") == 0) {
    environmentDisabled = false;
  }
  else if(std::strcmp(disableEnvironmentValue,"1") == 0) {
    environmentDisabled = true;
  }
  else {
    throw StringError("KATAGO_DISABLE_INT8 must be exactly 0 or 1");
  }

  return CudaInt8Policy{
    configEnabled,
    environmentDisabled,
    configEnabled && !environmentDisabled
  };
}

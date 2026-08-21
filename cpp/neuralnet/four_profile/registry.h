#ifndef NEURALNET_FOUR_PROFILE_REGISTRY_H_
#define NEURALNET_FOUR_PROFILE_REGISTRY_H_

#include "provider_v1.h"

#include <memory>
#include <vector>

namespace FourProfile {

struct ResolutionV1 {
  const FactoryV1* factory = nullptr;

  bool matched() const { return factory != nullptr; }
};

class RegistryV1 {
public:
  RegistryV1() = default;
  RegistryV1(const RegistryV1&) = delete;
  RegistryV1& operator=(const RegistryV1&) = delete;

  void add(std::unique_ptr<FactoryV1> factory);
  ResolutionV1 resolve(const ProfileKeyV1& key) const;
  size_t size() const { return factories.size(); }

private:
  std::vector<std::unique_ptr<FactoryV1>> factories;
};

}  // namespace FourProfile

#endif  // NEURALNET_FOUR_PROFILE_REGISTRY_H_

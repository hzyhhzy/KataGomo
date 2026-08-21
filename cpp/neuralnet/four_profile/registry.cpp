#include "registry.h"

#include <cstring>

namespace FourProfile {

void RegistryV1::add(std::unique_ptr<FactoryV1> factory) {
  if(factory == nullptr)
    throw FatalErrorV1("four-profile registry rejected a null factory");
  const char* id = factory->factoryId();
  if(id == nullptr || id[0] == '\0')
    throw FatalErrorV1("four-profile registry rejected a factory without an id");
  for(const std::unique_ptr<FactoryV1>& existing: factories) {
    if(std::strcmp(existing->factoryId(),id) == 0)
      throw FatalErrorV1(std::string("four-profile registry duplicate factory id: ") + id);
  }
  factories.push_back(std::move(factory));
}

ResolutionV1 RegistryV1::resolve(const ProfileKeyV1& key) const {
  ResolutionV1 resolution;
  for(const std::unique_ptr<FactoryV1>& factory: factories) {
    if(!factory->matches(key))
      continue;
    if(resolution.factory != nullptr) {
      throw FatalErrorV1(
        std::string("four-profile registry overlap: ") +
        resolution.factory->factoryId() + " and " + factory->factoryId()
      );
    }
    resolution.factory = factory.get();
  }
  return resolution;
}

}  // namespace FourProfile

#ifndef NEURALNET_FOUR_PROFILE_MANAGER_H_
#define NEURALNET_FOUR_PROFILE_MANAGER_H_

#include "provider_v1.h"
#include "registry.h"

#include <memory>

namespace FourProfile {

struct BuildReportV1 {
  RouteV1 route = RouteV1::Official;
  ReasonV1 reason = ReasonV1::Unmatched;
  std::string profileId = "official";
  std::string detail;
  TransformerSpanV1 span;
  bool hasProfileKey = false;
  ProfileKeyV1 profileKey;
};

class ManagerV1 {
public:
  static std::unique_ptr<ManagerV1> create(
    const RegistryV1& registry,
    const ModelViewV1& model,
    const RuntimeKeyV1& runtime,
    ModeV1 mode
  );

  ManagerV1(const ManagerV1&) = delete;
  ManagerV1& operator=(const ManagerV1&) = delete;
  ~ManagerV1() = default;

  const BuildReportV1& report() const { return buildReport; }
  ReasonV1 lastRuntimeReason() const { return runtimeReason; }

  // A specialized return arms exactly one enqueue call. An official return
  // guarantees the provider enqueued nothing and the caller may execute the
  // complete official transformer span.
  RouteV1 preflight(const RuntimeCallV1& call);
  RouteV1 enqueue(const RuntimeCallV1& call);

private:
  ManagerV1(ModeV1 mode, BuildReportV1 report);

  RouteV1 runtimeFailure(ReasonV1 reason, const std::string& detail);

  ModeV1 mode;
  BuildReportV1 buildReport;
  ReasonV1 runtimeReason;
  std::unique_ptr<ProviderV1> provider;
  uint64_t planGeneration;
  uint64_t armedRunToken;
  RuntimeCallV1 armedCall;
  bool enqueueArmed;
};

}  // namespace FourProfile

#endif  // NEURALNET_FOUR_PROFILE_MANAGER_H_

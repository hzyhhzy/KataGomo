#include "manager.h"

#include <exception>
#include <limits>
#include <utility>

namespace FourProfile {
namespace {

std::string contextMessage(
  const char* phase,
  const char* profileId,
  const std::string& detail
) {
  std::string message = "four-profile ";
  message += phase;
  message += " failed for ";
  message += profileId == nullptr ? "<unknown>" : profileId;
  if(!detail.empty()) {
    message += ": ";
    message += detail;
  }
  return message;
}

BuildReportV1 officialReport(
  ReasonV1 reason,
  const TransformerSpanV1& span,
  const std::string& detail
) {
  BuildReportV1 report;
  report.reason = reason;
  report.span = span;
  report.detail = detail;
  return report;
}

void requireCompleteOfficialResources(
  const ModelViewV1& model,
  const TransformerSpanV1& span
) {
  for(size_t layer = 0; layer < span.pairCount; layer++) {
    const BlockViewV1& attention = model.blocks[span.beginBlock + layer * 2];
    const BlockViewV1& ffn = model.blocks[span.beginBlock + layer * 2 + 1];
    if(!attention.officialAttention.complete()) {
      throw FatalErrorV1(
        "four-profile model view is missing complete official attention resources at layer " +
        std::to_string(layer)
      );
    }
    if(!ffn.officialFfn.complete()) {
      throw FatalErrorV1(
        "four-profile model view is missing complete official FFN resources at layer " +
        std::to_string(layer)
      );
    }
  }
}

}  // namespace

ManagerV1::ManagerV1(ModeV1 mode_, BuildReportV1 report_)
  : mode(mode_),
    buildReport(std::move(report_)),
    runtimeReason(buildReport.reason),
    planGeneration(0),
    armedRunToken(0),
    enqueueArmed(false) {}

std::unique_ptr<ManagerV1> ManagerV1::create(
  const RegistryV1& registry,
  const ModelViewV1& model,
  const RuntimeKeyV1& runtime,
  ModeV1 mode
) {
  if(mode == ModeV1::Off) {
    TransformerSpanV1 disabledSpan;
    return std::unique_ptr<ManagerV1>(new ManagerV1(
      mode,officialReport(ReasonV1::Disabled,disabledSpan,"four-profile providers are disabled")
    ));
  }

  const TransformerSpanV1 span = analyzeTransformerSpanV1(model.blocks);
  if(!span.hasTransformer) {
    return std::unique_ptr<ManagerV1>(new ManagerV1(
      mode,officialReport(ReasonV1::NoTransformer,span,span.detail)
    ));
  }
  if(!span.valid) {
    // Invalid or split transformer topology is not a registered profile. It is
    // always kept on the strict official route, including Required mode.
    return std::unique_ptr<ManagerV1>(new ManagerV1(
      mode,officialReport(ReasonV1::InvalidTransformerSpan,span,span.detail)
    ));
  }

  requireCompleteOfficialResources(model,span);

  ProfileKeyV1 key;
  std::string keyDetail;
  if(!deriveProfileKeyV1(model,span,runtime,key,keyDetail)) {
    return std::unique_ptr<ManagerV1>(new ManagerV1(
      mode,officialReport(ReasonV1::NonUniformTransformerSpan,span,keyDetail)
    ));
  }

  BuildReportV1 report = officialReport(ReasonV1::Unmatched,span,"no registered profile matched");
  report.hasProfileKey = true;
  report.profileKey = key;

  auto failureOrOfficial = [&](ReasonV1 reason, const char* profileId, const std::string& detail) {
    if(mode == ModeV1::Required)
      throw ErrorV1(contextMessage("construction",profileId,detail));
    report.route = RouteV1::Official;
    report.reason = reason;
    report.profileId = "official";
    report.detail = detail;
    return std::unique_ptr<ManagerV1>(new ManagerV1(mode,std::move(report)));
  };

  const ResolutionV1 resolution = registry.resolve(key);
  if(!resolution.matched()) {
    // Required means a matched provider may not silently degrade. It does not
    // reinterpret an unmatched model as a specialization candidate.
    return std::unique_ptr<ManagerV1>(new ManagerV1(mode,std::move(report)));
  }

  const FactoryV1& factory = *resolution.factory;
  const AvailabilityResultV1 availability = factory.availability(key);
  if(availability.availability != AvailabilityV1::Available) {
    const ReasonV1 reason = availability.availability == AvailabilityV1::Uncertified ?
      ReasonV1::ProviderUncertified : ReasonV1::ProviderUnavailable;
    const std::string detail = availability.detail.empty() ?
      std::string("factory is ") + availabilityNameV1(availability.availability) :
      availability.detail;
    return failureOrOfficial(reason,factory.factoryId(),detail);
  }

  std::unique_ptr<ProviderV1> provider;
  try {
    provider = factory.create(key);
  }
  catch(const FatalErrorV1&) {
    throw;
  }
  catch(const std::exception& e) {
    return failureOrOfficial(ReasonV1::ProviderCreationFailed,factory.factoryId(),e.what());
  }
  if(provider == nullptr) {
    return failureOrOfficial(
      ReasonV1::ProviderCreationFailed,factory.factoryId(),
      "available factory returned no provider"
    );
  }

  PreparedSpanV1 prepared;
  prepared.attention.reserve(span.pairCount);
  prepared.ffn.reserve(span.pairCount);
  for(size_t layer = 0; layer < span.pairCount; layer++) {
    const BlockViewV1& attentionBlock = model.blocks[span.beginBlock + layer * 2];
    const BlockViewV1& ffnBlock = model.blocks[span.beginBlock + layer * 2 + 1];

    PrepareAttentionResultV1 attentionResult;
    try {
      attentionResult = provider->prepareAttention(
        layer,
        attentionBlock.attention,
        attentionBlock.attentionScalars,
        attentionBlock.officialAttention
      );
    }
    catch(const FatalErrorV1&) {
      throw;
    }
    catch(const std::exception& e) {
      attentionResult.detail = e.what();
    }
    if(attentionResult.prepared == nullptr) {
      const std::string detail = attentionResult.detail.empty() ?
        "attention prepare returned no staged resource at layer " + std::to_string(layer) :
        attentionResult.detail;
      return failureOrOfficial(ReasonV1::PrepareFailed,factory.factoryId(),detail);
    }
    prepared.attention.push_back(std::move(attentionResult.prepared));

    PrepareFfnResultV1 ffnResult;
    try {
      ffnResult = provider->prepareFfn(
        layer,ffnBlock.ffn,ffnBlock.ffnScalars,ffnBlock.officialFfn
      );
    }
    catch(const FatalErrorV1&) {
      throw;
    }
    catch(const std::exception& e) {
      ffnResult.detail = e.what();
    }
    if(ffnResult.prepared == nullptr) {
      const std::string detail = ffnResult.detail.empty() ?
        "FFN prepare returned no staged resource at layer " + std::to_string(layer) :
        ffnResult.detail;
      return failureOrOfficial(ReasonV1::PrepareFailed,factory.factoryId(),detail);
    }
    prepared.ffn.push_back(std::move(ffnResult.prepared));
  }

  std::string commitDetail;
  bool committed = false;
  try {
    committed = provider->commit(std::move(prepared),commitDetail);
  }
  catch(const FatalErrorV1&) {
    throw;
  }
  catch(const std::exception& e) {
    commitDetail = e.what();
  }
  if(!committed) {
    if(commitDetail.empty())
      commitDetail = "provider rejected atomic span commit";
    return failureOrOfficial(ReasonV1::CommitFailed,factory.factoryId(),commitDetail);
  }

  const uint64_t committedGeneration = provider->committedPlanGeneration();
  if(committedGeneration == 0) {
    return failureOrOfficial(
      ReasonV1::CommitFailed,factory.factoryId(),
      "committed provider did not publish a nonzero plan generation"
    );
  }

  report.route = RouteV1::Specialized;
  report.reason = ReasonV1::Selected;
  report.profileId = provider->profileId() == nullptr ? factory.factoryId() : provider->profileId();
  report.detail = "all N/N resources prepared and committed atomically";
  std::unique_ptr<ManagerV1> manager(new ManagerV1(mode,std::move(report)));
  manager->planGeneration = committedGeneration;
  manager->provider = std::move(provider);
  return manager;
}

RouteV1 ManagerV1::runtimeFailure(ReasonV1 reason, const std::string& detail) {
  runtimeReason = reason;
  enqueueArmed = false;
  armedRunToken = 0;
  if(mode == ModeV1::Required)
    throw ErrorV1(contextMessage("runtime",buildReport.profileId.c_str(),detail));
  return RouteV1::Official;
}

RouteV1 ManagerV1::preflight(const RuntimeCallV1& call) {
  if(buildReport.route == RouteV1::Official)
    return RouteV1::Official;
  if(provider == nullptr)
    throw FatalErrorV1("four-profile selected route has no committed provider");
  if(enqueueArmed)
    throw FatalErrorV1("four-profile preflight called twice without enqueue");
  if(call.key != buildReport.profileKey.runtime) {
    return runtimeFailure(
      ReasonV1::RuntimePreflightFailed,
      "runtime key changed after provider preparation"
    );
  }
  const int64_t sequenceSize =
    static_cast<int64_t>(call.key.boardX) * static_cast<int64_t>(call.key.boardY);
  const bool sequenceSizeValid =
    call.key.boardX > 0 && call.key.boardY > 0 &&
    sequenceSize <= std::numeric_limits<int>::max() &&
    call.sequenceSize == static_cast<int>(sequenceSize);
  const bool maskEvidenceValid =
    (call.key.maskMode == MaskModeV1::None && call.key.maskNull && call.mask == nullptr) ||
    (call.key.maskMode == MaskModeV1::Dense && !call.key.maskNull && call.mask != nullptr);
  if(call.actualBatchSize <= 0 ||
     call.actualBatchSize != call.key.physicalBatchSize ||
     !sequenceSizeValid ||
     call.transformerBeginBlock != buildReport.span.beginBlock ||
     call.transformerPairCount != buildReport.span.pairCount ||
     !maskEvidenceValid ||
     call.stream == nullptr ||
     call.trunk == nullptr ||
     call.trunkScratch == nullptr ||
     ((call.workspace == nullptr) != (call.workspaceBytes == 0))) {
    return runtimeFailure(
      ReasonV1::RuntimePreflightFailed,
      "runtime B/S/mask/span evidence does not match the committed plan"
    );
  }

  ProviderOpResultV1 result;
  try {
    result = provider->preflight(call);
  }
  catch(const FatalErrorV1&) {
    throw;
  }
  catch(const std::exception& e) {
    // Only an explicit result can prove that preflight respected its
    // zero-enqueue contract. An exception has unknown side effects.
    throw FatalErrorV1(contextMessage(
      "preflight-exception",buildReport.profileId.c_str(),e.what()
    ));
  }
  catch(...) {
    throw FatalErrorV1(contextMessage(
      "preflight-exception",buildReport.profileId.c_str(),"non-standard exception"
    ));
  }
  if(result.enqueued != 0) {
    throw FatalErrorV1(contextMessage(
      "preflight-enqueue-contract",buildReport.profileId.c_str(),result.detail
    ));
  }
  if(!result.ok) {
    return runtimeFailure(
      ReasonV1::RuntimePreflightFailed,
      result.detail.empty() ? "provider preflight rejected runtime" : result.detail
    );
  }
  if(result.planGeneration != planGeneration || result.runToken == 0) {
    throw FatalErrorV1(contextMessage(
      "preflight-generation-contract",buildReport.profileId.c_str(),
      "provider returned a stale plan generation or zero run token"
    ));
  }

  runtimeReason = ReasonV1::Selected;
  armedCall = call;
  armedRunToken = result.runToken;
  enqueueArmed = true;
  return RouteV1::Specialized;
}

RouteV1 ManagerV1::enqueue(const RuntimeCallV1& call) {
  if(buildReport.route == RouteV1::Official)
    return RouteV1::Official;
  if(provider == nullptr)
    throw FatalErrorV1("four-profile selected route has no committed provider");
  if(!enqueueArmed)
    throw FatalErrorV1("four-profile enqueue called without a successful zero-enqueue preflight");
  if(!armedCall.sameIdentity(call)) {
    enqueueArmed = false;
    armedRunToken = 0;
    throw FatalErrorV1(
      "four-profile runtime call identity changed between preflight and enqueue"
    );
  }
  const uint64_t runToken = armedRunToken;
  enqueueArmed = false;
  armedRunToken = 0;

  ProviderOpResultV1 result;
  try {
    result = provider->enqueue(call,runToken);
  }
  catch(const FatalErrorV1&) {
    throw;
  }
  catch(const std::exception& e) {
    // Once provider enqueue has been entered, an exception cannot prove that
    // zero work was launched. Never run the official path after this point.
    throw FatalErrorV1(contextMessage(
      "enqueue-exception",buildReport.profileId.c_str(),e.what()
    ));
  }
  catch(...) {
    throw FatalErrorV1(contextMessage(
      "enqueue-exception",buildReport.profileId.c_str(),"non-standard exception"
    ));
  }
  if(result.ok) {
    if(result.enqueued == 0) {
      throw FatalErrorV1(contextMessage(
        "enqueue-contract",buildReport.profileId.c_str(),
        "provider reported specialized success without enqueueing work"
      ));
    }
    if(result.planGeneration != planGeneration || result.runToken != runToken) {
      throw FatalErrorV1(contextMessage(
        "enqueue-generation-contract",buildReport.profileId.c_str(),
        "provider completed against a stale plan generation or run token"
      ));
    }
    runtimeReason = ReasonV1::Selected;
    return RouteV1::Specialized;
  }
  if(result.enqueued != 0) {
    throw FatalErrorV1(contextMessage(
      "enqueue-after-work",buildReport.profileId.c_str(),result.detail
    ));
  }
  return runtimeFailure(
    ReasonV1::EnqueueFailedBeforeWork,
    result.detail.empty() ? "provider enqueue rejected before work" : result.detail
  );
}

}  // namespace FourProfile

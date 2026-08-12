#include "../neuralnet/cudaopregistry.h"

#include <algorithm>
#include <limits>
#include <tuple>

#include "../core/global.h"
#include "../core/sha2.h"
#include "../neuralnet/desc.h"

using namespace std;

namespace CudaOpRegistry {
namespace {

class FingerprintWriter {
 public:
  vector<uint8_t> bytes;

  void u8(uint8_t value) {
    bytes.push_back(value);
  }
  void u32(uint32_t value) {
    for(int i = 0; i < 4; i++)
      bytes.push_back((uint8_t)((value >> (i * 8)) & 0xFFu));
  }
  void i32(int32_t value) {
    u32((uint32_t)value);
  }
  void u64(uint64_t value) {
    for(int i = 0; i < 8; i++)
      bytes.push_back((uint8_t)((value >> (i * 8)) & 0xFFu));
  }
  void raw(const uint8_t* value, size_t len) {
    bytes.insert(bytes.end(),value,value + len);
  }
  void raw(const char* value, size_t len) {
    bytes.insert(bytes.end(),value,value + len);
  }
};

static string digestToHex(const array<uint8_t,32>& digest) {
  static const char* digits = "0123456789abcdef";
  string result(digest.size() * 2,'0');
  for(size_t i = 0; i < digest.size(); i++) {
    result[i * 2] = digits[digest[i] >> 4];
    result[i * 2 + 1] = digits[digest[i] & 0x0F];
  }
  return result;
}

static void encodeCapabilityKey(FingerprintWriter& out, const CapabilityKey& key) {
  out.u32(key.schemaVersion);
  out.u32((uint32_t)key.kind);
  out.u32(key.flags);
  out.u32((uint32_t)key.inputType);
  out.u32((uint32_t)key.outputType);
  out.u32((uint32_t)key.computeType);
  out.u32((uint32_t)key.layout);
  out.u32((uint32_t)key.maskMode);
  out.i32(key.batchSize);
  out.i32(key.spatialArea);
  out.i32(key.boardX);
  out.i32(key.boardY);
  out.i32(key.inChannels);
  out.i32(key.outChannels);
  out.i32(key.auxiliaryChannels);
  out.i32(key.kernelX);
  out.i32(key.kernelY);
  out.i32(key.dilationX);
  out.i32(key.dilationY);
  out.i32(key.numHeads);
  out.i32(key.numKVHeads);
  out.i32(key.qHeadDim);
  out.i32(key.vHeadDim);
  out.u32(key.semanticScalar0Bits);
  out.u32(key.semanticScalar1Bits);
  out.u32(key.deviceComputeCapability);
  out.u32(key.streamCount);
  out.u64(key.runtimeLibraryFingerprint);
}

static bool powerOfTwoOrZero(uint32_t value) {
  return value == 0 || (value & (value - 1)) == 0;
}

}  // namespace

bool CapabilityKey::operator==(const CapabilityKey& other) const {
  return
    schemaVersion == other.schemaVersion &&
    kind == other.kind &&
    flags == other.flags &&
    inputType == other.inputType &&
    outputType == other.outputType &&
    computeType == other.computeType &&
    layout == other.layout &&
    maskMode == other.maskMode &&
    batchSize == other.batchSize &&
    spatialArea == other.spatialArea &&
    boardX == other.boardX &&
    boardY == other.boardY &&
    inChannels == other.inChannels &&
    outChannels == other.outChannels &&
    auxiliaryChannels == other.auxiliaryChannels &&
    kernelX == other.kernelX &&
    kernelY == other.kernelY &&
    dilationX == other.dilationX &&
    dilationY == other.dilationY &&
    numHeads == other.numHeads &&
    numKVHeads == other.numKVHeads &&
    qHeadDim == other.qHeadDim &&
    vHeadDim == other.vHeadDim &&
    semanticScalar0Bits == other.semanticScalar0Bits &&
    semanticScalar1Bits == other.semanticScalar1Bits &&
    deviceComputeCapability == other.deviceComputeCapability &&
    streamCount == other.streamCount &&
    runtimeLibraryFingerprint == other.runtimeLibraryFingerprint;
}

bool CapabilityKey::operator!=(const CapabilityKey& other) const {
  return !(*this == other);
}

bool TacticId::operator==(const TacticId& other) const {
  return family == other.family && variant == other.variant;
}

bool TacticId::operator!=(const TacticId& other) const {
  return !(*this == other);
}

bool TacticId::operator<(const TacticId& other) const {
  return tie(family,variant) < tie(other.family,other.variant);
}

bool RecipeFingerprint::operator==(const RecipeFingerprint& other) const {
  return schemaVersion == other.schemaVersion && digest == other.digest;
}

bool RecipeFingerprint::operator!=(const RecipeFingerprint& other) const {
  return !(*this == other);
}

string RecipeFingerprint::toHex() const {
  return digestToHex(digest);
}

bool PlanFingerprint::operator==(const PlanFingerprint& other) const {
  return schemaVersion == other.schemaVersion && digest == other.digest;
}

bool PlanFingerprint::operator!=(const PlanFingerprint& other) const {
  return !(*this == other);
}

string PlanFingerprint::toHex() const {
  return digestToHex(digest);
}

void Registry::registerTactic(const TacticRegistration& registration) {
  if(registration.match == nullptr || registration.prepare == nullptr)
    throw StringError("Cannot register a CUDA tactic with null match or prepare callback");
  for(const TacticRegistration& existing: tactics) {
    if(existing.id == registration.id)
      throw StringError("Cannot register duplicate CUDA tactic id");
  }
  tactics.push_back(registration);
}

ResolveResult Registry::resolveAtConstruction(const OpRequest& request) const {
  struct Match {
    const TacticRegistration* registration;
    SupportClass support;
  };
  vector<Match> matches;
  matches.reserve(tactics.size());
  for(const TacticRegistration& tactic: tactics) {
    SupportClass support = tactic.match(request,tactic.userData);
    if(support != SupportClass::Unsupported)
      matches.push_back(Match{&tactic,support});
  }
  sort(matches.begin(),matches.end(),[](const Match& a, const Match& b) {
    if(a.support != b.support)
      return (uint32_t)a.support > (uint32_t)b.support;
    if(a.registration->priority != b.registration->priority)
      return a.registration->priority > b.registration->priority;
    return a.registration->id < b.registration->id;
  });

  for(const Match& match: matches) {
    PreparedOp prepared{};
    if(!match.registration->prepare(request,prepared,match.registration->userData))
      continue;
    if(!powerOfTwoOrZero(prepared.workspaceAlignment))
      continue;
    // Identity and proof level are registry-owned; callbacks may only fill the
    // resource requirements and implementation cookie.
    prepared.tactic = match.registration->id;
    prepared.recipe = match.registration->recipe;
    prepared.support = match.support;
    return ResolveResult{true,prepared};
  }
  return ResolveResult{};
}

size_t Registry::size() const {
  return tactics.size();
}

CapabilityKey makeCapabilityKey(
  const NeuralNetArchitecture::ArchitectureOpDesc& op,
  const RuntimeOpContext& runtime
) {
  if((op.runtimeDependencies & NeuralNetArchitecture::OP_RUNTIME_BATCH) != 0 && runtime.batchSize <= 0)
    throw StringError("Cannot construct capability key with nonpositive batch size");

  const bool usesArea =
    (op.runtimeDependencies & NeuralNetArchitecture::OP_RUNTIME_SPATIAL_AREA) != 0 ||
    (op.runtimeDependencies & NeuralNetArchitecture::OP_RUNTIME_SPATIAL_XY) != 0;
  if(usesArea && (runtime.boardX <= 0 || runtime.boardY <= 0))
    throw StringError("Cannot construct spatial capability key with nonpositive board dimensions");
  int64_t area64 = (int64_t)runtime.boardX * (int64_t)runtime.boardY;
  if(usesArea && area64 > numeric_limits<int32_t>::max())
    throw StringError("Cannot construct capability key: spatial area overflows int32");

  CapabilityKey key{};
  key.schemaVersion = CAPABILITY_KEY_SCHEMA_VERSION;
  key.kind = op.kind;
  key.flags = op.flags;
  key.inputType = runtime.inputType;
  key.outputType = runtime.outputType;
  key.computeType = runtime.computeType;
  key.layout = runtime.layout;
  key.maskMode = (op.runtimeDependencies & NeuralNetArchitecture::OP_RUNTIME_MASK) != 0 ?
    runtime.maskMode : MaskMode::None;
  key.batchSize = (op.runtimeDependencies & NeuralNetArchitecture::OP_RUNTIME_BATCH) != 0 ?
    runtime.batchSize : 0;
  key.spatialArea = usesArea ? (int32_t)area64 : 0;
  if((op.runtimeDependencies & NeuralNetArchitecture::OP_RUNTIME_SPATIAL_XY) != 0) {
    key.boardX = runtime.boardX;
    key.boardY = runtime.boardY;
  }
  key.inChannels = op.inChannels;
  key.outChannels = op.outChannels;
  key.auxiliaryChannels = op.auxiliaryChannels;
  key.kernelX = op.kernelX;
  key.kernelY = op.kernelY;
  key.dilationX = op.dilationX;
  key.dilationY = op.dilationY;
  key.numHeads = op.numHeads;
  key.numKVHeads = op.numKVHeads;
  key.qHeadDim = op.qHeadDim;
  key.vHeadDim = op.vHeadDim;
  key.semanticScalar0Bits = op.semanticScalar0Bits;
  key.semanticScalar1Bits = op.semanticScalar1Bits;
  key.deviceComputeCapability = runtime.deviceComputeCapability;
  key.streamCount = runtime.streamCount;
  key.runtimeLibraryFingerprint = runtime.runtimeLibraryFingerprint;
  return key;
}

vector<OpRequest> buildOpRequests(
  const NeuralNetArchitecture::ArchitectureDesc& architecture,
  const RuntimeOpContext& runtime
) {
  vector<OpRequest> result;
  result.reserve(architecture.operators.size());
  for(const NeuralNetArchitecture::ArchitectureOpDesc& op: architecture.operators) {
    OpRequest request{};
    request.key = makeCapabilityKey(op,runtime);
    request.architecture = architecture.signature;
    request.topologyIndex = op.topologyIndex;
    result.push_back(request);
  }
  return result;
}

vector<OpRequest> buildOpRequests(const ModelDesc& model, const RuntimeOpContext& runtime) {
  return buildOpRequests(NeuralNetArchitecture::buildArchitectureDesc(model),runtime);
}

RecipeFingerprint fingerprintRecipe(const uint8_t* bytes, size_t len) {
  if(bytes == nullptr && len != 0)
    throw StringError("Cannot fingerprint a null nonempty recipe");
  FingerprintWriter out;
  static const char magic[] = "KataGoCudaRecipe";
  out.raw(magic,sizeof(magic) - 1);
  out.u32(RECIPE_FINGERPRINT_SCHEMA_VERSION);
  out.u64((uint64_t)len);
  if(len > 0)
    out.raw(bytes,len);
  RecipeFingerprint result{};
  result.schemaVersion = RECIPE_FINGERPRINT_SCHEMA_VERSION;
  SHA2::get256(out.bytes.data(),out.bytes.size(),result.digest.data());
  return result;
}

RecipeFingerprint fingerprintRecipe(const string& stableRecipeEncoding) {
  return fingerprintRecipe(
    (const uint8_t*)stableRecipeEncoding.data(),
    stableRecipeEncoding.size()
  );
}

PlanFingerprint fingerprintPreparedPlan(
  const vector<OpRequest>& requests,
  const vector<PreparedOp>& prepared
) {
  if(requests.size() != prepared.size())
    throw StringError("Cannot fingerprint plan: request and prepared-op counts differ");
  FingerprintWriter out;
  static const char magic[] = "KataGoCudaPreparedPlan";
  out.raw(magic,sizeof(magic) - 1);
  out.u32(PLAN_FINGERPRINT_SCHEMA_VERSION);
  out.u64((uint64_t)requests.size());
  for(size_t i = 0; i < requests.size(); i++) {
    const OpRequest& request = requests[i];
    const PreparedOp& op = prepared[i];
    encodeCapabilityKey(out,request.key);
    out.u32(request.architecture.schemaVersion);
    out.raw(request.architecture.digest.data(),request.architecture.digest.size());
    out.u32(request.topologyIndex);
    out.u64(op.tactic.family);
    out.u64(op.tactic.variant);
    out.u32(op.recipe.schemaVersion);
    out.raw(op.recipe.digest.data(),op.recipe.digest.size());
    out.u32((uint32_t)op.support);
    out.u32(op.workspaceAlignment);
    out.u64(op.workspaceBytes);
  }
  PlanFingerprint result{};
  result.schemaVersion = PLAN_FINGERPRINT_SCHEMA_VERSION;
  SHA2::get256(out.bytes.data(),out.bytes.size(),result.digest.data());
  return result;
}

}  // namespace CudaOpRegistry

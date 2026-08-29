#include "p4_provider.h"

#include "../cudaincludes.h"
#include "../cudaandrocmhelpers.h"
#include "../desc.h"
#include "../cuda_specialized/sm120/c384/experimental_int8/kernels.h"
#include "../cuda_specialized/sm120/c384/experimental_int8/weights.h"
#include "../cuda_specialized/sm120/c384/fixed_batch/fa4.h"
#include "../../external/half-2.2.0/include/half.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifndef KATAGO_P4_EXTERNAL_PACKAGES_AVAILABLE
#define KATAGO_P4_EXTERNAL_PACKAGES_AVAILABLE 0
#endif

namespace FourProfile {
namespace {

constexpr const char* kProfileId =
  "P4-c384-h12-s225-qkn-positive-clip-int8-b28-s2";
constexpr const char* kGenericProfileId =
  "GINT8-c384-h12-qkn-positive-clip-dynamic-bs-mma";
constexpr int kBatch = C384Int8Experiment::kBatch;
constexpr int kSequence = C384Int8Experiment::kSequence;
constexpr int kChannels = C384Int8Experiment::kChannels;
constexpr int kHeads = C384Int8Experiment::kHeads;
constexpr int kHeadDim = C384Int8Experiment::kHeadDim;
constexpr int kFfnChannels = C384Int8Experiment::kFfnChannels;
constexpr int kPackedQkvChannels = C384Int8Experiment::kQkvChannels;
constexpr int kRopePairsTotal = kHeads * (kHeadDim / 2);
constexpr uint32_t kRmsEpsilon1e6Bits = 0x358637BDu;
constexpr uint64_t kPlanGeneration = 1;
constexpr int kMaxProviderRows = 1 << 20;

uint32_t floatBits(float value) {
  uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value),"32-bit float required");
  std::memcpy(&bits,&value,sizeof(bits));
  return bits;
}

bool positiveFinite(float value) {
  return value > 0.0f && std::isfinite(value);
}

bool aligned16(const void* pointer) {
  return pointer != nullptr &&
    (reinterpret_cast<uintptr_t>(pointer) & uintptr_t(15)) == 0;
}

bool baseRuntimeMatches(const RuntimeKeyV1& runtime) {
  const int64_t sequence =
    static_cast<int64_t>(runtime.boardX) * runtime.boardY;
  const bool validShape = runtime.boardX > 0 && runtime.boardY > 0 &&
    sequence > 0 && sequence <= kMaxProviderRows &&
    runtime.physicalBatchSize > 0 &&
    runtime.physicalBatchSize <= kMaxProviderRows / sequence;
  return runtime.deviceComputeCapability == 120 &&
    validShape &&
    runtime.sameGpuConcurrency > 0 && runtime.exactBoard &&
    runtime.maskMode == MaskModeV1::None && runtime.maskNull &&
    runtime.inputStorage == StorageTypeV1::Fp16 &&
    runtime.outputStorage == StorageTypeV1::Fp16 &&
    runtime.requestedExecution == RequestedExecutionV1::Int8 &&
    runtime.layout == TensorLayoutV1::Nhwc;
}

bool exactRuntimeMatches(const RuntimeKeyV1& runtime) {
  return baseRuntimeMatches(runtime) &&
    runtime.boardX == 15 && runtime.boardY == 15 &&
    runtime.physicalBatchSize == kBatch && runtime.sameGpuConcurrency == 2;
}

bool genericRuntimeMatches(const RuntimeKeyV1& runtime) {
  return baseRuntimeMatches(runtime) && !exactRuntimeMatches(runtime);
}

bool attentionMatches(const AttentionSpecV1& attention) {
  return attention.channels == kChannels &&
    attention.numHeads == kHeads && attention.numKVHeads == kHeads &&
    attention.qHeadDim == kHeadDim && attention.vHeadDim == kHeadDim &&
    attention.useRope && attention.learnableRope && attention.useQKNorm &&
    attention.hasInputQuantRange && attention.hasOutputQuantRange;
}

bool ffnMatches(const FfnSpecV1& ffn) {
  return ffn.channels == kChannels &&
    ffn.hiddenChannels == kFfnChannels && ffn.useSwiGLU &&
    ffn.clipClass == ClipClassV1::PositiveFinite &&
    ffn.hasInputQuantRange && ffn.hasProductQuantRange;
}

class DeviceBuffer {
public:
  DeviceBuffer() noexcept : pointer(nullptr),bytes(0) {}
  ~DeviceBuffer() {
    if(pointer != nullptr)
      (void)cudaFree(pointer);
  }
  DeviceBuffer(DeviceBuffer&& other) noexcept
    : pointer(other.pointer),bytes(other.bytes) {
    other.pointer = nullptr;
    other.bytes = 0;
  }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if(this != &other) {
      if(pointer != nullptr)
        (void)cudaFree(pointer);
      pointer = other.pointer;
      bytes = other.bytes;
      other.pointer = nullptr;
      other.bytes = 0;
    }
    return *this;
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void* get() const noexcept { return pointer; }
  size_t size() const noexcept { return bytes; }
  explicit operator bool() const noexcept { return pointer != nullptr; }

  static DeviceBuffer allocate(const std::string& label, size_t byteCount) {
    if(byteCount == 0)
      throw ErrorV1(label + ": zero-byte CUDA allocation");
    DeviceBuffer result;
    const cudaError_t status = cudaMalloc(&result.pointer,byteCount);
    if(status != cudaSuccess)
      throw ErrorV1(label + ": cudaMalloc failed: " + cudaGetErrorString(status));
    result.bytes = byteCount;
    return result;
  }

private:
  void* pointer;
  size_t bytes;
};

DeviceBuffer uploadBytes(
  const std::string& label,
  const void* source,
  size_t bytes
) {
  if(source == nullptr || bytes == 0)
    throw ErrorV1(label + ": empty upload");
  DeviceBuffer result = DeviceBuffer::allocate(label,bytes);
  const cudaError_t status =
    cudaMemcpy(result.get(),source,bytes,cudaMemcpyHostToDevice);
  if(status != cudaSuccess)
    throw ErrorV1(label + ": synchronous H2D copy failed: " +
      cudaGetErrorString(status));
  return result;
}

DeviceBuffer uploadFp16(
  const std::string& label,
  const std::vector<float>& source
) {
  if(source.empty())
    throw ErrorV1(label + ": empty FP16 upload");
  std::vector<half_float::half> converted(source.size());
  for(size_t i = 0; i < source.size(); i++)
    converted[i] = half_float::half_cast<half_float::half>(source[i]);
  return uploadBytes(
    label,converted.data(),converted.size() * sizeof(converted[0]));
}

DeviceBuffer uploadInt8(
  const std::string& label,
  const C384Int8Experiment::PackedWeights& packed
) {
  if(packed.values.empty() || !positiveFinite(packed.scale))
    throw ErrorV1(label + ": invalid packed INT8 weights");
  return uploadBytes(label,packed.values.data(),packed.values.size());
}

std::vector<float> learnedRopeTable(
  const TransformerAttentionDesc& desc,
  int boardX,
  int boardY,
  int sequenceSize
) {
  std::vector<float> cosTable;
  std::vector<float> sinTable;
  desc.computeRopeCosSin(boardX,boardY,sequenceSize,cosTable,sinTable);
  if(cosTable.size() != static_cast<size_t>(sequenceSize) * kRopePairsTotal ||
     sinTable.size() != cosTable.size())
    throw ErrorV1(desc.name + ": invalid learned RoPE table shape");
  std::vector<float> interleaved(cosTable.size() * 2);
  for(int xy = 0; xy < sequenceSize; xy++) {
    for(int pair = 0; pair < kRopePairsTotal; pair++) {
      const size_t source = static_cast<size_t>(pair) * sequenceSize + xy;
      const size_t destination =
        (static_cast<size_t>(xy) * kRopePairsTotal + pair) * 2;
      interleaved[destination] = cosTable[source];
      interleaved[destination + 1] = sinTable[source];
    }
  }
  return interleaved;
}

struct ProjectionDeleter {
  void operator()(void* pointer) const noexcept {
    C384Int8Experiment::destroyProjection(pointer);
  }
};
struct DualFfnDeleter {
  void operator()(void* pointer) const noexcept {
    C384Int8Experiment::destroyDualFfn(pointer);
  }
};
struct DownDeleter {
  void operator()(void* pointer) const noexcept {
    C384Int8Experiment::destroyDown(pointer);
  }
};
struct AttentionOutDeleter {
  void operator()(void* pointer) const noexcept {
    C384Int8Experiment::destroyAttentionOut(pointer);
  }
};

using ProjectionHandle = std::unique_ptr<void,ProjectionDeleter>;
using DualFfnHandle = std::unique_ptr<void,DualFfnDeleter>;
using DownHandle = std::unique_ptr<void,DownDeleter>;
using AttentionOutHandle = std::unique_ptr<void,AttentionOutDeleter>;

class P4PreparedAttention final : public PreparedAttentionV1 {
public:
  size_t layer = 0;
  const void* officialAnchor = nullptr;
  float preEpsilon = 0.0f;
  float qEpsilon = 0.0f;
  float kEpsilon = 0.0f;
  float inputQuantMaxAbs = 0.0f;
  float outputQuantMaxAbs = 0.0f;
  DeviceBuffer preGamma;
  DeviceBuffer qkvWeights;
  DeviceBuffer qGamma;
  DeviceBuffer kGamma;
  DeviceBuffer ropeTable;
  DeviceBuffer outWeights;
  ProjectionHandle projection;
  AttentionOutHandle attentionOut;
};

class P4PreparedFfn final : public PreparedFfnV1 {
public:
  size_t layer = 0;
  const void* officialAnchor = nullptr;
  float preEpsilon = 0.0f;
  float inputQuantMaxAbs = 0.0f;
  DeviceBuffer preGamma;
  DeviceBuffer upWeights;
  DeviceBuffer gateWeights;
  DeviceBuffer downWeights;
  DualFfnHandle dual;
  DownHandle down;
};

class P4Provider final : public ProviderV1 {
public:
  explicit P4Provider(const ProfileKeyV1& key_)
    : key(key_),physicalBatch(key_.runtime.physicalBatchSize),
      sequenceSize(key_.runtime.boardX * key_.runtime.boardY),tokenRows(0),
      exactFa4(exactRuntimeMatches(key_.runtime)),
      deviceOrdinal(-1),deviceMajor(0),deviceMinor(0),
      committed(false),runCounter(0),armedToken(0) {
    if(key.modelVersion != 105 || !baseRuntimeMatches(key.runtime) ||
       !attentionMatches(key.attention) || !ffnMatches(key.ffn))
      throw ErrorV1("C384 INT8 factory created for an unsupported profile key");
    tokenRows = physicalBatch * sequenceSize;
    cudaError_t status = cudaGetDevice(&deviceOrdinal);
    if(status != cudaSuccess || deviceOrdinal < 0)
      throw ErrorV1("P4 could not resolve the current CUDA device");
    status = cudaDeviceGetAttribute(
      &deviceMajor,cudaDevAttrComputeCapabilityMajor,deviceOrdinal);
    if(status == cudaSuccess)
      status = cudaDeviceGetAttribute(
        &deviceMinor,cudaDevAttrComputeCapabilityMinor,deviceOrdinal);
    if(status != cudaSuccess || deviceMajor != 12 || deviceMinor != 0)
      throw ErrorV1("P4 requires an exact SM120 CUDA device");
    if(!exactFa4 &&
       (!customCudaFlashAttentionMmaSupportsShape(
          kHeads,kHeads,kHeadDim,kHeadDim) ||
        !customCudaFlashAttentionMmaSupported()))
      throw ErrorV1("generic C384 INT8 route requires official MMA attention");
  }

  const char* profileId() const noexcept override {
    return exactFa4 ? kProfileId : kGenericProfileId;
  }

  bool acceptsActualBatchSize(
    int actualBatchSize,
    int physicalBatchSize
  ) const noexcept override {
    return physicalBatchSize == physicalBatch && actualBatchSize > 0 &&
      (exactFa4 ? actualBatchSize == physicalBatch :
        actualBatchSize <= physicalBatch);
  }

  PrepareAttentionResultV1 prepareAttention(
    size_t layer,
    const AttentionSpecV1& spec,
    const AttentionLayerScalarsV1& scalars,
    const OfficialAttentionResourcesV1& official
  ) override {
    PrepareAttentionResultV1 result;
    if(committed || !attentionMatches(spec) || !official.complete()) {
      result.detail = "P4 attention prepare rejected state or resources";
      return result;
    }
    const auto* desc =
      static_cast<const TransformerAttentionDesc*>(official.descriptor);
    if(desc == nullptr || !validateAttentionDescriptor(*desc,scalars,result.detail))
      return result;

    auto prepared = std::make_unique<P4PreparedAttention>();
    prepared->layer = layer;
    prepared->officialAnchor = official.executable;
    prepared->preEpsilon = desc->preLN.epsilon;
    prepared->qEpsilon = desc->qNorm.epsilon;
    prepared->kEpsilon = desc->kNorm.epsilon;
    prepared->inputQuantMaxAbs = desc->attentionInputQuantMaxAbs;
    prepared->outputQuantMaxAbs = desc->attentionOutputQuantMaxAbs;
    prepared->preGamma = uploadFp16(desc->name + ":p4-preln",desc->preLN.weight);
    prepared->qGamma = uploadFp16(desc->name + ":p4-qgamma",desc->qNorm.weight);
    prepared->kGamma = uploadFp16(desc->name + ":p4-kgamma",desc->kNorm.weight);
    prepared->ropeTable =
      uploadFp16(
        desc->name + ":p4-learned-rope",
        learnedRopeTable(
          *desc,key.runtime.boardX,key.runtime.boardY,sequenceSize));

    const C384Int8Experiment::PackedWeights packedQkv =
      C384Int8Experiment::packProjection(
        desc->qProj.weights,desc->kProj.weights,desc->vProj.weights,
        C384Int8Experiment::EngineMode::Aggressive);
    prepared->qkvWeights = uploadInt8(desc->name + ":p4-qkv",packedQkv);
    C384Int8Experiment::ProjectionConfig projectionConfig;
    projectionConfig.mode = C384Int8Experiment::ProjectionMode::AggressiveQkv;
    projectionConfig.tactic =
      C384Int8Experiment::ProjectionTactic::M128N128K64S3Sw2;
    projectionConfig.maxTokenRows = tokenRows;
    projectionConfig.sequenceSize = sequenceSize;
    projectionConfig.packedWeights =
      static_cast<const int8_t*>(prepared->qkvWeights.get());
    projectionConfig.weightScale = packedQkv.scale;
    projectionConfig.inputQuantMaxAbs = prepared->inputQuantMaxAbs;
    prepared->projection.reset(
      C384Int8Experiment::createProjection(projectionConfig));
    if(prepared->projection == nullptr ||
       !C384Int8Experiment::projectionInputQuantizationMatches(
         prepared->projection.get(),prepared->inputQuantMaxAbs) ||
       !C384Int8Experiment::projectionQknormRopeSupports(
         prepared->projection.get(),tokenRows,kChannels,kPackedQkvChannels,
         prepared->qEpsilon,prepared->kEpsilon)) {
      result.detail = "P4 fused INT8 QKV/QKN/RoPE preparation failed";
      return result;
    }

    const C384Int8Experiment::PackedWeights packedOut =
      C384Int8Experiment::packAttentionOut(desc->outProj.weights);
    prepared->outWeights = uploadInt8(desc->name + ":p4-out",packedOut);
    C384Int8Experiment::AttentionOutConfig outConfig;
    outConfig.tactic =
      C384Int8Experiment::AttentionOutTactic::M128N128K64S3Sw2;
    outConfig.maxTokenRows = tokenRows;
    outConfig.packedWeights =
      static_cast<const int8_t*>(prepared->outWeights.get());
    outConfig.weightScale = packedOut.scale;
    outConfig.inputQuantMaxAbs = prepared->outputQuantMaxAbs;
    prepared->attentionOut.reset(
      C384Int8Experiment::createAttentionOut(outConfig));
    if(prepared->attentionOut == nullptr ||
       !C384Int8Experiment::attentionOutInputQuantizationMatches(
         prepared->attentionOut.get(),prepared->outputQuantMaxAbs) ||
       !C384Int8Experiment::attentionOutSupports(
         prepared->attentionOut.get(),tokenRows)) {
      result.detail = "P4 INT8 attention-out preparation failed";
      return result;
    }

    result.prepared = std::move(prepared);
    result.detail = "P4 attention resources staged synchronously";
    return result;
  }

  PrepareFfnResultV1 prepareFfn(
    size_t layer,
    const FfnSpecV1& spec,
    const FfnLayerScalarsV1& scalars,
    const OfficialFfnResourcesV1& official
  ) override {
    PrepareFfnResultV1 result;
    if(committed || !ffnMatches(spec) || !official.complete()) {
      result.detail = "P4 FFN prepare rejected state or resources";
      return result;
    }
    const auto* desc = static_cast<const TransformerFFNDesc*>(official.descriptor);
    if(desc == nullptr || !validateFfnDescriptor(*desc,scalars,result.detail))
      return result;

    auto prepared = std::make_unique<P4PreparedFfn>();
    prepared->layer = layer;
    prepared->officialAnchor = official.executable;
    prepared->preEpsilon = desc->preLN.epsilon;
    prepared->inputQuantMaxAbs = desc->ffnInputQuantMaxAbs;
    prepared->preGamma = uploadFp16(desc->name + ":p4-preln",desc->preLN.weight);

    const C384Int8Experiment::PackedWeights packedUp =
      C384Int8Experiment::packMatrix(
        desc->linear1.weights,kChannels,kFfnChannels);
    const C384Int8Experiment::PackedWeights packedGate =
      C384Int8Experiment::packMatrix(
        desc->linearGate.weights,kChannels,kFfnChannels);
    prepared->upWeights = uploadInt8(desc->name + ":p4-up",packedUp);
    prepared->gateWeights = uploadInt8(desc->name + ":p4-gate",packedGate);
    C384Int8Experiment::DualFfnConfig dualConfig;
    dualConfig.tactic =
      C384Int8Experiment::DualFfnTactic::M128N128K64S3Sw4Interleaved;
    dualConfig.outputMode = C384Int8Experiment::DualFfnOutputMode::Int8Product;
    dualConfig.divide127Tactic =
      C384Int8Experiment::DualFfnDivide127Tactic::Auto;
    dualConfig.productPathTactic =
      C384Int8Experiment::DualFfnProductPathTactic::Auto;
    dualConfig.maxTokenRows = tokenRows;
    dualConfig.packedUpWeights =
      static_cast<const int8_t*>(prepared->upWeights.get());
    dualConfig.packedGateWeights =
      static_cast<const int8_t*>(prepared->gateWeights.get());
    dualConfig.upWeightScale = packedUp.scale;
    dualConfig.gateWeightScale = packedGate.scale;
    dualConfig.inputQuantMaxAbs = prepared->inputQuantMaxAbs;
    dualConfig.swigluClip = desc->swigluClip;
    dualConfig.productQuantMaxAbs = desc->productQuantMaxAbs;
    prepared->dual.reset(C384Int8Experiment::createDualFfn(dualConfig));
    if(prepared->dual == nullptr ||
       !C384Int8Experiment::dualFfnInputQuantizationMatches(
         prepared->dual.get(),prepared->inputQuantMaxAbs) ||
       !C384Int8Experiment::dualFfnSupports(
         prepared->dual.get(),
         C384Int8Experiment::DualFfnOutputMode::Int8Product,tokenRows)) {
      result.detail = "P4 adjustable-clip INT8 dual-FFN preparation failed";
      return result;
    }

    const C384Int8Experiment::PackedWeights packedDown =
      C384Int8Experiment::packMatrix(
        desc->linear2.weights,kFfnChannels,kChannels);
    prepared->downWeights = uploadInt8(desc->name + ":p4-down",packedDown);
    C384Int8Experiment::DownConfig downConfig;
    downConfig.tactic = C384Int8Experiment::DownTactic::M128N128K64S3Sw2;
    downConfig.maxTokenRows = tokenRows;
    downConfig.packedWeights =
      static_cast<const int8_t*>(prepared->downWeights.get());
    downConfig.weightScale = packedDown.scale;
    downConfig.productQuantMaxAbs = desc->productQuantMaxAbs;
    prepared->down.reset(C384Int8Experiment::createDown(downConfig));
    if(prepared->down == nullptr ||
       !C384Int8Experiment::downSupports(prepared->down.get(),tokenRows) ||
       !C384Int8Experiment::dualFfnDownProductQuantizationMatches(
         prepared->dual.get(),prepared->down.get())) {
      result.detail = "P4 INT8 FFN-down preparation failed";
      return result;
    }

    result.prepared = std::move(prepared);
    result.detail = "P4 FFN resources staged synchronously";
    return result;
  }

  bool commit(PreparedSpanV1&& prepared, std::string& detail) override {
    if(committed || prepared.attention.empty() ||
       prepared.attention.size() != prepared.ffn.size()) {
      detail = "P4 commit requires one nonempty dynamic N/N span";
      return false;
    }

    std::vector<std::unique_ptr<P4PreparedAttention>> stagedAttention;
    std::vector<std::unique_ptr<P4PreparedFfn>> stagedFfn;
    stagedAttention.reserve(prepared.attention.size());
    stagedFfn.reserve(prepared.ffn.size());
    for(size_t layer = 0; layer < prepared.attention.size(); layer++) {
      auto* a = dynamic_cast<P4PreparedAttention*>(prepared.attention[layer].get());
      auto* f = dynamic_cast<P4PreparedFfn*>(prepared.ffn[layer].get());
      if(a == nullptr || f == nullptr || a->layer != layer || f->layer != layer ||
         a->officialAnchor == nullptr || f->officialAnchor == nullptr) {
        detail = "P4 commit received malformed or reordered staged resources";
        return false;
      }
      stagedAttention.emplace_back(
        static_cast<P4PreparedAttention*>(prepared.attention[layer].release()));
      stagedFfn.emplace_back(
        static_cast<P4PreparedFfn*>(prepared.ffn[layer].release()));
    }

    C384H12Fa4Sm120::PreparedProof stagedFa4;
    if(exactFa4 && C384H12Fa4Sm120::prepareProofForExactBatch(
         kBatch,deviceOrdinal,C384H12Fa4Sm120::InputLayout::PackedTokenQkv,
         stagedFa4) != cudaSuccess) {
      detail = "P4 B28 packed FA4 preparation failed";
      return false;
    }
    DeviceBuffer stagedQkv = DeviceBuffer::allocate(
      "P4 packed QKV scratch",
      static_cast<size_t>(tokenRows) * kPackedQkvChannels * sizeof(half));
    DeviceBuffer stagedAttentionOut = DeviceBuffer::allocate(
      "P4 attention output scratch",
      static_cast<size_t>(tokenRows) * kChannels * sizeof(half));
    DeviceBuffer stagedProduct = DeviceBuffer::allocate(
      "P4 INT8 FFN product scratch",
      static_cast<size_t>(tokenRows) * kFfnChannels * sizeof(int8_t));

    attention = std::move(stagedAttention);
    ffn = std::move(stagedFfn);
    fa4Proof = stagedFa4;
    qkvScratch = std::move(stagedQkv);
    attentionOutScratch = std::move(stagedAttentionOut);
    productScratch = std::move(stagedProduct);
    committed = true;
    detail = exactFa4 ?
      "P4 dynamic N/N INT8 plan committed without inference enqueue" :
      "generic dynamic-batch C384 INT8 plan committed with official MMA attention";
    return true;
  }

  uint64_t committedPlanGeneration() const noexcept override {
    return committed ? kPlanGeneration : 0;
  }

  ProviderOpResultV1 preflight(const RuntimeCallV1& call) override {
    armedToken = 0;
    if(!committed || call.key != key.runtime ||
       !acceptsActualBatchSize(call.actualBatchSize,call.key.physicalBatchSize) ||
       call.sequenceSize != sequenceSize ||
       call.transformerPairCount != attention.size() || call.mask != nullptr ||
       call.stream == nullptr || !aligned16(call.trunk) ||
       !aligned16(call.trunkScratch) || !aligned16(qkvScratch.get()) ||
       !aligned16(attentionOutScratch.get()) || !aligned16(productScratch.get()))
      return ProviderOpResultV1::failure(
        "P4 full-chain preflight rejected runtime identity or scratch");
    const int activeRows = call.actualBatchSize * sequenceSize;

    int currentDevice = -1;
    if(cudaGetDevice(&currentDevice) != cudaSuccess || currentDevice != deviceOrdinal)
      return ProviderOpResultV1::failure(
        "P4 current CUDA device changed after commit");
    if(exactFa4 && !C384H12Fa4Sm120::proofCompatible(
         fa4Proof,kBatch,deviceOrdinal,
         C384H12Fa4Sm120::InputLayout::PackedTokenQkv))
      return ProviderOpResultV1::failure("P4 packed FA4 proof drifted");

    for(const auto& a: attention) {
      if(!C384Int8Experiment::projectionQknormRopeSupports(
           a->projection.get(),activeRows,kChannels,kPackedQkvChannels,
           a->qEpsilon,a->kEpsilon) ||
         !C384Int8Experiment::projectionInputQuantizationMatches(
           a->projection.get(),a->inputQuantMaxAbs) ||
         !C384Int8Experiment::attentionOutInputQuantizationMatches(
           a->attentionOut.get(),a->outputQuantMaxAbs) ||
         !C384Int8Experiment::attentionOutSupports(a->attentionOut.get(),activeRows))
        return ProviderOpResultV1::failure(
          "P4 attention layer failed kernel preflight");
    }
    for(const auto& f: ffn) {
      if(!C384Int8Experiment::dualFfnSupports(
           f->dual.get(),C384Int8Experiment::DualFfnOutputMode::Int8Product,
           activeRows) ||
         !C384Int8Experiment::dualFfnInputQuantizationMatches(
           f->dual.get(),f->inputQuantMaxAbs) ||
         !C384Int8Experiment::downSupports(f->down.get(),activeRows) ||
         !C384Int8Experiment::dualFfnDownProductQuantizationMatches(
           f->dual.get(),f->down.get()))
        return ProviderOpResultV1::failure(
          "P4 FFN layer failed kernel preflight");
    }

    runCounter += 1;
    if(runCounter == 0)
      runCounter += 1;
    armedToken = runCounter;
    return ProviderOpResultV1::success(0,kPlanGeneration,armedToken);
  }

  ProviderOpResultV1 enqueue(
    const RuntimeCallV1& call,
    uint64_t runToken
  ) override {
    if(!committed || runToken == 0 || runToken != armedToken)
      return ProviderOpResultV1::failure(
        "P4 enqueue did not match the armed RunToken");
    armedToken = 0;
    size_t enqueued = 0;
    const auto stream = reinterpret_cast<cudaStream_t>(call.stream);
    const int activeBatch = call.actualBatchSize;
    const int activeRows = activeBatch * sequenceSize;
    half* const trunk = static_cast<half*>(call.trunk);
    int8_t* const activation = static_cast<int8_t*>(call.trunkScratch);
    half* const qkv = static_cast<half*>(qkvScratch.get());
    half* const attentionOutput =
      static_cast<half*>(attentionOutScratch.get());
    int8_t* const product = static_cast<int8_t*>(productScratch.get());
    half* const q = qkv;
    half* const k = qkv + kChannels;
    half* const v = qkv + 2 * kChannels;

    for(size_t layer = 0; layer < attention.size(); layer++) {
      const P4PreparedAttention& a = *attention[layer];
      enqueued++;
      cudaError_t status = C384Int8Experiment::launchRmsNormInt8(
        trunk,activation,static_cast<const half*>(a.preGamma.get()),
        activeRows,a.preEpsilon,a.inputQuantMaxAbs,stream);
      if(status != cudaSuccess)
        return cudaFailure("attention RMS-to-INT8",status,enqueued);

      enqueued++;
      status = C384Int8Experiment::launchProjectionQknormRope(
        a.projection.get(),activeRows,activation,qkv,kPackedQkvChannels,
        static_cast<const half*>(a.qGamma.get()),
        static_cast<const half*>(a.kGamma.get()),
        static_cast<const half2*>(a.ropeTable.get()),
        a.qEpsilon,a.kEpsilon,stream);
      if(status != cudaSuccess)
        return cudaFailure("fused INT8 QKV/QKN/RoPE",status,enqueued);

      enqueued++;
      if(exactFa4) {
        const C384H12Fa4Sm120::LaunchResult fa4 = C384H12Fa4Sm120::launch(
          q,k,v,attentionOutput,kBatch,kSequence,kHeads,kHeads,kHeadDim,kHeadDim,
          true,true,C384H12Fa4Sm120::InputLayout::PackedTokenQkv,nullptr,true,
          &fa4Proof,deviceOrdinal,deviceMajor,deviceMinor,stream);
        if(!fa4.attempted || fa4.status != cudaSuccess)
          return cudaFailure(
            "B28 packed FA4",
            fa4.attempted ? fa4.status : cudaErrorNotSupported,enqueued);
      }
      else {
        const bool launched = customCudaFlashAttentionMma(
          q,k,v,nullptr,attentionOutput,activeBatch,sequenceSize,
          kHeads,kHeads,kHeadDim,kHeadDim,kPackedQkvChannels,
          kPackedQkvChannels,stream);
        status = launched ? cudaPeekAtLastError() : cudaErrorNotSupported;
        if(status != cudaSuccess)
          return cudaFailure("official MMA attention",status,enqueued);
      }

      enqueued++;
      status = C384Int8Experiment::launchQuantizeAttentionOutput(
        a.attentionOut.get(),attentionOutput,activation,activeRows,stream);
      if(status != cudaSuccess)
        return cudaFailure("attention output quantization",status,enqueued);

      enqueued++;
      status = C384Int8Experiment::launchAttentionOutResidual(
        a.attentionOut.get(),activeRows,activation,trunk,trunk,stream);
      if(status != cudaSuccess)
        return cudaFailure("INT8 attention-out residual",status,enqueued);

      const P4PreparedFfn& f = *ffn[layer];
      enqueued++;
      status = C384Int8Experiment::launchRmsNormInt8(
        trunk,activation,static_cast<const half*>(f.preGamma.get()),
        activeRows,f.preEpsilon,f.inputQuantMaxAbs,stream);
      if(status != cudaSuccess)
        return cudaFailure("FFN RMS-to-INT8",status,enqueued);

      enqueued++;
      status = C384Int8Experiment::launchDualFfnInt8(
        f.dual.get(),activeRows,activation,product,stream);
      if(status != cudaSuccess)
        return cudaFailure("adjustable-clip INT8 dual-FFN",status,enqueued);

      enqueued++;
      status = C384Int8Experiment::launchDownResidual(
        f.down.get(),activeRows,product,trunk,trunk,stream);
      if(status != cudaSuccess)
        return cudaFailure("INT8 FFN-down residual",status,enqueued);
    }
    return ProviderOpResultV1::success(enqueued,kPlanGeneration,runToken);
  }

private:
  bool validateAttentionDescriptor(
    const TransformerAttentionDesc& desc,
    const AttentionLayerScalarsV1& scalars,
    std::string& detail
  ) const {
    const bool dimensions =
      desc.numHeads == kHeads && desc.numKVHeads == kHeads &&
      desc.qHeadDim == kHeadDim && desc.vHeadDim == kHeadDim &&
      desc.useRope && desc.learnableRope && desc.useQKNorm &&
      desc.preLN.numChannels == kChannels &&
      desc.qProj.inChannels == kChannels && desc.qProj.outChannels == kChannels &&
      desc.kProj.inChannels == kChannels && desc.kProj.outChannels == kChannels &&
      desc.vProj.inChannels == kChannels && desc.vProj.outChannels == kChannels &&
      desc.outProj.inChannels == kChannels &&
      desc.outProj.outChannels == kChannels &&
      desc.qNorm.numChannels == kHeadDim && desc.kNorm.numChannels == kHeadDim &&
      desc.ropeNumKVHeads == kHeads && desc.ropeNumPairs == kHeadDim / 2 &&
      desc.preLN.weight.size() == kChannels &&
      desc.qNorm.weight.size() == kHeadDim && desc.kNorm.weight.size() == kHeadDim &&
      desc.qProj.weights.size() == static_cast<size_t>(kChannels) * kChannels &&
      desc.kProj.weights.size() == desc.qProj.weights.size() &&
      desc.vProj.weights.size() == desc.qProj.weights.size() &&
      desc.outProj.weights.size() == desc.qProj.weights.size() &&
      desc.ropeFreqs.size() == static_cast<size_t>(kHeads) * kHeadDim;
    const bool semantics =
      floatBits(desc.qNorm.epsilon) == kRmsEpsilon1e6Bits &&
      floatBits(desc.kNorm.epsilon) == kRmsEpsilon1e6Bits &&
      positiveFinite(desc.preLN.epsilon) &&
      scalars.inputQuantMaxAbsBits ==
        floatBits(desc.attentionInputQuantMaxAbs) &&
      scalars.outputQuantMaxAbsBits ==
        floatBits(desc.attentionOutputQuantMaxAbs) &&
      positiveFinite(desc.attentionInputQuantMaxAbs) &&
      positiveFinite(desc.attentionOutputQuantMaxAbs);
    if(!dimensions || !semantics) {
      detail = "P4 attention descriptor is outside the INT8 kernel contract";
      return false;
    }
    return true;
  }

  bool validateFfnDescriptor(
    const TransformerFFNDesc& desc,
    const FfnLayerScalarsV1& scalars,
    std::string& detail
  ) const {
    const bool dimensions =
      desc.numChannels == kChannels && desc.ffnChannels == kFfnChannels &&
      desc.useSwiGLU && desc.preLN.numChannels == kChannels &&
      desc.linear1.inChannels == kChannels &&
      desc.linear1.outChannels == kFfnChannels &&
      desc.linearGate.inChannels == kChannels &&
      desc.linearGate.outChannels == kFfnChannels &&
      desc.linear2.inChannels == kFfnChannels &&
      desc.linear2.outChannels == kChannels &&
      desc.preLN.weight.size() == kChannels &&
      desc.linear1.weights.size() ==
        static_cast<size_t>(kChannels) * kFfnChannels &&
      desc.linearGate.weights.size() == desc.linear1.weights.size() &&
      desc.linear2.weights.size() ==
        static_cast<size_t>(kFfnChannels) * kChannels;
    const bool semantics =
      positiveFinite(desc.preLN.epsilon) && positiveFinite(desc.swigluClip) &&
      positiveFinite(desc.ffnInputQuantMaxAbs) &&
      positiveFinite(desc.productQuantMaxAbs) &&
      scalars.swigluClipBits == floatBits(desc.swigluClip) &&
      scalars.inputQuantMaxAbsBits == floatBits(desc.ffnInputQuantMaxAbs) &&
      scalars.productQuantMaxAbsBits == floatBits(desc.productQuantMaxAbs);
    if(!dimensions || !semantics) {
      detail = "P4 FFN descriptor is outside the adjustable-clip INT8 contract";
      return false;
    }
    return true;
  }

  ProviderOpResultV1 cudaFailure(
    const char* operation,
    cudaError_t status,
    size_t enqueued
  ) const {
    return ProviderOpResultV1::failure(
      std::string("P4 ") + operation + " failed: " +
      cudaGetErrorString(status),enqueued);
  }

  ProfileKeyV1 key;
  int physicalBatch;
  int sequenceSize;
  int tokenRows;
  bool exactFa4;
  int deviceOrdinal;
  int deviceMajor;
  int deviceMinor;
  bool committed;
  uint64_t runCounter;
  uint64_t armedToken;
  std::vector<std::unique_ptr<P4PreparedAttention>> attention;
  std::vector<std::unique_ptr<P4PreparedFfn>> ffn;
  C384H12Fa4Sm120::PreparedProof fa4Proof;
  DeviceBuffer qkvScratch;
  DeviceBuffer attentionOutScratch;
  DeviceBuffer productScratch;
};

class P4Factory final : public FactoryV1 {
public:
  const char* factoryId() const noexcept override { return kProfileId; }

  bool matches(const ProfileKeyV1& key) const override {
    return key.modelVersion == 105 && exactRuntimeMatches(key.runtime) &&
      attentionMatches(key.attention) && ffnMatches(key.ffn);
  }

  AvailabilityResultV1 availability(const ProfileKeyV1&) const override {
    AvailabilityResultV1 result;
#if KATAGO_P4_EXTERNAL_PACKAGES_AVAILABLE
    result.availability = AvailabilityV1::Available;
    result.detail = "P4 INT8 kernels and B28 packed FA4 package are linked";
#else
    result.availability = AvailabilityV1::Unavailable;
    result.detail = "P4 was compiled without the B28 packed FA4 package";
#endif
    return result;
  }

  std::unique_ptr<ProviderV1> create(const ProfileKeyV1& key) const override {
#if KATAGO_P4_EXTERNAL_PACKAGES_AVAILABLE
    return std::make_unique<P4Provider>(key);
#else
    (void)key;
    return nullptr;
#endif
  }
};

class GenericC384Int8Factory final : public FactoryV1 {
public:
  const char* factoryId() const noexcept override { return kGenericProfileId; }

  bool matches(const ProfileKeyV1& key) const override {
    return key.modelVersion == 105 && genericRuntimeMatches(key.runtime) &&
      attentionMatches(key.attention) && ffnMatches(key.ffn);
  }

  AvailabilityResultV1 availability(const ProfileKeyV1&) const override {
    AvailabilityResultV1 result;
#if KATAGO_P4_EXTERNAL_PACKAGES_AVAILABLE
    result.availability = AvailabilityV1::Available;
    result.detail =
      "generic C384 INT8 kernels are linked; runtime uses official MMA attention";
#else
    result.availability = AvailabilityV1::Unavailable;
    result.detail = "generic C384 INT8 kernels are not linked";
#endif
    return result;
  }

  std::unique_ptr<ProviderV1> create(const ProfileKeyV1& key) const override {
#if KATAGO_P4_EXTERNAL_PACKAGES_AVAILABLE
    return std::make_unique<P4Provider>(key);
#else
    (void)key;
    return nullptr;
#endif
  }
};

}  // namespace

std::unique_ptr<FactoryV1> makeP4FactoryV1() {
  return std::make_unique<P4Factory>();
}

std::unique_ptr<FactoryV1> makeGenericC384Int8FactoryV1() {
  return std::make_unique<GenericC384Int8Factory>();
}

}  // namespace FourProfile

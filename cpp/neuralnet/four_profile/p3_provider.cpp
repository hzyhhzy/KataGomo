#include "p3_provider.h"

#include "../cudafusedffn.h"
#include "../cudaincludes.h"
#include "../desc.h"
#include "../cuda_specialized/sm120/c384/fixed_batch/fa4.h"
#include "../cuda_specialized/sm120/c384/fixed_batch/ffn_down.h"
#include "../cuda_specialized/sm120/c384/fixed_batch/kernels.h"
#include "../cuda_specialized/sm120/c384/fixed_batch/qknorm_rope.h"
#include "../cuda_specialized/sm120/c384/fixed_batch/weights.h"
#include "../cuda_specialized/sm120/shared/rms_norm.h"
#include "../../external/half-2.2.0/include/half.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifndef KATAGO_P3_EXTERNAL_PACKAGES_AVAILABLE
#define KATAGO_P3_EXTERNAL_PACKAGES_AVAILABLE 0
#endif
#ifndef KATAGO_P3_QKV_TACTIC_ID
#define KATAGO_P3_QKV_TACTIC_ID ""
#endif
#ifndef KATAGO_P3_FFN_DOWN_TACTIC_ID
#define KATAGO_P3_FFN_DOWN_TACTIC_ID ""
#endif

namespace FourProfile {
namespace {

constexpr const char* kProfileId =
  "P3-c384-h12-s225-qkn-positive-clip-fp16-b28-s2";
constexpr int kBatch = 28;
constexpr int kSequence = 225;
constexpr int kRows = kBatch * kSequence;
constexpr int kChannels = 384;
constexpr int kHeads = 12;
constexpr int kHeadDim = 32;
constexpr int kFfnChannels = 1024;
constexpr int kPackedQkvChannels = 3 * kChannels;
constexpr uint32_t kRmsEpsilon1e6Bits = 0x358637BDu;
constexpr uint64_t kPlanGeneration = 1;

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

bool runtimeMatches(const RuntimeKeyV1& runtime) {
  return runtime.deviceComputeCapability == 120 &&
    runtime.boardX == 15 && runtime.boardY == 15 &&
    runtime.physicalBatchSize == kBatch &&
    runtime.exactBoard &&
    runtime.maskMode == MaskModeV1::None && runtime.maskNull &&
    runtime.inputStorage == StorageTypeV1::Fp16 &&
    runtime.outputStorage == StorageTypeV1::Fp16 &&
    runtime.requestedExecution == RequestedExecutionV1::Fp16 &&
    runtime.layout == TensorLayoutV1::Nhwc;
}

bool attentionMatches(const AttentionSpecV1& attention) {
  return attention.channels == kChannels &&
    attention.numHeads == kHeads && attention.numKVHeads == kHeads &&
    attention.qHeadDim == kHeadDim && attention.vHeadDim == kHeadDim &&
    attention.useRope && attention.learnableRope && attention.useQKNorm &&
    !attention.hasInputQuantRange && !attention.hasOutputQuantRange;
}

bool ffnMatches(const FfnSpecV1& ffn) {
  return ffn.channels == kChannels &&
    ffn.hiddenChannels == kFfnChannels && ffn.useSwiGLU &&
    ffn.clipClass == ClipClassV1::PositiveFinite &&
    !ffn.hasInputQuantRange && !ffn.hasProductQuantRange;
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

DeviceBuffer uploadFp16(
  const std::string& label,
  const std::vector<float>& source
) {
  if(source.empty())
    throw ErrorV1(label + ": empty FP16 upload");
  std::vector<half_float::half> converted(source.size());
  for(size_t i = 0; i < source.size(); i++)
    converted[i] = half_float::half_cast<half_float::half>(source[i]);
  DeviceBuffer result =
    DeviceBuffer::allocate(label,converted.size() * sizeof(converted[0]));
  const cudaError_t status = cudaMemcpy(
    result.get(),converted.data(),result.size(),cudaMemcpyHostToDevice);
  if(status != cudaSuccess)
    throw ErrorV1(label + ": synchronous H2D copy failed: " +
      cudaGetErrorString(status));
  return result;
}

std::vector<float> transposeFfnWeight(const MatMulLayerDesc& desc) {
  if(desc.inChannels != kChannels || desc.outChannels != kFfnChannels ||
     desc.weights.size() != static_cast<size_t>(kChannels) * kFfnChannels)
    throw ErrorV1(desc.name + ": invalid P3 FFN projection shape");
  std::vector<float> packed(
    static_cast<size_t>(kFfnChannels) * kChannels);
  for(int output = 0; output < kFfnChannels; output++) {
    for(int input = 0; input < kChannels; input++) {
      packed[static_cast<size_t>(output) * kChannels + input] =
        desc.weights[static_cast<size_t>(input) * kFfnChannels + output];
    }
  }
  return packed;
}

std::vector<float> learnedRopeTable(const TransformerAttentionDesc& desc) {
  std::vector<float> cosTable;
  std::vector<float> sinTable;
  desc.computeRopeCosSin(15,15,kSequence,cosTable,sinTable);
  if(cosTable.size() != static_cast<size_t>(kSequence) *
       C384ExactFixedAot::kRopePairsTotal ||
     sinTable.size() != cosTable.size())
    throw ErrorV1(desc.name + ": invalid learned RoPE table shape");
  std::vector<float> interleaved(cosTable.size() * 2);
  for(int xy = 0; xy < kSequence; xy++) {
    for(int pair = 0; pair < C384ExactFixedAot::kRopePairsTotal; pair++) {
      const size_t source =
        static_cast<size_t>(pair) * kSequence + xy;
      const size_t destination =
        (static_cast<size_t>(xy) * C384ExactFixedAot::kRopePairsTotal + pair) * 2;
      interleaved[destination] = cosTable[source];
      interleaved[destination + 1] = sinTable[source];
    }
  }
  return interleaved;
}

std::vector<float> identityRopeTable() {
  std::vector<float> result(
    static_cast<size_t>(kSequence) *
    C384ExactFixedAot::kRopePairsTotal * 2);
  for(size_t i = 0; i < result.size(); i += 2) {
    result[i] = 1.0f;
    result[i + 1] = 0.0f;
  }
  return result;
}

class P3PreparedAttention final : public PreparedAttentionV1 {
public:
  size_t layer = 0;
  const void* officialAnchor = nullptr;
  float preEpsilon = 0.0f;
  float qEpsilon = 0.0f;
  float kEpsilon = 0.0f;
  DeviceBuffer preGamma;
  DeviceBuffer qkvWeights;
  DeviceBuffer qGamma;
  DeviceBuffer kGamma;
  DeviceBuffer ropeTable;
  DeviceBuffer outWeights;
};

class P3PreparedFfn final : public PreparedFfnV1 {
public:
  size_t layer = 0;
  const void* officialAnchor = nullptr;
  float preEpsilon = 0.0f;
  float clip = 0.0f;
  DeviceBuffer preGamma;
  DeviceBuffer upWeights;
  DeviceBuffer gateWeights;
  DeviceBuffer downWeights;
};

class OwnedCublas {
public:
  OwnedCublas() : handle(nullptr) {
    if(cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
      throw ErrorV1("P3 cublasCreate failed");
  }
  ~OwnedCublas() {
    if(handle != nullptr)
      (void)cublasDestroy(handle);
  }
  OwnedCublas(const OwnedCublas&) = delete;
  OwnedCublas& operator=(const OwnedCublas&) = delete;
  cublasHandle_t get() const noexcept { return handle; }
private:
  cublasHandle_t handle;
};

class P3Provider final : public ProviderV1 {
public:
  explicit P3Provider(const ProfileKeyV1& key_)
    : key(key_),deviceOrdinal(-1),deviceMajor(0),deviceMinor(0),
      committed(false),runCounter(0),armedToken(0) {
    if(!runtimeMatches(key.runtime) || key.modelVersion != 102 ||
       !attentionMatches(key.attention) || !ffnMatches(key.ffn))
      throw ErrorV1("P3 factory created for a non-P3 profile key");
    cudaError_t status = cudaGetDevice(&deviceOrdinal);
    if(status != cudaSuccess || deviceOrdinal < 0)
      throw ErrorV1("P3 could not resolve the current CUDA device");
    status = cudaDeviceGetAttribute(
      &deviceMajor,cudaDevAttrComputeCapabilityMajor,deviceOrdinal);
    if(status == cudaSuccess)
      status = cudaDeviceGetAttribute(
        &deviceMinor,cudaDevAttrComputeCapabilityMinor,deviceOrdinal);
    if(status != cudaSuccess || deviceMajor != 12 || deviceMinor != 0)
      throw ErrorV1("P3 requires an exact SM120 CUDA device");
  }

  const char* profileId() const noexcept override { return kProfileId; }

  PrepareAttentionResultV1 prepareAttention(
    size_t layer,
    const AttentionSpecV1& spec,
    const AttentionLayerScalarsV1& scalars,
    const OfficialAttentionResourcesV1& official
  ) override {
    PrepareAttentionResultV1 result;
    if(committed) {
      result.detail = "P3 attention prepare called after commit";
      return result;
    }
    if(!attentionMatches(spec) || !official.complete()) {
      result.detail = "P3 attention prepare received incomplete profile or official resources";
      return result;
    }
    const auto* desc =
      static_cast<const TransformerAttentionDesc*>(official.descriptor);
    if(desc == nullptr || !validateAttentionDescriptor(*desc,scalars,result.detail))
      return result;

    auto prepared = std::make_unique<P3PreparedAttention>();
    prepared->layer = layer;
    prepared->officialAnchor = official.executable;
    prepared->preEpsilon = desc->preLN.epsilon;
    prepared->qEpsilon = desc->qNorm.epsilon;
    prepared->kEpsilon = desc->kNorm.epsilon;
    prepared->preGamma = uploadFp16(desc->name + ":p3-preln",desc->preLN.weight);
    prepared->qkvWeights = uploadFp16(
      desc->name + ":p3-qkv",
      C384ExactFixedAot::packQkvWeights(
        desc->qProj.weights,desc->kProj.weights,desc->vProj.weights));
    prepared->qGamma = uploadFp16(desc->name + ":p3-qgamma",desc->qNorm.weight);
    prepared->kGamma = uploadFp16(desc->name + ":p3-kgamma",desc->kNorm.weight);
    prepared->ropeTable =
      uploadFp16(desc->name + ":p3-learned-rope",learnedRopeTable(*desc));
    prepared->outWeights =
      uploadFp16(desc->name + ":p3-out",desc->outProj.weights);
    result.prepared = std::move(prepared);
    result.detail = "P3 attention resources staged synchronously";
    return result;
  }

  PrepareFfnResultV1 prepareFfn(
    size_t layer,
    const FfnSpecV1& spec,
    const FfnLayerScalarsV1& scalars,
    const OfficialFfnResourcesV1& official
  ) override {
    PrepareFfnResultV1 result;
    if(committed) {
      result.detail = "P3 FFN prepare called after commit";
      return result;
    }
    if(!ffnMatches(spec) || !official.complete()) {
      result.detail = "P3 FFN prepare received incomplete profile or official resources";
      return result;
    }
    const auto* desc =
      static_cast<const TransformerFFNDesc*>(official.descriptor);
    if(desc == nullptr || !validateFfnDescriptor(*desc,scalars,result.detail))
      return result;

    auto prepared = std::make_unique<P3PreparedFfn>();
    prepared->layer = layer;
    prepared->officialAnchor = official.executable;
    prepared->preEpsilon = desc->preLN.epsilon;
    prepared->clip = desc->swigluClip;
    prepared->preGamma = uploadFp16(desc->name + ":p3-preln",desc->preLN.weight);
    prepared->upWeights =
      uploadFp16(desc->name + ":p3-up",transposeFfnWeight(desc->linear1));
    prepared->gateWeights =
      uploadFp16(desc->name + ":p3-gate",transposeFfnWeight(desc->linearGate));
    prepared->downWeights =
      uploadFp16(desc->name + ":p3-down",desc->linear2.weights);
    if(!CudaFusedFFN::supportsProblem(
         kRows,kFfnChannels,kChannels,prepared->clip) ||
       !CudaFusedFFN::supportsPreparedWeights(
         static_cast<const half*>(prepared->upWeights.get()),
         static_cast<const half*>(prepared->gateWeights.get()),
         kRows,kFfnChannels,kChannels,prepared->clip)) {
      result.detail =
        "P3 runtime-clipped CUTLASS FFN rejected the prepared problem";
      return result;
    }
    result.prepared = std::move(prepared);
    result.detail = "P3 FFN resources staged synchronously";
    return result;
  }

  bool commit(PreparedSpanV1&& prepared, std::string& detail) override {
    if(committed || prepared.attention.empty() ||
       prepared.attention.size() != prepared.ffn.size()) {
      detail = "P3 commit requires one nonempty dynamic N/N span";
      return false;
    }

    std::vector<std::unique_ptr<P3PreparedAttention>> stagedAttention;
    std::vector<std::unique_ptr<P3PreparedFfn>> stagedFfn;
    stagedAttention.reserve(prepared.attention.size());
    stagedFfn.reserve(prepared.ffn.size());
    for(size_t layer = 0; layer < prepared.attention.size(); layer++) {
      auto* a = dynamic_cast<P3PreparedAttention*>(prepared.attention[layer].get());
      auto* f = dynamic_cast<P3PreparedFfn*>(prepared.ffn[layer].get());
      if(a == nullptr || f == nullptr ||
         a->layer != layer || f->layer != layer ||
         a->officialAnchor == nullptr || f->officialAnchor == nullptr) {
        detail = "P3 commit received malformed or reordered staged resources";
        return false;
      }
      stagedAttention.emplace_back(
        static_cast<P3PreparedAttention*>(prepared.attention[layer].release()));
      stagedFfn.emplace_back(
        static_cast<P3PreparedFfn*>(prepared.ffn[layer].release()));
    }

    const int depth = static_cast<int>(stagedAttention.size());
    C384H12Fa4Sm120::PreparedProof stagedFa4;
    if(C384H12Fa4Sm120::prepareProofForExactBatch(
         kBatch,deviceOrdinal,C384H12Fa4Sm120::InputLayout::PackedTokenQkv,
         stagedFa4) != cudaSuccess) {
      detail = "P3 B28 packed FA4 eager preparation failed";
      return false;
    }

    C384ExactFixedAot::PreparedPackedFa4 stagedPortable;
    stagedPortable.abiVersion =
      C384ExactFixedAot::kPackedFa4ProofAbiVersion;
    stagedPortable.batchSize = stagedFa4.batch;
    stagedPortable.sequenceLength = stagedFa4.sequence;
    stagedPortable.numHeads = stagedFa4.heads;
    stagedPortable.numKvHeads = stagedFa4.heads;
    stagedPortable.qHeadDim = stagedFa4.headDim;
    stagedPortable.vHeadDim = stagedFa4.headDim;
    stagedPortable.deviceOrdinal = stagedFa4.deviceOrdinal;
    stagedPortable.acceptsPackedTokenQkv =
      stagedFa4.inputLayout ==
        C384H12Fa4Sm120::InputLayout::PackedTokenQkv;
    stagedPortable.id = stagedFa4.id;
    stagedPortable.implementationCookie = stagedFa4.implementationCookie;

    C384ExactFixedAot::PreparedCudaSelection stagedQkv =
      C384ExactFixedAot::prepareCudaSelection(
        makeExactShape(depth,true),KATAGO_P3_QKV_TACTIC_ID,nullptr,
        &stagedPortable);
    if(stagedQkv.qkvRope == nullptr || stagedQkv.packedFa4 == nullptr) {
      detail = std::string("P3 exact QKV preparation failed: ") +
        C384ExactFixedAot::rejectReasonName(stagedQkv.qkvRopeReason);
      return false;
    }

    C384ExactFfnDownAot::RuntimeShape stagedDownShape =
      makeDownShape(depth);
    C384ExactFfnDownAot::PreparedSelection stagedDown =
      C384ExactFfnDownAot::prepareSelection(
        stagedDownShape,KATAGO_P3_FFN_DOWN_TACTIC_ID);
    if(!stagedDown.selected()) {
      detail = std::string("P3 exact FFN-down preparation failed: ") +
        C384ExactFfnDownAot::rejectReasonName(stagedDown.reason);
      return false;
    }

    DeviceBuffer stagedIdentity =
      uploadFp16("P3 identity RoPE",identityRopeTable());
    DeviceBuffer stagedWide = DeviceBuffer::allocate(
      "P3 wide scratch",
      static_cast<size_t>(kRows) * kPackedQkvChannels * sizeof(half));
    DeviceBuffer stagedNarrow = DeviceBuffer::allocate(
      "P3 narrow scratch",
      static_cast<size_t>(kRows) * kChannels * sizeof(half));
    std::unique_ptr<OwnedCublas> stagedCublas =
      std::make_unique<OwnedCublas>();

    attention = std::move(stagedAttention);
    ffn = std::move(stagedFfn);
    fa4Proof = stagedFa4;
    portableFa4Proof = stagedPortable;
    qkvSelection = stagedQkv;
    qkvSelection.packedFa4 = &portableFa4Proof;
    downShape = stagedDownShape;
    downSelection = stagedDown;
    identityRope = std::move(stagedIdentity);
    wideScratch = std::move(stagedWide);
    narrowScratch = std::move(stagedNarrow);
    cublas = std::move(stagedCublas);
    committed = true;
    detail = "P3 dynamic N/N plan committed without inference enqueue";
    return true;
  }

  uint64_t committedPlanGeneration() const noexcept override {
    return committed ? kPlanGeneration : 0;
  }

  ProviderOpResultV1 preflight(const RuntimeCallV1& call) override {
    armedToken = 0;
    if(!committed || call.key != key.runtime ||
       call.actualBatchSize != kBatch || call.sequenceSize != kSequence ||
       call.transformerPairCount != attention.size() ||
       call.mask != nullptr || call.stream == nullptr ||
       !aligned16(call.trunk) || !aligned16(call.trunkScratch) ||
       !wideScratch || !narrowScratch || !identityRope ||
       cublas == nullptr || qkvSelection.qkvRope == nullptr ||
       qkvSelection.packedFa4 != &portableFa4Proof ||
       !downSelection.selected()) {
      return ProviderOpResultV1::failure(
        "P3 full-chain preflight rejected runtime identity or resources");
    }
    int currentDevice = -1;
    if(cudaGetDevice(&currentDevice) != cudaSuccess ||
       currentDevice != deviceOrdinal)
      return ProviderOpResultV1::failure(
        "P3 current CUDA device changed after commit");
    const auto stream = reinterpret_cast<cudaStream_t>(call.stream);
    if(cublasSetStream(cublas->get(),stream) != CUBLAS_STATUS_SUCCESS)
      return ProviderOpResultV1::failure(
        "P3 could not bind its private cuBLAS handle to the RunToken stream");

    if(!C384H12Fa4Sm120::proofCompatible(
         fa4Proof,kBatch,deviceOrdinal,
         C384H12Fa4Sm120::InputLayout::PackedTokenQkv))
      return ProviderOpResultV1::failure("P3 packed FA4 proof drifted");
    if(qkvSelection.qkvRope->key.tokenRows != kRows ||
       !qkvSelection.qkvRope->key.runtimeRopeTableDriven ||
       !aligned16(wideScratch.get()) || !aligned16(narrowScratch.get()))
      return ProviderOpResultV1::failure("P3 exact QKV ABI drifted");

    const half* const wide = static_cast<const half*>(wideScratch.get());
    for(const auto& a: attention) {
      if(!C384QKNormRopeSm120::supports(
           qknParams(a->qEpsilon,a->kEpsilon)) ||
         !aligned16(a->preGamma.get()) || !aligned16(a->qkvWeights.get()) ||
         !aligned16(a->qGamma.get()) || !aligned16(a->kGamma.get()) ||
         !aligned16(a->ropeTable.get()) || !aligned16(a->outWeights.get()))
        return ProviderOpResultV1::failure(
          "P3 attention layer failed full-chain preflight");
    }
    for(const auto& f: ffn) {
      if(!aligned16(f->preGamma.get()) ||
         !CudaFusedFFN::canImplement(
           static_cast<const half*>(call.trunkScratch),
           static_cast<const half*>(f->upWeights.get()),
           static_cast<const half*>(f->gateWeights.get()),
           static_cast<half*>(wideScratch.get()),
           kRows,kFfnChannels,kChannels,f->clip) ||
         !C384ExactFfnDownAot::supports(
           downSelection,downShape,wide,f->downWeights.get(),call.trunk))
        return ProviderOpResultV1::failure(
          "P3 FFN layer failed full-chain preflight");
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
        "P3 enqueue did not match the armed RunToken");
    armedToken = 0;
    size_t enqueued = 0;
    const auto stream = reinterpret_cast<cudaStream_t>(call.stream);
    if(cublasSetStream(cublas->get(),stream) != CUBLAS_STATUS_SUCCESS)
      return ProviderOpResultV1::failure(
        "P3 RunToken stream bind failed before work");

    half* const trunk = static_cast<half*>(call.trunk);
    half* const normalized = static_cast<half*>(call.trunkScratch);
    half* const wide = static_cast<half*>(wideScratch.get());
    half* const narrow = static_cast<half*>(narrowScratch.get());
    half* const q = wide;
    half* const k = wide + kChannels;
    half* const v = wide + 2 * kChannels;
    try {
      for(size_t layer = 0; layer < attention.size(); layer++) {
        const P3PreparedAttention& a = *attention[layer];
        enqueued++;
        cudaError_t status = Renju15Sm120::launchRmsNorm384(
          trunk,normalized,static_cast<const half*>(a.preGamma.get()),
          kRows,a.preEpsilon,Renju15Sm120::RmsNorm384Tactic::Warp4Vec4x3,
          stream);
        if(status != cudaSuccess)
          return cudaFailure("attention RMS",status,enqueued);

        enqueued++;
        status = qkvSelection.qkvRope->launch(
          normalized,static_cast<const half*>(a.qkvWeights.get()),
          static_cast<const half2*>(identityRope.get()),wide,kRows,
          deviceOrdinal,stream);
        if(status != cudaSuccess)
          return cudaFailure("exact QKV",status,enqueued);

        enqueued++;
        status = C384QKNormRopeSm120::launchInPlace(
          qknParams(a.qEpsilon,a.kEpsilon),wide,
          static_cast<const half*>(a.qGamma.get()),
          static_cast<const half*>(a.kGamma.get()),
          static_cast<const half2*>(a.ropeTable.get()),stream);
        if(status != cudaSuccess)
          return cudaFailure("QKN learned RoPE",status,enqueued);

        enqueued++;
        const C384H12Fa4Sm120::LaunchResult fa4 =
          C384H12Fa4Sm120::launch(
            q,k,v,narrow,kBatch,kSequence,kHeads,kHeads,kHeadDim,kHeadDim,
            true,true,C384H12Fa4Sm120::InputLayout::PackedTokenQkv,nullptr,
            true,&fa4Proof,deviceOrdinal,deviceMajor,deviceMinor,stream);
        if(!fa4.attempted || fa4.status != cudaSuccess)
          return cudaFailure("B28 packed FA4",
            fa4.attempted ? fa4.status : cudaErrorNotSupported,enqueued);

        const half alpha = __float2half(1.0f);
        enqueued++;
        const cublasStatus_t outStatus = cublasHgemm(
          cublas->get(),CUBLAS_OP_N,CUBLAS_OP_N,
          kChannels,kRows,kChannels,&alpha,
          static_cast<const half*>(a.outWeights.get()),kChannels,
          narrow,kChannels,&alpha,trunk,kChannels);
        if(outStatus != CUBLAS_STATUS_SUCCESS)
          return cublasFailure("attention out residual",outStatus,enqueued);

        const P3PreparedFfn& f = *ffn[layer];
        enqueued++;
        status = Renju15Sm120::launchRmsNorm384(
          trunk,normalized,static_cast<const half*>(f.preGamma.get()),
          kRows,f.preEpsilon,Renju15Sm120::RmsNorm384Tactic::Warp4Vec4x3,
          stream);
        if(status != cudaSuccess)
          return cudaFailure("FFN RMS",status,enqueued);

        enqueued++;
        CudaFusedFFN::runSwiGLU(
          normalized,static_cast<const half*>(f.upWeights.get()),
          static_cast<const half*>(f.gateWeights.get()),wide,
          kRows,kFfnChannels,kChannels,f.clip,stream);

        enqueued++;
        status = downSelection.tactic->launch(
          wide,static_cast<const half*>(f.downWeights.get()),trunk,kRows,
          deviceOrdinal,stream);
        if(status != cudaSuccess)
          return cudaFailure("exact FFN down residual",status,enqueued);
      }
    }
    catch(const std::exception& e) {
      return ProviderOpResultV1::failure(
        std::string("P3 execution failure after provider entry: ") + e.what(),
        enqueued);
    }
    return ProviderOpResultV1::success(enqueued,kPlanGeneration,runToken);
  }

private:
  bool validateAttentionDescriptor(
    const TransformerAttentionDesc& desc,
    const AttentionLayerScalarsV1& scalars,
    std::string& detail
  ) const {
    const bool dimensionsMatch =
      desc.numHeads == kHeads && desc.numKVHeads == kHeads &&
      desc.qHeadDim == kHeadDim && desc.vHeadDim == kHeadDim &&
      desc.useRope && desc.learnableRope && desc.useQKNorm &&
      desc.preLN.numChannels == kChannels &&
      desc.qProj.inChannels == kChannels &&
      desc.qProj.outChannels == kChannels &&
      desc.kProj.inChannels == kChannels &&
      desc.kProj.outChannels == kChannels &&
      desc.vProj.inChannels == kChannels &&
      desc.vProj.outChannels == kChannels &&
      desc.outProj.inChannels == kChannels &&
      desc.outProj.outChannels == kChannels &&
      desc.qNorm.numChannels == kHeadDim &&
      desc.kNorm.numChannels == kHeadDim &&
      desc.ropeNumKVHeads == kHeads &&
      desc.ropeNumPairs == kHeadDim / 2 &&
      desc.preLN.weight.size() == kChannels &&
      desc.qNorm.weight.size() == kHeadDim &&
      desc.kNorm.weight.size() == kHeadDim &&
      desc.qProj.weights.size() ==
        static_cast<size_t>(kChannels) * kChannels &&
      desc.kProj.weights.size() == desc.qProj.weights.size() &&
      desc.vProj.weights.size() == desc.qProj.weights.size() &&
      desc.outProj.weights.size() == desc.qProj.weights.size() &&
      desc.ropeFreqs.size() ==
        static_cast<size_t>(kHeads) * (kHeadDim / 2) * 2;
    const bool qknSemantics =
      floatBits(desc.qNorm.epsilon) == kRmsEpsilon1e6Bits &&
      floatBits(desc.kNorm.epsilon) == kRmsEpsilon1e6Bits &&
      positiveFinite(desc.preLN.epsilon);
    const bool noPtqRanges =
      scalars.inputQuantMaxAbsBits == 0 &&
      scalars.outputQuantMaxAbsBits == 0 &&
      desc.attentionInputQuantMaxAbs == 0.0f &&
      desc.attentionOutputQuantMaxAbs == 0.0f;
    if(!dimensionsMatch || !qknSemantics || !noPtqRanges) {
      detail = "P3 attention descriptor failed complete v102 semantic validation";
      return false;
    }
    return true;
  }

  bool validateFfnDescriptor(
    const TransformerFFNDesc& desc,
    const FfnLayerScalarsV1& scalars,
    std::string& detail
  ) const {
    const bool scalarsMatch =
      scalars.swigluClipBits == floatBits(desc.swigluClip) &&
      scalars.inputQuantMaxAbsBits == 0 &&
      scalars.productQuantMaxAbsBits == 0 &&
      desc.ffnInputQuantMaxAbs == 0.0f &&
      desc.productQuantMaxAbs == 0.0f &&
      positiveFinite(desc.swigluClip);
    const bool dimensionsMatch =
      desc.numChannels == kChannels &&
      desc.ffnChannels == kFfnChannels && desc.useSwiGLU &&
      desc.preLN.numChannels == kChannels &&
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
        static_cast<size_t>(kFfnChannels) * kChannels &&
      positiveFinite(desc.preLN.epsilon);
    if(!scalarsMatch || !dimensionsMatch) {
      detail = "P3 FFN descriptor failed complete v102 semantic validation";
      return false;
    }
    return true;
  }

  C384ExactFixedAot::RuntimeShape makeExactShape(
    int depth,
    bool attentionShape
  ) const {
    C384ExactFixedAot::RuntimeShape shape;
    shape.modelDepth = depth;
    shape.attentionBlockCount = depth;
    shape.ffnBlockCount = depth;
    shape.alternatingAttentionFfn = true;
    shape.batchSize = kBatch;
    shape.enqueuedRows = kRows;
    shape.boardX = 15;
    shape.boardY = 15;
    shape.sequenceLength = kSequence;
    shape.channels = kChannels;
    shape.numHeads = attentionShape ? kHeads : 0;
    shape.numKvHeads = attentionShape ? kHeads : 0;
    shape.qHeadDim = attentionShape ? kHeadDim : 0;
    shape.vHeadDim = attentionShape ? kHeadDim : 0;
    shape.ffnChannels = attentionShape ? 0 : kFfnChannels;
    shape.ropePairsTotal =
      attentionShape ? C384ExactFixedAot::kRopePairsTotal : 0;
    shape.deviceOrdinal = deviceOrdinal;
    shape.computeCapability = 120;
    shape.usingFp16 = true;
    shape.usingNhwc = true;
    shape.exactNoMask = true;
    shape.learnedRope = attentionShape;
    shape.swiglu = !attentionShape;
    shape.qkNorm = attentionShape;
    shape.swigluClipBits = 0;
    return shape;
  }

  C384ExactFfnDownAot::RuntimeShape makeDownShape(int depth) const {
    C384ExactFfnDownAot::RuntimeShape shape;
    shape.modelDepth = depth;
    shape.attentionBlockCount = depth;
    shape.ffnBlockCount = depth;
    shape.alternatingAttentionFfn = true;
    shape.batchSize = kBatch;
    shape.tokenRows = kRows;
    shape.boardX = 15;
    shape.boardY = 15;
    shape.sequenceLength = kSequence;
    shape.channels = kChannels;
    shape.ffnChannels = kFfnChannels;
    shape.deviceOrdinal = deviceOrdinal;
    shape.computeCapability = 120;
    shape.usingFp16 = true;
    shape.usingNhwc = true;
    shape.exactNoMask = true;
    shape.swiglu = true;
    return shape;
  }

  C384QKNormRopeSm120::LaunchParams qknParams(
    float qEpsilon,
    float kEpsilon
  ) const {
    C384QKNormRopeSm120::LaunchParams params;
    params.abiVersion = C384QKNormRopeSm120::kAbiVersion;
    params.batch = kBatch;
    params.sequence = kSequence;
    params.heads = kHeads;
    params.kvHeads = kHeads;
    params.headDim = kHeadDim;
    params.tokenRows = kRows;
    params.deviceOrdinal = deviceOrdinal;
    params.computeCapability = 120;
    params.usingFp16 = true;
    params.usingNhwc = true;
    params.learnedRope = true;
    params.qkNorm = true;
    params.inputSemantic =
      C384QKNormRopeSm120::InputSemantic::RawPackedQkv;
    params.qEpsilon = qEpsilon;
    params.kEpsilon = kEpsilon;
    return params;
  }

  ProviderOpResultV1 cudaFailure(
    const char* operation,
    cudaError_t status,
    size_t enqueued
  ) const {
    return ProviderOpResultV1::failure(
      std::string("P3 ") + operation + " failed: " +
      cudaGetErrorString(status),enqueued);
  }

  ProviderOpResultV1 cublasFailure(
    const char* operation,
    cublasStatus_t status,
    size_t enqueued
  ) const {
    return ProviderOpResultV1::failure(
      std::string("P3 ") + operation +
      " failed with cuBLAS status " + std::to_string(static_cast<int>(status)),
      enqueued);
  }

  ProfileKeyV1 key;
  int deviceOrdinal;
  int deviceMajor;
  int deviceMinor;
  bool committed;
  uint64_t runCounter;
  uint64_t armedToken;
  std::vector<std::unique_ptr<P3PreparedAttention>> attention;
  std::vector<std::unique_ptr<P3PreparedFfn>> ffn;
  C384H12Fa4Sm120::PreparedProof fa4Proof;
  C384ExactFixedAot::PreparedPackedFa4 portableFa4Proof;
  C384ExactFixedAot::PreparedCudaSelection qkvSelection;
  C384ExactFfnDownAot::RuntimeShape downShape;
  C384ExactFfnDownAot::PreparedSelection downSelection;
  DeviceBuffer identityRope;
  DeviceBuffer wideScratch;
  DeviceBuffer narrowScratch;
  std::unique_ptr<OwnedCublas> cublas;
};

class P3Factory final : public FactoryV1 {
public:
  const char* factoryId() const noexcept override { return kProfileId; }

  bool matches(const ProfileKeyV1& key) const override {
    return key.modelVersion == 102 && runtimeMatches(key.runtime) &&
      attentionMatches(key.attention) && ffnMatches(key.ffn);
  }

  AvailabilityResultV1 availability(const ProfileKeyV1&) const override {
    AvailabilityResultV1 result;
#if KATAGO_P3_EXTERNAL_PACKAGES_AVAILABLE
    result.availability = AvailabilityV1::Available;
    result.detail =
      "P3 provider and all three SHA-locked B28 packages are linked";
#else
    result.availability = AvailabilityV1::Unavailable;
    result.detail =
      "P3 provider was compiled without the complete SHA-locked AOT/FA4/down package set";
#endif
    return result;
  }

  std::unique_ptr<ProviderV1> create(
    const ProfileKeyV1& key
  ) const override {
#if KATAGO_P3_EXTERNAL_PACKAGES_AVAILABLE
    return std::make_unique<P3Provider>(key);
#else
    (void)key;
    return nullptr;
#endif
  }
};

}  // namespace

std::unique_ptr<FactoryV1> makeP3FactoryV1() {
  return std::make_unique<P3Factory>();
}

}  // namespace FourProfile

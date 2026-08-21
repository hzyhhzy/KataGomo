#include "p1_provider.h"

#if !defined(KATAGO_ENABLE_P1_SM120_PROVIDER) || !KATAGO_ENABLE_P1_SM120_PROVIDER
#error "p1_provider.cpp may only be compiled with the qualified P1 package"
#endif
#if !defined(KATAGO_ENABLE_RENJU15_RMS_SM120) || !KATAGO_ENABLE_RENJU15_RMS_SM120
#error "P1 provider requires the qualified SM120 RMSNorm kernel"
#endif
#if !defined(KATAGO_ENABLE_RENJU15_QKV_ROPE_GEMM_SM120) || !KATAGO_ENABLE_RENJU15_QKV_ROPE_GEMM_SM120
#error "P1 provider requires the qualified SM120 QKV+RoPE kernel"
#endif

#include "../cuda_specialized/sm120/c256/fixed_batch/fa4.h"
#include "../cuda_specialized/sm120/c256/fixed_batch/qkv_rope_gemm.h"
#include "../cuda_specialized/sm120/shared/residual_gemm.h"
#include "../cuda_specialized/sm120/shared/rms_norm.h"
#include "../cudafusedffn.h"
#include "../cudautils.h"
#include "../desc.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace FourProfile {
namespace {

constexpr const char* P1_PROFILE_ID = "P1-c256-h8-s225-fp16-b36-s2";
constexpr int P1_BOARD = 15;
constexpr int P1_SEQUENCE = 225;
constexpr int P1_BATCH = 36;
constexpr int P1_CHANNELS = 256;
constexpr int P1_HEADS = 8;
constexpr int P1_HEAD_DIM = 32;
constexpr int P1_ROPE_PAIRS = 16;
constexpr int P1_FFN_CHANNELS = 768;
constexpr int P1_TOKEN_ROWS = P1_BATCH * P1_SEQUENCE;

std::string cudaFailure(const char* operation, cudaError_t status, size_t layer) {
  std::ostringstream out;
  out << P1_PROFILE_ID << " layer " << layer << " " << operation
      << " failed: " << cudaGetErrorString(status);
  return out.str();
}

bool aligned16(const void* pointer) {
  return pointer != nullptr &&
    (reinterpret_cast<std::uintptr_t>(pointer) & std::uintptr_t(15)) == 0;
}

bool exactP1Runtime(const RuntimeKeyV1& runtime) {
  return runtime.deviceComputeCapability == 120 &&
    runtime.boardX == P1_BOARD && runtime.boardY == P1_BOARD &&
    runtime.physicalBatchSize == P1_BATCH &&
    runtime.sameGpuConcurrency == 2 && runtime.exactBoard &&
    runtime.maskMode == MaskModeV1::None && runtime.maskNull &&
    runtime.inputStorage == StorageTypeV1::Fp16 &&
    runtime.outputStorage == StorageTypeV1::Fp16 &&
    runtime.requestedExecution == RequestedExecutionV1::Fp16 &&
    runtime.layout == TensorLayoutV1::Nhwc;
}

bool exactP1Attention(const AttentionSpecV1& spec) {
  return spec.channels == P1_CHANNELS &&
    spec.numHeads == P1_HEADS && spec.numKVHeads == P1_HEADS &&
    spec.qHeadDim == P1_HEAD_DIM && spec.vHeadDim == P1_HEAD_DIM &&
    spec.useRope && spec.learnableRope && !spec.useQKNorm &&
    !spec.hasInputQuantRange && !spec.hasOutputQuantRange;
}

bool exactP1Ffn(const FfnSpecV1& spec) {
  return spec.channels == P1_CHANNELS &&
    spec.hiddenChannels == P1_FFN_CHANNELS && spec.useSwiGLU &&
    spec.clipClass == ClipClassV1::Zero &&
    !spec.hasInputQuantRange && !spec.hasProductQuantRange;
}

bool exactP1Key(const ProfileKeyV1& key) {
  return key.modelVersion == 102 && exactP1Runtime(key.runtime) &&
    exactP1Attention(key.attention) && exactP1Ffn(key.ffn);
}

class DeviceBuffer {
public:
  DeviceBuffer() : pointer(nullptr) {}
  ~DeviceBuffer() {
    if(pointer != nullptr)
      (void)cudaFree(pointer);
  }
  DeviceBuffer(DeviceBuffer&& other) noexcept : pointer(other.pointer) {
    other.pointer = nullptr;
  }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if(this != &other) {
      if(pointer != nullptr)
        (void)cudaFree(pointer);
      pointer = other.pointer;
      other.pointer = nullptr;
    }
    return *this;
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void uploadHalf(const std::string& name, const std::vector<float>& values) {
    if(pointer != nullptr || values.empty())
      throw ErrorV1(name + ": invalid P1 FP16 upload request");
    CudaUtils::mallocAndCopyToDevice(name,values,pointer,true);
  }

  void allocateBytes(const std::string& name, size_t bytes) {
    if(pointer != nullptr || bytes == 0)
      throw ErrorV1(name + ": invalid P1 scratch allocation request");
    const cudaError_t status = cudaMalloc(&pointer,bytes);
    if(status != cudaSuccess)
      throw ErrorV1(name + ": " + cudaGetErrorString(status));
  }

  void* get() const noexcept { return pointer; }

private:
  void* pointer;
};

class QkvRopeHandle {
public:
  QkvRopeHandle() : handle(nullptr) {}
  ~QkvRopeHandle() {
    katago_renju15_qkv_rope_gemm_sm120_destroy(handle);
  }
  QkvRopeHandle(QkvRopeHandle&& other) noexcept : handle(other.handle) {
    other.handle = nullptr;
  }
  QkvRopeHandle& operator=(QkvRopeHandle&& other) noexcept {
    if(this != &other) {
      katago_renju15_qkv_rope_gemm_sm120_destroy(handle);
      handle = other.handle;
      other.handle = nullptr;
    }
    return *this;
  }
  QkvRopeHandle(const QkvRopeHandle&) = delete;
  QkvRopeHandle& operator=(const QkvRopeHandle&) = delete;

  void prepare() {
    if(handle != nullptr)
      throw ErrorV1("P1 QKV+RoPE handle prepared twice");
    handle = katago_renju15_qkv_rope_gemm_sm120_create(
      KATAGO_RENJU15_QKV_ROPE_GEMM_M128_N128_K32_S3,P1_BATCH);
    if(handle == nullptr)
      throw ErrorV1("P1 QKV+RoPE handle is unavailable on the current device");
  }

  void* get() const noexcept { return handle; }

private:
  void* handle;
};

class ResidualHandle {
public:
  ResidualHandle() : handle(nullptr) {}
  ~ResidualHandle() {
    katago_renju15_residual_gemm_sm120_destroy(handle);
  }
  ResidualHandle(ResidualHandle&& other) noexcept : handle(other.handle) {
    other.handle = nullptr;
  }
  ResidualHandle& operator=(ResidualHandle&& other) noexcept {
    if(this != &other) {
      katago_renju15_residual_gemm_sm120_destroy(handle);
      handle = other.handle;
      other.handle = nullptr;
    }
    return *this;
  }
  ResidualHandle(const ResidualHandle&) = delete;
  ResidualHandle& operator=(const ResidualHandle&) = delete;

  void prepare(int family) {
    if(handle != nullptr)
      throw ErrorV1("P1 residual handle prepared twice");
    handle = katago_renju15_residual_gemm_sm120_create(
      family,KATAGO_RENJU15_RESIDUAL_GEMM_M128_N128_K32_S3,
      P1_TOKEN_ROWS);
    if(handle == nullptr)
      throw ErrorV1("P1 residual GEMM handle is unavailable on the current device");
  }

  void* get() const noexcept { return handle; }

private:
  void* handle;
};

void requireMatrix(
  const MatMulLayerDesc& matrix,
  int inChannels,
  int outChannels,
  const std::string& context
) {
  const size_t expected = static_cast<size_t>(inChannels) * outChannels;
  if(matrix.inChannels != inChannels || matrix.outChannels != outChannels ||
     matrix.weights.size() != expected)
    throw ErrorV1(context + ": descriptor matrix shape is not the exact P1 shape");
}

void requireRms(
  const TransformerRMSNormDesc& rms,
  const std::string& context
) {
  if(rms.numChannels != P1_CHANNELS ||
     rms.weight.size() != static_cast<size_t>(P1_CHANNELS) ||
     !std::isfinite(rms.epsilon) || rms.epsilon <= 0.0f || rms.epsilon > 1.0f)
    throw ErrorV1(context + ": descriptor RMSNorm is not the exact P1 shape");
}

std::vector<float> packQkv(const TransformerAttentionDesc& desc) {
  const size_t matrixElements =
    static_cast<size_t>(P1_CHANNELS) * P1_CHANNELS;
  std::vector<float> packed;
  packed.reserve(3 * matrixElements);
  packed.insert(packed.end(),desc.qProj.weights.begin(),desc.qProj.weights.end());
  packed.insert(packed.end(),desc.kProj.weights.begin(),desc.kProj.weights.end());
  packed.insert(packed.end(),desc.vProj.weights.begin(),desc.vProj.weights.end());
  return packed;
}

std::vector<float> packRope(const TransformerAttentionDesc& desc) {
  std::vector<float> cosTable;
  std::vector<float> sinTable;
  desc.computeRopeCosSin(
    P1_BOARD,P1_BOARD,P1_SEQUENCE,cosTable,sinTable);
  const int totalPairs = P1_HEADS * P1_ROPE_PAIRS;
  const size_t expected = static_cast<size_t>(totalPairs) * P1_SEQUENCE;
  if(cosTable.size() != expected || sinTable.size() != expected)
    throw ErrorV1("P1 learned-RoPE table has an unexpected size");

  std::vector<float> packed(expected * 2);
  for(int xy = 0; xy < P1_SEQUENCE; xy++) {
    for(int pair = 0; pair < totalPairs; pair++) {
      const size_t source = static_cast<size_t>(pair) * P1_SEQUENCE + xy;
      const size_t destination =
        (static_cast<size_t>(xy) * totalPairs + pair) * 2;
      packed[destination] = cosTable[source];
      packed[destination + 1] = sinTable[source];
    }
  }
  return packed;
}

std::vector<float> packFusedFfnWeight(const MatMulLayerDesc& matrix) {
  std::vector<float> packed(
    static_cast<size_t>(P1_FFN_CHANNELS) * P1_CHANNELS);
  for(int output = 0; output < P1_FFN_CHANNELS; output++) {
    for(int input = 0; input < P1_CHANNELS; input++) {
      packed[static_cast<size_t>(output) * P1_CHANNELS + input] =
        matrix.weights[static_cast<size_t>(input) * P1_FFN_CHANNELS + output];
    }
  }
  return packed;
}

class P1PreparedAttention final : public PreparedAttentionV1 {
public:
  P1PreparedAttention(size_t layer_, const TransformerAttentionDesc& desc)
    : layer(layer_), epsilon(desc.preLN.epsilon) {
    requireRms(desc.preLN,desc.name + ":attention");
    if(desc.numHeads != P1_HEADS || desc.numKVHeads != P1_HEADS ||
       desc.qHeadDim != P1_HEAD_DIM || desc.vHeadDim != P1_HEAD_DIM ||
       !desc.useRope || !desc.learnableRope || desc.useQKNorm ||
       desc.ropeNumKVHeads != P1_HEADS || desc.ropeNumPairs != P1_ROPE_PAIRS)
      throw ErrorV1(desc.name + ": attention descriptor violates the P1 semantic contract");
    requireMatrix(desc.qProj,P1_CHANNELS,P1_CHANNELS,desc.name + ":qProj");
    requireMatrix(desc.kProj,P1_CHANNELS,P1_CHANNELS,desc.name + ":kProj");
    requireMatrix(desc.vProj,P1_CHANNELS,P1_CHANNELS,desc.name + ":vProj");
    requireMatrix(desc.outProj,P1_CHANNELS,P1_CHANNELS,desc.name + ":outProj");

    gamma.uploadHalf(desc.name + ":p1PreLn",desc.preLN.weight);
    qkvWeights.uploadHalf(desc.name + ":p1Qkv",packQkv(desc));
    ropeCosSin.uploadHalf(desc.name + ":p1Rope",packRope(desc));
    outWeights.uploadHalf(desc.name + ":p1Out",desc.outProj.weights);
    qkv.prepare();
    out.prepare(KATAGO_RENJU15_RESIDUAL_GEMM_OUT_PROJ);
  }

  const size_t layer;
  const float epsilon;
  DeviceBuffer gamma;
  DeviceBuffer qkvWeights;
  DeviceBuffer ropeCosSin;
  DeviceBuffer outWeights;
  QkvRopeHandle qkv;
  ResidualHandle out;
};

class P1PreparedFfn final : public PreparedFfnV1 {
public:
  P1PreparedFfn(size_t layer_, const TransformerFFNDesc& desc)
    : layer(layer_), epsilon(desc.preLN.epsilon) {
    requireRms(desc.preLN,desc.name + ":ffn");
    if(desc.numChannels != P1_CHANNELS ||
       desc.ffnChannels != P1_FFN_CHANNELS || !desc.useSwiGLU ||
       desc.swigluClip != 0.0f)
      throw ErrorV1(desc.name + ": FFN descriptor violates the P1 semantic contract");
    requireMatrix(desc.linear1,P1_CHANNELS,P1_FFN_CHANNELS,desc.name + ":linear1");
    requireMatrix(desc.linearGate,P1_CHANNELS,P1_FFN_CHANNELS,desc.name + ":linearGate");
    requireMatrix(desc.linear2,P1_FFN_CHANNELS,P1_CHANNELS,desc.name + ":linear2");

    gamma.uploadHalf(desc.name + ":p1PreLn",desc.preLN.weight);
    linearWeights.uploadHalf(
      desc.name + ":p1FastLinear",packFusedFfnWeight(desc.linear1));
    gateWeights.uploadHalf(
      desc.name + ":p1FastGate",packFusedFfnWeight(desc.linearGate));
    downWeights.uploadHalf(desc.name + ":p1Down",desc.linear2.weights);
    down.prepare(KATAGO_RENJU15_RESIDUAL_GEMM_FFN_DOWN);
  }

  const size_t layer;
  const float epsilon;
  DeviceBuffer gamma;
  DeviceBuffer linearWeights;
  DeviceBuffer gateWeights;
  DeviceBuffer downWeights;
  ResidualHandle down;
};

class P1Provider final : public ProviderV1 {
public:
  explicit P1Provider(ProfileKeyV1 key_)
    : key(std::move(key_)), generation(0), nextRunToken(0), armedRunToken(0) {
    if(!exactP1Key(key))
      throw ErrorV1("P1 provider was created with a non-P1 key");
    if(!CudaFusedFFN::supportsShape(P1_FFN_CHANNELS,P1_CHANNELS))
      throw ErrorV1("8a fast fused FFN does not support C256/F768");
  }

  const char* profileId() const noexcept override { return P1_PROFILE_ID; }

  PrepareAttentionResultV1 prepareAttention(
    size_t layer,
    const AttentionSpecV1& spec,
    const AttentionLayerScalarsV1& scalars,
    const OfficialAttentionResourcesV1& official
  ) override {
    PrepareAttentionResultV1 result;
    if(generation != 0) {
      result.detail = "P1 attention prepare called after commit";
      return result;
    }
    if(!exactP1Attention(spec) || scalars.inputQuantMaxAbsBits != 0 ||
       scalars.outputQuantMaxAbsBits != 0 || !official.complete()) {
      result.detail = "P1 attention prepare received non-exact semantics or incomplete official resources";
      return result;
    }
    const auto* desc =
      static_cast<const TransformerAttentionDesc*>(official.descriptor);
    result.prepared = std::make_unique<P1PreparedAttention>(layer,*desc);
    return result;
  }

  PrepareFfnResultV1 prepareFfn(
    size_t layer,
    const FfnSpecV1& spec,
    const FfnLayerScalarsV1& scalars,
    const OfficialFfnResourcesV1& official
  ) override {
    PrepareFfnResultV1 result;
    if(generation != 0) {
      result.detail = "P1 FFN prepare called after commit";
      return result;
    }
    if(!exactP1Ffn(spec) || scalars.swigluClipBits != 0 ||
       scalars.inputQuantMaxAbsBits != 0 ||
       scalars.productQuantMaxAbsBits != 0 || !official.complete()) {
      result.detail = "P1 FFN prepare received non-exact semantics or incomplete official resources";
      return result;
    }
    const auto* desc = static_cast<const TransformerFFNDesc*>(official.descriptor);
    result.prepared = std::make_unique<P1PreparedFfn>(layer,*desc);
    return result;
  }

  bool commit(PreparedSpanV1&& prepared, std::string& detail) override {
    if(generation != 0) {
      detail = "P1 provider commit called more than once";
      return false;
    }
    if(prepared.attention.empty() ||
       prepared.attention.size() != prepared.ffn.size()) {
      detail = "P1 provider requires nonempty N/N staged resources";
      return false;
    }

    std::vector<std::unique_ptr<P1PreparedAttention>> stagedAttention;
    std::vector<std::unique_ptr<P1PreparedFfn>> stagedFfn;
    stagedAttention.reserve(prepared.attention.size());
    stagedFfn.reserve(prepared.ffn.size());
    for(size_t layer = 0; layer < prepared.attention.size(); layer++) {
      auto* attention = dynamic_cast<P1PreparedAttention*>(
        prepared.attention[layer].get());
      auto* ffn = dynamic_cast<P1PreparedFfn*>(prepared.ffn[layer].get());
      if(attention == nullptr || ffn == nullptr ||
         attention->layer != layer || ffn->layer != layer) {
        detail = "P1 provider received a foreign, reordered, or missing staged layer";
        return false;
      }
    }
    for(auto& preparedAttention: prepared.attention) {
      stagedAttention.emplace_back(static_cast<P1PreparedAttention*>(
        preparedAttention.release()));
    }
    for(auto& preparedFfn: prepared.ffn) {
      stagedFfn.emplace_back(static_cast<P1PreparedFfn*>(preparedFfn.release()));
    }

    DeviceBuffer stagedQkvScratch;
    DeviceBuffer stagedOperationScratch;
    constexpr size_t qkvElements =
      static_cast<size_t>(3) * P1_TOKEN_ROWS * P1_CHANNELS;
    constexpr size_t operationElements =
      static_cast<size_t>(P1_TOKEN_ROWS) * P1_FFN_CHANNELS;
    stagedQkvScratch.allocateBytes(
      "P1 qkv scratch",qkvElements * sizeof(half));
    stagedOperationScratch.allocateBytes(
      "P1 attention/ffn scratch",operationElements * sizeof(half));

    // Publish all N/N weights, handles, and scratch together. No member used
    // by enqueue is observable as committed before this point.
    attention = std::move(stagedAttention);
    ffn = std::move(stagedFfn);
    qkvScratch = std::move(stagedQkvScratch);
    operationScratch = std::move(stagedOperationScratch);
    generation = 1;
    detail = "P1 dynamic N/N span committed atomically";
    return true;
  }

  uint64_t committedPlanGeneration() const noexcept override {
    return generation;
  }

  ProviderOpResultV1 preflight(const RuntimeCallV1& call) override {
    if(generation == 0 || attention.empty() || attention.size() != ffn.size())
      return ProviderOpResultV1::failure("P1 plan is not committed");
    if(armedRunToken != 0)
      throw FatalErrorV1("P1 preflight called while a prior run token is armed");
    if(call.key != key.runtime || !exactP1Runtime(call.key) ||
       call.actualBatchSize != P1_BATCH || call.sequenceSize != P1_SEQUENCE ||
       call.transformerPairCount != attention.size() || call.mask != nullptr ||
       call.stream == nullptr || !aligned16(call.trunk) ||
       !aligned16(call.trunkScratch) || !aligned16(qkvScratch.get()) ||
       !aligned16(operationScratch.get()))
      return ProviderOpResultV1::failure(
        "P1 exact B36/S225/S2/no-mask/pointer contract rejected before enqueue");

    for(size_t layer = 0; layer < attention.size(); layer++) {
      const P1PreparedAttention& attn = *attention[layer];
      const P1PreparedFfn& feedForward = *ffn[layer];
      if(!aligned16(attn.gamma.get()) || !aligned16(attn.qkvWeights.get()) ||
         !aligned16(attn.ropeCosSin.get()) || !aligned16(attn.outWeights.get()) ||
         !aligned16(feedForward.gamma.get()) ||
         !aligned16(feedForward.linearWeights.get()) ||
         !aligned16(feedForward.gateWeights.get()) ||
         !aligned16(feedForward.downWeights.get()) ||
         !katago_renju15_qkv_rope_gemm_sm120_supports(
           attn.qkv.get(),P1_BATCH,P1_SEQUENCE,P1_CHANNELS,P1_CHANNELS,
           P1_HEADS,P1_HEADS,P1_HEAD_DIM,P1_ROPE_PAIRS,
           true,true,true,true,true) ||
         !katago_renju15_residual_gemm_sm120_supports(
           attn.out.get(),P1_TOKEN_ROWS,P1_CHANNELS,P1_CHANNELS,true,true) ||
         !katago_renju15_residual_gemm_sm120_supports(
           feedForward.down.get(),P1_TOKEN_ROWS,P1_FFN_CHANNELS,
           P1_CHANNELS,true,true))
        return ProviderOpResultV1::failure(
          "P1 prepared layer failed its zero-enqueue launch-shape gate");
    }

    nextRunToken++;
    if(nextRunToken == 0)
      nextRunToken++;
    armedCall = call;
    armedRunToken = nextRunToken;
    return ProviderOpResultV1::success(0,generation,armedRunToken);
  }

  ProviderOpResultV1 enqueue(
    const RuntimeCallV1& call,
    uint64_t runToken
  ) override {
    if(generation == 0 || runToken == 0 || runToken != armedRunToken ||
       !armedCall.sameIdentity(call))
      throw FatalErrorV1("P1 enqueue received a stale token or changed runtime identity");
    armedRunToken = 0;

    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(call.stream);
    half* const trunk = static_cast<half*>(call.trunk);
    half* const normalized = static_cast<half*>(call.trunkScratch);
    half* const qkv = static_cast<half*>(qkvScratch.get());
    half* const q = qkv;
    half* const k = q + static_cast<size_t>(P1_TOKEN_ROWS) * P1_CHANNELS;
    half* const v = k + static_cast<size_t>(P1_TOKEN_ROWS) * P1_CHANNELS;
    half* const operation = static_cast<half*>(operationScratch.get());
    size_t enqueued = 0;

    for(size_t layer = 0; layer < attention.size(); layer++) {
      const P1PreparedAttention& attn = *attention[layer];
      const P1PreparedFfn& feedForward = *ffn[layer];
      launchOrFatal(
        "attention RMSNorm",layer,
        Renju15Sm120::launchRmsNorm256(
          trunk,normalized,static_cast<const half*>(attn.gamma.get()),
          P1_TOKEN_ROWS,attn.epsilon,Renju15Sm120::RmsNorm256Tactic::Warp4Vec8,
          stream),enqueued);
      launchOrFatal(
        "QKV+RoPE",layer,
        katago_renju15_qkv_rope_gemm_sm120_launch(
          attn.qkv.get(),normalized,
          static_cast<const half*>(attn.qkvWeights.get()),
          static_cast<const half2*>(attn.ropeCosSin.get()),qkv,P1_BATCH,stream),
        enqueued);

      const Renju15Fa4Sm120::LaunchResult fa4 = Renju15Fa4Sm120::launch(
        Renju15Fa4Sm120::Tactic::B36Tm128Tn128S1Both16,
        q,k,v,operation,P1_BATCH,P1_SEQUENCE,P1_HEADS,P1_HEADS,
        P1_HEAD_DIM,P1_HEAD_DIM,true,true,nullptr,true,12,0,stream);
      if(!fa4.attempted)
        throw FatalErrorV1(
          std::string(P1_PROFILE_ID) + " committed FA4 refused its exact runtime");
      launchOrFatal("FA4",layer,fa4.status,enqueued);
      launchOrFatal(
        "attention residual",layer,
        katago_renju15_residual_gemm_sm120_launch(
          attn.out.get(),operation,
          static_cast<const half*>(attn.outWeights.get()),trunk,
          P1_TOKEN_ROWS,stream),enqueued);

      launchOrFatal(
        "FFN RMSNorm",layer,
        Renju15Sm120::launchRmsNorm256(
          trunk,normalized,static_cast<const half*>(feedForward.gamma.get()),
          P1_TOKEN_ROWS,feedForward.epsilon,
          Renju15Sm120::RmsNorm256Tactic::Warp4Vec8,stream),enqueued);
      try {
        CudaFusedFFN::runSwiGLU(
          normalized,
          static_cast<const half*>(feedForward.linearWeights.get()),
          static_cast<const half*>(feedForward.gateWeights.get()),operation,
          P1_TOKEN_ROWS,P1_FFN_CHANNELS,P1_CHANNELS,stream);
      }
      catch(const std::exception& e) {
        throw FatalErrorV1(
          std::string(P1_PROFILE_ID) + " layer " + std::to_string(layer) +
          " fast fused FFN failed after enqueue entry: " + e.what());
      }
      enqueued++;
      launchOrFatal(
        "FFN residual",layer,
        katago_renju15_residual_gemm_sm120_launch(
          feedForward.down.get(),operation,
          static_cast<const half*>(feedForward.downWeights.get()),trunk,
          P1_TOKEN_ROWS,stream),enqueued);
    }
    return ProviderOpResultV1::success(enqueued,generation,runToken);
  }

private:
  static void launchOrFatal(
    const char* operation,
    size_t layer,
    cudaError_t status,
    size_t& enqueued
  ) {
    if(status != cudaSuccess)
      throw FatalErrorV1(cudaFailure(operation,status,layer));
    enqueued++;
  }

  const ProfileKeyV1 key;
  std::vector<std::unique_ptr<P1PreparedAttention>> attention;
  std::vector<std::unique_ptr<P1PreparedFfn>> ffn;
  DeviceBuffer qkvScratch;
  DeviceBuffer operationScratch;
  uint64_t generation;
  uint64_t nextRunToken;
  uint64_t armedRunToken;
  RuntimeCallV1 armedCall;
};

}  // namespace

std::unique_ptr<ProviderV1> createP1ProviderV1(const ProfileKeyV1& key) {
  return std::make_unique<P1Provider>(key);
}

}  // namespace FourProfile

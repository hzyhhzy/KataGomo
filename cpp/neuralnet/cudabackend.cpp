#ifdef USE_CUDA_BACKEND
#include "../neuralnet/cudaerrorcheck.h"
#include "../neuralnet/cudaincludes.h"

// cuDNN Frontend is vendored and header-only. Include it before KataGo headers because it bundles
// nlohmann/json 3.11.x under the same include guard as KataGo's older copy. The source-level version
// gate preserves the in-tree fallback when building against older cuDNN headers.
#if defined(KATAGO_ENABLE_CUDNN_FRONTEND_SDPA) && KATAGO_ENABLE_CUDNN_FRONTEND_SDPA && CUDNN_VERSION >= 8903
  #define KATAGO_CUDA_HAS_SDPA 1
  #include <cudnn_frontend.h>
#else
  #define KATAGO_CUDA_HAS_SDPA 0
#endif

#include "../neuralnet/cudahelpers.h"
#include "../neuralnet/cudautils.h"
#include "../neuralnet/architecturedesc.h"
#include "../neuralnet/cudabackend_qkv_planar.h"
#include "../neuralnet/cudabackend_transformer_winner.h"
#include "../neuralnet/cudaopregistry.h"
#include "../neuralnet/int8policy.h"
#if defined(KATAGO_ENABLE_RENJU15_DUAL_FFN_SM120) && KATAGO_ENABLE_RENJU15_DUAL_FFN_SM120
#include "../neuralnet/renju15_dual_ffn_sm120.h"
#endif
#if defined(KATAGO_ENABLE_RENJU15_FA4_SM120) && KATAGO_ENABLE_RENJU15_FA4_SM120
#include "../neuralnet/renju15_fa4_sm120.h"
#endif
#if defined(KATAGO_ENABLE_RENJU15_GEMM_TACTICS_SM120) && KATAGO_ENABLE_RENJU15_GEMM_TACTICS_SM120
#include "../neuralnet/renju15_residual_gemm_sm120.h"
#endif
#if defined(KATAGO_ENABLE_RENJU15_RMS_SM120) && KATAGO_ENABLE_RENJU15_RMS_SM120
#include "../neuralnet/cudabackend_sm120_renju15_kernels.h"
#endif
#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
#include "../neuralnet/renju15_int8_fused_sm120.h"
#include "../neuralnet/renju15_int8_quantization.h"
#endif
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nninterface.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/desc.h"

#include "../core/hash.h"
#include "../core/simpleallocator.h"

#include "../external/half-2.2.0/include/half.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <unordered_map>

//------------------------
#include "../core/using.h"
//------------------------

using half_t = half_float::half;

#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
namespace {

struct CudaDeviceBufferDeleter {
  void operator()(void* pointer) const noexcept {
    if(pointer != nullptr)
      (void)cudaFree(pointer);
  }
};

struct Int8QkKernelDeleter {
  void operator()(void* pointer) const noexcept {
    katago_renju15_int8_qk_sm120_destroy(pointer);
  }
};

struct Int8DualFfnKernelDeleter {
  void operator()(void* pointer) const noexcept {
    katago_renju15_int8_dual_ffn_sm120_destroy(pointer);
  }
};

using UniqueCudaDeviceBuffer = std::unique_ptr<void,CudaDeviceBufferDeleter>;
using UniqueInt8QkKernel = std::unique_ptr<void,Int8QkKernelDeleter>;
using UniqueInt8DualFfnKernel =
  std::unique_ptr<void,Int8DualFfnKernelDeleter>;

void uploadPackedInt8(
  const string& name,
  const vector<int8_t>& packed,
  void*& deviceBuffer
) {
  deviceBuffer = nullptr;
  void* allocated = nullptr;
  CUDA_ERR(name.c_str(),cudaMalloc(&allocated,packed.size()));
  cudaError_t status = cudaMemcpy(
    allocated,packed.data(),packed.size(),cudaMemcpyHostToDevice);
  if(status != cudaSuccess) {
    cudaFree(allocated);
    CUDA_ERR(name.c_str(),status);
  }
  deviceBuffer = allocated;
}

void preparePackedInt8Matrix(
  const string& name,
  const vector<float>& weights,
  int inputChannels,
  int outputChannels,
  void*& deviceBuffer,
  float& scale
) {
  scale = Renju15Int8Quantization::perMatrixScale(weights);
  const vector<int8_t> packed = Renju15Int8Quantization::quantizeAndPackMatrix(
    weights,inputChannels,outputChannels,scale);
  uploadPackedInt8(name,packed,deviceBuffer);
}

void preparePackedInt8Qk(
  const string& name,
  const MatMulLayerDesc& q,
  const MatMulLayerDesc& k,
  void*& deviceBuffer,
  float& scale
) {
  if(q.weights.size() != (size_t)256 * 256 ||
     k.weights.size() != (size_t)256 * 256)
    throw StringError(name + ": INT8 QK weight count mismatch");
  if(q.inChannels != 256 || k.inChannels != 256 ||
     q.outChannels != 256 || k.outChannels != 256)
    throw StringError(name + ": incompatible INT8 QK shape");
  float maxAbs = 0.0f;
  for(float value: q.weights) {
    if(!std::isfinite(value))
      throw StringError(name + ": Q matrix contains NaN or infinity");
    maxAbs = std::max(maxAbs,std::fabs(value));
  }
  for(float value: k.weights) {
    if(!std::isfinite(value))
      throw StringError(name + ": K matrix contains NaN or infinity");
    maxAbs = std::max(maxAbs,std::fabs(value));
  }
  scale = std::max(
    maxAbs / 127.0f,std::numeric_limits<float>::min());
  if(!(scale > 0.0f) || !std::isfinite(scale))
    throw StringError(name + ": QK matrices produced an invalid scale");
  Renju15Int8Quantization::validateContract();

  vector<int8_t> packed((size_t)256 * 512);
  for(int inner = 0; inner < 256; inner++) {
    for(int channel = 0; channel < 256; channel++) {
      packed[(size_t)channel * 256 + inner] =
        Renju15Int8Quantization::quantizeWeight(
        q.weights[(size_t)inner * 256 + channel],scale);
      packed[(size_t)(256 + channel) * 256 + inner] =
        Renju15Int8Quantization::quantizeWeight(
        k.weights[(size_t)inner * 256 + channel],scale);
    }
  }
  uploadPackedInt8(name,packed,deviceBuffer);
}

void prepareEmbeddedPackedInt8(
  const string& name,
  const NativeInt8Quant::Entry& entry,
  void*& deviceBuffer,
  float& scale
) {
  // readTrailer() has already hash-bound these exact packed bytes to their
  // FP32 masters. Do not rebuild or requantize them in the backend.
  scale = NativeInt8Quant::weightScale(entry);
  uploadPackedInt8(name,entry.packedWeights,deviceBuffer);
}

} // namespace
#endif

//Define this to print out some of the intermediate values of the neural net
//#define DEBUG_INTERMEDIATE_VALUES

void NeuralNet::globalInitialize() {
  //Empty for cudnn backend
}

void NeuralNet::globalCleanup() {
  cudaDeviceReset();
}

//---------------------------------------------------------------------------------
// cudnn SDPA support. Graphs + execution plans cached lazily per (batchSize, hasMask).
// Used only when useFP16=true and cudnn supports SDPA at runtime. Otherwise falls back to
// customCudaFlashAttention (see cudahelpers).
//
// Tensor layout: BSHD physical, with strides chosen so that the (B,H,S,D)-dim graph view matches
// the existing CUDA backend's Q/K/V/output buffers from MatMulLayer:
//   element at (n, xy, h, d) lives at offset (h*headDim + d) + (n*seqLen + xy) * (numHeads*headDim).
//
// Masking: when a mask is present, we build a fully-materialized additive attention bias of shape
// [B, 1, S, S] from the [B, S] mask: bias[b,q,k] = (mask[b,k] != 0 ? 0 : -3e4). cudnn does not have
// plans for the [B,1,1,S] broadcast pattern that would let us avoid this materialization, but the
// full bias is correct for arbitrary (non-prefix) masks, which we need to support sub-board games.
// The bias dtype is FP16, exactly matching Q/K/V. A float bias with half Q/K/V can pass graph
// validation on cuDNN 9.x and then be reinterpreted incorrectly by the fused kernel.
//
// When mask is NULL (full-board, requireExactNNLen case), we build a no-bias graph instead, which
// avoids both the extra memory and the bias build kernel.

#if KATAGO_CUDA_HAS_SDPA
struct SDPAPlanForBatchSize {
  std::shared_ptr<cudnn_frontend::graph::Graph> graph;
  int64_t workspaceBytes;
  bool hasMask;  // true if the graph expects a bias variant-pack entry

  // UIDs for the variant pack, fixed at graph build time.
  static constexpr int64_t Q_UID = 1;
  static constexpr int64_t K_UID = 2;
  static constexpr int64_t V_UID = 3;
  static constexpr int64_t O_UID = 4;
  static constexpr int64_t BIAS_UID = 5;
};

// Full discriminating key for an SDPA execution plan. Every field that changes the cudnn graph shape
// must be here: if any attention layer in a future model differs in head count/dim/seqLen, it gets its
// own plan rather than incorrectly reusing another layer's. (batchSize and hasMask vary at runtime.)
struct SDPAGraphKey {
  int numHeads;
  int numKVHeads;
  int qHeadDim;
  int vHeadDim;
  int seqLen;
  int batchSize;
  bool hasMask;
  bool usingFP16;

  bool operator==(const SDPAGraphKey& o) const {
    return
      numHeads == o.numHeads &&
      numKVHeads == o.numKVHeads &&
      qHeadDim == o.qHeadDim &&
      vHeadDim == o.vHeadDim &&
      seqLen == o.seqLen &&
      batchSize == o.batchSize &&
      hasMask == o.hasMask &&
      usingFP16 == o.usingFP16;
  }
};
struct SDPAGraphKeyHash {
  uint64_t operator()(const SDPAGraphKey& k) const noexcept {
    uint64_t acc = (uint64_t)123456789;
    auto mix = [&acc](uint64_t x) {
      acc += x;
      acc += acc << 13;
      acc ^= acc >> 6;
    };
    mix((uint64_t)k.numHeads);
    mix((uint64_t)k.numKVHeads);
    mix((uint64_t)k.qHeadDim);
    mix((uint64_t)k.vHeadDim);
    mix((uint64_t)k.seqLen);
    mix((uint64_t)k.batchSize);
    mix(k.hasMask ? 1 : 0);
    mix(k.usingFP16 ? 1 : 0);
    acc = Hash::basicLCong(acc);
    return (size_t)(acc ^ (acc >> 32));
  }
};

struct SDPAGraphCache {
  std::unordered_map<SDPAGraphKey, std::shared_ptr<SDPAPlanForBatchSize>, SDPAGraphKeyHash> plansByKey;
  // A capability miss is local to the full key. An unsupported mask mode, sequence length, or batch
  // must not disable SDPA for another otherwise-supported attention shape on the same handle.
  std::unordered_map<SDPAGraphKey, string, SDPAGraphKeyHash> unsupportedByKey;

  SDPAGraphCache() :
    plansByKey(),
    unsupportedByKey()
  {}

  // Build (or fetch from cache) an execution plan for the given attention shape + batchSize + hasMask.
  // Construction and support-check failures happen before graph execution and are safely cached as
  // fallback decisions. Once execute is called, a failure is fatal because work might be enqueued.
  std::shared_ptr<SDPAPlanForBatchSize> getOrBuildPlan(cudnnHandle_t cudnn, const SDPAGraphKey& key, Logger* logger) {
    // This route intentionally handles only half IO with FP32 accumulation. FP32 uses the in-tree
    // online-softmax implementation without trying to construct a frontend graph.
    if(!key.usingFP16)
      return nullptr;

    auto it = plansByKey.find(key);
    if(it != plansByKey.end())
      return it->second;
    if(unsupportedByKey.find(key) != unsupportedByKey.end())
      return nullptr;

    namespace fe = cudnn_frontend;

    auto rejectForKey = [&](const string& reason) -> std::shared_ptr<SDPAPlanForBatchSize> {
      unsupportedByKey[key] = reason;
      if(logger != NULL)
        logger->write(
          "Cuda backend: cudnn SDPA unavailable for B=" + Global::intToString(key.batchSize) +
          " Hq=" + Global::intToString(key.numHeads) +
          " Hkv=" + Global::intToString(key.numKVHeads) +
          " S=" + Global::intToString(key.seqLen) +
          " Dq=" + Global::intToString(key.qHeadDim) +
          " Dv=" + Global::intToString(key.vHeadDim) +
          " mask=" + Global::boolToString(key.hasMask) +
          "; using custom attention fallback: " + reason
        );
      return nullptr;
    };

    // Keep a runtime-library check in addition to the compile-time header gate. This makes a
    // mismatched deployment fail closed to the generic attention implementation.
    if(cudnnGetVersion() < 8903)
      return rejectForKey("runtime cuDNN is older than 8.9.3");
    if(key.batchSize <= 0 || key.seqLen <= 0 || key.numHeads <= 0 || key.numKVHeads <= 0 ||
       key.qHeadDim <= 0 || key.vHeadDim <= 0 || key.numHeads % key.numKVHeads != 0)
      return rejectForKey("invalid or unsupported attention dimensions");
    auto plan = std::make_shared<SDPAPlanForBatchSize>();
    plan->hasMask = key.hasMask;
    auto graph = std::make_shared<fe::graph::Graph>();

    bool useFP16 = key.usingFP16;

    fe::DataType_t ioType = useFP16 ? fe::DataType_t::HALF : fe::DataType_t::FLOAT;
    graph->set_io_data_type(ioType)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);

    int64_t B = key.batchSize;
    int64_t Hq = key.numHeads;
    int64_t Hkv = key.numKVHeads;
    int64_t S = key.seqLen;
    int64_t Dq = key.qHeadDim;
    int64_t Dv = key.vHeadDim;

    // BSHD physical layout, with logical dim ordering (B, H, S, D):
    // stride for B = S * H_inner * D
    // stride for H = D
    // stride for S = H_inner * D
    // stride for D = 1
    // where H_inner is the number of heads packed for this tensor (numHeads or numKVHeads).
    int64_t qHinner = key.numHeads;
    int64_t kHinner = key.numKVHeads;
    int64_t vHinner = key.numKVHeads;

    auto Q = graph->tensor(
      fe::graph::Tensor_attributes()
      .set_name("Q")
      .set_uid(SDPAPlanForBatchSize::Q_UID)
      .set_dim({B, Hq, S, Dq})
      .set_stride({S * qHinner * Dq, Dq, qHinner * Dq, 1})
    );
    auto K = graph->tensor(
      fe::graph::Tensor_attributes()
      .set_name("K")
      .set_uid(SDPAPlanForBatchSize::K_UID)
      .set_dim({B, Hkv, S, Dq})
      .set_stride({S * kHinner * Dq, Dq, kHinner * Dq, 1})
    );
    auto V = graph->tensor(
      fe::graph::Tensor_attributes()
      .set_name("V")
      .set_uid(SDPAPlanForBatchSize::V_UID)
      .set_dim({B, Hkv, S, Dv})
      .set_stride({S * vHinner * Dv, Dv, vHinner * Dv, 1})
    );

    float scale = 1.0f / std::sqrt((float)key.qHeadDim);
    auto sdpa_options = (
      fe::graph::SDPA_attributes()
      .set_name("sdpa_fwd")
      .set_generate_stats(false)
      .set_attn_scale(scale)
    );

    if(key.hasMask) {
      // Full [B, 1, S, S] additive bias, broadcast over heads only. Per cudnn 9.8 empirical
      // testing the broadcast-over-q variant ([B,1,1,S]) has no supported plans for our shape.
      auto bias = graph->tensor(
        fe::graph::Tensor_attributes()
        .set_name("bias")
        .set_uid(SDPAPlanForBatchSize::BIAS_UID)
        .set_dim({B, 1, S, S})
        .set_stride({S * S, S * S, S, 1})
      );
      sdpa_options.set_bias(bias);
    }

    auto [O, Stats] = graph->sdpa(Q, K, V, sdpa_options);
    (void)Stats;

    // Output O also uses BSHD physical layout (matches what outProj expects).
    int64_t oHinner = key.numHeads;
    O->set_output(true)
      .set_dim({B, Hq, S, Dv})
      .set_stride({S * oHinner * Dv, Dv, oHinner * Dv, 1})
      .set_uid(SDPAPlanForBatchSize::O_UID);

    auto status = graph->validate();
    if(status.is_bad())
      return rejectForKey(string("graph validate failed: ") + status.get_message());
    status = graph->build_operation_graph(cudnn);
    if(status.is_bad())
      return rejectForKey(string("build_operation_graph failed: ") + status.get_message());
    status = graph->create_execution_plans({fe::HeurMode_t::A});
    if(status.is_bad())
      return rejectForKey(string("create_execution_plans failed: ") + status.get_message());
    status = graph->check_support(cudnn);
    if(status.is_bad())
      return rejectForKey(string("check_support failed: ") + status.get_message());
    status = graph->build_plans(cudnn);
    if(status.is_bad())
      return rejectForKey(string("build_plans failed: ") + status.get_message());

    int64_t ws = 0;
    status = graph->get_workspace_size(ws);
    if(status.is_bad())
      return rejectForKey(string("get_workspace_size failed: ") + status.get_message());
    if(ws < 0)
      return rejectForKey("get_workspace_size returned a negative size");

    plan->graph = graph;
    plan->workspaceBytes = ws;
    plansByKey[key] = plan;
    return plan;
  }
};
#else
struct SDPAGraphCache {
  SDPAGraphCache() {}
};
#endif


struct CudaHandles {
  cublasHandle_t cublas;
  cudnnHandle_t cudnn;
  // Every library operation, custom kernel, copy, and event for this handle is ordered on one
  // owned nonblocking stream. Distinct ComputeHandles can therefore execute independently.
  cudaStream_t stream;
  const int majorComputeCapability;
  const int minorComputeCapability;
  std::unique_ptr<SDPAGraphCache> sdpaCache;
  std::unique_ptr<CudaTransformerWinner::PreparedPlan> transformerPlan;
  bool loggedRms;
  bool loggedPlanarQkv;
  bool loggedLearnedRope;
  bool loggedQkvRope;
  bool loggedFa4;
  bool loggedDualFfn;
  bool loggedInt8Qk;
  bool loggedInt8DualFfn;
  bool loggedInt8Experiment;
  bool loggedOutProjection;
  bool loggedFfnDown;
  bool loggedCublasResidual;
  bool loggedWinner;
#if KATAGO_CUDA_HAS_SDPA
  std::unordered_set<SDPAGraphKey,SDPAGraphKeyHash> loggedSdpaKeys;
#endif
  bool exactWinnerPlan;
#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
  bool int8ExperimentPlan;
  int expectedInt8Qk;
  int expectedInt8DualFfn;
  int preparedInt8Qk;
  int preparedInt8DualFfn;
  int activeInt8Qk;
  int activeInt8DualFfn;
  NativeInt8Quant::WeightSource int8WeightSource;
  const NativeInt8Quant::Metadata* embeddedInt8Metadata;
  // Construction-only rollback hooks. They are discarded after both handle
  // validation and persistent scratch preparation have committed.
  std::vector<std::function<void()>> int8PreparedCleanupRegistry;
#endif
  int expectedWinnerRms;
  int expectedWinnerQkvRope;
  int expectedWinnerFa4;
  int expectedWinnerDualFfn;
  int expectedWinnerOutProjection;
  int expectedWinnerFfnDown;
  int preparedWinnerQkvRope;
  int preparedWinnerDualFfn;
  int preparedWinnerOutProjection;
  int preparedWinnerFfnDown;
  int activeWinnerRms;
  int activeWinnerQkvRope;
  int activeWinnerFa4;
  int activeWinnerDualFfn;
  int activeWinnerOutProjection;
  int activeWinnerFfnDown;
  // Logger for this handle's server thread; may be NULL. Used to report cudnn SDPA falling back.
  Logger* logger;

  CudaHandles(int major, int minor)
    : cublas(NULL),
      cudnn(NULL),
      stream(NULL),
      majorComputeCapability(major),
      minorComputeCapability(minor),
      sdpaCache(std::make_unique<SDPAGraphCache>()),
      transformerPlan(nullptr),
      loggedRms(false),
      loggedPlanarQkv(false),
      loggedLearnedRope(false),
      loggedQkvRope(false),
      loggedFa4(false),
      loggedDualFfn(false),
      loggedInt8Qk(false),
      loggedInt8DualFfn(false),
      loggedInt8Experiment(false),
      loggedOutProjection(false),
      loggedFfnDown(false),
      loggedCublasResidual(false),
      loggedWinner(false),
#if KATAGO_CUDA_HAS_SDPA
      loggedSdpaKeys(),
#endif
      exactWinnerPlan(false),
#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
      int8ExperimentPlan(false),
      expectedInt8Qk(0),
      expectedInt8DualFfn(0),
      preparedInt8Qk(0),
      preparedInt8DualFfn(0),
      activeInt8Qk(0),
      activeInt8DualFfn(0),
      int8WeightSource(NativeInt8Quant::WeightSource::None),
      embeddedInt8Metadata(nullptr),
      int8PreparedCleanupRegistry(),
#endif
      expectedWinnerRms(0),
      expectedWinnerQkvRope(0),
      expectedWinnerFa4(0),
      expectedWinnerDualFfn(0),
      expectedWinnerOutProjection(0),
      expectedWinnerFfnDown(0),
      preparedWinnerQkvRope(0),
      preparedWinnerDualFfn(0),
      preparedWinnerOutProjection(0),
      preparedWinnerFfnDown(0),
      activeWinnerRms(0),
      activeWinnerQkvRope(0),
      activeWinnerFa4(0),
      activeWinnerDualFfn(0),
      activeWinnerOutProjection(0),
      activeWinnerFfnDown(0),
      logger(NULL)
  {
    CUDA_ERR("CudaHandles",cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    try {
      CUBLAS_ERR("CudaHandles",cublasCreate(&cublas));
      CUDNN_ERR("CudaHandles",cudnnCreate(&cudnn));
      CUBLAS_ERR("CudaHandles",cublasSetStream(cublas,stream));
      CUDNN_ERR("CudaHandles",cudnnSetStream(cudnn,stream));
    }
    catch(...) {
      if(cudnn != NULL)
        cudnnDestroy(cudnn);
      if(cublas != NULL)
        cublasDestroy(cublas);
      cudaStreamDestroy(stream);
      throw;
    }
  }

  ~CudaHandles() {
    // Destructors cannot report CUDA errors. All ordinary inference/benchmark synchronizations use
    // CUDA_ERR; this final wait only preserves lifetime ordering for teardown.
    if(stream != NULL)
      cudaStreamSynchronize(stream);
    if(cublas != NULL)
      cublasDestroy(cublas);
    if(cudnn != NULL)
      cudnnDestroy(cudnn);
    if(stream != NULL)
      cudaStreamDestroy(stream);
  }

  void configureWinnerExpectations(
    bool int8RuntimeEnabled,
    const ModelDesc& modelDesc
  ) {
    if(transformerPlan == nullptr)
      return;
#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
    int8WeightSource = NativeInt8Quant::weightSourceForModel(modelDesc);
    if(int8WeightSource == NativeInt8Quant::WeightSource::EmbeddedV104) {
      embeddedInt8Metadata = &modelDesc.nativeInt8Quant;
    }
    else
      embeddedInt8Metadata = nullptr;
    const char* weightSource = NativeInt8Quant::weightSourceName(int8WeightSource);
    if(logger != NULL)
      logger->write(
        string("RENJU15_SM120_INT8_EXPERIMENT_RUNTIME_GATE enabled=") +
        (int8RuntimeEnabled ? "1" : "0") + " weight_source=" + weightSource);
    if(int8RuntimeEnabled &&
       int8WeightSource == NativeInt8Quant::WeightSource::LegacyImplicitV102 &&
       logger != NULL)
      logger->write(
        "RENJU15_SM120_INT8_LEGACY_COMPAT model_version=102 "
        "weights=load-time-quant prefer=native-v104-embedded");
    const CudaTransformerWinner::Int8ExperimentEligibility int8Eligibility =
      CudaTransformerWinner::evaluateInt8ExperimentEligibility(*transformerPlan);
    if(int8RuntimeEnabled && logger != NULL)
      logger->write(
        string("RENJU15_SM120_INT8_EXPERIMENT_ELIGIBILITY architecture=") +
        (int8Eligibility.architectureSignatureMatches ? "1" : "0") +
        " explicit_v104=" +
        (int8Eligibility.explicitV104ArchitectureSignatureMatches ? "1" : "0") +
        " legacy_v102=" +
        (int8Eligibility.legacyV102ArchitectureSignatureMatches ? "1" : "0") +
        " plan_fingerprint=" +
        (int8Eligibility.preparedPlanFingerprintValid ? "1" : "0") +
        " runtime_contract=" +
        (int8Eligibility.runtimeContractEligible ? "1" : "0") +
        " records_prepared=" +
        (int8Eligibility.allTransformerRecordsPrepared ? "1" : "0") +
        " attention=" +
        Global::intToString(int8Eligibility.attentionCount) + " ffn=" +
        Global::intToString(int8Eligibility.ffnCount) + " all=" +
        (int8Eligibility.allTransformerShapesEligible ? "1" : "0"));
    // This first experiment has only been accuracy-qualified for the current
    // 24-attention/24-FFN model on a 15x15 board. Runtime batch remains dynamic.
    const bool modelFormatMatches =
      (int8WeightSource == NativeInt8Quant::WeightSource::EmbeddedV104 &&
       int8Eligibility.explicitV104ArchitectureSignatureMatches) ||
      (int8WeightSource == NativeInt8Quant::WeightSource::LegacyImplicitV102 &&
       int8Eligibility.legacyV102ArchitectureSignatureMatches);
    int8ExperimentPlan = int8RuntimeEnabled && modelFormatMatches &&
      int8Eligibility.exactCurrent24LayerModel();
    if(int8ExperimentPlan) {
      expectedInt8Qk = int8Eligibility.attentionCount;
      expectedInt8DualFfn = int8Eligibility.ffnCount;
    }
#else
    (void)int8RuntimeEnabled;
    (void)modelDesc;
#endif
    for(const CudaTransformerWinner::PreparedRecord& record:
        transformerPlan->records) {
      if(record.found &&
         record.operation.support == CudaOpRegistry::SupportClass::CertifiedFast &&
         record.request.key.kind ==
           NeuralNetArchitecture::ArchitectureOpKind::TransformerAttention &&
         transformerPlan->attentionFor(record.request.topologyIndex).attention ==
           CudaTransformerWinner::AttentionTactic::Fa4Sm120B36S225Tm128Tn128S1Both16) {
        exactWinnerPlan = true;
        break;
      }
    }
    if(!exactWinnerPlan)
      return;
    for(const CudaTransformerWinner::PreparedRecord& record:
        transformerPlan->records) {
      if(record.request.key.kind ==
         NeuralNetArchitecture::ArchitectureOpKind::TransformerAttention) {
        const CudaTransformerWinner::AttentionRecipe recipe =
          transformerPlan->attentionFor(record.request.topologyIndex);
        if(recipe.rmsNorm ==
           CudaTransformerWinner::RmsNormTactic::Sm120C256Warp4Vec8)
          expectedWinnerRms++;
        if(recipe.qkvRope != CudaTransformerWinner::QkvRopeTactic::Disabled)
          expectedWinnerQkvRope++;
        if(recipe.attention != CudaTransformerWinner::AttentionTactic::Generic)
          expectedWinnerFa4++;
        if(recipe.outProjection ==
           CudaTransformerWinner::ResidualTactic::Sm120M128N128K32S3Sw1)
          expectedWinnerOutProjection++;
      }
      else if(record.request.key.kind ==
              NeuralNetArchitecture::ArchitectureOpKind::TransformerFFN) {
        const CudaTransformerWinner::FfnRecipe recipe =
          transformerPlan->ffnFor(record.request.topologyIndex);
        if(recipe.rmsNorm ==
           CudaTransformerWinner::RmsNormTactic::Sm120C256Warp4Vec8)
          expectedWinnerRms++;
        if(recipe.dualFfn != CudaTransformerWinner::DualFfnTactic::Disabled)
          expectedWinnerDualFfn++;
        if(recipe.downProjection ==
           CudaTransformerWinner::ResidualTactic::Sm120M128N128K32S3Sw1)
          expectedWinnerFfnDown++;
      }
    }
  }

#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
  bool usesEmbeddedInt8Weights() const {
    return int8WeightSource == NativeInt8Quant::WeightSource::EmbeddedV104;
  }

  const NativeInt8Quant::Entry& requireEmbeddedInt8Entry(
    uint32_t topologyIndex,
    NativeInt8Quant::Role role,
    const vector<string>& layerNames,
    uint32_t inputChannels,
    uint32_t outputChannels
  ) const {
    if(!usesEmbeddedInt8Weights() || embeddedInt8Metadata == nullptr)
      throw StringError("CUDA embedded INT8 entry requested without v104 metadata");
    return NativeInt8Quant::requireEntry(
      *embeddedInt8Metadata,topologyIndex,role,layerNames,
      inputChannels,outputChannels);
  }

  void finishEmbeddedInt8MetadataConsumption() {
    // Every block has copied its packed bytes to model-owned device buffers.
    // Avoid retaining a non-owning LoadedModel pointer past construction.
    embeddedInt8Metadata = nullptr;
  }

  void registerInt8PreparedCleanup(std::function<void()> cleanup) {
    int8PreparedCleanupRegistry.push_back(std::move(cleanup));
  }

  void disableInt8Experiment(const string& reason) {
    if(!int8ExperimentPlan)
      return;
    const int preparedQkBeforeCleanup = preparedInt8Qk;
    const int preparedDualBeforeCleanup = preparedInt8DualFfn;
    int8ExperimentPlan = false;
    for(auto cleanup = int8PreparedCleanupRegistry.rbegin();
        cleanup != int8PreparedCleanupRegistry.rend(); ++cleanup)
      (*cleanup)();
    int8PreparedCleanupRegistry.clear();
    preparedInt8Qk = 0;
    preparedInt8DualFfn = 0;
    activeInt8Qk = 0;
    activeInt8DualFfn = 0;
    if(logger != NULL) {
      logger->write(
        "RENJU15_SM120_INT8_EXPERIMENT_UNAVAILABLE fallback=fp16 reason=" +
        reason + " prepared_qk=" +
        Global::intToString(preparedQkBeforeCleanup) + "/" +
        Global::intToString(expectedInt8Qk) + " prepared_dual_ffn=" +
        Global::intToString(preparedDualBeforeCleanup) + "/" +
        Global::intToString(expectedInt8DualFfn));
    }
  }

  void noteInt8PreparationFailure(const char* reason) noexcept {
    // Preparation happens before inference can enqueue INT8 work. Consume a
    // sticky allocation/copy error and roll back every already-prepared layer;
    // the independently prepared FP16 model remains authoritative.
    (void)cudaGetLastError();
    try {
      disableInt8Experiment(reason);
    }
    catch(...) {
      // Logging/allocation must never turn optional INT8 preparation into a
      // ComputeHandle construction failure.
      int8ExperimentPlan = false;
      for(auto cleanup = int8PreparedCleanupRegistry.rbegin();
          cleanup != int8PreparedCleanupRegistry.rend(); ++cleanup)
        (*cleanup)();
      int8PreparedCleanupRegistry.clear();
      preparedInt8Qk = 0;
      preparedInt8DualFfn = 0;
      activeInt8Qk = 0;
      activeInt8DualFfn = 0;
    }
  }

  void commitInt8PreparedResources() {
    // Once persistent scratch also exists, model-owned RAII fields become the
    // sole owners and no construction rollback callback may outlive a block.
    int8PreparedCleanupRegistry.clear();
  }
#endif

  void validateWinnerPrepared() {
#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
    if(int8ExperimentPlan &&
       (preparedInt8Qk != expectedInt8Qk ||
         preparedInt8DualFfn != expectedInt8DualFfn)) {
      // A preparation miss is known before inference enqueues any work. Fall
      // back coherently to the already-prepared FP16 plan for every block.
      noteInt8PreparationFailure("incomplete-handle-preparation");
    }
#endif
    if(!exactWinnerPlan)
      return;
    if(preparedWinnerQkvRope != expectedWinnerQkvRope ||
       preparedWinnerDualFfn != expectedWinnerDualFfn ||
       preparedWinnerOutProjection != expectedWinnerOutProjection ||
       preparedWinnerFfnDown != expectedWinnerFfnDown)
      throw StringError("Certified CUDA transformer winner failed to prepare every block handle");
  }

  void noteWinnerLaunch(bool& blockCounted, int& activeCount) {
    if(!exactWinnerPlan || blockCounted)
      return;
    blockCounted = true;
    activeCount++;
    maybeLogWinnerActive();
  }

  void maybeLogWinnerActive() {
#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
    // INT8 changes the arithmetic recipe, so it must never borrow the certified
    // FP16 winner marker even when all unchanged downstream kernels are active.
    if(int8ExperimentPlan)
      return;
#endif
    if(loggedWinner || logger == NULL || transformerPlan == nullptr ||
       !exactWinnerPlan || activeWinnerRms != expectedWinnerRms ||
       activeWinnerQkvRope != expectedWinnerQkvRope ||
       activeWinnerFa4 != expectedWinnerFa4 ||
       activeWinnerDualFfn != expectedWinnerDualFfn ||
       activeWinnerOutProjection != expectedWinnerOutProjection ||
       activeWinnerFfnDown != expectedWinnerFfnDown)
      return;
    logger->write(
      "CUDA_TRANSFORMER_WINNER_ACTIVE qualification=certified-fast plan=" +
      transformerPlan->fingerprint.toHex() + " streams=" +
      Global::intToString((int)transformerPlan->runtime.streamCount));
    loggedWinner = true;
  }

#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
  void noteInt8ExperimentLaunch(bool& blockCounted, bool qk) {
    if(!int8ExperimentPlan || blockCounted)
      return;
    blockCounted = true;
    if(qk)
      activeInt8Qk++;
    else
      activeInt8DualFfn++;
    if(!loggedInt8Experiment && logger != NULL &&
       activeInt8Qk == expectedInt8Qk &&
       activeInt8DualFfn == expectedInt8DualFfn) {
      logger->write(
        "RENJU15_SM120_INT8_EXPERIMENT_ACTIVE quant=clip4-pt "
        "qk_ops=24 dual_ffn_ops=24 upgate_matrices=48 down=fp16 board=15 "
        "weights=" + string(
          int8WeightSource == NativeInt8Quant::WeightSource::EmbeddedV104 ?
            "embedded-v104" : "legacy-v102-load-time-quant") +
        " int8_scratch=persistent-max-batch");
      loggedInt8Experiment = true;
    }
  }
#endif

  static CudaHandles* cudaHandlesTesting() {
    const int gpuIdxForThisThread = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,gpuIdxForThisThread);
    return new CudaHandles(prop.major, prop.minor);
  }

  CudaHandles(const CudaHandles&) = delete;
  CudaHandles& operator=(const CudaHandles&) = delete;
};

//---------------------------------------------------------------------------------

template<typename T>
struct ByBatchSize {
  const int maxBatchSize;
  T* data;
  cudnnStatus_t (*destroyFunc)(T);

  ByBatchSize()
    : maxBatchSize(0), data(nullptr), destroyFunc(nullptr)
  {}

  ByBatchSize(
    int maxBatchSize_
  ) : maxBatchSize(maxBatchSize_), data(nullptr), destroyFunc(nullptr) {
    data = new T[maxBatchSize];
  }

  ByBatchSize(const ByBatchSize&) = delete;
  ByBatchSize& operator=(const ByBatchSize&) = delete;

  ~ByBatchSize() {
    if(destroyFunc != nullptr && data != nullptr) {
      for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
        (*destroyFunc)(data[batchSize-1]);
      }
    }
    if(data != nullptr) {
      delete[] data;
      data = nullptr;
    }
  }
  T& operator[](int batchSize) {
    return data[batchSize-1];
  }
  const T& operator[](int batchSize) const {
    return data[batchSize-1];
  }
};

template<typename T>
struct ByBatchSizeView {
  int maxBatchSize;
  T* data;

  ByBatchSizeView()
    : maxBatchSize(0), data(nullptr)
  {}

  ByBatchSizeView(const ByBatchSize<T>& toView)
    : maxBatchSize(toView.maxBatchSize), data(toView.data)
  {}
  ByBatchSizeView& operator=(const ByBatchSize<T>& toView) {
    maxBatchSize = toView.maxBatchSize;
    data = toView.data;
  }

  ~ByBatchSizeView() {
  }
  T& operator[](int batchSize) {
    return data[batchSize-1];
  }
  const T& operator[](int batchSize) const {
    return data[batchSize-1];
  }
};

//---------------------------------------------------------------------------------


//channels, useFP16, useNHWC
typedef std::tuple<int, bool, bool> CudnnTensorDesc4DKey;

struct CudnnManager {
  const string name;
  const int maxBatchSize;
  const int nnXLen;
  const int nnYLen;
  std::map<CudnnTensorDesc4DKey, ByBatchSize<cudnnTensorDescriptor_t>*> tensorDesc4DByBatchSizeByKey;

  CudnnManager(string name_, int maxBatchSize_, int nnXLen_, int nnYLen_)
    :name(name_),
     maxBatchSize(maxBatchSize_),
     nnXLen(nnXLen_),
     nnYLen(nnYLen_),
     tensorDesc4DByBatchSizeByKey()
  {
  }

  ~CudnnManager() {
    for(auto& iter: tensorDesc4DByBatchSizeByKey) {
      delete iter.second;
    }
  }

  ByBatchSizeView<cudnnTensorDescriptor_t> getTensorDesc4DByBatchSize(
    int channels, bool useFP16, bool useNHWC
  ) {
    auto iter = tensorDesc4DByBatchSizeByKey.find({channels, useFP16, useNHWC});
    if(iter != tensorDesc4DByBatchSizeByKey.end()) {
      return ByBatchSizeView<cudnnTensorDescriptor_t>(*(iter->second));
    }
    ByBatchSize<cudnnTensorDescriptor_t>* descs = new ByBatchSize<cudnnTensorDescriptor_t>(maxBatchSize);
    for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
      cudnnTensorDescriptor_t& desc = (*descs)[batchSize];
      CUDNN_ERR(name.c_str(),cudnnCreateTensorDescriptor(&desc));
      CUDNN_ERR(name.c_str(),cudnnSetTensor4dDescriptor(
                  desc,
                  (useNHWC ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW),
                  (useFP16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT),
                  batchSize,
                  channels,
                  nnYLen,
                  nnXLen
                ));
    }
    descs->destroyFunc = cudnnDestroyTensorDescriptor;
    tensorDesc4DByBatchSizeByKey[{channels, useFP16, useNHWC}] = descs;
    return ByBatchSizeView<cudnnTensorDescriptor_t>(*descs);
  }
};

//---------------------------------------------------------------------------------

struct ScratchBuffers {

  const size_t batchXYFloatBytes;
  const size_t batchFloatBytes;
  const size_t batchXYBytes;
  const size_t batchBytes;

  SimpleAllocator<void*>* allocator;

  // Not scratch, but convenient to have here
  void* zeroBuf;
  void* oneBuf;
#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
  // One persistent pair per ComputeHandle/owned stream. Transformer blocks are
  // sequential on that stream, so all layers and all actual batch sizes reuse
  // these max-batch allocations without lifetime overlap or allocator lookup.
  void* int8NormBuf;
  void* int8QkTempBuf;
  cudaError_t int8ScratchPrepareStatus;
#endif

  ScratchBuffers() = delete;
  ScratchBuffers(const ScratchBuffers&) = delete;
  ScratchBuffers& operator=(const ScratchBuffers&) = delete;

  ScratchBuffers(
    int maxBatchSize,
    int nnXLen,
    int nnYLen,
    bool useFP16
  )
    : batchXYFloatBytes((size_t)maxBatchSize * nnXLen * nnYLen * sizeof(float)),
      batchFloatBytes((size_t)maxBatchSize * sizeof(float)),
      batchXYBytes((size_t)maxBatchSize * nnXLen * nnYLen * (useFP16 ? sizeof(half_t) : sizeof(float))),
      batchBytes((size_t)maxBatchSize * (useFP16 ? sizeof(half_t) : sizeof(float))),
      allocator(nullptr),
      zeroBuf(nullptr),
      oneBuf(nullptr)
#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
      , int8NormBuf(nullptr),
      int8QkTempBuf(nullptr),
      int8ScratchPrepareStatus(cudaSuccess)
#endif
  {
    std::function<void*(size_t)> allocateFunc = [](size_t size) {
      void* buf;
      CUDA_ERR("ScratchBuffers",cudaMalloc(&buf, size));
      return buf;
    };
    std::function<void(void*)> releaseFunc = [](void* buf) {
      cudaFree(buf);
    };

    try {
      allocator = new SimpleAllocator<void*>(allocateFunc, releaseFunc);
      CudaUtils::hostMallocZeroOneBufs(zeroBuf, oneBuf, useFP16);
    }
    catch(...) {
      delete allocator;
      allocator = nullptr;
#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
      cudaFree(int8QkTempBuf);
      cudaFree(int8NormBuf);
      int8QkTempBuf = nullptr;
      int8NormBuf = nullptr;
#endif
      if(zeroBuf != nullptr)
        free(zeroBuf);
      if(oneBuf != nullptr)
        free(oneBuf);
      zeroBuf = nullptr;
      oneBuf = nullptr;
      throw;
    }
  }
  ~ScratchBuffers() {
#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
    cudaFree(int8QkTempBuf);
    cudaFree(int8NormBuf);
#endif
    delete allocator;
    free(zeroBuf);
    free(oneBuf);
  }

#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
  cudaError_t tryPrepareInt8ExperimentScratch(
    int maxBatchSize,
    int nnXLen,
    int nnYLen
  ) {
    if(hasInt8ExperimentScratch())
      return cudaSuccess;
    int8ScratchPrepareStatus = cudaSuccess;
    const size_t maxTokenRows = (size_t)maxBatchSize * nnXLen * nnYLen;
    int8ScratchPrepareStatus = cudaMalloc(
      &int8NormBuf,maxTokenRows * 256);
    if(int8ScratchPrepareStatus == cudaSuccess) {
      int8ScratchPrepareStatus = cudaMalloc(
        &int8QkTempBuf,maxTokenRows * 512 * sizeof(half));
    }
    if(int8ScratchPrepareStatus != cudaSuccess) {
      (void)cudaFree(int8QkTempBuf);
      (void)cudaFree(int8NormBuf);
      int8QkTempBuf = nullptr;
      int8NormBuf = nullptr;
      // This miss is consumed here rather than leaking into a later
      // cudaPeekAtLastError on the coherent FP16 fallback path.
      (void)cudaGetLastError();
    }
    return int8ScratchPrepareStatus;
  }

  bool hasInt8ExperimentScratch() const {
    return int8NormBuf != nullptr && int8QkTempBuf != nullptr &&
      int8ScratchPrepareStatus == cudaSuccess;
  }

#endif

  size_t getBufSizeXY(int channels) const {
    return channels * batchXYBytes;
  }
  size_t getBufSizeXYFloat(int channels) const {
    return channels * batchXYFloatBytes;
  }
  size_t getBufSizeFloat(int channels) const {
    return channels * batchFloatBytes;
  }
  size_t getBufSize(int channels) const {
    return channels * batchBytes;
  }

};


//---------------------------------------------------------------------------------

struct ConvLayer {
  const string name;
  const int inChannels;
  const int outChannels;
  ByBatchSizeView<cudnnTensorDescriptor_t> inputDescriptors;
  ByBatchSizeView<cudnnTensorDescriptor_t> outputDescriptors;
  cudnnFilterDescriptor_t filterDescriptor;
  cudnnConvolutionDescriptor_t convolutionDescriptor;
#if CUDNN_MAJOR >= 8
  ByBatchSize<cudnnConvolutionFwdAlgoPerf_t>* convolutionAlgorithms; //array of one for each batch size
#else
  ByBatchSize<cudnnConvolutionFwdAlgo_t>* convolutionAlgorithms; //array of one for each batch size
#endif
  void* filterBuf;

  ConvLayer() = delete;
  ConvLayer(const ConvLayer&) = delete;
  ConvLayer& operator=(const ConvLayer&) = delete;

  ConvLayer(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const ConvLayerDesc* desc,
    bool useFP16,
    bool useNHWC
  ) : ConvLayer(cudaHandles, manager, desc, useFP16, useNHWC, useNHWC)
  {}

  ConvLayer(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const ConvLayerDesc* desc,
    bool useFP16,
    bool useNHWCIn,
    bool useNHWCOut
  ) :
    name(desc->name),
    inChannels(desc->inChannels),
    outChannels(desc->outChannels)
  {
    int convYSize = desc->convYSize;
    int convXSize = desc->convXSize;
    int dilationY = desc->dilationY;
    int dilationX = desc->dilationX;
    int paddingX = (convXSize / 2) * dilationX;
    int paddingY = (convYSize / 2) * dilationY;

    assert(convXSize % 2 == 1);
    assert(convYSize % 2 == 1);

    inputDescriptors = manager->getTensorDesc4DByBatchSize(inChannels,useFP16,useNHWCIn);
    outputDescriptors = manager->getTensorDesc4DByBatchSize(outChannels,useFP16,useNHWCOut);
    int maxBatchSize = manager->maxBatchSize;

    bool filterNHWC = useNHWCOut && dilationY == 1 && dilationX == 1;

    CUDNN_ERR(name.c_str(),cudnnCreateFilterDescriptor(&filterDescriptor));
    CUDNN_ERR(name.c_str(),cudnnSetFilter4dDescriptor(
      filterDescriptor,
      (useFP16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT),
      (filterNHWC ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW),
      outChannels,
      inChannels,
      convYSize,
      convXSize
    ));

    int yStride = 1;
    int xStride = 1;

    //NVIDIA compute capability 7 is when we first hit Volta architecture, with tensor cores
    //See https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications
    bool tensorCoresSupported = cudaHandles->majorComputeCapability >= 7;

    CUDNN_ERR(name.c_str(),cudnnCreateConvolutionDescriptor(&convolutionDescriptor));
    CUDNN_ERR(name.c_str(),cudnnSetConvolution2dDescriptor(
      convolutionDescriptor,
      paddingY,
      paddingX,
      yStride,
      xStride,
      dilationY,
      dilationX,
      CUDNN_CROSS_CORRELATION,
      (useFP16 && !tensorCoresSupported) ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT
    ));
    if(useFP16 && tensorCoresSupported)
      CUDNN_ERR(name.c_str(),cudnnSetConvolutionMathType(convolutionDescriptor, CUDNN_TENSOR_OP_MATH));

#if CUDNN_MAJOR >= 8
    convolutionAlgorithms = new ByBatchSize<cudnnConvolutionFwdAlgoPerf_t>(maxBatchSize);
#else
    convolutionAlgorithms = new ByBatchSize<cudnnConvolutionFwdAlgo_t>(maxBatchSize);
#endif

    for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
      if(useFP16 && dilationX <= 1 && dilationY <= 1) {
#if CUDNN_MAJOR >= 8
        (*convolutionAlgorithms)[batchSize].algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
#else
        (*convolutionAlgorithms)[batchSize] = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
#endif
      }
      else {
        const cudnnTensorDescriptor_t& inputDescriptor = inputDescriptors[batchSize];
        const cudnnTensorDescriptor_t& outputDescriptor = outputDescriptors[batchSize];

#if CUDNN_MAJOR >= 8
        int requestedAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
        int returnedAlgoCount = -1;
        cudnnConvolutionFwdAlgoPerf_t results[2 * CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
        CUDNN_ERR(name.c_str(),cudnnGetConvolutionForwardAlgorithm_v7(
          cudaHandles->cudnn,
          inputDescriptor,
          filterDescriptor,
          convolutionDescriptor,
          outputDescriptor,
          requestedAlgoCount,
          &returnedAlgoCount,
          results
        ));
        if(returnedAlgoCount <= 0)
          throw StringError("cudnnGetConvolutionForwardAlgorithm_v7 returned no algorithms?");
        (*convolutionAlgorithms)[batchSize] = results[0];
#else
        size_t bytesMemoryLimit = 0;
        CUDNN_ERR(name.c_str(),cudnnGetConvolutionForwardAlgorithm(
           cudaHandles->cudnn,
           inputDescriptor,
           filterDescriptor,
           convolutionDescriptor,
           outputDescriptor,
           CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
           bytesMemoryLimit,
           &((*convolutionAlgorithms)[batchSize])
         ));
#endif
      }
    }

    assert(desc->weights.size() == convYSize * convXSize * inChannels * outChannels);

    if(filterNHWC) {
      vector<float> weightsTransposed(desc->weights.size());
      for(int y = 0; y < convYSize; y++) {
        for(int x = 0; x < convXSize; x++) {
          for(int ic = 0; ic < inChannels; ic++) {
            for(int oc = 0; oc < outChannels; oc++) {
              weightsTransposed[((oc*convYSize + y)*convXSize + x)*inChannels + ic] =
                desc->weights[((oc*inChannels + ic)*convYSize + y)*convXSize + x];
            }
          }
        }
      }
      CudaUtils::mallocAndCopyToDevice(name,weightsTransposed,filterBuf,useFP16);
      CUDA_ERR(name.c_str(),cudaStreamSynchronize(cudaHandles->stream));
    }
    else
      CudaUtils::mallocAndCopyToDevice(name,desc->weights,filterBuf,useFP16);
  }

  ~ConvLayer() {
    cudaFree(filterBuf);
    cudnnDestroyFilterDescriptor(filterDescriptor);
    cudnnDestroyConvolutionDescriptor(convolutionDescriptor);
    delete convolutionAlgorithms;
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t workspaceBytes = 0;
#if CUDNN_MAJOR >= 8
    CUDNN_ERR(name.c_str(),cudnnGetConvolutionForwardWorkspaceSize(
      cudaHandles->cudnn,
      inputDescriptors[batchSize],
      filterDescriptor,
      convolutionDescriptor,
      outputDescriptors[batchSize],
      (*convolutionAlgorithms)[batchSize].algo,
      &workspaceBytes
    ));
#else
    CUDNN_ERR(name.c_str(),cudnnGetConvolutionForwardWorkspaceSize(
      cudaHandles->cudnn,
      inputDescriptors[batchSize],
      filterDescriptor,
      convolutionDescriptor,
      outputDescriptors[batchSize],
      (*convolutionAlgorithms)[batchSize],
      &workspaceBytes
    ));
#endif
    return workspaceBytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    int batchSize,
    bool accumulate,
    void* inputBuf,
    void* outputBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    const float alpha = 1.0f;
    const float beta = accumulate ? 1.0f : 0.0f;
#if CUDNN_MAJOR >= 8
    CUDNN_ERR(name.c_str(),cudnnConvolutionForward(
      cudaHandles->cudnn,
      &alpha,
      inputDescriptors[batchSize],
      inputBuf,
      filterDescriptor,
      filterBuf,
      convolutionDescriptor,
      (*convolutionAlgorithms)[batchSize].algo,
      workspaceBuf,
      workspaceBytes,
      &beta,
      outputDescriptors[batchSize],
      outputBuf
    ));
#else
    CUDNN_ERR(name.c_str(),cudnnConvolutionForward(
      cudaHandles->cudnn,
      &alpha,
      inputDescriptors[batchSize],
      inputBuf,
      filterDescriptor,
      filterBuf,
      convolutionDescriptor,
      (*convolutionAlgorithms)[batchSize],
      workspaceBuf,
      workspaceBytes,
      &beta,
      outputDescriptors[batchSize],
      outputBuf
    ));
#endif
  }

};


//---------------------------------------------------------------------------------

struct BatchNormLayer {
  const string name;
  const int numChannels;
  const float epsilon;
  const int activation;
  const int nnXLen;
  const int nnYLen;

  const bool usingFP16;
  const bool usingNHWC;

  void* meanBuf;
  void* varianceBuf;
  void* scaleBuf;
  void* biasBuf;

  void* mergedScaleBuf;
  void* mergedBiasBuf;

  BatchNormLayer() = delete;
  BatchNormLayer(const BatchNormLayer&) = delete;
  BatchNormLayer& operator=(const BatchNormLayer&) = delete;

  BatchNormLayer(
    CudaHandles* cudaHandles,
    const BatchNormLayerDesc* desc,
    const ActivationLayerDesc* actDesc,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ) :
    name(desc->name),
    numChannels(desc->numChannels),
    epsilon(desc->epsilon),
    activation(actDesc->activation),
    nnXLen(nnX),
    nnYLen(nnY),
    usingFP16(useFP16),
    usingNHWC(useNHWC)
  {
    (void)cudaHandles;

    assert(desc->mean.size() == numChannels);
    CudaUtils::mallocAndCopyToDevice(name,desc->mean,meanBuf,useFP16);

    assert(desc->variance.size() == numChannels);
    CudaUtils::mallocAndCopyToDevice(name,desc->variance,varianceBuf,useFP16);

    assert(desc->scale.size() == numChannels);
    CudaUtils::mallocAndCopyToDevice(name,desc->scale,scaleBuf,useFP16);

    assert(desc->bias.size() == numChannels);
    CudaUtils::mallocAndCopyToDevice(name,desc->bias,biasBuf,useFP16);

    vector<float> mergedScale(numChannels);
    vector<float> mergedBias(numChannels);
    for(int i = 0; i<numChannels; i++) {
      mergedScale[i] = desc->scale[i] / sqrt(desc->variance[i] + epsilon);
      mergedBias[i] = desc->bias[i] - mergedScale[i] * desc->mean[i];
    }
    CudaUtils::mallocAndCopyToDevice(name,mergedScale,mergedScaleBuf,useFP16);
    CudaUtils::mallocAndCopyToDevice(name,mergedBias,mergedBiasBuf,useFP16);
  }
  ~BatchNormLayer() {
    cudaFree(meanBuf);
    cudaFree(varianceBuf);
    cudaFree(scaleBuf);
    cudaFree(biasBuf);
    cudaFree(mergedScaleBuf);
    cudaFree(mergedBiasBuf);
  }

  void apply(
    CudaHandles* cudaHandles,
    int batchSize,
    void* inputBuf,
    const void* maskBuf, //ok to be null
    void* outputBuf
  ) const {
    (void)cudaHandles;
    if(!usingFP16) {
      if(!usingNHWC)
        customCudaApplyCScaleBiasNCHW((const float*)inputBuf,(float*)outputBuf,(const float*)mergedScaleBuf,(const float*)mergedBiasBuf,
                                      (const float*)maskBuf,
                                      batchSize,numChannels,nnXLen*nnYLen,activation,cudaHandles->stream);
      else
        customCudaApplyCScaleBiasNHWC((const float*)inputBuf,(float*)outputBuf,(const float*)mergedScaleBuf,(const float*)mergedBiasBuf,
                                      (const float*)maskBuf,
                                      batchSize,nnXLen*nnYLen,numChannels,activation,cudaHandles->stream);
    }
    else {
      if(!usingNHWC)
        customCudaApplyCScaleBiasNCHW((const half*)inputBuf,(half*)outputBuf,(const half*)mergedScaleBuf,(const half*)mergedBiasBuf,
                                      (const half*)maskBuf,
                                      batchSize,numChannels,nnXLen*nnYLen,activation,cudaHandles->stream);
      else
        customCudaApplyCScaleBiasNHWC((const half*)inputBuf,(half*)outputBuf,(const half*)mergedScaleBuf,(const half*)mergedBiasBuf,
                                      (const half*)maskBuf,
                                      batchSize,nnXLen*nnYLen,numChannels,activation,cudaHandles->stream);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }

  }

};


//---------------------------------------------------------------------------------

struct MatMulLayer {
  const string name;
  const int inChannels;
  const int outChannels;
  const bool usingFP16;
  void* matBuf;

  MatMulLayer() = delete;
  MatMulLayer(const MatMulLayer&) = delete;
  MatMulLayer& operator=(const MatMulLayer&) = delete;

  MatMulLayer(
    CudaHandles* cudaHandles,
    const MatMulLayerDesc* desc,
    bool useFP16
  ) :
    name(desc->name),
    inChannels(desc->inChannels),
    outChannels(desc->outChannels),
    usingFP16(useFP16)
  {
    (void)cudaHandles;

    assert(desc->weights.size() == inChannels * outChannels);
    CudaUtils::mallocAndCopyToDevice(name,desc->weights,matBuf,useFP16);
  }

  ~MatMulLayer() {
    cudaFree(matBuf);
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles
  ) const {
    (void)cudaHandles;
    size_t workspaceBytes = 0;
    return workspaceBytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* inputBuf,
    void* outputBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    (void)workspaceBuf;
    (void)workspaceBytes;

    if(!usingFP16) {
      const float alpha = 1.0f;
      const float beta = 0.0f;
      CUBLAS_ERR(name.c_str(),cublasSgemm(
        cudaHandles->cublas,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        outChannels,
        batchSize,
        inChannels,
        &alpha,
        (const float*)matBuf,outChannels,
        (const float*)inputBuf,inChannels,
        &beta,
        (float*)outputBuf,outChannels
      ));
    }
    else {
      const half* alpha = (const half*)scratch->oneBuf;
      const half* beta = (const half*)scratch->zeroBuf;
      CUBLAS_ERR(name.c_str(),cublasHgemm(
        cudaHandles->cublas,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        outChannels,
        batchSize,
        inChannels,
        alpha,
        (const half*)matBuf,outChannels,
        (const half*)inputBuf,inChannels,
        beta,
        (half*)outputBuf,outChannels
      ));
    }

  }

  bool applyPreparedResidual(
    CudaHandles* cudaHandles,
    CudaTransformerWinner::ResidualTactic tactic,
    void* preparedKernel,
    int matBatchSize,
    const void* inputBuf,
    void* residualBuf,
    const void* maskBuf,
    bool ffnDown,
    bool& usedSpecialized
  ) const {
    usedSpecialized = false;
    if(!usingFP16 || maskBuf != nullptr || tactic ==
       CudaTransformerWinner::ResidualTactic::GenericAdd)
      return false;
#if defined(KATAGO_ENABLE_RENJU15_GEMM_TACTICS_SM120) && KATAGO_ENABLE_RENJU15_GEMM_TACTICS_SM120
    const bool specializedSm120 =
      tactic == CudaTransformerWinner::ResidualTactic::Sm120M128N128K32S3Sw1 ||
      tactic == CudaTransformerWinner::ResidualTactic::Sm120C384M128N128K32S3Sw1;
    if(specializedSm120 &&
       katago_renju15_residual_gemm_sm120_supports(
         preparedKernel,matBatchSize,inChannels,outChannels,true,true)) {
      CUDA_ERR(name.c_str(),katago_renju15_residual_gemm_sm120_launch(
        preparedKernel,(const half*)inputBuf,(const half*)matBuf,
        (half*)residualBuf,cudaHandles->stream));
      usedSpecialized = true;
      bool& logged = ffnDown ? cudaHandles->loggedFfnDown :
        cudaHandles->loggedOutProjection;
      if(!logged && cudaHandles->logger != NULL) {
        const char* marker = katago_renju15_residual_gemm_sm120_active_marker(
          preparedKernel);
        cudaHandles->logger->write(
          string("RENJU15_SM120_RESIDUAL_GEMM_ACTIVE family=") +
          (ffnDown ? "ffn-down" : "out-proj") + " marker=" +
          (marker == nullptr ? "missing" : marker));
        logged = true;
      }
      return true;
    }
#endif
    (void)preparedKernel;
    // A specialized handle/shape miss is known before enqueue. Preserve the
    // generic no-mask FP16 fusion by degrading to cuBLAS beta=1 rather than
    // paying a separate projection and residual-add kernel.
    if(tactic != CudaTransformerWinner::ResidualTactic::CublasHgemmBetaOne &&
       tactic != CudaTransformerWinner::ResidualTactic::Sm120M128N128K32S3Sw1 &&
       tactic != CudaTransformerWinner::ResidualTactic::Sm120C384M128N128K32S3Sw1)
      return false;
    const half one = __float2half(1.0f);
    CUBLAS_ERR(name.c_str(),cublasHgemm(
      cudaHandles->cublas,CUBLAS_OP_N,CUBLAS_OP_N,
      outChannels,matBatchSize,inChannels,&one,
      (const half*)matBuf,outChannels,(const half*)inputBuf,inChannels,
      &one,(half*)residualBuf,outChannels));
    if(!cudaHandles->loggedCublasResidual && cudaHandles->logger != NULL) {
      cudaHandles->logger->write(
        "CUDA_TRANSFORMER_RESIDUAL_ACTIVE marker=cublas-hgemm-beta1");
      cudaHandles->loggedCublasResidual = true;
    }
    return true;
  }

};

//---------------------------------------------------------------------------------

struct MatBiasLayer {
  const string name;
  const int numChannels;
  const bool usingFP16;
  const int activation;

  void* biasBuf;

  MatBiasLayer() = delete;
  MatBiasLayer(const MatBiasLayer&) = delete;
  MatBiasLayer& operator=(const MatBiasLayer&) = delete;

  MatBiasLayer(
    CudaHandles* cudaHandles,
    const MatBiasLayerDesc* desc,
    bool useFP16,
    int activation_
  ) :
    name(desc->name),
    numChannels(desc->numChannels),
    usingFP16(useFP16),
    activation(activation_)
  {
    (void)cudaHandles;
    assert(desc->weights.size() == numChannels);
    CudaUtils::mallocAndCopyToDevice(name,desc->weights,biasBuf,useFP16);
  }

  ~MatBiasLayer() {
    cudaFree(biasBuf);
  }

  void apply(
    CudaHandles* cudaHandles,
    int batchSize,
    void* matBuf
  ) const {
    if(!usingFP16) {
      customCudaAddCBiasInplaceNC((float*)matBuf,(const float*)biasBuf,batchSize,numChannels,activation,cudaHandles->stream);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }
    else {
      customCudaAddCBiasInplaceNC((half*)matBuf,(const half*)biasBuf,batchSize,numChannels,activation,cudaHandles->stream);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }
  }

};

//---------------------------------------------------------------------------------

struct NormActConv {
  const BatchNormLayer norm;
  const ConvLayer conv;

  const int inChannels;
  const int outChannels;
  const int nnXLen;
  const int nnYLen;
  const bool usingFP16;
  const bool usingNHWC;

  NormActConv() = delete;
  NormActConv(const NormActConv&) = delete;
  NormActConv& operator=(const NormActConv&) = delete;

  NormActConv(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const BatchNormLayerDesc* normDesc,
    const ActivationLayerDesc* actDesc,
    const ConvLayerDesc* convDesc,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ): norm(cudaHandles,normDesc,actDesc,nnX,nnY,useFP16,useNHWC),
     conv(cudaHandles,manager,convDesc,useFP16,useNHWC),
     inChannels(norm.numChannels),
     outChannels(conv.outChannels),
     nnXLen(nnX),
     nnYLen(nnY),
     usingFP16(useFP16),
     usingNHWC(useNHWC)
  {
    assert(norm.numChannels == conv.inChannels);
  }

  ~NormActConv()
  {}

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;
    b = conv.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    int batchSize,
    bool accumulate,
    void* inBuf,
    void* inScratchBuf,
    void* outBuf,
    void* maskBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    norm.apply(cudaHandles,batchSize,inBuf,maskBuf,inScratchBuf);
#ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint3D(string("AFTER NORM "), inScratchBuf, batchSize, inChannels, nnXLen*nnYLen, usingNHWC, usingFP16);
#endif
    conv.apply(cudaHandles,batchSize,accumulate,inScratchBuf,outBuf,workspaceBuf,workspaceBytes);
  }

};


//---------------------------------------------------------------------------------

struct ResidualBlock {
  const string name;
  const NormActConv normActConv1;
  const NormActConv normActConv2;

  ResidualBlock() = delete;
  ResidualBlock(const ResidualBlock&) = delete;
  ResidualBlock& operator=(const ResidualBlock&) = delete;

  ResidualBlock(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const ResidualBlockDesc* desc,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ): name(desc->name),
     normActConv1(cudaHandles,manager,&desc->preBN,&desc->preActivation,&desc->regularConv,nnX,nnY,useFP16,useNHWC),
     normActConv2(cudaHandles,manager,&desc->midBN,&desc->midActivation,&desc->finalConv,nnX,nnY,useFP16,useNHWC)
  {
  }

  ~ResidualBlock()
  {}

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;
    b = normActConv1.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = normActConv2.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* maskBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    SizedBuf<void*> midIn(scratch->allocator, scratch->getBufSizeXY(normActConv1.outChannels));
    SizedBuf<void*> midScratch(scratch->allocator, scratch->getBufSizeXY(normActConv1.outChannels));
    normActConv1.apply(cudaHandles,batchSize,false,trunkBuf,trunkScratchBuf,midIn.buf,maskBuf,workspaceBuf,workspaceBytes);
    normActConv2.apply(cudaHandles,batchSize,true,midIn.buf,midScratch.buf,trunkBuf,maskBuf,workspaceBuf,workspaceBytes);
  }

};


//----------------------------------------------------------------------------


struct GlobalPoolingResidualBlock {
  const string name;
  const BatchNormLayer preBN;
  const ConvLayer regularConv;
  const ConvLayer gpoolConv;
  const BatchNormLayer gpoolBN;
  const MatMulLayer gpoolToBiasMul;
  const NormActConv normActConv2;

  const int nnXLen;
  const int nnYLen;
  const int regularChannels;
  const int gpoolChannels;
  const bool usingFP16;
  const bool usingNHWC;

  GlobalPoolingResidualBlock() = delete;
  GlobalPoolingResidualBlock(const GlobalPoolingResidualBlock&) = delete;
  GlobalPoolingResidualBlock& operator=(const GlobalPoolingResidualBlock&) = delete;

  GlobalPoolingResidualBlock(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const GlobalPoolingResidualBlockDesc* desc,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ): name(desc->name),
     preBN(cudaHandles,&desc->preBN,&desc->preActivation,nnX,nnY,useFP16,useNHWC),
     regularConv(cudaHandles,manager,&desc->regularConv,useFP16,useNHWC),
     gpoolConv(cudaHandles,manager,&desc->gpoolConv,useFP16,useNHWC),
     gpoolBN(cudaHandles,&desc->gpoolBN,&desc->gpoolActivation,nnX,nnY,useFP16,useNHWC),
     gpoolToBiasMul(cudaHandles,&desc->gpoolToBiasMul,useFP16),
     normActConv2(cudaHandles,manager,&desc->midBN,&desc->midActivation,&desc->finalConv,nnX,nnY,useFP16,useNHWC),
     nnXLen(nnX),
     nnYLen(nnY),
     regularChannels(desc->regularConv.outChannels),
     gpoolChannels(desc->gpoolConv.outChannels),
     usingFP16(useFP16),
     usingNHWC(useNHWC)
  {
  }

  ~GlobalPoolingResidualBlock() {
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;
    b = regularConv.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = gpoolConv.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = gpoolToBiasMul.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = normActConv2.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = sizeof(float)*batchSize*gpoolChannels*nnXLen*nnYLen;
    bytes = std::max(bytes,b);
    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* maskBuf,
    float* maskSumBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    SizedBuf<void*> regularOut(scratch->allocator, scratch->getBufSizeXY(regularChannels));
    SizedBuf<void*> regularScratch(scratch->allocator, scratch->getBufSizeXY(regularChannels));
    SizedBuf<void*> gpoolOut(scratch->allocator, scratch->getBufSizeXY(gpoolChannels));
    SizedBuf<void*> gpoolOut2(scratch->allocator, scratch->getBufSizeXY(gpoolChannels));
    SizedBuf<void*> gpoolConcat(scratch->allocator, scratch->getBufSize(gpoolChannels*3));
    SizedBuf<void*> gpoolBias(scratch->allocator, scratch->getBufSize(regularChannels));

    preBN.apply(cudaHandles,batchSize,trunkBuf,maskBuf,trunkScratchBuf);
    regularConv.apply(cudaHandles,batchSize,false,trunkScratchBuf,regularOut.buf,workspaceBuf,workspaceBytes);
    gpoolConv.apply(cudaHandles,batchSize,false,trunkScratchBuf,gpoolOut.buf,workspaceBuf,workspaceBytes);
    gpoolBN.apply(cudaHandles,batchSize,gpoolOut.buf,maskBuf,gpoolOut2.buf);

    if(!usingFP16) {
      if(!usingNHWC)
        customCudaPoolRowsGPoolNCHW((const float*)gpoolOut2.buf,(float*)gpoolConcat.buf,batchSize,gpoolChannels,nnXLen*nnYLen,(const float*)maskBuf,maskSumBuf,cudaHandles->stream);
      else
        customCudaPoolRowsGPoolNHWC((const float*)gpoolOut2.buf,(float*)gpoolConcat.buf,batchSize,nnXLen*nnYLen,gpoolChannels,(const float*)maskBuf,maskSumBuf,cudaHandles->stream);
    }
    else {
      if(!usingNHWC)
        customCudaPoolRowsGPoolNCHW((const half*)gpoolOut2.buf,(half*)gpoolConcat.buf,batchSize,gpoolChannels,nnXLen*nnYLen,(const half*)maskBuf,maskSumBuf,cudaHandles->stream);
      else
        customCudaPoolRowsGPoolNHWC((const half*)gpoolOut2.buf,(half*)gpoolConcat.buf,batchSize,nnXLen*nnYLen,gpoolChannels,(const half*)maskBuf,maskSumBuf,cudaHandles->stream);
    }
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());

    gpoolToBiasMul.apply(cudaHandles,scratch,batchSize,gpoolConcat.buf,gpoolBias.buf,workspaceBuf,workspaceBytes);

    if(!usingFP16) {
      if(!usingNHWC)
        customCudaAddNCBiasInplaceNCHW((float*)regularOut.buf,(const float*)gpoolBias.buf,batchSize,regularChannels,nnXLen*nnYLen,cudaHandles->stream);
      else
        customCudaAddNCBiasInplaceNHWC((float*)regularOut.buf,(const float*)gpoolBias.buf,batchSize,nnXLen*nnYLen,regularChannels,cudaHandles->stream);
    }
    else {
      if(!usingNHWC)
        customCudaAddNCBiasInplaceNCHW((half*)regularOut.buf,(const half*)gpoolBias.buf,batchSize,regularChannels,nnXLen*nnYLen,cudaHandles->stream);
      else
        customCudaAddNCBiasInplaceNHWC((half*)regularOut.buf,(const half*)gpoolBias.buf,batchSize,nnXLen*nnYLen,regularChannels,cudaHandles->stream);
    }
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());

    normActConv2.apply(cudaHandles,batchSize,true,regularOut.buf,regularScratch.buf,trunkBuf,maskBuf,workspaceBuf,workspaceBytes);
  }

};

//------------------------------------------------------------------------------

struct TransformerPlanCursor {
  const CudaTransformerWinner::PreparedPlan* plan;
  size_t nextRecord;

  explicit TransformerPlanCursor(
    const CudaTransformerWinner::PreparedPlan* preparedPlan
  ) : plan(preparedPlan), nextRecord(0) {}

  uint32_t nextTopologyIndex(NeuralNetArchitecture::ArchitectureOpKind expected) {
    if(plan == nullptr)
      return UINT32_MAX;
    while(nextRecord < plan->records.size()) {
      const CudaTransformerWinner::PreparedRecord& record =
        plan->records[nextRecord++];
      const auto kind = record.request.key.kind;
      if(kind != NeuralNetArchitecture::ArchitectureOpKind::TransformerAttention &&
         kind != NeuralNetArchitecture::ArchitectureOpKind::TransformerFFN)
        continue;
      if(kind != expected)
        throw StringError("Prepared CUDA transformer plan does not match block topology");
      return record.request.topologyIndex;
    }
    throw StringError("Prepared CUDA transformer plan ended before block topology");
  }

  CudaTransformerWinner::AttentionRecipe nextAttention(
    uint32_t& topologyIndex
  ) {
    topologyIndex = nextTopologyIndex(
      NeuralNetArchitecture::ArchitectureOpKind::TransformerAttention);
    return plan == nullptr ? CudaTransformerWinner::AttentionRecipe{} :
      plan->attentionFor(topologyIndex);
  }

  CudaTransformerWinner::FfnRecipe nextFfn(
    uint32_t& topologyIndex
  ) {
    topologyIndex = nextTopologyIndex(
      NeuralNetArchitecture::ArchitectureOpKind::TransformerFFN);
    return plan == nullptr ? CudaTransformerWinner::FfnRecipe{} :
      plan->ffnFor(topologyIndex);
  }
};

struct BlockStack {
  const int numBlocks;
  const int trunkNumChannels;
  const int nnXLen;
  const int nnYLen;
  const bool usingFP16;
  const bool usingNHWC;
  vector<pair<int,unique_ptr_void>> blocks;

  BlockStack() = delete;
  BlockStack(const BlockStack&) = delete;
  BlockStack& operator=(const BlockStack&) = delete;

  BlockStack(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    int nBlocks,
    int trunkChannels,
    const std::vector<std::pair<int, unique_ptr_void>>& descBlocks,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC,
    TransformerPlanCursor* planCursor
  );
  ~BlockStack();

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const;

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* maskBuf,
    float* maskSumBuf,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const;

};

//------------------------------------------------------------------------------

struct NestedBottleneckResidualBlock {
  const string name;
  const NormActConv normActConv1;
  const BlockStack blocks;
  const NormActConv normActConv2;

  NestedBottleneckResidualBlock() = delete;
  NestedBottleneckResidualBlock(const NestedBottleneckResidualBlock&) = delete;
  NestedBottleneckResidualBlock& operator=(const NestedBottleneckResidualBlock&) = delete;

  NestedBottleneckResidualBlock(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const NestedBottleneckResidualBlockDesc* desc,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC,
    TransformerPlanCursor* planCursor
  ): name(desc->name),
     normActConv1(cudaHandles,manager,&desc->preBN,&desc->preActivation,&desc->preConv,nnX,nnY,useFP16,useNHWC),
     blocks(cudaHandles,manager,desc->numBlocks,desc->preConv.outChannels,desc->blocks,nnX,nnY,useFP16,useNHWC,planCursor),
     normActConv2(cudaHandles,manager,&desc->postBN,&desc->postActivation,&desc->postConv,nnX,nnY,useFP16,useNHWC)
  {
  }

  ~NestedBottleneckResidualBlock()
  {}

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;
    b = normActConv1.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = blocks.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = normActConv2.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* maskBuf,
    float* maskSumBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    SizedBuf<void*> mid(scratch->allocator, scratch->getBufSizeXY(normActConv1.outChannels));
    SizedBuf<void*> midScratch(scratch->allocator, scratch->getBufSizeXY(normActConv1.outChannels));
    assert(normActConv1.outChannels == normActConv2.inChannels);
    normActConv1.apply(cudaHandles,batchSize,false,trunkBuf,trunkScratchBuf,mid.buf,maskBuf,workspaceBuf,workspaceBytes);
    blocks.apply(
      cudaHandles,
      scratch,
      batchSize,
      maskBuf,
      maskSumBuf,
      mid.buf,
      midScratch.buf,
      workspaceBuf,
      workspaceBytes
    );
    normActConv2.apply(cudaHandles,batchSize,true,mid.buf,midScratch.buf,trunkBuf,maskBuf,workspaceBuf,workspaceBytes);
  }

};

//------------------------------------------------------------------------------

struct TransformerRMSNormLayer {
  const string name;
  const int numChannels;
  const float epsilon;
  const bool usingFP16;
  const CudaTransformerWinner::RmsNormTactic tactic;
  mutable bool countedWinnerRms;
  void* weightBuf;
  void* zeroBetaBuf;

  TransformerRMSNormLayer() = delete;
  TransformerRMSNormLayer(const TransformerRMSNormLayer&) = delete;
  TransformerRMSNormLayer& operator=(const TransformerRMSNormLayer&) = delete;

  TransformerRMSNormLayer(
    CudaHandles* cudaHandles,
    const TransformerRMSNormDesc* desc,
    bool useFP16,
    CudaTransformerWinner::RmsNormTactic selectedTactic
  ) :
    name(desc->name),
    numChannels(desc->numChannels),
    epsilon(desc->epsilon),
    usingFP16(useFP16),
    tactic(selectedTactic),
    countedWinnerRms(false)
  {
    (void)cudaHandles;
    if((int)desc->weight.size() != numChannels)
      throw StringError(name + ": RMSNorm weight count does not match numChannels");
    CudaUtils::mallocAndCopyToDevice(name, desc->weight, weightBuf, useFP16);
    // Allocate a zero buffer for beta (TransformerRMSNorm has no bias)
    vector<float> zeros(numChannels, 0.0f);
    CudaUtils::mallocAndCopyToDevice(name + ":zeroBeta", zeros, zeroBetaBuf, useFP16);
  }

  ~TransformerRMSNormLayer() {
    cudaFree(weightBuf);
    cudaFree(zeroBetaBuf);
  }

#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
  bool canApplyFp16Int8(const void* maskBuf) const {
    return usingFP16 && numChannels == 256 && maskBuf == nullptr;
  }

  // The caller performs every capability check before invoking this method.
  // Once selected, a launch error is fatal; inference never replays the block
  // through FP16 after potentially enqueueing INT8 work.
  void applyFp16Int8(
    CudaHandles* cudaHandles,
    int batchSize,
    int xySize,
    const void* inputBuf,
    void* outputFp16,
    void* outputInt8,
    const void* maskBuf
  ) const {
    if(!canApplyFp16Int8(maskBuf))
      throw StringError(name + ": incompatible experimental FP16+INT8 RMSNorm");
    CUDA_ERR(name.c_str(),Renju15Sm120::launchRmsNorm256Fp16Int8(
      (const half*)inputBuf,(half*)outputFp16,(int8_t*)outputInt8,
      (const half*)weightBuf,batchSize * xySize,epsilon,
      Renju15Sm120::RmsNorm256Tactic::Warp4Vec8,cudaHandles->stream));
    if(!cudaHandles->loggedRms && cudaHandles->logger != NULL) {
      cudaHandles->logger->write(
        "RENJU15_SM120_RMS_ACTIVE marker=warp4-vec8-fp16-int8-clip4");
      cudaHandles->loggedRms = true;
    }
    // This arithmetic recipe is tracked by the independent INT8 experiment
    // counters; do not increment the certified FP16 RMS winner counter.
  }
#endif

  // Apply RMSNorm on NHWC data [N, XY, C], applying mask [N, XY] to zero padded positions.
  // Uses the RMSNormGammaBeta kernel with gamma=weight, beta=0, no activation.
  void apply(
    CudaHandles* cudaHandles,
    int batchSize,
    int xySize,
    void* inputBuf,
    void* outputBuf,
    const void* maskBuf
  ) const {
#if defined(KATAGO_ENABLE_RENJU15_RMS_SM120) && KATAGO_ENABLE_RENJU15_RMS_SM120
    if(tactic == CudaTransformerWinner::RmsNormTactic::Sm120C256Warp4Vec8 &&
       usingFP16 && numChannels == 256 && maskBuf == nullptr) {
      CUDA_ERR(name.c_str(),Renju15Sm120::launchRmsNorm256(
        (const half*)inputBuf,(half*)outputBuf,(const half*)weightBuf,
        batchSize * xySize,epsilon,Renju15Sm120::RmsNorm256Tactic::Warp4Vec8,
        cudaHandles->stream));
      if(!cudaHandles->loggedRms && cudaHandles->logger != NULL) {
        cudaHandles->logger->write(
          "RENJU15_SM120_RMS_ACTIVE marker=warp4-vec8");
        cudaHandles->loggedRms = true;
      }
      cudaHandles->noteWinnerLaunch(
        countedWinnerRms,cudaHandles->activeWinnerRms);
      return;
    }
    if(tactic == CudaTransformerWinner::RmsNormTactic::Sm120C384Warp4Vec4x3 &&
       usingFP16 && numChannels == 384 && maskBuf == nullptr) {
      CUDA_ERR(name.c_str(),Renju15Sm120::launchRmsNorm384(
        (const half*)inputBuf,(half*)outputBuf,(const half*)weightBuf,
        batchSize * xySize,epsilon,Renju15Sm120::RmsNorm384Tactic::Warp4Vec4x3,
        cudaHandles->stream));
      if(!cudaHandles->loggedRms && cudaHandles->logger != NULL) {
        cudaHandles->logger->write(
          "KATAGO_C384_SM120_RMS_ACTIVE marker=warp4-vec4x3");
        cudaHandles->loggedRms = true;
      }
      return;
    }
#endif
    // RMSNormGammaBetaNHWC with gamma=weight, beta=zero, mask, identity activation.
    if(!usingFP16) {
      customCudaRMSNormGammaBetaNHWC(
        (const float*)inputBuf, (float*)outputBuf,
        (const float*)weightBuf, (const float*)zeroBetaBuf,
        (const float*)maskBuf,
        batchSize, xySize, numChannels, epsilon, ACTIVATION_IDENTITY,cudaHandles->stream);
    }
    else {
      customCudaRMSNormGammaBetaNHWC(
        (const half*)inputBuf, (half*)outputBuf,
        (const half*)weightBuf, (const half*)zeroBetaBuf,
        (const half*)maskBuf,
        batchSize, xySize, numChannels, epsilon, ACTIVATION_IDENTITY,cudaHandles->stream);
    }
    CUDA_ERR(name.c_str(), cudaPeekAtLastError());
  }
};

//------------------------------------------------------------------------------

#if 0
// Gom2026 v101/v102/v103 uses BatchNorm at the trunk tip. Keep the unrelated
// newer trunk-RMSNorm implementation out of this minimal transformer port.
struct RMSNormLayer {
  const string name;
  const int numChannels;
  const bool spatial;
  const int activation;
  const float epsilon;
  const int nnXLen;
  const int nnYLen;
  const bool usingFP16;
  const bool usingNHWC;

  void* gammaBuf;
  void* betaBuf;

  RMSNormLayer() = delete;
  RMSNormLayer(const RMSNormLayer&) = delete;
  RMSNormLayer& operator=(const RMSNormLayer&) = delete;

  RMSNormLayer(
    CudaHandles* cudaHandles,
    const RMSNormLayerDesc* desc,
    int act,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ) :
    name(desc->name),
    numChannels(desc->numChannels),
    spatial(desc->spatial),
    activation(act),
    epsilon(desc->epsilon),
    nnXLen(nnX),
    nnYLen(nnY),
    usingFP16(useFP16),
    usingNHWC(useNHWC)
  {
    (void)cudaHandles;
    testAssert((int)desc->gamma.size() == numChannels);
    testAssert((int)desc->beta.size() == numChannels);
    CudaUtils::mallocAndCopyToDevice(name, desc->gamma, gammaBuf, useFP16);
    CudaUtils::mallocAndCopyToDevice(name, desc->beta, betaBuf, useFP16);
  }

  ~RMSNormLayer() {
    cudaFree(gammaBuf);
    cudaFree(betaBuf);
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* inputBuf,
    void* outputBuf,
    const void* maskBuf,
    const float* maskSumBuf
  ) const {
    int xySize = nnXLen * nnYLen;
    if(!spatial) {
      if(!usingFP16) {
        if(!usingNHWC)
          customCudaRMSNormGammaBetaNCHW(
            (const float*)inputBuf, (float*)outputBuf, (const float*)gammaBuf, (const float*)betaBuf,
            (const float*)maskBuf, batchSize, numChannels, xySize, epsilon, activation,cudaHandles->stream);
        else
          customCudaRMSNormGammaBetaNHWC(
            (const float*)inputBuf, (float*)outputBuf, (const float*)gammaBuf, (const float*)betaBuf,
            (const float*)maskBuf, batchSize, xySize, numChannels, epsilon, activation,cudaHandles->stream);
      }
      else {
        if(!usingNHWC)
          customCudaRMSNormGammaBetaNCHW(
            (const half*)inputBuf, (half*)outputBuf, (const half*)gammaBuf, (const half*)betaBuf,
            (const half*)maskBuf, batchSize, numChannels, xySize, epsilon, activation,cudaHandles->stream);
        else
          customCudaRMSNormGammaBetaNHWC(
            (const half*)inputBuf, (half*)outputBuf, (const half*)gammaBuf, (const half*)betaBuf,
            (const half*)maskBuf, batchSize, xySize, numChannels, epsilon, activation,cudaHandles->stream);
      }
    }
    else {
      // Allocate temp buffer for spatial reduction from scratch (float regardless of FP16 mode).
      // Holds per-block partial sums plus the final reduced value per batch element; see
      // SPATIAL_RMSNORM_BLOCKS_PER_BATCH in cudahelpers.cu (partialStride = that + 1).
      SizedBuf<void*> sumSqBuf(scratch->allocator, (size_t)batchSize * CUDA_SPATIAL_RMSNORM_SUMSQ_STRIDE * sizeof(float));
      if(!usingFP16) {
        if(!usingNHWC)
          customCudaSpatialRMSNormNCHW(
            (const float*)inputBuf, (float*)outputBuf, (const float*)gammaBuf, (const float*)betaBuf,
            (const float*)maskBuf, maskSumBuf, batchSize, numChannels, xySize, epsilon, activation, (float*)sumSqBuf.buf,cudaHandles->stream);
        else
          customCudaSpatialRMSNormNHWC(
            (const float*)inputBuf, (float*)outputBuf, (const float*)gammaBuf, (const float*)betaBuf,
            (const float*)maskBuf, maskSumBuf, batchSize, xySize, numChannels, epsilon, activation, (float*)sumSqBuf.buf,cudaHandles->stream);
      }
      else {
        if(!usingNHWC)
          customCudaSpatialRMSNormNCHW(
            (const half*)inputBuf, (half*)outputBuf, (const half*)gammaBuf, (const half*)betaBuf,
            (const half*)maskBuf, maskSumBuf, batchSize, numChannels, xySize, epsilon, activation, (float*)sumSqBuf.buf,cudaHandles->stream);
        else
          customCudaSpatialRMSNormNHWC(
            (const half*)inputBuf, (half*)outputBuf, (const half*)gammaBuf, (const half*)betaBuf,
            (const half*)maskBuf, maskSumBuf, batchSize, xySize, numChannels, epsilon, activation, (float*)sumSqBuf.buf,cudaHandles->stream);
      }
    }
    CUDA_ERR(name.c_str(), cudaPeekAtLastError());
  }
};

//------------------------------------------------------------------------------
#endif

struct TransformerAttentionBlock {
  const string name;
  const int numHeads;
  const int numKVHeads;
  const int qHeadDim;
  const int vHeadDim;
  const bool useRope;
  const bool learnableRope;
  const int inChannels;

  const int nnXLen;
  const int nnYLen;
  const bool usingFP16;
  const bool usingNHWC;

  const TransformerRMSNormLayer preLN;
  const MatMulLayer qProj;
  const MatMulLayer kProj;
  const MatMulLayer vProj;
  const MatMulLayer outProj;

  const CudaTransformerWinner::AttentionRecipe recipe;
  std::unique_ptr<CudaQKVPlanar::Projection> qkvPlanarProjection;
  void* outProjectionKernel;
#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
  UniqueCudaDeviceBuffer int8QkWeightBuf;
  UniqueInt8QkKernel int8QkKernel;
  mutable bool countedInt8Qk;
#endif
  mutable bool countedWinnerQkvRope;
  mutable bool countedWinnerFa4;
  mutable bool countedWinnerOutProjection;

  // Precomputed RoPE cos/sin tables on device
  void* ropeCosTable;
  void* ropeSinTable;
  void* ropeCosSinTable;
  int ropeNumPairs;

  TransformerAttentionBlock() = delete;
  TransformerAttentionBlock(const TransformerAttentionBlock&) = delete;
  TransformerAttentionBlock& operator=(const TransformerAttentionBlock&) = delete;

#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
  void discardInt8Prepared() noexcept {
    int8QkKernel.reset();
    int8QkWeightBuf.reset();
  }
#endif

  void destroyRawPreparedState() noexcept {
#if defined(KATAGO_ENABLE_RENJU15_GEMM_TACTICS_SM120) && KATAGO_ENABLE_RENJU15_GEMM_TACTICS_SM120
    katago_renju15_residual_gemm_sm120_destroy(outProjectionKernel);
#endif
    outProjectionKernel = nullptr;
    if(ropeCosTable != NULL) (void)cudaFree(ropeCosTable);
    if(ropeSinTable != NULL) (void)cudaFree(ropeSinTable);
    if(ropeCosSinTable != NULL) (void)cudaFree(ropeCosSinTable);
    ropeCosTable = NULL;
    ropeSinTable = NULL;
    ropeCosSinTable = NULL;
  }

  TransformerAttentionBlock(
    CudaHandles* cudaHandles,
    const TransformerAttentionDesc* desc,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC,
    const CudaTransformerWinner::AttentionRecipe& selectedRecipe,
    int fixedBatchSize,
    uint32_t topologyIndex
  ) :
    name(desc->name),
    numHeads(desc->numHeads),
    numKVHeads(desc->numKVHeads),
    qHeadDim(desc->qHeadDim),
    vHeadDim(desc->vHeadDim),
    useRope(desc->useRope),
    learnableRope(desc->learnableRope),
    inChannels(desc->qProj.inChannels),
    nnXLen(nnX),
    nnYLen(nnY),
    usingFP16(useFP16),
    usingNHWC(useNHWC),
    preLN(cudaHandles, &desc->preLN, useFP16, selectedRecipe.rmsNorm),
    qProj(cudaHandles, &desc->qProj, useFP16),
    kProj(cudaHandles, &desc->kProj, useFP16),
    vProj(cudaHandles, &desc->vProj, useFP16),
    outProj(cudaHandles, &desc->outProj, useFP16),
    recipe(selectedRecipe),
    qkvPlanarProjection(),
    outProjectionKernel(nullptr),
#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
    int8QkWeightBuf(nullptr),
    int8QkKernel(nullptr),
    countedInt8Qk(false),
#endif
    countedWinnerQkvRope(false),
    countedWinnerFa4(false),
    countedWinnerOutProjection(false),
    ropeCosTable(NULL),
    ropeSinTable(NULL),
    ropeCosSinTable(NULL),
    ropeNumPairs(0)
  {
    if(!useNHWC) {
      throw StringError("Transformer blocks with NCHW layout are not yet supported by the CUDA backend");
    }
    const bool squareQkv =
      desc->qProj.inChannels > 0 &&
      desc->qProj.inChannels == desc->qProj.outChannels &&
      desc->kProj.inChannels == desc->qProj.inChannels &&
      desc->kProj.outChannels == desc->qProj.outChannels &&
      desc->vProj.inChannels == desc->qProj.inChannels &&
      desc->vProj.outChannels == desc->qProj.outChannels;
    if(recipe.planarQkv ==
         CudaTransformerWinner::PlanarQkvTactic::CublasHgemmStridedBatchedSquare &&
       useFP16 && useNHWC && squareQkv) {
      qkvPlanarProjection = std::make_unique<CudaQKVPlanar::Projection>(
        desc->qProj.weights,desc->kProj.weights,desc->vProj.weights,
        desc->qProj.inChannels,desc->qProj.outChannels,recipe,fixedBatchSize);
    }
    if(useRope) {
      try {
      ropeNumPairs = qHeadDim / 2;
      int seqLen = nnXLen * nnYLen;
      vector<float> cosTableData;
      vector<float> sinTableData;
      desc->computeRopeCosSin(nnXLen, nnYLen, seqLen, cosTableData, sinTableData);
      CudaUtils::mallocAndCopyToDevice(name + ":ropeCos", cosTableData.data(), (int)cosTableData.size(), ropeCosTable, useFP16);
      CudaUtils::mallocAndCopyToDevice(name + ":ropeSin", sinTableData.data(), (int)sinTableData.size(), ropeSinTable, useFP16);
      if(recipe.rope == CudaTransformerWinner::RopeTactic::LearnedHalf2 &&
         useFP16 && learnableRope) {
        const int totalKVPairs = numKVHeads * ropeNumPairs;
        vector<float> cosSinTableData((size_t)seqLen * totalKVPairs * 2);
        for(int xy = 0; xy < seqLen; xy++) {
          for(int hp = 0; hp < totalKVPairs; hp++) {
            const size_t source = (size_t)hp * seqLen + xy;
            const size_t destination = ((size_t)xy * totalKVPairs + hp) * 2;
            cosSinTableData[destination] = cosTableData[source];
            cosSinTableData[destination + 1] = sinTableData[source];
          }
        }
        CudaUtils::mallocAndCopyToDevice(
          name + ":ropeCosSinHalf2",cosSinTableData.data(),
          (int)cosSinTableData.size(),ropeCosSinTable,true);
      }
      }
      catch(...) {
        destroyRawPreparedState();
        throw;
      }
    }
    try {
#if defined(KATAGO_ENABLE_RENJU15_GEMM_TACTICS_SM120) && KATAGO_ENABLE_RENJU15_GEMM_TACTICS_SM120
    // Raw prepared state is created last so any throwing weight/table copy
    // above cannot bypass its destructor during partial construction.
    if(recipe.outProjection ==
       CudaTransformerWinner::ResidualTactic::Sm120M128N128K32S3Sw1) {
      outProjectionKernel = katago_renju15_residual_gemm_sm120_create(
        KATAGO_RENJU15_RESIDUAL_GEMM_OUT_PROJ,
        KATAGO_RENJU15_RESIDUAL_GEMM_M128_N128_K32_S3,
        fixedBatchSize * nnXLen * nnYLen);
    }
    else if(recipe.outProjection ==
       CudaTransformerWinner::ResidualTactic::Sm120C384M128N128K32S3Sw1) {
      outProjectionKernel = katago_renju15_residual_gemm_sm120_create(
        KATAGO_RENJU15_RESIDUAL_GEMM_C384_OUT_PROJ,
        KATAGO_RENJU15_RESIDUAL_GEMM_C384_M128_N128_K32_S3,
        fixedBatchSize * nnXLen * nnYLen);
    }
#endif
    if(cudaHandles->exactWinnerPlan) {
#if defined(KATAGO_ENABLE_RENJU15_QKV_ROPE_GEMM_SM120) && KATAGO_ENABLE_RENJU15_QKV_ROPE_GEMM_SM120
      if(recipe.qkvRope != CudaTransformerWinner::QkvRopeTactic::Disabled) {
        if(qkvPlanarProjection == nullptr ||
           !qkvPlanarProjection->hasFusedQKVRoPEState())
          throw StringError("Certified CUDA transformer QKV-RoPE handle preparation failed");
        cudaHandles->preparedWinnerQkvRope++;
      }
#endif
      if(recipe.outProjection ==
         CudaTransformerWinner::ResidualTactic::Sm120M128N128K32S3Sw1) {
        if(outProjectionKernel == nullptr)
          throw StringError("Certified CUDA transformer out-projection handle preparation failed");
        cudaHandles->preparedWinnerOutProjection++;
      }
    }
#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
    const bool int8ShapeEligible = cudaHandles->int8ExperimentPlan &&
      nnXLen == 15 && nnYLen == 15 && useFP16 && useNHWC && useRope &&
      learnableRope && ropeCosSinTable != nullptr && numHeads == 8 &&
      numKVHeads == 8 && qHeadDim == 32 && vHeadDim == 32 &&
      desc->qProj.inChannels == 256 && desc->qProj.outChannels == 256 &&
      desc->kProj.inChannels == 256 && desc->kProj.outChannels == 256;
    const NativeInt8Quant::Entry* embeddedQkEntry = nullptr;
    if(int8ShapeEligible && cudaHandles->usesEmbeddedInt8Weights()) {
      // Metadata mismatch is a model-format failure, not an optional CUDA
      // preparation miss. Keep this lookup outside the fallback transaction.
      embeddedQkEntry = &cudaHandles->requireEmbeddedInt8Entry(
        topologyIndex,NativeInt8Quant::Role::QK,
        vector<string>{desc->qProj.name,desc->kProj.name},256,512);
    }
    if(int8ShapeEligible) {
      try {
        float qkScale = 0.0f;
        void* packedQkWeights = nullptr;
        if(embeddedQkEntry != nullptr)
          prepareEmbeddedPackedInt8(
            name + ":int8Qk",*embeddedQkEntry,packedQkWeights,qkScale);
        else
          preparePackedInt8Qk(
            name + ":int8Qk",desc->qProj,desc->kProj,
            packedQkWeights,qkScale);
        int8QkWeightBuf.reset(packedQkWeights);
        int8QkKernel.reset(katago_renju15_int8_qk_sm120_create(
          fixedBatchSize * nnXLen * nnYLen,
          (const int8_t*)int8QkWeightBuf.get(),qkScale));
        if(int8QkKernel == nullptr)
          throw StringError(name + ": INT8 QK handle preparation failed");
        cudaHandles->registerInt8PreparedCleanup(
          [this]() { discardInt8Prepared(); });
        cudaHandles->preparedInt8Qk++;
      }
      catch(...) {
        discardInt8Prepared();
        cudaHandles->noteInt8PreparationFailure("qk-preparation-failed");
      }
    }
#endif
    }
    catch(...) {
#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
      discardInt8Prepared();
      // Model construction is aborting, so earlier registered callbacks would
      // otherwise outlive the already-destroyed BlockStack members.
      cudaHandles->int8PreparedCleanupRegistry.clear();
#endif
      destroyRawPreparedState();
      throw;
    }
  }

  ~TransformerAttentionBlock() {
#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
    discardInt8Prepared();
#endif
    destroyRawPreparedState();
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    (void)cudaHandles;
    (void)batchSize;
    return 0;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* maskBuf,
    float* maskSumBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    (void)maskSumBuf;
    (void)workspaceBuf;
    (void)workspaceBytes;

    int seqLen = nnXLen * nnYLen;
    int qTotalDim = numHeads * qHeadDim;
    int kTotalDim = numKVHeads * qHeadDim;
    int vTotalDim = numKVHeads * vHeadDim;
    size_t bytesPerElt = usingFP16 ? sizeof(half) : sizeof(float);
    int matBatchSize = batchSize * seqLen;

#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
    const bool useInt8Qk = cudaHandles->int8ExperimentPlan &&
      preLN.canApplyFp16Int8(maskBuf) && ropeCosSinTable != nullptr &&
      scratch->int8NormBuf != nullptr && scratch->int8QkTempBuf != nullptr &&
      qTotalDim == 256 && kTotalDim == 256 && vTotalDim == 256 &&
      katago_renju15_int8_qk_sm120_supports(
        int8QkKernel.get(),matBatchSize,inChannels,qTotalDim,kTotalDim,
        usingFP16,usingNHWC,maskBuf == nullptr);
    if(useInt8Qk) {
      preLN.applyFp16Int8(
        cudaHandles,batchSize,seqLen,trunkBuf,trunkScratchBuf,
        scratch->int8NormBuf,maskBuf);
    }
    else
#endif
    {
      preLN.apply(cudaHandles,batchSize,seqLen,trunkBuf,trunkScratchBuf,maskBuf);
    }

    // NHWC: trunk is [N, XY, C]. RMSNorm + mask zeroing.

#ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint3D("CUDA Attn RMSNorm out", trunkScratchBuf, batchSize, inChannels, seqLen, usingNHWC, usingFP16, maskBuf);
#endif

    // Step 2: Q/K/V projections
    // trunkScratchBuf is [N, XY, C] NHWC = [C, N*seqLen] column-major.
    // MatMulLayer expects input as [inChannels, batchSize], which matches.
    const size_t qElements = (size_t)qTotalDim * matBatchSize;
    const size_t kElements = (size_t)kTotalDim * matBatchSize;
    const size_t vElements = (size_t)vTotalDim * matBatchSize;
    SizedBuf<void*> qkvBuf(
      scratch->allocator,(qElements + kElements + vElements) * bytesPerElt);
    void* qData = qkvBuf.buf;
    void* kData = (char*)qkvBuf.buf + qElements * bytesPerElt;
    void* vData = (char*)kData + kElements * bytesPerElt;

    bool usedFusedQkvRope = false;
#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
    if(useInt8Qk) {
      CUDA_ERR(name.c_str(),katago_renju15_int8_qk_sm120_launch(
        int8QkKernel.get(),matBatchSize,(const int8_t*)scratch->int8NormBuf,
        (half*)scratch->int8QkTempBuf,cudaHandles->stream));
      CUDA_ERR(name.c_str(),katago_renju15_int8_qk_split_rope_sm120_launch(
        (const half*)scratch->int8QkTempBuf,(half*)qData,(half*)kData,
        (const half2*)ropeCosSinTable,batchSize,seqLen,cudaHandles->stream));
      // V intentionally remains on the FP16 path for this accuracy contract.
      vProj.apply(cudaHandles,scratch,matBatchSize,trunkScratchBuf,vData,
                  workspaceBuf,workspaceBytes);
      usedFusedQkvRope = true;
      if(!cudaHandles->loggedInt8Qk && cudaHandles->logger != NULL) {
        cudaHandles->logger->write(
          string("RENJU15_SM120_INT8_QK_ACTIVE marker=") +
          katago_renju15_int8_qk_sm120_marker());
        cudaHandles->loggedInt8Qk = true;
      }
      cudaHandles->noteInt8ExperimentLaunch(countedInt8Qk,true);
    }
#endif
#if defined(KATAGO_ENABLE_RENJU15_QKV_ROPE_GEMM_SM120) && KATAGO_ENABLE_RENJU15_QKV_ROPE_GEMM_SM120
    if(!usedFusedQkvRope && qkvPlanarProjection != nullptr &&
       recipe.qkvRope ==
         CudaTransformerWinner::QkvRopeTactic::Sm120C256H8D32M128N128K32S3 &&
       qkvPlanarProjection->supportsFusedQKVRoPE(
         batchSize,seqLen,numHeads,numKVHeads,qHeadDim,ropeNumPairs,
         usingFP16,usingNHWC,maskBuf == nullptr,ropeCosSinTable != nullptr,
         (const half2*)ropeCosSinTable)) {
      CUDA_ERR(name.c_str(),qkvPlanarProjection->applyFusedQKVRoPE(
        (const half*)trunkScratchBuf,(const half2*)ropeCosSinTable,
        (half*)qkvBuf.buf,batchSize,cudaHandles->stream));
      usedFusedQkvRope = true;
      if(!cudaHandles->loggedQkvRope && cudaHandles->logger != NULL) {
        const char* marker = qkvPlanarProjection->fusedQKVRoPEMarker();
        cudaHandles->logger->write(
          string("RENJU15_SM120_QKV_ROPE_ACTIVE marker=") +
          (marker == nullptr ? "missing" : marker));
        cudaHandles->loggedQkvRope = true;
      }
      cudaHandles->noteWinnerLaunch(
        countedWinnerQkvRope,cudaHandles->activeWinnerQkvRope);
    }
#endif
    if(!usedFusedQkvRope) {
      if(qkvPlanarProjection != nullptr) {
        const half* alpha = (const half*)scratch->oneBuf;
        const half* beta = (const half*)scratch->zeroBuf;
        CUBLAS_ERR(name.c_str(),qkvPlanarProjection->apply(
          cudaHandles->cublas,alpha,(const half*)trunkScratchBuf,beta,
          (half*)qkvBuf.buf,matBatchSize));
        if(!cudaHandles->loggedPlanarQkv && cudaHandles->logger != NULL) {
          cudaHandles->logger->write(
            "KATAGO_QKV_PLANAR_ACTIVE marker=cublas-hgemm-strided-batched");
          cudaHandles->loggedPlanarQkv = true;
        }
      }
      else {
        qProj.apply(cudaHandles, scratch, matBatchSize, trunkScratchBuf,
                    qData, workspaceBuf, workspaceBytes);
        kProj.apply(cudaHandles, scratch, matBatchSize, trunkScratchBuf,
                    kData, workspaceBuf, workspaceBytes);
        vProj.apply(cudaHandles, scratch, matBatchSize, trunkScratchBuf,
                    vData, workspaceBuf, workspaceBytes);
      }
    }

#ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint2D("CUDA Attn Q", qData, matBatchSize, qTotalDim, usingFP16);
#endif

    // Step 3: Apply RoPE to Q and K
    // Q is [qTotalDim, seqLen*batchSize] column-major = [batchSize*seqLen, qTotalDim] row-major
    if(useRope && !usedFusedQkvRope) {
      bool usedLearnedHalf2 = false;
      if(usingFP16 && learnableRope &&
         recipe.rope == CudaTransformerWinner::RopeTactic::LearnedHalf2 &&
         ropeCosSinTable != nullptr) {
        usedLearnedHalf2 = customCudaApplyLearnedQKRoPEHalf2(
          (half*)qData,(half*)kData,(const half2*)ropeCosSinTable,
          batchSize,seqLen,numHeads,numKVHeads,qHeadDim,ropeNumPairs,
          cudaHandles->stream);
        if(usedLearnedHalf2 && !cudaHandles->loggedLearnedRope &&
           cudaHandles->logger != NULL) {
          cudaHandles->logger->write(
            "KATAGO_LEARNED_ROPE_HALF2_ACTIVE marker=qk-fused-precomputed");
          cudaHandles->loggedLearnedRope = true;
        }
      }
      if(!usedLearnedHalf2 && !usingFP16) {
        customCudaApplyRoPE((float*)qData, (const float*)ropeCosTable, (const float*)ropeSinTable,
          batchSize, seqLen, numHeads, numKVHeads, qHeadDim, ropeNumPairs, learnableRope,cudaHandles->stream);
        customCudaApplyRoPE((float*)kData, (const float*)ropeCosTable, (const float*)ropeSinTable,
          batchSize, seqLen, numKVHeads, numKVHeads, qHeadDim, ropeNumPairs, learnableRope,cudaHandles->stream);
      }
      else if(!usedLearnedHalf2) {
        customCudaApplyRoPE((half*)qData, (const half*)ropeCosTable, (const half*)ropeSinTable,
          batchSize, seqLen, numHeads, numKVHeads, qHeadDim, ropeNumPairs, learnableRope,cudaHandles->stream);
        customCudaApplyRoPE((half*)kData, (const half*)ropeCosTable, (const half*)ropeSinTable,
          batchSize, seqLen, numKVHeads, numKVHeads, qHeadDim, ropeNumPairs, learnableRope,cudaHandles->stream);
      }
      CUDA_ERR(name.c_str(), cudaPeekAtLastError());
    }

    // Step 4: Scaled dot-product attention.
    // We use cudnn SDPA (FlashAttention-style, fused, no score-matrix materialization) when available
    // (FP16 + cudnn >= 8.9.3 + supported GPU). Otherwise fall back to a custom online-softmax CUDA kernel.
    // Both paths consume Q/K/V in BSHD layout and produce attnOut in the same layout as expected by outProj:
    //   attnOut: [numHeads*vHeadDim, seqLen*batchSize] col-major = [batchSize*seqLen, numHeads*vHeadDim] row-major.

    SizedBuf<void*> attnOutBuf(scratch->allocator, (size_t)numHeads * vHeadDim * seqLen * batchSize * bytesPerElt);

    bool usedSDPA = false;
#if defined(KATAGO_ENABLE_RENJU15_FA4_SM120) && KATAGO_ENABLE_RENJU15_FA4_SM120
    if(recipe.attention ==
       CudaTransformerWinner::AttentionTactic::Fa4Sm120B36S225Tm128Tn128S1Both16) {
      const bool preparedNoMask = cudaHandles->transformerPlan != nullptr &&
        cudaHandles->transformerPlan->runtime.maskMode == CudaOpRegistry::MaskMode::None;
      Renju15Fa4Sm120::LaunchResult result = Renju15Fa4Sm120::launch(
        Renju15Fa4Sm120::Tactic::B36Tm128Tn128S1Both16,
        (half*)qData,(half*)kData,(half*)vData,(half*)attnOutBuf.buf,
        batchSize,seqLen,numHeads,numKVHeads,qHeadDim,vHeadDim,
        usingFP16,usingNHWC,maskBuf,preparedNoMask,
        cudaHandles->majorComputeCapability,cudaHandles->minorComputeCapability,
        cudaHandles->stream);
      if(result.attempted) {
        CUDA_ERR(name.c_str(),result.status);
        usedSDPA = true;
        if(!cudaHandles->loggedFa4 && cudaHandles->logger != NULL) {
          cudaHandles->logger->write(
            string("RENJU15_SM120_FA4_ACTIVE marker=") +
            (result.marker == nullptr ? "missing" : result.marker));
          cudaHandles->loggedFa4 = true;
        }
        cudaHandles->noteWinnerLaunch(
          countedWinnerFa4,cudaHandles->activeWinnerFa4);
      }
    }
#endif
#if KATAGO_CUDA_HAS_SDPA
    SDPAGraphCache* sdpaCache = cudaHandles->sdpaCache.get();
    if(!usedSDPA && usingFP16 && sdpaCache != NULL) {
      bool hasMask = (maskBuf != NULL);
      SDPAGraphKey sdpaKey = {numHeads, numKVHeads, qHeadDim, vHeadDim, seqLen, batchSize, hasMask, usingFP16};
      auto plan = sdpaCache->getOrBuildPlan(cudaHandles->cudnn, sdpaKey, cudaHandles->logger);
      if(plan != nullptr) {
        std::unordered_map<int64_t, void*> variant_pack = {
          {SDPAPlanForBatchSize::Q_UID, qData},
          {SDPAPlanForBatchSize::K_UID, kData},
          {SDPAPlanForBatchSize::V_UID, vData},
          {SDPAPlanForBatchSize::O_UID, attnOutBuf.buf},
        };

        // Variable-board inference materializes a half [B,1,S,S] additive key bias. Exact-board
        // inference receives maskBuf=NULL and allocates no bias tensor.
        std::unique_ptr<SizedBuf<void*>> biasBuf;
        if(hasMask) {
          biasBuf = std::make_unique<SizedBuf<void*>>(
            scratch->allocator,
            (size_t)batchSize * seqLen * seqLen * bytesPerElt
          );
          customCudaMaskToAttnBiasFull(
            (const half*)maskBuf,
            (half*)biasBuf->buf,
            batchSize,
            seqLen,
            cudaHandles->stream
          );
          CUDA_ERR(name.c_str(),cudaPeekAtLastError());
          variant_pack[SDPAPlanForBatchSize::BIAS_UID] = biasBuf->buf;
        }

        // Allocate one byte when cuDNN reports no workspace because cudaMalloc(0) is not portable.
        const size_t sdpaWorkspaceBytes =
          plan->workspaceBytes > 0 ? (size_t)plan->workspaceBytes : (size_t)1;
        SizedBuf<void*> sdpaWs(scratch->allocator,sdpaWorkspaceBytes);

        auto status = plan->graph->execute(cudaHandles->cudnn, variant_pack, sdpaWs.buf);
        // Capability misses fall back before this point. Retrying another kernel after execute may
        // have partially enqueued work, so an execution error is deliberately fatal.
        if(status.is_bad())
          throw StringError(string("cudnn SDPA execute failed: ") + status.get_message());
        usedSDPA = true;
        if(cudaHandles->logger != NULL &&
           cudaHandles->loggedSdpaKeys.insert(sdpaKey).second) {
          cudaHandles->logger->write(
            "CUDA_CUDNN_SDPA_ACTIVE marker=frontend-graph mask=" +
            string(hasMask ? "dense" : "none") +
            " B=" + Global::intToString(batchSize) +
            " S=" + Global::intToString(seqLen) +
            " Hq=" + Global::intToString(numHeads) +
            " Hkv=" + Global::intToString(numKVHeads) +
            " Dq=" + Global::intToString(qHeadDim) +
            " Dv=" + Global::intToString(vHeadDim));
        }
      }
    }
#endif

    if(!usedSDPA) {
      if(!usingFP16) {
        customCudaFlashAttention(
          (const float*)qData, (const float*)kData, (const float*)vData,
          (const float*)maskBuf, (float*)attnOutBuf.buf,
          batchSize, seqLen, numHeads, numKVHeads, qHeadDim, vHeadDim,cudaHandles->stream);
      }
      else {
        customCudaFlashAttention(
          (const half*)qData, (const half*)kData, (const half*)vData,
          (const half*)maskBuf, (half*)attnOutBuf.buf,
          batchSize, seqLen, numHeads, numKVHeads, qHeadDim, vHeadDim,cudaHandles->stream);
      }
      CUDA_ERR(name.c_str(), cudaPeekAtLastError());
    }

    // Step 5-6: output projection and residual epilogue. The prepared path
    // writes directly into trunkBuf with beta=1.
    bool usedSpecializedResidual = false;
    const bool usedPreparedResidual = outProj.applyPreparedResidual(
      cudaHandles,recipe.outProjection,outProjectionKernel,matBatchSize,
      attnOutBuf.buf,trunkBuf,maskBuf,false,usedSpecializedResidual);
    if(usedSpecializedResidual && recipe.outProjection ==
         CudaTransformerWinner::ResidualTactic::Sm120M128N128K32S3Sw1)
      cudaHandles->noteWinnerLaunch(
        countedWinnerOutProjection,cudaHandles->activeWinnerOutProjection);
    if(!usedPreparedResidual) {
      // attnOutBuf is [numHeads*vHeadDim, seqLen*batchSize] col-major.
      outProj.apply(cudaHandles, scratch, matBatchSize, attnOutBuf.buf,
                    trunkScratchBuf, workspaceBuf, workspaceBytes);

#ifdef DEBUG_INTERMEDIATE_VALUES
      CudaUtils::debugPrint3D("CUDA Attn outProj", trunkScratchBuf, batchSize, inChannels, seqLen, usingNHWC, usingFP16, maskBuf);
#endif

      // NHWC: trunk is [N, XY, C], mask is [N, XY].
      if(!usingFP16) {
        customCudaMaskedResidualAddNHWC((float*)trunkBuf, (const float*)trunkScratchBuf, (const float*)maskBuf, batchSize, seqLen, inChannels,cudaHandles->stream);
      }
      else {
        customCudaMaskedResidualAddNHWC((half*)trunkBuf, (const half*)trunkScratchBuf, (const half*)maskBuf, batchSize, seqLen, inChannels,cudaHandles->stream);
      }
      CUDA_ERR(name.c_str(), cudaPeekAtLastError());
    }

#ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint3D("CUDA Attn residual", trunkBuf, batchSize, inChannels, seqLen, usingNHWC, usingFP16, maskBuf);
#endif
  }
};

//------------------------------------------------------------------------------

struct TransformerFFNBlock {
  const string name;
  const int numChannels;
  const int ffnChannels;
  const bool useSwiGLU;

  const int nnXLen;
  const int nnYLen;
  const bool usingFP16;
  const bool usingNHWC;

  const TransformerRMSNormLayer preLN;
  const MatMulLayer linear1;
  std::unique_ptr<MatMulLayer> linearGate;
  const MatMulLayer linear2;
  const CudaTransformerWinner::FfnRecipe recipe;
  void* dualFfnKernel;
  void* downProjectionKernel;
#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
  UniqueCudaDeviceBuffer int8UpWeightBuf;
  UniqueCudaDeviceBuffer int8GateWeightBuf;
  UniqueInt8DualFfnKernel int8DualFfnKernel;
  mutable bool countedInt8DualFfn;
#endif
  mutable bool countedWinnerDualFfn;
  mutable bool countedWinnerFfnDown;

  TransformerFFNBlock() = delete;
  TransformerFFNBlock(const TransformerFFNBlock&) = delete;
  TransformerFFNBlock& operator=(const TransformerFFNBlock&) = delete;

#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
  void discardInt8Prepared() noexcept {
    int8DualFfnKernel.reset();
    int8GateWeightBuf.reset();
    int8UpWeightBuf.reset();
  }
#endif

  void destroyRawPreparedState() noexcept {
#if defined(KATAGO_ENABLE_RENJU15_DUAL_FFN_SM120) && KATAGO_ENABLE_RENJU15_DUAL_FFN_SM120
    katago_renju15_dual_ffn_sm120_destroy(dualFfnKernel);
#endif
    dualFfnKernel = nullptr;
#if defined(KATAGO_ENABLE_RENJU15_GEMM_TACTICS_SM120) && KATAGO_ENABLE_RENJU15_GEMM_TACTICS_SM120
    katago_renju15_residual_gemm_sm120_destroy(downProjectionKernel);
#endif
    downProjectionKernel = nullptr;
  }

  TransformerFFNBlock(
    CudaHandles* cudaHandles,
    const TransformerFFNDesc* desc,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC,
    const CudaTransformerWinner::FfnRecipe& selectedRecipe,
    int fixedBatchSize,
    uint32_t topologyIndex
  ) :
    name(desc->name),
    numChannels(desc->numChannels),
    ffnChannels(desc->ffnChannels),
    useSwiGLU(desc->useSwiGLU),
    nnXLen(nnX),
    nnYLen(nnY),
    usingFP16(useFP16),
    usingNHWC(useNHWC),
    preLN(cudaHandles, &desc->preLN, useFP16, selectedRecipe.rmsNorm),
    linear1(cudaHandles, &desc->linear1, useFP16),
    linear2(cudaHandles, &desc->linear2, useFP16),
    recipe(selectedRecipe),
    dualFfnKernel(nullptr),
    downProjectionKernel(nullptr),
#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
    int8UpWeightBuf(nullptr),
    int8GateWeightBuf(nullptr),
    int8DualFfnKernel(nullptr),
    countedInt8DualFfn(false),
#endif
    countedWinnerDualFfn(false),
    countedWinnerFfnDown(false)
  {
    if(!useSwiGLU) {
      throw StringError("Non-SwiGLU transformer FFN is not yet supported in CUDA backend");
    }
    linearGate = std::make_unique<MatMulLayer>(cudaHandles, &desc->linearGate, useFP16);
    if(!useNHWC) {
      throw StringError("Transformer blocks with NCHW layout are not yet supported by the CUDA backend");
    }
    try {
#if defined(KATAGO_ENABLE_RENJU15_DUAL_FFN_SM120) && KATAGO_ENABLE_RENJU15_DUAL_FFN_SM120
    if(recipe.dualFfn ==
       CudaTransformerWinner::DualFfnTactic::Sm120C256F768M128N64K32S3Sw4) {
      dualFfnKernel = katago_renju15_dual_ffn_sm120_create(
        KATAGO_RENJU15_DUAL_FFN_M128_N64_K32_S3_SW4,
        fixedBatchSize * nnXLen * nnYLen);
    }
    else if(recipe.dualFfn ==
       CudaTransformerWinner::DualFfnTactic::Sm120C384F1024M128N64K32S3Sw4) {
      dualFfnKernel = katago_renju15_dual_ffn_sm120_create(
        KATAGO_RENJU15_DUAL_FFN_C384_F1024_M128_N64_K32_S3_SW4,
        fixedBatchSize * nnXLen * nnYLen);
    }
#endif
#if defined(KATAGO_ENABLE_RENJU15_GEMM_TACTICS_SM120) && KATAGO_ENABLE_RENJU15_GEMM_TACTICS_SM120
    if(recipe.downProjection ==
       CudaTransformerWinner::ResidualTactic::Sm120M128N128K32S3Sw1) {
      downProjectionKernel = katago_renju15_residual_gemm_sm120_create(
        KATAGO_RENJU15_RESIDUAL_GEMM_FFN_DOWN,
        KATAGO_RENJU15_RESIDUAL_GEMM_M128_N128_K32_S3,
        fixedBatchSize * nnXLen * nnYLen);
    }
    else if(recipe.downProjection ==
       CudaTransformerWinner::ResidualTactic::Sm120C384M128N128K32S3Sw1) {
      downProjectionKernel = katago_renju15_residual_gemm_sm120_create(
        KATAGO_RENJU15_RESIDUAL_GEMM_C384_FFN_DOWN,
        KATAGO_RENJU15_RESIDUAL_GEMM_C384_M128_N128_K32_S3,
        fixedBatchSize * nnXLen * nnYLen);
    }
#endif
    if(cudaHandles->exactWinnerPlan) {
      const bool expectsDual =
        recipe.dualFfn != CudaTransformerWinner::DualFfnTactic::Disabled;
      const bool expectsDown = recipe.downProjection ==
        CudaTransformerWinner::ResidualTactic::Sm120M128N128K32S3Sw1;
      if((expectsDual && dualFfnKernel == nullptr) ||
         (expectsDown && downProjectionKernel == nullptr)) {
#if defined(KATAGO_ENABLE_RENJU15_DUAL_FFN_SM120) && KATAGO_ENABLE_RENJU15_DUAL_FFN_SM120
        katago_renju15_dual_ffn_sm120_destroy(dualFfnKernel);
        dualFfnKernel = nullptr;
#endif
#if defined(KATAGO_ENABLE_RENJU15_GEMM_TACTICS_SM120) && KATAGO_ENABLE_RENJU15_GEMM_TACTICS_SM120
        katago_renju15_residual_gemm_sm120_destroy(downProjectionKernel);
        downProjectionKernel = nullptr;
#endif
        throw StringError("Certified CUDA transformer FFN handle preparation failed");
      }
      if(expectsDual)
        cudaHandles->preparedWinnerDualFfn++;
      if(expectsDown)
        cudaHandles->preparedWinnerFfnDown++;
    }
#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
    const bool int8ShapeEligible = cudaHandles->int8ExperimentPlan &&
      nnXLen == 15 && nnYLen == 15 && useFP16 && useNHWC && useSwiGLU &&
      numChannels == 256 && ffnChannels == 768 &&
      desc->linear1.inChannels == 256 && desc->linear1.outChannels == 768 &&
      desc->linearGate.inChannels == 256 &&
      desc->linearGate.outChannels == 768;
    const NativeInt8Quant::Entry* embeddedUpEntry = nullptr;
    const NativeInt8Quant::Entry* embeddedGateEntry = nullptr;
    if(int8ShapeEligible && cudaHandles->usesEmbeddedInt8Weights()) {
      // As with QK, a topology/name/shape mismatch is fatal model metadata,
      // while allocation/upload failures below remain coherent FP16 fallback.
      embeddedUpEntry = &cudaHandles->requireEmbeddedInt8Entry(
        topologyIndex,NativeInt8Quant::Role::FfnUp,
        vector<string>{desc->linear1.name},256,768);
      embeddedGateEntry = &cudaHandles->requireEmbeddedInt8Entry(
        topologyIndex,NativeInt8Quant::Role::FfnGate,
        vector<string>{desc->linearGate.name},256,768);
    }
    if(int8ShapeEligible) {
      try {
        float upScale = 0.0f;
        float gateScale = 0.0f;
        void* packedUpWeights = nullptr;
        if(embeddedUpEntry != nullptr)
          prepareEmbeddedPackedInt8(
            name + ":int8Up",*embeddedUpEntry,packedUpWeights,upScale);
        else
          preparePackedInt8Matrix(
            name + ":int8Up",desc->linear1.weights,256,768,
            packedUpWeights,upScale);
        int8UpWeightBuf.reset(packedUpWeights);
        void* packedGateWeights = nullptr;
        if(embeddedGateEntry != nullptr)
          prepareEmbeddedPackedInt8(
            name + ":int8Gate",*embeddedGateEntry,
            packedGateWeights,gateScale);
        else
          preparePackedInt8Matrix(
            name + ":int8Gate",desc->linearGate.weights,256,768,
            packedGateWeights,gateScale);
        int8GateWeightBuf.reset(packedGateWeights);
        int8DualFfnKernel.reset(katago_renju15_int8_dual_ffn_sm120_create(
          fixedBatchSize * nnXLen * nnYLen,
          (const int8_t*)int8UpWeightBuf.get(),
          (const int8_t*)int8GateWeightBuf.get(),upScale,gateScale));
        if(int8DualFfnKernel == nullptr)
          throw StringError(name + ": INT8 dual-FFN handle preparation failed");
        cudaHandles->registerInt8PreparedCleanup(
          [this]() { discardInt8Prepared(); });
        cudaHandles->preparedInt8DualFfn++;
      }
      catch(...) {
        discardInt8Prepared();
        cudaHandles->noteInt8PreparationFailure("dual-ffn-preparation-failed");
      }
    }
#endif
    }
    catch(...) {
#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
      discardInt8Prepared();
      // Model construction is aborting, so earlier registered callbacks would
      // otherwise outlive the already-destroyed BlockStack members.
      cudaHandles->int8PreparedCleanupRegistry.clear();
#endif
      destroyRawPreparedState();
      throw;
    }
  }

  ~TransformerFFNBlock()
  {
#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
    discardInt8Prepared();
#endif
    destroyRawPreparedState();
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    (void)cudaHandles;
    (void)batchSize;
    return 0;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* maskBuf,
    float* maskSumBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    (void)maskSumBuf;

    int seqLen = nnXLen * nnYLen;
    int matBatchSize = batchSize * seqLen;
    size_t bytesPerElt = usingFP16 ? sizeof(half) : sizeof(float);

    // Step 1: RMSNorm
#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
    const bool useInt8DualFfn = cudaHandles->int8ExperimentPlan &&
      preLN.canApplyFp16Int8(maskBuf) && scratch->int8NormBuf != nullptr &&
      katago_renju15_int8_dual_ffn_sm120_supports(
        int8DualFfnKernel.get(),matBatchSize,numChannels,ffnChannels,
        usingFP16,usingNHWC,maskBuf == nullptr);
    if(useInt8DualFfn) {
      preLN.applyFp16Int8(
        cudaHandles,batchSize,seqLen,trunkBuf,trunkScratchBuf,
        scratch->int8NormBuf,maskBuf);
    }
    else
#endif
    {
      preLN.apply(cudaHandles,batchSize,seqLen,trunkBuf,trunkScratchBuf,maskBuf);
    }

#ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint3D("CUDA FFN RMSNorm out", trunkScratchBuf, batchSize, numChannels, seqLen, usingNHWC, usingFP16, maskBuf);
#endif

    // Step 2-3: dual projection + SwiGLU. A failed handle or launch-shape
    // capability check falls back before any work is enqueued.
    SizedBuf<void*> ffnBuf(scratch->allocator, (size_t)ffnChannels * matBatchSize * bytesPerElt);
    bool usedDualFfn = false;
#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
    if(useInt8DualFfn) {
      CUDA_ERR(name.c_str(),katago_renju15_int8_dual_ffn_sm120_launch(
        int8DualFfnKernel.get(),matBatchSize,
        (const int8_t*)scratch->int8NormBuf,
        (half*)ffnBuf.buf,cudaHandles->stream));
      usedDualFfn = true;
      if(!cudaHandles->loggedInt8DualFfn && cudaHandles->logger != NULL) {
        cudaHandles->logger->write(
          string("RENJU15_SM120_INT8_DUAL_FFN_ACTIVE marker=") +
          katago_renju15_int8_dual_ffn_sm120_marker());
        cudaHandles->loggedInt8DualFfn = true;
      }
      cudaHandles->noteInt8ExperimentLaunch(countedInt8DualFfn,false);
    }
#endif
#if defined(KATAGO_ENABLE_RENJU15_DUAL_FFN_SM120) && KATAGO_ENABLE_RENJU15_DUAL_FFN_SM120
    const bool specializedDualFfn =
      recipe.dualFfn ==
        CudaTransformerWinner::DualFfnTactic::Sm120C256F768M128N64K32S3Sw4 ||
      recipe.dualFfn ==
        CudaTransformerWinner::DualFfnTactic::Sm120C384F1024M128N64K32S3Sw4;
    if(!usedDualFfn && specializedDualFfn &&
       katago_renju15_dual_ffn_sm120_supports(
         dualFfnKernel,matBatchSize,numChannels,ffnChannels,
         usingFP16,usingNHWC,maskBuf == nullptr)) {
      CUDA_ERR(name.c_str(),katago_renju15_dual_ffn_sm120_launch(
        dualFfnKernel,(const half*)trunkScratchBuf,
        (const half*)linear1.matBuf,(const half*)linearGate->matBuf,
        (half*)ffnBuf.buf,cudaHandles->stream));
      usedDualFfn = true;
      if(!cudaHandles->loggedDualFfn && cudaHandles->logger != NULL) {
        const char* marker = katago_renju15_dual_ffn_sm120_active_marker(
          dualFfnKernel);
        cudaHandles->logger->write(
          string("RENJU15_SM120_DUAL_FFN_ACTIVE marker=") +
          (marker == nullptr ? "missing" : marker));
        cudaHandles->loggedDualFfn = true;
      }
      if(recipe.dualFfn ==
           CudaTransformerWinner::DualFfnTactic::Sm120C256F768M128N64K32S3Sw4)
        cudaHandles->noteWinnerLaunch(
          countedWinnerDualFfn,cudaHandles->activeWinnerDualFfn);
    }
#endif
    if(!usedDualFfn) {
      linear1.apply(cudaHandles, scratch, matBatchSize, trunkScratchBuf,
                    ffnBuf.buf, workspaceBuf, workspaceBytes);
      SizedBuf<void*> gateBuf(scratch->allocator, (size_t)ffnChannels * matBatchSize * bytesPerElt);
      linearGate->apply(cudaHandles, scratch, matBatchSize, trunkScratchBuf, gateBuf.buf, workspaceBuf, workspaceBytes);

      int totalSize = (int)((size_t)ffnChannels * matBatchSize);
      if(!usingFP16) {
        customCudaSwiGLU((const float*)ffnBuf.buf, (const float*)gateBuf.buf, (float*)ffnBuf.buf, totalSize,cudaHandles->stream);
      }
      else {
        customCudaSwiGLU((const half*)ffnBuf.buf, (const half*)gateBuf.buf, (half*)ffnBuf.buf, totalSize,cudaHandles->stream);
      }
      CUDA_ERR(name.c_str(), cudaPeekAtLastError());
    }

#ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint2D("CUDA FFN SwiGLU", ffnBuf.buf, matBatchSize, ffnChannels, usingFP16);
#endif

    // Step 4-5: down projection and residual epilogue.
    bool usedSpecializedResidual = false;
    const bool usedPreparedResidual = linear2.applyPreparedResidual(
      cudaHandles,recipe.downProjection,
      downProjectionKernel,
      matBatchSize,
      ffnBuf.buf,trunkBuf,maskBuf,true,usedSpecializedResidual);
    if(usedSpecializedResidual && recipe.downProjection ==
         CudaTransformerWinner::ResidualTactic::Sm120M128N128K32S3Sw1)
      cudaHandles->noteWinnerLaunch(
        countedWinnerFfnDown,cudaHandles->activeWinnerFfnDown);
    if(!usedPreparedResidual) {
      linear2.apply(cudaHandles, scratch, matBatchSize, ffnBuf.buf,
                    trunkScratchBuf, workspaceBuf, workspaceBytes);
      if(!usingFP16) {
        customCudaMaskedResidualAddNHWC((float*)trunkBuf, (const float*)trunkScratchBuf, (const float*)maskBuf, batchSize, seqLen, numChannels,cudaHandles->stream);
      }
      else {
        customCudaMaskedResidualAddNHWC((half*)trunkBuf, (const half*)trunkScratchBuf, (const half*)maskBuf, batchSize, seqLen, numChannels,cudaHandles->stream);
      }
      CUDA_ERR(name.c_str(), cudaPeekAtLastError());
    }

#ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint3D("CUDA FFN residual", trunkBuf, batchSize, numChannels, seqLen, usingNHWC, usingFP16, maskBuf);
#endif
  }
};

//------------------------------------------------------------------------------

BlockStack::BlockStack(
  CudaHandles* cudaHandles,
  CudnnManager* manager,
  int nBlocks,
  int trunkChannels,
  const std::vector<std::pair<int, unique_ptr_void>>& descBlocks,
  int nnX,
  int nnY,
  bool useFP16,
  bool useNHWC,
  TransformerPlanCursor* planCursor
) :
  numBlocks(nBlocks),
  trunkNumChannels(trunkChannels),
  nnXLen(nnX),
  nnYLen(nnY),
  usingFP16(useFP16),
  usingNHWC(useNHWC)
{
  assert(numBlocks == descBlocks.size());
  for(int i = 0; i<numBlocks; i++) {
    if(descBlocks[i].first == ORDINARY_BLOCK_KIND) {
      ResidualBlockDesc* blockDesc = (ResidualBlockDesc*)descBlocks[i].second.get();
      unique_ptr_void blockPtr = make_unique_void(
        new ResidualBlock(
          cudaHandles,
          manager,
          blockDesc,
          nnXLen,
          nnYLen,
          useFP16,
          useNHWC
        )
      );
      blocks.push_back(make_pair(ORDINARY_BLOCK_KIND,std::move(blockPtr)));
    }
    else if(descBlocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
      GlobalPoolingResidualBlockDesc* blockDesc = (GlobalPoolingResidualBlockDesc*)descBlocks[i].second.get();
      unique_ptr_void blockPtr = make_unique_void(
        new GlobalPoolingResidualBlock(
          cudaHandles,
          manager,
          blockDesc,
          nnXLen,
          nnYLen,
          useFP16,
          useNHWC
        )
      );
      blocks.push_back(make_pair(GLOBAL_POOLING_BLOCK_KIND,std::move(blockPtr)));
    }
    else if(descBlocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      NestedBottleneckResidualBlockDesc* blockDesc = (NestedBottleneckResidualBlockDesc*)descBlocks[i].second.get();
      unique_ptr_void blockPtr = make_unique_void(
        new NestedBottleneckResidualBlock(
          cudaHandles,
          manager,
          blockDesc,
          nnXLen,
          nnYLen,
          useFP16,
          useNHWC,
          planCursor
        )
      );
      blocks.push_back(make_pair(NESTED_BOTTLENECK_BLOCK_KIND,std::move(blockPtr)));
    }
    else if(descBlocks[i].first == TRANSFORMER_ATTENTION_BLOCK_KIND) {
      TransformerAttentionDesc* blockDesc = (TransformerAttentionDesc*)descBlocks[i].second.get();
      uint32_t topologyIndex = UINT32_MAX;
      const CudaTransformerWinner::AttentionRecipe selectedRecipe =
        planCursor == nullptr ? CudaTransformerWinner::AttentionRecipe{} :
        planCursor->nextAttention(topologyIndex);
      unique_ptr_void blockPtr = make_unique_void(
        new TransformerAttentionBlock(
          cudaHandles,
          blockDesc,
          nnXLen,
          nnYLen,
          useFP16,
          useNHWC,
          selectedRecipe,
          manager->maxBatchSize,
          topologyIndex
        )
      );
      blocks.push_back(make_pair(TRANSFORMER_ATTENTION_BLOCK_KIND,std::move(blockPtr)));
    }
    else if(descBlocks[i].first == TRANSFORMER_FFN_BLOCK_KIND) {
      TransformerFFNDesc* blockDesc = (TransformerFFNDesc*)descBlocks[i].second.get();
      uint32_t topologyIndex = UINT32_MAX;
      const CudaTransformerWinner::FfnRecipe selectedRecipe =
        planCursor == nullptr ? CudaTransformerWinner::FfnRecipe{} :
        planCursor->nextFfn(topologyIndex);
      unique_ptr_void blockPtr = make_unique_void(
        new TransformerFFNBlock(
          cudaHandles,
          blockDesc,
          nnXLen,
          nnYLen,
          useFP16,
          useNHWC,
          selectedRecipe,
          manager->maxBatchSize,
          topologyIndex
        )
      );
      blocks.push_back(make_pair(TRANSFORMER_FFN_BLOCK_KIND,std::move(blockPtr)));
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }
}
BlockStack::~BlockStack() {
}

size_t BlockStack::requiredWorkspaceBytes(
  CudaHandles* cudaHandles,
  int batchSize
) const {
  size_t bytes = 0;
  size_t b;

  for(int i = 0; i<blocks.size(); i++) {
    if(blocks[i].first == ORDINARY_BLOCK_KIND) {
      ResidualBlock* block = (ResidualBlock*)blocks[i].second.get();
      b = block->requiredWorkspaceBytes(cudaHandles,batchSize);
      bytes = std::max(bytes,b);
    }
    else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
      GlobalPoolingResidualBlock* block = (GlobalPoolingResidualBlock*)blocks[i].second.get();
      b = block->requiredWorkspaceBytes(cudaHandles,batchSize);
      bytes = std::max(bytes,b);
    }
    else if(blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      NestedBottleneckResidualBlock* block = (NestedBottleneckResidualBlock*)blocks[i].second.get();
      b = block->requiredWorkspaceBytes(cudaHandles,batchSize);
      bytes = std::max(bytes,b);
    }
    else if(blocks[i].first == TRANSFORMER_ATTENTION_BLOCK_KIND) {
      TransformerAttentionBlock* block = (TransformerAttentionBlock*)blocks[i].second.get();
      b = block->requiredWorkspaceBytes(cudaHandles,batchSize);
      bytes = std::max(bytes,b);
    }
    else if(blocks[i].first == TRANSFORMER_FFN_BLOCK_KIND) {
      TransformerFFNBlock* block = (TransformerFFNBlock*)blocks[i].second.get();
      b = block->requiredWorkspaceBytes(cudaHandles,batchSize);
      bytes = std::max(bytes,b);
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }
  return bytes;
}

void BlockStack::apply(
  CudaHandles* cudaHandles,
  ScratchBuffers* scratch,
  int batchSize,
  void* maskBuf,
  float* maskSumBuf,
  void* trunkBuf,
  void* trunkScratchBuf,
  void* workspaceBuf,
  size_t workspaceBytes
) const {

  for(int i = 0; i<blocks.size(); i++) {
#ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint3D("CUDA Blockstack block " + Global::intToString(i), trunkBuf, batchSize, trunkNumChannels, nnXLen*nnYLen, usingNHWC, usingFP16, maskBuf);
#endif

    if(blocks[i].first == ORDINARY_BLOCK_KIND) {
      ResidualBlock* block = (ResidualBlock*)blocks[i].second.get();
      block->apply(
        cudaHandles,
        scratch,
        batchSize,
        trunkBuf,
        trunkScratchBuf,
        maskBuf,
        workspaceBuf,
        workspaceBytes
      );
    }
    else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
      GlobalPoolingResidualBlock* block = (GlobalPoolingResidualBlock*)blocks[i].second.get();
      block->apply(
        cudaHandles,
        scratch,
        batchSize,
        trunkBuf,
        trunkScratchBuf,
        maskBuf,
        maskSumBuf,
        workspaceBuf,
        workspaceBytes
      );
    }
    else if(blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      NestedBottleneckResidualBlock* block = (NestedBottleneckResidualBlock*)blocks[i].second.get();
      block->apply(
        cudaHandles,
        scratch,
        batchSize,
        trunkBuf,
        trunkScratchBuf,
        maskBuf,
        maskSumBuf,
        workspaceBuf,
        workspaceBytes
      );
    }
    else if(blocks[i].first == TRANSFORMER_ATTENTION_BLOCK_KIND) {
      TransformerAttentionBlock* block = (TransformerAttentionBlock*)blocks[i].second.get();
      block->apply(
        cudaHandles,
        scratch,
        batchSize,
        trunkBuf,
        trunkScratchBuf,
        maskBuf,
        maskSumBuf,
        workspaceBuf,
        workspaceBytes
      );
    }
    else if(blocks[i].first == TRANSFORMER_FFN_BLOCK_KIND) {
      TransformerFFNBlock* block = (TransformerFFNBlock*)blocks[i].second.get();
      block->apply(
        cudaHandles,
        scratch,
        batchSize,
        trunkBuf,
        trunkScratchBuf,
        maskBuf,
        maskSumBuf,
        workspaceBuf,
        workspaceBytes
      );
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }
}

//----------------------------------------------------------------------------


struct Trunk {
  const string name;
  const int version;
  const int numBlocks;
  const int trunkNumChannels;

  const int nnXLen;
  const int nnYLen;
  const bool usingFP16;
  const bool usingNHWC;

  std::unique_ptr<ConvLayer> initialConv;
  std::unique_ptr<MatMulLayer> initialMatMul;
  const BlockStack blocks;
  std::unique_ptr<BatchNormLayer> trunkTipBN;

  Trunk() = delete;
  Trunk(const Trunk&) = delete;
  Trunk& operator=(const Trunk&) = delete;

  Trunk(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const TrunkDesc* desc,
    int nnX,
    int nnY,
    bool inputsUseNHWC,
    bool useFP16,
    bool useNHWC,
    TransformerPlanCursor* planCursor
  ) :
    name(desc->name),
    version(desc->version),
    numBlocks(desc->numBlocks),
    trunkNumChannels(desc->trunkNumChannels),
    nnXLen(nnX),
    nnYLen(nnY),
    usingFP16(useFP16),
    usingNHWC(useNHWC),
    blocks(cudaHandles,manager,desc->numBlocks,desc->trunkNumChannels,desc->blocks,
           nnX,nnY,useFP16,useNHWC,planCursor)
  {
    int midNumChannels = desc->midNumChannels;
    int regularNumChannels = desc->regularNumChannels;
    int gpoolNumChannels = desc->gpoolNumChannels;

    int maxBatchSize = manager->maxBatchSize;
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,trunkNumChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,midNumChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,regularNumChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,gpoolNumChannels);

    initialConv = std::make_unique<ConvLayer>(cudaHandles,manager,&desc->initialConv,useFP16,inputsUseNHWC,useNHWC);
    initialMatMul = std::make_unique<MatMulLayer>(cudaHandles,&desc->initialMatMul,useFP16);

    trunkTipBN = std::make_unique<BatchNormLayer>(cudaHandles,&desc->trunkTipBN,&desc->trunkTipActivation,nnXLen,nnYLen,useFP16,useNHWC);
    assert(desc->blocks.size() == numBlocks);
  }

  ~Trunk()
  {
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;

    b = initialConv->requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);

    b = initialMatMul->requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);

    b = blocks.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* inputBuf,
    void* inputGlobalBuf,
    void* maskBuf,
    float* maskSumBuf,
    void* trunkBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {

    SizedBuf<void*> trunkScratch(scratch->allocator, scratch->getBufSizeXY(trunkNumChannels));

    //Feed the conv into trunkScratch.buf, not trunkBuf
    initialConv->apply(cudaHandles,batchSize,false,inputBuf,trunkScratch.buf,workspaceBuf,workspaceBytes);

    #ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint3D(string("After initial conv"), trunkScratch.buf, batchSize, trunkNumChannels, nnXLen*nnYLen, usingNHWC, usingFP16);
    #endif

    //Feed the matmul into trunkBuf
    initialMatMul->apply(cudaHandles,scratch,batchSize,inputGlobalBuf,trunkBuf,workspaceBuf,workspaceBytes);
    //Then accumulate it into trunkScratch.buf, broadcasting during the process
    if(!usingFP16) {
      if(!usingNHWC)
        customCudaAddNCBiasInplaceNCHW((float*)trunkScratch.buf,(const float*)trunkBuf,batchSize,trunkNumChannels,nnXLen*nnYLen,cudaHandles->stream);
      else
        customCudaAddNCBiasInplaceNHWC((float*)trunkScratch.buf,(const float*)trunkBuf,batchSize,nnXLen*nnYLen,trunkNumChannels,cudaHandles->stream);
    }
    else {
      if(!usingNHWC)
        customCudaAddNCBiasInplaceNCHW((half*)trunkScratch.buf,(const half*)trunkBuf,batchSize,trunkNumChannels,nnXLen*nnYLen,cudaHandles->stream);
      else
        customCudaAddNCBiasInplaceNHWC((half*)trunkScratch.buf,(const half*)trunkBuf,batchSize,nnXLen*nnYLen,trunkNumChannels,cudaHandles->stream);
    }
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());

    //Flip trunkBuf and trunkScratch.buf so that the result gets accumulated in trunkScratch.buf
    blocks.apply(
      cudaHandles,
      scratch,
      batchSize,
      maskBuf,
      maskSumBuf,
      trunkScratch.buf,
      trunkBuf,
      workspaceBuf,
      workspaceBytes
    );

    //And now with the final BN port it from trunkScratch.buf to trunkBuf.
    trunkTipBN->apply(cudaHandles,batchSize,trunkScratch.buf,maskBuf,trunkBuf);
    #ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint4D(string("Trunk tip"), trunkBuf, batchSize, trunkNumChannels, nnXLen, nnYLen, usingNHWC, usingFP16);
    #endif
  }

};

//------------------------------------------------------------------------------

static void fillMaskFloatBufAndMaskSumBuf(CudaHandles* cudaHandles, void* maskBuf, float*& maskFloatBuf, float*& maskSumBuf, bool usingFP16, int batchSize, int nnXLen, int nnYLen) {
  if(!usingFP16) {
    maskFloatBuf = (float*)maskBuf;
    customCudaPoolRowsSumNCHW((const float*)maskFloatBuf,maskSumBuf,batchSize,1,nnXLen*nnYLen,1.0,cudaHandles->stream);
    CUDA_ERR("sumMask",cudaPeekAtLastError());
  }
  else {
    customCudaCopyFromHalf((const half*)maskBuf,maskFloatBuf,batchSize*nnXLen*nnYLen,cudaHandles->stream);
    CUDA_ERR("copyMaskFromHalf",cudaPeekAtLastError());
    customCudaPoolRowsSumNCHW((const float*)maskFloatBuf,maskSumBuf,batchSize,1,nnXLen*nnYLen,1.0,cudaHandles->stream);
    CUDA_ERR("sumMask",cudaPeekAtLastError());
  }
}


//------------------------------------------------------------------------------

struct PolicyHead {
  const string name;
  const int version;
  const int nnXLen;
  const int nnYLen;
  const int p1Channels;
  const int g1Channels;
  const int p2Channels;
  const bool usingFP16;
  const bool usingNHWC;

  const ConvLayer p1Conv;
  const ConvLayer g1Conv;
  const BatchNormLayer g1BN;
  const MatMulLayer gpoolToBiasMul;
  const BatchNormLayer p1BN;
  const ConvLayer p2Conv;
  const MatMulLayer gpoolToPassMul;

  PolicyHead() = delete;
  PolicyHead(const PolicyHead&) = delete;
  PolicyHead& operator=(const PolicyHead&) = delete;

  PolicyHead(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const PolicyHeadDesc* desc,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ) :
    name(desc->name),
    version(desc->version),
    nnXLen(nnX),
    nnYLen(nnY),
    p1Channels(desc->p1Conv.outChannels),
    g1Channels(desc->g1Conv.outChannels),
    p2Channels(desc->p2Conv.outChannels),
    usingFP16(useFP16),
    usingNHWC(useNHWC),
    p1Conv(cudaHandles,manager,&desc->p1Conv,useFP16,useNHWC),
    g1Conv(cudaHandles,manager,&desc->g1Conv,useFP16,useNHWC),
    g1BN(cudaHandles,&desc->g1BN,&desc->g1Activation,nnX,nnY,useFP16,useNHWC),
    gpoolToBiasMul(cudaHandles,&desc->gpoolToBiasMul,false),
    p1BN(cudaHandles,&desc->p1BN,&desc->p1Activation,nnX,nnY,false,useNHWC),
    p2Conv(cudaHandles,manager,&desc->p2Conv,false,useNHWC),
    gpoolToPassMul(cudaHandles,&desc->gpoolToPassMul,false)
  {
  }

  ~PolicyHead()
  {
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;

    b = p1Conv.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = g1Conv.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = gpoolToBiasMul.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = p2Conv.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = gpoolToPassMul.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = sizeof(float)*batchSize*g1Channels*nnXLen*nnYLen;
    bytes = std::max(bytes,b);

    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* maskBuf,
    float* maskFloatBuf,
    float* maskSumBuf,
    void* trunkBuf,
    float* policyBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {

    SizedBuf<void*> p1Out(scratch->allocator, scratch->getBufSizeXYFloat(p1Channels)); //Need to hold floats, not just halfs
    SizedBuf<void*> p1Out2(scratch->allocator, scratch->getBufSizeXYFloat(p1Channels)); //Need to hold floats, not just halfs
    SizedBuf<void*> g1Out(scratch->allocator, scratch->getBufSizeXY(g1Channels));
    SizedBuf<void*> g1Out2(scratch->allocator, scratch->getBufSizeXY(g1Channels));
    SizedBuf<void*> g1Concat(scratch->allocator, scratch->getBufSizeFloat(g1Channels*3));
    SizedBuf<void*> g1Bias(scratch->allocator, scratch->getBufSizeFloat(p1Channels));
    SizedBuf<void*> p2Out(scratch->allocator, scratch->getBufSizeXYFloat(p2Channels));
    SizedBuf<void*> g1Pass(scratch->allocator, scratch->getBufSizeFloat(p2Channels));

    p1Conv.apply(cudaHandles,batchSize,false,trunkBuf,p1Out.buf,workspaceBuf,workspaceBytes);
    g1Conv.apply(cudaHandles,batchSize,false,trunkBuf,g1Out.buf,workspaceBuf,workspaceBytes);
    g1BN.apply(cudaHandles,batchSize,g1Out.buf,maskBuf,g1Out2.buf);

    if(!usingFP16) {
      if(!usingNHWC)
        customCudaPoolRowsGPoolNCHW((const float*)g1Out2.buf,(float*)g1Concat.buf,batchSize,g1Channels,nnXLen*nnYLen,maskFloatBuf,maskSumBuf,cudaHandles->stream);
      else
        customCudaPoolRowsGPoolNHWC((const float*)g1Out2.buf,(float*)g1Concat.buf,batchSize,nnXLen*nnYLen,g1Channels,maskFloatBuf,maskSumBuf,cudaHandles->stream);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }
    else {
      customCudaCopyFromHalf((const half*)g1Out2.buf,(float*)workspaceBuf,batchSize*g1Channels*nnXLen*nnYLen,cudaHandles->stream);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      if(!usingNHWC)
        customCudaPoolRowsGPoolNCHW((const float*)workspaceBuf,(float*)g1Concat.buf,batchSize,g1Channels,nnXLen*nnYLen,maskFloatBuf,maskSumBuf,cudaHandles->stream);
      else
        customCudaPoolRowsGPoolNHWC((const float*)workspaceBuf,(float*)g1Concat.buf,batchSize,nnXLen*nnYLen,g1Channels,maskFloatBuf,maskSumBuf,cudaHandles->stream);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }

    gpoolToBiasMul.apply(cudaHandles,scratch,batchSize,g1Concat.buf,g1Bias.buf,workspaceBuf,workspaceBytes);

    #ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint4D(string("p1 pre-gpool-sum"), p1Out.buf, batchSize, p1Channels, nnXLen, nnYLen, usingNHWC, usingFP16);
    CudaUtils::debugPrint4D(string("g1 pre-gpool"), g1Out.buf, batchSize, g1Channels, nnXLen, nnYLen, usingNHWC, usingFP16);
    CudaUtils::debugPrint2D(string("g1 pooled"), g1Concat.buf, batchSize, g1Channels*3, usingFP16);
    CudaUtils::debugPrint2D(string("g1 biases"), g1Bias.buf, batchSize, p1Channels, usingFP16);
    #endif

    float* p1OutBufA;
    float* p1OutBufB;
    if(!usingFP16) {
      p1OutBufA = (float*)p1Out.buf;
      p1OutBufB = (float*)p1Out2.buf;
    }
    else {
      customCudaCopyFromHalf((const half*)p1Out.buf,(float*)p1Out2.buf,batchSize*p1Channels*nnXLen*nnYLen,cudaHandles->stream);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      p1OutBufA = (float*)p1Out2.buf;
      p1OutBufB = (float*)p1Out.buf;
    }

    if(!usingNHWC)
      customCudaAddNCBiasInplaceNCHW(p1OutBufA,(float*)g1Bias.buf,batchSize,p1Channels,nnXLen*nnYLen,cudaHandles->stream);
    else
      customCudaAddNCBiasInplaceNHWC(p1OutBufA,(float*)g1Bias.buf,batchSize,nnXLen*nnYLen,p1Channels,cudaHandles->stream);
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());

    p1BN.apply(cudaHandles,batchSize,p1OutBufA,maskFloatBuf,p1OutBufB);
    p2Conv.apply(cudaHandles,batchSize,false,p1OutBufB,(float*)p2Out.buf,workspaceBuf,workspaceBytes);

    gpoolToPassMul.apply(cudaHandles,scratch,batchSize,g1Concat.buf,g1Pass.buf,workspaceBuf,workspaceBytes);

    #ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint4D(string("p1 after-gpool-sum"), p1Out.buf, batchSize, p1Channels, nnXLen, nnYLen, usingNHWC, usingFP16);
    CudaUtils::debugPrint4D(string("p2"), p2Out.buf, batchSize, p2Channels, nnXLen, nnYLen, usingNHWC, usingFP16);
    CudaUtils::debugPrint2D(string("p2pass"), g1Pass.buf, batchSize, 1, usingFP16);
    #endif

    customCudaChannelConcat(
      (float*)p2Out.buf,(float*)g1Pass.buf,policyBuf,
      nnXLen*nnYLen,
      1,
      batchSize,
      cudaHandles->stream
    );
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());

  }

};

//------------------------------------------------------------------------------

struct ValueHead {
  const string name;
  const int version;
  const int nnXLen;
  const int nnYLen;
  const int v1Channels;
  const int v2Channels;
  const int valueChannels;
  const int scoreValueChannels;
  const int ownershipChannels;
  const bool usingFP16;
  const bool usingNHWC;

  const ConvLayer v1Conv;
  const BatchNormLayer v1BN;
  const MatMulLayer v2Mul;
  const MatBiasLayer v2Bias;
  const MatMulLayer v3Mul;
  const MatBiasLayer v3Bias;
  const MatMulLayer sv3Mul;
  const MatBiasLayer sv3Bias;
  const ConvLayer vOwnershipConv;

  ValueHead() = delete;
  ValueHead(const ValueHead&) = delete;
  ValueHead& operator=(const ValueHead&) = delete;

  ValueHead(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const ValueHeadDesc* desc,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ) :
    name(desc->name),
    version(desc->version),
    nnXLen(nnX),
    nnYLen(nnY),
    v1Channels(desc->v1Conv.outChannels),
    v2Channels(desc->v2Mul.outChannels),
    valueChannels(desc->v3Mul.outChannels),
    scoreValueChannels(desc->sv3Mul.outChannels),
    ownershipChannels(desc->vOwnershipConv.outChannels),
    usingFP16(useFP16),
    usingNHWC(useNHWC),
    v1Conv(cudaHandles,manager,&desc->v1Conv,useFP16,useNHWC),
    v1BN(cudaHandles,&desc->v1BN,&desc->v1Activation,nnX,nnY,useFP16,useNHWC),
    v2Mul(cudaHandles,&desc->v2Mul,false),
    v2Bias(cudaHandles,&desc->v2Bias,false,desc->v2Activation.activation),
    v3Mul(cudaHandles,&desc->v3Mul,false),
    v3Bias(cudaHandles,&desc->v3Bias,false,ACTIVATION_IDENTITY),
    sv3Mul(cudaHandles,&desc->sv3Mul,false),
    sv3Bias(cudaHandles,&desc->sv3Bias,false,ACTIVATION_IDENTITY),
    vOwnershipConv(cudaHandles,manager,&desc->vOwnershipConv,useFP16,useNHWC)
  {
  }

  ~ValueHead()
  {
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;

    b = v1Conv.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = v2Mul.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = v3Mul.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = sizeof(float)*batchSize*v1Channels*nnXLen*nnYLen;
    bytes = std::max(bytes,b);

    b = sv3Mul.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = vOwnershipConv.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = sizeof(float)*batchSize*ownershipChannels*nnXLen*nnYLen;
    bytes = std::max(bytes,b);

    return bytes;
  }


  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* maskBuf,
    float* maskSumBuf,
    void* trunkBuf,
    float* valueBuf,
    float* scoreValueBuf,
    void* ownershipBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    SizedBuf<void*> v1Out(scratch->allocator, scratch->getBufSizeXY(v1Channels));
    SizedBuf<void*> v1Out2(scratch->allocator, scratch->getBufSizeXY(v1Channels));
    SizedBuf<void*> v1Mean(scratch->allocator, scratch->getBufSizeFloat(v1Channels*3));
    SizedBuf<void*> v2Out(scratch->allocator, scratch->getBufSizeFloat(v2Channels));
    SizedBuf<void*> ownershipScratch(scratch->allocator, scratch->getBufSizeXYFloat(ownershipChannels));

    v1Conv.apply(cudaHandles,batchSize,false,trunkBuf,v1Out.buf,workspaceBuf,workspaceBytes);
    v1BN.apply(cudaHandles,batchSize,v1Out.buf,maskBuf,v1Out2.buf);

    void* bufToBePooled = v1Out2.buf;
    if(usingFP16) {
      customCudaCopyFromHalf((const half*)v1Out2.buf,(float*)workspaceBuf,batchSize*v1Channels*nnXLen*nnYLen,cudaHandles->stream);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      bufToBePooled = workspaceBuf;
    }

    if(!usingNHWC)
      customCudaValueHeadPoolNCHW((float*)bufToBePooled,(float*)v1Mean.buf,batchSize,v1Channels,nnXLen*nnYLen,maskSumBuf,cudaHandles->stream);
    else
      customCudaValueHeadPoolNHWC((const float*)bufToBePooled,(float*)v1Mean.buf,batchSize,nnXLen*nnYLen,v1Channels,maskSumBuf,cudaHandles->stream);
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());

    v2Mul.apply(cudaHandles,scratch,batchSize,v1Mean.buf,v2Out.buf,workspaceBuf,workspaceBytes);
    v2Bias.apply(cudaHandles,batchSize,v2Out.buf);
    v3Mul.apply(cudaHandles,scratch,batchSize,v2Out.buf,valueBuf,workspaceBuf,workspaceBytes);
    v3Bias.apply(cudaHandles,batchSize,valueBuf);

    sv3Mul.apply(cudaHandles,scratch,batchSize,v2Out.buf,scoreValueBuf,workspaceBuf,workspaceBytes);
    sv3Bias.apply(cudaHandles,batchSize,scoreValueBuf);

    #ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint4D(string("v1"), v1Out.buf, batchSize, v1Channels, nnXLen, nnYLen, usingNHWC, usingFP16);
    CudaUtils::debugPrint2D(string("v1 pooled"), v1Mean.buf, batchSize, v1Channels, usingFP16);
    CudaUtils::debugPrint2D(string("v2"), v2Out.buf, batchSize, v1Channels, usingFP16);
    #endif

    if(!usingFP16) {
      vOwnershipConv.apply(cudaHandles,batchSize,false,v1Out2.buf,ownershipBuf,workspaceBuf,workspaceBytes);
    }
    else {
      vOwnershipConv.apply(cudaHandles,batchSize,false,v1Out2.buf,ownershipScratch.buf,workspaceBuf,workspaceBytes);
      customCudaCopyFromHalf((const half*)ownershipScratch.buf,(float*)ownershipBuf,batchSize*ownershipChannels*nnXLen*nnYLen,cudaHandles->stream);
      CUDA_ERR("vOwnership copy",cudaPeekAtLastError());
    }

  }

};

//------------------------------------------------------------------------------

struct Model {
  const string name;
  const int version;
  const int maxBatchSize;
  const int nnXLen;
  const int nnYLen;
  const int numInputChannels;
  const int numInputGlobalChannels;
  const int numValueChannels;
  const int numScoreValueChannels;
  const int numOwnershipChannels;
  const bool usingFP16;
  const bool usingNHWC;
  const bool inputsUsingNHWC;

  std::unique_ptr<Trunk> trunk;
  std::unique_ptr<PolicyHead> policyHead;
  std::unique_ptr<ValueHead> valueHead;
  std::unique_ptr<CudnnManager> manager;

  Model() = delete;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  Model(
    CudaHandles* cudaHandles,
    const ModelDesc* desc,
    int maxBatchSz,
    int nnX,
    int nnY,
    bool inputsUseNHWC,
    bool useFP16,
    bool useNHWC
  ) :
    name(desc->name),
    version(desc->version),
    maxBatchSize(maxBatchSz),
    nnXLen(nnX),
    nnYLen(nnY),
    numInputChannels(desc->numInputChannels),
    numInputGlobalChannels(desc->numInputGlobalChannels),
    numValueChannels(desc->numValueChannels),
    numScoreValueChannels(desc->numScoreValueChannels),
    numOwnershipChannels(desc->numOwnershipChannels),
    usingFP16(useFP16),
    usingNHWC(useNHWC),
    inputsUsingNHWC(inputsUseNHWC)
  {
    if(nnXLen > NNPos::MAX_BOARD_LEN)
      throw StringError(Global::strprintf("nnXLen (%d) is greater than NNPos::MAX_BOARD_LEN (%d)",
        nnXLen, NNPos::MAX_BOARD_LEN
      ));
    if(nnYLen > NNPos::MAX_BOARD_LEN)
      throw StringError(Global::strprintf("nnYLen (%d) is greater than NNPos::MAX_BOARD_LEN (%d)",
        nnYLen, NNPos::MAX_BOARD_LEN
      ));

    int numFeatures = NNModelVersion::getNumSpatialFeatures(version);
    if(numInputChannels != numFeatures)
      throw StringError(Global::strprintf("Neural net numInputChannels (%d) was not the expected number based on version (%d)",
        numInputChannels, numFeatures
      ));
    int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(version);
    if(numInputGlobalChannels != numGlobalFeatures)
      throw StringError(Global::strprintf("Neural net numInputGlobalChannels (%d) was not the expected number based on version (%d)",
        numInputGlobalChannels, numGlobalFeatures
      ));

    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,numInputChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,numInputGlobalChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,numValueChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,numScoreValueChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,numOwnershipChannels);

    manager = std::make_unique<CudnnManager>(name, maxBatchSize, nnXLen, nnYLen);
    TransformerPlanCursor planCursor(cudaHandles->transformerPlan.get());
    trunk = std::make_unique<Trunk>(
      cudaHandles,manager.get(),&desc->trunk,nnXLen,nnYLen,
      inputsUseNHWC,useFP16,useNHWC,&planCursor);
    policyHead = std::make_unique<PolicyHead>(cudaHandles,manager.get(),&desc->policyHead,nnXLen,nnYLen,useFP16,useNHWC);
    valueHead = std::make_unique<ValueHead>(cudaHandles,manager.get(),&desc->valueHead,nnXLen,nnYLen,useFP16,useNHWC);
  }

  ~Model()
  {
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;

    b = trunk->requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = policyHead->requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = valueHead->requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);

    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    bool requireExactNNLen,

    void* inputBuf,
    void* inputGlobalBuf,

    float* policyBuf,

    float* valueBuf,
    float* scoreValueBuf,
    void* ownershipBuf,

    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    SizedBuf<void*> mask(scratch->allocator, scratch->getBufSizeXY(1));
    SizedBuf<void*> maskFloat(scratch->allocator, scratch->getBufSizeXYFloat(1));
    SizedBuf<void*> maskSum(scratch->allocator, scratch->getBufSizeFloat(1));

    void* maskBuf = mask.buf;
    float* maskFloatBuf = (float*)maskFloat.buf;
    float* maskSumBuf = (float*)maskSum.buf;

    if(!usingFP16) {
      if(inputsUsingNHWC)
        customCudaChannel0ExtractNHWC((const float*)inputBuf, (float*)maskBuf, batchSize, nnXLen*nnYLen, numInputChannels,cudaHandles->stream);
      else
        customCudaChannel0ExtractNCHW((const float*)inputBuf, (float*)maskBuf, batchSize, numInputChannels, nnXLen*nnYLen,cudaHandles->stream);
      CUDA_ERR("modelExtractMask",cudaPeekAtLastError());
    }
    else {
      if(inputsUsingNHWC)
        customCudaChannel0ExtractNHWC((const half*)inputBuf, (half*)maskBuf, batchSize, nnXLen*nnYLen, numInputChannels,cudaHandles->stream);
      else
        customCudaChannel0ExtractNCHW((const half*)inputBuf, (half*)maskBuf, batchSize, numInputChannels, nnXLen*nnYLen,cudaHandles->stream);
      CUDA_ERR("modelExtractMask",cudaPeekAtLastError());
    }

    fillMaskFloatBufAndMaskSumBuf(cudaHandles,maskBuf,maskFloatBuf,maskSumBuf,usingFP16,batchSize,nnXLen,nnYLen);

    //Don't do any masking if we know the board is exactly the desired size
    if(requireExactNNLen) {
      //Set to NULL to signal downstream that this buf doesn't need to be used
      maskBuf = NULL;
      maskFloatBuf = NULL;
      //The global pooling structures need this no matter what, for normalizing based on this and its sqrt.
      //maskSumBuf = NULL;
    }

    #ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint4D(string("Initial bin features"), inputBuf, batchSize, trunk->initialConv->inChannels, nnXLen, nnYLen, inputsUsingNHWC, usingFP16);
    CudaUtils::debugPrint2D(string("Initial global features"), inputGlobalBuf, batchSize, trunk->initialMatMul->inChannels, usingFP16);
    #endif

    SizedBuf<void*> trunkBuf(scratch->allocator, scratch->getBufSizeXY(trunk->trunkNumChannels));

    trunk->apply(
      cudaHandles,
      scratch,
      batchSize,
      inputBuf,
      inputGlobalBuf,
      maskBuf,
      maskSumBuf,
      trunkBuf.buf,
      workspaceBuf,
      workspaceBytes
    );
    policyHead->apply(
      cudaHandles,
      scratch,
      batchSize,
      maskBuf,
      maskFloatBuf,
      maskSumBuf,
      trunkBuf.buf,
      policyBuf,
      workspaceBuf,
      workspaceBytes
    );
    valueHead->apply(
      cudaHandles,
      scratch,
      batchSize,
      maskBuf,
      maskSumBuf,
      trunkBuf.buf,
      valueBuf,
      scoreValueBuf,
      ownershipBuf,
      workspaceBuf,
      workspaceBytes
    );
  }

};


//------------------------------------------------------------------------------

struct LoadedModel {
  ModelDesc modelDesc;

  LoadedModel(const string& fileName, const string& expectedSha256) {
    ModelDesc::loadFromFileMaybeGZipped(fileName,modelDesc,expectedSha256);
  }

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

LoadedModel* NeuralNet::loadModelFile(const string& file, const string& expectedSha256) {
  LoadedModel* loadedModel = new LoadedModel(file,expectedSha256);
  return loadedModel;
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}

string NeuralNet::getModelName(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc.name;
}

int NeuralNet::getModelVersion(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc.version;
}

Rules NeuralNet::getSupportedRules(const LoadedModel* loadedModel, const Rules& desiredRules, bool& supported) {
  return loadedModel->modelDesc.getSupportedRules(desiredRules, supported);
}

const ModelDesc& NeuralNet::getModelDesc(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc;
}
//------------------------------------------------------------------------------

struct Buffers {
  //All of these are device pointers

  float* inputBufFloat;
  void* inputBuf;
  float* inputGlobalBufFloat;
  void* inputGlobalBuf;
  size_t inputBufBytesFloat;
  size_t inputBufBytes;
  size_t inputGlobalBufBytesFloat;
  size_t inputGlobalBufBytes;

  float* policyBuf;
  size_t policyBufBytes;

  float* valueBuf;
  size_t valueBufBytes;
  float* scoreValueBuf;
  size_t scoreValueBufBytes;
  void* ownershipBuf;
  size_t ownershipBufBytes;

  void* workspaceBuf;
  size_t workspaceBytes;

  Buffers() = delete;
  Buffers(const Buffers&) = delete;
  Buffers& operator=(const Buffers&) = delete;

  Buffers(CudaHandles* cudaHandles, const Model& m, const ScratchBuffers& scratch) {
    size_t batchXYFloatBytes = (size_t)scratch.batchXYFloatBytes;
    size_t batchFloatBytes = (size_t)scratch.batchFloatBytes;
    size_t batchXYBytes = (size_t)scratch.batchXYBytes;
    size_t batchBytes = (size_t)scratch.batchBytes;

    inputBufBytesFloat = m.numInputChannels * batchXYFloatBytes;
    inputBufBytes = m.numInputChannels * batchXYBytes;
    inputGlobalBufBytesFloat = m.numInputGlobalChannels * batchFloatBytes;
    inputGlobalBufBytes = m.numInputGlobalChannels * batchBytes;

    CUDA_ERR("Buffers", cudaMalloc(reinterpret_cast<void**>(&inputBufFloat), inputBufBytesFloat));
    CUDA_ERR("Buffers",cudaMalloc(&inputBuf, inputBufBytes));
    CUDA_ERR("Buffers",cudaMalloc(reinterpret_cast<void**>(&inputGlobalBufFloat), inputGlobalBufBytesFloat));
    CUDA_ERR("Buffers",cudaMalloc(&inputGlobalBuf, inputGlobalBufBytes));

    policyBufBytes = m.policyHead->p2Channels * (batchXYFloatBytes + batchFloatBytes);
    CUDA_ERR("Buffers",cudaMalloc(reinterpret_cast<void**>(&policyBuf), policyBufBytes));
    assert(m.policyHead->p2Channels == 1);

    valueBufBytes = m.valueHead->valueChannels * batchFloatBytes;
    CUDA_ERR("Buffers",cudaMalloc(reinterpret_cast<void**>(&valueBuf), valueBufBytes));

    scoreValueBufBytes = m.valueHead->scoreValueChannels * batchFloatBytes;
    CUDA_ERR("Buffers",cudaMalloc(reinterpret_cast<void**>(&scoreValueBuf), scoreValueBufBytes));

    //This buf is used for both an intermdiate fp16 result in fp16 mode, and ALSO the final fp32 output, so always must be fp32-sized
    ownershipBufBytes = m.valueHead->ownershipChannels * batchXYFloatBytes;
    CUDA_ERR("Buffers",cudaMalloc(&ownershipBuf, ownershipBufBytes));

    //In theory the requiredWorkspaceBytes calls could give us values non-monotone in batch size
    //such as if the convolution algorithm changes between batch size 1 and larger.
    //So we call it for all the batch sizes.
    size_t bytes = 0;
    size_t b;
    for(int batchSize = 1; batchSize <= m.maxBatchSize; batchSize++) {
      b = m.requiredWorkspaceBytes(cudaHandles,batchSize);
      bytes = std::max(bytes,b);
    }

    CUDA_ERR("Buffers",cudaMalloc(&workspaceBuf, bytes));
    workspaceBytes = bytes;
  }

  ~Buffers() {
    cudaFree(inputBufFloat);
    cudaFree(inputBuf);
    cudaFree(inputGlobalBufFloat);
    cudaFree(inputGlobalBuf);

    cudaFree(policyBuf);

    cudaFree(valueBuf);
    cudaFree(scoreValueBuf);
    cudaFree(ownershipBuf);

    cudaFree(workspaceBuf);
  }

};

//------------------------------------------------------------------------------

struct ComputeContext {
  int nnXLen;
  int nnYLen;
  enabled_t useFP16Mode;
  enabled_t useNHWCMode;
  bool useINT8;
};

ComputeContext* NeuralNet::createComputeContext(
  const std::vector<int>& gpuIdxs,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const string& openCLTunerFile,
  const string& homeDataDirOverride,
  bool openCLReTunePerBoardSize,
  enabled_t useFP16Mode,
  enabled_t useNHWCMode,
  bool useINT8,
  const LoadedModel* loadedModel
) {
  (void)gpuIdxs;
  (void)openCLTunerFile;
  (void)homeDataDirOverride;
  (void)openCLReTunePerBoardSize;
  (void)loadedModel;

  const CudaInt8Policy requestedInt8Policy = resolveCudaInt8Policy(
    useINT8,std::getenv("KATAGO_DISABLE_INT8"));
#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
  const bool int8Compiled = true;
#else
  const bool int8Compiled = false;
#endif
  const bool allowINT8 = requestedInt8Policy.enabled && int8Compiled;
  if(logger != NULL) {
    logger->write(
      string("CUDA_INT8_POLICY config=") +
      (requestedInt8Policy.configEnabled ? "1" : "0") +
      " env_disabled=" +
      (requestedInt8Policy.environmentDisabled ? "1" : "0") +
      " compiled=" + (int8Compiled ? "1" : "0") +
      " allowed=" + (allowINT8 ? "1" : "0"));
  }

  ComputeContext* context = new ComputeContext();
  context->nnXLen = nnXLen;
  context->nnYLen = nnYLen;
  context->useFP16Mode = useFP16Mode;
  context->useNHWCMode = useNHWCMode;
  context->useINT8 = allowINT8;
  return context;
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

//------------------------------------------------------------------------------

struct ComputeHandle {
  std::unique_ptr<CudaHandles> cudaHandles;
  std::unique_ptr<Model> model;
  std::unique_ptr<ScratchBuffers> scratch;
  std::unique_ptr<Buffers> buffers;
  const bool usingFP16;
  const int nnXLen;
  const int nnYLen;
  const bool requireExactNNLen;
  const bool inputsUseNHWC;
  const int policySize;

  ComputeHandle(
    const ComputeContext* context,
    const LoadedModel* loadedModel,
    int majorComputeCapability,
    int minorComputeCapability,
    int deviceWarpSize,
    size_t sharedBytesPerBlockOptin,
    int maxBatchSize,
    bool requireExactNNLen_,
    bool inputsUseNHWC_,
    bool useFP16,
    bool useNHWC,
    int streamCount,
    Logger* logger
  ) :
    usingFP16(useFP16),
    nnXLen(context->nnXLen),
    nnYLen(context->nnYLen),
    requireExactNNLen(requireExactNNLen_),
    inputsUseNHWC(inputsUseNHWC_),
    policySize(NNPos::getPolicySize(context->nnXLen, context->nnYLen))
  {
    cudaHandles = std::make_unique<CudaHandles>(majorComputeCapability,minorComputeCapability);
    cudaHandles->logger = logger;
    if(loadedModel->modelDesc.trunk.hasAnyTransformerBlocks() &&
       !loadedModel->modelDesc.onnxHeader.isOnnx) {
      int cudaRuntimeVersion = 0;
      int cudaDriverVersion = 0;
      int cublasVersion = 0;
      CUDA_ERR("ComputeHandle",cudaRuntimeGetVersion(&cudaRuntimeVersion));
      CUDA_ERR("ComputeHandle",cudaDriverGetVersion(&cudaDriverVersion));
      CUBLAS_ERR("ComputeHandle",cublasGetVersion(
        cudaHandles->cublas,&cublasVersion));
      const size_t cudnnVersion = cudnnGetVersion();
      if(logger != NULL) {
        logger->write(
          "CUDA_TRANSFORMER_RUNTIME_ABI runtime=" + Global::intToString(cudaRuntimeVersion) +
          " driver=" + Global::intToString(cudaDriverVersion) +
          " cublas=" + Global::intToString(cublasVersion) +
          " cudnn=" + Global::uint64ToString((uint64_t)cudnnVersion)
        );
      }

      CudaOpRegistry::RuntimeOpContext runtime{};
      runtime.batchSize = maxBatchSize;
      runtime.boardX = nnXLen;
      runtime.boardY = nnYLen;
      runtime.maskMode = requireExactNNLen ? CudaOpRegistry::MaskMode::None :
        CudaOpRegistry::MaskMode::Dense;
      runtime.inputType = useFP16 ? CudaOpRegistry::NumericType::Float16 :
        CudaOpRegistry::NumericType::Float32;
      runtime.outputType = runtime.inputType;
      runtime.computeType = CudaOpRegistry::NumericType::Float32;
      runtime.layout = useNHWC ? CudaOpRegistry::TensorLayout::NHWC :
        CudaOpRegistry::TensorLayout::NCHW;
      runtime.deviceComputeCapability =
        (uint32_t)(majorComputeCapability * 10 + minorComputeCapability);
      runtime.streamCount = (uint32_t)std::max(1,streamCount);
      runtime.runtimeLibraryFingerprint =
        CudaTransformerWinner::makeRuntimeLibraryFingerprint(
          cudaRuntimeVersion,cudaDriverVersion,cublasVersion,cudnnVersion);

      CudaTransformerWinner::DeviceCapability device{};
      device.computeCapability = runtime.deviceComputeCapability;
      device.warpSize = (uint32_t)deviceWarpSize;
      device.sharedBytesPerBlockOptin = sharedBytesPerBlockOptin;
      device.cudaRuntimeVersion = cudaRuntimeVersion;
      device.cudaDriverVersion = cudaDriverVersion;
      device.cublasVersion = cublasVersion;
      device.cudnnVersion = cudnnVersion;
#if defined(KATAGO_ENABLE_RENJU15_FA4_SM120) && KATAGO_ENABLE_RENJU15_FA4_SM120
      device.specializedSm120KernelsAvailable = true;
#endif
      const NeuralNetArchitecture::ArchitectureDesc architecture =
        NeuralNetArchitecture::buildArchitectureDesc(loadedModel->modelDesc);
      cudaHandles->transformerPlan =
        std::make_unique<CudaTransformerWinner::PreparedPlan>(
          CudaTransformerWinner::preparePlan(architecture,runtime,device));
      cudaHandles->configureWinnerExpectations(
        context->useINT8,loadedModel->modelDesc);
    }
    model = std::make_unique<Model>(
      cudaHandles.get(), &(loadedModel->modelDesc), maxBatchSize,
      nnXLen, nnYLen, inputsUseNHWC, useFP16, useNHWC
    );
#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
    cudaHandles->finishEmbeddedInt8MetadataConsumption();
#endif
    cudaHandles->validateWinnerPrepared();
    auto allocateBaseline = [&]() {
      scratch = std::make_unique<ScratchBuffers>(
        maxBatchSize,nnXLen,nnYLen,useFP16);
      buffers = std::make_unique<Buffers>(cudaHandles.get(), *model, *scratch);
    };
    try {
      allocateBaseline();
    }
    catch(...) {
#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
      if(cudaHandles->int8ExperimentPlan) {
        // Optional packed weights may have consumed the margin required by
        // the authoritative FP16 buffers. Destroy the partial baseline state,
        // release all INT8 resources, clear a sticky CUDA allocation error,
        // and retry the baseline exactly once.
        buffers.reset();
        scratch.reset();
        cudaHandles->noteInt8PreparationFailure("baseline-allocation-retry");
        allocateBaseline();
      }
      else
#endif
        throw;
    }
#if defined(KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT) && KATAGO_ENABLE_RENJU15_INT8_EXPERIMENT
    if(cudaHandles->int8ExperimentPlan) {
      const cudaError_t status = scratch->tryPrepareInt8ExperimentScratch(
        maxBatchSize,nnXLen,nnYLen);
      if(status != cudaSuccess) {
        cudaHandles->noteInt8PreparationFailure(
          "persistent-scratch-allocation-failed");
      }
    }
    cudaHandles->commitInt8PreparedResources();
#endif

    //Synchronize after creating buffers and copying all the weights, just in case
    CUDA_ERR("ComputeHandle", cudaStreamSynchronize(cudaHandles->stream));
  }
  ~ComputeHandle() {
    // Model/buffer destructors free device weights, RoPE tables, and prepared
    // CUTLASS state before CudaHandles itself is destroyed. Drain this owned
    // stream first so no queued launch can observe freed state. Destructors
    // cannot report CUDA errors, so this final defensive wait is best-effort.
    if(cudaHandles != nullptr && cudaHandles->stream != NULL)
      (void)cudaStreamSynchronize(cudaHandles->stream);
  }

  ComputeHandle() = delete;
  ComputeHandle(const ComputeHandle&) = delete;
  ComputeHandle& operator=(const ComputeHandle&) = delete;
};

ComputeHandle* NeuralNet::createComputeHandle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  bool requireExactNNLen,
  bool inputsUseNHWC,
  int gpuIdxForThisThread,
  int serverThreadIdx,
  int backendNumThreads) {
  //Use whatever CUDA believes GPU 0 to be.
  if(gpuIdxForThisThread == -1)
    gpuIdxForThisThread = 0;

  CUDA_ERR("createComputeHandle",cudaSetDevice(gpuIdxForThisThread));

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop,gpuIdxForThisThread);

  bool useFP16 = false;
  bool useNHWC = false;
  //Old GPUs - use FP32 and explicitly fail if FP16 enabled
  if(prop.major < 5 || (prop.major == 5 && prop.minor < 3)) {
    if(context->useFP16Mode == enabled_t::True)
      throw StringError("Cuda device versions below 5.3 do not support useFP16=true");
    if(context->useNHWCMode == enabled_t::True)
      useNHWC = true;
  }
  //In theory these GPUs support FP16, so allow if the user wants.
  else if(prop.major < 6) {
    if(context->useFP16Mode == enabled_t::True)
      useFP16 = true;
    if(context->useNHWCMode == enabled_t::True)
      useNHWC = true;
  }
  //On Pascal architecture, default to using FP16 operations
  //Actually, just use FP32 - there's a risk that on certain cards this might just be a lot worse.
  //A user manually fine-tuning for performance can just enable it themselves if they know how.
  else if(prop.major < 7) {
    if(context->useFP16Mode == enabled_t::True)
      useFP16 = true;
    if(context->useNHWCMode == enabled_t::True)
      useNHWC = true;
  }
  //On Volta and higher, use FP16 and NHWC together because we have tensor cores.
  else {
    if(context->useFP16Mode == enabled_t::True || context->useFP16Mode == enabled_t::Auto)
      useFP16 = true;
    if(context->useNHWCMode == enabled_t::True || (context->useNHWCMode == enabled_t::Auto && useFP16))
      useNHWC = true;
  }

  // Transformer matmuls use a channel-contiguous [B,S,C] view. This is also
  // required for the FP32 correctness fallback, where NHWC would otherwise
  // remain disabled under the ordinary auto heuristic.
  if(loadedModel->modelDesc.trunk.hasAnyTransformerBlocks() && !useNHWC) {
    useNHWC = true;
    if(logger != NULL)
      logger->write("Cuda backend: forcing NHWC for transformer trunk");
  }

  if(logger != NULL) {
    logger->write(
      "Cuda backend thread " + Global::intToString(serverThreadIdx) + ": Found GPU " + string(prop.name)
      + " memory " + Global::uint64ToString(prop.totalGlobalMem)
      + " compute capability major " + Global::intToString(prop.major)
      + " minor " + Global::intToString(prop.minor)
    );
    logger->write(
      "Cuda backend thread " + Global::intToString(serverThreadIdx) + ": Model version " + Global::intToString(loadedModel->modelDesc.version) +
      " useFP16 = " + Global::boolToString(useFP16) +
      " useNHWC = " + Global::boolToString(useNHWC)
    );
    logger->write(
      "Cuda backend thread " + Global::intToString(serverThreadIdx) + ": Model name: " + loadedModel->modelDesc.name
    );
  }

  ComputeHandle* gpuHandle = new ComputeHandle(
    context,loadedModel,prop.major,prop.minor,prop.warpSize,
    (size_t)prop.sharedMemPerBlockOptin,maxBatchSize,requireExactNNLen,
    inputsUseNHWC,useFP16,useNHWC,backendNumThreads,logger
  );
  // SDPA plans are selected lazily on the first preflight, after model construction. The evaluator
  // logger therefore outlives every capability-probe message emitted by this handle.
  gpuHandle->cudaHandles->logger = logger;
  if(logger != NULL) {
    logger->write(
      "CUDA_HANDLE_STREAM_ACTIVE marker=owned-nonblocking" +
      string(" serverThread=") + Global::intToString(serverThreadIdx) +
      " device=" + Global::intToString(gpuIdxForThisThread) +
      " stream=" + Global::uint64ToString(
        (uint64_t)(uintptr_t)gpuHandle->cudaHandles->stream
      )
    );
  }
  return gpuHandle;
}

void NeuralNet::freeComputeHandle(ComputeHandle* gpuHandle) {
  delete gpuHandle;
}

bool NeuralNet::isUsingFP16(const ComputeHandle* handle) {
  return handle->usingFP16;
}

//------------------------------------------------------------------------------

void NeuralNet::printDevices() {
  int numDevices = 0;
  cudaGetDeviceCount(&numDevices);
  for(int i = 0; i<numDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    cout << "Found CUDA device " << i << ": " << prop.name << endl;
  }
}


//------------------------------------------------------------------------------

struct InputBuffers {
  int maxBatchSize;

  size_t singleInputElts;
  size_t singleInputBytes;
  size_t singleInputGlobalElts;
  size_t singleInputGlobalBytes;
  size_t singlePolicyResultElts;
  size_t singlePolicyResultBytes;
  size_t singleValueResultElts;
  size_t singleValueResultBytes;
  size_t singleScoreValueResultElts;
  size_t singleScoreValueResultBytes;
  size_t singleOwnershipResultElts;
  size_t singleOwnershipResultBytes;

  size_t userInputBufferBytes;
  size_t userInputGlobalBufferBytes;
  size_t policyResultBufferBytes;
  size_t valueResultBufferBytes;
  size_t scoreValueResultBufferBytes;
  size_t ownershipResultBufferBytes;

  float* userInputBuffer; //Host pointer
  float* userInputGlobalBuffer; //Host pointer

  float* policyResults; //Host pointer
  float* valueResults; //Host pointer
  float* scoreValueResults; //Host pointer
  float* ownershipResults; //Host pointer

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
    const ModelDesc& m = loadedModel->modelDesc;

    maxBatchSize = maxBatchSz;
    singleInputElts = (size_t)m.numInputChannels * nnXLen * nnYLen;
    singleInputBytes = (size_t)m.numInputChannels * nnXLen * nnYLen * sizeof(float);
    singleInputGlobalElts = (size_t)m.numInputGlobalChannels;
    singleInputGlobalBytes = (size_t)m.numInputGlobalChannels * sizeof(float);
    singlePolicyResultElts = (size_t)(1 + nnXLen * nnYLen);
    singlePolicyResultBytes = (size_t)(1 + nnXLen * nnYLen) * sizeof(float);
    singleValueResultElts = (size_t)m.numValueChannels;
    singleValueResultBytes = (size_t)m.numValueChannels * sizeof(float);
    singleScoreValueResultElts = (size_t)m.numScoreValueChannels;
    singleScoreValueResultBytes = (size_t)m.numScoreValueChannels * sizeof(float);
    singleOwnershipResultElts = (size_t)m.numOwnershipChannels * nnXLen * nnYLen;
    singleOwnershipResultBytes = (size_t)m.numOwnershipChannels * nnXLen * nnYLen * sizeof(float);

    assert(NNModelVersion::getNumSpatialFeatures(m.version) == m.numInputChannels);
    assert(NNModelVersion::getNumGlobalFeatures(m.version) == m.numInputGlobalChannels);

    userInputBufferBytes = (size_t)m.numInputChannels * maxBatchSize * nnXLen * nnYLen * sizeof(float);
    userInputGlobalBufferBytes = (size_t)m.numInputGlobalChannels * maxBatchSize * sizeof(float);
    policyResultBufferBytes = (size_t)maxBatchSize * (1 + nnXLen * nnYLen) * sizeof(float);
    valueResultBufferBytes = (size_t)maxBatchSize * m.numValueChannels * sizeof(float);
    scoreValueResultBufferBytes = (size_t)maxBatchSize * m.numScoreValueChannels * sizeof(float);
    ownershipResultBufferBytes = (size_t)maxBatchSize * nnXLen * nnYLen * m.numOwnershipChannels * sizeof(float);

    userInputBuffer = new float[(size_t)m.numInputChannels * maxBatchSize * nnXLen * nnYLen];
    userInputGlobalBuffer = new float[(size_t)m.numInputGlobalChannels * maxBatchSize];

    policyResults = new float[(size_t)maxBatchSize * (1 + nnXLen * nnYLen)];
    valueResults = new float[(size_t)maxBatchSize * m.numValueChannels];

    scoreValueResults = new float[(size_t)maxBatchSize * m.numScoreValueChannels];
    ownershipResults = new float[(size_t)maxBatchSize * nnXLen * nnYLen * m.numOwnershipChannels];
  }

  ~InputBuffers() {
    delete[] userInputBuffer;
    delete[] userInputGlobalBuffer;
    delete[] policyResults;
    delete[] valueResults;
    delete[] scoreValueResults;
    delete[] ownershipResults;
  }

  InputBuffers() = delete;
  InputBuffers(const InputBuffers&) = delete;
  InputBuffers& operator=(const InputBuffers&) = delete;

};

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  return new InputBuffers(loadedModel,maxBatchSize,nnXLen,nnYLen);
}
void NeuralNet::freeInputBuffers(InputBuffers* inputBuffers) {
  delete inputBuffers;
}

//---------------------------------------------------------------------------------------


void NeuralNet::getOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs,
  float* outputPolicys
) {
  assert(numBatchEltsFilled <= inputBuffers->maxBatchSize);
  assert(numBatchEltsFilled > 0);
  int batchSize = numBatchEltsFilled;
  int nnXLen = gpuHandle->nnXLen;
  int nnYLen = gpuHandle->nnYLen;
  int version = gpuHandle->model->version;

  int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(version);
  int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(version);
  assert(numSpatialFeatures == gpuHandle->model->numInputChannels);
  assert(numSpatialFeatures * nnXLen * nnYLen == inputBuffers->singleInputElts);
  assert(numGlobalFeatures == inputBuffers->singleInputGlobalElts);

  for(int nIdx = 0; nIdx<batchSize; nIdx++) {
    float* rowSpatialInput = inputBuffers->userInputBuffer + (inputBuffers->singleInputElts * nIdx);
    float* rowGlobalInput = inputBuffers->userInputGlobalBuffer + (inputBuffers->singleInputGlobalElts * nIdx);

    const float* rowGlobal = inputBufs[nIdx]->rowGlobal;
    const float* rowSpatial = inputBufs[nIdx]->rowSpatial;
    std::copy(rowGlobal,rowGlobal+numGlobalFeatures,rowGlobalInput);
    SymmetryHelpers::copyInputsWithSymmetry(rowSpatial, rowSpatialInput, 1, nnYLen, nnXLen, numSpatialFeatures, gpuHandle->inputsUseNHWC, inputBufs[nIdx]->symmetry);
  }

  Buffers* buffers = gpuHandle->buffers.get();
  ScratchBuffers* scratch = gpuHandle->scratch.get();

  if(!gpuHandle->usingFP16) {
    assert(inputBuffers->userInputBufferBytes == buffers->inputBufBytes);
    assert(inputBuffers->userInputGlobalBufferBytes == buffers->inputGlobalBufBytes);
    assert(inputBuffers->policyResultBufferBytes == buffers->policyBufBytes);
    assert(inputBuffers->valueResultBufferBytes == buffers->valueBufBytes);
    assert(inputBuffers->singleInputBytes == inputBuffers->singleInputElts*4);
    assert(inputBuffers->singleInputGlobalBytes == inputBuffers->singleInputGlobalElts*4);
    assert(inputBuffers->singlePolicyResultElts == gpuHandle->policySize);
    assert(inputBuffers->singlePolicyResultBytes == gpuHandle->policySize * sizeof(float));
    assert(inputBuffers->scoreValueResultBufferBytes == buffers->scoreValueBufBytes);
    assert(inputBuffers->ownershipResultBufferBytes == buffers->ownershipBufBytes);
    assert(inputBuffers->singleOwnershipResultElts == nnXLen*nnYLen);
    assert(inputBuffers->singleOwnershipResultBytes == nnXLen*nnYLen * sizeof(float));

    CUDA_ERR("getOutput",cudaMemcpyAsync(buffers->inputBuf, inputBuffers->userInputBuffer, inputBuffers->singleInputBytes*batchSize, cudaMemcpyHostToDevice, gpuHandle->cudaHandles->stream));
    CUDA_ERR("getOutput",cudaMemcpyAsync(buffers->inputGlobalBuf, inputBuffers->userInputGlobalBuffer, inputBuffers->singleInputGlobalBytes*batchSize, cudaMemcpyHostToDevice, gpuHandle->cudaHandles->stream));
  }
  else {
    assert(inputBuffers->userInputBufferBytes == buffers->inputBufBytesFloat);
    assert(inputBuffers->userInputGlobalBufferBytes == buffers->inputGlobalBufBytesFloat);
    assert(inputBuffers->policyResultBufferBytes == buffers->policyBufBytes);
    assert(inputBuffers->valueResultBufferBytes == buffers->valueBufBytes);
    assert(inputBuffers->userInputBufferBytes == buffers->inputBufBytes*2);
    assert(inputBuffers->userInputGlobalBufferBytes == buffers->inputGlobalBufBytes*2);
    assert(inputBuffers->singleInputBytes == inputBuffers->singleInputElts*4);
    assert(inputBuffers->singleInputGlobalBytes == inputBuffers->singleInputGlobalElts*4);
    assert(inputBuffers->singlePolicyResultElts == gpuHandle->policySize);
    assert(inputBuffers->singlePolicyResultBytes == gpuHandle->policySize * sizeof(float));
    assert(inputBuffers->scoreValueResultBufferBytes == buffers->scoreValueBufBytes);
    assert(inputBuffers->ownershipResultBufferBytes == buffers->ownershipBufBytes);
    assert(inputBuffers->singleOwnershipResultElts == nnXLen*nnYLen);
    assert(inputBuffers->singleOwnershipResultBytes == nnXLen*nnYLen * sizeof(float));

    CUDA_ERR("getOutput",cudaMemcpyAsync(buffers->inputBufFloat, inputBuffers->userInputBuffer, inputBuffers->singleInputBytes*batchSize, cudaMemcpyHostToDevice, gpuHandle->cudaHandles->stream));
    CUDA_ERR("getOutput",cudaMemcpyAsync(buffers->inputGlobalBufFloat, inputBuffers->userInputGlobalBuffer, inputBuffers->singleInputGlobalBytes*batchSize, cudaMemcpyHostToDevice, gpuHandle->cudaHandles->stream));

    customCudaCopyToHalf((const float*)buffers->inputBufFloat,(half*)buffers->inputBuf,inputBuffers->singleInputElts*batchSize,gpuHandle->cudaHandles->stream);
    CUDA_ERR("getOutput",cudaPeekAtLastError());
    customCudaCopyToHalf((const float*)buffers->inputGlobalBufFloat,(half*)buffers->inputGlobalBuf,inputBuffers->singleInputGlobalElts*batchSize,gpuHandle->cudaHandles->stream);
    CUDA_ERR("getOutput",cudaPeekAtLastError());
  }

  gpuHandle->model->apply(
    gpuHandle->cudaHandles.get(),
    scratch,
    batchSize,
    gpuHandle->requireExactNNLen,

    buffers->inputBuf,
    buffers->inputGlobalBuf,

    buffers->policyBuf,

    buffers->valueBuf,
    buffers->scoreValueBuf,
    buffers->ownershipBuf,

    buffers->workspaceBuf,
    buffers->workspaceBytes
  );

  CUDA_ERR("getOutput",cudaMemcpyAsync(inputBuffers->policyResults, buffers->policyBuf, inputBuffers->singlePolicyResultBytes*batchSize, cudaMemcpyDeviceToHost, gpuHandle->cudaHandles->stream));
  CUDA_ERR("getOutput",cudaMemcpyAsync(inputBuffers->valueResults, buffers->valueBuf, inputBuffers->singleValueResultBytes*batchSize, cudaMemcpyDeviceToHost, gpuHandle->cudaHandles->stream));
  CUDA_ERR("getOutput",cudaMemcpyAsync(inputBuffers->scoreValueResults, buffers->scoreValueBuf, inputBuffers->singleScoreValueResultBytes*batchSize, cudaMemcpyDeviceToHost, gpuHandle->cudaHandles->stream));
  CUDA_ERR("getOutput",cudaMemcpyAsync(inputBuffers->ownershipResults, buffers->ownershipBuf, inputBuffers->singleOwnershipResultBytes*batchSize, cudaMemcpyDeviceToHost, gpuHandle->cudaHandles->stream));
  CUDA_ERR("getOutput",cudaStreamSynchronize(gpuHandle->cudaHandles->stream));

  assert(outputs.size() == batchSize);

  for(int row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];
    assert(output->nnXLen == nnXLen);
    assert(output->nnYLen == nnYLen);

    const float* policySrcBuf = inputBuffers->policyResults + row * gpuHandle->policySize;
    float* policyProbs = outputPolicys + row * NNPos::MAX_NN_POLICY_SIZE;

    //These are not actually correct, the client does the postprocessing to turn them into
    //policy probabilities and white game outcome probabilities
    //Also we don't fill in the nnHash here either
    SymmetryHelpers::copyOutputsWithSymmetry(policySrcBuf, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    policyProbs[gpuHandle->policySize-1] = policySrcBuf[gpuHandle->policySize-1];

    int numValueChannels = gpuHandle->model->numValueChannels;
    assert(numValueChannels == 3);
    output->whiteWinProb = inputBuffers->valueResults[row * numValueChannels];
    output->whiteLossProb = inputBuffers->valueResults[row * numValueChannels + 1];
    output->whiteNoResultProb = inputBuffers->valueResults[row * numValueChannels + 2];

    if(version >= 9) {
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 6);
      output->varTimeLeft = inputBuffers->scoreValueResults[row * numScoreValueChannels + 3];
      output->shorttermWinlossError = inputBuffers->scoreValueResults[row * numScoreValueChannels + 4];
    }
    else if(version >= 8) {
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 4);
      output->varTimeLeft = inputBuffers->scoreValueResults[row * numScoreValueChannels + 3];
      output->shorttermWinlossError = 0;
    }
    else if(version >= 4) {
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 2);
      output->varTimeLeft = 0;
      output->shorttermWinlossError = 0;
    }
    else if(version >= 3) {
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 1);
      output->varTimeLeft = 0;
      output->shorttermWinlossError = 0;
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }

}

static void cudaUploadBenchmarkInputs(ComputeHandle* gpuHandle, InputBuffers* inputBuffers, int batchSize) {
  Buffers* buffers = gpuHandle->buffers.get();
  if(!gpuHandle->usingFP16) {
    CUDA_ERR("benchmarkOutput",cudaMemcpyAsync(buffers->inputBuf, inputBuffers->userInputBuffer, inputBuffers->singleInputBytes*batchSize, cudaMemcpyHostToDevice,gpuHandle->cudaHandles->stream));
    CUDA_ERR("benchmarkOutput",cudaMemcpyAsync(buffers->inputGlobalBuf, inputBuffers->userInputGlobalBuffer, inputBuffers->singleInputGlobalBytes*batchSize, cudaMemcpyHostToDevice,gpuHandle->cudaHandles->stream));
  }
  else {
    CUDA_ERR("benchmarkOutput",cudaMemcpyAsync(buffers->inputBufFloat, inputBuffers->userInputBuffer, inputBuffers->singleInputBytes*batchSize, cudaMemcpyHostToDevice,gpuHandle->cudaHandles->stream));
    CUDA_ERR("benchmarkOutput",cudaMemcpyAsync(buffers->inputGlobalBufFloat, inputBuffers->userInputGlobalBuffer, inputBuffers->singleInputGlobalBytes*batchSize, cudaMemcpyHostToDevice,gpuHandle->cudaHandles->stream));

    customCudaCopyToHalf((const float*)buffers->inputBufFloat,(half*)buffers->inputBuf,inputBuffers->singleInputElts*batchSize,gpuHandle->cudaHandles->stream);
    CUDA_ERR("benchmarkOutput",cudaPeekAtLastError());
    customCudaCopyToHalf((const float*)buffers->inputGlobalBufFloat,(half*)buffers->inputGlobalBuf,inputBuffers->singleInputGlobalElts*batchSize,gpuHandle->cudaHandles->stream);
    CUDA_ERR("benchmarkOutput",cudaPeekAtLastError());
  }
}

static void cudaPrepareBenchmarkHostInputs(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  NNResultBuf** inputBufs,
  int batchSize
) {
  const int nnXLen = gpuHandle->nnXLen;
  const int nnYLen = gpuHandle->nnYLen;
  const int version = gpuHandle->model->version;
  const int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(version);
  const int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(version);
  for(int nIdx = 0; nIdx < batchSize; nIdx++) {
    if(inputBufs[nIdx] == NULL || inputBufs[nIdx]->rowSpatial == NULL || inputBufs[nIdx]->rowGlobal == NULL)
      throw StringError("benchmarkOutput: null input row");
    float* rowSpatialInput = inputBuffers->userInputBuffer + inputBuffers->singleInputElts * nIdx;
    float* rowGlobalInput = inputBuffers->userInputGlobalBuffer + inputBuffers->singleInputGlobalElts * nIdx;
    std::copy(inputBufs[nIdx]->rowGlobal,inputBufs[nIdx]->rowGlobal+numGlobalFeatures,rowGlobalInput);
    SymmetryHelpers::copyInputsWithSymmetry(
      inputBufs[nIdx]->rowSpatial,
      rowSpatialInput,
      1,
      nnYLen,
      nnXLen,
      numSpatialFeatures,
      gpuHandle->inputsUseNHWC,
      inputBufs[nIdx]->symmetry
    );
  }
}

bool NeuralNet::benchmarkOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  NNResultBuf** inputBufs,
  int batchSize,
  int numWarmups,
  int numIterations,
  bool forceMaskAllOnes,
  vector<double>& iterationSeconds,
  const std::function<void()>& beforeTimedLoop,
  const std::function<void()>& afterTimedLoop
) {
  assert(batchSize > 0 && batchSize <= inputBuffers->maxBatchSize);
  if(numWarmups < 0 || numIterations <= 0)
    throw StringError("benchmarkOutput: invalid warmup/iteration count");

  iterationSeconds.clear();

  // One-time host packing and H2D preparation, excluded from the timed loop.
  cudaPrepareBenchmarkHostInputs(gpuHandle,inputBuffers,inputBufs,batchSize);
  cudaUploadBenchmarkInputs(gpuHandle, inputBuffers, batchSize);

  Buffers* buffers = gpuHandle->buffers.get();
  ScratchBuffers* scratch = gpuHandle->scratch.get();
  const bool effectiveRequireExactNNLen =
    gpuHandle->requireExactNNLen && !forceMaskAllOnes;

  for(int w = 0; w < numWarmups; w++) {
    gpuHandle->model->apply(
      gpuHandle->cudaHandles.get(),
      scratch,
      batchSize,
      effectiveRequireExactNNLen,
      buffers->inputBuf,
      buffers->inputGlobalBuf,
      buffers->policyBuf,
      buffers->valueBuf,
      buffers->scoreValueBuf,
      buffers->ownershipBuf,
      buffers->workspaceBuf,
      buffers->workspaceBytes
    );
  }
  CUDA_ERR("benchmarkOutput",cudaStreamSynchronize(gpuHandle->cudaHandles->stream));

  std::vector<cudaEvent_t> startEvents(numIterations);
  std::vector<cudaEvent_t> endEvents(numIterations);
  for(int i = 0; i < numIterations; i++) {
    CUDA_ERR("benchmarkOutput",cudaEventCreate(&startEvents[i]));
    CUDA_ERR("benchmarkOutput",cudaEventCreate(&endEvents[i]));
  }

  try {
    // Events and all other setup are complete before the common-wall release, so aggregate timing
    // contains only the repeated device forwards and the slowest lane's final stream wait.
    if(beforeTimedLoop)
      beforeTimedLoop();
    for(int i = 0; i < numIterations; i++) {
      CUDA_ERR("benchmarkOutput",cudaEventRecord(startEvents[i],gpuHandle->cudaHandles->stream));
      gpuHandle->model->apply(
        gpuHandle->cudaHandles.get(),
        scratch,
        batchSize,
        effectiveRequireExactNNLen,
        buffers->inputBuf,
        buffers->inputGlobalBuf,
        buffers->policyBuf,
        buffers->valueBuf,
        buffers->scoreValueBuf,
        buffers->ownershipBuf,
        buffers->workspaceBuf,
        buffers->workspaceBytes
      );
      CUDA_ERR("benchmarkOutput",cudaEventRecord(endEvents[i],gpuHandle->cudaHandles->stream));
    }
    CUDA_ERR("benchmarkOutput",cudaStreamSynchronize(gpuHandle->cudaHandles->stream));
    if(afterTimedLoop)
      afterTimedLoop();

    iterationSeconds.reserve(numIterations);
    for(int i = 0; i < numIterations; i++) {
      float milliseconds = 0.0f;
      CUDA_ERR("benchmarkOutput",cudaEventElapsedTime(&milliseconds,startEvents[i],endEvents[i]));
      iterationSeconds.push_back((double)milliseconds / 1000.0);
    }
  }
  catch(...) {
    for(int i = 0; i < numIterations; i++) {
      cudaEventDestroy(startEvents[i]);
      cudaEventDestroy(endEvents[i]);
    }
    throw;
  }

  for(int i = 0; i < numIterations; i++) {
    cudaEventDestroy(startEvents[i]);
    cudaEventDestroy(endEvents[i]);
  }
  return true;
}

//TESTING ----------------------------------------------------------------------------------


bool NeuralNet::testEvaluateConv(
  const ConvLayerDesc* desc,
  int desiredBatchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  vector<float>& outputBuffer
) {
  cudaDeviceSynchronize();
  CudaHandles* cudaHandles = CudaHandles::cudaHandlesTesting();

  size_t numInputFloats = (size_t)desiredBatchSize * nnXLen * nnYLen * desc->inChannels;
  size_t numOutputFloats = (size_t)desiredBatchSize * nnXLen * nnYLen * desc->outChannels;
  if(numInputFloats != inputBuffer.size())
    throw StringError("testEvaluateConv: unexpected input buffer size");

  void* deviceInput;
  void* deviceOutput;
  CudaUtils::mallocAndCopyToDevice("deviceInput", inputBuffer.data(), numInputFloats, deviceInput, useFP16);
  CudaUtils::mallocOnDevice("deviceOutput", numOutputFloats, deviceOutput, useFP16);

  int maxBatchSize = desiredBatchSize;

  CudnnManager* manager = new CudnnManager("manager",maxBatchSize,nnXLen,nnYLen);
  ConvLayer* convLayer = new ConvLayer(cudaHandles,manager,desc,useFP16,useNHWC);

  size_t workspaceBytes =
    convLayer->requiredWorkspaceBytes(cudaHandles,desiredBatchSize);
  void* deviceWorkspace;
  CUDA_ERR("deviceWorkspace",cudaMalloc(&deviceWorkspace, workspaceBytes));


  bool accumulate = false;
  convLayer->apply(
    cudaHandles,
    desiredBatchSize,
    accumulate,
    deviceInput,
    deviceOutput,
    deviceWorkspace,
    workspaceBytes
  );

  outputBuffer.resize(numOutputFloats);
  CudaUtils::expensiveCopyFromDevice("copyResultsToHost", outputBuffer.data(), numOutputFloats, deviceOutput, useFP16);

  cudaFree(deviceWorkspace);

  delete convLayer;
  delete manager;
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  delete cudaHandles;

  return true;
}


bool NeuralNet::testEvaluateBatchNorm(
  const BatchNormLayerDesc* desc,
  int desiredBatchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer
) {
  cudaDeviceSynchronize();
  CudaHandles* cudaHandles = CudaHandles::cudaHandlesTesting();

  size_t numInputFloats = (size_t)desiredBatchSize * nnXLen * nnYLen * desc->numChannels;
  size_t numMaskFloats = (size_t)desiredBatchSize * nnXLen * nnYLen;
  size_t numOutputFloats = (size_t)desiredBatchSize * nnXLen * nnYLen * desc->numChannels;
  if(numInputFloats != inputBuffer.size())
    throw StringError("testEvaluateBatchNorm: unexpected input buffer size");
  if(numMaskFloats != maskBuffer.size())
    throw StringError("testEvaluateBatchNorm: unexpected mask buffer size");

  ActivationLayerDesc actDesc;
  actDesc.activation = ACTIVATION_IDENTITY;

  void* deviceInput;
  void* deviceMask;
  void* deviceOutput;
  CudaUtils::mallocAndCopyToDevice("deviceInput", inputBuffer.data(), numInputFloats, deviceInput, useFP16);
  CudaUtils::mallocAndCopyToDevice("deviceMask", maskBuffer.data(), numMaskFloats, deviceMask, useFP16);
  CudaUtils::mallocOnDevice("deviceOutput", numOutputFloats, deviceOutput, useFP16);

  BatchNormLayer* batchNormLayer = new BatchNormLayer(cudaHandles,desc,&actDesc,nnXLen,nnYLen,useFP16,useNHWC);

  batchNormLayer->apply(
    cudaHandles,
    desiredBatchSize,
    deviceInput,
    deviceMask,
    deviceOutput
  );

  outputBuffer.resize(numOutputFloats);
  CudaUtils::expensiveCopyFromDevice("copyResultsToHost", outputBuffer.data(), numOutputFloats, deviceOutput, useFP16);

  delete batchNormLayer;

  cudaFree(deviceInput);
  cudaFree(deviceMask);
  cudaFree(deviceOutput);
  delete cudaHandles;

  return true;
}


bool NeuralNet::testEvaluateResidualBlock(
  const ResidualBlockDesc* desc,
  int desiredBatchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer
) {
  cudaDeviceSynchronize();
  CudaHandles* cudaHandles = CudaHandles::cudaHandlesTesting();

  size_t numInputFloats = (size_t)desiredBatchSize * nnXLen * nnYLen * desc->preBN.numChannels;
  size_t numMaskFloats = (size_t)desiredBatchSize * nnXLen * nnYLen;
  size_t numOutputFloats = (size_t)desiredBatchSize * nnXLen * nnYLen * desc->finalConv.outChannels;
  if(numInputFloats != inputBuffer.size())
    throw StringError("testEvaluateResidualBlock: unexpected input buffer size");
  if(numMaskFloats != maskBuffer.size())
    throw StringError("testEvaluateResidualBlock: unexpected mask buffer size");

  ScratchBuffers* scratch = new ScratchBuffers(desiredBatchSize, nnXLen, nnYLen, useFP16);

  void* deviceInput;
  void* deviceMask;
  void* deviceScratch;
  CudaUtils::mallocAndCopyToDevice("deviceInput", inputBuffer.data(), numInputFloats, deviceInput, useFP16);
  CudaUtils::mallocAndCopyToDevice("deviceMask", maskBuffer.data(), numMaskFloats, deviceMask, useFP16);
  CudaUtils::mallocOnDevice("deviceScratch", numInputFloats, deviceScratch, useFP16);

  int maxBatchSize = desiredBatchSize;

  CudnnManager* manager = new CudnnManager("manager",maxBatchSize,nnXLen,nnYLen);
  ResidualBlock* residualBlock = new ResidualBlock(cudaHandles,manager,desc,nnXLen,nnYLen,useFP16,useNHWC);

  size_t workspaceBytes =
    residualBlock->requiredWorkspaceBytes(cudaHandles,desiredBatchSize);
  void* deviceWorkspace;
  CUDA_ERR("deviceWorkspace",cudaMalloc(&deviceWorkspace, workspaceBytes));

  residualBlock->apply(
    cudaHandles,
    scratch,
    desiredBatchSize,
    deviceInput,
    deviceScratch,
    deviceMask,
    deviceWorkspace,
    workspaceBytes
  );

  outputBuffer.resize(numOutputFloats);
  CudaUtils::expensiveCopyFromDevice("copyResultsToHost", outputBuffer.data(), numOutputFloats, deviceInput, useFP16);

  cudaFree(deviceWorkspace);

  delete residualBlock;
  delete manager;
  cudaFree(deviceInput);
  cudaFree(deviceMask);
  cudaFree(deviceScratch);
  delete scratch;
  delete cudaHandles;

  return true;
}

bool NeuralNet::testEvaluateGlobalPoolingResidualBlock(
  const GlobalPoolingResidualBlockDesc* desc,
  int desiredBatchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer
) {
  cudaDeviceSynchronize();
  CudaHandles* cudaHandles = CudaHandles::cudaHandlesTesting();

  size_t numInputFloats = (size_t)desiredBatchSize * nnXLen * nnYLen * desc->preBN.numChannels;
  size_t numMaskFloats = (size_t)desiredBatchSize * nnXLen * nnYLen;
  size_t numMaskSumFloats = (size_t)desiredBatchSize;
  size_t numOutputFloats = (size_t)desiredBatchSize * nnXLen * nnYLen * desc->finalConv.outChannels;

  if(numInputFloats != inputBuffer.size())
    throw StringError("testEvaluateGlobalPoolingResidualBlock: unexpected input buffer size");
  if(numMaskFloats != maskBuffer.size())
    throw StringError("testEvaluateGlobalPoolingResidualBlock: unexpected mask buffer size");

  ScratchBuffers* scratch = new ScratchBuffers(desiredBatchSize, nnXLen, nnYLen, useFP16);

  void* deviceInput;
  void* deviceMask;
  float* deviceMaskFloatOrig;
  float* deviceMaskFloat;
  float* deviceMaskSum;
  void* deviceScratch;

  CudaUtils::mallocAndCopyToDevice("deviceInput", inputBuffer.data(), numInputFloats, deviceInput, useFP16);
  CudaUtils::mallocAndCopyToDevice("deviceMask", maskBuffer.data(), numMaskFloats, deviceMask, useFP16);
  CUDA_ERR("deviceMaskFloat",cudaMalloc(reinterpret_cast<void**>(&deviceMaskFloat), numMaskFloats * sizeof(float)));
  CUDA_ERR("deviceMaskSum",cudaMalloc(reinterpret_cast<void**>(&deviceMaskSum), numMaskSumFloats * sizeof(float)));
  deviceMaskFloatOrig = deviceMaskFloat;
  CudaUtils::mallocOnDevice("deviceScratch", numInputFloats, deviceScratch, useFP16);

  fillMaskFloatBufAndMaskSumBuf(cudaHandles,deviceMask, deviceMaskFloat, deviceMaskSum, useFP16, desiredBatchSize, nnXLen, nnYLen);

  int maxBatchSize = desiredBatchSize;

  CudnnManager* manager = new CudnnManager("manager",maxBatchSize,nnXLen,nnYLen);
  GlobalPoolingResidualBlock* residualBlock = new GlobalPoolingResidualBlock(
    cudaHandles,manager,desc,nnXLen,nnYLen,useFP16,useNHWC
  );

  size_t workspaceBytes =
    residualBlock->requiredWorkspaceBytes(
      cudaHandles,desiredBatchSize
    );

  void* deviceWorkspace;
  CUDA_ERR("deviceWorkspace",cudaMalloc(&deviceWorkspace, workspaceBytes));

  residualBlock->apply(
    cudaHandles,
    scratch,
    desiredBatchSize,
    deviceInput,
    deviceScratch,
    deviceMask,
    deviceMaskSum,
    deviceWorkspace,
    workspaceBytes
  );

  outputBuffer.resize(numOutputFloats);
  CudaUtils::expensiveCopyFromDevice("copyResultsToHost", outputBuffer.data(), numOutputFloats, deviceInput, useFP16);

  cudaFree(deviceWorkspace);

  delete residualBlock;
  delete manager;

  cudaFree(deviceInput);
  cudaFree(deviceMask);
  cudaFree(deviceMaskFloatOrig);
  cudaFree(deviceMaskSum);
  cudaFree(deviceScratch);
  delete scratch;
  delete cudaHandles;

  return true;
}


#endif  // USE_CUDA_BACKEND

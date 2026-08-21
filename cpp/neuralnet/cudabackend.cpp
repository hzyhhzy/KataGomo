#ifdef USE_CUDA_BACKEND
#include <atomic>
#include <cstdlib>
#include <limits>

#include "../neuralnet/cudaerrorcheck.h"
#include "../neuralnet/cudaincludes.h"

// The dependency-minimal generic fallback uses the in-tree online-softmax
// attention kernel. Shape-specific attention will be selected separately.
#define KATAGO_CUDA_HAS_SDPA 0

#include "../neuralnet/cudahelpers.h"
#if defined(KATAGO_CUDA_FUSED_FFN_AVAILABLE) && KATAGO_CUDA_FUSED_FFN_AVAILABLE
#include "../neuralnet/cudafusedffn.h"
#endif
#include "../neuralnet/cudautils.h"
#include "../neuralnet/int8policy.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/v105policy.h"
#include "../neuralnet/four_profile/manager.h"
#include "../neuralnet/four_profile/mode.h"
#include "../neuralnet/four_profile/native_model_view.h"
#include "../neuralnet/four_profile/stub_factories.h"
#if defined(KATAGO_P3_PROVIDER_COMPILED) && KATAGO_P3_PROVIDER_COMPILED
#include "../neuralnet/four_profile/p3_provider.h"
#endif
#if defined(KATAGO_P4_PROVIDER_COMPILED) && KATAGO_P4_PROVIDER_COMPILED
#include "../neuralnet/four_profile/p4_provider.h"
#endif
#include "../neuralnet/nninterface.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/desc.h"

#include "../core/simpleallocator.h"

#include "../external/half-2.2.0/include/half.hpp"

//------------------------
#include "../core/using.h"
//------------------------

using half_t = half_float::half;

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
// [B, 1, S, S] from the [B, S] mask: bias[b,q,k] = (mask[b,k] != 0 ? 0 : -1e4). cudnn does not have
// plans for the [B,1,1,S] broadcast pattern that would let us avoid this materialization, but the
// full bias is correct for arbitrary (non-prefix) masks, which we need to support sub-board games.
// The bias is built once per inference (the mask is the same across all 20 attention blocks).
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
  bool sdpaSupported;
  string disableReason;

  SDPAGraphCache() :
    plansByKey(),
    sdpaSupported(true),
    disableReason()
  {}

  // Build (or fetch from cache) an execution plan for the given attention shape + batchSize + hasMask.
  // Returns nullptr if SDPA is not supported for this configuration; caller should use fallback.
  // On a build failure during warmup, SDPA is disabled going forward and nullptr is returned (the
  // caller falls back to the custom kernel); outside of warmup such a failure is fatal. logger (if
  // non-NULL) is used to report a disable.
  std::shared_ptr<SDPAPlanForBatchSize> getOrBuildPlan(cudnnHandle_t cudnn, const SDPAGraphKey& key, Logger* logger, bool isWarmup) {
    if(!sdpaSupported)
      return nullptr;

    // Cuda graphs for SDPA path only well-tested for FP16/BF16; FP32 uses fallback
    if(!key.usingFP16)
      return nullptr;

    auto it = plansByKey.find(key);
    if(it != plansByKey.end())
      return it->second;

    namespace fe = cudnn_frontend;

    // Disable SDPA and report the reason. Outside of warmup a build failure is fatal; during warmup
    // we tolerate it and fall back to the custom kernel (returning nullptr to the caller).
    auto disable = [&](const string& reason) -> std::shared_ptr<SDPAPlanForBatchSize> {
      if(!isWarmup)
        throw StringError(reason);
      sdpaSupported = false;
      disableReason = reason;
      if(logger != NULL)
        logger->write("Cuda backend: disabling cudnn SDPA and falling back to custom attention kernel: " + reason);
      return nullptr;
    };
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
      return disable(string("cudnn SDPA graph validate failed: ") + status.get_message());
    status = graph->build_operation_graph(cudnn);
    if(status.is_bad())
      return disable(string("cudnn SDPA build_operation_graph failed: ") + status.get_message());
    status = graph->create_execution_plans({fe::HeurMode_t::A});
    if(status.is_bad())
      return disable(string("cudnn SDPA create_execution_plans failed: ") + status.get_message());
    status = graph->check_support(cudnn);
    if(status.is_bad())
      return disable(string("cudnn SDPA check_support failed: ") + status.get_message());
    status = graph->build_plans(cudnn);
    if(status.is_bad())
      return disable(string("cudnn SDPA build_plans failed: ") + status.get_message());

    int64_t ws = 0;
    status = graph->get_workspace_size(ws);
    if(status.is_bad())
      return disable(string("cudnn SDPA get_workspace_size failed: ") + status.get_message());

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

// Per-handle RAII owners. Declaring the stream first makes it outlive both
// vendor handles, while partial construction failures unwind every resource
// that was successfully created before the failure.
struct OwnedComputeStream {
  cudaStream_t stream;

  OwnedComputeStream() {
    CUDA_ERR("CudaHandles",cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  }
  ~OwnedComputeStream() {
    (void)cudaStreamDestroy(stream);
  }
  operator cudaStream_t() const { return stream; }

  OwnedComputeStream(const OwnedComputeStream&) = delete;
  OwnedComputeStream& operator=(const OwnedComputeStream&) = delete;
};

struct OwnedCublasHandle {
  cublasHandle_t handle;

  OwnedCublasHandle() {
    CUBLAS_ERR("CudaHandles",cublasCreate(&handle));
  }
  ~OwnedCublasHandle() {
    (void)cublasDestroy(handle);
  }
  operator cublasHandle_t() const { return handle; }

  OwnedCublasHandle(const OwnedCublasHandle&) = delete;
  OwnedCublasHandle& operator=(const OwnedCublasHandle&) = delete;
};

struct OwnedCudnnHandle {
  cudnnHandle_t handle;

  OwnedCudnnHandle() {
    CUDNN_ERR("CudaHandles",cudnnCreate(&handle));
  }
  ~OwnedCudnnHandle() {
    (void)cudnnDestroy(handle);
  }
  operator cudnnHandle_t() const { return handle; }

  OwnedCudnnHandle(const OwnedCudnnHandle&) = delete;
  OwnedCudnnHandle& operator=(const OwnedCudnnHandle&) = delete;
};

#ifdef KATAGO_BUILD_BENCHMARKNN
struct BenchmarkRouteCountsInternal {
  int attention;
  int ffn;
  int qkn;
  int orderedClippedSwiGLU;
  int combinedQKV;
  int learnedRopeFp32;
  int fixedRope;
  int mma;
  int scalar;
  int planar;
  int cudnn;
  int fallback;

  BenchmarkRouteCountsInternal()
    : attention(0),
      ffn(0),
      qkn(0),
      orderedClippedSwiGLU(0),
      combinedQKV(0),
      learnedRopeFp32(0),
      fixedRope(0),
      mma(0),
      scalar(0),
      planar(0),
      cudnn(0),
      fallback(0)
  {}
};

struct BenchmarkRouteStateInternal {
  bool prepared;
  bool hasSuccessfulInvocation;
  bool loggedPrepared;
  bool loggedActive;
  bool currentModelApplyComplete;
  uint64_t invocationSerial;
  uint64_t streamIdentity;
  int nnXLen;
  int nnYLen;
  int lastBatchSize;
  int currentBatchSize;
  bool lastFp16;
  bool lastNhwc;
  bool lastExact;
  bool lastMaskNull;
  bool currentFp16;
  bool currentNhwc;
  bool currentExact;
  bool currentMaskNull;
  BenchmarkRouteCountsInternal expected;
  BenchmarkRouteCountsInternal preparedCounts;
  BenchmarkRouteCountsInternal current;
  BenchmarkRouteCountsInternal last;

  BenchmarkRouteStateInternal()
    : prepared(false),
      hasSuccessfulInvocation(false),
      loggedPrepared(false),
      loggedActive(false),
      currentModelApplyComplete(false),
      invocationSerial(0),
      streamIdentity(0),
      nnXLen(0),
      nnYLen(0),
      lastBatchSize(0),
      currentBatchSize(0),
      lastFp16(false),
      lastNhwc(false),
      lastExact(false),
      lastMaskNull(false),
      currentFp16(false),
      currentNhwc(false),
      currentExact(false),
      currentMaskNull(false),
      expected(),
      preparedCounts(),
      current(),
      last()
  {}

  void beginInvocation() noexcept {
    current = BenchmarkRouteCountsInternal();
    currentModelApplyComplete = false;
  }

  void finishModelApply(
    int batchSize,
    bool fp16,
    bool nhwc,
    bool exact,
    bool maskNull
  ) noexcept {
    currentBatchSize = batchSize;
    currentFp16 = fp16;
    currentNhwc = nhwc;
    currentExact = exact;
    currentMaskNull = maskNull;
    currentModelApplyComplete = true;
  }

  void publishSuccessfulInvocation() noexcept {
    if(!currentModelApplyComplete)
      return;
    last = current;
    lastBatchSize = currentBatchSize;
    lastFp16 = currentFp16;
    lastNhwc = currentNhwc;
    lastExact = currentExact;
    lastMaskNull = currentMaskNull;
    invocationSerial += 1;
    hasSuccessfulInvocation = true;
    currentModelApplyComplete = false;
  }
};
#endif


struct CudaHandles {
  // Every inference-time transfer, custom kernel, and vendor operation for
  // this handle is ordered on this nonblocking stream. Distinct NN server
  // threads can therefore overlap without legacy-default-stream coupling.
  OwnedComputeStream stream;
  OwnedCublasHandle cublas;
  OwnedCudnnHandle cudnn;
  const int majorComputeCapability;
  const int minorComputeCapability;
  int multiprocessorCount;
  // Set once, before Model construction. Attention blocks may commit their
  // QKV weights to the official interleaved layout only when this is true.
  bool mmaAttentionEnabled;
  bool loggedUsingCombinedQKV;
  bool loggedUsingFusedQKNormRoPE;
  bool loggedUsingMmaAttention;
  bool loggedUsingScalarAttention;
#if defined(KATAGO_CUDA_FUSED_FFN_AVAILABLE) && KATAGO_CUDA_FUSED_FFN_AVAILABLE
  bool genericFusedFfnAvailable;
  bool loggedUsingGenericFusedFfn;
#endif
  std::unique_ptr<SDPAGraphCache> sdpaCache;
  // Logger for this handle's server thread; may be NULL. Used to report cudnn SDPA falling back.
  Logger* logger;
  // Set while warming up (see NNEvaluator::maybeWarmupComputeHandle). When true, a failed cudnn SDPA
  // execution is tolerated (fall back to the custom kernel); when false such a failure is fatal.
  bool isWarmup;
#ifdef KATAGO_BUILD_BENCHMARKNN
  BenchmarkRouteStateInternal benchmarkRoute;
#endif

  CudaHandles(int major, int minor)
    : majorComputeCapability(major),
      minorComputeCapability(minor),
      multiprocessorCount(0),
      mmaAttentionEnabled(false),
      loggedUsingCombinedQKV(false),
      loggedUsingFusedQKNormRoPE(false),
      loggedUsingMmaAttention(false),
      loggedUsingScalarAttention(false),
#if defined(KATAGO_CUDA_FUSED_FFN_AVAILABLE) && KATAGO_CUDA_FUSED_FFN_AVAILABLE
      genericFusedFfnAvailable(false),
      loggedUsingGenericFusedFfn(false),
#endif
      sdpaCache(std::make_unique<SDPAGraphCache>()),
      logger(NULL),
      isWarmup(false)
  {
    int device = 0;
    CUDA_ERR("CudaHandles",cudaGetDevice(&device));
    CUDA_ERR(
      "CudaHandles",
      cudaDeviceGetAttribute(
        &multiprocessorCount,cudaDevAttrMultiProcessorCount,device));
    CUBLAS_ERR("CudaHandles",cublasSetStream(cublas.handle,stream.stream));
    CUDNN_ERR("CudaHandles",cudnnSetStream(cudnn.handle,stream.stream));
#if defined(KATAGO_CUDA_FUSED_FFN_AVAILABLE) && KATAGO_CUDA_FUSED_FFN_AVAILABLE
    genericFusedFfnAvailable = CudaFusedFFN::supportedOnCurrentDevice();
#endif
  }

  static CudaHandles* cudaHandlesTesting() {
    const int gpuIdxForThisThread = 0;
    cudaDeviceProp prop;
    CUDA_ERR("cudaHandlesTesting",cudaGetDeviceProperties(&prop,gpuIdxForThisThread));
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

  ScratchBuffers() = delete;
  ScratchBuffers(const ScratchBuffers&) = delete;
  ScratchBuffers& operator=(const ScratchBuffers&) = delete;

  ScratchBuffers(int maxBatchSize, int nnXLen, int nnYLen, bool useFP16)
    : batchXYFloatBytes((size_t)maxBatchSize * nnXLen * nnYLen * sizeof(float)),
      batchFloatBytes((size_t)maxBatchSize * sizeof(float)),
      batchXYBytes((size_t)maxBatchSize * nnXLen * nnYLen * (useFP16 ? sizeof(half_t) : sizeof(float))),
      batchBytes((size_t)maxBatchSize * (useFP16 ? sizeof(half_t) : sizeof(float)))
  {
    std::function<void*(size_t)> allocateFunc = [](size_t size) {
      void* buf;
      CUDA_ERR("ScratchBuffers",cudaMalloc(&buf, size));
      return buf;
    };
    std::function<void(void*)> releaseFunc = [](void* buf) {
      cudaFree(buf);
    };

    allocator = new SimpleAllocator<void*>(allocateFunc, releaseFunc);

    CudaUtils::hostMallocZeroOneBufs(zeroBuf, oneBuf, useFP16);
  }
  ~ScratchBuffers() {
    delete allocator;
    free(zeroBuf);
    free(oneBuf);
  }

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

// Exception-safe ownership for device allocations used by the transformer
// adapter and its MatMul construction chain. Keep this deliberately small
// instead of broadening the change into legacy convolution/head allocations.
struct OwnedDeviceBuf {
  void* buf = nullptr;

  OwnedDeviceBuf() {}
  ~OwnedDeviceBuf() {
    if(buf != nullptr)
      (void)cudaFree(buf);
  }

  OwnedDeviceBuf(const OwnedDeviceBuf&) = delete;
  OwnedDeviceBuf& operator=(const OwnedDeviceBuf&) = delete;
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
  OwnedDeviceBuf matBuf;

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
    CudaUtils::mallocAndCopyToDevice(name,desc->weights,matBuf.buf,useFP16);
  }

  ~MatMulLayer() {}

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
    size_t workspaceBytes,
    bool accumulate = false
  ) const {
    (void)workspaceBuf;
    (void)workspaceBytes;

    if(!usingFP16) {
      const float alpha = 1.0f;
      const float beta = accumulate ? 1.0f : 0.0f;
      CUBLAS_ERR(name.c_str(),cublasSgemm(
        cudaHandles->cublas,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        outChannels,
        batchSize,
        inChannels,
        &alpha,
        (const float*)matBuf.buf,outChannels,
        (const float*)inputBuf,inChannels,
        &beta,
        (float*)outputBuf,outChannels
      ));
    }
    else {
      const half* alpha = (const half*)scratch->oneBuf;
      const half* beta = (const half*)(accumulate ? scratch->oneBuf : scratch->zeroBuf);
      CUBLAS_ERR(name.c_str(),cublasHgemm(
        cudaHandles->cublas,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        outChannels,
        batchSize,
        inChannels,
        alpha,
        (const half*)matBuf.buf,outChannels,
        (const half*)inputBuf,inChannels,
        beta,
        (half*)outputBuf,outChannels
      ));
    }

  }

};

//---------------------------------------------------------------------------------

// Official transformer's shared-input batched GEMM primitive. Q/K/V and the
// two FFN input projections use identical B matrices, so a single cuBLAS call
// avoids three/two independent launches while retaining the legacy FP16
// numerical contract.
static void applySharedInputStridedMatMuls(
  CudaHandles* cudaHandles,
  ScratchBuffers* scratch,
  bool usingFP16,
  const string& name,
  int outChannels,
  int batchSize,
  int inChannels,
  const void* packedWeights,
  const void* inputBuf,
  void* outputBuf,
  long long outputStrideElts,
  int count
) {
  const long long weightStrideElts = (long long)inChannels * outChannels;
  if(!usingFP16) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_ERR(name.c_str(),cublasSgemmStridedBatched(
      cudaHandles->cublas,CUBLAS_OP_N,CUBLAS_OP_N,
      outChannels,batchSize,inChannels,
      &alpha,(const float*)packedWeights,outChannels,weightStrideElts,
      (const float*)inputBuf,inChannels,0LL,
      &beta,(float*)outputBuf,outChannels,outputStrideElts,count
    ));
  }
  else {
    const half* alpha = (const half*)scratch->oneBuf;
    const half* beta = (const half*)scratch->zeroBuf;
    CUBLAS_ERR(name.c_str(),cublasHgemmStridedBatched(
      cudaHandles->cublas,CUBLAS_OP_N,CUBLAS_OP_N,
      outChannels,batchSize,inChannels,
      alpha,(const half*)packedWeights,outChannels,weightStrideElts,
      (const half*)inputBuf,inChannels,0LL,
      beta,(half*)outputBuf,outChannels,outputStrideElts,count
    ));
  }
}

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
      customCudaAddCBiasInplaceNC(
        (float*)matBuf,(const float*)biasBuf,batchSize,numChannels,activation,cudaHandles->stream
      );
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }
    else {
      customCudaAddCBiasInplaceNC(
        (half*)matBuf,(const half*)biasBuf,batchSize,numChannels,activation,cudaHandles->stream
      );
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
    bool useNHWC
  );
  ~BlockStack();

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const;

#ifdef KATAGO_BUILD_BENCHMARKNN
  void collectBenchmarkPreparedRoutes(
    const CudaHandles* cudaHandles,
    BenchmarkRouteCountsInternal& counts
  ) const;
#endif

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* maskBuf,
    float* maskSumBuf,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* workspaceBuf,
    size_t workspaceBytes,
    FourProfile::ManagerV1* fourProfileManager = nullptr,
    const FourProfile::RuntimeKeyV1* fourProfileRuntime = nullptr
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
    bool useNHWC
  ): name(desc->name),
     normActConv1(cudaHandles,manager,&desc->preBN,&desc->preActivation,&desc->preConv,nnX,nnY,useFP16,useNHWC),
     blocks(cudaHandles,manager,desc->numBlocks,desc->preConv.outChannels,desc->blocks,nnX,nnY,useFP16,useNHWC),
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
  OwnedDeviceBuf weightBuf;
  OwnedDeviceBuf zeroBetaBuf;

  TransformerRMSNormLayer() = delete;
  TransformerRMSNormLayer(const TransformerRMSNormLayer&) = delete;
  TransformerRMSNormLayer& operator=(const TransformerRMSNormLayer&) = delete;

  TransformerRMSNormLayer(
    CudaHandles* cudaHandles,
    const TransformerRMSNormDesc* desc,
    bool useFP16
  ) :
    name(desc->name),
    numChannels(desc->numChannels),
    epsilon(desc->epsilon),
    usingFP16(useFP16)
  {
    (void)cudaHandles;
    if((int)desc->weight.size() != numChannels)
      throw StringError(name + ": RMSNorm weight count does not match numChannels");
    CudaUtils::mallocAndCopyToDevice(name, desc->weight, weightBuf.buf, useFP16);
    // Allocate a zero buffer for beta (TransformerRMSNorm has no bias)
    vector<float> zeros(numChannels, 0.0f);
    CudaUtils::mallocAndCopyToDevice(name + ":zeroBeta", zeros, zeroBetaBuf.buf, useFP16);
  }

  ~TransformerRMSNormLayer() {}

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
    // RMSNormGammaBetaNHWC with gamma=weight, beta=zero, mask, identity activation.
    if(!usingFP16) {
      customCudaRMSNormGammaBetaNHWC(
        (const float*)inputBuf, (float*)outputBuf,
        (const float*)weightBuf.buf, (const float*)zeroBetaBuf.buf,
        (const float*)maskBuf,
        batchSize, xySize, numChannels, epsilon, ACTIVATION_IDENTITY, cudaHandles->stream);
    }
    else {
      customCudaRMSNormGammaBetaNHWC(
        (const half*)inputBuf, (half*)outputBuf,
        (const half*)weightBuf.buf, (const half*)zeroBetaBuf.buf,
        (const half*)maskBuf,
        batchSize, xySize, numChannels, epsilon, ACTIVATION_IDENTITY, cudaHandles->stream);
    }
    CUDA_ERR(name.c_str(), cudaPeekAtLastError());
  }

  const half* fp16Weight() const {
    if(!usingFP16)
      throw StringError(name + ": requested FP16 RMSNorm weight from FP32 layer");
    return (const half*)weightBuf.buf;
  }
};

//------------------------------------------------------------------------------

#if 0
// Gom's transformer topology uses BatchNorm at the trunk tip. Keep the
// unrelated official trunk-RMSNorm variant out of this compatibility outer;
// later metadata/PTQ model versions retain this topology.
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
            (const float*)maskBuf, batchSize, numChannels, xySize, epsilon, activation, cudaHandles->stream);
        else
          customCudaRMSNormGammaBetaNHWC(
            (const float*)inputBuf, (float*)outputBuf, (const float*)gammaBuf, (const float*)betaBuf,
            (const float*)maskBuf, batchSize, xySize, numChannels, epsilon, activation, cudaHandles->stream);
      }
      else {
        if(!usingNHWC)
          customCudaRMSNormGammaBetaNCHW(
            (const half*)inputBuf, (half*)outputBuf, (const half*)gammaBuf, (const half*)betaBuf,
            (const half*)maskBuf, batchSize, numChannels, xySize, epsilon, activation, cudaHandles->stream);
        else
          customCudaRMSNormGammaBetaNHWC(
            (const half*)inputBuf, (half*)outputBuf, (const half*)gammaBuf, (const half*)betaBuf,
            (const half*)maskBuf, batchSize, xySize, numChannels, epsilon, activation, cudaHandles->stream);
      }
    }
    else {
      // Allocate temp buffer for spatial reduction from scratch (float regardless of FP16 mode).
      // Holds per-block partial sums plus the final reduced value per batch element; see
      // SPATIAL_RMSNORM_BLOCKS_PER_BATCH in cudahelpers.cu (partialStride = that + 1).
      SizedBuf<void*> sumSqBuf(
        scratch->allocator,scratch->getBufSizeFloat(CUDA_SPATIAL_RMSNORM_SUMSQ_STRIDE)
      );
      if(!usingFP16) {
        if(!usingNHWC)
          customCudaSpatialRMSNormNCHW(
            (const float*)inputBuf, (float*)outputBuf, (const float*)gammaBuf, (const float*)betaBuf,
            (const float*)maskBuf, maskSumBuf, batchSize, numChannels, xySize, epsilon, activation, (float*)sumSqBuf.buf, cudaHandles->stream);
        else
          customCudaSpatialRMSNormNHWC(
            (const float*)inputBuf, (float*)outputBuf, (const float*)gammaBuf, (const float*)betaBuf,
            (const float*)maskBuf, maskSumBuf, batchSize, xySize, numChannels, epsilon, activation, (float*)sumSqBuf.buf, cudaHandles->stream);
      }
      else {
        if(!usingNHWC)
          customCudaSpatialRMSNormNCHW(
            (const half*)inputBuf, (half*)outputBuf, (const half*)gammaBuf, (const half*)betaBuf,
            (const half*)maskBuf, maskSumBuf, batchSize, numChannels, xySize, epsilon, activation, (float*)sumSqBuf.buf, cudaHandles->stream);
        else
          customCudaSpatialRMSNormNHWC(
            (const half*)inputBuf, (half*)outputBuf, (const half*)gammaBuf, (const half*)betaBuf,
            (const half*)maskBuf, maskSumBuf, batchSize, xySize, numChannels, epsilon, activation, (float*)sumSqBuf.buf, cudaHandles->stream);
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
  const bool useQKNorm;
  const int inChannels;

  const int nnXLen;
  const int nnYLen;
  const bool usingFP16;
  const bool usingNHWC;

  const TransformerRMSNormLayer preLN;
  std::unique_ptr<TransformerRMSNormLayer> qNorm;
  std::unique_ptr<TransformerRMSNormLayer> kNorm;
  const MatMulLayer outProj;

  // Official projection plan. On an MMA-capable FP16 device use one interleaved
  // QKV GEMM. Otherwise equal-shape Q/K/V share one strided-batched GEMM.
  const bool sameQKVShapes;
  const bool useCombinedQKV;
  OwnedDeviceBuf qkvPackedWeights;
  std::unique_ptr<MatMulLayer> qProj;
  std::unique_ptr<MatMulLayer> kProj;
  std::unique_ptr<MatMulLayer> vProj;

  // Fixed RoPE uses device tables. Learned RoPE keeps the tiny frequency
  // tensor in FP32 and recomputes sin/cos in the official fused QK kernel.
  OwnedDeviceBuf ropeCosTable;
  OwnedDeviceBuf ropeSinTable;
  OwnedDeviceBuf ropeFreqsBuf;
  OwnedDeviceBuf learnedRopeCosSinTableFp32;
  int ropeNumPairs;

  static bool shouldCombineQKV(
    CudaHandles* cudaHandles,
    const TransformerAttentionDesc* desc,
    bool useFP16
  ) {
    const bool otherwiseEligible =
      useFP16 &&
      cudaHandles->mmaAttentionEnabled &&
      desc->qProj.inChannels == desc->kProj.inChannels &&
      desc->qProj.inChannels == desc->vProj.inChannels &&
      customCudaFlashAttentionMmaSupportsShape(
        desc->numHeads,desc->numKVHeads,desc->qHeadDim,desc->vHeadDim
      ) &&
      (!desc->useQKNorm ||
       customCudaFusedQKNormRoPESupportsShape(desc->qHeadDim));
    return V105CudaPolicy::shouldUseCombinedQKV(desc->useQKNorm,otherwiseEligible);
  }

  static bool scalarAttentionSupportsShape(int qDim,int vDim) {
    return
      (qDim == 32 && vDim == 32) ||
      (qDim == 32 && vDim == 16) ||
      (qDim == 64 && vDim == 64) ||
      (qDim == 64 && vDim == 32) ||
      (qDim == 32 && vDim == 64);
  }

  TransformerAttentionBlock() = delete;
  TransformerAttentionBlock(const TransformerAttentionBlock&) = delete;
  TransformerAttentionBlock& operator=(const TransformerAttentionBlock&) = delete;

  TransformerAttentionBlock(
    CudaHandles* cudaHandles,
    const TransformerAttentionDesc* desc,
    int maxBatchSize,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ) :
    name(desc->name),
    numHeads(desc->numHeads),
    numKVHeads(desc->numKVHeads),
    qHeadDim(desc->qHeadDim),
    vHeadDim(desc->vHeadDim),
    useRope(desc->useRope),
    learnableRope(desc->learnableRope),
    useQKNorm(desc->useQKNorm),
    inChannels(desc->qProj.inChannels),
    nnXLen(nnX),
    nnYLen(nnY),
    usingFP16(useFP16),
    usingNHWC(useNHWC),
    preLN(cudaHandles,&desc->preLN,useFP16),
    qNorm(nullptr),
    kNorm(nullptr),
    outProj(cudaHandles,&desc->outProj,useFP16),
    sameQKVShapes(
      desc->qProj.inChannels == desc->kProj.inChannels &&
      desc->qProj.inChannels == desc->vProj.inChannels &&
      desc->qProj.outChannels == desc->kProj.outChannels &&
      desc->qProj.outChannels == desc->vProj.outChannels
    ),
    useCombinedQKV(shouldCombineQKV(cudaHandles,desc,useFP16)),
    ropeNumPairs(0)
  {
    if(!useNHWC)
      throw StringError("Transformer blocks with NCHW layout are not supported by the CUDA backend");

    const int qTotalDim = numHeads * qHeadDim;
    const int kTotalDim = numKVHeads * qHeadDim;
    const int vTotalDim = numKVHeads * vHeadDim;
    if(qTotalDim % 8 != 0 || kTotalDim % 8 != 0 || vTotalDim % 8 != 0)
      throw StringError(name + ": CUDA attention projection widths must be multiples of 8");

    if(useQKNorm) {
      if(!useFP16)
        throw StringError(name + ": v105 Q/K normalization requires FP16");
      const long long seqLen = (long long)nnXLen * nnYLen;
      const long long maxQElements =
        (long long)maxBatchSize * seqLen * numHeads * qHeadDim;
      const long long maxKElements =
        (long long)maxBatchSize * seqLen * numKVHeads * qHeadDim;
      if(maxQElements >= 2147483647LL || maxKElements >= 2147483647LL)
        throw StringError(name + ": Q/K normalization exceeds the CUDA 32-bit index limit");
      qNorm = std::make_unique<TransformerRMSNormLayer>(cudaHandles,&desc->qNorm,useFP16);
      kNorm = std::make_unique<TransformerRMSNormLayer>(cudaHandles,&desc->kNorm,useFP16);
    }

    // Fail closed during model construction, before adapter-specific QKV/RoPE
    // allocations and before any inference work. The scalar attention kernel
    // flattens batch and query-head into grid.y, while RoPE uses
    // (seqLen,maxBatchSize) as grid.(x,y) and one thread per
    // (query-head,coordinate-pair).
    if(maxBatchSize <= 0 || (long long)maxBatchSize * numHeads > 65535LL)
      throw StringError(name + ": maxBatchSize * numHeads exceeds CUDA attention grid.y limit");
    if(useRope) {
      const long long seqLen = (long long)nnXLen * nnYLen;
      const long long headPairs = (long long)numHeads * (qHeadDim / 2);
      if(seqLen <= 0 || seqLen > 2147483647LL || maxBatchSize > 65535)
        throw StringError(name + ": board or max batch exceeds CUDA RoPE launch grid limits");
      if(headPairs > 1024)
        throw StringError(name + ": numHeads * RoPE coordinate pairs exceeds 1024 threads");
    }
    if(!useCombinedQKV && !scalarAttentionSupportsShape(qHeadDim,vHeadDim))
      throw StringError(
        name + ": attention shape has neither a committed MMA plan nor an official scalar kernel"
      );

    if(useCombinedQKV) {
      if(!cudaHandles->loggedUsingCombinedQKV) {
        cudaHandles->loggedUsingCombinedQKV = true;
        if(cudaHandles->logger != NULL)
          cudaHandles->logger->write("CUDA_TRANSFORMER_ROUTE qkv=official_combined");
      }
      const int combinedDim = qTotalDim + kTotalDim + vTotalDim;
      assert(desc->qProj.weights.size() == (size_t)inChannels * qTotalDim);
      assert(desc->kProj.weights.size() == (size_t)inChannels * kTotalDim);
      assert(desc->vProj.weights.size() == (size_t)inChannels * vTotalDim);
      vector<float> packed((size_t)inChannels * combinedDim);
      for(int i = 0; i < inChannels; i++) {
        std::copy(
          desc->qProj.weights.begin() + (size_t)i * qTotalDim,
          desc->qProj.weights.begin() + (size_t)(i+1) * qTotalDim,
          packed.begin() + (size_t)i * combinedDim
        );
        std::copy(
          desc->kProj.weights.begin() + (size_t)i * kTotalDim,
          desc->kProj.weights.begin() + (size_t)(i+1) * kTotalDim,
          packed.begin() + (size_t)i * combinedDim + qTotalDim
        );
        std::copy(
          desc->vProj.weights.begin() + (size_t)i * vTotalDim,
          desc->vProj.weights.begin() + (size_t)(i+1) * vTotalDim,
          packed.begin() + (size_t)i * combinedDim + qTotalDim + kTotalDim
        );
      }
      CudaUtils::mallocAndCopyToDevice(name + ":qkvCombined",packed,qkvPackedWeights.buf,useFP16);
    }
    else if(sameQKVShapes) {
      vector<float> packed;
      packed.reserve(desc->qProj.weights.size() * 3);
      packed.insert(packed.end(),desc->qProj.weights.begin(),desc->qProj.weights.end());
      packed.insert(packed.end(),desc->kProj.weights.begin(),desc->kProj.weights.end());
      packed.insert(packed.end(),desc->vProj.weights.begin(),desc->vProj.weights.end());
      CudaUtils::mallocAndCopyToDevice(name + ":qkvPacked",packed,qkvPackedWeights.buf,useFP16);
    }
    else {
      qProj = std::make_unique<MatMulLayer>(cudaHandles,&desc->qProj,useFP16);
      kProj = std::make_unique<MatMulLayer>(cudaHandles,&desc->kProj,useFP16);
      vProj = std::make_unique<MatMulLayer>(cudaHandles,&desc->vProj,useFP16);
    }

    if(useRope) {
      ropeNumPairs = qHeadDim / 2;
      if(learnableRope) {
        if(desc->ropeFreqs.size() != (size_t)numKVHeads * ropeNumPairs * 2)
          throw StringError(name + ": invalid learned RoPE frequency tensor");
        CudaUtils::mallocAndCopyToDevice(
          name + ":ropeFreqs",desc->ropeFreqs.data(),(int)desc->ropeFreqs.size(),ropeFreqsBuf.buf,false
        );
        if(useQKNorm && useCombinedQKV) {
          const long long tablePairs =
            (long long)nnXLen * nnYLen * numKVHeads * ropeNumPairs;
          if(tablePairs <= 0 ||
             (unsigned long long)tablePairs >
               (unsigned long long)std::numeric_limits<size_t>::max() /
                 (2 * sizeof(float)))
            throw StringError(name + ": learned RoPE table size overflow");
          CUDA_ERR(
            name.c_str(),
            cudaMalloc(
              &learnedRopeCosSinTableFp32.buf,
              (size_t)tablePairs * 2 * sizeof(float)));
          customCudaBuildLearnedRopeTableFP32(
            (const float*)ropeFreqsBuf.buf,
            (float*)learnedRopeCosSinTableFp32.buf,
            nnXLen * nnYLen,numKVHeads,ropeNumPairs,nnXLen,
            cudaHandles->stream);
          CUDA_ERR(name.c_str(),cudaPeekAtLastError());
        }
      }
      else {
        const int seqLen = nnXLen * nnYLen;
        vector<float> cosTableData;
        vector<float> sinTableData;
        desc->computeRopeCosSin(nnXLen,nnYLen,seqLen,cosTableData,sinTableData);
        CudaUtils::mallocAndCopyToDevice(
          name + ":ropeCos",cosTableData.data(),(int)cosTableData.size(),ropeCosTable.buf,useFP16
        );
        CudaUtils::mallocAndCopyToDevice(
          name + ":ropeSin",sinTableData.data(),(int)sinTableData.size(),ropeSinTable.buf,useFP16
        );
      }
    }
  }

  ~TransformerAttentionBlock() {}

  size_t requiredWorkspaceBytes(CudaHandles* cudaHandles,int batchSize) const {
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

    const int seqLen = nnXLen * nnYLen;
    const int matBatchSize = batchSize * seqLen;
    const int qTotalDim = numHeads * qHeadDim;
    const int kTotalDim = numKVHeads * qHeadDim;
    const int vTotalDim = numKVHeads * vHeadDim;
    const size_t bytesPerElt = usingFP16 ? sizeof(half) : sizeof(float);

    preLN.apply(cudaHandles,batchSize,seqLen,trunkBuf,trunkScratchBuf,maskBuf);

    // Size transients by the handle's max batch. SimpleAllocator pools by exact
    // byte count, so actual-batch sizing would retain one allocation per batch
    // size seen over the process lifetime.
    SizedBuf<void*> qkvBuf(
      scratch->allocator,scratch->getBufSizeXY(qTotalDim + kTotalDim + vTotalDim)
    );
    void* qPtr;
    void* kPtr;
    void* vPtr;
    int qStrideElts;
    int kvStrideElts;

    if(useCombinedQKV) {
      const int combinedDim = qTotalDim + kTotalDim + vTotalDim;
      qPtr = qkvBuf.buf;
      kPtr = (char*)qkvBuf.buf + (size_t)qTotalDim * sizeof(half);
      vPtr = (char*)qkvBuf.buf + (size_t)(qTotalDim + kTotalDim) * sizeof(half);
      qStrideElts = combinedDim;
      kvStrideElts = combinedDim;
      applySharedInputStridedMatMuls(
        cudaHandles,scratch,usingFP16,name,
        combinedDim,matBatchSize,inChannels,
        qkvPackedWeights.buf,trunkScratchBuf,qkvBuf.buf,0LL,1
      );
    }
    else {
      qPtr = qkvBuf.buf;
      kPtr = (char*)qkvBuf.buf + scratch->getBufSizeXY(qTotalDim);
      vPtr = (char*)qkvBuf.buf + scratch->getBufSizeXY(qTotalDim + kTotalDim);
      qStrideElts = qTotalDim;
      kvStrideElts = kTotalDim;
      if(sameQKVShapes) {
        const long long outputStrideElts =
          (long long)(scratch->getBufSizeXY(qTotalDim) / bytesPerElt);
        applySharedInputStridedMatMuls(
          cudaHandles,scratch,usingFP16,name,
          qTotalDim,matBatchSize,inChannels,
          qkvPackedWeights.buf,trunkScratchBuf,qPtr,outputStrideElts,3
        );
      }
      else {
        qProj->apply(cudaHandles,scratch,matBatchSize,trunkScratchBuf,qPtr,workspaceBuf,workspaceBytes);
        kProj->apply(cudaHandles,scratch,matBatchSize,trunkScratchBuf,kPtr,workspaceBuf,workspaceBytes);
        vProj->apply(cudaHandles,scratch,matBatchSize,trunkScratchBuf,vPtr,workspaceBuf,workspaceBytes);
      }
    }

    // Combined QKV uses one stride-aware kernel for both per-head RMSNorm and
    // RoPE. The planar fallback retains the two ordinary RMSNorm launches.
    bool usedFusedQKNormRoPE = false;
    if(useQKNorm) {
      if(qNorm == nullptr || kNorm == nullptr)
        throw StringError(name + ": Q/K normalization descriptors were not prepared");
      if(useCombinedQKV) {
        const CudaFusedQKNormRopeMode ropeMode = !useRope ?
          CudaFusedQKNormRopeMode::None :
          (learnableRope ? CudaFusedQKNormRopeMode::LearnedTable :
                           CudaFusedQKNormRopeMode::Fixed);
        customCudaFusedQKNormRoPEFP16(
          (half*)qPtr,(half*)kPtr,
          qNorm->fp16Weight(),kNorm->fp16Weight(),
          (const half*)ropeCosTable.buf,(const half*)ropeSinTable.buf,
          (const float*)learnedRopeCosSinTableFp32.buf,
          matBatchSize,seqLen,numHeads,numKVHeads,qHeadDim,
          qStrideElts,kvStrideElts,ropeNumPairs,
          qNorm->epsilon,kNorm->epsilon,ropeMode,
          cudaHandles->multiprocessorCount,cudaHandles->stream);
        CUDA_ERR(name.c_str(),cudaPeekAtLastError());
        usedFusedQKNormRoPE = true;
        if(!cudaHandles->loggedUsingFusedQKNormRoPE) {
          cudaHandles->loggedUsingFusedQKNormRoPE = true;
          if(cudaHandles->logger != NULL)
            cudaHandles->logger->write(
              "CUDA_TRANSFORMER_ROUTE qkn_rope=fused_combined_fp16");
        }
      }
      else {
        // [M,H,D] is contiguous, so viewing it as [M*H,1,D] applies the
        // learned D-vector independently to every head.
        qNorm->apply(cudaHandles,matBatchSize * numHeads,1,qPtr,qPtr,nullptr);
        kNorm->apply(cudaHandles,matBatchSize * numKVHeads,1,kPtr,kPtr,nullptr);
      }
#ifdef KATAGO_BUILD_BENCHMARKNN
      cudaHandles->benchmarkRoute.current.qkn += 1;
#endif
    }

    if(useRope && !usedFusedQKNormRoPE) {
      const bool fuseQK = numHeads % numKVHeads == 0;
      assert(!useCombinedQKV || fuseQK);

      if(learnableRope) {
        if(!usingFP16) {
          if(fuseQK)
            customCudaApplyRoPEQKLearnableRecompute(
              (float*)qPtr,(float*)kPtr,(const float*)ropeFreqsBuf.buf,
              batchSize,seqLen,numHeads,numKVHeads,qHeadDim,
              qStrideElts,kvStrideElts,ropeNumPairs,nnXLen,cudaHandles->stream
            );
          else {
            customCudaApplyRoPELearnableRecompute(
              (float*)qPtr,(const float*)ropeFreqsBuf.buf,batchSize,seqLen,numHeads,numKVHeads,
              qHeadDim,ropeNumPairs,nnXLen,cudaHandles->stream
            );
            customCudaApplyRoPELearnableRecompute(
              (float*)kPtr,(const float*)ropeFreqsBuf.buf,batchSize,seqLen,numKVHeads,numKVHeads,
              qHeadDim,ropeNumPairs,nnXLen,cudaHandles->stream
            );
          }
        }
        else {
          if(fuseQK)
            customCudaApplyRoPEQKLearnableRecompute(
              (half*)qPtr,(half*)kPtr,(const float*)ropeFreqsBuf.buf,
              batchSize,seqLen,numHeads,numKVHeads,qHeadDim,
              qStrideElts,kvStrideElts,ropeNumPairs,nnXLen,cudaHandles->stream
            );
          else {
            customCudaApplyRoPELearnableRecompute(
              (half*)qPtr,(const float*)ropeFreqsBuf.buf,batchSize,seqLen,numHeads,numKVHeads,
              qHeadDim,ropeNumPairs,nnXLen,cudaHandles->stream
            );
            customCudaApplyRoPELearnableRecompute(
              (half*)kPtr,(const float*)ropeFreqsBuf.buf,batchSize,seqLen,numKVHeads,numKVHeads,
              qHeadDim,ropeNumPairs,nnXLen,cudaHandles->stream
            );
          }
        }
      }
      else {
        if(!usingFP16) {
          if(fuseQK)
            customCudaApplyRoPEQK(
              (float*)qPtr,(float*)kPtr,(const float*)ropeCosTable.buf,(const float*)ropeSinTable.buf,
              batchSize,seqLen,numHeads,numKVHeads,qHeadDim,qStrideElts,kvStrideElts,
              ropeNumPairs,false,cudaHandles->stream
            );
          else {
            customCudaApplyRoPE(
              (float*)qPtr,(const float*)ropeCosTable.buf,(const float*)ropeSinTable.buf,
              batchSize,seqLen,numHeads,numKVHeads,qHeadDim,ropeNumPairs,false,cudaHandles->stream
            );
            customCudaApplyRoPE(
              (float*)kPtr,(const float*)ropeCosTable.buf,(const float*)ropeSinTable.buf,
              batchSize,seqLen,numKVHeads,numKVHeads,qHeadDim,ropeNumPairs,false,cudaHandles->stream
            );
          }
        }
        else {
          if(fuseQK)
            customCudaApplyRoPEQK(
              (half*)qPtr,(half*)kPtr,(const half*)ropeCosTable.buf,(const half*)ropeSinTable.buf,
              batchSize,seqLen,numHeads,numKVHeads,qHeadDim,qStrideElts,kvStrideElts,
              ropeNumPairs,false,cudaHandles->stream
            );
          else {
            customCudaApplyRoPE(
              (half*)qPtr,(const half*)ropeCosTable.buf,(const half*)ropeSinTable.buf,
              batchSize,seqLen,numHeads,numKVHeads,qHeadDim,ropeNumPairs,false,cudaHandles->stream
            );
            customCudaApplyRoPE(
              (half*)kPtr,(const half*)ropeCosTable.buf,(const half*)ropeSinTable.buf,
              batchSize,seqLen,numKVHeads,numKVHeads,qHeadDim,ropeNumPairs,false,cudaHandles->stream
            );
          }
        }
      }
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }

    SizedBuf<void*> attnOutBuf(
      scratch->allocator,scratch->getBufSizeXY(numHeads * vHeadDim)
    );
    bool usedMma = false;
    if(usingFP16 && cudaHandles->mmaAttentionEnabled) {
      usedMma = customCudaFlashAttentionMma(
        (const half*)qPtr,(const half*)kPtr,(const half*)vPtr,(const half*)maskBuf,
        (half*)attnOutBuf.buf,batchSize,seqLen,numHeads,numKVHeads,qHeadDim,vHeadDim,
        qStrideElts,kvStrideElts,cudaHandles->stream
      );
      if(usedMma)
        CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      if(usedMma && !cudaHandles->loggedUsingMmaAttention) {
        cudaHandles->loggedUsingMmaAttention = true;
        if(cudaHandles->logger != NULL)
          cudaHandles->logger->write("CUDA_TRANSFORMER_ROUTE attention=official_mma");
      }
    }

    // Interleaved QKV is an atomic construction-time plan: scalar attention
    // cannot consume it. A rejected launch therefore signals an invariant bug,
    // rather than silently running with the wrong strides.
    if(!usedMma && useCombinedQKV) {
      // Construction preflight covers every rejection condition in the MMA
      // launcher. Reaching this point means that contract drifted after QKV
      // work was already enqueued, so fail fatally instead of throwing into a
      // caller that might attempt an unsafe fallback on interleaved tensors.
      std::terminate();
    }

    if(!usedMma) {
      if(!cudaHandles->loggedUsingScalarAttention) {
        cudaHandles->loggedUsingScalarAttention = true;
        if(cudaHandles->logger != NULL)
          cudaHandles->logger->write("CUDA_TRANSFORMER_ROUTE attention=official_scalar");
      }
      if(!usingFP16)
        customCudaFlashAttention(
          (const float*)qPtr,(const float*)kPtr,(const float*)vPtr,(const float*)maskBuf,
          (float*)attnOutBuf.buf,batchSize,seqLen,numHeads,numKVHeads,qHeadDim,vHeadDim,cudaHandles->stream
        );
      else
        customCudaFlashAttention(
          (const half*)qPtr,(const half*)kPtr,(const half*)vPtr,(const half*)maskBuf,
          (half*)attnOutBuf.buf,batchSize,seqLen,numHeads,numKVHeads,qHeadDim,vHeadDim,cudaHandles->stream
        );
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }

    if(maskBuf == NULL) {
      outProj.apply(
        cudaHandles,scratch,matBatchSize,attnOutBuf.buf,trunkBuf,workspaceBuf,workspaceBytes,true
      );
    }
    else {
      outProj.apply(
        cudaHandles,scratch,matBatchSize,attnOutBuf.buf,trunkScratchBuf,workspaceBuf,workspaceBytes
      );
      if(!usingFP16)
        customCudaMaskedResidualAddNHWC(
          (float*)trunkBuf,(const float*)trunkScratchBuf,(const float*)maskBuf,
          batchSize,seqLen,inChannels,cudaHandles->stream
        );
      else
        customCudaMaskedResidualAddNHWC(
          (half*)trunkBuf,(const half*)trunkScratchBuf,(const half*)maskBuf,
          batchSize,seqLen,inChannels,cudaHandles->stream
        );
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }

#ifdef KATAGO_BUILD_BENCHMARKNN
    BenchmarkRouteCountsInternal& active = cudaHandles->benchmarkRoute.current;
    active.attention += 1;
    if(useCombinedQKV)
      active.combinedQKV += 1;
    else
      active.planar += 1;
    if(useRope) {
      if(learnableRope)
        active.learnedRopeFp32 += 1;
      else
        active.fixedRope += 1;
    }
    if(usedMma)
      active.mma += 1;
    else
      active.scalar += 1;
    const bool mmaWasPrepared =
      usingFP16 &&
      cudaHandles->mmaAttentionEnabled &&
      customCudaFlashAttentionMmaSupportsShape(numHeads,numKVHeads,qHeadDim,vHeadDim);
    if(mmaWasPrepared && !usedMma)
      active.fallback += 1;
#endif
  }
};

//------------------------------------------------------------------------------

struct TransformerFFNBlock {
  const string name;
  const int numChannels;
  const int ffnChannels;
  const bool useSwiGLU;
  const float swigluClip;
  const V105CudaPolicy::SwiGLUPlan swiGLUPlan;

  const int nnXLen;
  const int nnYLen;
  const bool usingFP16;
  const bool usingNHWC;
  const V105CudaPolicy::ProjectedScratchLayout projectedScratchLayout;

  const TransformerRMSNormLayer preLN;
  const MatMulLayer linear2;
  OwnedDeviceBuf ffnPackedWeights;
#if defined(KATAGO_CUDA_FUSED_FFN_AVAILABLE) && KATAGO_CUDA_FUSED_FFN_AVAILABLE
  OwnedDeviceBuf fusedFfnWeights;
  bool fusedFfnPrepared;
#endif

  TransformerFFNBlock() = delete;
  TransformerFFNBlock(const TransformerFFNBlock&) = delete;
  TransformerFFNBlock& operator=(const TransformerFFNBlock&) = delete;

  TransformerFFNBlock(
    CudaHandles* cudaHandles,
    const TransformerFFNDesc* desc,
    int maxBatchSize,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ) :
    name(desc->name),
    numChannels(desc->numChannels),
    ffnChannels(desc->ffnChannels),
    useSwiGLU(desc->useSwiGLU),
    swigluClip(desc->swigluClip),
    swiGLUPlan(V105CudaPolicy::selectSwiGLUPlan(desc->swigluClip)),
    nnXLen(nnX),
    nnYLen(nnY),
    usingFP16(useFP16),
    usingNHWC(useNHWC),
    projectedScratchLayout(V105CudaPolicy::makeProjectedScratchLayout(
      (size_t)maxBatchSize,(size_t)nnX,(size_t)nnY,(size_t)desc->ffnChannels,
      useFP16 ? sizeof(half) : sizeof(float)
    )),
    preLN(cudaHandles,&desc->preLN,useFP16),
    linear2(cudaHandles,&desc->linear2,useFP16)
#if defined(KATAGO_CUDA_FUSED_FFN_AVAILABLE) && KATAGO_CUDA_FUSED_FFN_AVAILABLE
    ,fusedFfnPrepared(false)
#endif
  {
    if(!useSwiGLU)
      throw StringError("Non-SwiGLU transformer FFN is not supported by the CUDA backend");
    if(!useNHWC)
      throw StringError("Transformer blocks with NCHW layout are not supported by the CUDA backend");
    if(swiGLUPlan == V105CudaPolicy::SwiGLUPlan::OrderedClippedFP32 && !useFP16)
      throw StringError(name + ": positive SwiGLU clip requires FP16");
    if(
      desc->linear1.inChannels != desc->linearGate.inChannels ||
      desc->linear1.outChannels != desc->linearGate.outChannels
    )
      throw StringError(name + ": FFN linear and gate projections must share a shape");

    vector<float> packed;
    packed.reserve(desc->linear1.weights.size() + desc->linearGate.weights.size());
    packed.insert(packed.end(),desc->linear1.weights.begin(),desc->linear1.weights.end());
    packed.insert(packed.end(),desc->linearGate.weights.begin(),desc->linearGate.weights.end());
    CudaUtils::mallocAndCopyToDevice(name + ":ffnPacked",packed,ffnPackedWeights.buf,useFP16);

#if defined(KATAGO_CUDA_FUSED_FFN_AVAILABLE) && KATAGO_CUDA_FUSED_FFN_AVAILABLE
    // Positive clip has an exact ordered-FP32 epilogue in this CUTLASS
    // family. Keep clip0 on the unchanged official helper, avoiding an
    // unnecessary numerical recipe change for v102 models.
    const long long maxRows64 =
      (long long)maxBatchSize * nnX * nnY;
    if(cudaHandles->genericFusedFfnAvailable && useFP16 &&
       swiGLUPlan == V105CudaPolicy::SwiGLUPlan::OrderedClippedFP32 &&
       maxRows64 > 0 && maxRows64 <= 2147483647LL &&
       CudaFusedFFN::supportsProblem(
         (int)maxRows64,ffnChannels,numChannels,swigluClip)) {
      vector<float> fusedPacked(
        (size_t)2 * (size_t)ffnChannels * (size_t)numChannels);
      for(int out = 0; out < ffnChannels; out++) {
        for(int in = 0; in < numChannels; in++) {
          const size_t source = (size_t)in * ffnChannels + out;
          const size_t destination = (size_t)out * numChannels + in;
          fusedPacked[destination] = desc->linear1.weights[source];
          fusedPacked[(size_t)ffnChannels * numChannels + destination] =
            desc->linearGate.weights[source];
        }
      }
      CudaUtils::mallocAndCopyToDevice(
        name + ":genericFusedFfn",fusedPacked,fusedFfnWeights.buf,true);
      const half* const up = (const half*)fusedFfnWeights.buf;
      const half* const gate = up + (size_t)ffnChannels * numChannels;
      if(CudaFusedFFN::supportsPreparedWeights(
           up,gate,(int)maxRows64,ffnChannels,numChannels,swigluClip)) {
        fusedFfnPrepared = true;
        if(!cudaHandles->loggedUsingGenericFusedFfn) {
          cudaHandles->loggedUsingGenericFusedFfn = true;
          if(cudaHandles->logger != NULL)
            cudaHandles->logger->write(
              "CUDA_TRANSFORMER_ROUTE ffn=cutlass_fused_clipped_fp16");
        }
      }
      else {
        (void)cudaFree(fusedFfnWeights.buf);
        fusedFfnWeights.buf = nullptr;
      }
    }
#endif
  }

  ~TransformerFFNBlock() {}

  size_t requiredWorkspaceBytes(CudaHandles* cudaHandles,int batchSize) const {
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

    const int seqLen = nnXLen * nnYLen;
    const int matBatchSize = batchSize * seqLen;
    preLN.apply(cudaHandles,batchSize,seqLen,trunkBuf,trunkScratchBuf,maskBuf);

    // The optional CUTLASS path writes the SwiGLU product directly to the
    // first plane. Unsupported runtime rows/pointers fall back before any
    // fused work is enqueued.
    SizedBuf<void*> projected(scratch->allocator,projectedScratchLayout.totalBytes);
    void* linearBuf = projected.buf;
    void* gateBuf = (char*)projected.buf + projectedScratchLayout.planeStrideBytes;
    const size_t totalElements = (size_t)ffnChannels * (size_t)matBatchSize;
    if(totalElements > projectedScratchLayout.planeElements ||
       totalElements > (size_t)std::numeric_limits<int>::max())
      throw StringError(name + ": FFN actual batch exceeds the constructed scratch layout");
    const int totalSize = (int)totalElements;
    bool usedFusedFfn = false;
#if defined(KATAGO_CUDA_FUSED_FFN_AVAILABLE) && KATAGO_CUDA_FUSED_FFN_AVAILABLE
    if(fusedFfnPrepared) {
      const half* const up = (const half*)fusedFfnWeights.buf;
      const half* const gate = up + (size_t)ffnChannels * numChannels;
      if(CudaFusedFFN::canImplement(
           (const half*)trunkScratchBuf,up,gate,(half*)linearBuf,
           matBatchSize,ffnChannels,numChannels,swigluClip)) {
        CudaFusedFFN::runSwiGLU(
          (const half*)trunkScratchBuf,up,gate,(half*)linearBuf,
          matBatchSize,ffnChannels,numChannels,swigluClip,cudaHandles->stream);
        usedFusedFfn = true;
      }
    }
#endif
    if(!usedFusedFfn) {
      applySharedInputStridedMatMuls(
        cudaHandles,scratch,usingFP16,name,
        ffnChannels,matBatchSize,numChannels,
        ffnPackedWeights.buf,trunkScratchBuf,linearBuf,
        (long long)projectedScratchLayout.planeStrideElements,2
      );
      if(!usingFP16)
        customCudaSwiGLU(
          (const float*)linearBuf,(const float*)gateBuf,(float*)linearBuf,totalSize,cudaHandles->stream
        );
      else if(swiGLUPlan == V105CudaPolicy::SwiGLUPlan::LegacyUnclipped)
        customCudaSwiGLU(
          (const half*)linearBuf,(const half*)gateBuf,(half*)linearBuf,totalSize,cudaHandles->stream
        );
      else
        customCudaSwiGLUOrderedClippedFP16(
          (const half*)linearBuf,(const half*)gateBuf,(half*)linearBuf,
          totalSize,swigluClip,cudaHandles->stream
        );
    }
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());
#ifdef KATAGO_BUILD_BENCHMARKNN
    // Publish the ordered-clip marker only after its helper launch was
    // enqueued and cudaPeekAtLastError accepted it.
    if(swiGLUPlan == V105CudaPolicy::SwiGLUPlan::OrderedClippedFP32)
      cudaHandles->benchmarkRoute.current.orderedClippedSwiGLU += 1;
#endif

    if(maskBuf == NULL) {
      linear2.apply(
        cudaHandles,scratch,matBatchSize,linearBuf,trunkBuf,workspaceBuf,workspaceBytes,true
      );
    }
    else {
      linear2.apply(
        cudaHandles,scratch,matBatchSize,linearBuf,trunkScratchBuf,workspaceBuf,workspaceBytes
      );
      if(!usingFP16)
        customCudaMaskedResidualAddNHWC(
          (float*)trunkBuf,(const float*)trunkScratchBuf,(const float*)maskBuf,
          batchSize,seqLen,numChannels,cudaHandles->stream
        );
      else
        customCudaMaskedResidualAddNHWC(
          (half*)trunkBuf,(const half*)trunkScratchBuf,(const half*)maskBuf,
          batchSize,seqLen,numChannels,cudaHandles->stream
        );
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }

#ifdef KATAGO_BUILD_BENCHMARKNN
    cudaHandles->benchmarkRoute.current.ffn += 1;
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
  bool useNHWC
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
          useNHWC
        )
      );
      blocks.push_back(make_pair(NESTED_BOTTLENECK_BLOCK_KIND,std::move(blockPtr)));
    }
    else if(descBlocks[i].first == TRANSFORMER_ATTENTION_BLOCK_KIND) {
      TransformerAttentionDesc* blockDesc = (TransformerAttentionDesc*)descBlocks[i].second.get();
      unique_ptr_void blockPtr = make_unique_void(
        new TransformerAttentionBlock(
          cudaHandles,
          blockDesc,
          manager->maxBatchSize,
          nnXLen,
          nnYLen,
          useFP16,
          useNHWC
        )
      );
      blocks.push_back(make_pair(TRANSFORMER_ATTENTION_BLOCK_KIND,std::move(blockPtr)));
    }
    else if(descBlocks[i].first == TRANSFORMER_FFN_BLOCK_KIND) {
      TransformerFFNDesc* blockDesc = (TransformerFFNDesc*)descBlocks[i].second.get();
      // TransformerFFNBlock::apply passes the flattened activation count to
      // the SwiGLU launcher as int. Reject oversized generic shapes before
      // constructing the layer or uploading any of its device weights.
      CudaUtils::checkBufferSize(
        manager->maxBatchSize,nnXLen,nnYLen,blockDesc->ffnChannels
      );
      unique_ptr_void blockPtr = make_unique_void(
        new TransformerFFNBlock(
          cudaHandles,
          blockDesc,
          manager->maxBatchSize,
          nnXLen,
          nnYLen,
          useFP16,
          useNHWC
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

#ifdef KATAGO_BUILD_BENCHMARKNN
static void collectBenchmarkExpectedRoutes(
  const std::vector<std::pair<int, unique_ptr_void>>& descBlocks,
  BenchmarkRouteCountsInternal& counts
) {
  for(const auto& descBlock: descBlocks) {
    if(descBlock.first == TRANSFORMER_ATTENTION_BLOCK_KIND) {
      counts.attention += 1;
      const TransformerAttentionDesc* attention =
        (const TransformerAttentionDesc*)descBlock.second.get();
      if(attention->useQKNorm)
        counts.qkn += 1;
    }
    else if(descBlock.first == TRANSFORMER_FFN_BLOCK_KIND) {
      counts.ffn += 1;
      const TransformerFFNDesc* ffn =
        (const TransformerFFNDesc*)descBlock.second.get();
      if(ffn->swigluClip > 0.0f)
        counts.orderedClippedSwiGLU += 1;
    }
    else if(descBlock.first == NESTED_BOTTLENECK_BLOCK_KIND) {
      const NestedBottleneckResidualBlockDesc* nested =
        (const NestedBottleneckResidualBlockDesc*)descBlock.second.get();
      collectBenchmarkExpectedRoutes(nested->blocks,counts);
    }
  }
}

void BlockStack::collectBenchmarkPreparedRoutes(
  const CudaHandles* cudaHandles,
  BenchmarkRouteCountsInternal& counts
) const {
  for(const auto& blockEntry: blocks) {
    if(blockEntry.first == TRANSFORMER_ATTENTION_BLOCK_KIND) {
      const TransformerAttentionBlock* block =
        (const TransformerAttentionBlock*)blockEntry.second.get();
      counts.attention += 1;
      if(block->useQKNorm)
        counts.qkn += 1;
      if(block->useCombinedQKV)
        counts.combinedQKV += 1;
      else
        counts.planar += 1;
      if(block->useRope) {
        if(block->learnableRope)
          counts.learnedRopeFp32 += 1;
        else
          counts.fixedRope += 1;
      }
      const bool mmaPrepared =
        block->usingFP16 &&
        cudaHandles->mmaAttentionEnabled &&
        customCudaFlashAttentionMmaSupportsShape(
          block->numHeads,block->numKVHeads,block->qHeadDim,block->vHeadDim
        );
      if(mmaPrepared)
        counts.mma += 1;
      else
        counts.scalar += 1;
    }
    else if(blockEntry.first == TRANSFORMER_FFN_BLOCK_KIND) {
      counts.ffn += 1;
      const TransformerFFNBlock* block =
        (const TransformerFFNBlock*)blockEntry.second.get();
      if(block->swiGLUPlan == V105CudaPolicy::SwiGLUPlan::OrderedClippedFP32)
        counts.orderedClippedSwiGLU += 1;
    }
    else if(blockEntry.first == NESTED_BOTTLENECK_BLOCK_KIND) {
      const NestedBottleneckResidualBlock* nested =
        (const NestedBottleneckResidualBlock*)blockEntry.second.get();
      nested->blocks.collectBenchmarkPreparedRoutes(cudaHandles,counts);
    }
  }
}
#endif

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
  size_t workspaceBytes,
  FourProfile::ManagerV1* fourProfileManager,
  const FourProfile::RuntimeKeyV1* fourProfileRuntime
) const {

  FourProfile::RouteV1 fourProfileRoute = FourProfile::RouteV1::Official;
  FourProfile::RuntimeCallV1 fourProfileCall;
  if(fourProfileManager != nullptr && fourProfileRuntime != nullptr) {
    const FourProfile::BuildReportV1& report = fourProfileManager->report();
    fourProfileCall.key = *fourProfileRuntime;
    fourProfileCall.actualBatchSize = batchSize;
    fourProfileCall.sequenceSize = nnXLen * nnYLen;
    fourProfileCall.transformerBeginBlock = report.span.beginBlock;
    fourProfileCall.transformerPairCount = report.span.pairCount;
    fourProfileCall.stream = reinterpret_cast<void*>(cudaHandles->stream.stream);
    fourProfileCall.trunk = trunkBuf;
    fourProfileCall.trunkScratch = trunkScratchBuf;
    fourProfileCall.mask = maskBuf;
    fourProfileCall.workspace = workspaceBuf;
    fourProfileCall.workspaceBytes = workspaceBytes;

    // H2D conversion and common official setup may already be queued, but no
    // provider-specific inference work has been queued. The provider must
    // complete this full-identity, zero-enqueue preflight before its first
    // specialized operation. An Auto miss therefore safely executes this
    // same official block span exactly once.
    fourProfileRoute = fourProfileManager->preflight(fourProfileCall);
  }

  for(int i = 0; i<blocks.size(); i++) {
    if(fourProfileRoute == FourProfile::RouteV1::Specialized &&
       static_cast<size_t>(i) == fourProfileCall.transformerBeginBlock) {
      const size_t end = fourProfileCall.transformerBeginBlock +
        fourProfileCall.transformerPairCount * 2;
      if(end > blocks.size() || end <= static_cast<size_t>(i))
        throw FourProfile::FatalErrorV1("four-profile committed span is outside CUDA block stack");
      const FourProfile::RouteV1 enqueueRoute =
        fourProfileManager->enqueue(fourProfileCall);
      if(enqueueRoute == FourProfile::RouteV1::Specialized) {
#ifdef KATAGO_BUILD_BENCHMARKNN
        // A whole-span provider performs these logical transformer operations
        // without entering the official per-layer implementations that update
        // the benchmark counters. Publish the covered logical counts here so
        // the test-only harness can validate and time the specialized route.
        cudaHandles->benchmarkRoute.current.attention =
          cudaHandles->benchmarkRoute.expected.attention;
        cudaHandles->benchmarkRoute.current.ffn =
          cudaHandles->benchmarkRoute.expected.ffn;
        cudaHandles->benchmarkRoute.current.qkn =
          cudaHandles->benchmarkRoute.expected.qkn;
        cudaHandles->benchmarkRoute.current.orderedClippedSwiGLU =
          cudaHandles->benchmarkRoute.expected.orderedClippedSwiGLU;
#endif
        i = static_cast<int>(end) - 1;
        continue;
      }
      // Only an explicit provider failure with enqueued==0 reaches here.
      // Execute the official transformer blocks from the current index.
      fourProfileRoute = FourProfile::RouteV1::Official;
    }
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
    bool useNHWC
  ) :
    name(desc->name),
    version(desc->version),
    numBlocks(desc->numBlocks),
    trunkNumChannels(desc->trunkNumChannels),
    nnXLen(nnX),
    nnYLen(nnY),
    usingFP16(useFP16),
    usingNHWC(useNHWC),
    blocks(cudaHandles,manager,desc->numBlocks,desc->trunkNumChannels,desc->blocks,nnX,nnY,useFP16,useNHWC)
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
    size_t workspaceBytes,
    FourProfile::ManagerV1* fourProfileManager,
    const FourProfile::RuntimeKeyV1* fourProfileRuntime
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
      workspaceBytes,
      fourProfileManager,
      fourProfileRuntime
    );

    //And now with the final BN port it from trunkScratch.buf to trunkBuf.
    trunkTipBN->apply(cudaHandles,batchSize,trunkScratch.buf,maskBuf,trunkBuf);
    #ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint4D(string("Trunk tip"), trunkBuf, batchSize, trunkNumChannels, nnXLen, nnYLen, usingNHWC, usingFP16);
    #endif
  }

};

//------------------------------------------------------------------------------

static void fillMaskFloatBufAndMaskSumBuf(
  CudaHandles* cudaHandles,
  void* maskBuf,
  float*& maskFloatBuf,
  float*& maskSumBuf,
  bool usingFP16,
  int batchSize,
  int nnXLen,
  int nnYLen
) {
  if(!usingFP16) {
    maskFloatBuf = (float*)maskBuf;
    customCudaPoolRowsSumNCHW(
      (const float*)maskFloatBuf,maskSumBuf,batchSize,1,nnXLen*nnYLen,1.0,cudaHandles->stream
    );
    CUDA_ERR("sumMask",cudaPeekAtLastError());
  }
  else {
    customCudaCopyFromHalf(
      (const half*)maskBuf,maskFloatBuf,batchSize*nnXLen*nnYLen,cudaHandles->stream
    );
    CUDA_ERR("copyMaskFromHalf",cudaPeekAtLastError());
    customCudaPoolRowsSumNCHW(
      (const float*)maskFloatBuf,maskSumBuf,batchSize,1,nnXLen*nnYLen,1.0,cudaHandles->stream
    );
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
      customCudaCopyFromHalf(
        (const half*)g1Out2.buf,(float*)workspaceBuf,
        batchSize*g1Channels*nnXLen*nnYLen,cudaHandles->stream
      );
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
      customCudaCopyFromHalf(
        (const half*)p1Out.buf,(float*)p1Out2.buf,
        batchSize*p1Channels*nnXLen*nnYLen,cudaHandles->stream
      );
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
      customCudaCopyFromHalf(
        (const half*)v1Out2.buf,(float*)workspaceBuf,
        batchSize*v1Channels*nnXLen*nnYLen,cudaHandles->stream
      );
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
      customCudaCopyFromHalf(
        (const half*)ownershipScratch.buf,(float*)ownershipBuf,
        batchSize*ownershipChannels*nnXLen*nnYLen,cudaHandles->stream
      );
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
    trunk = std::make_unique<Trunk>(cudaHandles,manager.get(),&desc->trunk,nnXLen,nnYLen,inputsUseNHWC,useFP16,useNHWC);
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
    size_t workspaceBytes,
    FourProfile::ManagerV1* fourProfileManager = nullptr,
    const FourProfile::RuntimeKeyV1* fourProfileRuntime = nullptr
  ) const {
#ifdef KATAGO_BUILD_BENCHMARKNN
    cudaHandles->benchmarkRoute.beginInvocation();
#endif
    SizedBuf<void*> mask(scratch->allocator, scratch->getBufSizeXY(1));
    SizedBuf<void*> maskFloat(scratch->allocator, scratch->getBufSizeXYFloat(1));
    SizedBuf<void*> maskSum(scratch->allocator, scratch->getBufSizeFloat(1));

    void* maskBuf = mask.buf;
    float* maskFloatBuf = (float*)maskFloat.buf;
    float* maskSumBuf = (float*)maskSum.buf;

    if(!usingFP16) {
      if(inputsUsingNHWC)
        customCudaChannel0ExtractNHWC(
          (const float*)inputBuf,(float*)maskBuf,batchSize,nnXLen*nnYLen,numInputChannels,cudaHandles->stream
        );
      else
        customCudaChannel0ExtractNCHW(
          (const float*)inputBuf,(float*)maskBuf,batchSize,numInputChannels,nnXLen*nnYLen,cudaHandles->stream
        );
      CUDA_ERR("modelExtractMask",cudaPeekAtLastError());
    }
    else {
      if(inputsUsingNHWC)
        customCudaChannel0ExtractNHWC(
          (const half*)inputBuf,(half*)maskBuf,batchSize,nnXLen*nnYLen,numInputChannels,cudaHandles->stream
        );
      else
        customCudaChannel0ExtractNCHW(
          (const half*)inputBuf,(half*)maskBuf,batchSize,numInputChannels,nnXLen*nnYLen,cudaHandles->stream
        );
      CUDA_ERR("modelExtractMask",cudaPeekAtLastError());
    }

    fillMaskFloatBufAndMaskSumBuf(
      cudaHandles,maskBuf,maskFloatBuf,maskSumBuf,usingFP16,batchSize,nnXLen,nnYLen
    );

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
      workspaceBytes,
      fourProfileManager,
      fourProfileRuntime
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
#ifdef KATAGO_BUILD_BENCHMARKNN
    // This only seals a candidate. getOutput publishes it after the handle
    // stream synchronizes successfully, so asynchronous failures leave the
    // preceding successful snapshot and serial untouched.
    cudaHandles->benchmarkRoute.finishModelApply(
      batchSize,usingFP16,usingNHWC,requireExactNNLen,maskBuf == NULL
    );
#endif
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

  Buffers(CudaHandles* cudaHandles, const Model& m, const ScratchBuffers& scratch)
    : inputBufFloat(nullptr),
      inputBuf(nullptr),
      inputGlobalBufFloat(nullptr),
      inputGlobalBuf(nullptr),
      inputBufBytesFloat(0),
      inputBufBytes(0),
      inputGlobalBufBytesFloat(0),
      inputGlobalBufBytes(0),
      policyBuf(nullptr),
      policyBufBytes(0),
      valueBuf(nullptr),
      valueBufBytes(0),
      scoreValueBuf(nullptr),
      scoreValueBufBytes(0),
      ownershipBuf(nullptr),
      ownershipBufBytes(0),
      workspaceBuf(nullptr),
      workspaceBytes(0)
  {
    try {
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
    catch(...) {
      freeDeviceBuffers();
      throw;
    }
  }

  ~Buffers() {
    freeDeviceBuffers();
  }

  void freeDeviceBuffers() noexcept {
    if(workspaceBuf != nullptr) {
      (void)cudaFree(workspaceBuf);
      workspaceBuf = nullptr;
    }
    if(ownershipBuf != nullptr) {
      (void)cudaFree(ownershipBuf);
      ownershipBuf = nullptr;
    }
    if(scoreValueBuf != nullptr) {
      (void)cudaFree(scoreValueBuf);
      scoreValueBuf = nullptr;
    }
    if(valueBuf != nullptr) {
      (void)cudaFree(valueBuf);
      valueBuf = nullptr;
    }
    if(policyBuf != nullptr) {
      (void)cudaFree(policyBuf);
      policyBuf = nullptr;
    }
    if(inputGlobalBuf != nullptr) {
      (void)cudaFree(inputGlobalBuf);
      inputGlobalBuf = nullptr;
    }
    if(inputGlobalBufFloat != nullptr) {
      (void)cudaFree(inputGlobalBufFloat);
      inputGlobalBufFloat = nullptr;
    }
    if(inputBuf != nullptr) {
      (void)cudaFree(inputBuf);
      inputBuf = nullptr;
    }
    if(inputBufFloat != nullptr) {
      (void)cudaFree(inputBufFloat);
      inputBufFloat = nullptr;
    }
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
  if(loadedModel->modelDesc.version == 105 && useFP16Mode == enabled_t::False)
    V105CudaPolicy::requireCurrentExecution(
      loadedModel->modelDesc.version,loadedModel->modelDesc.trunk,false
    );
  (void)gpuIdxs;
  (void)openCLTunerFile;
  (void)homeDataDirOverride;
  (void)openCLReTunePerBoardSize;

  const CudaInt8Policy requestedInt8 = resolveCudaInt8Policy(
    useINT8,std::getenv("KATAGO_DISABLE_INT8"));
#if defined(KATAGO_P4_PROVIDER_COMPILED) && KATAGO_P4_PROVIDER_COMPILED
  const bool int8Compiled = true;
#else
  const bool int8Compiled = false;
#endif
  const bool allowINT8 = requestedInt8.enabled && int8Compiled;
  if(logger != nullptr) {
    logger->write(
      string("CUDA_INT8_POLICY config=") +
      (requestedInt8.configEnabled ? "1" : "0") +
      " env_disabled=" + (requestedInt8.environmentDisabled ? "1" : "0") +
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

#ifdef KATAGO_BUILD_BENCHMARKNN
static bool benchmarkRouteHasTransformer(const BenchmarkRouteStateInternal& route) noexcept {
  return route.expected.attention > 0 || route.expected.ffn > 0;
}

static void logBenchmarkPreparedRouteNoThrow(CudaHandles* cudaHandles) noexcept {
  BenchmarkRouteStateInternal& route = cudaHandles->benchmarkRoute;
  if(route.loggedPrepared || !benchmarkRouteHasTransformer(route))
    return;
  route.loggedPrepared = true;
  if(cudaHandles->logger == NULL)
    return;
  try {
    const BenchmarkRouteCountsInternal& expected = route.expected;
    const BenchmarkRouteCountsInternal& prepared = route.preparedCounts;
    cudaHandles->logger->write(
      "CUDA_TRANSFORMER_ROUTE_PROOF phase=PREPARED prepared=1 expected_attention=" + Global::intToString(expected.attention) +
      " prepared_attention=" + Global::intToString(prepared.attention) +
      " expected_ffn=" + Global::intToString(expected.ffn) +
      " prepared_ffn=" + Global::intToString(prepared.ffn) +
      " expected_qkn=" + Global::intToString(expected.qkn) +
      " prepared_qkn=" + Global::intToString(prepared.qkn) +
      " expected_ordered_clipped_swiglu=" + Global::intToString(expected.orderedClippedSwiGLU) +
      " prepared_ordered_clipped_swiglu=" + Global::intToString(prepared.orderedClippedSwiGLU) +
      " combined_qkv=" + Global::intToString(prepared.combinedQKV) +
      " learned_rope_fp32=" + Global::intToString(prepared.learnedRopeFp32) +
      " fixed_rope=" + Global::intToString(prepared.fixedRope) +
      " mma=" + Global::intToString(prepared.mma) +
      " scalar=" + Global::intToString(prepared.scalar) +
      " planar=" + Global::intToString(prepared.planar) +
      " cudnn=" + Global::intToString(prepared.cudnn) +
      " fallback=" + Global::intToString(prepared.fallback) +
      " board=" + Global::intToString(route.nnXLen) + "x" + Global::intToString(route.nnYLen)
    );
  }
  catch(...) {
    // Route diagnostics are observational and must never affect inference.
  }
}

static void logBenchmarkActiveRouteNoThrow(CudaHandles* cudaHandles) noexcept {
  BenchmarkRouteStateInternal& route = cudaHandles->benchmarkRoute;
  if(route.loggedActive || !benchmarkRouteHasTransformer(route) || !route.hasSuccessfulInvocation)
    return;
  route.loggedActive = true;
  if(cudaHandles->logger == NULL)
    return;
  try {
    const BenchmarkRouteCountsInternal& active = route.last;
    cudaHandles->logger->write(
      "CUDA_TRANSFORMER_ROUTE_PROOF phase=ACTIVE invocation=" + Global::uint64ToString(route.invocationSerial) +
      " attention=" + Global::intToString(active.attention) + "/" + Global::intToString(route.expected.attention) +
      " ffn=" + Global::intToString(active.ffn) + "/" + Global::intToString(route.expected.ffn) +
      " qkn=" + Global::intToString(active.qkn) + "/" + Global::intToString(route.expected.qkn) +
      " ordered_clipped_swiglu=" + Global::intToString(active.orderedClippedSwiGLU) + "/" +
        Global::intToString(route.expected.orderedClippedSwiGLU) +
      " combined_qkv=" + Global::intToString(active.combinedQKV) +
      " learned_rope_fp32=" + Global::intToString(active.learnedRopeFp32) +
      " fixed_rope=" + Global::intToString(active.fixedRope) +
      " mma=" + Global::intToString(active.mma) +
      " scalar=" + Global::intToString(active.scalar) +
      " planar=" + Global::intToString(active.planar) +
      " cudnn=" + Global::intToString(active.cudnn) +
      " fallback=" + Global::intToString(active.fallback) +
      " fp16=" + Global::boolToString(route.lastFp16) +
      " nhwc=" + Global::boolToString(route.lastNhwc) +
      " exact=" + Global::boolToString(route.lastExact) +
      " mask_null=" + Global::boolToString(route.lastMaskNull) +
      " batch=" + Global::intToString(route.lastBatchSize) +
      " board=" + Global::intToString(route.nnXLen) + "x" + Global::intToString(route.nnYLen)
    );
  }
  catch(...) {
    // Route diagnostics are observational and must never affect inference.
  }
}
#endif

//------------------------------------------------------------------------------

static FourProfile::RuntimeKeyV1 makeFourProfileRuntimeKey(
  int majorComputeCapability,
  int minorComputeCapability,
  int nnXLen,
  int nnYLen,
  int physicalBatchSize,
  int sameGpuConcurrency,
  bool requireExactNNLen,
  bool useFP16,
  bool useNHWC,
  bool useINT8
) {
  FourProfile::RuntimeKeyV1 key;
  key.deviceComputeCapability = majorComputeCapability * 10 + minorComputeCapability;
  key.boardX = nnXLen;
  key.boardY = nnYLen;
  key.physicalBatchSize = physicalBatchSize;
  key.sameGpuConcurrency = sameGpuConcurrency;
  key.exactBoard = requireExactNNLen;
  key.maskMode = requireExactNNLen ?
    FourProfile::MaskModeV1::None : FourProfile::MaskModeV1::Dense;
  key.maskNull = requireExactNNLen;
  key.inputStorage = useFP16 ?
    FourProfile::StorageTypeV1::Fp16 : FourProfile::StorageTypeV1::Fp32;
  key.outputStorage = key.inputStorage;
  key.requestedExecution = useINT8 ?
    FourProfile::RequestedExecutionV1::Int8 :
    FourProfile::RequestedExecutionV1::Fp16;
  key.layout = useNHWC ?
    FourProfile::TensorLayoutV1::Nhwc : FourProfile::TensorLayoutV1::Nchw;
  return key;
}

static FourProfile::NativeExecutableBlocksV1 collectFourProfileExecutables(
  const BlockStack& blocks
) {
  FourProfile::NativeExecutableBlocksV1 result;
  result.reserve(blocks.blocks.size());
  for(const auto& block: blocks.blocks)
    result.emplace_back(block.first,block.second.get());
  return result;
}

//------------------------------------------------------------------------------

struct ComputeHandle {
  std::unique_ptr<CudaHandles> cudaHandles;
  std::unique_ptr<Model> model;
  std::unique_ptr<ScratchBuffers> scratch;
  std::unique_ptr<Buffers> buffers;
  // Declared after every official resource so ordinary member unwinding also
  // destroys provider plans first if constructor work below later throws.
  std::unique_ptr<FourProfile::ManagerV1> fourProfileManager;
  FourProfile::RuntimeKeyV1 fourProfileRuntime;
  const bool usingFP16;
  const int nnXLen;
  const int nnYLen;
  const bool requireExactNNLen;
  const bool inputsUseNHWC;
  const int policySize;

  ComputeHandle(
    const ComputeContext* context,
    const LoadedModel* loadedModel,
    Logger* logger,
    int majorComputeCapability,
    int minorComputeCapability,
    int maxBatchSize,
    int sameGpuConcurrency,
    FourProfile::ModeV1 fourProfileMode,
    bool requireExactNNLen_,
    bool inputsUseNHWC_,
    bool useFP16,
    bool useNHWC
  ) :
    usingFP16(useFP16),
    nnXLen(context->nnXLen),
    nnYLen(context->nnYLen),
    requireExactNNLen(requireExactNNLen_),
    inputsUseNHWC(inputsUseNHWC_),
    policySize(NNPos::getPolicySize(context->nnXLen, context->nnYLen))
  {
    V105CudaPolicy::requireCurrentExecution(
      loadedModel->modelDesc.version,loadedModel->modelDesc.trunk,useFP16
    );
#ifdef KATAGO_BUILD_BENCHMARKNN
    BenchmarkRouteCountsInternal expectedRoute;
    BenchmarkRouteCountsInternal preparedRoute;
    collectBenchmarkExpectedRoutes(loadedModel->modelDesc.trunk.blocks,expectedRoute);
#endif
    cudaHandles = std::make_unique<CudaHandles>(majorComputeCapability,minorComputeCapability);
    cudaHandles->logger = logger;
    // Probe before constructing the model because an enabled MMA path commits
    // attention weights to the official interleaved combined-QKV layout.
    const bool wantMmaAttention =
      useFP16 && majorComputeCapability >= 8 && loadedModel->modelDesc.trunk.hasAnyTransformerBlocks();
    cudaHandles->mmaAttentionEnabled =
      wantMmaAttention && customCudaFlashAttentionMmaSupported();
    if(wantMmaAttention && !cudaHandles->mmaAttentionEnabled) {
      const string warning =
        "WARNING CUDA_TRANSFORMER_ROUTE attention=official_mma_probe_failed fallback=official_scalar";
      if(logger != NULL)
        logger->write(warning);
      else
        std::cerr << warning << std::endl;
    }
    model = std::make_unique<Model>(
      cudaHandles.get(), &(loadedModel->modelDesc), maxBatchSize,
      nnXLen, nnYLen, inputsUseNHWC, useFP16, useNHWC
    );
#ifdef KATAGO_BUILD_BENCHMARKNN
    model->trunk->blocks.collectBenchmarkPreparedRoutes(cudaHandles.get(),preparedRoute);
    if(
      preparedRoute.attention != expectedRoute.attention ||
      preparedRoute.ffn != expectedRoute.ffn ||
      preparedRoute.qkn != expectedRoute.qkn ||
      preparedRoute.orderedClippedSwiGLU != expectedRoute.orderedClippedSwiGLU ||
      preparedRoute.combinedQKV + preparedRoute.planar != preparedRoute.attention ||
      preparedRoute.mma + preparedRoute.scalar + preparedRoute.cudnn != preparedRoute.attention
    ) {
      throw StringError("CUDA benchmark route proof: constructed transformer routes do not match the model descriptor");
    }
#endif
    scratch = std::make_unique<ScratchBuffers>(maxBatchSize, nnXLen, nnYLen, useFP16);
    buffers = std::make_unique<Buffers>(cudaHandles.get(), *model, *scratch);

    // Ensure handle-local setup has completed without serializing unrelated
    // handles on the same device.
    CUDA_ERR("ComputeHandle",cudaStreamSynchronize(cudaHandles->stream.stream));

    // The complete official Model, ScratchBuffers, and Buffers are all alive
    // and their setup stream is synchronized before any provider factory may
    // prepare or commit resources. Auto can therefore retain a fully formed
    // official route after any explicit zero-enqueue provider failure.
    fourProfileRuntime = makeFourProfileRuntimeKey(
      majorComputeCapability,minorComputeCapability,nnXLen,nnYLen,maxBatchSize,
      sameGpuConcurrency,requireExactNNLen,useFP16,useNHWC,
      context->useINT8 && loadedModel->modelDesc.version == 105
    );
    FourProfile::RegistryV1 fourProfileRegistry;
#if defined(KATAGO_P3_PROVIDER_COMPILED) && KATAGO_P3_PROVIDER_COMPILED && \
    defined(KATAGO_P4_PROVIDER_COMPILED) && KATAGO_P4_PROVIDER_COMPILED
    FourProfile::registerBuiltinStubFactoriesV1(
      fourProfileRegistry,FourProfile::makeP3FactoryV1(),
      FourProfile::makeP4FactoryV1());
#elif defined(KATAGO_P3_PROVIDER_COMPILED) && KATAGO_P3_PROVIDER_COMPILED
    FourProfile::registerBuiltinStubFactoriesV1(
      fourProfileRegistry,FourProfile::makeP3FactoryV1());
#elif defined(KATAGO_P4_PROVIDER_COMPILED) && KATAGO_P4_PROVIDER_COMPILED
    FourProfile::registerBuiltinStubFactoriesV1(
      fourProfileRegistry,nullptr,FourProfile::makeP4FactoryV1());
#else
    FourProfile::registerBuiltinStubFactoriesV1(fourProfileRegistry);
#endif
#if defined(KATAGO_P4_PROVIDER_COMPILED) && KATAGO_P4_PROVIDER_COMPILED
    fourProfileRegistry.add(FourProfile::makeGenericC384Int8FactoryV1());
#endif
    const FourProfile::ModelViewV1 fourProfileModel =
      FourProfile::buildNativeModelViewV1(
        loadedModel->modelDesc,collectFourProfileExecutables(model->trunk->blocks)
      );
    fourProfileManager = FourProfile::ManagerV1::create(
      fourProfileRegistry,fourProfileModel,fourProfileRuntime,fourProfileMode
    );
    if(logger != NULL) {
      logger->write(
        "CUDA_FOUR_PROFILE_ROUTE mode=" + string(FourProfile::modeNameV1(fourProfileMode)) +
        " route=" +
        string(fourProfileManager->report().route == FourProfile::RouteV1::Specialized ?
          "specialized" : "official") +
        " reason=" + FourProfile::reasonNameV1(fourProfileManager->report().reason) +
        " profile=" + fourProfileManager->report().profileId
      );
    }
#ifdef KATAGO_BUILD_BENCHMARKNN
    BenchmarkRouteStateInternal& route = cudaHandles->benchmarkRoute;
    route.expected = expectedRoute;
    route.preparedCounts = preparedRoute;
    route.streamIdentity = (uint64_t)(uintptr_t)cudaHandles->stream.stream;
    route.nnXLen = nnXLen;
    route.nnYLen = nnYLen;
    route.prepared = true;
    logBenchmarkPreparedRouteNoThrow(cudaHandles.get());
#endif
  }
  ~ComputeHandle() noexcept {
    // Device buffers and model weights must outlive every operation enqueued
    // on this handle. An asynchronous execution error cannot be recovered by
    // freeing those resources, so report it and fail fatally first.
    if(cudaHandles != nullptr) {
      cudaError_t status = cudaStreamSynchronize(cudaHandles->stream.stream);
      if(status != cudaSuccess) {
        std::cerr
          << "FATAL CUDA asynchronous error while destroying ComputeHandle: "
          << cudaGetErrorString(status) << std::endl;
        std::terminate();
      }
    }
    // Provider plans may retain non-owning references to every official
    // resource passed during prepare, so destroy them before any such target.
    fourProfileManager.reset();
    buffers.reset();
    scratch.reset();
    model.reset();
    cudaHandles.reset();
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
  int sameGpuConcurrency) {
  const char* fourProfileModeSetting = std::getenv("KATAGO_FOUR_PROFILE_MODE");
  const FourProfile::ModeV1 fourProfileMode =
    FourProfile::parseModeSettingV1(fourProfileModeSetting);
  if(logger != NULL) {
    logger->write(
      "CUDA_FOUR_PROFILE_MODE mode=" + string(FourProfile::modeNameV1(fourProfileMode)) +
      " source=" + (fourProfileModeSetting == nullptr ? string("default") : string("environment"))
    );
  }
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

  V105CudaPolicy::requireCurrentExecution(
    loadedModel->modelDesc.version,loadedModel->modelDesc.trunk,useFP16
  );

  ComputeHandle* gpuHandle = new ComputeHandle(
    context,loadedModel,logger,prop.major,prop.minor,maxBatchSize,sameGpuConcurrency,
    fourProfileMode,requireExactNNLen,inputsUseNHWC,useFP16,useNHWC
  );
  return gpuHandle;
}

void NeuralNet::freeComputeHandle(ComputeHandle* gpuHandle) {
  delete gpuHandle;
}

bool NeuralNet::isUsingFP16(const ComputeHandle* handle) {
  return handle->usingFP16;
}

#ifdef KATAGO_BUILD_BENCHMARKNN
bool NeuralNet::getBenchmarkRouteProof(
  const ComputeHandle* handle,
  BenchmarkRouteProof& proof
) {
  proof = BenchmarkRouteProof();
  if(handle == nullptr || handle->cudaHandles == nullptr)
    return false;

  const BenchmarkRouteStateInternal& route = handle->cudaHandles->benchmarkRoute;
  proof.prepared = route.prepared;
  proof.hasSuccessfulInvocation = route.hasSuccessfulInvocation;
  proof.invocationSerial = route.invocationSerial;
  proof.streamIdentity = route.streamIdentity;
  proof.nnXLen = route.nnXLen;
  proof.nnYLen = route.nnYLen;

  proof.expectedAttention = route.expected.attention;
  proof.expectedFfn = route.expected.ffn;
  proof.expectedQkn = route.expected.qkn;
  proof.expectedOrderedClippedSwiGLU = route.expected.orderedClippedSwiGLU;
  proof.preparedAttention = route.preparedCounts.attention;
  proof.preparedFfn = route.preparedCounts.ffn;
  proof.preparedQkn = route.preparedCounts.qkn;
  proof.preparedOrderedClippedSwiGLU = route.preparedCounts.orderedClippedSwiGLU;
  proof.preparedCombinedQKV = route.preparedCounts.combinedQKV;
  proof.preparedLearnedRopeFp32 = route.preparedCounts.learnedRopeFp32;
  proof.preparedFixedRope = route.preparedCounts.fixedRope;
  proof.preparedMma = route.preparedCounts.mma;
  proof.preparedScalar = route.preparedCounts.scalar;
  proof.preparedPlanar = route.preparedCounts.planar;
  proof.preparedCudnn = route.preparedCounts.cudnn;
  proof.preparedFallback = route.preparedCounts.fallback;

  proof.lastActiveAttention = route.last.attention;
  proof.lastActiveFfn = route.last.ffn;
  proof.lastActiveQkn = route.last.qkn;
  proof.lastActiveOrderedClippedSwiGLU = route.last.orderedClippedSwiGLU;
  proof.lastActiveCombinedQKV = route.last.combinedQKV;
  proof.lastActiveLearnedRopeFp32 = route.last.learnedRopeFp32;
  proof.lastActiveFixedRope = route.last.fixedRope;
  proof.lastActiveMma = route.last.mma;
  proof.lastActiveScalar = route.last.scalar;
  proof.lastActivePlanar = route.last.planar;
  proof.lastActiveCudnn = route.last.cudnn;
  proof.lastActiveFallback = route.last.fallback;
  proof.lastBatchSize = route.lastBatchSize;
  proof.lastFp16 = route.lastFp16;
  proof.lastNhwc = route.lastNhwc;
  proof.lastExact = route.lastExact;
  proof.lastMaskNull = route.lastMaskNull;
  return true;
}
#endif

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

  // Track allocation flavor so fallback pageable buffers are released with
  // free(), while pinned buffers use cudaFreeHost().
  std::vector<void*> pinnedPtrs;
  std::vector<void*> pageablePtrs;

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

    // Pinned host memory lets the H2D and D2H cudaMemcpyAsync operations use
    // true DMA and overlap work issued by independent ComputeHandles.
    userInputBuffer = nullptr;
    userInputGlobalBuffer = nullptr;
    policyResults = nullptr;
    valueResults = nullptr;
    scoreValueResults = nullptr;
    ownershipResults = nullptr;
    try {
      pinnedPtrs.reserve(6);
      pageablePtrs.reserve(6);
      userInputBuffer = mallocPinned(
        (size_t)m.numInputChannels * maxBatchSize * nnXLen * nnYLen
      );
      userInputGlobalBuffer = mallocPinned(
        (size_t)m.numInputGlobalChannels * maxBatchSize
      );
      policyResults = mallocPinned(
        (size_t)maxBatchSize * (1 + nnXLen * nnYLen)
      );
      valueResults = mallocPinned(
        (size_t)maxBatchSize * m.numValueChannels
      );
      scoreValueResults = mallocPinned(
        (size_t)maxBatchSize * m.numScoreValueChannels
      );
      ownershipResults = mallocPinned(
        (size_t)maxBatchSize * nnXLen * nnYLen * m.numOwnershipChannels
      );
    }
    catch(...) {
      freeBuffers();
      throw;
    }
  }

  float* mallocPinned(size_t numFloats) {
    void* buf = nullptr;
    if(cudaHostAlloc(&buf,numFloats * sizeof(float),cudaHostAllocPortable) == cudaSuccess) {
      pinnedPtrs.push_back(buf);
      return (float*)buf;
    }

    // cudaHostAlloc failures are recoverable here. Clear the sticky runtime
    // error before any later launch-error check, then degrade to pageable host
    // memory rather than failing model initialization.
    (void)cudaGetLastError();
    static std::atomic<bool> warned(false);
    if(!warned.exchange(true)) {
      std::cerr
        << "WARNING: pinned host memory allocation failed, falling back to pageable memory for NN input/output buffers (GPU transfers may overlap less)"
        << std::endl;
    }
    buf = malloc(numFloats * sizeof(float));
    if(buf == nullptr)
      throw StringError("InputBuffers: out of host memory");
    pageablePtrs.push_back(buf);
    return (float*)buf;
  }

  void freeBuffers() noexcept {
    for(void* ptr: pinnedPtrs)
      (void)cudaFreeHost(ptr);
    for(void* ptr: pageablePtrs)
      free(ptr);
    pinnedPtrs.clear();
    pageablePtrs.clear();
  }

  ~InputBuffers() {
    freeBuffers();
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

#ifdef KATAGO_BUILD_NNRAWGATE
void NeuralNet::getRawNNGateOutputs(const InputBuffers* inputBuffers, RawNNGateOutputs& out) {
  if(inputBuffers == nullptr)
    throw StringError("nnrawgate: null input buffers for raw output view");
  out.policy = inputBuffers->policyResults;
  out.value = inputBuffers->valueResults;
  out.scoreValue = inputBuffers->scoreValueResults;
  out.ownership = inputBuffers->ownershipResults;
  out.policyElts = inputBuffers->singlePolicyResultElts;
  out.valueElts = inputBuffers->singleValueResultElts;
  out.scoreValueElts = inputBuffers->singleScoreValueResultElts;
  out.ownershipElts = inputBuffers->singleOwnershipResultElts;
}
#endif

//---------------------------------------------------------------------------------------


void NeuralNet::getOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs
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
  CudaHandles* cudaHandles = gpuHandle->cudaHandles.get();
  cudaStream_t stream = cudaHandles->stream;

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

    CUDA_ERR("getOutput",cudaMemcpyAsync(buffers->inputBuf, inputBuffers->userInputBuffer, inputBuffers->singleInputBytes*batchSize, cudaMemcpyHostToDevice, stream));
    CUDA_ERR("getOutput",cudaMemcpyAsync(buffers->inputGlobalBuf, inputBuffers->userInputGlobalBuffer, inputBuffers->singleInputGlobalBytes*batchSize, cudaMemcpyHostToDevice, stream));
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

    CUDA_ERR("getOutput",cudaMemcpyAsync(buffers->inputBufFloat, inputBuffers->userInputBuffer, inputBuffers->singleInputBytes*batchSize, cudaMemcpyHostToDevice, stream));
    CUDA_ERR("getOutput",cudaMemcpyAsync(buffers->inputGlobalBufFloat, inputBuffers->userInputGlobalBuffer, inputBuffers->singleInputGlobalBytes*batchSize, cudaMemcpyHostToDevice, stream));

    customCudaCopyToHalf(
      (const float*)buffers->inputBufFloat,(half*)buffers->inputBuf,
      inputBuffers->singleInputElts*batchSize,stream
    );
    CUDA_ERR("getOutput",cudaPeekAtLastError());
    customCudaCopyToHalf(
      (const float*)buffers->inputGlobalBufFloat,(half*)buffers->inputGlobalBuf,
      inputBuffers->singleInputGlobalElts*batchSize,stream
    );
    CUDA_ERR("getOutput",cudaPeekAtLastError());
  }

  FourProfile::ManagerV1* selectedFourProfileManager = nullptr;
  FourProfile::RuntimeKeyV1 fourProfileRuntime;
  const FourProfile::RuntimeKeyV1* selectedFourProfileRuntime = nullptr;
  if(gpuHandle->fourProfileManager != nullptr &&
     gpuHandle->fourProfileManager->report().route == FourProfile::RouteV1::Specialized) {
    selectedFourProfileManager = gpuHandle->fourProfileManager.get();
    fourProfileRuntime = gpuHandle->fourProfileRuntime;
    selectedFourProfileRuntime = &fourProfileRuntime;
  }
  gpuHandle->model->apply(
    cudaHandles,
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
    buffers->workspaceBytes,
    selectedFourProfileManager,
    selectedFourProfileRuntime
  );

  CUDA_ERR("getOutput",cudaMemcpyAsync(inputBuffers->policyResults, buffers->policyBuf, inputBuffers->singlePolicyResultBytes*batchSize, cudaMemcpyDeviceToHost, stream));
  CUDA_ERR("getOutput",cudaMemcpyAsync(inputBuffers->valueResults, buffers->valueBuf, inputBuffers->singleValueResultBytes*batchSize, cudaMemcpyDeviceToHost, stream));
  CUDA_ERR("getOutput",cudaMemcpyAsync(inputBuffers->scoreValueResults, buffers->scoreValueBuf, inputBuffers->singleScoreValueResultBytes*batchSize, cudaMemcpyDeviceToHost, stream));
  CUDA_ERR("getOutput",cudaMemcpyAsync(inputBuffers->ownershipResults, buffers->ownershipBuf, inputBuffers->singleOwnershipResultBytes*batchSize, cudaMemcpyDeviceToHost, stream));
  CUDA_ERR("getOutput",cudaStreamSynchronize(stream));
#ifdef KATAGO_BUILD_BENCHMARKNN
  cudaHandles->benchmarkRoute.publishSuccessfulInvocation();
  logBenchmarkActiveRouteNoThrow(cudaHandles);
#endif

  assert(outputs.size() == batchSize);

  for(int row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];
    assert(output->nnXLen == nnXLen);
    assert(output->nnYLen == nnYLen);

    const float* policySrcBuf = inputBuffers->policyResults + row * gpuHandle->policySize;
    float* policyProbs = output->policyProbs;

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

#ifdef KATAGO_BUILD_BENCHMARKNN

namespace {

void benchmarkHashMixByte(uint64_t& hash, uint8_t value) {
  hash ^= value;
  hash *= UINT64_C(1099511628211);
}

void benchmarkHashMixUint64(uint64_t& hash, uint64_t value) {
  for(int shift = 0; shift < 64; shift += 8)
    benchmarkHashMixByte(hash,(uint8_t)(value >> shift));
}

void benchmarkHashMixFloat(uint64_t& hash, float value) {
  uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value),"unexpected float width");
  std::memcpy(&bits,&value,sizeof(bits));
  for(int shift = 0; shift < 32; shift += 8)
    benchmarkHashMixByte(hash,(uint8_t)(bits >> shift));
}

uint64_t benchmarkHashInputRow(
  const InputBuffers* buffers,
  int row
) {
  uint64_t hash = UINT64_C(1469598103934665603);
  const float* spatial = buffers->userInputBuffer + buffers->singleInputElts * row;
  const float* global = buffers->userInputGlobalBuffer + buffers->singleInputGlobalElts * row;
  for(size_t i = 0; i < buffers->singleInputElts; i++) {
    if(!std::isfinite(spatial[i]))
      throw StringError("benchmarknn: packed spatial input is non-finite");
    benchmarkHashMixFloat(hash,spatial[i]);
  }
  for(size_t i = 0; i < buffers->singleInputGlobalElts; i++) {
    if(!std::isfinite(global[i]))
      throw StringError("benchmarknn: packed global input is non-finite");
    benchmarkHashMixFloat(hash,global[i]);
  }
  return hash;
}

void benchmarkHashRawOutputSpan(
  uint64_t category,
  const float* values,
  uint64_t count,
  NeuralNet::BenchmarkRawIOProof& proof,
  bool& sawFirst,
  uint32_t& firstBits
) {
  benchmarkHashMixUint64(proof.outputChecksum,category);
  benchmarkHashMixUint64(proof.outputChecksum,count);
  for(uint64_t i = 0; i < count; i++) {
    const float value = values[i];
    uint32_t bits = 0;
    std::memcpy(&bits,&value,sizeof(bits));
    benchmarkHashMixUint64(proof.outputChecksum,i);
    benchmarkHashMixFloat(proof.outputChecksum,value);
    proof.outputsFinite = proof.outputsFinite && std::isfinite(value);
    proof.outputsNonzero = proof.outputsNonzero || value != 0.0f;
    if(!sawFirst) {
      firstBits = bits;
      sawFirst = true;
    }
    else if(bits != firstBits)
      proof.outputsNonconstant = true;
  }
}

void benchmarkHashOutputRowSpan(
  uint64_t& hash,
  uint64_t category,
  const float* values,
  size_t count
) {
  benchmarkHashMixUint64(hash,category);
  benchmarkHashMixUint64(hash,(uint64_t)count);
  for(size_t i = 0; i < count; i++) {
    benchmarkHashMixUint64(hash,(uint64_t)i);
    benchmarkHashMixFloat(hash,values[i]);
  }
}

uint64_t benchmarkHashOutputRow(const InputBuffers* buffers, int row) {
  uint64_t hash = UINT64_C(1469598103934665603);
  benchmarkHashOutputRowSpan(
    hash,UINT64_C(1),
    buffers->policyResults + buffers->singlePolicyResultElts * row,
    buffers->singlePolicyResultElts
  );
  benchmarkHashOutputRowSpan(
    hash,UINT64_C(2),
    buffers->valueResults + buffers->singleValueResultElts * row,
    buffers->singleValueResultElts
  );
  benchmarkHashOutputRowSpan(
    hash,UINT64_C(3),
    buffers->scoreValueResults + buffers->singleScoreValueResultElts * row,
    buffers->singleScoreValueResultElts
  );
  benchmarkHashOutputRowSpan(
    hash,UINT64_C(4),
    buffers->ownershipResults + buffers->singleOwnershipResultElts * row,
    buffers->singleOwnershipResultElts
  );
  return hash;
}

NeuralNet::BenchmarkRawIOProof makeBenchmarkRawIOProof(
  const ComputeHandle* gpuHandle,
  const InputBuffers* buffers,
  int batchSize,
  int laneIdx
) {
  if(gpuHandle == nullptr || buffers == nullptr)
    throw StringError("benchmarknn: null handle or buffers for raw proof");
  if(batchSize <= 0 || batchSize > buffers->maxBatchSize)
    throw StringError("benchmarknn: invalid raw-proof batch size");
  if(laneIdx < 0)
    throw StringError("benchmarknn: invalid lane index");
  if(buffers->singlePolicyResultElts != (size_t)gpuHandle->policySize)
    throw StringError("benchmarknn: raw-proof policy shape mismatch");

  NeuralNet::BenchmarkRawIOProof proof = NeuralNet::BenchmarkRawIOProof();
  proof.batchSize = batchSize;
  proof.inputChecksum = UINT64_C(1469598103934665603);
  proof.inputContentChecksum = UINT64_C(1469598103934665603);
  proof.outputChecksum = UINT64_C(1469598103934665603);
  proof.outputsFinite = true;
  benchmarkHashMixUint64(proof.outputChecksum,(uint64_t)laneIdx);
  benchmarkHashMixUint64(proof.outputChecksum,(uint64_t)batchSize);

  std::set<uint64_t> distinctInputRows;
  for(int row = 0; row < batchSize; row++) {
    const uint64_t rowHash = benchmarkHashInputRow(buffers,row);
    distinctInputRows.insert(rowHash);
    benchmarkHashMixUint64(proof.inputChecksum,(uint64_t)laneIdx);
    benchmarkHashMixUint64(proof.inputChecksum,(uint64_t)row);
    benchmarkHashMixUint64(proof.inputChecksum,rowHash);
    benchmarkHashMixUint64(proof.inputContentChecksum,(uint64_t)row);
    benchmarkHashMixUint64(proof.inputContentChecksum,rowHash);
    proof.inputRowsHashed += 1;
  }
  proof.uniqueInputRows = (int)distinctInputRows.size();

  proof.policyFloatsHashed = (uint64_t)buffers->singlePolicyResultElts * (uint64_t)batchSize;
  proof.valueFloatsHashed = (uint64_t)buffers->singleValueResultElts * (uint64_t)batchSize;
  proof.scoreValueFloatsHashed =
    (uint64_t)buffers->singleScoreValueResultElts * (uint64_t)batchSize;
  proof.ownershipFloatsHashed =
    (uint64_t)buffers->singleOwnershipResultElts * (uint64_t)batchSize;
  proof.rawOutputFloatsHashed =
    proof.policyFloatsHashed + proof.valueFloatsHashed +
    proof.scoreValueFloatsHashed + proof.ownershipFloatsHashed;

  bool sawFirst = false;
  uint32_t firstBits = 0;
  benchmarkHashRawOutputSpan(
    UINT64_C(1),buffers->policyResults,proof.policyFloatsHashed,proof,sawFirst,firstBits
  );
  benchmarkHashRawOutputSpan(
    UINT64_C(2),buffers->valueResults,proof.valueFloatsHashed,proof,sawFirst,firstBits
  );
  benchmarkHashRawOutputSpan(
    UINT64_C(3),buffers->scoreValueResults,proof.scoreValueFloatsHashed,proof,sawFirst,firstBits
  );
  benchmarkHashRawOutputSpan(
    UINT64_C(4),buffers->ownershipResults,proof.ownershipFloatsHashed,proof,sawFirst,firstBits
  );

  std::set<uint64_t> distinctOutputRows;
  for(int row = 0; row < batchSize; row++) {
    distinctOutputRows.insert(benchmarkHashOutputRow(buffers,row));
    proof.outputRowsHashed += 1;
  }
  proof.uniqueOutputRows = (int)distinctOutputRows.size();

  proof.valid =
    proof.inputRowsHashed == batchSize && proof.uniqueInputRows == batchSize &&
    proof.outputRowsHashed == batchSize && proof.uniqueOutputRows == batchSize &&
    proof.rawOutputFloatsHashed > 0 && sawFirst;
  return proof;
}

void prepareBenchmarkHostInputs(
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
  const size_t rowSpatialElts = (size_t)numSpatialFeatures * nnXLen * nnYLen;
  for(int nIdx = 0; nIdx < batchSize; nIdx++) {
    if(inputBufs[nIdx] == nullptr || inputBufs[nIdx]->rowSpatial == nullptr ||
       inputBufs[nIdx]->rowGlobal == nullptr)
      throw StringError("benchmarkDeviceOnlyOutput: null input row");
    if(inputBufs[nIdx]->rowSpatialSize < (int)rowSpatialElts ||
       inputBufs[nIdx]->rowGlobalSize < numGlobalFeatures)
      throw StringError("benchmarkDeviceOnlyOutput: undersized input row");
    float* rowSpatialInput = inputBuffers->userInputBuffer + inputBuffers->singleInputElts * nIdx;
    float* rowGlobalInput =
      inputBuffers->userInputGlobalBuffer + inputBuffers->singleInputGlobalElts * nIdx;
    std::copy(
      inputBufs[nIdx]->rowGlobal,inputBufs[nIdx]->rowGlobal+numGlobalFeatures,rowGlobalInput
    );
    SymmetryHelpers::copyInputsWithSymmetry(
      inputBufs[nIdx]->rowSpatial,rowSpatialInput,1,nnYLen,nnXLen,numSpatialFeatures,
      gpuHandle->inputsUseNHWC,inputBufs[nIdx]->symmetry
    );
  }
}

void uploadBenchmarkInputs(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int batchSize,
  cudaStream_t stream
) {
  Buffers* buffers = gpuHandle->buffers.get();
  if(!gpuHandle->usingFP16) {
    CUDA_ERR(
      "benchmarkDeviceOnlyOutput",
      cudaMemcpyAsync(
        buffers->inputBuf,inputBuffers->userInputBuffer,
        inputBuffers->singleInputBytes*batchSize,cudaMemcpyHostToDevice,stream
      )
    );
    CUDA_ERR(
      "benchmarkDeviceOnlyOutput",
      cudaMemcpyAsync(
        buffers->inputGlobalBuf,inputBuffers->userInputGlobalBuffer,
        inputBuffers->singleInputGlobalBytes*batchSize,cudaMemcpyHostToDevice,stream
      )
    );
  }
  else {
    CUDA_ERR(
      "benchmarkDeviceOnlyOutput",
      cudaMemcpyAsync(
        buffers->inputBufFloat,inputBuffers->userInputBuffer,
        inputBuffers->singleInputBytes*batchSize,cudaMemcpyHostToDevice,stream
      )
    );
    CUDA_ERR(
      "benchmarkDeviceOnlyOutput",
      cudaMemcpyAsync(
        buffers->inputGlobalBufFloat,inputBuffers->userInputGlobalBuffer,
        inputBuffers->singleInputGlobalBytes*batchSize,cudaMemcpyHostToDevice,stream
      )
    );
    customCudaCopyToHalf(
      (const float*)buffers->inputBufFloat,(half*)buffers->inputBuf,
      inputBuffers->singleInputElts*batchSize,stream
    );
    CUDA_ERR("benchmarkDeviceOnlyOutput",cudaPeekAtLastError());
    customCudaCopyToHalf(
      (const float*)buffers->inputGlobalBufFloat,(half*)buffers->inputGlobalBuf,
      inputBuffers->singleInputGlobalElts*batchSize,stream
    );
    CUDA_ERR("benchmarkDeviceOnlyOutput",cudaPeekAtLastError());
  }
}

void poisonBenchmarkDeviceOutputs(
  Buffers* buffers,
  const InputBuffers* inputBuffers,
  int batchSize,
  cudaStream_t stream
) {
  CUDA_ERR(
    "benchmarkDeviceOnlyOutput",
    cudaMemsetAsync(
      buffers->policyBuf,0xFF,inputBuffers->singlePolicyResultBytes*batchSize,stream
    )
  );
  CUDA_ERR(
    "benchmarkDeviceOnlyOutput",
    cudaMemsetAsync(
      buffers->valueBuf,0xFF,inputBuffers->singleValueResultBytes*batchSize,stream
    )
  );
  CUDA_ERR(
    "benchmarkDeviceOnlyOutput",
    cudaMemsetAsync(
      buffers->scoreValueBuf,0xFF,inputBuffers->singleScoreValueResultBytes*batchSize,stream
    )
  );
  CUDA_ERR(
    "benchmarkDeviceOnlyOutput",
    cudaMemsetAsync(
      buffers->ownershipBuf,0xFF,inputBuffers->singleOwnershipResultBytes*batchSize,stream
    )
  );
}

void poisonBenchmarkHostOutputs(
  InputBuffers* inputBuffers,
  int batchSize
) {
  std::memset(inputBuffers->policyResults,0xFF,inputBuffers->singlePolicyResultBytes*batchSize);
  std::memset(inputBuffers->valueResults,0xFF,inputBuffers->singleValueResultBytes*batchSize);
  std::memset(
    inputBuffers->scoreValueResults,0xFF,inputBuffers->singleScoreValueResultBytes*batchSize
  );
  std::memset(
    inputBuffers->ownershipResults,0xFF,inputBuffers->singleOwnershipResultBytes*batchSize
  );
}

void copyBenchmarkRawOutputsToHost(
  Buffers* buffers,
  InputBuffers* inputBuffers,
  int batchSize,
  cudaStream_t stream
) {
  CUDA_ERR(
    "benchmarkDeviceOnlyOutput",
    cudaMemcpyAsync(
      inputBuffers->policyResults,buffers->policyBuf,
      inputBuffers->singlePolicyResultBytes*batchSize,cudaMemcpyDeviceToHost,stream
    )
  );
  CUDA_ERR(
    "benchmarkDeviceOnlyOutput",
    cudaMemcpyAsync(
      inputBuffers->valueResults,buffers->valueBuf,
      inputBuffers->singleValueResultBytes*batchSize,cudaMemcpyDeviceToHost,stream
    )
  );
  CUDA_ERR(
    "benchmarkDeviceOnlyOutput",
    cudaMemcpyAsync(
      inputBuffers->scoreValueResults,buffers->scoreValueBuf,
      inputBuffers->singleScoreValueResultBytes*batchSize,cudaMemcpyDeviceToHost,stream
    )
  );
  CUDA_ERR(
    "benchmarkDeviceOnlyOutput",
    cudaMemcpyAsync(
      inputBuffers->ownershipResults,buffers->ownershipBuf,
      inputBuffers->singleOwnershipResultBytes*batchSize,cudaMemcpyDeviceToHost,stream
    )
  );
}

void destroyBenchmarkEventsNoThrow(std::vector<cudaEvent_t>& events) noexcept {
  for(cudaEvent_t event : events) {
    if(event != nullptr)
      (void)cudaEventDestroy(event);
  }
}

}

bool NeuralNet::getBenchmarkRawIOProof(
  const ComputeHandle* gpuHandle,
  const InputBuffers* buffers,
  int batchSize,
  int laneIdx,
  BenchmarkRawIOProof& proof
) {
  proof = makeBenchmarkRawIOProof(gpuHandle,buffers,batchSize,laneIdx);
  return true;
}

bool NeuralNet::benchmarkDeviceOnlyOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  NNResultBuf** inputBufs,
  int batchSize,
  int laneIdx,
  int numWarmups,
  int numIterations,
  std::vector<double>& iterationSeconds,
  BenchmarkRawIOProof& proof,
  const std::function<void()>& beforeTimedLoop,
  const std::function<void()>& afterTimedLoop
) {
  if(gpuHandle == nullptr || inputBuffers == nullptr || inputBufs == nullptr)
    throw StringError("benchmarkDeviceOnlyOutput: null argument");
  if(batchSize <= 0 || batchSize > inputBuffers->maxBatchSize)
    throw StringError("benchmarkDeviceOnlyOutput: invalid batch size");
  if(laneIdx < 0)
    throw StringError("benchmarkDeviceOnlyOutput: invalid lane index");
  if(numWarmups < 0 || numWarmups > 10000 ||
     numIterations <= 0 || numIterations > 4096)
    throw StringError("benchmarkDeviceOnlyOutput: invalid warmup/iteration count");

  proof = BenchmarkRawIOProof();
  iterationSeconds.clear();
  CudaHandles* cudaHandles = gpuHandle->cudaHandles.get();
  const cudaStream_t stream = cudaHandles->stream;
  prepareBenchmarkHostInputs(gpuHandle,inputBuffers,inputBufs,batchSize);
  uploadBenchmarkInputs(gpuHandle,inputBuffers,batchSize,stream);

  Buffers* buffers = gpuHandle->buffers.get();
  ScratchBuffers* scratch = gpuHandle->scratch.get();
  FourProfile::ManagerV1* selectedFourProfileManager = nullptr;
  FourProfile::RuntimeKeyV1 fourProfileRuntime;
  const FourProfile::RuntimeKeyV1* selectedFourProfileRuntime = nullptr;
  if(gpuHandle->fourProfileManager != nullptr &&
     gpuHandle->fourProfileManager->report().route == FourProfile::RouteV1::Specialized) {
    selectedFourProfileManager = gpuHandle->fourProfileManager.get();
    fourProfileRuntime = gpuHandle->fourProfileRuntime;
    selectedFourProfileRuntime = &fourProfileRuntime;
  }
  for(int i = 0; i < numWarmups; i++) {
    gpuHandle->model->apply(
      cudaHandles,scratch,batchSize,gpuHandle->requireExactNNLen,
      buffers->inputBuf,buffers->inputGlobalBuf,buffers->policyBuf,buffers->valueBuf,
      buffers->scoreValueBuf,buffers->ownershipBuf,buffers->workspaceBuf,buffers->workspaceBytes,
      selectedFourProfileManager,selectedFourProfileRuntime
    );
  }

  // Poison after warmup and finish all setup work before the shared wall
  // release. A final raw proof containing poison/NaN therefore cannot pass.
  poisonBenchmarkDeviceOutputs(buffers,inputBuffers,batchSize,stream);
  poisonBenchmarkHostOutputs(inputBuffers,batchSize);
  CUDA_ERR("benchmarkDeviceOnlyOutput",cudaStreamSynchronize(stream));

  std::vector<cudaEvent_t> startEvents((size_t)numIterations,nullptr);
  std::vector<cudaEvent_t> endEvents((size_t)numIterations,nullptr);
  try {
    for(int i = 0; i < numIterations; i++) {
      CUDA_ERR(
        "benchmarkDeviceOnlyOutput",
        cudaEventCreateWithFlags(&startEvents[i],cudaEventDefault)
      );
      CUDA_ERR(
        "benchmarkDeviceOnlyOutput",
        cudaEventCreateWithFlags(&endEvents[i],cudaEventDefault)
      );
    }

    if(beforeTimedLoop)
      beforeTimedLoop();
    for(int i = 0; i < numIterations; i++) {
      CUDA_ERR("benchmarkDeviceOnlyOutput",cudaEventRecord(startEvents[i],stream));
      gpuHandle->model->apply(
        cudaHandles,scratch,batchSize,gpuHandle->requireExactNNLen,
        buffers->inputBuf,buffers->inputGlobalBuf,buffers->policyBuf,buffers->valueBuf,
        buffers->scoreValueBuf,buffers->ownershipBuf,buffers->workspaceBuf,buffers->workspaceBytes,
        selectedFourProfileManager,selectedFourProfileRuntime
      );
      CUDA_ERR("benchmarkDeviceOnlyOutput",cudaEventRecord(endEvents[i],stream));
    }
    CUDA_ERR("benchmarkDeviceOnlyOutput",cudaStreamSynchronize(stream));
    if(afterTimedLoop)
      afterTimedLoop();

    // This is deliberately the first operation after the common wall closes:
    // consume the exact final timed outputs, on this handle's own stream,
    // without launching another inference.
    copyBenchmarkRawOutputsToHost(buffers,inputBuffers,batchSize,stream);
    CUDA_ERR("benchmarkDeviceOnlyOutput",cudaStreamSynchronize(stream));
    proof = makeBenchmarkRawIOProof(gpuHandle,inputBuffers,batchSize,laneIdx);
    cudaHandles->benchmarkRoute.publishSuccessfulInvocation();
    logBenchmarkActiveRouteNoThrow(cudaHandles);

    iterationSeconds.reserve((size_t)numIterations);
    for(int i = 0; i < numIterations; i++) {
      float milliseconds = 0.0f;
      CUDA_ERR(
        "benchmarkDeviceOnlyOutput",
        cudaEventElapsedTime(&milliseconds,startEvents[i],endEvents[i])
      );
      const double seconds = (double)milliseconds / 1000.0;
      if(!(seconds > 0.0) || !std::isfinite(seconds))
        throw StringError("benchmarkDeviceOnlyOutput: invalid CUDA event interval");
      iterationSeconds.push_back(seconds);
    }
  }
  catch(...) {
    destroyBenchmarkEventsNoThrow(startEvents);
    destroyBenchmarkEventsNoThrow(endEvents);
    throw;
  }

  destroyBenchmarkEventsNoThrow(startEvents);
  destroyBenchmarkEventsNoThrow(endEvents);
  return true;
}

#endif

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

  fillMaskFloatBufAndMaskSumBuf(
    cudaHandles,deviceMask,deviceMaskFloat,deviceMaskSum,useFP16,desiredBatchSize,nnXLen,nnYLen
  );

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

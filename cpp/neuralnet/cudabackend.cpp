#ifdef USE_CUDA_BACKEND
#include "../neuralnet/cudaerrorcheck.h"
#include "../neuralnet/cudaincludes.h"

// The dependency-minimal generic fallback uses the in-tree online-softmax
// attention kernel. Shape-specific attention will be selected separately.
#define KATAGO_CUDA_HAS_SDPA 0

#include "../neuralnet/cudahelpers.h"
#include "../neuralnet/cudautils.h"
#include "../neuralnet/modelversion.h"
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

// Gom2026's legacy outer backend predates the explicit-stream launcher API used
// by the official CUDA core. Keep the adapter local: all legacy calls below are
// forwarded to the official kernels on the default stream. This preserves the
// old synchronous getOutput contract without maintaining a second set of GPU
// helpers.
#define KATAGO_LEGACY_STREAM nullptr
inline void customCudaChannelConcat(const float* a,const float* b,float* o,int ca,int cb,int n) { customCudaChannelConcat(a,b,o,ca,cb,n,KATAGO_LEGACY_STREAM); }
inline void customCudaChannelConcat(const half* a,const half* b,half* o,int ca,int cb,int n) { customCudaChannelConcat(a,b,o,ca,cb,n,KATAGO_LEGACY_STREAM); }
inline void customCudaChannel0ExtractNCHW(const float* i,float* o,int n,int c,int xy) { customCudaChannel0ExtractNCHW(i,o,n,c,xy,KATAGO_LEGACY_STREAM); }
inline void customCudaChannel0ExtractNCHW(const half* i,half* o,int n,int c,int xy) { customCudaChannel0ExtractNCHW(i,o,n,c,xy,KATAGO_LEGACY_STREAM); }
inline void customCudaChannel0ExtractNHWC(const float* i,float* o,int n,int xy,int c) { customCudaChannel0ExtractNHWC(i,o,n,xy,c,KATAGO_LEGACY_STREAM); }
inline void customCudaChannel0ExtractNHWC(const half* i,half* o,int n,int xy,int c) { customCudaChannel0ExtractNHWC(i,o,n,xy,c,KATAGO_LEGACY_STREAM); }
inline void customCudaPoolRowsSumNCHW(const float* i,float* o,int n,int c,int xy,float s) { customCudaPoolRowsSumNCHW(i,o,n,c,xy,s,KATAGO_LEGACY_STREAM); }
inline void customCudaValueHeadPoolNCHW(const float* i,float* o,int n,int c,int xy,const float* m) { customCudaValueHeadPoolNCHW(i,o,n,c,xy,m,KATAGO_LEGACY_STREAM); }
inline void customCudaValueHeadPoolNHWC(const float* i,float* o,int n,int xy,int c,const float* m) { customCudaValueHeadPoolNHWC(i,o,n,xy,c,m,KATAGO_LEGACY_STREAM); }
#define KATAGO_WRAP_GPOOL(T) \
  inline void customCudaPoolRowsGPoolNCHW(const T* i,T* o,int n,int c,int xy,const T* m,const float* ms) { customCudaPoolRowsGPoolNCHW(i,o,n,c,xy,m,ms,KATAGO_LEGACY_STREAM); } \
  inline void customCudaPoolRowsGPoolNHWC(const T* i,T* o,int n,int xy,int c,const T* m,const float* ms) { customCudaPoolRowsGPoolNHWC(i,o,n,xy,c,m,ms,KATAGO_LEGACY_STREAM); }
KATAGO_WRAP_GPOOL(float)
KATAGO_WRAP_GPOOL(half)
inline void customCudaCopyToHalf(const float* i,half* o,int n) { customCudaCopyToHalf(i,o,n,KATAGO_LEGACY_STREAM); }
inline void customCudaCopyFromHalf(const half* i,float* o,int n) { customCudaCopyFromHalf(i,o,n,KATAGO_LEGACY_STREAM); }
#define KATAGO_WRAP_BIAS(T) \
  inline void customCudaAddCBiasInplaceNC(T* b,const T* x,int n,int c,int a) { customCudaAddCBiasInplaceNC(b,x,n,c,a,KATAGO_LEGACY_STREAM); } \
  inline void customCudaAddNCBiasInplaceNCHW(T* b,const T* x,int n,int c,int xy) { customCudaAddNCBiasInplaceNCHW(b,x,n,c,xy,KATAGO_LEGACY_STREAM); } \
  inline void customCudaAddNCBiasInplaceNHWC(T* b,const T* x,int n,int xy,int c) { customCudaAddNCBiasInplaceNHWC(b,x,n,xy,c,KATAGO_LEGACY_STREAM); }
KATAGO_WRAP_BIAS(float)
KATAGO_WRAP_BIAS(half)
#define KATAGO_WRAP_SCALE_BIAS(T) \
  inline void customCudaApplyCScaleBiasNCHW(const T* i,T* o,const T* s,const T* b,const T* m,int n,int c,int xy,int a) { customCudaApplyCScaleBiasNCHW(i,o,s,b,m,n,c,xy,a,KATAGO_LEGACY_STREAM); } \
  inline void customCudaApplyCScaleBiasNHWC(const T* i,T* o,const T* s,const T* b,const T* m,int n,int xy,int c,int a) { customCudaApplyCScaleBiasNHWC(i,o,s,b,m,n,xy,c,a,KATAGO_LEGACY_STREAM); }
KATAGO_WRAP_SCALE_BIAS(float)
KATAGO_WRAP_SCALE_BIAS(half)
#define KATAGO_WRAP_RMS(T) \
  inline void customCudaRMSNormGammaBetaNCHW(const T* i,T* o,const T* g,const T* b,const T* m,int n,int c,int xy,float e,int a) { customCudaRMSNormGammaBetaNCHW(i,o,g,b,m,n,c,xy,e,a,KATAGO_LEGACY_STREAM); } \
  inline void customCudaRMSNormGammaBetaNHWC(const T* i,T* o,const T* g,const T* b,const T* m,int n,int xy,int c,float e,int a) { customCudaRMSNormGammaBetaNHWC(i,o,g,b,m,n,xy,c,e,a,KATAGO_LEGACY_STREAM); }
KATAGO_WRAP_RMS(float)
KATAGO_WRAP_RMS(half)
#define KATAGO_WRAP_ROPE(T) \
  inline void customCudaApplyRoPE(T* x,const T* co,const T* si,int n,int s,int h,int kh,int d,int p,bool l) { customCudaApplyRoPE(x,co,si,n,s,h,kh,d,p,l,KATAGO_LEGACY_STREAM); }
KATAGO_WRAP_ROPE(float)
KATAGO_WRAP_ROPE(half)
#define KATAGO_WRAP_ATTN(T) \
  inline void customCudaFlashAttention(const T* q,const T* k,const T* v,const T* m,T* o,int n,int s,int h,int kh,int qd,int vd) { customCudaFlashAttention(q,k,v,m,o,n,s,h,kh,qd,vd,KATAGO_LEGACY_STREAM); } \
  inline void customCudaMaskedResidualAddNHWC(T* t,const T* r,const T* m,int n,int xy,int c) { customCudaMaskedResidualAddNHWC(t,r,m,n,xy,c,KATAGO_LEGACY_STREAM); } \
  inline void customCudaSwiGLU(const T* a,const T* b,T* o,int z) { customCudaSwiGLU(a,b,o,z,KATAGO_LEGACY_STREAM); }
KATAGO_WRAP_ATTN(float)
KATAGO_WRAP_ATTN(half)
#undef KATAGO_WRAP_GPOOL
#undef KATAGO_WRAP_BIAS
#undef KATAGO_WRAP_SCALE_BIAS
#undef KATAGO_WRAP_RMS
#undef KATAGO_WRAP_ROPE
#undef KATAGO_WRAP_ATTN

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


struct CudaHandles {
  cublasHandle_t cublas;
  cudnnHandle_t cudnn;
  const int majorComputeCapability;
  const int minorComputeCapability;
  // Set once, before Model construction. Attention blocks may commit their
  // QKV weights to the official interleaved layout only when this is true.
  bool mmaAttentionEnabled;
  bool loggedUsingCombinedQKV;
  bool loggedUsingMmaAttention;
  bool loggedUsingScalarAttention;
  std::unique_ptr<SDPAGraphCache> sdpaCache;
  // Logger for this handle's server thread; may be NULL. Used to report cudnn SDPA falling back.
  Logger* logger;
  // Set while warming up (see NNEvaluator::maybeWarmupComputeHandle). When true, a failed cudnn SDPA
  // execution is tolerated (fall back to the custom kernel); when false such a failure is fatal.
  bool isWarmup;

  CudaHandles(int major, int minor)
    : majorComputeCapability(major),
      minorComputeCapability(minor),
      mmaAttentionEnabled(false),
      loggedUsingCombinedQKV(false),
      loggedUsingMmaAttention(false),
      loggedUsingScalarAttention(false),
      sdpaCache(std::make_unique<SDPAGraphCache>()),
      logger(NULL),
      isWarmup(false)
  {
    CUBLAS_ERR("CudaHandles",cublasCreate(&cublas));
    CUDNN_ERR("CudaHandles",cudnnCreate(&cudnn));
  }

  ~CudaHandles() {
    cublasDestroy(cublas);
    cudnnDestroy(cudnn);
  }

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
      cudaDeviceSynchronize();
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
                                      batchSize,numChannels,nnXLen*nnYLen,activation);
      else
        customCudaApplyCScaleBiasNHWC((const float*)inputBuf,(float*)outputBuf,(const float*)mergedScaleBuf,(const float*)mergedBiasBuf,
                                      (const float*)maskBuf,
                                      batchSize,nnXLen*nnYLen,numChannels,activation);
    }
    else {
      if(!usingNHWC)
        customCudaApplyCScaleBiasNCHW((const half*)inputBuf,(half*)outputBuf,(const half*)mergedScaleBuf,(const half*)mergedBiasBuf,
                                      (const half*)maskBuf,
                                      batchSize,numChannels,nnXLen*nnYLen,activation);
      else
        customCudaApplyCScaleBiasNHWC((const half*)inputBuf,(half*)outputBuf,(const half*)mergedScaleBuf,(const half*)mergedBiasBuf,
                                      (const half*)maskBuf,
                                      batchSize,nnXLen*nnYLen,numChannels,activation);
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
        (const float*)matBuf,outChannels,
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
        (const half*)matBuf,outChannels,
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
    (void)cudaHandles;
    if(!usingFP16) {
      customCudaAddCBiasInplaceNC((float*)matBuf,(const float*)biasBuf,batchSize,numChannels,activation);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }
    else {
      customCudaAddCBiasInplaceNC((half*)matBuf,(const half*)biasBuf,batchSize,numChannels,activation);
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
        customCudaPoolRowsGPoolNCHW((const float*)gpoolOut2.buf,(float*)gpoolConcat.buf,batchSize,gpoolChannels,nnXLen*nnYLen,(const float*)maskBuf,maskSumBuf);
      else
        customCudaPoolRowsGPoolNHWC((const float*)gpoolOut2.buf,(float*)gpoolConcat.buf,batchSize,nnXLen*nnYLen,gpoolChannels,(const float*)maskBuf,maskSumBuf);
    }
    else {
      if(!usingNHWC)
        customCudaPoolRowsGPoolNCHW((const half*)gpoolOut2.buf,(half*)gpoolConcat.buf,batchSize,gpoolChannels,nnXLen*nnYLen,(const half*)maskBuf,maskSumBuf);
      else
        customCudaPoolRowsGPoolNHWC((const half*)gpoolOut2.buf,(half*)gpoolConcat.buf,batchSize,nnXLen*nnYLen,gpoolChannels,(const half*)maskBuf,maskSumBuf);
    }
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());

    gpoolToBiasMul.apply(cudaHandles,scratch,batchSize,gpoolConcat.buf,gpoolBias.buf,workspaceBuf,workspaceBytes);

    if(!usingFP16) {
      if(!usingNHWC)
        customCudaAddNCBiasInplaceNCHW((float*)regularOut.buf,(const float*)gpoolBias.buf,batchSize,regularChannels,nnXLen*nnYLen);
      else
        customCudaAddNCBiasInplaceNHWC((float*)regularOut.buf,(const float*)gpoolBias.buf,batchSize,nnXLen*nnYLen,regularChannels);
    }
    else {
      if(!usingNHWC)
        customCudaAddNCBiasInplaceNCHW((half*)regularOut.buf,(const half*)gpoolBias.buf,batchSize,regularChannels,nnXLen*nnYLen);
      else
        customCudaAddNCBiasInplaceNHWC((half*)regularOut.buf,(const half*)gpoolBias.buf,batchSize,nnXLen*nnYLen,regularChannels);
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
  void* weightBuf;
  void* zeroBetaBuf;

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
    CudaUtils::mallocAndCopyToDevice(name, desc->weight, weightBuf, useFP16);
    // Allocate a zero buffer for beta (TransformerRMSNorm has no bias)
    vector<float> zeros(numChannels, 0.0f);
    CudaUtils::mallocAndCopyToDevice(name + ":zeroBeta", zeros, zeroBetaBuf, useFP16);
  }

  ~TransformerRMSNormLayer() {
    cudaFree(weightBuf);
    cudaFree(zeroBetaBuf);
  }

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
    (void)cudaHandles;
    // RMSNormGammaBetaNHWC with gamma=weight, beta=zero, mask, identity activation.
    if(!usingFP16) {
      customCudaRMSNormGammaBetaNHWC(
        (const float*)inputBuf, (float*)outputBuf,
        (const float*)weightBuf, (const float*)zeroBetaBuf,
        (const float*)maskBuf,
        batchSize, xySize, numChannels, epsilon, ACTIVATION_IDENTITY);
    }
    else {
      customCudaRMSNormGammaBetaNHWC(
        (const half*)inputBuf, (half*)outputBuf,
        (const half*)weightBuf, (const half*)zeroBetaBuf,
        (const half*)maskBuf,
        batchSize, xySize, numChannels, epsilon, ACTIVATION_IDENTITY);
    }
    CUDA_ERR(name.c_str(), cudaPeekAtLastError());
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
    (void)cudaHandles;
    int xySize = nnXLen * nnYLen;
    if(!spatial) {
      if(!usingFP16) {
        if(!usingNHWC)
          customCudaRMSNormGammaBetaNCHW(
            (const float*)inputBuf, (float*)outputBuf, (const float*)gammaBuf, (const float*)betaBuf,
            (const float*)maskBuf, batchSize, numChannels, xySize, epsilon, activation);
        else
          customCudaRMSNormGammaBetaNHWC(
            (const float*)inputBuf, (float*)outputBuf, (const float*)gammaBuf, (const float*)betaBuf,
            (const float*)maskBuf, batchSize, xySize, numChannels, epsilon, activation);
      }
      else {
        if(!usingNHWC)
          customCudaRMSNormGammaBetaNCHW(
            (const half*)inputBuf, (half*)outputBuf, (const half*)gammaBuf, (const half*)betaBuf,
            (const half*)maskBuf, batchSize, numChannels, xySize, epsilon, activation);
        else
          customCudaRMSNormGammaBetaNHWC(
            (const half*)inputBuf, (half*)outputBuf, (const half*)gammaBuf, (const half*)betaBuf,
            (const half*)maskBuf, batchSize, xySize, numChannels, epsilon, activation);
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
            (const float*)maskBuf, maskSumBuf, batchSize, numChannels, xySize, epsilon, activation, (float*)sumSqBuf.buf);
        else
          customCudaSpatialRMSNormNHWC(
            (const float*)inputBuf, (float*)outputBuf, (const float*)gammaBuf, (const float*)betaBuf,
            (const float*)maskBuf, maskSumBuf, batchSize, xySize, numChannels, epsilon, activation, (float*)sumSqBuf.buf);
      }
      else {
        if(!usingNHWC)
          customCudaSpatialRMSNormNCHW(
            (const half*)inputBuf, (half*)outputBuf, (const half*)gammaBuf, (const half*)betaBuf,
            (const half*)maskBuf, maskSumBuf, batchSize, numChannels, xySize, epsilon, activation, (float*)sumSqBuf.buf);
        else
          customCudaSpatialRMSNormNHWC(
            (const half*)inputBuf, (half*)outputBuf, (const half*)gammaBuf, (const half*)betaBuf,
            (const half*)maskBuf, maskSumBuf, batchSize, xySize, numChannels, epsilon, activation, (float*)sumSqBuf.buf);
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
  const MatMulLayer outProj;

  // Official projection plan. On an MMA-capable FP16 device use one interleaved
  // QKV GEMM. Otherwise equal-shape Q/K/V share one strided-batched GEMM.
  const bool sameQKVShapes;
  const bool useCombinedQKV;
  void* qkvPackedWeights;
  std::unique_ptr<MatMulLayer> qProj;
  std::unique_ptr<MatMulLayer> kProj;
  std::unique_ptr<MatMulLayer> vProj;

  // Fixed RoPE uses device tables. Learned RoPE keeps the tiny frequency
  // tensor in FP32 and recomputes sin/cos in the official fused QK kernel.
  void* ropeCosTable;
  void* ropeSinTable;
  void* ropeFreqsBuf;
  int ropeNumPairs;

  static bool shouldCombineQKV(
    CudaHandles* cudaHandles,
    const TransformerAttentionDesc* desc,
    bool useFP16
  ) {
    return
      useFP16 &&
      cudaHandles->mmaAttentionEnabled &&
      desc->qProj.inChannels == desc->kProj.inChannels &&
      desc->qProj.inChannels == desc->vProj.inChannels &&
      customCudaFlashAttentionMmaSupportsShape(
        desc->numHeads,desc->numKVHeads,desc->qHeadDim,desc->vHeadDim
      );
  }

  TransformerAttentionBlock() = delete;
  TransformerAttentionBlock(const TransformerAttentionBlock&) = delete;
  TransformerAttentionBlock& operator=(const TransformerAttentionBlock&) = delete;

  TransformerAttentionBlock(
    CudaHandles* cudaHandles,
    const TransformerAttentionDesc* desc,
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
    inChannels(desc->qProj.inChannels),
    nnXLen(nnX),
    nnYLen(nnY),
    usingFP16(useFP16),
    usingNHWC(useNHWC),
    preLN(cudaHandles,&desc->preLN,useFP16),
    outProj(cudaHandles,&desc->outProj,useFP16),
    sameQKVShapes(
      desc->qProj.inChannels == desc->kProj.inChannels &&
      desc->qProj.inChannels == desc->vProj.inChannels &&
      desc->qProj.outChannels == desc->kProj.outChannels &&
      desc->qProj.outChannels == desc->vProj.outChannels
    ),
    useCombinedQKV(shouldCombineQKV(cudaHandles,desc,useFP16)),
    qkvPackedWeights(NULL),
    ropeCosTable(NULL),
    ropeSinTable(NULL),
    ropeFreqsBuf(NULL),
    ropeNumPairs(0)
  {
    if(!useNHWC)
      throw StringError("Transformer blocks with NCHW layout are not supported by the CUDA backend");

    const int qTotalDim = numHeads * qHeadDim;
    const int kTotalDim = numKVHeads * qHeadDim;
    const int vTotalDim = numKVHeads * vHeadDim;
    if(qTotalDim % 8 != 0 || kTotalDim % 8 != 0 || vTotalDim % 8 != 0)
      throw StringError(name + ": CUDA attention projection widths must be multiples of 8");

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
      CudaUtils::mallocAndCopyToDevice(name + ":qkvCombined",packed,qkvPackedWeights,useFP16);
    }
    else if(sameQKVShapes) {
      vector<float> packed;
      packed.reserve(desc->qProj.weights.size() * 3);
      packed.insert(packed.end(),desc->qProj.weights.begin(),desc->qProj.weights.end());
      packed.insert(packed.end(),desc->kProj.weights.begin(),desc->kProj.weights.end());
      packed.insert(packed.end(),desc->vProj.weights.begin(),desc->vProj.weights.end());
      CudaUtils::mallocAndCopyToDevice(name + ":qkvPacked",packed,qkvPackedWeights,useFP16);
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
          name + ":ropeFreqs",desc->ropeFreqs.data(),(int)desc->ropeFreqs.size(),ropeFreqsBuf,false
        );
      }
      else {
        const int seqLen = nnXLen * nnYLen;
        vector<float> cosTableData;
        vector<float> sinTableData;
        desc->computeRopeCosSin(nnXLen,nnYLen,seqLen,cosTableData,sinTableData);
        CudaUtils::mallocAndCopyToDevice(
          name + ":ropeCos",cosTableData.data(),(int)cosTableData.size(),ropeCosTable,useFP16
        );
        CudaUtils::mallocAndCopyToDevice(
          name + ":ropeSin",sinTableData.data(),(int)sinTableData.size(),ropeSinTable,useFP16
        );
      }
    }
  }

  ~TransformerAttentionBlock() {
    if(qkvPackedWeights != NULL) cudaFree(qkvPackedWeights);
    if(ropeCosTable != NULL) cudaFree(ropeCosTable);
    if(ropeSinTable != NULL) cudaFree(ropeSinTable);
    if(ropeFreqsBuf != NULL) cudaFree(ropeFreqsBuf);
  }

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
        qkvPackedWeights,trunkScratchBuf,qkvBuf.buf,0LL,1
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
          qkvPackedWeights,trunkScratchBuf,qPtr,outputStrideElts,3
        );
      }
      else {
        qProj->apply(cudaHandles,scratch,matBatchSize,trunkScratchBuf,qPtr,workspaceBuf,workspaceBytes);
        kProj->apply(cudaHandles,scratch,matBatchSize,trunkScratchBuf,kPtr,workspaceBuf,workspaceBytes);
        vProj->apply(cudaHandles,scratch,matBatchSize,trunkScratchBuf,vPtr,workspaceBuf,workspaceBytes);
      }
    }

    if(useRope) {
      const bool fuseQK = numHeads % numKVHeads == 0;
      if(useCombinedQKV && !fuseQK)
        throw StringError(name + ": combined QKV requires integral grouped-query heads");

      if(learnableRope) {
        if(!usingFP16) {
          if(fuseQK)
            customCudaApplyRoPEQKLearnableRecompute(
              (float*)qPtr,(float*)kPtr,(const float*)ropeFreqsBuf,
              batchSize,seqLen,numHeads,numKVHeads,qHeadDim,
              qStrideElts,kvStrideElts,ropeNumPairs,nnXLen,KATAGO_LEGACY_STREAM
            );
          else {
            customCudaApplyRoPELearnableRecompute(
              (float*)qPtr,(const float*)ropeFreqsBuf,batchSize,seqLen,numHeads,numKVHeads,
              qHeadDim,ropeNumPairs,nnXLen,KATAGO_LEGACY_STREAM
            );
            customCudaApplyRoPELearnableRecompute(
              (float*)kPtr,(const float*)ropeFreqsBuf,batchSize,seqLen,numKVHeads,numKVHeads,
              qHeadDim,ropeNumPairs,nnXLen,KATAGO_LEGACY_STREAM
            );
          }
        }
        else {
          if(fuseQK)
            customCudaApplyRoPEQKLearnableRecompute(
              (half*)qPtr,(half*)kPtr,(const float*)ropeFreqsBuf,
              batchSize,seqLen,numHeads,numKVHeads,qHeadDim,
              qStrideElts,kvStrideElts,ropeNumPairs,nnXLen,KATAGO_LEGACY_STREAM
            );
          else {
            customCudaApplyRoPELearnableRecompute(
              (half*)qPtr,(const float*)ropeFreqsBuf,batchSize,seqLen,numHeads,numKVHeads,
              qHeadDim,ropeNumPairs,nnXLen,KATAGO_LEGACY_STREAM
            );
            customCudaApplyRoPELearnableRecompute(
              (half*)kPtr,(const float*)ropeFreqsBuf,batchSize,seqLen,numKVHeads,numKVHeads,
              qHeadDim,ropeNumPairs,nnXLen,KATAGO_LEGACY_STREAM
            );
          }
        }
      }
      else {
        if(!usingFP16) {
          if(fuseQK)
            customCudaApplyRoPEQK(
              (float*)qPtr,(float*)kPtr,(const float*)ropeCosTable,(const float*)ropeSinTable,
              batchSize,seqLen,numHeads,numKVHeads,qHeadDim,qStrideElts,kvStrideElts,
              ropeNumPairs,false,KATAGO_LEGACY_STREAM
            );
          else {
            customCudaApplyRoPE(
              (float*)qPtr,(const float*)ropeCosTable,(const float*)ropeSinTable,
              batchSize,seqLen,numHeads,numKVHeads,qHeadDim,ropeNumPairs,false
            );
            customCudaApplyRoPE(
              (float*)kPtr,(const float*)ropeCosTable,(const float*)ropeSinTable,
              batchSize,seqLen,numKVHeads,numKVHeads,qHeadDim,ropeNumPairs,false
            );
          }
        }
        else {
          if(fuseQK)
            customCudaApplyRoPEQK(
              (half*)qPtr,(half*)kPtr,(const half*)ropeCosTable,(const half*)ropeSinTable,
              batchSize,seqLen,numHeads,numKVHeads,qHeadDim,qStrideElts,kvStrideElts,
              ropeNumPairs,false,KATAGO_LEGACY_STREAM
            );
          else {
            customCudaApplyRoPE(
              (half*)qPtr,(const half*)ropeCosTable,(const half*)ropeSinTable,
              batchSize,seqLen,numHeads,numKVHeads,qHeadDim,ropeNumPairs,false
            );
            customCudaApplyRoPE(
              (half*)kPtr,(const half*)ropeCosTable,(const half*)ropeSinTable,
              batchSize,seqLen,numKVHeads,numKVHeads,qHeadDim,ropeNumPairs,false
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
        qStrideElts,kvStrideElts,KATAGO_LEGACY_STREAM
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
    if(!usedMma && useCombinedQKV)
      throw StringError(name + ": combined QKV plan was rejected by MMA attention");

    if(!usedMma) {
      if(!cudaHandles->loggedUsingScalarAttention) {
        cudaHandles->loggedUsingScalarAttention = true;
        if(cudaHandles->logger != NULL)
          cudaHandles->logger->write("CUDA_TRANSFORMER_ROUTE attention=official_scalar");
      }
      if(!usingFP16)
        customCudaFlashAttention(
          (const float*)qPtr,(const float*)kPtr,(const float*)vPtr,(const float*)maskBuf,
          (float*)attnOutBuf.buf,batchSize,seqLen,numHeads,numKVHeads,qHeadDim,vHeadDim
        );
      else
        customCudaFlashAttention(
          (const half*)qPtr,(const half*)kPtr,(const half*)vPtr,(const half*)maskBuf,
          (half*)attnOutBuf.buf,batchSize,seqLen,numHeads,numKVHeads,qHeadDim,vHeadDim
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
          batchSize,seqLen,inChannels
        );
      else
        customCudaMaskedResidualAddNHWC(
          (half*)trunkBuf,(const half*)trunkScratchBuf,(const half*)maskBuf,
          batchSize,seqLen,inChannels
        );
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }
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
  const MatMulLayer linear2;
  void* ffnPackedWeights;

  TransformerFFNBlock() = delete;
  TransformerFFNBlock(const TransformerFFNBlock&) = delete;
  TransformerFFNBlock& operator=(const TransformerFFNBlock&) = delete;

  TransformerFFNBlock(
    CudaHandles* cudaHandles,
    const TransformerFFNDesc* desc,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ) :
    name(desc->name),
    numChannels(desc->numChannels),
    ffnChannels(desc->ffnChannels),
    useSwiGLU(desc->useSwiGLU),
    nnXLen(nnX),
    nnYLen(nnY),
    usingFP16(useFP16),
    usingNHWC(useNHWC),
    preLN(cudaHandles,&desc->preLN,useFP16),
    linear2(cudaHandles,&desc->linear2,useFP16),
    ffnPackedWeights(NULL)
  {
    if(!useSwiGLU)
      throw StringError("Non-SwiGLU transformer FFN is not supported by the CUDA backend");
    if(!useNHWC)
      throw StringError("Transformer blocks with NCHW layout are not supported by the CUDA backend");
    if(
      desc->linear1.inChannels != desc->linearGate.inChannels ||
      desc->linear1.outChannels != desc->linearGate.outChannels
    )
      throw StringError(name + ": FFN linear and gate projections must share a shape");

    vector<float> packed;
    packed.reserve(desc->linear1.weights.size() + desc->linearGate.weights.size());
    packed.insert(packed.end(),desc->linear1.weights.begin(),desc->linear1.weights.end());
    packed.insert(packed.end(),desc->linearGate.weights.begin(),desc->linearGate.weights.end());
    CudaUtils::mallocAndCopyToDevice(name + ":ffnPacked",packed,ffnPackedWeights,useFP16);
  }

  ~TransformerFFNBlock() {
    if(ffnPackedWeights != NULL) cudaFree(ffnPackedWeights);
  }

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

    // Official unfused FFN plan: one strided-batched GEMM produces both
    // linear and gate buffers, followed by the shared SwiGLU kernel.
    SizedBuf<void*> projected(scratch->allocator,scratch->getBufSizeXY(2 * ffnChannels));
    void* linearBuf = projected.buf;
    void* gateBuf = (char*)projected.buf + scratch->getBufSizeXY(ffnChannels);
    applySharedInputStridedMatMuls(
      cudaHandles,scratch,usingFP16,name,
      ffnChannels,matBatchSize,numChannels,
      ffnPackedWeights,trunkScratchBuf,linearBuf,
      (long long)(scratch->getBufSizeXY(ffnChannels) / (usingFP16 ? sizeof(half) : sizeof(float))),2
    );

    const int totalSize = (int)((size_t)ffnChannels * matBatchSize);
    if(!usingFP16)
      customCudaSwiGLU((const float*)linearBuf,(const float*)gateBuf,(float*)linearBuf,totalSize);
    else
      customCudaSwiGLU((const half*)linearBuf,(const half*)gateBuf,(half*)linearBuf,totalSize);
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());

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
          batchSize,seqLen,numChannels
        );
      else
        customCudaMaskedResidualAddNHWC(
          (half*)trunkBuf,(const half*)trunkScratchBuf,(const half*)maskBuf,
          batchSize,seqLen,numChannels
        );
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }
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
      unique_ptr_void blockPtr = make_unique_void(
        new TransformerFFNBlock(
          cudaHandles,
          blockDesc,
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
        customCudaAddNCBiasInplaceNCHW((float*)trunkScratch.buf,(const float*)trunkBuf,batchSize,trunkNumChannels,nnXLen*nnYLen);
      else
        customCudaAddNCBiasInplaceNHWC((float*)trunkScratch.buf,(const float*)trunkBuf,batchSize,nnXLen*nnYLen,trunkNumChannels);
    }
    else {
      if(!usingNHWC)
        customCudaAddNCBiasInplaceNCHW((half*)trunkScratch.buf,(const half*)trunkBuf,batchSize,trunkNumChannels,nnXLen*nnYLen);
      else
        customCudaAddNCBiasInplaceNHWC((half*)trunkScratch.buf,(const half*)trunkBuf,batchSize,nnXLen*nnYLen,trunkNumChannels);
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

static void fillMaskFloatBufAndMaskSumBuf(void* maskBuf, float*& maskFloatBuf, float*& maskSumBuf, bool usingFP16, int batchSize, int nnXLen, int nnYLen) {
  if(!usingFP16) {
    maskFloatBuf = (float*)maskBuf;
    customCudaPoolRowsSumNCHW((const float*)maskFloatBuf,maskSumBuf,batchSize,1,nnXLen*nnYLen,1.0);
    CUDA_ERR("sumMask",cudaPeekAtLastError());
  }
  else {
    customCudaCopyFromHalf((const half*)maskBuf,maskFloatBuf,batchSize*nnXLen*nnYLen);
    CUDA_ERR("copyMaskFromHalf",cudaPeekAtLastError());
    customCudaPoolRowsSumNCHW((const float*)maskFloatBuf,maskSumBuf,batchSize,1,nnXLen*nnYLen,1.0);
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
        customCudaPoolRowsGPoolNCHW((const float*)g1Out2.buf,(float*)g1Concat.buf,batchSize,g1Channels,nnXLen*nnYLen,maskFloatBuf,maskSumBuf);
      else
        customCudaPoolRowsGPoolNHWC((const float*)g1Out2.buf,(float*)g1Concat.buf,batchSize,nnXLen*nnYLen,g1Channels,maskFloatBuf,maskSumBuf);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }
    else {
      customCudaCopyFromHalf((const half*)g1Out2.buf,(float*)workspaceBuf,batchSize*g1Channels*nnXLen*nnYLen);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      if(!usingNHWC)
        customCudaPoolRowsGPoolNCHW((const float*)workspaceBuf,(float*)g1Concat.buf,batchSize,g1Channels,nnXLen*nnYLen,maskFloatBuf,maskSumBuf);
      else
        customCudaPoolRowsGPoolNHWC((const float*)workspaceBuf,(float*)g1Concat.buf,batchSize,nnXLen*nnYLen,g1Channels,maskFloatBuf,maskSumBuf);
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
      customCudaCopyFromHalf((const half*)p1Out.buf,(float*)p1Out2.buf,batchSize*p1Channels*nnXLen*nnYLen);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      p1OutBufA = (float*)p1Out2.buf;
      p1OutBufB = (float*)p1Out.buf;
    }

    if(!usingNHWC)
      customCudaAddNCBiasInplaceNCHW(p1OutBufA,(float*)g1Bias.buf,batchSize,p1Channels,nnXLen*nnYLen);
    else
      customCudaAddNCBiasInplaceNHWC(p1OutBufA,(float*)g1Bias.buf,batchSize,nnXLen*nnYLen,p1Channels);
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
      batchSize
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
      customCudaCopyFromHalf((const half*)v1Out2.buf,(float*)workspaceBuf,batchSize*v1Channels*nnXLen*nnYLen);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      bufToBePooled = workspaceBuf;
    }

    if(!usingNHWC)
      customCudaValueHeadPoolNCHW((float*)bufToBePooled,(float*)v1Mean.buf,batchSize,v1Channels,nnXLen*nnYLen,maskSumBuf);
    else
      customCudaValueHeadPoolNHWC((const float*)bufToBePooled,(float*)v1Mean.buf,batchSize,nnXLen*nnYLen,v1Channels,maskSumBuf);
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
      customCudaCopyFromHalf((const half*)ownershipScratch.buf,(float*)ownershipBuf,batchSize*ownershipChannels*nnXLen*nnYLen);
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
        customCudaChannel0ExtractNHWC((const float*)inputBuf, (float*)maskBuf, batchSize, nnXLen*nnYLen, numInputChannels);
      else
        customCudaChannel0ExtractNCHW((const float*)inputBuf, (float*)maskBuf, batchSize, numInputChannels, nnXLen*nnYLen);
      CUDA_ERR("modelExtractMask",cudaPeekAtLastError());
    }
    else {
      if(inputsUsingNHWC)
        customCudaChannel0ExtractNHWC((const half*)inputBuf, (half*)maskBuf, batchSize, nnXLen*nnYLen, numInputChannels);
      else
        customCudaChannel0ExtractNCHW((const half*)inputBuf, (half*)maskBuf, batchSize, numInputChannels, nnXLen*nnYLen);
      CUDA_ERR("modelExtractMask",cudaPeekAtLastError());
    }

    fillMaskFloatBufAndMaskSumBuf(maskBuf,maskFloatBuf,maskSumBuf,usingFP16,batchSize,nnXLen,nnYLen);

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
  const LoadedModel* loadedModel
) {
  (void)gpuIdxs;
  (void)logger;
  (void)openCLTunerFile;
  (void)homeDataDirOverride;
  (void)openCLReTunePerBoardSize;
  (void)loadedModel;

  ComputeContext* context = new ComputeContext();
  context->nnXLen = nnXLen;
  context->nnYLen = nnYLen;
  context->useFP16Mode = useFP16Mode;
  context->useNHWCMode = useNHWCMode;
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
    Logger* logger,
    int majorComputeCapability,
    int minorComputeCapability,
    int maxBatchSize,
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
    cudaHandles = std::make_unique<CudaHandles>(majorComputeCapability,minorComputeCapability);
    cudaHandles->logger = logger;
    // Probe before constructing the model because an enabled MMA path commits
    // attention weights to the official interleaved combined-QKV layout.
    cudaHandles->mmaAttentionEnabled =
      useFP16 && majorComputeCapability >= 8 && customCudaFlashAttentionMmaSupported();
    model = std::make_unique<Model>(
      cudaHandles.get(), &(loadedModel->modelDesc), maxBatchSize,
      nnXLen, nnYLen, inputsUseNHWC, useFP16, useNHWC
    );
    scratch = std::make_unique<ScratchBuffers>(maxBatchSize, nnXLen, nnYLen, useFP16);
    buffers = std::make_unique<Buffers>(cudaHandles.get(), *model, *scratch);

    //Synchronize after creating buffers and copying all the weights, just in case
    CUDA_ERR("ComputeHandle", cudaDeviceSynchronize());
  }
  ~ComputeHandle() {
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
  (void)backendNumThreads; // Unused
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
    context,loadedModel,logger,prop.major,prop.minor,maxBatchSize,requireExactNNLen,inputsUseNHWC,useFP16,useNHWC
  );
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

    CUDA_ERR("getOutput",cudaMemcpy(buffers->inputBuf, inputBuffers->userInputBuffer, inputBuffers->singleInputBytes*batchSize, cudaMemcpyHostToDevice));
    CUDA_ERR("getOutput",cudaMemcpy(buffers->inputGlobalBuf, inputBuffers->userInputGlobalBuffer, inputBuffers->singleInputGlobalBytes*batchSize, cudaMemcpyHostToDevice));
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

    CUDA_ERR("getOutput",cudaMemcpy(buffers->inputBufFloat, inputBuffers->userInputBuffer, inputBuffers->singleInputBytes*batchSize, cudaMemcpyHostToDevice));
    CUDA_ERR("getOutput",cudaMemcpy(buffers->inputGlobalBufFloat, inputBuffers->userInputGlobalBuffer, inputBuffers->singleInputGlobalBytes*batchSize, cudaMemcpyHostToDevice));

    customCudaCopyToHalf((const float*)buffers->inputBufFloat,(half*)buffers->inputBuf,inputBuffers->singleInputElts*batchSize);
    CUDA_ERR("getOutput",cudaPeekAtLastError());
    customCudaCopyToHalf((const float*)buffers->inputGlobalBufFloat,(half*)buffers->inputGlobalBuf,inputBuffers->singleInputGlobalElts*batchSize);
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

  CUDA_ERR("getOutput",cudaMemcpy(inputBuffers->policyResults, buffers->policyBuf, inputBuffers->singlePolicyResultBytes*batchSize, cudaMemcpyDeviceToHost));
  CUDA_ERR("getOutput",cudaMemcpy(inputBuffers->valueResults, buffers->valueBuf, inputBuffers->singleValueResultBytes*batchSize, cudaMemcpyDeviceToHost));
  CUDA_ERR("getOutput",cudaMemcpy(inputBuffers->scoreValueResults, buffers->scoreValueBuf, inputBuffers->singleScoreValueResultBytes*batchSize, cudaMemcpyDeviceToHost));
  CUDA_ERR("getOutput",cudaMemcpy(inputBuffers->ownershipResults, buffers->ownershipBuf, inputBuffers->singleOwnershipResultBytes*batchSize, cudaMemcpyDeviceToHost));

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

  fillMaskFloatBufAndMaskSumBuf(deviceMask, deviceMaskFloat, deviceMaskSum, useFP16, desiredBatchSize, nnXLen, nnYLen);

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

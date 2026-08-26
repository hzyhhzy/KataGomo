#ifdef USE_CPU_PTQ_BACKEND

#include "../neuralnet/nninterface.h"

#include "../neuralnet/cpuptq/kernel.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nneval.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#include <immintrin.h>
#endif

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using namespace std;

namespace {

[[noreturn]] void failBoundary(const string& message) {
  assert(false && "CPU PTQ backend boundary mismatch");
  throw StringError("CPU PTQ backend boundary mismatch: " + message);
}

void requireBoundary(bool condition, const string& message) {
  if(!condition)
    failBoundary(message);
}

bool hasRequiredCpuIsa() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
  int regs[4] = {};
  __cpuid(regs,0);
  if(regs[0] < 7)
    return false;
  __cpuidex(regs,1,0);
  const bool osxsave = (regs[2] & (1 << 27)) != 0;
  const bool avx = (regs[2] & (1 << 28)) != 0;
  if(!osxsave || !avx)
    return false;
  const unsigned __int64 xcr0 = _xgetbv(0);
#if defined(CPU_PTQ_AVX2_ONLY)
  if((xcr0 & 0x6) != 0x6)
    return false;
  const bool sse41 = (regs[2] & (1 << 19)) != 0;
  const bool sse42 = (regs[2] & (1 << 20)) != 0;
  const bool popcnt = (regs[2] & (1 << 23)) != 0;
  const bool fma = (regs[2] & (1 << 12)) != 0;
  __cpuidex(regs,7,0);
  const bool avx2 = (regs[1] & (1 << 5)) != 0;
  return sse41 && sse42 && popcnt && fma && avx2;
#else
  if((xcr0 & 0xE6) != 0xE6)
    return false;
  __cpuidex(regs,7,0);
  const bool avx512f = (regs[1] & (1 << 16)) != 0;
  const bool avx512dq = (regs[1] & (1 << 17)) != 0;
  const bool avx512bw = (regs[1] & (1 << 30)) != 0;
  const bool avx512vl = (regs[1] & (1u << 31)) != 0;
  const bool avx512vnni = (regs[2] & (1 << 11)) != 0;
  return avx512f && avx512dq && avx512bw && avx512vl && avx512vnni;
#endif
#elif (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
  unsigned int eax = 0;
  unsigned int ebx = 0;
  unsigned int ecx = 0;
  unsigned int edx = 0;
  if(!__get_cpuid_max(0,nullptr) || !__get_cpuid(1,&eax,&ebx,&ecx,&edx))
    return false;
  if((ecx & bit_OSXSAVE) == 0 || (ecx & bit_AVX) == 0)
    return false;
  const uint64_t xcr0 = _xgetbv(0);
#if defined(CPU_PTQ_AVX2_ONLY)
  if((xcr0 & 0x6) != 0x6)
    return false;
  if((ecx & bit_SSE4_1) == 0 || (ecx & bit_SSE4_2) == 0 ||
     (ecx & bit_POPCNT) == 0 || (ecx & bit_FMA) == 0)
    return false;
  if(!__get_cpuid_count(7,0,&eax,&ebx,&ecx,&edx))
    return false;
  return (ebx & bit_AVX2) != 0;
#else
  if((xcr0 & 0xE6) != 0xE6)
    return false;
  if(!__get_cpuid_count(7,0,&eax,&ebx,&ecx,&edx))
    return false;
  return (ebx & bit_AVX512F) != 0 &&
         (ebx & bit_AVX512DQ) != 0 &&
         (ebx & bit_AVX512BW) != 0 &&
         (ebx & bit_AVX512VL) != 0 &&
         (ecx & bit_AVX512VNNI) != 0;
#endif
#else
  return false;
#endif
}

}  // namespace

struct LoadedModel {
  ModelDesc modelDesc;

  LoadedModel(const string& fileName, const string& expectedSha256) {
    ModelDesc::loadFromFileMaybeGZipped(fileName,modelDesc,expectedSha256);
    (void)CpuPtq::selectProfile(modelDesc);
  }

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

LoadedModel* NeuralNet::loadModelFile(const string& file, const string& expectedSha256) {
  return new LoadedModel(file,expectedSha256);
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

Rules NeuralNet::getSupportedRules(
  const LoadedModel* loadedModel,
  const Rules& desiredRules,
  bool& supported
) {
  return loadedModel->modelDesc.getSupportedRules(desiredRules,supported);
}

const ModelDesc& NeuralNet::getModelDesc(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc;
}

struct ComputeContext {
  int nnXLen;
  int nnYLen;
  const CpuPtq::ProfileSpec* profile;
};

struct ComputeHandle {
  const ComputeContext* context;
  std::unique_ptr<CpuPtq::Kernel> kernel;

  ComputeHandle(const ComputeContext* ctx, const LoadedModel& model)
    : context(ctx) {
    CpuPtq::TensorMap tensors = CpuPtq::makeKernelTensors(model.modelDesc,*ctx->profile);
    using Factory = std::unique_ptr<CpuPtq::Kernel> (*)(const CpuPtq::TensorMap&);
    struct Entry {
      CpuPtq::ProfileKind kind;
      Factory factory;
    };
    static constexpr Entry entries[] = {
      {CpuPtq::ProfileKind::B24C192H6F512,CpuPtq::createB24Kernel},
      {CpuPtq::ProfileKind::B16C128H4F384,CpuPtq::createB16Kernel},
      {CpuPtq::ProfileKind::B11C96H3F256,CpuPtq::createB11Kernel},
    };
    for(const Entry& entry: entries) {
      if(entry.kind == ctx->profile->kind) {
        kernel = entry.factory(tensors);
        return;
      }
    }
    failBoundary("selected profile has no compiled kernel");
  }
};

struct InputBuffers {
  int maxBatchSize;
  vector<float> spatialInput;
  vector<float> globalInput;
  vector<float> policy;
  vector<float> value;
  vector<float> scoreValue;
  vector<float> ownership;

  InputBuffers(int maxBatchSz, int nnXLen, int nnYLen)
    : maxBatchSize(maxBatchSz),
      spatialInput((size_t)CpuPtq::SPATIAL_INPUTS * nnXLen * nnYLen),
      globalInput(CpuPtq::GLOBAL_INPUTS),
      policy(CpuPtq::POLICY_SIZE),
      value(CpuPtq::VALUE_SIZE),
      scoreValue(CpuPtq::SCORE_VALUE_SIZE),
      ownership(CpuPtq::OWNERSHIP_SIZE) {}
};

void NeuralNet::globalInitialize() {
#ifdef _WIN32
  constexpr DWORD noDialogs =
    SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX;
  SetErrorMode(noDialogs);
  DWORD previous = 0;
  SetThreadErrorMode(noDialogs,&previous);
#endif
}

void NeuralNet::globalCleanup() {}

void NeuralNet::printDevices() {
#if defined(CPU_PTQ_AVX2_ONLY)
  cout << "CPU-PTQ device 0: AVX2/FMA single-thread (AVX-512/VNNI disabled)" << endl;
#else
  cout << "CPU-PTQ device 0: AVX-512 VNNI single-thread" << endl;
#endif
}

ComputeContext* NeuralNet::createComputeContext(
  const vector<int>& gpuIdxs,
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
  (void)useINT8;
  requireBoundary(loadedModel != nullptr,"loaded model is null");
  requireBoundary(nnXLen == CpuPtq::BOARD_LEN && nnYLen == CpuPtq::BOARD_LEN,
                  "board must be exactly 15x15");
  requireBoundary(useFP16Mode != enabled_t::True,"FP16 is unsupported");
  requireBoundary(useNHWCMode != enabled_t::True,"backend internal NHWC mode is unsupported");
#if defined(CPU_PTQ_AVX2_ONLY)
  requireBoundary(
    hasRequiredCpuIsa(),
    "CPU/OS must expose SSE4.1/SSE4.2/POPCNT/AVX/AVX2/FMA and AVX XSAVE state");
#else
  requireBoundary(
    hasRequiredCpuIsa(),
    "CPU/OS must expose AVX-512F/DQ/BW/VL and VNNI");
#endif
  const CpuPtq::ProfileSpec& profile = CpuPtq::selectProfile(loadedModel->modelDesc);
  if(logger != nullptr)
    logger->write("CPU-PTQ backend: selected " + string(profile.name));
  return new ComputeContext{nnXLen,nnYLen,&profile};
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

ComputeHandle* NeuralNet::createComputeHandle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  bool requireExactNNLen,
  bool inputsUseNHWC,
  int gpuIdxForThisThread,
  int serverThreadIdx,
  int backendNumThreads
) {
  (void)gpuIdxForThisThread;
  requireBoundary(context != nullptr && loadedModel != nullptr,"null compute input");
  requireBoundary(maxBatchSize == 1,"maxBatchSize must be exactly one");
  requireBoundary(requireExactNNLen,"requireExactNNLen must be true");
  requireBoundary(!inputsUseNHWC,"engine inputs must use NCHW");
  requireBoundary(serverThreadIdx == 0,"only server thread zero is supported");
  requireBoundary(backendNumThreads == 1,"backendNumThreads must be one");
  if(logger != nullptr) {
    logger->write(
      "CPU-PTQ backend thread 0: model v106 " + loadedModel->modelDesc.name +
      " profile=" + context->profile->name);
  }
  return new ComputeHandle(context,*loadedModel);
}

void NeuralNet::freeComputeHandle(ComputeHandle* computeHandle) {
  delete computeHandle;
}

bool NeuralNet::isUsingFP16(const ComputeHandle* computeHandle) {
  (void)computeHandle;
  return false;
}

InputBuffers* NeuralNet::createInputBuffers(
  const LoadedModel* loadedModel,
  int maxBatchSize,
  int nnXLen,
  int nnYLen
) {
  requireBoundary(loadedModel != nullptr,"loaded model is null");
  requireBoundary(maxBatchSize == 1,"input buffer maxBatchSize must be one");
  requireBoundary(nnXLen == CpuPtq::BOARD_LEN && nnYLen == CpuPtq::BOARD_LEN,
                  "input buffer board must be 15x15");
  return new InputBuffers(maxBatchSize,nnXLen,nnYLen);
}

void NeuralNet::freeInputBuffers(InputBuffers* buffers) {
  delete buffers;
}

void NeuralNet::getOutput(
  ComputeHandle* computeHandle,
  InputBuffers* buffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs,
  float* outputPolicys
) {
  requireBoundary(computeHandle != nullptr && buffers != nullptr,"null inference handle");
  requireBoundary(numBatchEltsFilled == 1,"runtime batch size must be one");
  requireBoundary(outputs.size() == 1 && outputs[0] != nullptr,"one output row is required");
  requireBoundary(inputBufs != nullptr && inputBufs[0] != nullptr,"one input row is required");
  requireBoundary(outputPolicys != nullptr,"policy output buffer is null");

  const NNResultBuf* input = inputBufs[0];
  SymmetryHelpers::copyInputsWithSymmetry(
    input->rowSpatial,buffers->spatialInput.data(),1,
    CpuPtq::BOARD_LEN,CpuPtq::BOARD_LEN,CpuPtq::SPATIAL_INPUTS,
    false,input->symmetry);
  std::copy(
    input->rowGlobal,input->rowGlobal + CpuPtq::GLOBAL_INPUTS,
    buffers->globalInput.begin());

  computeHandle->kernel->infer(
    buffers->spatialInput.data(),buffers->globalInput.data(),
    buffers->policy.data(),buffers->value.data(),buffers->scoreValue.data(),
    buffers->ownership.data());

  NNOutput* output = outputs[0];
  requireBoundary(
    output->nnXLen == CpuPtq::BOARD_LEN && output->nnYLen == CpuPtq::BOARD_LEN,
    "NNOutput board mismatch");
  SymmetryHelpers::copyOutputsWithSymmetry(
    buffers->policy.data(),outputPolicys,1,
    CpuPtq::BOARD_LEN,CpuPtq::BOARD_LEN,input->symmetry);
  outputPolicys[CpuPtq::BOARD_AREA] = buffers->policy[CpuPtq::BOARD_AREA];
  output->whiteWinProb = buffers->value[0];
  output->whiteLossProb = buffers->value[1];
  output->whiteNoResultProb = buffers->value[2];
  output->varTimeLeft = buffers->scoreValue[3];
  output->shorttermWinlossError = buffers->scoreValue[4];
}

#ifdef KATAGO_BUILD_NNRAWGATE
void NeuralNet::getRawNNGateOutputs(const InputBuffers* buffers, RawNNGateOutputs& out) {
  requireBoundary(buffers != nullptr,"raw-output buffers are null");
  out.policy = buffers->policy.data();
  out.value = buffers->value.data();
  out.scoreValue = buffers->scoreValue.data();
  out.ownership = buffers->ownership.data();
  out.policyElts = CpuPtq::POLICY_SIZE;
  out.valueElts = CpuPtq::VALUE_SIZE;
  out.scoreValueElts = CpuPtq::SCORE_VALUE_SIZE;
  out.ownershipElts = CpuPtq::OWNERSHIP_SIZE;
}
#endif

bool NeuralNet::testEvaluateConv(
  const ConvLayerDesc*, int, int, int, bool, bool,
  const vector<float>&, vector<float>&
) { return false; }

bool NeuralNet::testEvaluateBatchNorm(
  const BatchNormLayerDesc*, int, int, int, bool, bool,
  const vector<float>&, const vector<float>&, vector<float>&
) { return false; }

bool NeuralNet::testEvaluateResidualBlock(
  const ResidualBlockDesc*, int, int, int, bool, bool,
  const vector<float>&, const vector<float>&, vector<float>&
) { return false; }

bool NeuralNet::testEvaluateGlobalPoolingResidualBlock(
  const GlobalPoolingResidualBlockDesc*, int, int, int, bool, bool,
  const vector<float>&, const vector<float>&, vector<float>&
) { return false; }

#endif  // USE_CPU_PTQ_BACKEND

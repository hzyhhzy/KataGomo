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

#if defined(__x86_64__) && defined(__GNUC__)
#include <cpuid.h>
#include <immintrin.h>
#endif

using namespace std;

namespace {

[[noreturn]] void failBoundary(const string& message) {
  assert(false && "CPU-PTQ backend boundary mismatch");
  throw StringError("CPU-PTQ backend boundary mismatch: " + message);
}

void requireBoundary(bool condition, const string& message) {
  if(!condition)
    failBoundary(message);
}

bool hasRequiredCpuIsa() {
#if defined(__x86_64__) && defined(__GNUC__)
  unsigned int eax = 0;
  unsigned int ebx = 0;
  unsigned int ecx = 0;
  unsigned int edx = 0;
  if(!__get_cpuid_max(0,nullptr) || !__get_cpuid(1,&eax,&ebx,&ecx,&edx))
    return false;
  if((ecx & bit_OSXSAVE) == 0 || (ecx & bit_AVX) == 0)
    return false;
  unsigned int xcr0Low = 0;
  unsigned int xcr0High = 0;
  __asm__ volatile("xgetbv" : "=a"(xcr0Low), "=d"(xcr0High) : "c"(0));
  const uint64_t xcr0 = static_cast<uint64_t>(xcr0Low) |
    (static_cast<uint64_t>(xcr0High) << 32);
  if((xcr0 & 0xE6U) != 0xE6U)
    return false;
  if(!__get_cpuid_count(7,0,&eax,&ebx,&ecx,&edx))
    return false;
  constexpr unsigned int avx512Vnni = 1U << 11;
  return
    (ebx & bit_AVX512F) != 0 &&
    (ebx & bit_AVX512DQ) != 0 &&
    (ebx & bit_AVX512BW) != 0 &&
    (ebx & bit_AVX512VL) != 0 &&
    (ecx & avx512Vnni) != 0;
#else
  return false;
#endif
}

}  // namespace

struct LoadedModel {
  CpuPtq::Model cpuModel;
  ModelDesc modelDesc;

  LoadedModel(const string& fileName, const string& expectedSha256)
    : cpuModel(CpuPtq::loadModelFile(fileName,expectedSha256)) {
    modelDesc.name = cpuModel.name;
    modelDesc.sha256 = cpuModel.sha256;
    modelDesc.version = CpuPtq::MODEL_VERSION;
    modelDesc.numInputChannels = CpuPtq::SPATIAL_INPUTS;
    modelDesc.numInputGlobalChannels = CpuPtq::GLOBAL_INPUTS;
    modelDesc.numValueChannels = CpuPtq::VALUE_SIZE;
    modelDesc.numScoreValueChannels = CpuPtq::MISC_VALUE_SIZE;
    modelDesc.numOwnershipChannels = 0;
  }

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

LoadedModel* NeuralNet::loadModelFile(
  const string& file,
  const string& expectedSha256
) {
  return new LoadedModel(file,expectedSha256);
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}

string NeuralNet::getModelName(const LoadedModel* loadedModel) {
  return loadedModel->cpuModel.name;
}

int NeuralNet::getModelVersion(const LoadedModel* loadedModel) {
  (void)loadedModel;
  return CpuPtq::MODEL_VERSION;
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

  ComputeHandle(const ComputeContext* context_, const LoadedModel& model)
    : context(context_), kernel(CpuPtq::createKernel(model.cpuModel)) {}
};

struct InputBuffers {
  vector<float> spatial;
  vector<float> global;
  vector<float> policy;
  vector<float> value;
  vector<float> miscValue;

  InputBuffers()
    : spatial(CpuPtq::SPATIAL_INPUTS * CpuPtq::BOARD_AREA),
      global(CpuPtq::GLOBAL_INPUTS),
      policy(CpuPtq::POLICY_SIZE),
      value(CpuPtq::VALUE_SIZE),
      miscValue(CpuPtq::MISC_VALUE_SIZE) {}
};

void NeuralNet::globalInitialize() {}
void NeuralNet::globalCleanup() {}

void NeuralNet::printDevices() {
  cout << "CPU-PTQ device 0: Ice Lake AVX-512 VNNI S8 single-thread" << endl;
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
  const LoadedModel* loadedModel
) {
  (void)gpuIdxs;
  (void)openCLTunerFile;
  (void)homeDataDirOverride;
  (void)openCLReTunePerBoardSize;
  requireBoundary(loadedModel != nullptr,"loaded model is null");
  requireBoundary(
    nnXLen == CpuPtq::BOARD_LEN && nnYLen == CpuPtq::BOARD_LEN,
    "board must be exactly 7x7");
  requireBoundary(useFP16Mode != enabled_t::True,"FP16 is unsupported");
  requireBoundary(useNHWCMode != enabled_t::True,"NHWC is unsupported");
  requireBoundary(
    hasRequiredCpuIsa(),
    "CPU/OS must expose AVX-512F/DQ/BW/VL, AVX-512 VNNI, and ZMM XSAVE state");
  if(logger != nullptr)
    logger->write(
      "CPU-PTQ backend: selected " + string(loadedModel->cpuModel.profile->name) +
      " for Ice Lake AVX-512 VNNI");
  return new ComputeContext{
    nnXLen,nnYLen,loadedModel->cpuModel.profile
  };
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
  requireBoundary(maxBatchSize == 1,"maxBatchSize must be one");
  requireBoundary(requireExactNNLen,"requireExactNNLen must be true");
  requireBoundary(!inputsUseNHWC,"engine input layout must be NCHW");
  requireBoundary(serverThreadIdx == 0,"only NN server thread zero is supported");
  requireBoundary(backendNumThreads == 1,"backendNumThreads must be one");
  if(logger != nullptr)
    logger->write(
      "CPU-PTQ backend thread 0: v206 " + loadedModel->cpuModel.name +
      " profile=" + context->profile->name);
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
  requireBoundary(maxBatchSize == 1,"input maxBatchSize must be one");
  requireBoundary(
    nnXLen == CpuPtq::BOARD_LEN && nnYLen == CpuPtq::BOARD_LEN,
    "input board must be 7x7");
  return new InputBuffers();
}

void NeuralNet::freeInputBuffers(InputBuffers* buffers) {
  delete buffers;
}

void NeuralNet::getOutput(
  ComputeHandle* computeHandle,
  InputBuffers* buffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs
) {
  requireBoundary(computeHandle != nullptr && buffers != nullptr,"null inference handle");
  requireBoundary(numBatchEltsFilled == 1,"runtime batch size must be one");
  requireBoundary(outputs.size() == 1 && outputs[0] != nullptr,"one output is required");
  requireBoundary(inputBufs != nullptr && inputBufs[0] != nullptr,"one input is required");

  const NNResultBuf* input = inputBufs[0];
  SymmetryHelpers::copyInputsWithSymmetry(
    input->rowSpatial,buffers->spatial.data(),1,
    CpuPtq::BOARD_LEN,CpuPtq::BOARD_LEN,CpuPtq::SPATIAL_INPUTS,
    false,input->symmetry);
  std::copy(
    input->rowGlobal,input->rowGlobal + CpuPtq::GLOBAL_INPUTS,
    buffers->global.begin());

  computeHandle->kernel->infer(
    buffers->spatial.data(),buffers->global.data(),buffers->policy.data(),
    buffers->value.data(),buffers->miscValue.data());

  NNOutput* output = outputs[0];
  requireBoundary(
    output->nnXLen == CpuPtq::BOARD_LEN && output->nnYLen == CpuPtq::BOARD_LEN,
    "NNOutput board mismatch");
  SymmetryHelpers::copyOutputsWithSymmetry(
    buffers->policy.data(),output->policyProbs,1,
    CpuPtq::BOARD_LEN,CpuPtq::BOARD_LEN,input->symmetry);
  output->policyProbs[CpuPtq::BOARD_AREA] = buffers->policy[CpuPtq::BOARD_AREA];
  output->whiteWinProb = buffers->value[0];
  output->whiteLossProb = buffers->value[1];
  output->whiteNoResultProb = buffers->value[2];
  output->varTimeLeft = buffers->miscValue[3];
  output->shorttermWinlossError = buffers->miscValue[4];
}

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

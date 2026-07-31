#include "../neuralnet/nninterface.h"
#include "../neuralnet/onnxprotoreader.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/modelversion.h"
#include "../core/os.h"
#include "../core/fileutils.h"
#include "../dataio/homedata.h"

#include <onnxruntime_cxx_api.h>
#ifdef USE_ONNX_DIRECTML_BACKEND
#ifndef _WIN32
#error "onnx-directml only supports Windows"
#endif

#if __has_include(<dml_provider_factory.h>)
#include <dml_provider_factory.h>
#else
#error "Please manually install the Microsoft.ML.OnnxRuntime.DirectML NuGet package"
#endif

#if __has_include("../build/packages/Microsoft.ML.OnnxRuntime.DirectML.1.24.4/runtimes/win-x64/native/onnxruntime.lib")
#pragma comment(lib,"../build/packages/Microsoft.ML.OnnxRuntime.DirectML.1.24.4/runtimes/win-x64/native/onnxruntime.lib")
#else
#error "Please check the location of the lib file"
#endif
#endif

#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <map>
#include <mutex>
#include <codecvt>
#include <locale>

using namespace std;

//---------------------------------------------------------------------------------------------------------

static void checkOrtStatus(OrtStatus* status) {
  if (status != NULL) {
    string msg = Ort::GetApi().GetErrorMessage(status);
    Ort::GetApi().ReleaseStatus(status);
    throw StringError("ONNX Runtime Error: " + msg);
  }
}

static constexpr int ONNX_V112_OUTPUT_HEADS = 6;
static constexpr int ONNX_SELECTED_OUTPUT_HEAD = 0;

static int getOnnxOutputHeadCount(int modelVersion) {
  return modelVersion == 112 ? ONNX_V112_OUTPUT_HEADS : 1;
}

static size_t getSelectedOnnxOutputHeadOffset(size_t singleHeadElts, int modelVersion) {
  int headCount = getOnnxOutputHeadCount(modelVersion);
  assert(ONNX_SELECTED_OUTPUT_HEAD >= 0 && ONNX_SELECTED_OUTPUT_HEAD < headCount);
  (void)headCount;
  return (size_t)ONNX_SELECTED_OUTPUT_HEAD * singleHeadElts;
}

//---------------------------------------------------------------------------------------------------------

void NeuralNet::globalInitialize() {
  // Initialize generic ONNX Runtime environment if needed?
  // Ort::Env is usually created in ComputeHandle or globally.
}

void NeuralNet::globalCleanup() {
}

struct ComputeContext {
  int nnXLen;
  int nnYLen;
  enabled_t useFP16Mode;
  string onnxModelPath;
};

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

struct LoadedModel {
  ModelDesc modelDesc;
  string fileName;

  LoadedModel(const string& fileName, const string& expectedSha256) {
    this->fileName = fileName;
    if (Global::isSuffix(fileName, ".onnx")) {
      try {
        ModelDesc::loadFromONNX(fileName, modelDesc);
      } catch (const StringError& e) {
        throw StringError("Failed to load ONNX model config: " + fileName + "\n" + e.what());
      }
    } else {
      throw StringError("ONNX backend only supports .onnx files");
    }
  }
};

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
  if(useNHWCMode == enabled_t::True) {
    throw StringError("ONNX backend: useNHWC = false required");
  }

  ComputeContext* context = new ComputeContext();
  context->nnXLen = nnXLen;
  context->nnYLen = nnYLen;
  context->useFP16Mode = useFP16Mode;
  context->onnxModelPath = loadedModel->fileName;
  return context;
}

LoadedModel* NeuralNet::loadModelFile(const string& file, const string& expectedSha256) {
  return new LoadedModel(file, expectedSha256);
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

//---------------------------------------------------------------------------------------------------------

struct ComputeHandle {
  ComputeContext* ctx;
  int maxBatchSize;
  int modelVersion;

  Ort::Env env;
  unique_ptr<Ort::Session> session;
  Ort::MemoryInfo memoryInfo;

  vector<const char*> inputNames;
  vector<const char*> outputNames;

  // Buffers are managed by InputBuffers, but we need to know shapes.

  ComputeHandle(ComputeContext* context, const LoadedModel* loadedModel, int maxBatchSz)
    : ctx(context),
      maxBatchSize(maxBatchSz),
      modelVersion(loadedModel->modelDesc.version),
      env(ORT_LOGGING_LEVEL_WARNING, "KataGo"),
      memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
  {
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef USE_ONNX_DIRECTML_BACKEND
#ifdef _WIN32
    // Enable DirectML
    // Using the C API to append provider to session options
    const OrtApi& ortApi = Ort::GetApi();
    OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_DML(sessionOptions, 0); // device_id = 0
    if (status) {
        // If DML fails, fall back or throw? Throwing is better as user requested DML.
        string msg = ortApi.GetErrorMessage(status);
        ortApi.ReleaseStatus(status);
        throw StringError("Failed to append DirectML execution provider: " + msg);
    }
#endif
#endif

    // Convert path to wide string on Windows if needed by ONNX Runtime
#ifdef _WIN32
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    std::wstring wpath = converter.from_bytes(ctx->onnxModelPath);
    session = make_unique<Ort::Session>(env, wpath.c_str(), sessionOptions);
#else
    session = make_unique<Ort::Session>(env, ctx->onnxModelPath.c_str(), sessionOptions);
#endif

    // Cache input/output names
    Ort::AllocatorWithDefaultOptions allocator;
    size_t numInputNodes = session->GetInputCount();
    for(size_t i = 0; i < numInputNodes; i++) {
      auto name = session->GetInputNameAllocated(i, allocator);
      inputNames.push_back(strdup(name.get()));
    }

    size_t numOutputNodes = session->GetOutputCount();
    for(size_t i = 0; i < numOutputNodes; i++) {
      auto name = session->GetOutputNameAllocated(i, allocator);
      outputNames.push_back(strdup(name.get()));
    }
  }

  ~ComputeHandle() {
    for(const char* name : inputNames) free((void*)name);
    for(const char* name : outputNames) free((void*)name);
  }
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
  int numThreads
) {
  (void)numThreads;
  if(inputsUseNHWC) throw StringError("ONNX backend: inputsUseNHWC = false required");

  // We ignore gpuIdxForThisThread for CPU backend, and use 0 for DirectML (implied).
  // If we wanted to support multiple GPUs with DirectML, we'd need to pass gpuIdx to SessionOptionsAppendExecutionProvider_DML.

  return new ComputeHandle(context, loadedModel, maxBatchSize);
}

void NeuralNet::freeComputeHandle(ComputeHandle* gpuHandle) {
  delete gpuHandle;
}

bool NeuralNet::isUsingFP16(const ComputeHandle* gpuHandle) {
  return false; // Or check if model is fp16? For now assume fp32 interface.
}

void NeuralNet::printDevices() {
  // TODO: query ONNX providers
  cout << "ONNX Runtime Devices: CPU";
#ifdef USE_ONNX_DIRECTML_BACKEND
  cout << ", DirectML";
#endif
  cout << endl;
}

//---------------------------------------------------------------------------------------------------------

struct InputBuffers {
  int maxBatchSize;

  size_t singleInputElts;
  size_t singleInputGlobalElts;

  // Outputs
  size_t singleout_policyHeadElts;
  size_t singleout_valueHeadElts;
  size_t singleout_miscvalueHeadElts;
  size_t singleout_moremiscvalueHeadElts;
  size_t singleout_ownershipHeadElts;

  size_t singleout_policyElts;
  size_t singleout_valueElts;
  size_t singleout_miscvalueElts;
  size_t singleout_moremiscvalueElts;
  size_t singleout_ownershipElts;

  unique_ptr<float[]> spatialInputs;
  unique_ptr<float[]> globalInputs;

  unique_ptr<float[]> out_policyResults;
  unique_ptr<float[]> out_valueResults;
  unique_ptr<float[]> out_miscvalueResults;
  unique_ptr<float[]> out_moremiscvalueResults;
  unique_ptr<float[]> out_ownershipResults;

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
    const ModelDesc& m = loadedModel->modelDesc;
    maxBatchSize = maxBatchSz;

    singleInputElts = m.numInputChannels * nnXLen * nnYLen;
    singleInputGlobalElts = m.numInputGlobalChannels;

    // Output sizes - following trtbackend.cpp logic for ONNX
    int policyNum = (m.version >= 12 && m.version <= 99) ? 6 : 4;
    if(m.version >= 12 && m.version <= 99)
      throw StringError("modelVersion >= 12 && modelVersion <= 99 not supported");
    int outputHeadCount = getOnnxOutputHeadCount(m.version);
    singleout_policyHeadElts = (size_t)policyNum * (nnXLen * nnYLen + 1);
    singleout_valueHeadElts = 3;
    singleout_miscvalueHeadElts = 10;
    singleout_moremiscvalueHeadElts = 8;
    singleout_ownershipHeadElts = (size_t)nnXLen * nnYLen;
    singleout_policyElts = (size_t)outputHeadCount * singleout_policyHeadElts;
    singleout_valueElts = (size_t)outputHeadCount * singleout_valueHeadElts;
    singleout_miscvalueElts = (size_t)outputHeadCount * singleout_miscvalueHeadElts;
    singleout_moremiscvalueElts = (size_t)outputHeadCount * singleout_moremiscvalueHeadElts;
    singleout_ownershipElts = (size_t)outputHeadCount * singleout_ownershipHeadElts;

    spatialInputs = make_unique<float[]>(maxBatchSize * singleInputElts);
    globalInputs = make_unique<float[]>(maxBatchSize * singleInputGlobalElts);

    out_policyResults = make_unique<float[]>(maxBatchSize * singleout_policyElts);
    out_valueResults = make_unique<float[]>(maxBatchSize * singleout_valueElts);
    out_miscvalueResults = make_unique<float[]>(maxBatchSize * singleout_miscvalueElts);
    out_moremiscvalueResults = make_unique<float[]>(maxBatchSize * singleout_moremiscvalueElts);
    out_ownershipResults = make_unique<float[]>(maxBatchSize * singleout_ownershipElts);
  }
};

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  return new InputBuffers(loadedModel, maxBatchSize, nnXLen, nnYLen);
}

void NeuralNet::freeInputBuffers(InputBuffers* inputBuffers) {
  delete inputBuffers;
}

void NeuralNet::getOutput(
  ComputeHandle* handle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs,
  float* outputPolicys
) {
  int batchSize = numBatchEltsFilled;
  int nnXLen = handle->ctx->nnXLen;
  int nnYLen = handle->ctx->nnYLen;
  int modelVersion = handle->modelVersion;

  const int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
  const int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);

  // Fill inputs
  for(int nIdx = 0; nIdx < batchSize; nIdx++) {
    float* rowSpatialInput = &inputBuffers->spatialInputs[inputBuffers->singleInputElts * nIdx];
    float* rowGlobalInput = &inputBuffers->globalInputs[inputBuffers->singleInputGlobalElts * nIdx];

    const float* rowGlobal = inputBufs[nIdx]->rowGlobal;
    const float* rowSpatial = inputBufs[nIdx]->rowSpatial;

    copy(rowGlobal, rowGlobal + numGlobalFeatures, rowGlobalInput);
    SymmetryHelpers::copyInputsWithSymmetry(
      rowSpatial, rowSpatialInput, 1, nnYLen, nnXLen, numSpatialFeatures, false, inputBufs[nIdx]->symmetry);
  }

  // Run ONNX inference
  vector<const char*> inputNames = {"input_spatial", "input_global"};
  vector<const char*> outputNames = {"out_policy", "out_value", "out_miscvalue", "out_moremiscvalue", "out_ownership"};

  // Shapes
  int64_t spatialShape[] = {batchSize, numSpatialFeatures, nnYLen, nnXLen};
  int64_t globalShape[] = {batchSize, numGlobalFeatures};

  vector<Ort::Value> inputTensors;
  inputTensors.push_back(Ort::Value::CreateTensor<float>(
      handle->memoryInfo, inputBuffers->spatialInputs.get(), inputBuffers->singleInputElts * batchSize, spatialShape, 4));
  inputTensors.push_back(Ort::Value::CreateTensor<float>(
      handle->memoryInfo, inputBuffers->globalInputs.get(), inputBuffers->singleInputGlobalElts * batchSize, globalShape, 2));

  // Output shapes and tensors
  // We use the buffers directly
  int64_t policyShape[] = {batchSize, (int64_t)inputBuffers->singleout_policyElts};
  int64_t valueShape[] = {batchSize, (int64_t)inputBuffers->singleout_valueElts};
  int64_t miscShape[] = {batchSize, (int64_t)inputBuffers->singleout_miscvalueElts};
  int64_t moremiscShape[] = {batchSize, (int64_t)inputBuffers->singleout_moremiscvalueElts};
  int64_t ownShape[] = {batchSize, (int64_t)inputBuffers->singleout_ownershipElts};

  // Note: The ONNX model output shapes might be slightly different (e.g. including 1s),
  // but CreateTensor with user buffer assumes we know the shape we want to view it as, or we should use what the model expects.
  // Ideally we should match model output shapes.
  // However, Ort::Session::Run will allocate outputs if we don't provide them, OR we provide pre-allocated values.
  // If we provide pre-allocated values, shapes must match.
  // For simplicity, let's let ORT allocate and then copy? No, we want to avoid copy.
  // But we reused InputBuffers buffers.
  // Let's assume the shapes match what we computed.

  // Actually, for "out_policy", trtbackend says "1 * policyNum * (nnXLen * nnYLen + 1)".
  // The shape in ONNX might be [batch, policyNum, nnYLen * nnXLen + 1] or similar.
  // We should probably check `handle->session->GetOutputTypeInfo`.
  // But for now let's try to pass the flat buffers if possible, or correct shapes.

  // To be safe, we can let ORT allocate and copy to our buffers.
  // But wait, we want performance.
  // Let's rely on the fact that trtbackend.cpp defines these sizes based on what the ONNX model produces.

  vector<Ort::Value> outputTensors;
  // We need correct shapes.
  // Re-reading trtbackend.cpp:
  // "profile->setDimensions("input_spatial", ... Dims4(1, spatialC, ctx->nnYLen, ctx->nnXLen));"
  // So input shapes are correct.

  // Output shapes?
  // We don't have them hardcoded in trtbackend.cpp other than total size.
  // But we can query them from the session in ComputeHandle constructor and store them!

  // For this implementation, I will just let ORT allocate outputs and copy them. It's safer and easier.
  // Performance hit is small compared to inference.

  auto outputValues = handle->session->Run(Ort::RunOptions{nullptr},
    inputNames.data(), inputTensors.data(), inputNames.size(),
    outputNames.data(), outputNames.size());

  // Copy outputs to buffers
  // Assumes output order matches outputNames

  auto copyToBuffer = [&](size_t idx, float* dest, size_t size) {
    const float* src = outputValues[idx].GetTensorData<float>();
    size_t count = outputValues[idx].GetTensorTypeAndShapeInfo().GetElementCount();
    if(count != size) {
      throw StringError(
        "ONNX backend: output " + string(outputNames[idx]) + " element count (" +
        Global::uint64ToString((uint64_t)count) + ") did not match expected (" +
        Global::uint64ToString((uint64_t)size) + ")");
    }
    copy(src, src + size, dest);
  };

  copyToBuffer(0, inputBuffers->out_policyResults.get(), inputBuffers->singleout_policyElts * batchSize);
  copyToBuffer(1, inputBuffers->out_valueResults.get(), inputBuffers->singleout_valueElts * batchSize);
  copyToBuffer(2, inputBuffers->out_miscvalueResults.get(), inputBuffers->singleout_miscvalueElts * batchSize);
  copyToBuffer(3, inputBuffers->out_moremiscvalueResults.get(), inputBuffers->singleout_moremiscvalueElts * batchSize);
  copyToBuffer(4, inputBuffers->out_ownershipResults.get(), inputBuffers->singleout_ownershipElts * batchSize);

  for(int row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];

    // Policy
    const float* policySrcBuf =
      &inputBuffers->out_policyResults[
        row * inputBuffers->singleout_policyElts +
        getSelectedOnnxOutputHeadOffset(inputBuffers->singleout_policyHeadElts, modelVersion)
      ];
    float* policyProbs = outputPolicys + row * NNPos::MAX_NN_POLICY_SIZE;

    // Logic from trtbackend.cpp
    if(modelVersion >= 12 && modelVersion <= 99)
      throw StringError("modelVersion >= 12 && modelVersion <= 99 not supported");

    SymmetryHelpers::copyOutputsWithSymmetry(policySrcBuf, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    policyProbs[nnXLen * nnYLen] = policySrcBuf[nnXLen * nnYLen];

    if(modelVersion == 112) {
      for(int head = 1; head<NNOutput::NUM_POLICY_HEADS; head++) {
        const float* policyByHeadSrcBuf =
          &inputBuffers->out_policyResults[
            row * inputBuffers->singleout_policyElts +
            (size_t)head * inputBuffers->singleout_policyHeadElts
          ];
        float* policyByHead = inputBufs[row]->policyResultsByExtraHead[head-1];
        SymmetryHelpers::copyOutputsWithSymmetry(
          policyByHeadSrcBuf, policyByHead, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
        policyByHead[nnXLen * nnYLen] = policyByHeadSrcBuf[nnXLen * nnYLen];
      }
    }

    // Value
    const float* valueBaseBuf =
      &inputBuffers->out_valueResults[
        row * inputBuffers->singleout_valueElts
      ];
    int valueHeadCount = getOnnxOutputHeadCount(modelVersion);
    for(int head = 0; head<NNOutput::NUM_VALUE_HEADS; head++) {
      int srcHead = head < valueHeadCount ? head : 0;
      const float* valueSrcBuf = valueBaseBuf + (size_t)srcHead * inputBuffers->singleout_valueHeadElts;
      output->whiteWinProbByHead[head] = valueSrcBuf[0];
      output->whiteLossProbByHead[head] = valueSrcBuf[1];
      output->whiteNoResultProbByHead[head] = valueSrcBuf[2];
    }
    output->whiteWinProb = output->whiteWinProbByHead[0];
    output->whiteLossProb = output->whiteLossProbByHead[0];
    output->whiteNoResultProb = output->whiteNoResultProbByHead[0];

    // Misc Value
    const float* miscValueSrcBuf =
      &inputBuffers->out_miscvalueResults[
        row * inputBuffers->singleout_miscvalueElts +
        getSelectedOnnxOutputHeadOffset(inputBuffers->singleout_miscvalueHeadElts, modelVersion)
      ];
    const float* moreMiscValueSrcBuf =
      &inputBuffers->out_moremiscvalueResults[
        row * inputBuffers->singleout_moremiscvalueElts +
        getSelectedOnnxOutputHeadOffset(inputBuffers->singleout_moremiscvalueHeadElts, modelVersion)
      ];

    if(modelVersion >= 9) {
      output->varTimeLeft = miscValueSrcBuf[3];
      output->shorttermWinlossError = moreMiscValueSrcBuf[0];
    }
  }
}

// Implement Test methods with false
bool NeuralNet::testEvaluateConv(const ConvLayerDesc* desc, int batchSize, int nnXLen, int nnYLen, bool useFP16, bool useNHWC, const std::vector<float>& inputBuffer, std::vector<float>& outputBuffer) { return false; }
bool NeuralNet::testEvaluateBatchNorm(const BatchNormLayerDesc* desc, int batchSize, int nnXLen, int nnYLen, bool useFP16, bool useNHWC, const std::vector<float>& inputBuffer, const std::vector<float>& maskBuffer, std::vector<float>& outputBuffer) { return false; }
bool NeuralNet::testEvaluateResidualBlock(const ResidualBlockDesc* desc, int batchSize, int nnXLen, int nnYLen, bool useFP16, bool useNHWC, const std::vector<float>& inputBuffer, const std::vector<float>& maskBuffer, std::vector<float>& outputBuffer) { return false; }
bool NeuralNet::testEvaluateGlobalPoolingResidualBlock(const GlobalPoolingResidualBlockDesc* desc, int batchSize, int nnXLen, int nnYLen, bool useFP16, bool useNHWC, const std::vector<float>& inputBuffer, const std::vector<float>& maskBuffer, std::vector<float>& outputBuffer) { return false; }

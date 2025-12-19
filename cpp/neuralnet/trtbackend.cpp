//modification for tensorrt-onnx: https://github.com/yehu3d/katago_onnx
#ifdef USE_TENSORRT_BACKEND

#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CACHE_TENSORRT_PLAN
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <istream>
#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../core/sha2.h"
#include "../dataio/homedata.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nninterface.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"

using namespace std;
using namespace nvinfer1;

// Define this to print out some of the intermediate values of the neural net
//#define DEBUG_INTERMEDIATE_VALUES

// Define this to use plan cache instead of timing cache, which enables instant
// initialization at the cost of excessive disk space usage
//#define CACHE_TENSORRT_PLAN

const int TensorRT_BuilderOptimizationLevel = 2; //0 for fast init, 2 is default, 5 is max

static void checkCudaError(const cudaError_t status, const char* opName, const char* file, const char* func, int line) {
  if(status != cudaSuccess)
    throw StringError(
      string("CUDA Error, for ") + opName + " file " + file + ", func " + func + ", line " + Global::intToString(line) +
      ", error " + cudaGetErrorString(status));
}
#define CUDA_ERR(opName, x) \
  { checkCudaError((x), opName, __FILE__, #x, __LINE__); }

bool isFileExists_ifstream(string& name) {
  ifstream f(name.c_str());
  return f.good();
}

void NeuralNet::globalInitialize() {
  // Empty for TensorRT backend
}

void NeuralNet::globalCleanup() {
  // Empty for TensorRT backend
}

struct ComputeContext {
  int nnXLen;
  int nnYLen;
  enabled_t useFP16Mode;
  string onnxfile;
  string homeDataDirOverride;
};

ComputeContext* NeuralNet::createComputeContext(
  const vector<int>& gpuIdxs,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const string& openCLTunerFile,
  const string& homeDataDirOverride,
  const string& onnxfile,
  bool openCLReTunePerBoardSize,
  enabled_t useFP16Mode,
  enabled_t useNHWCMode,
  const LoadedModel* loadedModel) {
  (void)gpuIdxs;
  (void)logger;
  (void)openCLTunerFile;
  (void)openCLReTunePerBoardSize;
  (void)loadedModel;

  if(useNHWCMode == enabled_t::True) {
    throw StringError("TensorRT backend: useNHWC = false required, other configurations not supported");
  }

  ComputeContext* context = new ComputeContext();
  context->nnXLen = nnXLen;
  context->nnYLen = nnYLen;
  context->useFP16Mode = useFP16Mode;
  context->onnxfile = onnxfile;
  context->homeDataDirOverride = homeDataDirOverride;
  return context;
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

struct LoadedModel {
  ModelDesc modelDesc;

  LoadedModel(const string& fileName, const string& expectedSha256) {
    ModelDesc::loadFromFileMaybeGZipped(fileName, modelDesc, expectedSha256);
  }

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

LoadedModel* NeuralNet::loadModelFile(const string& file, const string& expectedSha256) {
  LoadedModel* loadedModel = new LoadedModel(file, expectedSha256);
  return loadedModel;
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}

string NeuralNet::getModelName(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc.name;
}

int NeuralNet::getModelVersion(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc.modelVersion;
}

int NeuralNet::getNumInputMetaChannels(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc.numInputMetaChannels;
}

Rules NeuralNet::getSupportedRules(const LoadedModel* loadedModel, const Rules& desiredRules, bool& supported) {
  return loadedModel->modelDesc.getSupportedRules(desiredRules, supported);
}

ModelPostProcessParams NeuralNet::getPostProcessParams(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc.postProcessParams;
}

struct TRTModel {
  int nnXLen;
  int nnYLen;
  int maxBatchSize;
  bool requireExactNNLen;

  // TensorRT keeps only reference to weights before engine is built
  const LoadedModel* rawModel;
  vector<unique_ptr<float[]>> extraWeights;

  int modelVersion;
  uint8_t tuneHash[32];
  IOptimizationProfile* profile;
  unique_ptr<INetworkDefinition> network;
  vector<pair<string, string>> debugOutputs;

  TRTModel() = default;
  TRTModel(TRTModel&&) = default;
  TRTModel(const TRTModel&) = delete;
  TRTModel& operator=(TRTModel&&) = default;
  TRTModel& operator=(const TRTModel&) = delete;
};

struct TRTLogger : ILogger {
  Logger* logger;
  Severity level;

  TRTLogger() {
    logger = nullptr;
    level = Severity::kERROR;
  }

  TRTLogger(const TRTLogger&) = delete;
  TRTLogger& operator=(const TRTLogger&) = delete;

  void log(Severity severity, const char* msg) noexcept override {
    if(logger && severity <= level)
      logger->write("TensorRT backend: " + string(msg));
    if(severity == Severity::kERROR && logger && !logger->isLoggingToStderr() && !logger->isLoggingToStdout()) {
      std::cerr << ("TensorRT backend: " + string(msg)) << std::endl;
    }
  }

  void setLogger(Logger* externalLogger) { logger = externalLogger; }
};

struct ComputeHandle {
  ComputeContext* ctx;

  int infer_times = 0;
  bool usingFP16;
  int maxBatchSize;
  int modelVersion;
  vector<pair<string, string>> debugOutputs;

  TRTLogger trtLogger;
  map<string, void*> buffers;
  unique_ptr<IRuntime> runtime;
  unique_ptr<ICudaEngine> engine;
  unique_ptr<IExecutionContext> exec;



  ComputeHandle(
    Logger* logger,
    const cudaDeviceProp* prop,
    ComputeContext* context,
    const LoadedModel* loadedModel,
    int maxBatchSz,
    bool requireExactNNLen) {
    ctx = context;

    maxBatchSize = maxBatchSz;
    modelVersion = loadedModel->modelDesc.modelVersion;

    // Certain minor versions of TensorRT uses a global logger, which is bad.
    // Since TensorRT maintains ABI compatibility between minor versions, a dynamic library mismatch
    // does not necessarily generate a dynamic link error, therefore, an extra check is required.
    if(getInferLibVersion() / 100 != NV_TENSORRT_VERSION / 100) {
      throw StringError("TensorRT backend: detected incompatible version of TensorRT library");
    }

    trtLogger.setLogger(logger);

    auto builder = unique_ptr<IBuilder>(createInferBuilder(trtLogger));
    if(!builder) {
      throw StringError("TensorRT backend: failed to create builder");
    }
    auto config = unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    if(!config) {
      throw StringError("TensorRT backend: failed to create builder config");
    }

    auto network = unique_ptr<INetworkDefinition>(
      builder->createNetworkV2(1U << static_cast<int>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
    if(!network) {
      throw StringError("TensorRT backend: failed to create network definition");
    }
    auto profile = builder->createOptimizationProfile();
    if(!profile) {
      throw StringError("TensorRT backend: failed to create optimization profile");
    }
    auto parser = nvonnxparser::createParser(*network, trtLogger);
    if(!parser) {
      throw StringError("TensorRT backend: failed to create ONNX parser");
    }
    
    string ONNX_sha256 = "";

    if(!parser->parseFromFile(ctx->onnxfile.c_str(), static_cast<int>(ILogger::Severity::kERROR))) {
      throw StringError("TensorRT backend: failed to parse ONNX model");
    }
    {
      string tmp;
      FileUtils::loadFileIntoString(ctx->onnxfile, "", tmp, &ONNX_sha256);
    }

    int64_t spatialC = NNModelVersion::getNumSpatialFeatures(modelVersion);
    int64_t globalC = NNModelVersion::getNumGlobalFeatures(modelVersion);
    profile->setDimensions("input_spatial", OptProfileSelector::kMIN, Dims4(1, spatialC, ctx->nnYLen, ctx->nnXLen));
    profile->setDimensions("input_spatial", OptProfileSelector::kOPT, Dims4(maxBatchSize, spatialC, ctx->nnYLen, ctx->nnXLen));
    profile->setDimensions("input_spatial", OptProfileSelector::kMAX, Dims4(maxBatchSize, spatialC, ctx->nnYLen, ctx->nnXLen));
    profile->setDimensions("input_global", OptProfileSelector::kMIN, Dims2(1, globalC));
    profile->setDimensions("input_global", OptProfileSelector::kOPT, Dims2(maxBatchSize, globalC));
    profile->setDimensions("input_global", OptProfileSelector::kMAX, Dims2(maxBatchSize, globalC));

    if(builder->platformHasFastFp16()) {
        if(ctx->useFP16Mode == enabled_t::True || ctx->useFP16Mode == enabled_t::Auto) {
          config->setFlag(BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
          config->setFlag(BuilderFlag::kFP16);
          usingFP16 = true;
        }
      } else if(ctx->useFP16Mode == enabled_t::True) {
        throw StringError("TensorRT backend: CUDA device does not support useFP16=true");
      }

    std::cout << "Number of inputs: " << network->getNbInputs() << std::endl;
    // for(int i = 0; i < network->getNbInputs(); ++i) {
    //  auto input = network->getInput(i);
    //  std::cout << "Input " << i << " name: " << input->getName() << std::endl;
    //}

    // //Print output tensor names
    // std::cout << "Number of outputs: " << network->getNbOutputs() << std::endl;
    // for(int i = 0; i < network->getNbOutputs(); ++i) {
    //  auto output = network->getOutput(i);
    //  std::cout << "Output " << i << " name: " << output->getName() << std::endl;
    //}
    config->addOptimizationProfile(profile);

    if(prop->major >= 8) {
      // This is to avoid tactics that have shape switching overhead
      config->setTacticSources(1U << static_cast<uint32_t>(TacticSource::kJIT_CONVOLUTIONS));
      //uint32_t sources = (1U << static_cast<uint32_t>(TacticSource::kCUBLAS)) |
      //                   (1U << static_cast<uint32_t>(TacticSource::kCUBLAS_LT)) |
      //                   (1U << static_cast<uint32_t>(TacticSource::kCUDNN)) |
      //                   (1U << static_cast<uint32_t>(TacticSource::kJIT_CONVOLUTIONS));
      //config->setTacticSources(sources);
      config->setBuilderOptimizationLevel(TensorRT_BuilderOptimizationLevel);
    }

    // So that there are no concurrent kernel executions probably from other parts of code while profiling
    // See CUDA Runtime API document for more details related to NULL stream and synchronization behaviors
    config->setProfileStream(cudaStreamLegacy);

    // Typical runtime allocation is much less than the 1 GiB specified below
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, static_cast<size_t>(1U) << 32);

    string plan;
    {
      static mutex tuneMutex;
      tuneMutex.lock();

      auto cacheDir = HomeData::getHomeDataDir(true, ctx->homeDataDirOverride);
      cacheDir += "/trtcache";
      MakeDir::make(cacheDir);

      uint8_t deviceHash[32];
      SHA2::get256(prop->name, deviceHash);

      // Truncated to 4 bytes
      char deviceIdent[4 * 2 + 1];
      for(int i = 0; i < 4; i++) {
        sprintf(deviceIdent + i * 2, "%02x", static_cast<unsigned char>(deviceHash[i]));
      }
      deviceIdent[sizeof(deviceIdent) - 1] = 0;

#ifdef CACHE_TENSORRT_PLAN

      string ONNX_sha256_short = ONNX_sha256.substr(0, 16);
      auto planCacheFile = Global::strprintf(
        "%s/trt-%d_gpu-%s_olv-%d_onnxnet-%s_%s%dx%d_batch%d_fp%d",
        cacheDir.c_str(),
        getInferLibVersion(),
        deviceIdent,
        TensorRT_BuilderOptimizationLevel,
        ONNX_sha256_short.c_str(),
        requireExactNNLen ? "exact" : "max",
        ctx->nnYLen,
        ctx->nnXLen,
        maxBatchSize,
        usingFP16 ? 16 : 32);
      string paramStr = Global::strprintf(
        "_%d_%s_%s_%d_%d_%d_%d",
        getInferLibVersion(),
        deviceIdent,
        requireExactNNLen ? "exact" : "max",
        ctx->nnYLen,
        ctx->nnXLen,
        maxBatchSize,
        usingFP16 ? 16 : 32);
      std::cout << "plan file:" << planCacheFile << std::endl;
      try {
        plan = FileUtils::readFileBinary(planCacheFile);
      } catch(const StringError& e) {
        (void)e;
      };

      if(plan.size() > 0) {
        if(plan.size() < 64 + paramStr.size()) {
          logger->write("Could not parse plan, unexpected size in " + planCacheFile);
          plan.clear();
        } else {
          string cachedParamStr = plan.substr(plan.size() - paramStr.size());
          string modelHash = plan.substr(plan.size() - 64 - paramStr.size(), 64);
          if(modelHash != loadedModel->modelDesc.sha256) {
            logger->write("Plan cache is corrupted or is for the wrong model in " + planCacheFile);
            plan.clear();
          } else if(cachedParamStr != paramStr) {
            logger->write("Plan cache is corrupted or is for the wrong parameters in " + planCacheFile);
            plan.clear();
          } else {
            plan.erase(plan.size() - 64 - paramStr.size());
          }
        }
      }

      if(plan.size() <= 0) {
        logger->write("Creating new plan cache");
        auto planBuffer = unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network, *config));
        if(!planBuffer) {
          throw StringError("TensorRT backend: failed to create plan");
        }
        plan.insert(
          plan.end(),
          static_cast<char*>(planBuffer->data()),
          static_cast<char*>(planBuffer->data()) + planBuffer->size());
        if(loadedModel->modelDesc.sha256.size() != 64) {
          throw StringError("Unexpected model hash size");
        }
        if(true)
        {
          ofstream ofs0;
          FileUtils::open(ofs0, planCacheFile + ".pure", ios::out | ios::binary);
          ofs0.write(plan.data(), plan.size());
          ofs0.close();
          logger->write("Saved new pure plan cache to " + planCacheFile + ".pure");
         
        }
        plan.insert(plan.end(), loadedModel->modelDesc.sha256.begin(), loadedModel->modelDesc.sha256.end());
        plan.insert(plan.end(), paramStr.begin(), paramStr.end());
        ofstream ofs;
        FileUtils::open(ofs, planCacheFile, ios::out | ios::binary);
        ofs.write(plan.data(), plan.size());
        ofs.close();
        logger->write("Saved new plan cache to " + planCacheFile);
        plan.erase(plan.size() - 64 - paramStr.size());
        tuneMutex.unlock();
      } else {
        tuneMutex.unlock();
        logger->write("Using existing plan cache at " + planCacheFile);
      }
#else
      // Truncated to 6 bytes
      char tuneIdent[6 * 2 + 1];
      for(int i = 0; i < 6; i++) {
        sprintf(tuneIdent + i * 2, "%02x", static_cast<unsigned char>(model->tuneHash[i]));
      }
      tuneIdent[sizeof(tuneIdent) - 1] = 0;

      auto timingCacheFile = Global::strprintf(
        "%s/trt-%d_gpu-%s_tune-%s_%s%dx%d_batch%d_fp%d",
        cacheDir.c_str(),
        getInferLibVersion(),
        deviceIdent,
        tuneIdent,
        requireExactNNLen ? "exact" : "max",
        ctx->nnYLen,
        ctx->nnXLen,
        maxBatchSize,
        usingFP16 ? 16 : 32);

      string timingCacheBlob;
      try {
        timingCacheBlob = FileUtils::readFileBinary(timingCacheFile);
      } catch(const StringError& e) {
        (void)e;
      };
      if(timingCacheBlob.size() > 0)
        logger->write("Using existing timing cache at " + timingCacheFile);
      else
        logger->write("Creating new timing cache");

      auto timingCache =
        unique_ptr<ITimingCache>(config->createTimingCache(timingCacheBlob.data(), timingCacheBlob.size()));
      auto invalidTimingCache = !config->setTimingCache(*timingCache, false);
      if(invalidTimingCache) {
        logger->write("Invalid timing cache, using new one instead");
        timingCache.reset(config->createTimingCache(nullptr, 0));
        config->setTimingCache(*timingCache, false);
      }

      unique_ptr<IHostMemory> planBuffer;
      if(invalidTimingCache || !timingCacheBlob.size()) {
        planBuffer.reset(builder->buildSerializedNetwork(*model->network, *config));
        if(!planBuffer) {
          throw StringError("TensorRT backend: failed to create plan");
        }
        auto serializedTimingCache = unique_ptr<IHostMemory>(config->getTimingCache()->serialize());
        ofstream ofs;
        FileUtils::open(ofs, timingCacheFile, ios::out | ios::binary);
        ofs.write(static_cast<char*>(serializedTimingCache->data()), serializedTimingCache->size());
        ofs.close();
        logger->write("Saved new timing cache to " + timingCacheFile);
        tuneMutex.unlock();
      } else {
        tuneMutex.unlock();
        planBuffer.reset(builder->buildSerializedNetwork(*model->network, *config));
        if(!planBuffer) {
          throw StringError("TensorRT backend: failed to create plan");
        }
      }
      plan.insert(
        plan.end(),
        static_cast<char*>(planBuffer->data()),
        static_cast<char*>(planBuffer->data()) + planBuffer->size());
#endif
    }

    runtime.reset(createInferRuntime(trtLogger));
    if(!runtime) {
      throw StringError("TensorRT backend: failed to create runtime");
    }
    std::cout << "Infer Runtime created successfully." << std::endl;

    std::cout << "Deserializing CUDA Engine..." << std::endl;
    engine.reset(runtime->deserializeCudaEngine(plan.data(), plan.size()));
    if(!engine) {
      throw StringError("TensorRT backend: failed to create cuda engine");
    }
    std::cout << "CUDA Engine deserialized successfully." << std::endl;

    std::cout << "Creating Execution Context..." << std::endl;
    exec.reset(engine->createExecutionContext());
    if(!exec) {
      throw StringError("TensorRT backend: failed to create execution context");
    }
    std::cout << "Execution Context created successfully." << std::endl;
    for(int i = 0; i < engine->getNbIOTensors(); i++) {
      void* buffer = nullptr;
      auto name = engine->getIOTensorName(i);

      auto dims = engine->getTensorShape(name);
      size_t bytes = accumulate(dims.d + 1, dims.d + dims.nbDims, maxBatchSize * sizeof(float), multiplies<size_t>());
      CUDA_ERR("ComputeHandle", cudaMalloc(&buffer, bytes));
      buffers.emplace(make_pair(name, buffer));
      exec->setTensorAddress(name, buffer);
    }
    int nbIOTensors = engine->getNbIOTensors();
    for(int i = 0; i < nbIOTensors; ++i) {
      const char* tensorName = engine->getIOTensorName(i);
      nvinfer1::Dims dims = engine->getTensorShape(tensorName);
      nvinfer1::DataType dtype = engine->getTensorDataType(tensorName);
      bool isInput = engine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT;

      std::cout << "Tensor " << i << " (" << (isInput ? "Input" : "Output") << "): " << tensorName << std::endl;
      std::cout << "Dims: ";
      for(int d = 0; d < dims.nbDims; ++d) {
        std::cout << dims.d[d] << " ";
      }
      std::cout << ", DataType: " << static_cast<int>(dtype) << std::endl;
    }

    exec->setOptimizationProfileAsync(0, cudaStreamPerThread);
    cudaStreamSynchronize(cudaStreamPerThread);
  }

  ~ComputeHandle() {
    for(auto ptr: buffers) {
      CUDA_ERR("~ComputeHandle", cudaFree(ptr.second));
    }
  }

  ComputeHandle() = delete;
  ComputeHandle(const ComputeHandle&) = delete;
  ComputeHandle& operator=(const ComputeHandle&) = delete;

  void* getBuffer(const char* name) {
    auto search = buffers.find(name);
    if(search != buffers.end()) {
      return search->second;
    } else {
      throw StringError(Global::strprintf("ComputeHandle: unknown tensor name %s", name));
    }
  }

  size_t getBufferBytes(const char* name) {
    auto dims = engine->getTensorShape(name);
    if(dims.nbDims != -1) {
      return accumulate(dims.d + 1, dims.d + dims.nbDims, maxBatchSize * sizeof(float), multiplies<size_t>());
    } else {
      throw StringError(Global::strprintf("ComputeHandle: unknown tensor name %s", name));
    }
  }

  size_t getBufferRowElts(const char* name) {
    auto dims = engine->getTensorShape(name);
    if(dims.nbDims != -1) {
      return accumulate(dims.d + 1, dims.d + dims.nbDims, 1, multiplies<size_t>());
    } else {
      throw StringError(Global::strprintf("ComputeHandle: unknown tensor name %s", name));
    }
  }

  Dims getBufferDynamicShape(const char* name, int batchSize) {
    auto dims = engine->getTensorShape(name);
    if(dims.nbDims != -1) {
      dims.d[0] = batchSize;
      return dims;
    } else {
      throw StringError(Global::strprintf("ComputeHandle: unknown tensor name %s", name));
    }
  }

  void printDebugOutput(int batchSize) {
    for(auto& debugOutput: debugOutputs) {
      auto name = debugOutput.first;
      auto desc = debugOutput.second;
      auto dims = getBufferDynamicShape(name.c_str(), batchSize);

      vector<float> values(accumulate(dims.d, dims.d + dims.nbDims, 1, multiplies<size_t>()));
      CUDA_ERR(
        "printDebugOutput",
        cudaMemcpy(values.data(), getBuffer(name.c_str()), values.size() * sizeof(float), cudaMemcpyDeviceToHost));

      cout << "=========================================================" << endl;
      cout << desc << endl;
      int i = 0;
      if(dims.nbDims == 2) {
        for(int n = 0; n < dims.d[0]; n++) {
          cout << "-(n=" << n << ")--------------------" << endl;
          for(int c = 0; c < dims.d[1]; c++) {
            cout << values[i++] << " ";
          }
          cout << endl;
        }
        cout << endl;
      } else if(dims.nbDims == 4) {
        for(int n = 0; n < dims.d[0]; n++) {
          cout << "-(n=" << n << ")--------------------" << endl;
          for(int c = 0; c < dims.d[1]; c++) {
            cout << "(c=" << c << ")" << endl;
            for(int y = 0; y < dims.d[2]; y++) {
              for(int x = 0; x < dims.d[3]; x++)
                cout << values[i++] << " ";
              cout << endl;
            }
            cout << endl;
          }
        }
      }
      cout << "=========================================================" << endl;
    }
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
  int serverThreadIdx) {
  if(inputsUseNHWC) {
    throw StringError("TensorRT backend: inputsUseNHWC = false required, other configurations not supported");
  }

  // Use whatever CUDA believes GPU 0 to be.
  if(gpuIdxForThisThread == -1)
    gpuIdxForThisThread = 0;
  CUDA_ERR("createComputeHandle", cudaSetDevice(gpuIdxForThisThread));

  cudaDeviceProp prop;
  CUDA_ERR("createComputeHandle", cudaGetDeviceProperties(&prop, gpuIdxForThisThread));

  if(logger != NULL) {
    logger->write(
      "TensorRT backend thread " + Global::intToString(serverThreadIdx) + ": Found GPU " + string(prop.name) +
      " memory " + Global::uint64ToString(prop.totalGlobalMem) + " compute capability major " +
      Global::intToString(prop.major) + " minor " + Global::intToString(prop.minor));
    logger->write(
      "TensorRT backend thread " + Global::intToString(serverThreadIdx) + ": Initializing (may take a long time)");
  }

  auto handle = new ComputeHandle(logger, &prop, context, loadedModel, maxBatchSize, requireExactNNLen);

  if(logger != NULL) {
    logger->write(
      "TensorRT backend thread " + Global::intToString(serverThreadIdx) + ": Model version " +
      Global::intToString(loadedModel->modelDesc.modelVersion) +
      " useFP16 = " + Global::boolToString(handle->usingFP16));
    logger->write(
      "TensorRT backend thread " + Global::intToString(serverThreadIdx) +
      ": Model name: " + loadedModel->modelDesc.name);
  }

  return handle;
}

void NeuralNet::freeComputeHandle(ComputeHandle* gpuHandle) {
  delete gpuHandle;
}

bool NeuralNet::isUsingFP16(const ComputeHandle* gpuHandle) {
  return gpuHandle->usingFP16;
}

void NeuralNet::printDevices() {
  int numDevices = 0;
  CUDA_ERR("printDevices", cudaGetDeviceCount(&numDevices));
  for(int i = 0; i < numDevices; i++) {
    cudaDeviceProp prop;
    CUDA_ERR("printDevices", cudaGetDeviceProperties(&prop, i));
    std::cout << "Found GPU device " << i << ": " << prop.name << endl;
  }
}

struct InputBuffers {
  int maxBatchSize;

  // size_t singleMaskElts;
  // size_t singleMaskBytes;
  size_t singleFeatureElts;
  size_t singleFeatureBytes;
  size_t singleGlobalFeatureElts;
  size_t singleGlobalFeatureBytes;
  // size_t singlePolicyPassResultElts;
  // size_t singlePolicyPassResultBytes;
  // size_t singlePolicyResultElts;
  // size_t singlePolicyResultBytes;
  // size_t singleValueResultElts;
  // size_t singleValueResultBytes;
  // size_t singleScoreValueResultElts;
  // size_t singleScoreValueResultBytes;
  // size_t singleOwnershipResultElts;
  // size_t singleOwnershipResultBytes;

  size_t singleout_policyElts;
  size_t singleout_policyBytes;
  size_t singleout_valueElts;
  size_t singleout_valueBytes;
  size_t singleout_miscvalueElts;
  size_t singleout_miscvalueBytes;
  size_t singleout_moremiscvalueElts;
  size_t singleout_moremiscvalueBytes;
  size_t singleout_ownershipElts;
  size_t singleout_ownershipBytes;
  // size_t singleout_scoringElts;
  // size_t singleout_scoringBytes;
  // size_t singleout_futureposElts;
  // size_t singleout_futureposBytes;
  // size_t singleout_sekiElts;
  // size_t singleout_sekiBytes;
  // size_t singleout_scorebelief_logprobsElts;
  // size_t singleout_scorebelief_logprobsBytes;
  // size_t singleiout_policyElts;
  // size_t singleiout_policyBytes;
  // size_t singleiout_valueElts;
  // size_t singleiout_valueBytes;
  // size_t singleiout_miscvalueElts;
  // size_t singleiout_miscvalueBytes;
  // size_t singleiout_moremiscvalueElts;
  // size_t singleiout_moremiscvalueBytes;
  // size_t singleiout_ownershipElts;
  // size_t singleiout_ownershipBytes;
  // size_t singleiout_scoringElts;
  // size_t singleiout_scoringBytes;
  // size_t singleiout_futureposElts;
  // size_t singleiout_futureposBytes;
  // size_t singleiout_sekiElts;
  // size_t singleiout_sekiBytes;
  // size_t singleiout_scorebelief_logprobsElts;
  // size_t singleiout_scorebelief_logprobsBytes;

  // size_t maskInputBufferBytes;
  size_t featureInputBufferBytes;

  size_t globalFeatureInputBufferBytes;
  // size_t policyPassResultBufferBytes;
  // size_t policyResultBufferBytes;
  // size_t valueResultBufferBytes;
  // size_t scoreValueResultBufferBytes;
  // size_t ownershipResultBufferBytes;

  size_t out_policyBufferBytes;
  size_t out_valueBufferBytes;
  size_t out_miscvalueBufferBytes;
  size_t out_moremiscvalueBufferBytes;
  size_t out_ownershipBufferBytes;
  // size_t out_scoringBufferBytes;
  // size_t out_futureposBufferBytes;
  // size_t out_sekiBufferBytes;
  // size_t out_scorebelief_logprobsBufferBytes;
  // size_t iout_policyBufferBytes;
  // size_t iout_valueBufferBytes;
  // size_t iout_miscvalueBufferBytes;
  // size_t iout_moremiscvalueBufferBytes;
  // size_t iout_ownershipBufferBytes;
  // size_t iout_scoringBufferBytes;
  // size_t iout_futureposBufferBytes;
  // size_t iout_sekiBufferBytes;
  // size_t iout_scorebelief_logprobsBufferBytes;

  // unique_ptr<float[]> maskInputs;           // Host pointer
  unique_ptr<float[]> featureInputs;        // Host pointer
  unique_ptr<float[]> globalFeatureInputs;  // Host pointer

  // unique_ptr<float[]> policyPassResults;    // Host pointer
  // unique_ptr<float[]> policyResults;        // Host pointer
  // unique_ptr<float[]> valueResults;         // Host pointer
  // unique_ptr<float[]> scoreValueResults;    // Host pointer
  // unique_ptr<float[]> ownershipResults;     // Host pointer

  unique_ptr<float[]> out_policyResults;
  unique_ptr<float[]> out_valueResults;
  unique_ptr<float[]> out_miscvalueResults;
  unique_ptr<float[]> out_moremiscvalueResults;
  unique_ptr<float[]> out_ownershipResults;
  // unique_ptr<float[]> out_scoringResults;
  // unique_ptr<float[]> out_futureposResults;
  // unique_ptr<float[]> out_sekiResults;
  // unique_ptr<float[]> out_scorebelief_logprobsResults;
  // unique_ptr<float[]> iout_policyResults;
  // unique_ptr<float[]> iout_valueResults;
  // unique_ptr<float[]> iout_miscvalueResults;
  // unique_ptr<float[]> iout_moremiscvalueResults;
  // unique_ptr<float[]> iout_ownershipResults;
  // unique_ptr<float[]> iout_scoringResults;
  // unique_ptr<float[]> iout_futureposResults;
  // unique_ptr<float[]> iout_sekiResults;
  // unique_ptr<float[]> iout_scorebelief_logprobsResults;

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
    const ModelDesc& m = loadedModel->modelDesc;

    if(nnXLen > NNPos::MAX_BOARD_LEN)
      throw StringError(
        Global::strprintf("nnXLen (%d) is greater than NNPos::MAX_BOARD_LEN (%d)", nnXLen, NNPos::MAX_BOARD_LEN));
    if(nnYLen > NNPos::MAX_BOARD_LEN)
      throw StringError(
        Global::strprintf("nnYLen (%d) is greater than NNPos::MAX_BOARD_LEN (%d)", nnYLen, NNPos::MAX_BOARD_LEN));

    maxBatchSize = maxBatchSz;
    // singleMaskElts = nnXLen * nnYLen;
    // singleMaskBytes = singleMaskElts * sizeof(float);
    singleFeatureElts = m.numInputChannels * nnXLen * nnYLen;
    singleFeatureBytes = singleFeatureElts * sizeof(float);
    singleGlobalFeatureElts = m.numInputGlobalChannels;
    singleGlobalFeatureBytes = singleGlobalFeatureElts * sizeof(float);
    // singlePolicyPassResultElts = (size_t)m.numPolicyChannels;
    // singlePolicyPassResultBytes = singlePolicyPassResultElts * sizeof(float);
    // singlePolicyResultElts = (size_t)m.numPolicyChannels * nnXLen * nnYLen;
    // singlePolicyResultBytes = singlePolicyResultElts * sizeof(float);
    // singleValueResultElts = m.numValueChannels;
    // singleValueResultBytes = singleValueResultElts * sizeof(float);
    // singleScoreValueResultElts = m.numScoreValueChannels;
    // singleScoreValueResultBytes = singleScoreValueResultElts * sizeof(float);
    // singleOwnershipResultElts = m.numOwnershipChannels * nnXLen * nnYLen;
    // singleOwnershipResultBytes = singleOwnershipResultElts * sizeof(float);
    int policyNum = (loadedModel->modelDesc.modelVersion >= 12 && loadedModel->modelDesc.modelVersion <= 99) ? 6 : 4;
    singleout_policyElts = 1 * policyNum * (nnXLen * nnYLen + 1);  // 1 6 362
    singleout_policyBytes = singleout_policyElts * sizeof(float);
    singleout_valueElts = 3;
    singleout_valueBytes = singleout_valueElts * sizeof(float);
    singleout_miscvalueElts = 10;
    singleout_miscvalueBytes = singleout_miscvalueElts * sizeof(float);
    singleout_moremiscvalueElts = 8;
    singleout_moremiscvalueBytes = singleout_moremiscvalueElts * sizeof(float);
    singleout_ownershipElts = 1 * nnXLen * nnYLen;
    singleout_ownershipBytes = singleout_ownershipElts * sizeof(float);
    // singleout_scoringelts = 1 * nnxlen * nnylen;
    // singleout_scoringbytes = singleout_scoringelts * sizeof(float);
    // singleout_futureposelts = 2 * nnxlen * nnylen;
    // singleout_futureposbytes = singleout_futureposelts * sizeof(float);
    // singleout_sekielts = 4 * nnxlen * nnylen;
    // singleout_sekibytes = singleout_sekielts * sizeof(float);
    // singleout_scorebelief_logprobselts = 2 * (nnxlen * nnylen + 1);
    // singleout_scorebelief_logprobsbytes = singleout_scorebelief_logprobselts * sizeof(float);
    // singleiout_policyElts = 6 * nnXLen * nnYLen;
    // singleiout_policyBytes = singleiout_policyElts * sizeof(float);
    // singleiout_valueElts = 3;
    // singleiout_valueBytes = singleiout_valueElts * sizeof(float);
    // singleiout_miscvalueElts = 10;
    // singleiout_miscvalueBytes = singleiout_miscvalueElts * sizeof(float);
    // singleiout_moremiscvalueElts = 8;
    // singleiout_moremiscvalueBytes = singleiout_moremiscvalueElts * sizeof(float);
    // singleiout_ownershipElts = 1 * nnXLen * nnYLen;
    // singleiout_ownershipBytes = singleiout_ownershipElts * sizeof(float);
    // singleiout_scoringElts = 1 * nnXLen * nnYLen;
    // singleiout_scoringBytes = singleiout_scoringElts * sizeof(float);
    // singleiout_futureposElts = 2 * nnXLen * nnYLen;
    // singleiout_futureposBytes = singleiout_futureposElts * sizeof(float);
    // singleiout_sekiElts = 4 * nnXLen * nnYLen;
    // singleiout_sekiBytes = singleiout_sekiElts * sizeof(float);
    // singleiout_scorebelief_logprobsElts = 2 * (nnXLen * nnYLen + 1);
    // singleiout_scorebelief_logprobsBytes = singleiout_scorebelief_logprobsElts * sizeof(float);

    assert(NNModelVersion::getNumSpatialFeatures(m.modelVersion) == m.numInputChannels);
    assert(NNModelVersion::getNumGlobalFeatures(m.modelVersion) == m.numInputGlobalChannels);

    // maskInputBufferBytes = maxBatchSize * singleMaskBytes;
    featureInputBufferBytes = maxBatchSize * singleFeatureBytes;
    globalFeatureInputBufferBytes = maxBatchSize * singleGlobalFeatureBytes;
    // policyPassResultBufferBytes = maxBatchSize * singlePolicyPassResultBytes;
    // policyResultBufferBytes = maxBatchSize * singlePolicyResultBytes;
    // valueResultBufferBytes = maxBatchSize * singleValueResultBytes;
    // scoreValueResultBufferBytes = maxBatchSize * singleScoreValueResultBytes;
    // ownershipResultBufferBytes = maxBatchSize * singleOwnershipResultBytes;

    // maskInputs = make_unique<float[]>(maxBatchSize * singleMaskElts);
    featureInputs = make_unique<float[]>(maxBatchSize * singleFeatureElts);
    globalFeatureInputs = make_unique<float[]>(maxBatchSize * singleGlobalFeatureElts);

    // policyPassResults = make_unique<float[]>(maxBatchSize * singlePolicyPassResultElts);
    // policyResults = make_unique<float[]>(maxBatchSize * singlePolicyResultElts);
    // valueResults = make_unique<float[]>(maxBatchSize * singleValueResultElts);
    // scoreValueResults = make_unique<float[]>(maxBatchSize * singleScoreValueResultElts);
    // ownershipResults = make_unique<float[]>(maxBatchSize * singleOwnershipResultElts);

    out_policyResults = std::make_unique<float[]>(maxBatchSize * singleout_policyElts);
    out_valueResults = std::make_unique<float[]>(maxBatchSize * singleout_policyElts);
    out_miscvalueResults = std::make_unique<float[]>(maxBatchSize * singleout_miscvalueElts);
    out_moremiscvalueResults = std::make_unique<float[]>(maxBatchSize * singleout_moremiscvalueElts);
    out_ownershipResults = std::make_unique<float[]>(maxBatchSize * singleout_ownershipElts);
    // out_scoringResults = std::make_unique<float[]>(maxBatchSize * singleout_scoringElts);
    // out_futureposResults = std::make_unique<float[]>(maxBatchSize * singleout_futureposElts);
    // out_sekiResults = std::make_unique<float[]>(maxBatchSize * singleout_sekiElts);
    // out_scorebelief_logprobsResults = std::make_unique<float[]>(maxBatchSize * singleout_scorebelief_logprobsElts);
    // iout_policyResults = std::make_unique<float[]>(maxBatchSize * singleiout_policyElts);
    // iout_valueResults = std::make_unique<float[]>(maxBatchSize * singleiout_policyElts);
    // iout_miscvalueResults = std::make_unique<float[]>(maxBatchSize * singleiout_miscvalueElts);
    // iout_moremiscvalueResults = std::make_unique<float[]>(maxBatchSize * singleiout_moremiscvalueElts);
    // iout_ownershipResults = std::make_unique<float[]>(maxBatchSize * singleiout_ownershipElts);
    // iout_scoringResults = std::make_unique<float[]>(maxBatchSize * singleiout_scoringElts);
    // iout_futureposResults = std::make_unique<float[]>(maxBatchSize * singleiout_futureposElts);
    // iout_sekiResults = std::make_unique<float[]>(maxBatchSize * singleiout_sekiElts);
    // iout_scorebelief_logprobsResults = std::make_unique<float[]>(maxBatchSize * singleiout_scorebelief_logprobsElts);
  }

  InputBuffers() = delete;
  InputBuffers(const InputBuffers&) = delete;
  InputBuffers& operator=(const InputBuffers&) = delete;
};

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  return new InputBuffers(loadedModel, maxBatchSize, nnXLen, nnYLen);
}

void NeuralNet::freeInputBuffers(InputBuffers* inputBuffers) {
  delete inputBuffers;
}

void NeuralNet::getOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs) {
  assert(numBatchEltsFilled <= inputBuffers->maxBatchSize);
  assert(numBatchEltsFilled > 0);

  int batchSize = numBatchEltsFilled;
  int nnXLen = gpuHandle->ctx->nnXLen;
  int nnYLen = gpuHandle->ctx->nnYLen;
  int modelVersion = gpuHandle->modelVersion;

  int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
  int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);

  for(int nIdx = 0; nIdx < batchSize; nIdx++) {
    // float* rowMaskInput = &inputBuffers->maskInputs[inputBuffers->singleMaskElts * nIdx];
    float* rowFeatureInput = &inputBuffers->featureInputs[inputBuffers->singleFeatureElts * nIdx];
    float* rowGlobalFeatureInput = &inputBuffers->globalFeatureInputs[inputBuffers->singleGlobalFeatureElts * nIdx];

    const float* rowFeature = inputBufs[nIdx]->rowSpatialBuf.data();
    const float* rowGlobalFeature = inputBufs[nIdx]->rowGlobalBuf.data();
    SymmetryHelpers::copyInputsWithSymmetry(
      rowFeature, rowFeatureInput, 1, nnYLen, nnXLen, numSpatialFeatures, false, inputBufs[nIdx]->symmetry);
    copy(rowGlobalFeature, rowGlobalFeature + numGlobalFeatures, rowGlobalFeatureInput);
    // copy(rowFeatureInput, rowFeatureInput + inputBuffers->singleMaskElts, rowMaskInput);
  }

  // assert(inputBuffers->singleMaskElts == gpuHandle->getBufferRowElts("InputMask"));
  assert(inputBuffers->singleFeatureElts == gpuHandle->getBufferRowElts("input_spatial"));       // InputFeature
  assert(inputBuffers->singleGlobalFeatureElts == gpuHandle->getBufferRowElts("input_global"));  // InputGlobalFeature

  // assert(inputBuffers->singlePolicyPassResultElts ==
  // gpuHandle->getBufferRowElts("OutputPolicyPass"));//OutputPolicyPass assert(inputBuffers->singlePolicyResultElts ==
  // gpuHandle->getBufferRowElts("OutputPolicy"));//OutputPolicy assert(inputBuffers->singleValueResultElts ==
  // gpuHandle->getBufferRowElts("OutputValue"));//OutputValue assert(inputBuffers->singleScoreValueResultElts ==
  // gpuHandle->getBufferRowElts("OutputScoreValue"));//OutputScoreValue assert(inputBuffers->singleOwnershipResultElts
  // == gpuHandle->getBufferRowElts("OutputOwnership"));//OutputOwnership
  assert(inputBuffers->singleout_policyElts == gpuHandle->getBufferRowElts("out_policy"));
  assert(inputBuffers->singleout_valueElts == gpuHandle->getBufferRowElts("out_value"));
  assert(inputBuffers->singleout_miscvalueElts == gpuHandle->getBufferRowElts("out_miscvalue"));
  assert(inputBuffers->singleout_moremiscvalueElts == gpuHandle->getBufferRowElts("out_moremiscvalue"));
  assert(inputBuffers->singleout_ownershipElts == gpuHandle->getBufferRowElts("out_ownership"));
  // assert(inputBuffers->singleout_scoringElts == gpuHandle->getBufferRowElts("out_scoring"));
  // assert(inputBuffers->singleout_futureposElts == gpuHandle->getBufferRowElts("out_futurepos"));
  // assert(inputBuffers->singleout_sekiElts == gpuHandle->getBufferRowElts("out_seki"));
  // assert(inputBuffers->singleout_scorebelief_logprobsElts ==
  // gpuHandle->getBufferRowElts("out_scorebelief_logprobs")); assert(inputBuffers->singleiout_policyElts ==
  // gpuHandle->getBufferRowElts("out_policy")); assert(inputBuffers->singleiout_valueElts ==
  // gpuHandle->getBufferRowElts("out_value")); assert(inputBuffers->singleiout_miscvalueElts ==
  // gpuHandle->getBufferRowElts("out_miscvalue")); assert(inputBuffers->singleiout_moremiscvalueElts ==
  // gpuHandle->getBufferRowElts("out_moremiscvalue")); assert(inputBuffers->singleiout_ownershipElts ==
  // gpuHandle->getBufferRowElts("out_ownership")); assert(inputBuffers->singleiout_scoringElts ==
  // gpuHandle->getBufferRowElts("out_scoring")); assert(inputBuffers->singleiout_futureposElts ==
  // gpuHandle->getBufferRowElts("out_futurepos")); assert(inputBuffers->singleiout_sekiElts ==
  // gpuHandle->getBufferRowElts("out_seki")); assert(inputBuffers->singleiout_scorebelief_logprobsElts ==
  // gpuHandle->getBufferRowElts("out_scorebelief_logprobs"));

  // assert(inputBuffers->maskInputBufferBytes == gpuHandle->getBufferBytes("InputMask"));
  assert(inputBuffers->featureInputBufferBytes == gpuHandle->getBufferBytes("input_spatial"));  // InputFeature
  assert(
    inputBuffers->globalFeatureInputBufferBytes == gpuHandle->getBufferBytes("input_global"));  // InputGlobalFeature

  // assert(inputBuffers->policyPassResultBufferBytes ==
  // gpuHandle->getBufferBytes("OutputPolicyPass"));//OutputPolicyPass assert(inputBuffers->policyResultBufferBytes ==
  // gpuHandle->getBufferBytes("OutputPolicy"));//OutputPolicy assert(inputBuffers->valueResultBufferBytes ==
  // gpuHandle->getBufferBytes("OutputValue"));//OutputValue assert(inputBuffers->scoreValueResultBufferBytes ==
  // gpuHandle->getBufferBytes("OutputScoreValue"));//OutputScoreValue assert(inputBuffers->ownershipResultBufferBytes ==
  // gpuHandle->getBufferBytes("OutputOwnership"));//OutputOwnership

  // const int numPolicyChannels = inputBuffers->singlePolicyPassResultElts;
  // assert(inputBuffers->singlePolicyResultElts == numPolicyChannels * nnXLen * nnYLen);

  // Transfers from host memory to device memory are asynchronous with respect to the host
  // CUDA_ERR(
  //"getOutput",
  // cudaMemcpyAsync(
  // gpuHandle->getBuffer("InputMask"),
  // inputBuffers->maskInputs.get(),
  // inputBuffers->singleMaskBytes * batchSize,
  // cudaMemcpyHostToDevice));
  CUDA_ERR(
    "getOutput",
    cudaMemcpyAsync(
      gpuHandle->getBuffer("input_spatial"),
      inputBuffers->featureInputs.get(),
      inputBuffers->singleFeatureBytes * batchSize,
      cudaMemcpyHostToDevice));
  CUDA_ERR(
    "getOutput",
    cudaMemcpyAsync(
      gpuHandle->getBuffer("input_global"),
      inputBuffers->globalFeatureInputs.get(),
      inputBuffers->singleGlobalFeatureBytes * batchSize,
      cudaMemcpyHostToDevice));

  // std::cout << "batchSize:" << batchSize << std::endl;
  // std::cout << "singleMaskBytes:" << inputBuffers->singleMaskBytes / sizeof(float) << std::endl;
  // std::cout << "singleFeatureBytes:" << inputBuffers->singleFeatureBytes / sizeof(float) << std::endl;
  // std::cout << "singleGlobalFeatureBytes:" << inputBuffers->singleGlobalFeatureBytes / sizeof(float) << std::endl;

  // auto maskInputDims = gpuHandle->getBufferDynamicShape("InputMask", batchSize);

  auto featureInputDims = gpuHandle->getBufferDynamicShape("input_spatial", batchSize);
  auto globalFeatureInputDims = gpuHandle->getBufferDynamicShape("input_global", batchSize);

  gpuHandle->exec->setInputShape("input_spatial", featureInputDims);
  gpuHandle->exec->setInputShape("input_global", globalFeatureInputDims);
  gpuHandle->exec->enqueueV3(cudaStreamPerThread);
  gpuHandle->infer_times++;
  // CUDA_ERR(
  //  "getOutput",
  //  cudaMemcpy(
  //    inputBuffers->policyPassResults.get(),
  //    gpuHandle->getBuffer("OutputPolicyPass"),
  //    inputBuffers->singlePolicyPassResultBytes * batchSize,
  //    cudaMemcpyDeviceToHost));
  // CUDA_ERR(
  //  "getOutput",
  //  cudaMemcpy(
  //    inputBuffers->policyResults.get(),
  //    gpuHandle->getBuffer("OutputPolicy"),
  //    inputBuffers->singlePolicyResultBytes * batchSize,
  //    cudaMemcpyDeviceToHost));
  // CUDA_ERR(
  //  "getOutput",
  //  cudaMemcpy(
  //    inputBuffers->valueResults.get(),
  //    gpuHandle->getBuffer("OutputValue"),
  //    inputBuffers->singleValueResultBytes * batchSize,
  //    cudaMemcpyDeviceToHost));
  // CUDA_ERR(
  //  "getOutput",
  //  cudaMemcpy(
  //    inputBuffers->scoreValueResults.get(),
  //    gpuHandle->getBuffer("OutputScoreValue"),
  //    inputBuffers->singleScoreValueResultBytes * batchSize,
  //    cudaMemcpyDeviceToHost));
  // CUDA_ERR(
  //  "getOutput",
  //  cudaMemcpy(
  //    inputBuffers->ownershipResults.get(),
  //    gpuHandle->getBuffer("OutputOwnership"),
  //    inputBuffers->singleOwnershipResultBytes * batchSize,
  //    cudaMemcpyDeviceToHost));

  CUDA_ERR(
    "getOutput",
    cudaMemcpy(
      inputBuffers->out_policyResults.get(),
      gpuHandle->getBuffer("out_policy"),
      inputBuffers->singleout_policyBytes * batchSize,
      cudaMemcpyDeviceToHost));
  CUDA_ERR(
    "getOutput",
    cudaMemcpy(
      inputBuffers->out_valueResults.get(),
      gpuHandle->getBuffer("out_value"),
      inputBuffers->singleout_valueBytes * batchSize,
      cudaMemcpyDeviceToHost));
  CUDA_ERR(
    "getOutput",
    cudaMemcpy(
      inputBuffers->out_miscvalueResults.get(),
      gpuHandle->getBuffer("out_miscvalue"),
      inputBuffers->singleout_miscvalueBytes * batchSize,
      cudaMemcpyDeviceToHost));
  CUDA_ERR(
    "getOutput",
    cudaMemcpy(
      inputBuffers->out_moremiscvalueResults.get(),
      gpuHandle->getBuffer("out_moremiscvalue"),
      inputBuffers->singleout_moremiscvalueBytes * batchSize,
      cudaMemcpyDeviceToHost));
  CUDA_ERR(
    "getOutput",
    cudaMemcpy(
      inputBuffers->out_ownershipResults.get(),
      gpuHandle->getBuffer("out_ownership"),
      inputBuffers->singleout_ownershipBytes * batchSize,
      cudaMemcpyDeviceToHost));

#if false
  if(gpuHandle->infer_times%10==0) {
    FILE* fp_policyPassResults = fopen(("Oresult/out_policy_" + std::to_string(gpuHandle->infer_times)).c_str(), "w+");
    FILE* fp_policyResults = fopen(("onnxresult/out_value" + std::to_string(gpuHandle->infer_times)).c_str(), "w+");
    FILE* fp_valueResults = fopen(("onnxresult/out_miscvalue_" + std::to_string(gpuHandle->infer_times)).c_str(), "w+");
    FILE* fp_scoreValueResults = fopen(("onnxresult/out_moremiscvalue_" + std::to_string(gpuHandle->infer_times)).c_str(), "w+");
    FILE* fp_ownershipResults = fopen(("onnxresult/out_ownership_" + std::to_string(gpuHandle->infer_times)).c_str(), "w+");

    
    for(int i = 0; i < inputBuffers->singleout_policyBytes * batchSize / sizeof(float); i++) {
      char buf[128];
      sprintf(buf, "%.4f ", inputBuffers->out_policyResults.get()[i]);
      fwrite(buf, strlen(buf), 1, fp_policyPassResults);
    }
    fclose(fp_policyPassResults);
    for(int i = 0; i < inputBuffers->singleout_valueBytes * batchSize / sizeof(float); i++) {
      char buf[128];
      sprintf(buf, "%.4f ", inputBuffers->out_valueResults.get()[i]);
      fwrite(buf, strlen(buf), 1, fp_policyResults);
    }
    fclose(fp_policyResults);
    for(int i = 0; i < inputBuffers->singleout_miscvalueBytes * batchSize / sizeof(float); i++) {
      char buf[128];
      sprintf(buf, "%.4f ", inputBuffers->out_miscvalueResults.get()[i]);
      fwrite(buf, strlen(buf), 1, fp_valueResults);
    }
    fclose(fp_valueResults);
    for(int i = 0; i < inputBuffers->singleout_moremiscvalueBytes * batchSize / sizeof(float); i++) {
      char buf[128];
      sprintf(buf, "%.4f ", inputBuffers->out_moremiscvalueResults.get()[i]);
      fwrite(buf, strlen(buf), 1, fp_scoreValueResults);
    }
    fclose(fp_scoreValueResults);
    for(int i = 0; i < inputBuffers->singleout_ownershipBytes * batchSize / sizeof(float); i++) {
      char buf[128];
      sprintf(buf, "%.4f ", inputBuffers->out_ownershipResults.get()[i]);
      fwrite(buf, strlen(buf), 1, fp_ownershipResults);
    }
    fclose(fp_ownershipResults);
  }
#endif
  // CUDA_ERR(
  //  "getOutput",
  //  cudaMemcpy(
  //    inputBuffers->out_scoringResults.get(),
  //    gpuHandle->getBuffer("out_scoring"),
  //    inputBuffers->singleout_scoringBytes * batchSize,
  //    cudaMemcpyDeviceToHost));
  // CUDA_ERR(
  //  "getOutput",
  //  cudaMemcpy(
  //    inputBuffers->out_futureposResults.get(),
  //    gpuHandle->getBuffer("out_futurepos"),
  //    inputBuffers->singleout_futureposBytes * batchSize,
  //    cudaMemcpyDeviceToHost));
  // CUDA_ERR(
  //  "getOutput",
  //  cudaMemcpy(
  //    inputBuffers->out_sekiResults.get(),
  //    gpuHandle->getBuffer("out_seki"),
  //    inputBuffers->singleout_sekiBytes * batchSize,
  //    cudaMemcpyDeviceToHost));
  // CUDA_ERR(
  //  "getOutput",
  //  cudaMemcpy(
  //    inputBuffers->out_scorebelief_logprobsResults.get(),
  //    gpuHandle->getBuffer("out_scorebelief_logprobs"),
  //    inputBuffers->singleout_scorebelief_logprobsBytes * batchSize,
  //    cudaMemcpyDeviceToHost));


  // CUDA_ERR(
  //  "getOutput",
  //  cudaMemcpy(
  //    inputBuffers->iout_policyResults.get(),
  //    gpuHandle->getBuffer("iout_policy"),
  //    inputBuffers->singleiout_policyBytes * batchSize,
  //    cudaMemcpyDeviceToHost));
  // CUDA_ERR(
  //  "getOutput",
  //  cudaMemcpy(
  //    inputBuffers->iout_valueResults.get(),
  //    gpuHandle->getBuffer("iout_value"),
  //    inputBuffers->singleiout_valueBytes * batchSize,
  //    cudaMemcpyDeviceToHost));
  // CUDA_ERR(
  //  "getOutput",
  //  cudaMemcpy(
  //    inputBuffers->iout_miscvalueResults.get(),
  //    gpuHandle->getBuffer("iout_miscvalue"),
  //    inputBuffers->singleiout_miscvalueBytes * batchSize,
  //    cudaMemcpyDeviceToHost));
  // CUDA_ERR(
  //  "getOutput",
  //  cudaMemcpy(
  //    inputBuffers->iout_moremiscvalueResults.get(),
  //    gpuHandle->getBuffer("iout_moremiscvalue"),
  //    inputBuffers->singleiout_moremiscvalueBytes * batchSize,
  //    cudaMemcpyDeviceToHost));
  // CUDA_ERR(
  //  "getOutput",
  //  cudaMemcpy(
  //    inputBuffers->iout_ownershipResults.get(),
  //    gpuHandle->getBuffer("iout_ownership"),
  //    inputBuffers->singleiout_ownershipBytes * batchSize,
  //    cudaMemcpyDeviceToHost));
  // CUDA_ERR(
  //  "getOutput",
  //  cudaMemcpy(
  //    inputBuffers->iout_scoringResults.get(),
  //    gpuHandle->getBuffer("iout_scoring"),
  //    inputBuffers->singleiout_scoringBytes * batchSize,
  //    cudaMemcpyDeviceToHost));
  // CUDA_ERR(
  //  "getOutput",
  //  cudaMemcpy(
  //    inputBuffers->iout_futureposResults.get(),
  //    gpuHandle->getBuffer("iout_futurepos"),
  //    inputBuffers->singleiout_futureposBytes * batchSize,
  //    cudaMemcpyDeviceToHost));
  // CUDA_ERR(
  //  "getOutput",
  //  cudaMemcpy(
  //    inputBuffers->iout_sekiResults.get(),
  //    gpuHandle->getBuffer("iout_seki"),
  //    inputBuffers->singleiout_sekiBytes * batchSize,
  //    cudaMemcpyDeviceToHost));
  // CUDA_ERR(
  //  "getOutput",
  //  cudaMemcpy(
  //    inputBuffers->iout_scorebelief_logprobsResults.get(),
  //    gpuHandle->getBuffer("iout_scorebelief_logprobs"),
  //    inputBuffers->singleiout_scorebelief_logprobsBytes * batchSize,
  //    cudaMemcpyDeviceToHost));

  gpuHandle->printDebugOutput(batchSize);

  assert(outputs.size() == batchSize);

  float policyProbsTmp[NNPos::MAX_NN_POLICY_SIZE];

  for(int row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];

    assert(output->nnXLen == nnXLen);
    assert(output->nnYLen == nnYLen);
    float policyOptimism = (float)inputBufs[row]->policyOptimism;

    // const float* policyPassSrcBuf = &inputBuffers->out_policyResults[row * inputBuffers->singleout_policyElts];
    const float* policySrcBuf = &inputBuffers->out_policyResults[row * inputBuffers->singleout_policyElts];
    float* policyProbs = output->policyProbs;

    // These are in logits, the client does the postprocessing to turn them into
    // policy probabilities and white game outcome probabilities
    // Also we don't fill in the nnHash here either
    // Handle version >= 12 policy optimism
    int numPolicyChannels = (modelVersion >= 12 && modelVersion <= 99) ? 2 : 1;
    if(numPolicyChannels == 2) {
      // TRT is all NCHW
      for(int i = 0; i < nnXLen * nnYLen; i++) {
        float p = policySrcBuf[i];
        // float pOpt = policySrcBuf[i + nnXLen * nnYLen];// ֹ۲  ԣ 362*5
        float pOpt = policySrcBuf[i + 5 * (nnXLen * nnYLen + 1)];
        policyProbsTmp[i] = p + (pOpt - p) * policyOptimism;
      }
      SymmetryHelpers::copyOutputsWithSymmetry(
        policyProbsTmp, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);  // 361,362*6-1 = 2171,   һλΪ2171
      // policyProbs[nnXLen * nnYLen] = policyPassSrcBuf[0] + (policyPassSrcBuf[1] - policyPassSrcBuf[0]) *
      // policyOptimism;
      policyProbs[nnXLen * nnYLen] = 
        policySrcBuf[nnXLen * nnYLen] + 
        (policySrcBuf[5 * (nnXLen * nnYLen + 1) + nnXLen * nnYLen] - policySrcBuf[nnXLen * nnYLen]) * policyOptimism;
    } 
    else {
      assert(numPolicyChannels == 1);
      SymmetryHelpers::copyOutputsWithSymmetry(policySrcBuf, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
      
      policyProbs[nnXLen * nnYLen] = policySrcBuf[nnXLen * nnYLen];
    }

    // int numValueChannels = inputBuffers->singleValueResultElts;
    int numValueChannels = inputBuffers->singleout_valueElts;
    assert(numValueChannels == 3);
    // output->whiteWinProb = inputBuffers->valueResults[row * numValueChannels];
    // output->whiteLossProb = inputBuffers->valueResults[row * numValueChannels + 1];
    // output->whiteNoResultProb = inputBuffers->valueResults[row * numValueChannels + 2];
    output->whiteWinProb = inputBuffers->out_valueResults[row * numValueChannels];
    output->whiteLossProb = inputBuffers->out_valueResults[row * numValueChannels + 1];
    output->whiteNoResultProb = inputBuffers->out_valueResults[row * numValueChannels + 2];

    // As above, these are NOT actually from white's perspective, but rather the player to move.
    // As usual the client does the postprocessing.
    if(output->whiteOwnerMap != NULL) {
      // const float* ownershipSrcBuf = &inputBuffers->ownershipResults[row * nnXLen * nnYLen];
      const float* ownershipSrcBuf = &inputBuffers->out_ownershipResults[row * nnXLen * nnYLen];
      assert(inputBuffers->singleout_ownershipElts == nnXLen * nnYLen);
      SymmetryHelpers::copyOutputsWithSymmetry(
        ownershipSrcBuf, output->whiteOwnerMap, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    }
    // int numScoreValueChannels = inputBuffers->singlevalueResultElts;
    int numScoreValueChannels = inputBuffers->singleout_valueElts;
    if(modelVersion >= 9) {
      // assert(numScoreValueChannels == 6);
      // output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      // output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
      // output->whiteLead = inputBuffers->scoreValueResults[row * numScoreValueChannels + 2];
      // output->varTimeLeft = inputBuffers->scoreValueResults[row * numScoreValueChannels + 3];
      // output->shorttermWinlossError = inputBuffers->scoreValueResults[row * numScoreValueChannels + 4];
      // output->shorttermScoreError = inputBuffers->scoreValueResults[row * numScoreValueChannels + 5];
      output->whiteScoreMean = inputBuffers->out_miscvalueResults[row * 10];
      output->whiteScoreMeanSq = inputBuffers->out_miscvalueResults[row * 10 + 1];
      output->whiteLead = inputBuffers->out_miscvalueResults[row * 10 + 2];
      output->varTimeLeft = inputBuffers->out_miscvalueResults[row * 10 + 3];
      output->shorttermWinlossError = inputBuffers->out_moremiscvalueResults[row * 8];
      output->shorttermScoreError = inputBuffers->out_moremiscvalueResults[row * 8 + 1];
    } else if(modelVersion >= 8) {
      // assert(numScoreValueChannels == 4);
      // output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      // output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
      // output->whiteLead = inputBuffers->scoreValueResults[row * numScoreValueChannels + 2];
      // output->varTimeLeft = inputBuffers->scoreValueResults[row * numScoreValueChannels + 3];
      // output->shorttermWinlossError = 0;
      // output->shorttermScoreError = 0;
      std::cout << "you need higher model version !" << endl;
    } else if(modelVersion >= 4) {
      std::cout << "you need higher model version !" << endl;
      // assert(numScoreValueChannels == 2);
      // output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      // output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
      // output->whiteLead = output->whiteScoreMean;
      // output->varTimeLeft = 0;
      // output->shorttermWinlossError = 0;
      // output->shorttermScoreError = 0;
    } else if(modelVersion >= 3) {
      std::cout << "you need higher model version !" << endl;
      // assert(numScoreValueChannels == 1);
      // output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      // //Version 3 neural nets don't have any second moment output, implicitly already folding it in, so we just use
      // the
      // //mean squared
      // output->whiteScoreMeanSq = output->whiteScoreMean * output->whiteScoreMean;
      // output->whiteLead = output->whiteScoreMean;
      // output->varTimeLeft = 0;
      // output->shorttermWinlossError = 0;
      // output->shorttermScoreError = 0;
    } else {
      ASSERT_UNREACHABLE;
    }
  }
}

bool NeuralNet::testEvaluateConv(
  const ConvLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  vector<float>& outputBuffer) {
  (void)desc;
  (void)batchSize;
  (void)nnXLen;
  (void)nnYLen;
  (void)useFP16;
  (void)useNHWC;
  (void)inputBuffer;
  (void)outputBuffer;
  return false;
}

// Mask should be in 'NHW' format (no "C" channel).
bool NeuralNet::testEvaluateBatchNorm(
  const BatchNormLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer) {
  (void)desc;
  (void)batchSize;
  (void)nnXLen;
  (void)nnYLen;
  (void)useFP16;
  (void)useNHWC;
  (void)inputBuffer;
  (void)maskBuffer;
  (void)outputBuffer;
  return false;
}

bool NeuralNet::testEvaluateResidualBlock(
  const ResidualBlockDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer) {
  (void)desc;
  (void)batchSize;
  (void)nnXLen;
  (void)nnYLen;
  (void)useFP16;
  (void)useNHWC;
  (void)inputBuffer;
  (void)maskBuffer;
  (void)outputBuffer;
  return false;
}

bool NeuralNet::testEvaluateGlobalPoolingResidualBlock(
  const GlobalPoolingResidualBlockDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer) {
  (void)desc;
  (void)batchSize;
  (void)nnXLen;
  (void)nnYLen;
  (void)useFP16;
  (void)useNHWC;
  (void)inputBuffer;
  (void)maskBuffer;
  (void)outputBuffer;
  return false;
}

#endif  // USE_TENSORRT_BACKEND

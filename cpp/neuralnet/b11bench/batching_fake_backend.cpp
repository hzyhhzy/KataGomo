// Only linked by batching_cpu_test.py, never by the production engine.
#include "batching_fake_backend.h"
#include "../nneval.h"
#include <chrono>
#include <condition_variable>
#include <mutex>

struct LoadedModel { ModelDesc desc; };
struct ComputeContext { int eligibleDevice; };
struct ComputeHandle { int capacity; int x; int y; int thread; bool eligible; mutable bool warmup; };
struct InputBuffers { int capacity; };

namespace {
std::mutex stateMutex;
std::condition_variable stateChanged;
std::vector<BatchingFake::Call> observed;
std::vector<std::string> failures;
bool blockNext = false;
bool blocked = false;
bool released = false;
int handles = 0;
int buffers = 0;
std::vector<int> expectedMapping;
void expect(bool ok,const std::string& message) {
  if(!ok) failures.push_back(message);
}
}

namespace BatchingFake {
void reset() {
  std::lock_guard<std::mutex> lock(stateMutex);
  observed.clear(); failures.clear(); blockNext=false; blocked=false; released=false;
}
void blockNextCall() {
  std::lock_guard<std::mutex> lock(stateMutex);
  blockNext=true; blocked=false; released=false;
}
bool waitUntilBlocked(int milliseconds) {
  std::unique_lock<std::mutex> lock(stateMutex);
  return stateChanged.wait_for(lock,std::chrono::milliseconds(milliseconds),[] { return blocked; });
}
void releaseCall() {
  std::lock_guard<std::mutex> lock(stateMutex);
  released=true; stateChanged.notify_all();
}
std::vector<Call> calls() { std::lock_guard<std::mutex> lock(stateMutex); return observed; }
std::vector<std::string> errors() { std::lock_guard<std::mutex> lock(stateMutex); return failures; }
int liveHandles() { std::lock_guard<std::mutex> lock(stateMutex); return handles; }
int liveBuffers() { std::lock_guard<std::mutex> lock(stateMutex); return buffers; }
std::vector<int> expectedThreadMapping() { std::lock_guard<std::mutex> lock(stateMutex); return expectedMapping; }
}

namespace NeuralNet {
void globalInitialize() {}
void globalCleanup() {}
void printDevices() {}
std::string getRuntimeBackendDetail(ConfigParser&) { return "batchingcpufake"; }
BatchPolicy getBatchPolicy(ConfigParser&) { return BatchPolicy::Dynamic; }
int getNumEffectiveDevices(ConfigParser&,const std::vector<int>& devices) {
  return std::max(1,(int)std::set<int>(devices.begin(),devices.end()).size());
}
LoadedModel* loadModelFile(const std::string& name,const std::string&) {
  auto* model = new LoadedModel;
  ModelDesc& d=model->desc;
  d.name="batching-cpu-fake"; d.modelVersion=17;
  d.numInputChannels=22; d.numInputGlobalChannels=19;
  d.numInputMetaChannels=name=="metadata" ? SGFMetadata::METADATA_INPUT_NUM_CHANNELS : 0;
  d.numPolicyChannels=2; d.numValueChannels=3; d.numScoreValueChannels=6; d.numOwnershipChannels=1;
  d.archSummary.present=true;
  return model;
}
void freeLoadedModel(LoadedModel* model) { delete model; }
const ModelDesc& getModelDesc(const LoadedModel* model) { return model->desc; }
ComputeContext* createComputeContext(const std::vector<int>&,Logger*,int,int,const std::string&,
  enabled_t,const LoadedModel*,ConfigParser& cfg) {
  return new ComputeContext{cfg.contains("batchingFakeEligibleDevice") ? cfg.getInt("batchingFakeEligibleDevice") : -99};
}
void freeComputeContext(ComputeContext* context) { delete context; }
ComputeHandle* createComputeHandle(ComputeContext* context,const LoadedModel*,Logger*,int capacity,
  bool,bool,int device,int thread) {
  std::lock_guard<std::mutex> lock(stateMutex); handles++;
  return new ComputeHandle{capacity,19,19,thread,(device==-1 ? 0 : device)==context->eligibleDevice,false};
}
#ifdef USE_CUDA_BACKEND
// Exercise NNEvaluator's CUDA-only selection and topology plumbing without
// including CUDA headers, loading its runtime, or pretending to test hardware.
void setExpectedConcurrentGpuThreads(ComputeContext*,const std::vector<int>& mapping) {
  std::lock_guard<std::mutex> lock(stateMutex); expectedMapping=mapping;
}
bool isB11BatchingEligible(const ComputeHandle* handle) { return handle && handle->eligible; }
#endif
void freeComputeHandle(ComputeHandle* handle) {
  if(handle) { std::lock_guard<std::mutex> lock(stateMutex); handles--; delete handle; }
}
bool isUsingFP16(const ComputeHandle*) { return false; }
bool setIsWarmup(const ComputeHandle* handle,bool warmup) {
  bool previous=handle->warmup; handle->warmup=warmup; return previous;
}
InputBuffers* createInputBuffers(const LoadedModel*,int capacity,int,int) {
  std::lock_guard<std::mutex> lock(stateMutex); buffers++;
  return new InputBuffers{capacity};
}
void freeInputBuffers(InputBuffers* input) {
  if(input) { std::lock_guard<std::mutex> lock(stateMutex); buffers--; delete input; }
}
void getOutput(ComputeHandle* handle,InputBuffers* input,int batch,NNResultBuf** rows,
  std::vector<NNOutput*>& outputs) {
  {
    std::unique_lock<std::mutex> lock(stateMutex);
    expect(batch>0 && batch<=handle->capacity,"physical batch outside handle capacity");
    expect(batch<=input->capacity,"physical batch outside input capacity");
    expect(outputs.size()==(size_t)batch,"output vector size differs from physical batch");
    BatchingFake::Call call{batch,handle->capacity,handle->thread,{}};
    for(int i=0;i<batch;i++) {
      const NNResultBuf& r=*rows[i];
      NNOutput& o=*outputs.at(i);
      expect(r.rowSpatialBuf.size()==19*19*22,"spatial buffer has wrong size");
      expect(r.rowGlobalBuf.size()==19,"global buffer has wrong size");
      expect(!r.hasRowMeta || r.rowMetaBuf.size()==SGFMetadata::METADATA_INPUT_NUM_CHANNELS,
        "metadata buffer has wrong size");
      expect(o.nnXLen==19 && o.nnYLen==19,"dummy or real output has wrong board size");
      call.rows.push_back(BatchingFake::Row{
        reinterpret_cast<uintptr_t>(rows[i]),reinterpret_cast<uintptr_t>(outputs[i]),
        reinterpret_cast<uintptr_t>(o.whiteOwnerMap),r.includeOwnerMap,r.hasRowMeta,
        r.boardXSizeForServer,r.boardYSizeForServer,r.symmetry,r.policyOptimism,
        r.rowSpatialBuf,r.rowGlobalBuf,r.rowMetaBuf});
    }
    observed.push_back(std::move(call));
    if(blockNext) {
      blockNext=false; blocked=true; stateChanged.notify_all();
      if(!stateChanged.wait_for(lock,std::chrono::seconds(5),[] { return released; }))
        failures.push_back("fake backend gate timed out");
      blocked=false;
    }
  }
  // Fill every physical row, including dummy ownership maps. Aliased/undersized
  // dummy allocations are therefore exercised rather than merely inspected.
  for(int i=0;i<batch;i++) {
    NNOutput& o=*outputs.at(i);
    const float tag=0.001f*rows[i]->boardXSizeForServer+0.0001f*rows[i]->boardYSizeForServer;
    std::fill(o.policyProbs,o.policyProbs+NNPos::MAX_NN_POLICY_SIZE,0.0f);
    o.whiteWinProb=tag; o.whiteLossProb=-tag; o.whiteNoResultProb=-4.0f;
    o.whiteScoreMean=tag; o.whiteScoreMeanSq=0.1f; o.whiteLead=tag;
    o.varTimeLeft=0.1f; o.shorttermWinlossError=0.01f; o.shorttermScoreError=0.01f;
    o.policyOptimismUsed=(float)rows[i]->policyOptimism;
    if(o.whiteOwnerMap)
      std::fill(o.whiteOwnerMap,o.whiteOwnerMap+19*19,tag);
  }
}
}

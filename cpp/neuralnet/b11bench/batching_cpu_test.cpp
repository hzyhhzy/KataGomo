// CPU integration tests of production NNEvaluator::serve(), with no CUDA or weights.
#include "batching_fake_backend.h"
#include "../nneval.h"
#include <chrono>
#include <future>
#include <iostream>
#include <set>
#include <thread>
#ifdef _WIN32
#include <windows.h>
#include <crtdbg.h>
#include <cstdlib>
#endif

// The production class grants this narrowly scoped friend access. Queue-size
// observation makes exact underfilled batches deterministic without changing
// production dispatch, exposing a public hook, or guessing a producer delay.
struct NNBatchingCpuTestAccess {
  static size_t queued(NNEvaluator& eval) { return eval.queryQueue.size(); }
  static int ongoing(NNEvaluator& eval) {
    std::lock_guard<std::mutex> lock(eval.bufferMutex);
    return eval.numOngoingEvals;
  }
};

namespace {
int checks=0;
int scenarios=0;
void require(bool condition,const std::string& message) {
  checks++;
  if(!condition) throw std::runtime_error(message);
}
template<class Predicate>
bool waitFor(Predicate predicate) {
  auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);
  while(!predicate()) {
    if(std::chrono::steady_clock::now()>=deadline) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

struct Request {
  Board board;
  BoardHistory history;
  SGFMetadata metadata;
  MiscNNInputParams params;
  NNResultBuf buffer;
  bool owner;
  Request(int index,bool withMetadata,bool partialBoards)
    : board(partialBoards && index%2==0 ? 13 : 19,partialBoards && index%2==0 ? 9 : 19),
      history(board,P_BLACK,Rules::getTrompTaylorish(),0,BoardHistoryModes(false,false)),
      metadata(SGFMetadata::makeDummyWarmupProfile()),owner(index%2!=0) {
    params.symmetry=index%8;
    params.policyOptimism=withMetadata ? 0.0 : 0.1*(index%3);
    metadata.inverseBRank=1+index%8;
    metadata.inverseWRank=3+index%8;
  }
};

std::future<void> submit(NNEvaluator& eval,Request& request,bool withMetadata) {
  return std::async(std::launch::async,[&eval,&request,withMetadata] {
    eval.evaluate(request.board,request.history,P_BLACK,
      withMetadata ? &request.metadata : nullptr,request.params,request.buffer,true,request.owner);
  });
}

bool sameInput(const BatchingFake::Row& a,const BatchingFake::Row& b) {
  return a.input==b.input && a.requestedOwner==b.requestedOwner && a.hasMeta==b.hasMeta &&
    a.boardX==b.boardX && a.boardY==b.boardY && a.symmetry==b.symmetry && a.optimism==b.optimism &&
    a.spatial==b.spatial && a.global==b.global && a.meta==b.meta;
}

void validateCall(const BatchingFake::Call& call,int real,int physical,bool metadata,bool nhwc) {
  require(call.batch==physical,"wrong physical batch: got "+std::to_string(call.batch)+
    ", expected "+std::to_string(physical));
  require((int)call.rows.size()==physical,"wrong number of recorded physical rows");
  std::set<uintptr_t> inputPointers;
  std::set<uintptr_t> outputPointers;
  std::set<uintptr_t> ownerPointers;
  for(int i=0;i<physical;i++) {
    const auto& row=call.rows[i];
    require(row.hasMeta==metadata,"metadata-presence flag changed");
    require(metadata ? row.meta.size()==192 : row.meta.empty(),"metadata data size changed");
    require(row.symmetry>=0 && row.symmetry<8,"symmetry was not resolved before padding");
    int ones=0;
    for(int xy=0;xy<19*19;xy++) {
      float mask=row.spatial[nhwc ? xy*22 : xy];
      require(mask==0.0f || mask==1.0f,"mask is not binary");
      ones+=(mask==1.0f);
    }
    require(ones==row.boardX*row.boardY,"mask area changed or padded row has all-zero mask");
    if(i<real) {
      require(inputPointers.insert(row.input).second,"real request duplicated within batch");
      require((row.owner!=0)==row.requestedOwner,"real output ownership allocation changed");
    }
    else {
      require(sameInput(row,call.rows[real-1]),"padding did not preserve the last real request");
      require(inputPointers.count(row.input)==1,"padding did not reference an existing real row");
    }
    require(outputPointers.insert(row.output).second,"real/dummy output buffers alias within batch");
    if(row.owner)
      require(ownerPointers.insert(row.owner).second,"ownership buffers alias within batch");
  }
}

void runWave(NNEvaluator& eval,int capacity,int count,bool pad,bool metadata,bool nhwc,bool partialBoards) {
  scenarios++;
  BatchingFake::reset();
  eval.clearStats();
  BatchingFake::blockNextCall();
  Request first(91,metadata,false);
  auto firstDone=submit(eval,first,metadata);
  require(BatchingFake::waitUntilBlocked(3000),"first evaluation did not reach backend promptly");

  std::vector<std::unique_ptr<Request>> requests;
  std::vector<std::future<void>> completions;
  for(int i=0;i<count;i++) {
    requests.emplace_back(new Request(i,metadata,partialBoards));
    completions.push_back(submit(eval,*requests.back(),metadata));
  }
  bool queued=waitFor([&] { return NNBatchingCpuTestAccess::queued(eval)==(size_t)count; });
  // Release even on a failed queue assertion, so outstanding clients can finish.
  BatchingFake::releaseCall();
  require(queued,"all real requests did not enqueue before releasing the first batch");
  require(firstDone.wait_for(std::chrono::seconds(3))==std::future_status::ready,"first request hung");
  firstDone.get();
  for(auto& done:completions) {
    require(done.wait_for(std::chrono::seconds(3))==std::future_status::ready,"real request hung");
    done.get();
  }
  require(waitFor([&] { return NNBatchingCpuTestAccess::ongoing(eval)==0; }),"real outstanding count did not drain");
  auto calls=BatchingFake::calls();
  require(calls.size()==2,"expected exactly isolated bootstrap and queued test batch");
  validateCall(calls[0],1,pad ? capacity : 1,metadata,nhwc);
  validateCall(calls[1],count,pad ? capacity : count,metadata,nhwc);
  if(pad) {
    for(int i=0;i<capacity-count;i++) {
      require(calls[0].rows[1+i].output==calls[1].rows[count+i].output,
        "per-thread dummy output was not reused between batches");
      require(calls[0].rows[1+i].owner==calls[1].rows[count+i].owner,
        "per-thread dummy ownership storage was not reused between batches");
    }
  }
  require(eval.numRowsProcessed()==(uint64_t)(count+1),"stats counted padding as real rows");
  require(eval.numBatchesProcessed()==2,"batch count is incorrect");
  require(eval.averageProcessedBatchSize()==(count+1)/2.0,"average used physical rows");
  require(eval.numCacheHits()==0,"skipCache requests unexpectedly hit cache");
  require(BatchingFake::errors().empty(),"fake backend detected an input/output contract failure");
  std::set<const NNOutput*> delivered;
  for(const auto& request:requests) {
    require(request->buffer.hasResult && request->buffer.result!=nullptr,"real result not delivered");
    const NNOutput& output=*request->buffer.result;
    const auto recorded=std::find_if(calls[1].rows.begin(),calls[1].rows.begin()+count,
      [&](const BatchingFake::Row& row) {
        return row.input==reinterpret_cast<uintptr_t>(&request->buffer);
      });
    require(recorded!=calls[1].rows.begin()+count,"real request did not reach the backend");
    require(recorded->output==reinterpret_cast<uintptr_t>(&output),"output delivered to the wrong real request");
    require(recorded->spatial==request->buffer.rowSpatialBuf && recorded->global==request->buffer.rowGlobalBuf,
      "real input feature buffers changed while completing the batch");
    if(metadata) {
      std::vector<float> expectedMeta(SGFMetadata::METADATA_INPUT_NUM_CHANNELS);
      SGFMetadata::fillMetadataRow(&request->metadata,expectedMeta.data(),P_BLACK,
        request->board.x_size*request->board.y_size);
      require(recorded->meta==expectedMeta,"real request metadata differs from its profile");
    }
    require(delivered.insert(&output).second,"clients received aliased outputs");
    require((output.whiteOwnerMap!=nullptr)==request->owner,"client ownership map changed");
    require(std::isfinite(output.whiteWinProb) && std::isfinite(output.whiteScoreMean),"invalid postprocessed result");
    if(output.whiteOwnerMap) {
      const float tag=0.001f*request->board.x_size+0.0001f*request->board.y_size;
      for(int y=0;y<19;y++) for(int x=0;x<19;x++) {
        const float expected=x<request->board.x_size && y<request->board.y_size ? -std::tanh(tag) : 0.0f;
        require(std::fabs(output.whiteOwnerMap[y*19+x]-expected)<1e-7f,
          "ownership value/sign/mask differs from the real request");
      }
    }
  }
}

void runScenario(const std::string& mode,int capacity,int count,bool metadata=false,bool nhwc=true,bool eligible=false) {
  std::map<std::string,std::string> values;
  if(mode!="default") values["nnBatchAwareDispatch"]=mode;
  if(eligible) values["batchingFakeEligibleDevice"]="0";
  ConfigParser cfg(values);
  Logger logger(nullptr,false,false,false);
  NNEvaluator eval("cpu-test",metadata ? "metadata" : "plain","",&logger,
    capacity,19,19,false,nhwc,-1,0,false,"",enabled_t::False,1,{0},"cpu-batching",false,0,cfg);
  require(!eval.isNeuralNetLess(),"test accidentally bypassed real backend path");
  eval.spawnServerThreads();
  const bool pad=mode=="true" || (eligible && (mode=="auto" || mode=="default"));
  runWave(eval,capacity,count,pad,metadata,nhwc,true);
  eval.killServerThreads();
  require(BatchingFake::liveHandles()==0 && BatchingFake::liveBuffers()==0,"backend resources leaked after stop");
  if((mode=="true" || eligible) && count==3) {
    // Reuse the same evaluator after changing its per-device mapping and row cap.
    eval.setNumThreads({1});
    eval.setMaxRowsToSendPerBatch(3);
    eval.spawnServerThreads();
    runWave(eval,capacity,3,mode=="true",metadata,nhwc,true);
    eval.killServerThreads();
    require(BatchingFake::liveHandles()==0 && BatchingFake::liveBuffers()==0,"backend resources leaked after respawn");
    if(eligible) {
      eval.setNumThreads({0});
      eval.spawnServerThreads();
      runWave(eval,capacity,3,pad,metadata,nhwc,true);
      eval.killServerThreads();
      require(BatchingFake::liveHandles()==0 && BatchingFake::liveBuffers()==0,"backend resources leaked after eligible respawn");
    }
  }
}

#ifdef USE_CUDA_BACKEND
void runMixedEligibility(const std::string& mode,int capacity) {
  scenarios++;
  ConfigParser cfg(std::map<std::string,std::string>{
    {"nnBatchAwareDispatch",mode},{"batchingFakeEligibleDevice","0"}});
  Logger logger(nullptr,false,false,false);
  NNEvaluator eval("mixed-cpu-test","plain","",&logger,capacity,19,19,false,true,-1,0,
    false,"",enabled_t::False,2,{0,1},"mixed-batching",false,0,cfg);
  eval.spawnServerThreads();
  require(BatchingFake::expectedThreadMapping()==std::vector<int>({0,1}),"CUDA topology was not supplied before handles");
  BatchingFake::reset();
  BatchingFake::blockNextCall();
  Request first(0,false,true),second(1,false,true);
  auto firstDone=submit(eval,first,false);
  require(BatchingFake::waitUntilBlocked(3000),"mixed first worker did not start");
  auto secondDone=submit(eval,second,false);
  bool otherDeviceReady=secondDone.wait_for(std::chrono::seconds(3))==std::future_status::ready;
  BatchingFake::releaseCall();
  require(otherDeviceReady,"busy device incorrectly blocked the other device's worker");
  secondDone.get(); firstDone.get();
  require(waitFor([&] { return NNBatchingCpuTestAccess::ongoing(eval)==0; }),"mixed outstanding count did not drain");
  auto calls=BatchingFake::calls();
  require(calls.size()==2,"mixed workers did not each process one real request");
  require(calls[0].serverThread!=calls[1].serverThread,"blocked worker processed both requests");
  for(const auto& call:calls) {
    bool pad=mode=="true" || (mode=="auto" && call.serverThread==0);
    validateCall(call,1,pad ? capacity : 1,false,true);
  }
  require(eval.numRowsProcessed()==2 && eval.numBatchesProcessed()==2,"mixed statistics counted padding");
  eval.killServerThreads();
  eval.setNumThreads({1});
  eval.spawnServerThreads();
  require(BatchingFake::expectedThreadMapping()==std::vector<int>({1}),"respawn retained old topology");
  runWave(eval,capacity,3,mode=="true",false,true,true);
  eval.killServerThreads();
  require(BatchingFake::liveHandles()==0 && BatchingFake::liveBuffers()==0,"mixed backend resources leaked");
}
#endif
}

int main() {
#ifdef _WIN32
  SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX|SEM_NOOPENFILEERRORBOX);
  _set_abort_behavior(0,_WRITE_ABORT_MSG|_CALL_REPORTFAULT);
  _CrtSetReportMode(_CRT_ASSERT,_CRTDBG_MODE_FILE);
  _CrtSetReportFile(_CRT_ASSERT,_CRTDBG_FILE_STDERR);
#endif
  try {
    Board::initHash(); ScoreValue::initTables();
    for(const std::string mode: {"true","false","auto","default"})
      for(int capacity: {13,16})
        for(int count: {1,3,12,capacity}) runScenario(mode,capacity,count);
    for(int capacity: {13,16}) {
      runScenario("true",capacity,3,true,true);
      runScenario("true",capacity,12,true,false);
      runScenario("auto",capacity,3,true,true);
    }
#ifdef USE_CUDA_BACKEND
    for(int capacity: {13,16}) {
      for(int count: {1,3,12,capacity}) runScenario("auto",capacity,count,false,true,true);
      runScenario("default",capacity,3,false,true,true);
      runScenario("false",capacity,3,false,true,true);
      runMixedEligibility("auto",capacity);
      runMixedEligibility("false",capacity);
    }
#endif
    ScoreValue::freeTables();
    std::cout<<"PASS "<<scenarios<<" real NNEvaluator "
#ifdef USE_CUDA_BACKEND
      <<"fake-CUDA selection "
#endif
      <<"CPU batching scenarios, "<<checks
      <<" checks; no GPU initialized and no model weights loaded\n";
    return 0;
  }
  catch(const std::exception& e) {
    BatchingFake::releaseCall();
    std::cerr<<"FAIL: "<<e.what()<<'\n';
    return 1;
  }
}

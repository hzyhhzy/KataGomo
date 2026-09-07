// Standalone REAL CUDA NNEvaluator queue/output regression. Not an engine target.
// Build with batching_cuda_test.py from an already-built CUDA Ninja tree.
#include "../nneval.h"
#include "../../core/rand.h"
#include "../../external/nlohmann_json/json.hpp"
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <thread>
#ifdef _WIN32
#include <windows.h>
#include <crtdbg.h>
#endif
#ifndef USE_CUDA_BACKEND
#error This integration test requires the real CUDA backend, not the dummy backend.
#endif

// Read-only observation through the existing test friend. No injected outputs,
// fake backend, debugger, altered inference call, or production test switch.
struct NNBatchingCpuTestAccess {
  static size_t queued(NNEvaluator& e) { return e.queryQueue.size(); }
  static int ongoing(NNEvaluator& e) {
    std::lock_guard<std::mutex> lock(e.bufferMutex);
    return e.numOngoingEvals;
  }
  static bool enabled(NNEvaluator& e) {
    std::lock_guard<std::mutex> lock(e.bufferMutex);
    return e.anyBatchAwareDispatch;
  }
};

namespace {
using json = nlohmann::json;
void require(bool ok,const std::string& message) {
  if(!ok) throw std::runtime_error(message);
}
template<class Predicate> bool waitFor(Predicate predicate,int seconds=10) {
  const auto end=std::chrono::steady_clock::now()+std::chrono::seconds(seconds);
  while(!predicate()) {
    if(std::chrono::steady_clock::now()>=end) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}
// A staged request can be waiting for a worker when a startup failure occurs.
// Exit immediately on that failure instead of joining a blocked async future.
[[noreturn]] void stagingFailure(const std::string& message) {
  std::cerr << "CUDA queue integration failure: " << message << std::endl;
  std::_Exit(2);
}

struct Request {
  Board board;
  BoardHistory history;
  Player nextPlayer;
  MiscNNInputParams params;
  NNResultBuf buffer;
  explicit Request(int index,bool partial)
    : board(partial ? (index%3==0 ? 13 : index%3==1 ? 9 : 19) : 19,
            partial ? (index%3==0 ? 9 : index%3==1 ? 9 : 19) : 19),
      history(board,P_BLACK,Rules::getTrompTaylorish(),0,BoardHistoryModes(false,false)),
      nextPlayer(P_BLACK) {
    Rand rand("b11-real-batching-row-"+std::to_string(index));
    for(int turn=0;turn<12+index*2+index%2;turn++) {
      bool moved=false;
      for(int attempt=0;attempt<1000 && !moved;attempt++) {
        Loc loc=Location::getLoc(rand.nextUInt(board.x_size),rand.nextUInt(board.y_size),board.x_size);
        if(history.isLegal(board,loc,nextPlayer)) {
          history.makeBoardMoveAssumeLegal(board,loc,nextPlayer,nullptr);
          nextPlayer=getOpp(nextPlayer);
          moved=true;
        }
      }
      require(moved,"could not generate a legal deterministic board");
    }
    params.symmetry=index%8;
    params.policyOptimism=0.0;
  }
};

struct Wave {
  std::vector<std::shared_ptr<NNOutput>> outputs;
  json evidence;
};

Wave runWave(NNEvaluator& eval,int capacity,int count,bool expectPadding,
             bool partial,int gpu,std::ostringstream& logs) {
  // Open the ordinary query queue with zero workers, then fill it using only
  // public evaluate(). Changing worker mappings is supported between generations.
  // Starting one real worker after N enqueues gives exactly one real N-row batch.
  eval.setNumThreads({});
  eval.spawnServerThreads();
  eval.clearStats();
  logs.str(""); logs.clear();
  std::vector<std::unique_ptr<Request>> requests;
  std::vector<std::future<void>> futures;
  for(int row=0;row<count;row++) {
    requests.emplace_back(new Request(row,partial));
    Request* request=requests.back().get();
    futures.push_back(std::async(std::launch::async,[&eval,request] {
      eval.evaluate(request->board,request->history,request->nextPlayer,
                    request->params,request->buffer,true,true);
    }));
    if(!waitFor([&] { return NNBatchingCpuTestAccess::queued(eval)==(size_t)(row+1); }))
      stagingFailure("request did not enter the queue before worker startup");
  }
  try {
    eval.setNumThreads({gpu});
    eval.spawnServerThreads();
  }
  catch(const std::exception& error) { stagingFailure(error.what()); }
  // Check only after serving/draining: failed assertions must not strand clients.
  const bool enabled=NNBatchingCpuTestAccess::enabled(eval);
  for(auto& future:futures) {
    if(future.wait_for(std::chrono::seconds(30))!=std::future_status::ready)
      stagingFailure("real CUDA evaluate() did not return within 30 seconds");
    try { future.get(); }
    catch(const std::exception& error) { stagingFailure(error.what()); }
  }
  if(!waitFor([&] { return NNBatchingCpuTestAccess::ongoing(eval)==0; }))
    stagingFailure("real evaluations did not drain");
  eval.killServerThreads();
  require(enabled==expectPadding,"Auto did not select the prepared B11 lane (or false unexpectedly enabled padding)");
  require(eval.numRowsProcessed()==(uint64_t)count,"padding was counted as real rows");
  require(eval.numBatchesProcessed()==1,"staged requests were not processed as exactly one batch");
  require(eval.numCacheHits()==0,"cache unexpectedly supplied an output");
  require(eval.averageProcessedBatchSize()==count,"average batch statistics included padding");
  const int padded=expectPadding ? capacity-count : 0;
  const std::string marker="processed "+std::to_string(count)+" rows 1 batches, "+
    std::to_string(padded)+" padding rows";
  require(logs.str().find(marker)!=std::string::npos,"production worker padding count did not match the requested physical batch");
  Wave wave;
  for(auto& request:requests) {
    require(request->buffer.hasResult && request->buffer.result!=nullptr,"missing real result");
    require(request->buffer.result->whiteOwnerMap!=nullptr,"missing requested ownership map");
    wave.outputs.push_back(request->buffer.result);
  }
  wave.evidence={{"dispatch_enabled",enabled},{"real_rows",eval.numRowsProcessed()},
    {"batches",eval.numBatchesProcessed()},{"padding_rows",padded},
    {"physical_rows",count+padded},{"cache_hits",eval.numCacheHits()},
    {"production_padding_log_verified",true}};
  return wave;
}

struct Metric {
  double atol,rtol,maxAbs=0.0,sumSquared=0.0;
  size_t n=0,failed=0;
  Metric(double a,double r=0.0):atol(a),rtol(r) {}
  void add(double a,double b) {
    require(std::isfinite(a) && std::isfinite(b),"non-finite postprocessed output");
    double d=std::abs(a-b);
    maxAbs=std::max(maxAbs,d); sumSquared+=d*d; n++;
    if(d>atol+rtol*std::max(std::abs(a),std::abs(b))) failed++;
  }
  json report() const {
    return {{"max_abs",maxAbs},{"rmse",std::sqrt(sumSquared/std::max(size_t(1),n))},
      {"samples",n},{"failed_samples",failed},{"atol",atol},{"rtol",rtol}};
  }
};

json compare(const Wave& baseline,const Wave& padded,double probabilityAtol,double ownerAtol) {
  Metric policy(probabilityAtol),value(probabilityAtol),owner(ownerAtol);
  Metric score(0.5,0.01),auxiliary(1.0,0.02);
  for(size_t row=0;row<baseline.outputs.size();row++) {
    const NNOutput& a=*baseline.outputs[row]; const NNOutput& b=*padded.outputs[row];
    require(a.nnXLen==19 && a.nnYLen==19 && b.nnXLen==19 && b.nnYLen==19,"output dimensions changed");
    require(a.nnHash==b.nnHash,"different position/input hash delivered to a real request");
    for(int pos=0;pos<NNPos::getPolicySize(19,19);pos++) {
      require((a.policyProbs[pos]<0)==(b.policyProbs[pos]<0),"legal policy mask changed");
      policy.add(a.policyProbs[pos],b.policyProbs[pos]);
    }
    value.add(a.whiteWinProb,b.whiteWinProb);
    value.add(a.whiteLossProb,b.whiteLossProb);
    value.add(a.whiteNoResultProb,b.whiteNoResultProb);
    score.add(a.whiteScoreMean,b.whiteScoreMean);
    score.add(a.whiteLead,b.whiteLead);
    // Compare the second moment as score standard deviation, in points rather
    // than squared points; mean and standard deviation determine both moments.
    require(std::isfinite(a.whiteScoreMeanSq) && std::isfinite(b.whiteScoreMeanSq),"non-finite score second moment");
    score.add(std::sqrt(std::max(0.0,(double)a.whiteScoreMeanSq-a.whiteScoreMean*a.whiteScoreMean)),
              std::sqrt(std::max(0.0,(double)b.whiteScoreMeanSq-b.whiteScoreMean*b.whiteScoreMean)));
    auxiliary.add(a.varTimeLeft,b.varTimeLeft);
    auxiliary.add(a.shorttermWinlossError,b.shorttermWinlossError);
    auxiliary.add(a.shorttermScoreError,b.shorttermScoreError);
    for(int pos=0;pos<19*19;pos++) owner.add(a.whiteOwnerMap[pos],b.whiteOwnerMap[pos]);
  }
  return {{"pass",policy.failed+value.failed+owner.failed+score.failed+auxiliary.failed==0},
    {"policy",policy.report()},{"value",value.report()},{"ownership",owner.report()},
    {"score_points",score.report()},{"auxiliary",auxiliary.report()}};
}
}

int main(int argc,char** argv) {
#ifdef _WIN32
  SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX|SEM_NOOPENFILEERRORBOX);
  _set_abort_behavior(0,_WRITE_ABORT_MSG|_CALL_REPORTFAULT);
  _CrtSetReportMode(_CRT_ASSERT,_CRTDBG_MODE_FILE);
  _CrtSetReportFile(_CRT_ASSERT,_CRTDBG_FILE_STDERR);
#endif
  try {
    if(argc<3 || argc>9) {
      std::cerr << "Usage: b11_batching_cuda_test MODEL CONFIG [CAPACITY=0] [ACTUAL=0] "
                   "[CASE=all|exact|fullmask|partialmask] [GPU=0] [PROB_ATOL=0.005] [OWNER_ATOL=0.02]\n";
      return 2;
    }
    int selectedCapacity=argc>3 ? std::stoi(argv[3]) : 0;
    int selectedActual=argc>4 ? std::stoi(argv[4]) : 0;
    std::string selectedCase=argc>5 ? argv[5] : "all";
    int gpu=argc>6 ? std::stoi(argv[6]) : 0;
    double probabilityAtol=argc>7 ? std::stod(argv[7]) : 0.005;
    double ownerAtol=argc>8 ? std::stod(argv[8]) : 0.02;
    require(selectedCapacity==0 || selectedCapacity==13 || selectedCapacity==16,"capacity must be 0, 13, or 16");
    require(selectedActual>=0 && selectedActual<=16,"actual batch must be between 0 and 16");
    require(selectedCase=="all" || selectedCase=="exact" || selectedCase=="fullmask" || selectedCase=="partialmask","unknown board/mask case");
    require(gpu>=0 && std::isfinite(probabilityAtol) && probabilityAtol>0 &&
            std::isfinite(ownerAtol) && ownerAtol>0,"invalid GPU index or tolerances");
    Board::initHash(); ScoreValue::initTables(); NeuralNet::globalInitialize();
    ConfigParser baseConfig(argv[2]);
    bool passed=true; int scenarios=0;
    for(int capacity:{13,16}) {
      if(selectedCapacity!=0 && selectedCapacity!=capacity) continue;
      for(const std::string boardCase:{"exact","fullmask","partialmask"}) {
        if(selectedCase!="all" && selectedCase!=boardCase) continue;
        std::vector<int> counts=capacity==13 ? std::vector<int>{1,3,12,13} : std::vector<int>{1,3,12,13,16};
        if(selectedActual!=0) counts=selectedActual<=capacity ? std::vector<int>{selectedActual} : std::vector<int>{};
        if(counts.empty()) continue;
        std::vector<Wave> baseline;
        for(const std::string mode:{"false","auto"}) {
          ConfigParser config(baseConfig); config.overrideKey("nnBatchAwareDispatch",mode);
          std::ostringstream logs; Logger logger(nullptr,false,true,false,false); logger.addOStream(logs,false);
          NNEvaluator eval("b11-real-batching-test",argv[1],"",&logger,capacity,19,19,
            boardCase=="exact",true,-1,0,false,"",enabled_t::True,1,{gpu},
            "b11-real-batching-fixed",false,0,config);
          for(size_t i=0;i<counts.size();i++) {
            std::cerr << "Real CUDA queue case capacity=" << capacity << " actual=" << counts[i]
                      << " case=" << boardCase << " mode=" << mode << std::endl;
            Wave wave=runWave(eval,capacity,counts[i],mode=="auto",boardCase=="partialmask",gpu,logs);
            if(mode=="false") baseline.push_back(std::move(wave));
            else {
              json result=compare(baseline[i],wave,probabilityAtol,ownerAtol);
              passed=passed && result.at("pass").get<bool>(); scenarios++;
              result["capacity"]=capacity; result["actual"]=counts[i]; result["case"]=boardCase;
              result["baseline"]=baseline[i].evidence; result["auto"]=wave.evidence;
              result["type"]="case"; std::cout << result.dump() << std::endl;
            }
          }
        }
      }
    }
    require(scenarios>0,"selection did not contain any test cases");
    NeuralNet::globalCleanup();
    std::cout << json({{"type","summary"},{"pass",passed},{"scenarios",scenarios},
      {"real_cuda_backend",true},{"fixed_symmetry",true},{"cache_disabled",true},
      {"performance_measurement",false}}).dump() << std::endl;
    return passed ? 0 : 1;
  }
  catch(const std::exception& error) {
    std::cerr << "CUDA queue integration failure: " << error.what() << std::endl;
    return 2;
  }
}

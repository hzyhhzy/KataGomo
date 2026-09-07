// Identical measurement harness for official, experimental and doomoooo CUDA backends.
// Compile in place of cudabackend.cpp and main.cpp, linking the other engine objects.
#include "neuralnet/cudabackend.cpp"
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <thread>

static void forwardOnly(ComputeHandle* h,int batch) {
  Buffers* b=h->buffers.get();
#ifdef DOOM_REFERENCE
  h->apply(
#else
  h->model->apply(
#endif
    h->cudaHandles.get(),h->scratch.get(),batch,h->requireExactNNLen
#ifdef USE_B11_FA4
      || h->cudaHandles->b11FullMask
#endif
    ,
    b->inputBuf,b->inputGlobalBuf,b->inputMetaBuf,b->policyPassBuf,b->policyBuf,
    b->valueBuf,b->scoreValueBuf,b->ownershipBuf,b->workspaceBuf,b->workspaceBytes);
}
static void dumpRaw(ComputeHandle* h,int batch,const ModelDesc& d,const std::string& path) {
  auto b=h->buffers.get();
  std::ofstream f(path,std::ios::binary);
  const int xy=h->nnXLen*h->nnYLen;
  auto dump=[&](const void* p,size_t count,bool fp16){
    std::vector<float> v(count);
    if(fp16){std::vector<half> tmp(count);CUDA_ERR("dump",cudaMemcpy(tmp.data(),p,count*2,cudaMemcpyDeviceToHost));
      for(size_t i=0;i<count;i++)v[i]=__half2float(tmp[i]);}
    else CUDA_ERR("dump",cudaMemcpy(v.data(),p,count*4,cudaMemcpyDeviceToHost));
    f.write(reinterpret_cast<const char*>(v.data()),count*4);
  };
  dump(b->policyPassBuf,(size_t)batch*d.numPolicyChannels,false);
  dump(b->policyBuf,(size_t)batch*xy*d.numPolicyChannels,false);
  dump(b->valueBuf,(size_t)batch*d.numValueChannels,false);
  dump(b->scoreValueBuf,(size_t)batch*d.numScoreValueChannels,false);
  // ValueHead converts ownership back to FP32 even in an FP16 engine.
  dump(b->ownershipBuf,(size_t)batch*xy,false);
  if(!f)throw std::runtime_error("raw output write failed");
}
int main(int argc,char**argv) {
  if(argc<9){std::cerr<<"model config batch exact iterations streams output-prefix corpus [nnlen=19] [activebatch=capacity] [FP32=0]\n";return 2;}
  try {
    const int capacity=std::stoi(argv[3]),exact=std::stoi(argv[4]),iterations=std::stoi(argv[5]),streams=std::stoi(argv[6]);
    const std::string prefix=argv[7],corpusPath=argv[8];
    const int len=argc>9?std::stoi(argv[9]):19,batch=argc>10?std::stoi(argv[10]):capacity;
    const bool fp32=argc>11 && std::stoi(argv[11])!=0;
    if(batch<1 || batch>capacity || streams<1 || iterations<1 ||
       len<1 || len>Board::MAX_LEN || (exact!=0 && exact!=1))
      throw std::runtime_error("invalid arguments");
    Board::initHash();ScoreValue::initTables();NeuralNet::globalInitialize();
    ConfigParser cfg(argv[2]);Logger logger(nullptr,false,true,false);
    auto model=NeuralNet::loadModelFile(argv[1],"");
    const auto& d=NeuralNet::getModelDesc(model);
    if(d.numInputChannels!=22 || d.numInputGlobalChannels!=19 || d.numInputMetaChannels!=0)
      throw std::runtime_error("benchmark corpus requires input version 7: 22 spatial, 19 global, no metadata");
    int stride=len*len*d.numInputChannels+d.numInputGlobalChannels;
    std::vector<float> corpus;
    std::ifstream cf(corpusPath,std::ios::binary|std::ios::ate);
    if(cf){auto bytes=cf.tellg();if(bytes<=0 || bytes%(stride*4)!=0)throw std::runtime_error("bad corpus length");
      corpus.resize(size_t(bytes)/4);cf.seekg(0);cf.read(reinterpret_cast<char*>(corpus.data()),bytes);
      if(!cf || cf.gcount()!=bytes)throw std::runtime_error("corpus read failed");}
    else {
      // Generate once with official feature code; all binaries subsequently load these exact bytes.
      Rand rand("official-go-b11c768-5090-v1");corpus.resize((size_t)256*stride);
      for(int row=0;row<256;row++) {
        const int bsize=exact?len:(row%3==0?len:row%3==1?std::min(13,len):std::min(9,len));
        Board board(bsize,bsize);
#ifdef DOOM_REFERENCE
        BoardHistory hist(board,P_BLACK,Rules::getTrompTaylorish(),0,false);
#else
        BoardHistory hist(board,P_BLACK,Rules::getTrompTaylorish(),0,BoardHistoryModes(false,false));
#endif
        Player pla=P_BLACK;
        for(int step=0;step<row%160;step++) {
          for(int tries=0;tries<200;tries++) {
            Loc loc=Location::getLoc(rand.nextUInt(bsize),rand.nextUInt(bsize),bsize);
            if(hist.isLegal(board,loc,pla)){hist.makeBoardMoveAssumeLegal(board,loc,pla,nullptr);pla=getOpp(pla);break;}
          }
        }
        NNInputs::fillRowV7(board,hist,pla,MiscNNInputParams(),len,len,true,
          corpus.data()+(size_t)row*stride,corpus.data()+(size_t)row*stride+len*len*d.numInputChannels);
      }
      std::ofstream out(corpusPath,std::ios::binary);out.write(reinterpret_cast<const char*>(corpus.data()),corpus.size()*4);
      if(!out)throw std::runtime_error("corpus write failed");
    }
    auto context=NeuralNet::createComputeContext({0},&logger,len,len,"",fp32?enabled_t::False:enabled_t::True,model,cfg);
    std::atomic<int> ready(0),done(0);std::atomic<bool> go(false),failed(false);
    std::mutex errorMutex;std::exception_ptr error;
    std::vector<double> durations(streams),gpuMs(streams);std::vector<std::thread> workers;
    for(int s=0;s<streams;s++)workers.emplace_back([&,s]{try{
#ifdef DOOM_REFERENCE
      auto ownedStream=NeuralNet::createComputeStream(0);
      auto h=NeuralNet::createComputeHandle(context,model,&logger,capacity,exact,true,0,s,ownedStream);
#else
      auto h=NeuralNet::createComputeHandle(context,model,&logger,capacity,exact,true,0,s);
#endif
      auto inputs=NeuralNet::createInputBuffers(model,capacity,len,len);
      std::vector<std::unique_ptr<NNResultBuf>> owned;
      std::vector<NNResultBuf*> rows;std::vector<std::unique_ptr<NNOutput>> oo;std::vector<NNOutput*> outputs;
      for(int r=0;r<batch;r++){
        owned.emplace_back(new NNResultBuf);auto b=owned.back().get();
        const float* src=corpus.data()+((s*batch+r)%(corpus.size()/stride))*stride;
        b->rowSpatialBuf.assign(src,src+len*len*d.numInputChannels);b->rowGlobalBuf.assign(src+len*len*d.numInputChannels,src+stride);
        b->hasRowMeta=false;b->symmetry=0;b->policyOptimism=0;b->boardXSizeForServer=len;b->boardYSizeForServer=len;
        rows.push_back(b);oo.emplace_back(new NNOutput);oo.back()->nnXLen=len;oo.back()->nnYLen=len;oo.back()->whiteOwnerMap=nullptr;
        outputs.push_back(oo.back().get());
      }
      NeuralNet::setIsWarmup(h,true);NeuralNet::getOutput(h,inputs,batch,rows.data(),outputs);
      for(int w=0;w<20;w++)forwardOnly(h,batch);
      CUDA_ERR("warmup",cudaStreamSynchronize(h->cudaHandles->stream));NeuralNet::setIsWarmup(h,false);
      cudaGraph_t graph=nullptr;cudaGraphExec_t graphExec=nullptr;
      if(std::getenv("DIRECT_GRAPH")) {
        if(std::getenv("DIRECT_E2E"))throw std::runtime_error("graph harness is device-only");
        CUDA_ERR("capture",cudaStreamBeginCapture(h->cudaHandles->stream,cudaStreamCaptureModeThreadLocal));
        forwardOnly(h,batch);
        CUDA_ERR("capture",cudaStreamEndCapture(h->cudaHandles->stream,&graph));
        CUDA_ERR("graph",cudaGraphInstantiate(&graphExec,graph,0));
        for(int w=0;w<20;w++)CUDA_ERR("graph warmup",cudaGraphLaunch(graphExec,h->cudaHandles->stream));
        CUDA_ERR("graph warmup",cudaStreamSynchronize(h->cudaHandles->stream));
      }
      cudaEvent_t begin,end;CUDA_ERR("event",cudaEventCreate(&begin));CUDA_ERR("event",cudaEventCreate(&end));
      ready++;while(!go.load() && !failed.load())std::this_thread::yield();
      auto start=std::chrono::steady_clock::now();
      CUDA_ERR("event",cudaEventRecord(begin,h->cudaHandles->stream));
      for(int i=0;i<iterations;i++){
        if(graphExec) { CUDA_ERR("graph",cudaGraphLaunch(graphExec,h->cudaHandles->stream)); }
        else if(std::getenv("DIRECT_E2E"))NeuralNet::getOutput(h,inputs,batch,rows.data(),outputs);
        else forwardOnly(h,batch);
      }
      CUDA_ERR("event",cudaEventRecord(end,h->cudaHandles->stream));CUDA_ERR("sync",cudaStreamSynchronize(h->cudaHandles->stream));
      durations[s]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
      float ms;CUDA_ERR("event",cudaEventElapsedTime(&ms,begin,end));gpuMs[s]=ms;
      done++;while(done.load()<streams && !failed.load())std::this_thread::yield();
      // All device work has completed; output is not part of the measured region.
      if(prefix!="-")dumpRaw(h,batch,d,prefix+"-s"+std::to_string(s)+".raw");
      if(std::getenv("DIRECT_REPLAY")) {
        const int replayRows=std::stoi(std::getenv("DIRECT_REPLAY"));
        if(streams!=1 || prefix=="-" || replayRows%batch!=0)
          throw std::runtime_error("replay needs one stream, output prefix and whole batches");
        for(int offset=0;offset<replayRows;offset+=batch) {
          for(int r=0;r<batch;r++) {
            const float* src=corpus.data()+((offset+r)%(corpus.size()/stride))*stride;
            std::copy(src,src+len*len*d.numInputChannels,owned[r]->rowSpatialBuf.begin());
            std::copy(src+len*len*d.numInputChannels,src+stride,owned[r]->rowGlobalBuf.begin());
          }
          NeuralNet::getOutput(h,inputs,batch,rows.data(),outputs);
          dumpRaw(h,batch,d,prefix+"-r"+std::to_string(offset)+".raw");
        }
      }
      if(graphExec)cudaGraphExecDestroy(graphExec);
      if(graph)cudaGraphDestroy(graph);
      cudaEventDestroy(begin);cudaEventDestroy(end);NeuralNet::freeInputBuffers(inputs);NeuralNet::freeComputeHandle(h);
#ifdef DOOM_REFERENCE
      NeuralNet::freeComputeStream(ownedStream);
#endif
    }catch(...){failed=true;std::lock_guard<std::mutex> lock(errorMutex);if(!error)error=std::current_exception();}});
    while(ready.load()<streams && !failed.load()) { std::this_thread::yield(); }
    go=true;
    for(auto& t:workers) { t.join(); }
    if(error)std::rethrow_exception(error);
    double wall=*std::max_element(durations.begin(),durations.end());
    std::cout<<std::setprecision(10)<<"{\"batch\":"<<batch<<",\"capacity\":"<<capacity<<",\"streams\":"<<streams
      <<",\"exact\":"<<(exact?"true":"false")<<",\"iterations\":"<<iterations<<",\"wall_seconds\":"<<wall
      <<",\"nnEvalPerSec\":"<<double(batch)*streams*iterations/wall<<",\"gpu_ms\":[";
    for(int i=0;i<streams;i++) { std::cout<<(i?",":"")<<gpuMs[i]; }
    std::cout<<"]}\n";
    NeuralNet::freeComputeContext(context);NeuralNet::freeLoadedModel(model);NeuralNet::globalCleanup();return 0;
  }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}

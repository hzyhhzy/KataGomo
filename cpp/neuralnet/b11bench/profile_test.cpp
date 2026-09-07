// CPU-only integration test of the production predicate using real descriptors.
// No CUDA includes, symbols, device initialization, or model-file writes.
#include "neuralnet/cudab11profile.h"
#include <climits>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
int checks = 0;

void require(bool value,const std::string& label) {
  checks++;
  if(!value) throw std::runtime_error("FAILED: " + label);
}

template<class T>
void rejectMutation(ModelDesc& d,T& field,T replacement,const std::string& label) {
  const T saved = field;
  field = replacement;
  const bool accepted = isOfficialB11C768(d);
  field = saved;
  require(!accepted,label);
  require(isOfficialB11C768(d),label + " restored");
}

void rejectNull(ModelDesc& d,unique_ptr_void& field,const std::string& label) {
  void* saved = field.release();
  const bool accepted = isOfficialB11C768(d);
  field.reset(saved);
  require(!accepted,label);
  require(isOfficialB11C768(d),label + " restored");
}

void testModel(ModelDesc& d) {
  require(isOfficialB11C768(d),"real official b11 model accepted");
  rejectMutation(d,d.modelVersion,16,"modelVersion");
  rejectMutation(d,d.numInputChannels,23,"input channels");
  rejectMutation(d,d.numInputGlobalChannels,20,"global channels");
  rejectMutation(d,d.numInputMetaChannels,1,"metadata channels");
  rejectMutation(d,d.trunk.trunkNumChannels,767,"trunk channels");
  rejectMutation(d,d.trunk.numBlocks,10,"trunk declared blocks");
  auto last = std::move(d.trunk.blocks.back());
  d.trunk.blocks.pop_back();
  require(!isOfficialB11C768(d),"trunk actual blocks");
  d.trunk.blocks.emplace_back(std::move(last));
  require(isOfficialB11C768(d),"trunk actual blocks restored");
  for(std::size_t oi=0;oi<d.trunk.blocks.size();oi++) {
    auto& outer = d.trunk.blocks[oi];
    const std::string tag = "outer " + std::to_string(oi) + ": ";
    rejectMutation(d,outer.first,ORDINARY_BLOCK_KIND,tag+"kind");
    rejectNull(d,outer.second,tag+"null pointer");
    auto& n = *static_cast<NestedBottleneckResidualBlockDesc*>(outer.second.get());
    rejectMutation(d,n.numBlocks,5,tag+"declared inner blocks");
    auto tail = std::move(n.blocks.back());
    n.blocks.pop_back();
    require(!isOfficialB11C768(d),tag+"actual inner blocks");
    n.blocks.emplace_back(std::move(tail));
    require(isOfficialB11C768(d),tag+"actual inner blocks restored");
    rejectMutation(d,n.preBN.numChannels,767,tag+"preBN channels");
    rejectMutation(d,n.postBN.numChannels,383,tag+"postBN channels");
    for(ConvLayerDesc* c : {&n.preConv,&n.postConv}) {
      rejectMutation(d,c->convXSize,3,tag+"conv X");
      rejectMutation(d,c->convYSize,3,tag+"conv Y");
      rejectMutation(d,c->dilationX,2,tag+"dilation X");
      rejectMutation(d,c->dilationY,2,tag+"dilation Y");
      rejectMutation(d,c->inChannels,c->inChannels+1,tag+"conv input");
      rejectMutation(d,c->outChannels,c->outChannels+1,tag+"conv output");
    }
    for(std::size_t ii=0;ii<n.blocks.size();ii++) {
      auto& item = n.blocks[ii];
      const std::string itag = tag + "inner " + std::to_string(ii) + ": ";
      rejectNull(d,item.second,itag+"null pointer");
      rejectMutation(d,item.first,ORDINARY_BLOCK_KIND,itag+"kind");
      if(ii%2 == 0) {
        auto& a = *static_cast<TransformerAttentionDesc*>(item.second.get());
        rejectMutation(d,a.numHeads,11,itag+"heads");
        rejectMutation(d,a.numKVHeads,11,itag+"KV heads");
        rejectMutation(d,a.qHeadDim,31,itag+"Q head dim");
        rejectMutation(d,a.vHeadDim,31,itag+"V head dim");
        rejectMutation(d,a.useRope,false,itag+"RoPE");
        rejectMutation(d,a.learnableRope,false,itag+"learned RoPE");
        rejectMutation(d,a.ropeNumKVHeads,11,itag+"RoPE KV heads");
        rejectMutation(d,a.ropeNumPairs,15,itag+"RoPE pairs");
        rejectMutation(d,a.preLN.numChannels,383,itag+"preLN channels");
        for(MatMulLayerDesc* m : {&a.qProj,&a.kProj,&a.vProj,&a.outProj}) {
          rejectMutation(d,m->inChannels,m->inChannels+1,itag+"projection input");
          rejectMutation(d,m->outChannels,m->outChannels+1,itag+"projection output");
        }
      }
      else {
        auto& f = *static_cast<TransformerFFNDesc*>(item.second.get());
        rejectMutation(d,f.numChannels,383,itag+"FFN channels");
        rejectMutation(d,f.ffnChannels,1151,itag+"FFN hidden channels");
        rejectMutation(d,f.useSwiGLU,false,itag+"SwiGLU");
        rejectMutation(d,f.preLN.numChannels,383,itag+"preLN channels");
        for(MatMulLayerDesc* m : {&f.linear1,&f.linearGate,&f.linear2}) {
          rejectMutation(d,m->inChannels,m->inChannels+1,itag+"projection input");
          rejectMutation(d,m->outChannels,m->outChannels+1,itag+"projection output");
        }
      }
    }
  }
  const std::string savedName = d.name;
  d.name = "independent-of-checkpoint-name";
  require(isOfficialB11C768(d),"checkpoint name is not part of profile");
  d.name = savedName;
  d.releaseWeights();
  require(isOfficialB11C768(d),"released-weight descriptor accepted");
}

void testHardwareAndBatch() {
  const char* name = "NVIDIA GeForce RTX 5090";
  require(isB11CudaHardwareLayoutEligible(name,12,0,true,true,19,19),"exact hardware/layout");
  for(const char* wrong : {"NVIDIA GeForce RTX 4090","NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
                          "NVIDIA GeForce RTX 5090D","NVIDIA GeForce RTX 5090 ",""})
    require(!isB11CudaHardwareLayoutEligible(wrong,12,0,true,true,19,19),"different GPU name");
  require(!isB11CudaHardwareLayoutEligible(nullptr,12,0,true,true,19,19),"null GPU name");
  for(int major : {8,9,11,13})
    require(!isB11CudaHardwareLayoutEligible(name,major,0,true,true,19,19),"different major");
  require(!isB11CudaHardwareLayoutEligible(name,12,1,true,true,19,19),"different minor");
  require(!isB11CudaHardwareLayoutEligible(name,12,0,false,true,19,19),"FP32");
  require(!isB11CudaHardwareLayoutEligible(name,12,0,true,false,19,19),"NCHW");
  for(int len : {0,1,9,13,17,20,25}) {
    require(!isB11CudaHardwareLayoutEligible(name,12,0,true,true,len,19),"different board X");
    require(!isB11CudaHardwareLayoutEligible(name,12,0,true,true,19,len),"different board Y");
  }
  const int screeningSet[] = {8,16,32}; // Test data, NOT a production batch policy.
  for(int batch=-2;batch<=100;batch++) {
    require(isB11SupportedBatch(batch,screeningSet) == (batch==8 || batch==16 || batch==32),"batch list");
    require(isB11SupportedBatch(batch,1,96) == (batch>=1 && batch<=96),"batch range");
    require(isB11OptimizedBatch(batch) == (batch==13 || batch==16),"final optimized batch set");
  }
  require(!isB11OptimizedBatch(INT_MIN),"final batch rejects INT_MIN");
  require(!isB11OptimizedBatch(INT_MAX),"final batch rejects INT_MAX");
  require(!isB11SupportedBatch(16,nullptr,0),"empty/null batch list");
  require(!isB11SupportedBatch(16,nullptr,3),"null nonempty batch list");
  require(!isB11SupportedBatch(16,screeningSet,0),"zero-length batch list");
  require(!isB11SupportedBatch(16,0,96),"invalid range minimum");
  require(!isB11SupportedBatch(16,96,1),"reversed range");
  require(isB11SupportedBatch(INT_MAX,INT_MAX,INT_MAX),"range has no overflow arithmetic");
}

void testConcurrencyAndL2() {
  require(getB11ExpectedConcurrentGpuThreads({},0) == 0,"unknown topology");
  require(getB11ExpectedConcurrentGpuThreads({0},0) == 1,"one explicit thread is not three");
  require(getB11ExpectedConcurrentGpuThreads({0,0,0},0) == 3,"three threads on GPU 0");
  require(getB11ExpectedConcurrentGpuThreads({-1,-1,-1},0) == 3,"default CUDA device is GPU 0");
  require(getB11ExpectedConcurrentGpuThreads({-1,0,-1},0) == 3,"default and explicit GPU 0 aliases");
  require(getB11ExpectedConcurrentGpuThreads({-1,-1,-1},1) == 0,"default does not mean any GPU");
  require(getB11ExpectedConcurrentGpuThreads({0,1,2},0) == 1,"three devices are not three streams each");
  require(getB11ExpectedConcurrentGpuThreads({0,1,0,1,0},0) == 3,"multi-GPU count for GPU 0");
  require(getB11ExpectedConcurrentGpuThreads({0,1,0,1,0},1) == 2,"multi-GPU count for GPU 1");
  require(getB11ExpectedConcurrentGpuThreads({0,1,0,1,0},2) == 0,"unused device");
  require(getB11ExpectedConcurrentGpuThreads({2,2,2,1},2) == 3,"nonzero device index");
  require(getB11ExpectedConcurrentGpuThreads({0,0,0,-2},0) == 0,"invalid mapping rejected as a whole");
  require(getB11ExpectedConcurrentGpuThreads({INT_MIN,0},0) == 0,"INT_MIN mapping rejected");
  require(getB11ExpectedConcurrentGpuThreads({0,0,0},-1) == 0,"actual device must be resolved");
  require(getB11ExpectedConcurrentGpuThreads({0},INT_MIN) == 0,"invalid actual device");
  require(getB11ExpectedConcurrentGpuThreads(std::vector<int>(1024,0),0) == 1024,"full Setup thread-count range");
  require(getB11ExpectedConcurrentGpuThreads({0,1},0) == 1,"deduplicated inventory cannot imply concurrency");
  for(int batch=-2;batch<=100;batch++) {
    for(int count=-1;count<=17;count++) {
      require(isB11AutoL2Eligible(batch,count) == ((batch==13 || batch==16) && count==3),
        "automatic L2 has explicit batch and concurrency pair");
    }
  }
  require(!isB11AutoL2Eligible(13,INT_MAX),"L2 rejects huge count");
  require(!isB11AutoL2Eligible(16,INT_MIN),"L2 rejects invalid count");
  require(!isB11AutoL2Eligible(INT_MAX,3),"L2 rejects huge batch");
  require(!isB11AutoL2Eligible(INT_MIN,3),"L2 rejects invalid batch");
}
}

int main(int argc,char** argv) {
  try {
    if(argc < 2) throw std::runtime_error("usage: b11_profile_test B11_MODEL [DIFFERENT_VALID_MODEL ...]");
    testHardwareAndBatch();
    testConcurrencyAndL2();
    ModelDesc d;
    ModelDesc::loadFromFileMaybeGZipped(argv[1],d,"");
    std::cout << "MODEL " << d.name << " version=" << d.modelVersion << '\n';
    testModel(d);
    for(int i=2;i<argc;i++) {
      ModelDesc other;
      ModelDesc::loadFromFileMaybeGZipped(argv[i],other,"");
      require(!isOfficialB11C768(other),"different valid model rejected: " + other.name);
      std::cout << "OTHER " << other.name << " rejected\n";
    }
    std::cout << "PASS " << checks << " CPU-only profile checks; no GPU initialized\n";
    return 0;
  }
  catch(const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

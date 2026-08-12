#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/logger.h"
#include "../core/rand.h"
#include "../neuralnet/nneval.h"
#include "../program/setup.h"
#include "../command/commandline.h"
#include "../main.h"

#include <array>
#include <cstring>
#include <fstream>
#include <memory>

using namespace std;

namespace {

struct Corpus {
  uint32_t numRows;
  array<unsigned char,32> identity;
  vector<unsigned char> packedInput;
  vector<float> globalInput;
};

uint32_t readU32(ifstream& in) {
  unsigned char b[4];
  in.read((char*)b,4);
  if(!in)
    throw StringError("replaynn15: truncated corpus header");
  return (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
         ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

void writeU32(ofstream& out, uint32_t x) {
  unsigned char b[4] = {
    (unsigned char)(x & 0xff),
    (unsigned char)((x >> 8) & 0xff),
    (unsigned char)((x >> 16) & 0xff),
    (unsigned char)((x >> 24) & 0xff),
  };
  out.write((const char*)b,4);
}

void readExact(ifstream& in, void* dst, size_t bytes, const char* what) {
  in.read((char*)dst,(streamsize)bytes);
  if(!in)
    throw StringError(string("replaynn15: truncated ") + what);
}

void writeExact(ofstream& out, const void* src, size_t bytes) {
  out.write((const char*)src,(streamsize)bytes);
  if(!out)
    throw StringError("replaynn15: output write failed");
}

Corpus readCorpus(const string& path) {
  ifstream in(path,ios::binary);
  if(!in)
    throw StringError("replaynn15: could not open corpus " + path);
  char magic[8];
  readExact(in,magic,8,"corpus magic");
  if(memcmp(magic,"R15CORP1",8) != 0)
    throw StringError("replaynn15: bad corpus magic");
  Corpus c;
  c.numRows = readU32(in);
  const uint32_t posLen = readU32(in);
  const uint32_t binFeatures = readU32(in);
  const uint32_t globalFeatures = readU32(in);
  const uint32_t packedWidth = readU32(in);
  const uint32_t policyDim = readU32(in);
  const uint32_t globalTargetDim = readU32(in);
  if(c.numRows == 0 || posLen != 15 || binFeatures != 22 || globalFeatures != 39 ||
     packedWidth != 29 || policyDim != 226 || globalTargetDim != 64)
    throw StringError("replaynn15: corpus dimensions do not match 15x15 v102");
  readExact(in,c.identity.data(),c.identity.size(),"corpus identity");
  c.packedInput.resize((size_t)c.numRows * 22 * 29);
  c.globalInput.resize((size_t)c.numRows * 39);
  readExact(in,c.packedInput.data(),c.packedInput.size(),"packed spatial input");
  readExact(in,c.globalInput.data(),c.globalInput.size()*sizeof(float),"global input");

  // The remainder contains policy and global targets. The comparator reads
  // those fields directly; the inference command only needs to prove it used
  // the same identity-bound corpus and row order.
  const uint64_t targetBytes =
    (uint64_t)c.numRows * 2 * 226 * sizeof(int16_t) +
    (uint64_t)c.numRows * 64 * sizeof(float);
  in.seekg((streamoff)targetBytes,ios::cur);
  if(!in || in.peek() != EOF)
    throw StringError("replaynn15: corpus size mismatch");
  return c;
}

}

int MainCmds::replaynn15(const vector<string>& args) {
  Board::initHash();
  Rand seedRand;
  ConfigParser cfg;
  string modelFile;
  string corpusFile;
  string outputFile;
  int batchSize = 36;

  try {
    KataGoCommandLine cmd(
      "Replay an identity-bound full-15x15 corpus at an exact physical batch and dump raw heads."
    );
    cmd.addConfigFileArg(KataGoCommandLine::defaultGtpConfigFileName(),"gtp_example.cfg");
    cmd.addModelFileArg();
    TCLAP::ValueArg<string> corpusArg("","corpus","R15CORP1 corpus",true,"","FILE");
    TCLAP::ValueArg<string> outputArg("","output","R15OUT1 raw output",true,"","FILE");
    TCLAP::ValueArg<int> batchArg("","batch-size","Exact physical batch",false,36,"N");
    cmd.add(corpusArg);
    cmd.add(outputArg);
    cmd.add(batchArg);
    cmd.setShortUsageArgLimit();
    cmd.addOverrideConfigArg();
    cmd.parseArgs(args);
    modelFile = cmd.getModelFile();
    corpusFile = corpusArg.getValue();
    outputFile = outputArg.getValue();
    batchSize = batchArg.getValue();
    cmd.getConfig(cfg);
    if(batchSize <= 0)
      throw StringError("replaynn15: batch size must be positive");
  }
  catch(TCLAP::ArgException& e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Corpus corpus = readCorpus(corpusFile);
  Logger logger(NULL,true,false,false);
  logger.write("Version " + Version::getGitRevisionWithBackend());
  logger.write("replaynn15 model " + modelFile);
  logger.write("replaynn15 corpus " + corpusFile);
  logger.write("replaynn15 rows " + Global::uint64ToString(corpus.numRows) +
               " exact physical batch " + Global::intToString(batchSize));

  NNEvaluator* nnEval = NULL;
  ComputeHandle* handle = NULL;
  InputBuffers* inputBuffers = NULL;
  try {
    nnEval = Setup::initializeNNEvaluator(
      modelFile,modelFile,"",cfg,logger,seedRand,batchSize*64,batchSize,
      15,15,batchSize,true,false,Setup::SETUP_FOR_BENCHMARK
    );
    nnEval->killServerThreads();
    if(nnEval->getNNXLen() != 15 || nnEval->getNNYLen() != 15 ||
       !nnEval->getRequireExactNNLen())
      throw StringError("replaynn15: evaluator did not initialize exact 15x15");
    if(nnEval->getNumServerThreads() < 1)
      throw StringError("replaynn15: evaluator has no GPU thread");

    handle = NeuralNet::createComputeHandle(
      nnEval->getComputeContext(),nnEval->getLoadedModel(),&logger,batchSize,
      nnEval->getRequireExactNNLen(),nnEval->getInputsUseNHWC(),
      nnEval->getGpuIdxByServerThread(0),0,nnEval->getBackendNumThreads()
    );
    inputBuffers = NeuralNet::createInputBuffers(nnEval->getLoadedModel(),batchSize,15,15);
    const bool usingFP16 = NeuralNet::isUsingFP16(handle);
    const bool useNHWC = nnEval->getInputsUseNHWC();

    vector<unique_ptr<NNResultBuf>> ownedRows;
    vector<NNResultBuf*> rows;
    vector<unique_ptr<NNOutput>> ownedOutputs;
    vector<NNOutput*> outputs;
    ownedRows.reserve(batchSize);
    rows.reserve(batchSize);
    ownedOutputs.reserve(batchSize);
    outputs.reserve(batchSize);
    for(int b = 0; b < batchSize; b++) {
      ownedRows.push_back(make_unique<NNResultBuf>());
      NNResultBuf* row = ownedRows.back().get();
      row->rowSpatial = new float[22*225];
      row->rowSpatialSize = 22*225;
      row->rowGlobal = new float[39];
      row->rowGlobalSize = 39;
      row->symmetry = 0;
      rows.push_back(row);
      ownedOutputs.push_back(make_unique<NNOutput>());
      ownedOutputs.back()->nnXLen = 15;
      ownedOutputs.back()->nnYLen = 15;
      outputs.push_back(ownedOutputs.back().get());
    }
    vector<float> postPolicy((size_t)batchSize * NNPos::MAX_NN_POLICY_SIZE);
    vector<float> policy((size_t)corpus.numRows * 226);
    vector<float> value((size_t)corpus.numRows * 3);
    vector<float> misc((size_t)corpus.numRows * 6);
    vector<float> ownership((size_t)corpus.numRows * 225);

    for(uint32_t rowStart = 0; rowStart < corpus.numRows; rowStart += batchSize) {
      const int realBatch = (int)min<uint32_t>((uint32_t)batchSize,corpus.numRows-rowStart);
      for(int b = 0; b < batchSize; b++) {
        const uint32_t srcRow = rowStart + (uint32_t)(b % realBatch);
        const unsigned char* packed =
          corpus.packedInput.data() + (size_t)srcRow * 22 * 29;
        float* spatial = rows[b]->rowSpatial;
        for(int c = 0; c < 22; c++) {
          for(int p = 0; p < 225; p++) {
            const int bit = (packed[c*29 + p/8] >> (7-(p%8))) & 1;
            spatial[useNHWC ? p*22+c : c*225+p] = (float)bit;
          }
        }
        const float* global = corpus.globalInput.data() + (size_t)srcRow * 39;
        copy(global,global+39,rows[b]->rowGlobal);
      }
      NeuralNet::getOutput(
        handle,inputBuffers,batchSize,rows.data(),outputs,postPolicy.data()
      );
      RawNNOutputs raw;
      NeuralNet::getRawNNOutputs(inputBuffers,raw);
      if(raw.policyElts != 226 || raw.valueElts != 3 ||
         raw.miscElts != 6 || raw.ownershipElts != 225)
        throw StringError("replaynn15: raw head dimensions differ from v102 contract");
      for(int b = 0; b < realBatch; b++) {
        const uint32_t dstRow = rowStart + (uint32_t)b;
        copy(raw.policy + (size_t)b*226,raw.policy + (size_t)(b+1)*226,policy.data() + (size_t)dstRow*226);
        copy(raw.value + (size_t)b*3,raw.value + (size_t)(b+1)*3,value.data() + (size_t)dstRow*3);
        copy(raw.misc + (size_t)b*6,raw.misc + (size_t)(b+1)*6,misc.data() + (size_t)dstRow*6);
        copy(raw.ownership + (size_t)b*225,raw.ownership + (size_t)(b+1)*225,ownership.data() + (size_t)dstRow*225);
      }
    }

    ofstream out(outputFile,ios::binary|ios::trunc);
    if(!out)
      throw StringError("replaynn15: could not open output " + outputFile);
    const char magic[8] = {'R','1','5','O','U','T','1','\0'};
    writeExact(out,magic,8);
    const uint32_t header[10] = {
      corpus.numRows,(uint32_t)batchSize,15,226,3,6,225,
      usingFP16 ? 1u : 0u,useNHWC ? 1u : 0u,1u
    };
    for(uint32_t x : header)
      writeU32(out,x);
    writeExact(out,corpus.identity.data(),corpus.identity.size());
    writeExact(out,policy.data(),policy.size()*sizeof(float));
    writeExact(out,value.data(),value.size()*sizeof(float));
    writeExact(out,misc.data(),misc.size()*sizeof(float));
    writeExact(out,ownership.data(),ownership.size()*sizeof(float));
    out.close();
    logger.write("replaynn15 wrote " + outputFile +
                 " usingFP16=" + string(usingFP16 ? "true" : "false") +
                 " fixedBatchTailPadding=true");
  }
  catch(...) {
    if(inputBuffers != NULL) NeuralNet::freeInputBuffers(inputBuffers);
    if(handle != NULL) NeuralNet::freeComputeHandle(handle);
    delete nnEval;
    NeuralNet::globalCleanup();
    throw;
  }
  NeuralNet::freeInputBuffers(inputBuffers);
  NeuralNet::freeComputeHandle(handle);
  delete nnEval;
  NeuralNet::globalCleanup();
  return 0;
}

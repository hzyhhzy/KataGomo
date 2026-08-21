#include "../core/config_parser.h"
#include "../core/global.h"
#include "../core/logger.h"
#include "../core/rand.h"
#include "../core/sha2.h"
#include "../game/board.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nneval.h"
#include "../program/setup.h"
#include "../command/commandline.h"
#include "../main.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <type_traits>

using namespace std;

namespace {

constexpr uint32_t RAW_GATE_SCHEMA = 1;
constexpr uint32_t SOURCE_R15_CORPUS = 1;
constexpr uint32_t SOURCE_SYNTHETIC = 2;
constexpr int SPATIAL_FEATURES = 22;
constexpr int GLOBAL_FEATURES = 39;
constexpr int V102_DEFAULT_PHYSICAL_BATCH_SIZE = 36;
constexpr int V105_LAYER_COUNT = 36;
constexpr int V105_PHYSICAL_BATCH_SIZE = 28;

enum class ModelContract {
  V102,
  V105,
};

enum class RouteContract : uint32_t {
  NONE = 0,
  OFFICIAL_STAGE1 = 1,
  OFFICIAL_V105_QKN_CLIP4 = 2,
};

struct Corpus {
  uint32_t numRows;
  int boardSize;
  uint32_t sourceKind;
  array<unsigned char,32> identity;
  // Canonical row-major, channel-major spatial input: [row][channel][point].
  vector<float> spatial;
  vector<float> global;
};

struct RawSections {
  int batchSize;
  vector<float> policy;
  vector<float> value;
  vector<float> scoreValue;
  vector<float> ownership;
};

uint32_t readU32(ifstream& in, const char* what) {
  unsigned char b[4];
  in.read((char*)b,4);
  if(!in)
    throw StringError(string("nnrawgate: truncated ") + what);
  return (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
         ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

void writeU32(ofstream& out, uint32_t value) {
  const unsigned char b[4] = {
    (unsigned char)(value & 0xff),
    (unsigned char)((value >> 8) & 0xff),
    (unsigned char)((value >> 16) & 0xff),
    (unsigned char)((value >> 24) & 0xff),
  };
  out.write((const char*)b,4);
  if(!out)
    throw StringError("nnrawgate: output write failed");
}

void readExact(ifstream& in, void* dst, size_t bytes, const char* what) {
  in.read((char*)dst,(streamsize)bytes);
  if(!in)
    throw StringError(string("nnrawgate: truncated ") + what);
}

void writeExact(ofstream& out, const void* src, size_t bytes) {
  out.write((const char*)src,(streamsize)bytes);
  if(!out)
    throw StringError("nnrawgate: output write failed");
}

Corpus readR15Corpus(const string& path) {
  ifstream in(path,ios::binary);
  if(!in)
    throw StringError("nnrawgate: could not open corpus " + path);
  char magic[8];
  readExact(in,magic,8,"corpus magic");
  if(memcmp(magic,"R15CORP1",8) != 0)
    throw StringError("nnrawgate: bad R15 corpus magic");

  Corpus corpus;
  corpus.numRows = readU32(in,"corpus row count");
  const uint32_t posLen = readU32(in,"corpus board size");
  const uint32_t spatialFeatures = readU32(in,"corpus spatial features");
  const uint32_t globalFeatures = readU32(in,"corpus global features");
  const uint32_t packedWidth = readU32(in,"corpus packed width");
  const uint32_t policyDim = readU32(in,"corpus policy dimension");
  const uint32_t globalTargetDim = readU32(in,"corpus target dimension");
  if(corpus.numRows == 0 || posLen != 15 || spatialFeatures != SPATIAL_FEATURES ||
     globalFeatures != GLOBAL_FEATURES || packedWidth != 29 || policyDim != 226 ||
     globalTargetDim != 64)
    throw StringError("nnrawgate: corpus dimensions do not match full-board 15x15 V101 input");

  corpus.boardSize = 15;
  corpus.sourceKind = SOURCE_R15_CORPUS;
  readExact(in,corpus.identity.data(),corpus.identity.size(),"corpus identity");
  vector<unsigned char> packed((size_t)corpus.numRows * SPATIAL_FEATURES * packedWidth);
  readExact(in,packed.data(),packed.size(),"packed spatial input");
  corpus.spatial.resize((size_t)corpus.numRows * SPATIAL_FEATURES * 225);
  for(uint32_t row = 0; row < corpus.numRows; row++) {
    for(int c = 0; c < SPATIAL_FEATURES; c++) {
      const unsigned char* src = packed.data() +
        ((size_t)row * SPATIAL_FEATURES + (size_t)c) * packedWidth;
      float* dst = corpus.spatial.data() +
        ((size_t)row * SPATIAL_FEATURES + (size_t)c) * 225;
      for(int p = 0; p < 225; p++)
        dst[p] = (float)((src[p/8] >> (7-(p%8))) & 1);
    }
  }
  corpus.global.resize((size_t)corpus.numRows * GLOBAL_FEATURES);
  readExact(
    in,corpus.global.data(),corpus.global.size()*sizeof(float),"global input"
  );
  const uint64_t targetBytes =
    (uint64_t)corpus.numRows * 2 * policyDim * sizeof(int16_t) +
    (uint64_t)corpus.numRows * globalTargetDim * sizeof(float);
  in.seekg((streamoff)targetBytes,ios::cur);
  if(!in || in.peek() != EOF)
    throw StringError("nnrawgate: corpus size mismatch");
  for(uint32_t row = 0; row < corpus.numRows; row++) {
    const float* mask = corpus.spatial.data() + (size_t)row * SPATIAL_FEATURES * 225;
    for(int p = 0; p < 225; p++) {
      if(mask[p] != 1.0f)
        throw StringError("nnrawgate: R15 gate corpus must use a full-board mask");
    }
  }
  return corpus;
}

uint64_t mix64(uint64_t x) {
  x += UINT64_C(0x9e3779b97f4a7c15);
  x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
  return x ^ (x >> 31);
}

void appendIdentityU32(vector<unsigned char>& bytes, uint32_t value) {
  bytes.push_back((unsigned char)(value & 0xff));
  bytes.push_back((unsigned char)((value >> 8) & 0xff));
  bytes.push_back((unsigned char)((value >> 16) & 0xff));
  bytes.push_back((unsigned char)((value >> 24) & 0xff));
}

void appendIdentityFloat(vector<unsigned char>& bytes, float value) {
  uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value),"unexpected float width");
  memcpy(&bits,&value,sizeof(bits));
  appendIdentityU32(bytes,bits);
}

Corpus makeSyntheticCorpus(int boardSize, int numRows) {
  if((boardSize != 15 && boardSize != 19) || numRows <= 0)
    throw StringError("nnrawgate: synthetic corpus requires board 15 or 19 and positive rows");
  const int area = boardSize * boardSize;
  Corpus corpus;
  corpus.numRows = (uint32_t)numRows;
  corpus.boardSize = boardSize;
  corpus.sourceKind = SOURCE_SYNTHETIC;
  corpus.spatial.resize((size_t)numRows * SPATIAL_FEATURES * area);
  corpus.global.resize((size_t)numRows * GLOBAL_FEATURES);

  const uint64_t seed = UINT64_C(0x4e4e524157474154);
  for(int row = 0; row < numRows; row++) {
    float* spatial = corpus.spatial.data() + (size_t)row * SPATIAL_FEATURES * area;
    for(int p = 0; p < area; p++)
      spatial[p] = 1.0f;
    for(int c = 1; c < SPATIAL_FEATURES; c++) {
      const int threshold = 1 + ((row + c * 3) % 5); // 1/16 through 5/16 density.
      for(int p = 0; p < area; p++) {
        const uint64_t key = seed ^ ((uint64_t)(row+1) * UINT64_C(0xd6e8feb86659fd93)) ^
          ((uint64_t)(c+1) * UINT64_C(0xa5a3564e27f8862f)) ^
          ((uint64_t)(p+1) * UINT64_C(0x9e3779b185ebca87));
        spatial[(size_t)c * area + p] = (mix64(key) >> 60) < (uint64_t)threshold ? 1.0f : 0.0f;
      }
    }
    float* global = corpus.global.data() + (size_t)row * GLOBAL_FEATURES;
    for(int c = 0; c < GLOBAL_FEATURES; c++) {
      const uint64_t key = seed ^ ((uint64_t)(row+1) * UINT64_C(0x94d049bb133111eb)) ^
        ((uint64_t)(c+1) * UINT64_C(0xbf58476d1ce4e5b9));
      const int code = (int)(mix64(key) % 5);
      global[c] = (float)(code - 2) * 0.5f;
    }
  }

  vector<unsigned char> identityBytes;
  identityBytes.reserve(16 + (corpus.spatial.size()+corpus.global.size())*sizeof(float));
  appendIdentityU32(identityBytes,(uint32_t)boardSize);
  appendIdentityU32(identityBytes,(uint32_t)numRows);
  appendIdentityU32(identityBytes,SPATIAL_FEATURES);
  appendIdentityU32(identityBytes,GLOBAL_FEATURES);
  for(float value: corpus.spatial)
    appendIdentityFloat(identityBytes,value);
  for(float value: corpus.global)
    appendIdentityFloat(identityBytes,value);
  SHA2::get256(identityBytes.data(),identityBytes.size(),corpus.identity.data());
  return corpus;
}

string normalizeSha256(const string& raw) {
  if(raw.size() != 64)
    throw StringError("nnrawgate: expected-model-sha256 must contain exactly 64 hex digits");
  string normalized;
  normalized.reserve(raw.size());
  for(char c: raw) {
    const unsigned char uc = (unsigned char)c;
    if(!std::isxdigit(uc))
      throw StringError("nnrawgate: expected-model-sha256 contains a non-hex character");
    normalized.push_back((char)std::tolower(uc));
  }
  return normalized;
}

vector<int> parseSchedule(const string& raw, int maxBatchSize) {
  vector<int> schedule;
  if(raw.empty())
    schedule = {maxBatchSize,1,maxBatchSize-1,2,7,maxBatchSize};
  else {
    for(const string& piece: Global::split(raw,',')) {
      if(piece.empty())
        throw StringError("nnrawgate: batch-schedule contains an empty item");
      schedule.push_back(Global::stringToInt(piece));
    }
  }
  if(maxBatchSize < 8)
    throw StringError("nnrawgate: max batch size must be at least 8 for the fixed dynamic schedule");
  if(schedule.size() < 6 || schedule.front() != maxBatchSize || schedule.back() != maxBatchSize)
    throw StringError("nnrawgate: batch-schedule must begin and end with max batch size");
  const int required[] = {1,2,7,maxBatchSize-1,maxBatchSize};
  for(int requiredBatch: required) {
    if(find(schedule.begin(),schedule.end(),requiredBatch) == schedule.end())
      throw StringError("nnrawgate: batch-schedule is missing required dynamic batch " + Global::intToString(requiredBatch));
  }
  for(int batch: schedule) {
    if(batch <= 0 || batch > maxBatchSize)
      throw StringError("nnrawgate: batch-schedule item is outside [1,max-batch]");
  }
  return schedule;
}

template<typename CreateHandleFn>
ComputeHandle* createRawGateHandleCompat(
  CreateHandleFn createHandle,
  ComputeContext* context,
  const LoadedModel* model,
  Logger* logger,
  int maxBatchSize,
  bool requireExact,
  bool inputsUseNHWC,
  int gpuIdx,
  int backendNumThreads
) {
  if constexpr(std::is_invocable_r_v<
    ComputeHandle*,CreateHandleFn,ComputeContext*,const LoadedModel*,Logger*,int,bool,bool,int,int,int
  >) {
    return createHandle(
      context,model,logger,maxBatchSize,requireExact,inputsUseNHWC,gpuIdx,0,backendNumThreads
    );
  }
  else {
    static_assert(std::is_invocable_r_v<
      ComputeHandle*,CreateHandleFn,ComputeContext*,const LoadedModel*,Logger*,int,bool,bool,int,int,int,int
    >,"unsupported createComputeHandle signature for nnrawgate");
    return createHandle(
      context,model,logger,maxBatchSize,requireExact,inputsUseNHWC,gpuIdx,0,1,backendNumThreads
    );
  }
}

#ifdef KATAGO_BUILD_BENCHMARKNN
bool isOfficialStage1Route(const NeuralNet::BenchmarkRouteProof& proof, int batchSize) {
  const int attention = proof.expectedAttention;
  const int ffn = proof.expectedFfn;
  return
    proof.prepared && proof.hasSuccessfulInvocation && proof.lastBatchSize == batchSize &&
    attention > 0 && attention == ffn &&
    proof.preparedAttention == attention && proof.preparedFfn == ffn &&
    proof.preparedCombinedQKV == attention && proof.preparedLearnedRopeFp32 == attention &&
    proof.preparedMma == attention && proof.preparedFixedRope == 0 &&
    proof.preparedScalar == 0 && proof.preparedPlanar == 0 && proof.preparedCudnn == 0 &&
    proof.preparedFallback == 0 &&
    proof.lastActiveAttention == attention && proof.lastActiveFfn == ffn &&
    proof.lastActiveCombinedQKV == attention && proof.lastActiveLearnedRopeFp32 == attention &&
    proof.lastActiveMma == attention && proof.lastActiveFixedRope == 0 &&
    proof.lastActiveScalar == 0 && proof.lastActivePlanar == 0 && proof.lastActiveCudnn == 0 &&
    proof.lastActiveFallback == 0 &&
    proof.lastFp16 && proof.lastNhwc && proof.lastExact && proof.lastMaskNull;
}

bool isOfficialV105QknClip4Route(const NeuralNet::BenchmarkRouteProof& proof, int batchSize) {
  return
    proof.prepared && proof.hasSuccessfulInvocation && proof.lastBatchSize == batchSize &&
    proof.expectedAttention == V105_LAYER_COUNT && proof.expectedFfn == V105_LAYER_COUNT &&
    proof.expectedQkn == V105_LAYER_COUNT &&
    proof.expectedOrderedClippedSwiGLU == V105_LAYER_COUNT &&
    proof.preparedAttention == V105_LAYER_COUNT && proof.preparedFfn == V105_LAYER_COUNT &&
    proof.preparedPlanar == V105_LAYER_COUNT && proof.preparedQkn == V105_LAYER_COUNT &&
    proof.preparedLearnedRopeFp32 == V105_LAYER_COUNT && proof.preparedMma == V105_LAYER_COUNT &&
    proof.preparedOrderedClippedSwiGLU == V105_LAYER_COUNT &&
    proof.preparedCombinedQKV == 0 && proof.preparedFixedRope == 0 &&
    proof.preparedScalar == 0 && proof.preparedCudnn == 0 && proof.preparedFallback == 0 &&
    proof.lastActiveAttention == V105_LAYER_COUNT && proof.lastActiveFfn == V105_LAYER_COUNT &&
    proof.lastActivePlanar == V105_LAYER_COUNT && proof.lastActiveQkn == V105_LAYER_COUNT &&
    proof.lastActiveLearnedRopeFp32 == V105_LAYER_COUNT && proof.lastActiveMma == V105_LAYER_COUNT &&
    proof.lastActiveOrderedClippedSwiGLU == V105_LAYER_COUNT &&
    proof.lastActiveCombinedQKV == 0 && proof.lastActiveFixedRope == 0 &&
    proof.lastActiveScalar == 0 && proof.lastActiveCudnn == 0 && proof.lastActiveFallback == 0 &&
    proof.lastFp16 && proof.lastNhwc && proof.lastExact && proof.lastMaskNull;
}
#endif

void verifyRouteAfterCall(
  const ComputeHandle* handle,
  RouteContract routeContract,
  int batchSize,
  uint64_t& previousSerial
) {
  if(routeContract == RouteContract::NONE)
    return;
#ifdef KATAGO_BUILD_BENCHMARKNN
  NeuralNet::BenchmarkRouteProof proof = NeuralNet::BenchmarkRouteProof();
  if(!NeuralNet::getBenchmarkRouteProof(handle,proof))
    throw StringError("nnrawgate: official route proof unavailable after inference");
  if(proof.invocationSerial != previousSerial + 1)
    throw StringError("nnrawgate: route serial did not advance exactly once for actual batch");
  const bool routeMatches = routeContract == RouteContract::OFFICIAL_STAGE1 ?
    isOfficialStage1Route(proof,batchSize) : isOfficialV105QknClip4Route(proof,batchSize);
  if(!routeMatches)
    throw StringError("nnrawgate: inference did not use the required official route contract");
  previousSerial = proof.invocationSerial;
#else
  (void)handle;
  (void)batchSize;
  (void)previousSerial;
  throw StringError("nnrawgate: an expected official route requires KATAGO_BUILD_BENCHMARKNN=ON");
#endif
}

void fillRows(
  const Corpus& corpus,
  uint32_t rowStart,
  int batchSize,
  bool useNHWC,
  vector<unique_ptr<NNResultBuf>>& ownedRows
) {
  const int area = corpus.boardSize * corpus.boardSize;
  for(int b = 0; b < batchSize; b++) {
    const uint32_t srcRow = rowStart + (uint32_t)b;
    if(srcRow >= corpus.numRows)
      throw StringError("nnrawgate: input row request exceeds corpus");
    NNResultBuf* row = ownedRows[b].get();
    const float* canonical = corpus.spatial.data() + (size_t)srcRow * SPATIAL_FEATURES * area;
    for(int c = 0; c < SPATIAL_FEATURES; c++) {
      for(int p = 0; p < area; p++)
        row->rowSpatial[useNHWC ? p*SPATIAL_FEATURES+c : c*area+p] = canonical[c*area+p];
    }
    const float* global = corpus.global.data() + (size_t)srcRow * GLOBAL_FEATURES;
    copy(global,global+GLOBAL_FEATURES,row->rowGlobal);
    row->symmetry = 0;
  }
}

RawSections runRawCall(
  ComputeHandle* handle,
  InputBuffers* inputBuffers,
  const Corpus& corpus,
  uint32_t rowStart,
  int batchSize,
  bool useNHWC,
  RouteContract routeContract,
  uint64_t& routeSerial,
  vector<unique_ptr<NNResultBuf>>& ownedRows,
  vector<NNResultBuf*>& rowPointers,
  vector<unique_ptr<NNOutput>>& ownedOutputs,
  vector<float>& postPolicy
) {
  fillRows(corpus,rowStart,batchSize,useNHWC,ownedRows);
  vector<NNOutput*> outputs;
  outputs.reserve(batchSize);
  for(int b = 0; b < batchSize; b++)
    outputs.push_back(ownedOutputs[b].get());
  NeuralNet::getOutput(
    handle,inputBuffers,batchSize,rowPointers.data(),outputs,postPolicy.data()
  );
  verifyRouteAfterCall(handle,routeContract,batchSize,routeSerial);

  RawNNGateOutputs raw;
  NeuralNet::getRawNNGateOutputs(inputBuffers,raw);
  const int area = corpus.boardSize * corpus.boardSize;
  if(raw.policyElts != (size_t)area+1 || raw.valueElts != 3 ||
     raw.scoreValueElts != 6 || raw.ownershipElts != (size_t)area)
    throw StringError("nnrawgate: raw head dimensions differ from the V101 contract");

  RawSections sections;
  sections.batchSize = batchSize;
  sections.policy.assign(raw.policy,raw.policy+(size_t)batchSize*raw.policyElts);
  sections.value.assign(raw.value,raw.value+(size_t)batchSize*raw.valueElts);
  sections.scoreValue.assign(
    raw.scoreValue,raw.scoreValue+(size_t)batchSize*raw.scoreValueElts
  );
  sections.ownership.assign(
    raw.ownership,raw.ownership+(size_t)batchSize*raw.ownershipElts
  );
  const vector<const vector<float>*> all = {
    &sections.policy,&sections.value,&sections.scoreValue,&sections.ownership
  };
  for(const vector<float>* values: all) {
    for(float value: *values) {
      if(!std::isfinite(value))
        throw StringError("nnrawgate: raw output contains a non-finite float");
    }
  }
  return sections;
}

void copyFullSection(vector<float>& dst, const vector<float>& src, uint32_t rowStart, int dim) {
  copy(src.begin(),src.end(),dst.begin()+(size_t)rowStart*dim);
}

class NeuralNetSessionScope {
 public:
  NeuralNetSessionScope() : active(false) {}
  ~NeuralNetSessionScope() {
    if(active)
      NeuralNet::globalCleanup();
  }
  void initialize(ConfigParser& cfg) {
    Setup::initializeSession(cfg);
    active = true;
  }
 private:
  bool active;
};

}

int MainCmds::nnrawgate(const vector<string>& args) {
  Board::initHash();
  Rand seedRand;
  ConfigParser cfg;
  string modelFile;
  string expectedModelSha256;
  string corpusFile;
  string outputFile;
  string scheduleText;
  int boardSize = 15;
  int maxBatchSize = V102_DEFAULT_PHYSICAL_BATCH_SIZE;
  int syntheticRows = 512;
  ModelContract modelContract = ModelContract::V102;
  RouteContract routeContract = RouteContract::NONE;

  try {
    KataGoCommandLine cmd(
      "Test-only deterministic raw-head replay. Runs no timed region and writes an identity-bound binary dump."
    );
    cmd.addConfigFileArg(KataGoCommandLine::defaultGtpConfigFileName(),"gtp_example.cfg");
    cmd.addModelFileArg();
    TCLAP::ValueArg<string> expectedShaArg(
      "","expected-model-sha256","Required full SHA-256 of the model file",true,"","HEX"
    );
    TCLAP::ValueArg<string> corpusArg(
      "","corpus","Optional R15CORP1 corpus; omit for deterministic synthetic input",false,"","FILE"
    );
    TCLAP::ValueArg<string> outputArg(
      "","output","NNRAWG1 raw output file",true,"","FILE"
    );
    TCLAP::ValueArg<int> boardArg(
      "","board","Exact square board size, 15 or 19 (default 15)",false,15,"N"
    );
    TCLAP::ValueArg<int> batchArg(
      "B","batch-size","Maximum/full physical batch (v102 default 36; v105 requires 28)",
      false,V102_DEFAULT_PHYSICAL_BATCH_SIZE,"N"
    );
    TCLAP::ValueArg<int> rowsArg(
      "","synthetic-rows","Rows when no corpus is supplied (default 512)",false,512,"N"
    );
    TCLAP::ValueArg<string> scheduleArg(
      "","batch-schedule","Same-handle actual-batch sequence (default B,1,B-1,2,7,B)",false,"","LIST"
    );
    TCLAP::ValueArg<string> modelContractArg(
      "","model-contract","Exact model contract: v102 or v105 (default v102)",false,"v102","NAME"
    );
    TCLAP::SwitchArg expectedOfficialArg(
      "","expected-official-stage1","Require official FP16/NHWC attention+FFN route after every call",false
    );
    TCLAP::SwitchArg expectedOfficialV105Arg(
      "","expected-official-v105-qkn-clip4",
      "Require the 36-layer FP16/NHWC planar-QKV/QKN/learned-RoPE/MMA/clipped-SwiGLU route",false
    );
    cmd.add(expectedShaArg);
    cmd.add(corpusArg);
    cmd.add(outputArg);
    cmd.add(boardArg);
    cmd.add(batchArg);
    cmd.add(rowsArg);
    cmd.add(scheduleArg);
    cmd.add(modelContractArg);
    cmd.add(expectedOfficialArg);
    cmd.add(expectedOfficialV105Arg);
    cmd.setShortUsageArgLimit();
    cmd.addOverrideConfigArg();
    cmd.parseArgs(args);
    modelFile = cmd.getModelFile();
    expectedModelSha256 = normalizeSha256(expectedShaArg.getValue());
    corpusFile = corpusArg.getValue();
    outputFile = outputArg.getValue();
    boardSize = boardArg.getValue();
    maxBatchSize = batchArg.getValue();
    syntheticRows = rowsArg.getValue();
    scheduleText = scheduleArg.getValue();
    if(modelContractArg.getValue() == "v102")
      modelContract = ModelContract::V102;
    else if(modelContractArg.getValue() == "v105")
      modelContract = ModelContract::V105;
    else
      throw StringError("nnrawgate: model-contract must be v102 or v105");
    if(expectedOfficialArg.getValue() && expectedOfficialV105Arg.getValue())
      throw StringError("nnrawgate: expected route flags are mutually exclusive");
    if(expectedOfficialArg.getValue())
      routeContract = RouteContract::OFFICIAL_STAGE1;
    else if(expectedOfficialV105Arg.getValue())
      routeContract = RouteContract::OFFICIAL_V105_QKN_CLIP4;
    if(boardSize != 15 && boardSize != 19)
      throw StringError("nnrawgate: board must be 15 or 19");
    if(maxBatchSize < 8 || maxBatchSize > 1024)
      throw StringError("nnrawgate: batch-size must be between 8 and 1024");
    if(syntheticRows <= 0 || syntheticRows > 65536)
      throw StringError("nnrawgate: synthetic-rows must be between 1 and 65536");
    if(!corpusFile.empty() && boardSize != 15)
      throw StringError("nnrawgate: R15CORP1 may only be used with board 15");
    if(modelContract == ModelContract::V105 &&
       (boardSize != 15 || maxBatchSize != V105_PHYSICAL_BATCH_SIZE))
      throw StringError("nnrawgate: v105 contract requires board 15 and physical batch-size 28");
    if(modelContract == ModelContract::V105 && corpusFile.empty())
      throw StringError("nnrawgate: v105 contract requires an R15CORP1 corpus");
    if(routeContract == RouteContract::OFFICIAL_STAGE1 && modelContract != ModelContract::V102)
      throw StringError("nnrawgate: expected-official-stage1 requires model-contract v102");
    if(routeContract == RouteContract::OFFICIAL_V105_QKN_CLIP4 && modelContract != ModelContract::V105)
      throw StringError("nnrawgate: expected-official-v105-qkn-clip4 requires model-contract v105");
    cmd.getConfig(cfg);
  }
  catch(TCLAP::ArgException& e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  const vector<int> schedule = parseSchedule(scheduleText,maxBatchSize);
  if(modelContract == ModelContract::V105 &&
     schedule != vector<int>{28,1,27,2,7,28})
    throw StringError("nnrawgate: v105 contract requires exact schedule 28,1,27,2,7,28");
  Corpus corpus = corpusFile.empty() ?
    makeSyntheticCorpus(boardSize,max(syntheticRows,maxBatchSize)) : readR15Corpus(corpusFile);
  if(corpus.numRows < (uint32_t)maxBatchSize)
    throw StringError("nnrawgate: corpus must contain at least max-batch rows");

  cfg.overrideKey("numNNServerThreadsPerModel","1");
  cfg.overrideKey("nnRandomize","false");
  cfg.overrideKey("nnForcedSymmetry","0");
  cfg.overrideKey("debugSkipNeuralNet","false");
  cfg.overrideKey("maxBoardXSizeForNNBuffer",Global::intToString(boardSize));
  cfg.overrideKey("maxBoardYSizeForNNBuffer",Global::intToString(boardSize));
  cfg.overrideKey("maxBoardXSizeForNNBuffer0",Global::intToString(boardSize));
  cfg.overrideKey("maxBoardYSizeForNNBuffer0",Global::intToString(boardSize));
  cfg.overrideKey("requireMaxBoardSize","true");
  cfg.overrideKey("requireMaxBoardSize0","true");
  if(routeContract != RouteContract::NONE) {
    cfg.overrideKey("cudaUseFP16","true");
    cfg.overrideKey("cudaUseFP16-0","true");
    cfg.overrideKey("cudaUseNHWC","true");
    cfg.overrideKey("cudaUseNHWC0","true");
    cfg.overrideKey("cudaInputsUseNHWC","true");
    cfg.overrideKey("cudaInputsUseNHWC0","true");
  }

  Logger logger(NULL,true,false,false);
  const string revision = Version::getGitRevisionWithBackend();
  logger.write("Version " + revision);
  logger.write(
    "nnrawgate board=" + Global::intToString(boardSize) +
    " rows=" + Global::uint64ToString(corpus.numRows) +
    " maxBatch=" + Global::intToString(maxBatchSize) +
    " source=" + string(corpus.sourceKind == SOURCE_R15_CORPUS ? "R15CORP1" : "synthetic-v1")
  );

  NeuralNetSessionScope session;
  session.initialize(cfg);
  unique_ptr<NNEvaluator> nnEval(
    Setup::initializeNNEvaluator(
      modelFile,modelFile,expectedModelSha256,cfg,logger,seedRand,maxBatchSize*64,maxBatchSize,
      boardSize,boardSize,maxBatchSize,true,false,Setup::SETUP_FOR_BENCHMARK
    )
  );
  nnEval->killServerThreads();
  if(nnEval->getNNXLen() != boardSize || nnEval->getNNYLen() != boardSize ||
     !nnEval->getRawGateRequireExactNNLen())
    throw StringError("nnrawgate: evaluator did not initialize the exact requested board");
  if(nnEval->getNumServerThreads() < 1)
    throw StringError("nnrawgate: evaluator has no configured GPU thread");

  LoadedModel* loadedModel = nnEval->getRawGateLoadedModel();
  const int modelVersion = NeuralNet::getModelVersion(loadedModel);
  const int inputsVersion = NNModelVersion::getInputsVersion(modelVersion);
  const int requiredModelVersion = modelContract == ModelContract::V105 ? 105 : 102;
  if(modelVersion != requiredModelVersion || inputsVersion != 101 ||
     NNModelVersion::getNumSpatialFeatures(modelVersion) != SPATIAL_FEATURES ||
     NNModelVersion::getNumGlobalFeatures(modelVersion) != GLOBAL_FEATURES)
    throw StringError(
      "nnrawgate: model does not satisfy the selected exact V101 22/39 model contract"
    );

  ComputeHandle* handle = nullptr;
  InputBuffers* inputBuffers = nullptr;
  try {
    handle = createRawGateHandleCompat(
      &NeuralNet::createComputeHandle,
      nnEval->getRawGateComputeContext(),loadedModel,&logger,maxBatchSize,true,
      nnEval->getRawGateInputsUseNHWC(),nnEval->getRawGateGpuIdx(0),
      nnEval->getRawGateBackendNumThreads()
    );
    inputBuffers = NeuralNet::createInputBuffers(
      loadedModel,maxBatchSize,boardSize,boardSize
    );
    const bool usingFP16 = NeuralNet::isUsingFP16(handle);
    const bool useNHWC = nnEval->getRawGateInputsUseNHWC();
    if(routeContract != RouteContract::NONE && (!usingFP16 || !useNHWC))
      throw StringError("nnrawgate: expected official route requires FP16 and NHWC");

    uint64_t routeSerial = 0;
#ifdef KATAGO_BUILD_BENCHMARKNN
    if(routeContract != RouteContract::NONE) {
      NeuralNet::BenchmarkRouteProof initial = NeuralNet::BenchmarkRouteProof();
      if(!NeuralNet::getBenchmarkRouteProof(handle,initial) || !initial.prepared || initial.hasSuccessfulInvocation)
        throw StringError("nnrawgate: invalid initial official-route proof state");
      routeSerial = initial.invocationSerial;
    }
#endif

    vector<unique_ptr<NNResultBuf>> ownedRows;
    vector<NNResultBuf*> rowPointers;
    vector<unique_ptr<NNOutput>> ownedOutputs;
    ownedRows.reserve(maxBatchSize);
    rowPointers.reserve(maxBatchSize);
    ownedOutputs.reserve(maxBatchSize);
    const int area = boardSize * boardSize;
    for(int b = 0; b < maxBatchSize; b++) {
      ownedRows.push_back(make_unique<NNResultBuf>());
      NNResultBuf* row = ownedRows.back().get();
      row->rowSpatial = new float[SPATIAL_FEATURES*area];
      row->rowSpatialSize = SPATIAL_FEATURES*area;
      row->rowGlobal = new float[GLOBAL_FEATURES];
      row->rowGlobalSize = GLOBAL_FEATURES;
      rowPointers.push_back(row);
      ownedOutputs.push_back(make_unique<NNOutput>());
      ownedOutputs.back()->nnXLen = boardSize;
      ownedOutputs.back()->nnYLen = boardSize;
    }
    vector<float> postPolicy((size_t)maxBatchSize * NNPos::MAX_NN_POLICY_SIZE);
    const int policyDim = area + 1;
    const int valueDim = 3;
    const int scoreValueDim = 6;
    const int ownershipDim = area;
    RawSections full;
    full.batchSize = maxBatchSize;
    full.policy.resize((size_t)corpus.numRows*policyDim);
    full.value.resize((size_t)corpus.numRows*valueDim);
    full.scoreValue.resize((size_t)corpus.numRows*scoreValueDim);
    full.ownership.resize((size_t)corpus.numRows*ownershipDim);
    vector<int> fullBatchSizes;
    for(uint32_t rowStart = 0; rowStart < corpus.numRows; rowStart += (uint32_t)maxBatchSize) {
      const int actualBatch = (int)min<uint32_t>(
        (uint32_t)maxBatchSize,corpus.numRows-rowStart
      );
      RawSections call = runRawCall(
        handle,inputBuffers,corpus,rowStart,actualBatch,useNHWC,routeContract,
        routeSerial,ownedRows,rowPointers,ownedOutputs,postPolicy
      );
      fullBatchSizes.push_back(actualBatch);
      copyFullSection(full.policy,call.policy,rowStart,policyDim);
      copyFullSection(full.value,call.value,rowStart,valueDim);
      copyFullSection(full.scoreValue,call.scoreValue,rowStart,scoreValueDim);
      copyFullSection(full.ownership,call.ownership,rowStart,ownershipDim);
    }

    vector<RawSections> dynamicCalls;
    dynamicCalls.reserve(schedule.size());
    for(int actualBatch: schedule) {
      dynamicCalls.push_back(runRawCall(
        handle,inputBuffers,corpus,0,actualBatch,useNHWC,routeContract,
        routeSerial,ownedRows,rowPointers,ownedOutputs,postPolicy
      ));
    }

    ofstream out(outputFile,ios::binary|ios::trunc);
    if(!out)
      throw StringError("nnrawgate: could not open output " + outputFile);
    const char magic[8] = {'N','N','R','A','W','G','1','\0'};
    writeExact(out,magic,8);
    const uint32_t header[] = {
      RAW_GATE_SCHEMA,(uint32_t)boardSize,corpus.numRows,(uint32_t)maxBatchSize,
      SPATIAL_FEATURES,GLOBAL_FEATURES,(uint32_t)policyDim,(uint32_t)valueDim,
      (uint32_t)scoreValueDim,(uint32_t)ownershipDim,usingFP16 ? 1u : 0u,
      useNHWC ? 1u : 0u,corpus.sourceKind,(uint32_t)schedule.size(),
      (uint32_t)fullBatchSizes.size(),(uint32_t)routeContract,
      (uint32_t)modelVersion,(uint32_t)inputsVersion,0u
    };
    for(uint32_t value: header)
      writeU32(out,value);
    writeExact(out,corpus.identity.data(),corpus.identity.size());
    writeExact(out,expectedModelSha256.data(),expectedModelSha256.size());
    writeU32(out,(uint32_t)revision.size());
    writeExact(out,revision.data(),revision.size());
    for(int batch: fullBatchSizes)
      writeU32(out,(uint32_t)batch);
    for(int batch: schedule)
      writeU32(out,(uint32_t)batch);
    writeExact(out,full.policy.data(),full.policy.size()*sizeof(float));
    writeExact(out,full.value.data(),full.value.size()*sizeof(float));
    writeExact(out,full.scoreValue.data(),full.scoreValue.size()*sizeof(float));
    writeExact(out,full.ownership.data(),full.ownership.size()*sizeof(float));
    for(const RawSections& call: dynamicCalls) {
      writeU32(out,(uint32_t)call.batchSize);
      writeExact(out,call.policy.data(),call.policy.size()*sizeof(float));
      writeExact(out,call.value.data(),call.value.size()*sizeof(float));
      writeExact(out,call.scoreValue.data(),call.scoreValue.size()*sizeof(float));
      writeExact(out,call.ownership.data(),call.ownership.size()*sizeof(float));
    }
    out.close();
    logger.write(
      "nnrawgate wrote " + outputFile + " usingFP16=" +
      string(usingFP16 ? "true" : "false") + " useNHWC=" +
      string(useNHWC ? "true" : "false") + " fullReplayTailUsesActualBatch=true"
    );
  }
  catch(...) {
    if(handle != nullptr)
      NeuralNet::freeComputeHandle(handle);
    if(inputBuffers != nullptr)
      NeuralNet::freeInputBuffers(inputBuffers);
    throw;
  }
  NeuralNet::freeComputeHandle(handle);
  NeuralNet::freeInputBuffers(inputBuffers);
  return 0;
}

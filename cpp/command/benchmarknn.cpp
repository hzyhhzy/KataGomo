#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/logger.h"
#include "../core/rand.h"
#include "../game/board.h"
#include "../neuralnet/nneval.h"
#include "../program/setup.h"
#include "../command/commandline.h"
#include "../main.h"

#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

using namespace std;

namespace {

string checksumString(uint64_t checksum) {
  ostringstream out;
  out << "0x" << hex << setw(16) << setfill('0') << checksum;
  return out.str();
}

bool validRate(double value) {
  return value > 0.0 && std::isfinite(value);
}

bool isOfficialStage1Route(const NeuralNet::BenchmarkRouteProof& proof) {
  const int attention = proof.expectedAttention;
  const int ffn = proof.expectedFfn;
  return
    proof.prepared && proof.hasSuccessfulInvocation &&
    attention > 0 && ffn > 0 && attention == ffn &&
    proof.expectedQkn == 0 && proof.expectedOrderedClippedSwiGLU == 0 &&
    proof.preparedAttention == attention && proof.preparedFfn == ffn &&
    proof.preparedQkn == 0 && proof.preparedOrderedClippedSwiGLU == 0 &&
    proof.preparedCombinedQKV == attention &&
    proof.preparedLearnedRopeFp32 == attention &&
    proof.preparedMma == attention &&
    proof.preparedFixedRope == 0 && proof.preparedScalar == 0 &&
    proof.preparedPlanar == 0 && proof.preparedCudnn == 0 &&
    proof.preparedFallback == 0 &&
    proof.lastActiveAttention == attention && proof.lastActiveFfn == ffn &&
    proof.lastActiveQkn == 0 && proof.lastActiveOrderedClippedSwiGLU == 0 &&
    proof.lastActiveCombinedQKV == attention &&
    proof.lastActiveLearnedRopeFp32 == attention &&
    proof.lastActiveMma == attention &&
    proof.lastActiveFixedRope == 0 && proof.lastActiveScalar == 0 &&
    proof.lastActivePlanar == 0 && proof.lastActiveCudnn == 0 &&
    proof.lastActiveFallback == 0 &&
    proof.lastFp16 && proof.lastNhwc && proof.lastExact && proof.lastMaskNull;
}

bool isOfficialV105Fp16Route(const NeuralNet::BenchmarkRouteProof& proof) {
  const int attention = proof.expectedAttention;
  const int ffn = proof.expectedFfn;
  return
    proof.prepared && proof.hasSuccessfulInvocation &&
    attention > 0 && ffn == attention &&
    proof.expectedQkn == attention &&
    proof.expectedOrderedClippedSwiGLU == ffn &&
    proof.preparedAttention == attention && proof.preparedFfn == ffn &&
    proof.preparedQkn == attention &&
    proof.preparedOrderedClippedSwiGLU == ffn &&
    proof.preparedCombinedQKV == 0 && proof.preparedPlanar == attention &&
    proof.preparedLearnedRopeFp32 == attention && proof.preparedFixedRope == 0 &&
    proof.preparedMma == attention && proof.preparedScalar == 0 &&
    proof.preparedCudnn == 0 && proof.preparedFallback == 0 &&
    proof.lastActiveAttention == attention && proof.lastActiveFfn == ffn &&
    proof.lastActiveQkn == attention &&
    proof.lastActiveOrderedClippedSwiGLU == ffn &&
    proof.lastActiveCombinedQKV == 0 && proof.lastActivePlanar == attention &&
    proof.lastActiveLearnedRopeFp32 == attention && proof.lastActiveFixedRope == 0 &&
    proof.lastActiveMma == attention && proof.lastActiveScalar == 0 &&
    proof.lastActiveCudnn == 0 && proof.lastActiveFallback == 0 &&
    proof.lastFp16 && proof.lastNhwc && proof.lastExact && proof.lastMaskNull;
}

NeuralNet::BenchmarkRouteProof makeOfficialV105Fp16RouteFixture(int numLayers) {
  NeuralNet::BenchmarkRouteProof proof = NeuralNet::BenchmarkRouteProof();
  proof.prepared = true;
  proof.hasSuccessfulInvocation = true;
  proof.invocationSerial = 8;
  proof.streamIdentity = 0x1234;
  proof.nnXLen = 15;
  proof.nnYLen = 15;
  proof.expectedAttention = numLayers;
  proof.expectedFfn = numLayers;
  proof.expectedQkn = numLayers;
  proof.expectedOrderedClippedSwiGLU = numLayers;
  proof.preparedAttention = numLayers;
  proof.preparedFfn = numLayers;
  proof.preparedQkn = numLayers;
  proof.preparedOrderedClippedSwiGLU = numLayers;
  proof.preparedPlanar = numLayers;
  proof.preparedLearnedRopeFp32 = numLayers;
  proof.preparedMma = numLayers;
  proof.lastActiveAttention = numLayers;
  proof.lastActiveFfn = numLayers;
  proof.lastActiveQkn = numLayers;
  proof.lastActiveOrderedClippedSwiGLU = numLayers;
  proof.lastActivePlanar = numLayers;
  proof.lastActiveLearnedRopeFp32 = numLayers;
  proof.lastActiveMma = numLayers;
  proof.lastBatchSize = 3;
  proof.lastFp16 = true;
  proof.lastNhwc = true;
  proof.lastExact = true;
  proof.lastMaskNull = true;
  return proof;
}

void verifyOfficialRoutePredicates() {
  NeuralNet::BenchmarkRouteProof proof = NeuralNet::BenchmarkRouteProof();
  if(isOfficialStage1Route(proof))
    throw StringError("benchmarknn: empty official-route fixture unexpectedly passed");

  proof.prepared = true;
  proof.hasSuccessfulInvocation = true;
  proof.expectedAttention = 2;
  proof.expectedFfn = 2;
  proof.preparedAttention = 2;
  proof.preparedFfn = 2;
  proof.preparedCombinedQKV = 2;
  proof.preparedLearnedRopeFp32 = 2;
  proof.preparedMma = 2;
  proof.lastActiveAttention = 2;
  proof.lastActiveFfn = 2;
  proof.lastActiveCombinedQKV = 2;
  proof.lastActiveLearnedRopeFp32 = 2;
  proof.lastActiveMma = 2;
  proof.lastFp16 = true;
  proof.lastNhwc = true;
  proof.lastExact = true;
  proof.lastMaskNull = true;
  if(!isOfficialStage1Route(proof))
    throw StringError("benchmarknn: valid official-route fixture unexpectedly failed");

  proof.lastActiveScalar = 1;
  if(isOfficialStage1Route(proof))
    throw StringError("benchmarknn: scalar fallback fixture unexpectedly passed official route");
  proof.lastActiveScalar = 0;
  proof.expectedFfn = 1;
  proof.preparedFfn = 1;
  proof.lastActiveFfn = 1;
  if(isOfficialStage1Route(proof))
    throw StringError("benchmarknn: unpaired attention/ffn fixture unexpectedly passed official route");

  NeuralNet::BenchmarkRouteProof v105 = makeOfficialV105Fp16RouteFixture(2);
  if(!isOfficialV105Fp16Route(v105))
    throw StringError("benchmarknn: valid v105 FP16 route fixture unexpectedly failed");
  if(isOfficialStage1Route(v105))
    throw StringError("benchmarknn: v105 route fixture unexpectedly passed Stage1 route");

  v105.lastActiveQkn = 1;
  if(isOfficialV105Fp16Route(v105))
    throw StringError("benchmarknn: incomplete active QKN fixture unexpectedly passed v105 route");
  v105 = makeOfficialV105Fp16RouteFixture(2);
  v105.preparedOrderedClippedSwiGLU = 1;
  if(isOfficialV105Fp16Route(v105))
    throw StringError("benchmarknn: incomplete prepared clipped SwiGLU fixture unexpectedly passed v105 route");
  v105 = makeOfficialV105Fp16RouteFixture(2);
  v105.preparedCombinedQKV = 1;
  v105.preparedPlanar = 1;
  if(isOfficialV105Fp16Route(v105))
    throw StringError("benchmarknn: mixed projection fixture unexpectedly passed v105 route");
  v105 = makeOfficialV105Fp16RouteFixture(2);
  v105.lastMaskNull = false;
  if(isOfficialV105Fp16Route(v105))
    throw StringError("benchmarknn: masked fixture unexpectedly passed v105 route");
}

bool sameRawProof(const NeuralNet::BenchmarkRawIOProof& a, const NeuralNet::BenchmarkRawIOProof& b) {
  return
    a.valid && b.valid && a.batchSize == b.batchSize &&
    a.inputRowsHashed == b.inputRowsHashed &&
    a.uniqueInputRows == b.uniqueInputRows &&
    a.inputChecksum == b.inputChecksum &&
    a.inputContentChecksum == b.inputContentChecksum &&
    a.outputChecksum == b.outputChecksum &&
    a.rawOutputFloatsHashed == b.rawOutputFloatsHashed &&
    a.policyFloatsHashed == b.policyFloatsHashed &&
    a.valueFloatsHashed == b.valueFloatsHashed &&
    a.scoreValueFloatsHashed == b.scoreValueFloatsHashed &&
    a.ownershipFloatsHashed == b.ownershipFloatsHashed &&
    a.outputRowsHashed == b.outputRowsHashed &&
    a.uniqueOutputRows == b.uniqueOutputRows &&
    a.outputsFinite && b.outputsFinite &&
    a.outputsNonzero && b.outputsNonzero &&
    a.outputsNonconstant && b.outputsNonconstant;
}

bool sameRouteFamily(
  const NeuralNet::BenchmarkRouteProof& a,
  const NeuralNet::BenchmarkRouteProof& b
) {
  return
    a.expectedAttention == b.expectedAttention && a.expectedFfn == b.expectedFfn &&
    a.expectedQkn == b.expectedQkn &&
    a.expectedOrderedClippedSwiGLU == b.expectedOrderedClippedSwiGLU &&
    a.preparedAttention == b.preparedAttention && a.preparedFfn == b.preparedFfn &&
    a.preparedQkn == b.preparedQkn &&
    a.preparedOrderedClippedSwiGLU == b.preparedOrderedClippedSwiGLU &&
    a.preparedCombinedQKV == b.preparedCombinedQKV &&
    a.preparedLearnedRopeFp32 == b.preparedLearnedRopeFp32 &&
    a.preparedFixedRope == b.preparedFixedRope && a.preparedMma == b.preparedMma &&
    a.preparedScalar == b.preparedScalar && a.preparedPlanar == b.preparedPlanar &&
    a.preparedCudnn == b.preparedCudnn && a.preparedFallback == b.preparedFallback &&
    a.lastActiveAttention == b.lastActiveAttention && a.lastActiveFfn == b.lastActiveFfn &&
    a.lastActiveQkn == b.lastActiveQkn &&
    a.lastActiveOrderedClippedSwiGLU == b.lastActiveOrderedClippedSwiGLU &&
    a.lastActiveCombinedQKV == b.lastActiveCombinedQKV &&
    a.lastActiveLearnedRopeFp32 == b.lastActiveLearnedRopeFp32 &&
    a.lastActiveFixedRope == b.lastActiveFixedRope && a.lastActiveMma == b.lastActiveMma &&
    a.lastActiveScalar == b.lastActiveScalar && a.lastActivePlanar == b.lastActivePlanar &&
    a.lastActiveCudnn == b.lastActiveCudnn && a.lastActiveFallback == b.lastActiveFallback &&
    a.lastFp16 == b.lastFp16 && a.lastNhwc == b.lastNhwc &&
    a.lastExact == b.lastExact && a.lastMaskNull == b.lastMaskNull;
}

const char* jsonBool(bool value) {
  return value ? "true" : "false";
}

void appendBenchmarkLaneJson(
  ostream& out,
  int lane,
  double medianSeconds,
  double nnEvalsPerSec,
  const NNEvalBenchmarkLaneProof& laneProof,
  const char* serialSemantics
) {
  const NeuralNet::BenchmarkRawIOProof& raw = laneProof.rawIO;
  const NeuralNet::BenchmarkRouteProof& before = laneProof.routeBefore;
  const NeuralNet::BenchmarkRouteProof& route = laneProof.routeAfter;
  out << "{"
      << "\"lane\":" << lane
      << ",\"medianSeconds\":" << medianSeconds
      << ",\"nnEvalsPerSec\":" << nnEvalsPerSec
      << ",\"input\":{"
      << "\"checksum\":\"" << checksumString(raw.inputChecksum) << "\""
      << ",\"contentChecksum\":\"" << checksumString(raw.inputContentChecksum) << "\""
      << ",\"rowsHashed\":" << raw.inputRowsHashed
      << ",\"uniqueRows\":" << raw.uniqueInputRows
      << ",\"generatedChecksumMatch\":"
      << jsonBool(raw.inputChecksum == laneProof.generatedInputChecksum)
      << "}"
      << ",\"rawOutput\":{"
      << "\"checksum\":\"" << checksumString(raw.outputChecksum) << "\""
      << ",\"floatsHashed\":" << raw.rawOutputFloatsHashed
      << ",\"policyFloats\":" << raw.policyFloatsHashed
      << ",\"valueFloats\":" << raw.valueFloatsHashed
      << ",\"scoreValueFloats\":" << raw.scoreValueFloatsHashed
      << ",\"ownershipFloats\":" << raw.ownershipFloatsHashed
      << ",\"rowsHashed\":" << raw.outputRowsHashed
      << ",\"uniqueRows\":" << raw.uniqueOutputRows
      << ",\"finite\":" << jsonBool(raw.outputsFinite)
      << ",\"nonzero\":" << jsonBool(raw.outputsNonzero)
      << ",\"nonconstant\":" << jsonBool(raw.outputsNonconstant)
      << "}"
      << ",\"route\":{"
      << "\"streamIdentity\":\"" << checksumString(route.streamIdentity) << "\""
      << ",\"serialSemantics\":\"" << serialSemantics << "\""
      << ",\"serialBefore\":" << before.invocationSerial
      << ",\"serialAfter\":" << route.invocationSerial
      << ",\"serialDelta\":" << route.invocationSerial - before.invocationSerial
      << ",\"batchSize\":" << route.lastBatchSize
      << ",\"nnXLen\":" << route.nnXLen
      << ",\"nnYLen\":" << route.nnYLen
      << ",\"fp16\":" << jsonBool(route.lastFp16)
      << ",\"nhwc\":" << jsonBool(route.lastNhwc)
      << ",\"exact\":" << jsonBool(route.lastExact)
      << ",\"maskNull\":" << jsonBool(route.lastMaskNull)
      << ",\"officialStage1\":" << jsonBool(isOfficialStage1Route(route))
      << ",\"officialV105Fp16\":" << jsonBool(isOfficialV105Fp16Route(route))
      << ",\"expected\":{"
      << "\"attention\":" << route.expectedAttention
      << ",\"ffn\":" << route.expectedFfn
      << ",\"qkn\":" << route.expectedQkn
      << ",\"orderedClippedSwiGLU\":" << route.expectedOrderedClippedSwiGLU
      << "}"
      << ",\"prepared\":{"
      << "\"attention\":" << route.preparedAttention
      << ",\"ffn\":" << route.preparedFfn
      << ",\"qkn\":" << route.preparedQkn
      << ",\"orderedClippedSwiGLU\":" << route.preparedOrderedClippedSwiGLU
      << ",\"combinedQKV\":" << route.preparedCombinedQKV
      << ",\"learnedRopeFp32\":" << route.preparedLearnedRopeFp32
      << ",\"fixedRope\":" << route.preparedFixedRope
      << ",\"mma\":" << route.preparedMma
      << ",\"scalar\":" << route.preparedScalar
      << ",\"planar\":" << route.preparedPlanar
      << ",\"cudnn\":" << route.preparedCudnn
      << ",\"fallback\":" << route.preparedFallback
      << "}"
      << ",\"active\":{"
      << "\"attention\":" << route.lastActiveAttention
      << ",\"ffn\":" << route.lastActiveFfn
      << ",\"qkn\":" << route.lastActiveQkn
      << ",\"orderedClippedSwiGLU\":" << route.lastActiveOrderedClippedSwiGLU
      << ",\"combinedQKV\":" << route.lastActiveCombinedQKV
      << ",\"learnedRopeFp32\":" << route.lastActiveLearnedRopeFp32
      << ",\"fixedRope\":" << route.lastActiveFixedRope
      << ",\"mma\":" << route.lastActiveMma
      << ",\"scalar\":" << route.lastActiveScalar
      << ",\"planar\":" << route.lastActivePlanar
      << ",\"cudnn\":" << route.lastActiveCudnn
      << ",\"fallback\":" << route.lastActiveFallback
      << "}"
      << "}"
      << "}";
}

int emitV105RouteContractFixture(const string& mode) {
  NNEvalBenchmarkLaneProof laneProof = NNEvalBenchmarkLaneProof();
  laneProof.generatedInputChecksum = 0x1111;
  laneProof.generatedInputContentChecksum = 0x2222;
  laneProof.generatedInputRows = 3;
  laneProof.uniqueGeneratedInputRows = 3;
  laneProof.rawIO.valid = true;
  laneProof.rawIO.batchSize = 3;
  laneProof.rawIO.inputRowsHashed = 3;
  laneProof.rawIO.uniqueInputRows = 3;
  laneProof.rawIO.inputChecksum = laneProof.generatedInputChecksum;
  laneProof.rawIO.inputContentChecksum = laneProof.generatedInputContentChecksum;
  laneProof.rawIO.outputChecksum = 0x3333;
  laneProof.rawIO.rawOutputFloatsHashed = 9;
  laneProof.rawIO.policyFloatsHashed = 3;
  laneProof.rawIO.valueFloatsHashed = 2;
  laneProof.rawIO.scoreValueFloatsHashed = 2;
  laneProof.rawIO.ownershipFloatsHashed = 2;
  laneProof.rawIO.outputsFinite = true;
  laneProof.rawIO.outputsNonzero = true;
  laneProof.rawIO.outputsNonconstant = true;
  laneProof.rawIO.outputRowsHashed = 3;
  laneProof.rawIO.uniqueOutputRows = 3;
  laneProof.routeAfter = makeOfficialV105Fp16RouteFixture(2);
  if(mode == "missing-qkn")
    laneProof.routeAfter.lastActiveQkn -= 1;
  else if(mode == "missing-clip")
    laneProof.routeAfter.lastActiveOrderedClippedSwiGLU -= 1;
  else if(mode != "pass")
    throw StringError("benchmarknn: invalid internal route fixture mode");
  laneProof.routeBefore = laneProof.routeAfter;
  laneProof.routeBefore.invocationSerial = laneProof.routeAfter.invocationSerial - 1;

  ostringstream laneJson;
  appendBenchmarkLaneJson(laneJson,0,0.125,24.0,laneProof,"fixture");
  const string serialized = laneJson.str();
  if(
    serialized.find(
      "\"expected\":{\"attention\":2,\"ffn\":2,\"qkn\":2,\"orderedClippedSwiGLU\":2}"
    ) == string::npos ||
    serialized.find(
      "\"prepared\":{\"attention\":2,\"ffn\":2,\"qkn\":2,\"orderedClippedSwiGLU\":2"
    ) == string::npos ||
    serialized.find(
      "\"active\":{\"attention\":2,\"ffn\":2,\"qkn\":"
    ) == string::npos ||
    serialized.find("\"orderedClippedSwiGLU\":",serialized.find("\"active\":{") + 1) == string::npos
  ) {
    throw StringError("benchmarknn: v105 route JSON fixture lost required route fields");
  }

  const bool routeAccepted = isOfficialV105Fp16Route(laneProof.routeAfter);
  if((mode == "pass") != routeAccepted)
    throw StringError("benchmarknn: v105 route fixture predicate result contradicted its mode");

  cout << "BENCHMARKNN_ROUTE_CONTRACT_FIXTURE_JSON "
       << "{\"schema\":\"katago.benchmarknn.route-fixture.v1\""
       << ",\"status\":\"" << (routeAccepted ? "pass" : "fail") << "\""
       << ",\"mode\":\"" << mode << "\""
       << ",\"routeAccepted\":" << jsonBool(routeAccepted)
       << ",\"expectedModelVersion\":105"
       << ",\"expectedInputsVersion\":101"
       << ",\"expectedSpatialFeatures\":22"
       << ",\"expectedGlobalFeatures\":39"
       << ",\"lane\":" << serialized
       << "}" << endl;
  return routeAccepted ? 0 : 1;
}

class NeuralNetSessionScope {
 public:
  NeuralNetSessionScope() : active(false) {}
  ~NeuralNetSessionScope() {
    if(active)
      NeuralNet::globalCleanup();
  }

  void initialize(ConfigParser& cfg) {
    if(active)
      throw StringError("benchmarknn: neural-net session initialized twice");
    Setup::initializeSession(cfg);
    active = true;
  }

  NeuralNetSessionScope(const NeuralNetSessionScope&) = delete;
  NeuralNetSessionScope& operator=(const NeuralNetSessionScope&) = delete;

 private:
  bool active;
};

}

int MainCmds::benchmarknn(const vector<string>& args) {
  verifyOfficialRoutePredicates();
  Board::initHash();
  Rand seedRand("benchmarknn-fixed-seed-v1");

  ConfigParser cfg;
  string modelFile;
  int batchSize = 36;
  int serverThreads = 2;
  int numWarmups = 10;
  int numIterations = 100;
  int boardXSize = 15;
  int boardYSize = 15;
  bool expectedOfficialStage1 = false;
  bool expectedV105Fp16 = false;

  try {
    KataGoCommandLine cmd(
      "Test-only raw neural-net benchmark. Reports full-I/O wall throughput and a separate "
      "device-only common-wall throughput. Inputs are fixed and reproducible."
    );
    cmd.addConfigFileArg(KataGoCommandLine::defaultGtpConfigFileName(),"gtp_example.cfg");
    cmd.addModelFileArg();
    TCLAP::ValueArg<int> batchSizeArg(
      "B","batch-size","Batch size per benchmark lane (default 36)",false,36,"N"
    );
    TCLAP::ValueArg<int> serverThreadsArg(
      "S","server-threads","Number of independent compute-handle lanes (default 2)",false,2,"N"
    );
    TCLAP::ValueArg<int> warmupArg(
      "W","warmup","Untimed forwards per lane before each timed region (default 10)",false,10,"N"
    );
    TCLAP::ValueArg<int> iterationsArg(
      "I","iterations","Timed forwards per lane in each mode (default 100)",false,100,"N"
    );
    TCLAP::ValueArg<int> boardArg(
      "","board","Exact square NN board size (default 15)",false,15,"N"
    );
    TCLAP::ValueArg<int> boardSizeAliasArg(
      "","boardsize","Alias for -board",false,-1,"N"
    );
    TCLAP::ValueArg<int> boardXArg(
      "","board-x","Exact NN board width (requires -board-y)",false,-1,"N"
    );
    TCLAP::ValueArg<int> boardYArg(
      "","board-y","Exact NN board height (requires -board-x)",false,-1,"N"
    );
    TCLAP::SwitchArg expectedOfficialStage1Arg(
      "","expected-official-stage1",
      "Require every lane to prove the official Stage1 FP16 transformer route",false
    );
    TCLAP::SwitchArg expectedV105Fp16Arg(
      "","expected-v105-fp16",
      "Require every lane to prove the native v105 FP16 transformer route",false
    );
    TCLAP::SwitchArg routeContractFixtureArg(
      "","route-contract-fixture",
      "Run the CPU-only v105 route predicate and JSON serialization fixture",false
    );
    TCLAP::SwitchArg routeContractFixtureMissingQknArg(
      "","route-contract-fixture-missing-qkn",
      "Run the CPU-only v105 route fixture with one missing active QKN route",false
    );
    TCLAP::SwitchArg routeContractFixtureMissingClipArg(
      "","route-contract-fixture-missing-clip",
      "Run the CPU-only v105 route fixture with one missing active clipped SwiGLU route",false
    );
    cmd.add(batchSizeArg);
    cmd.add(serverThreadsArg);
    cmd.add(warmupArg);
    cmd.add(iterationsArg);
    cmd.add(boardArg);
    cmd.add(boardSizeAliasArg);
    cmd.add(boardXArg);
    cmd.add(boardYArg);
    cmd.add(expectedOfficialStage1Arg);
    cmd.add(expectedV105Fp16Arg);
    cmd.add(routeContractFixtureArg);
    cmd.add(routeContractFixtureMissingQknArg);
    cmd.add(routeContractFixtureMissingClipArg);
    cmd.setShortUsageArgLimit();
    cmd.addOverrideConfigArg();
    cmd.parseArgs(args);

    expectedOfficialStage1 = expectedOfficialStage1Arg.getValue();
    expectedV105Fp16 = expectedV105Fp16Arg.getValue();
    if(expectedOfficialStage1 && expectedV105Fp16)
      throw StringError(
        "benchmarknn: -expected-official-stage1 and -expected-v105-fp16 are mutually exclusive"
      );
    const int numRouteFixtureModes =
      (routeContractFixtureArg.getValue() ? 1 : 0) +
      (routeContractFixtureMissingQknArg.getValue() ? 1 : 0) +
      (routeContractFixtureMissingClipArg.getValue() ? 1 : 0);
    if(numRouteFixtureModes > 1)
      throw StringError("benchmarknn: specify only one route-contract fixture mode");
    if(routeContractFixtureArg.getValue())
      return emitV105RouteContractFixture("pass");
    if(routeContractFixtureMissingQknArg.getValue())
      return emitV105RouteContractFixture("missing-qkn");
    if(routeContractFixtureMissingClipArg.getValue())
      return emitV105RouteContractFixture("missing-clip");

    modelFile = cmd.getModelFile();
    batchSize = batchSizeArg.getValue();
    serverThreads = serverThreadsArg.getValue();
    numWarmups = warmupArg.getValue();
    numIterations = iterationsArg.getValue();
    const bool squareBoardIsSet = boardArg.isSet() || boardSizeAliasArg.isSet();
    const bool rectangularBoardIsSet = boardXArg.isSet() || boardYArg.isSet();
    if(boardArg.isSet() && boardSizeAliasArg.isSet())
      throw StringError("benchmarknn: specify only one of -board and -boardsize");
    if(squareBoardIsSet && rectangularBoardIsSet)
      throw StringError("benchmarknn: square and X/Y board options are mutually exclusive");
    if(boardXArg.isSet() != boardYArg.isSet())
      throw StringError("benchmarknn: -board-x and -board-y must be specified together");
    if(rectangularBoardIsSet) {
      boardXSize = boardXArg.getValue();
      boardYSize = boardYArg.getValue();
    }
    else {
      const int boardSize =
        boardSizeAliasArg.isSet() ? boardSizeAliasArg.getValue() : boardArg.getValue();
      boardXSize = boardSize;
      boardYSize = boardSize;
    }
    if(batchSize <= 0 || batchSize > 1024)
      throw StringError("benchmarknn: batch-size must be between 1 and 1024");
    if(serverThreads <= 0 || serverThreads > 64)
      throw StringError("benchmarknn: server-threads must be between 1 and 64");
    if(numWarmups < 0 || numWarmups > 10000)
      throw StringError("benchmarknn: warmup must be between 0 and 10000");
    if(numIterations <= 0 || numIterations > 4096)
      throw StringError("benchmarknn: iterations must be between 1 and 4096");
    if(boardXSize < 2 || boardXSize > Board::MAX_LEN ||
       boardYSize < 2 || boardYSize > Board::MAX_LEN)
      throw StringError(
        "benchmarknn: board dimensions must be between 2 and " +
        Global::intToString(Board::MAX_LEN)
      );
    const int64_t simultaneousRows = (int64_t)batchSize * serverThreads;
    if(simultaneousRows > 4096)
      throw StringError("benchmarknn: B*S must be at most 4096");
    if((int64_t)serverThreads * numIterations > 16384)
      throw StringError("benchmarknn: S*I must be at most 16384 CUDA event pairs");

    cmd.getConfig(cfg);
  }
  catch(TCLAP::ArgException& e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  // Setup's legacy benchmark path starts ordinary server threads. Keep that
  // startup topology to one lane, stop it before timing, then install exactly S
  // benchmark lanes on the same configured device.
  cfg.overrideKey("numNNServerThreadsPerModel","1");
  cfg.overrideKey("nnRandomize","false");
  cfg.overrideKey("nnForcedSymmetry","0");
  cfg.overrideKey("debugSkipNeuralNet","false");
  cfg.overrideKey("maxBoardXSizeForNNBuffer",Global::intToString(boardXSize));
  cfg.overrideKey("maxBoardYSizeForNNBuffer",Global::intToString(boardYSize));
  cfg.overrideKey("maxBoardXSizeForNNBuffer0",Global::intToString(boardXSize));
  cfg.overrideKey("maxBoardYSizeForNNBuffer0",Global::intToString(boardYSize));
  cfg.overrideKey("requireMaxBoardSize","true");
  cfg.overrideKey("requireMaxBoardSize0","true");
  if(expectedOfficialStage1 || expectedV105Fp16) {
    cfg.overrideKey("cudaUseFP16","true");
    cfg.overrideKey("cudaUseFP16-0","true");
    cfg.overrideKey("cudaUseNHWC","true");
    cfg.overrideKey("cudaUseNHWC0","true");
    cfg.overrideKey("cudaInputsUseNHWC","true");
    cfg.overrideKey("cudaInputsUseNHWC0","true");
  }

  Logger logger(NULL,true,false,false);
  logger.write("Version " + Version::getGitRevisionWithBackend());
  cout << "BENCHMARKNN_CONFIG"
       << " B=" << batchSize
       << " S=" << serverThreads
       << " W=" << numWarmups
       << " I=" << numIterations
       << " boardX=" << boardXSize
       << " boardY=" << boardYSize
       << " exact=1"
       << " noMask=1"
       << " expectedOfficialStage1=" << (expectedOfficialStage1 ? 1 : 0)
       << " expectedV105Fp16=" << (expectedV105Fp16 ? 1 : 0)
       << " input=fixed-lane-row-distinct"
       << " policy=spatial-area-plus-pass"
       << endl;

  const string expectedSha256 = "";
  const int expectedConcurrentEvals = batchSize * serverThreads;
  const int maxConcurrentEvals = expectedConcurrentEvals + batchSize;
  const bool defaultRequireExactNNLen = true;
  const bool disableFP16 = false;

  NeuralNetSessionScope session;
  session.initialize(cfg);
  unique_ptr<NNEvaluator> nnEval(
    Setup::initializeNNEvaluator(
      modelFile,modelFile,expectedSha256,cfg,logger,seedRand,maxConcurrentEvals,expectedConcurrentEvals,
      boardXSize,boardYSize,batchSize,defaultRequireExactNNLen,disableFP16,
      Setup::SETUP_FOR_BENCHMARK
    )
  );
  nnEval->killServerThreads();

  if(nnEval->getNNXLen() != boardXSize || nnEval->getNNYLen() != boardYSize ||
     nnEval->getMaxBatchSize() != batchSize ||
     !nnEval->getBenchmarkRequireExactNNLen() ||
     !nnEval->supportsBoardSize(boardXSize,boardYSize))
    throw StringError("benchmarknn: exact board/batch configuration readback failed");
  const int actualModelVersion = nnEval->getBenchmarkModelVersion();
  const int actualInputsVersion = nnEval->getBenchmarkInputsVersion();
  const int actualSpatialFeatures = nnEval->getBenchmarkNumSpatialFeatures();
  const int actualGlobalFeatures = nnEval->getBenchmarkNumGlobalFeatures();
  const bool officialStage1InputAbi =
    actualModelVersion == 102 && actualInputsVersion == 101 &&
    actualSpatialFeatures == 22 && actualGlobalFeatures == 39;
  const bool v105InputAbi =
    actualModelVersion == 105 && actualInputsVersion == 101 &&
    actualSpatialFeatures == 22 && actualGlobalFeatures == 39;
  if(expectedOfficialStage1 && !officialStage1InputAbi)
    throw StringError("benchmarknn: expected official Stage1 requires model v102 with V101 22/39 inputs");
  if(expectedV105Fp16 && !v105InputAbi)
    throw StringError("benchmarknn: expected v105 FP16 requires model v105 with V101 22/39 inputs");
  cout << "BENCHMARKNN_TOPOLOGY_VERIFIED"
       << " B=" << nnEval->getMaxBatchSize()
       << " boardX=" << nnEval->getNNXLen()
       << " boardY=" << nnEval->getNNYLen()
       << " exact=1"
       << " noMaskRequired=1"
       << " modelVersion=" << actualModelVersion
       << " inputsVersion=" << actualInputsVersion
       << " spatialFeatures=" << actualSpatialFeatures
       << " globalFeatures=" << actualGlobalFeatures
       << " officialStage1InputAbi=" << (officialStage1InputAbi ? 1 : 0)
       << " v105InputAbi=" << (v105InputAbi ? 1 : 0)
       << endl;

  const set<int> configuredGpuIdxs = nnEval->getGpuIdxs();
  const int gpuIdx = configuredGpuIdxs.empty() ? -1 : *configuredGpuIdxs.begin();
  nnEval->setNumThreads(vector<int>((size_t)serverThreads,gpuIdx));
  if(nnEval->getNumServerThreads() != serverThreads)
    throw StringError("benchmarknn: server-thread topology readback failed");

  NNEvalFullIOBenchmarkResult full = nnEval->benchmarkFullIO(numWarmups,numIterations);
  bool allLanesOfficialStage1 = true;
  bool allLanesOfficialV105Fp16 = true;
  for(int lane = 0; lane < full.numThreads; lane++) {
    const NNEvalBenchmarkLaneProof& laneProof = full.perThreadProofs[lane];
    const NeuralNet::BenchmarkRawIOProof& raw = laneProof.rawIO;
    const NeuralNet::BenchmarkRouteProof& route = laneProof.routeAfter;
    const uint64_t serialDelta = route.invocationSerial - laneProof.routeBefore.invocationSerial;
    const bool official = isOfficialStage1Route(route);
    const bool officialV105 = isOfficialV105Fp16Route(route);
    allLanesOfficialStage1 = allLanesOfficialStage1 && official;
    allLanesOfficialV105Fp16 = allLanesOfficialV105Fp16 && officialV105;
    cout << "BENCHMARKNN_FULL_IO_LANE"
         << " lane=" << lane
         << " medianMs=" << setprecision(10) << full.perThreadMedianSeconds[lane] * 1000.0
         << " nnEvalsPerSec=" << setprecision(12) << full.perThreadNNEvalsPerSec[lane]
         << " inputChecksum=" << checksumString(raw.inputChecksum)
         << " inputContentChecksum=" << checksumString(raw.inputContentChecksum)
         << " outputChecksum=" << checksumString(raw.outputChecksum)
         << " rawFloats=" << raw.rawOutputFloatsHashed
         << " outputRowsHashed=" << raw.outputRowsHashed
         << " uniqueOutputRows=" << raw.uniqueOutputRows
         << " finite=" << (raw.outputsFinite ? 1 : 0)
         << " nonzero=" << (raw.outputsNonzero ? 1 : 0)
         << " nonconstant=" << (raw.outputsNonconstant ? 1 : 0)
         << endl;
    cout << "BENCHMARKNN_ROUTE_PROOF"
         << " mode=full-io"
         << " lane=" << lane
         << " serialSemantics=per-getOutput"
         << " serialDelta=" << serialDelta
         << " stream=" << checksumString(route.streamIdentity)
         << " batch=" << route.lastBatchSize
         << " attention=" << route.lastActiveAttention << "/" << route.expectedAttention
         << " ffn=" << route.lastActiveFfn << "/" << route.expectedFfn
         << " expectedQkn=" << route.expectedQkn
         << " preparedQkn=" << route.preparedQkn
         << " activeQkn=" << route.lastActiveQkn
         << " expectedOrderedClippedSwiGLU=" << route.expectedOrderedClippedSwiGLU
         << " preparedOrderedClippedSwiGLU=" << route.preparedOrderedClippedSwiGLU
         << " activeOrderedClippedSwiGLU=" << route.lastActiveOrderedClippedSwiGLU
         << " combinedQKV=" << route.lastActiveCombinedQKV
         << " learnedRopeFp32=" << route.lastActiveLearnedRopeFp32
         << " mma=" << route.lastActiveMma
         << " scalar=" << route.lastActiveScalar
         << " exact=" << (route.lastExact ? 1 : 0)
         << " noMask=" << (route.lastMaskNull ? 1 : 0)
         << " officialStage1=" << (official ? 1 : 0)
         << " officialV105Fp16=" << (officialV105 ? 1 : 0)
         << endl;
  }
  const bool fullFinite =
    full.outputsFinite && full.outputsNonzero && full.outputsNonconstant &&
    validRate(full.actualWallNNEvalsPerSec) &&
    full.actualWallSeconds > 0.0 && std::isfinite(full.actualWallSeconds);
  cout << "BENCHMARKNN_FULL_IO_RESULT"
       << " actualWallSeconds=" << setprecision(12) << full.actualWallSeconds
       << " actualWallNNEvalsPerSec=" << setprecision(12) << full.actualWallNNEvalsPerSec
       << " finite=" << (fullFinite ? 1 : 0)
       << " inputChecksum=" << checksumString(full.inputChecksum)
       << " outputChecksum=" << checksumString(full.outputChecksum)
       << endl;

  NNEvalDeviceOnlyBenchmarkResult device =
    nnEval->benchmarkDeviceOnly(numWarmups,numIterations);
  for(int lane = 0; lane < device.numThreads; lane++) {
    const NNEvalBenchmarkLaneProof& laneProof = device.perThreadProofs[lane];
    const NeuralNet::BenchmarkRawIOProof& raw = laneProof.rawIO;
    const NeuralNet::BenchmarkRouteProof& route = laneProof.routeAfter;
    const uint64_t serialDelta = route.invocationSerial - laneProof.routeBefore.invocationSerial;
    const bool official = isOfficialStage1Route(route);
    const bool officialV105 = isOfficialV105Fp16Route(route);
    allLanesOfficialStage1 = allLanesOfficialStage1 && official;
    allLanesOfficialV105Fp16 = allLanesOfficialV105Fp16 && officialV105;
    cout << "BENCHMARKNN_DEVICE_ONLY_LANE"
         << " lane=" << lane
         << " medianMs=" << setprecision(10) << device.perThreadMedianSeconds[lane] * 1000.0
         << " nnEvalsPerSec=" << setprecision(12) << device.perThreadNNEvalsPerSec[lane]
         << " inputChecksum=" << checksumString(raw.inputChecksum)
         << " inputContentChecksum=" << checksumString(raw.inputContentChecksum)
         << " outputChecksum=" << checksumString(raw.outputChecksum)
         << " rawFloats=" << raw.rawOutputFloatsHashed
         << " outputRowsHashed=" << raw.outputRowsHashed
         << " uniqueOutputRows=" << raw.uniqueOutputRows
         << " finite=" << (raw.outputsFinite ? 1 : 0)
         << " nonzero=" << (raw.outputsNonzero ? 1 : 0)
         << " nonconstant=" << (raw.outputsNonconstant ? 1 : 0)
         << endl;
    cout << "BENCHMARKNN_ROUTE_PROOF"
         << " mode=device-only"
         << " lane=" << lane
         << " serialSemantics=one-device-session"
         << " serialDelta=" << serialDelta
         << " stream=" << checksumString(route.streamIdentity)
         << " batch=" << route.lastBatchSize
         << " attention=" << route.lastActiveAttention << "/" << route.expectedAttention
         << " ffn=" << route.lastActiveFfn << "/" << route.expectedFfn
         << " expectedQkn=" << route.expectedQkn
         << " preparedQkn=" << route.preparedQkn
         << " activeQkn=" << route.lastActiveQkn
         << " expectedOrderedClippedSwiGLU=" << route.expectedOrderedClippedSwiGLU
         << " preparedOrderedClippedSwiGLU=" << route.preparedOrderedClippedSwiGLU
         << " activeOrderedClippedSwiGLU=" << route.lastActiveOrderedClippedSwiGLU
         << " combinedQKV=" << route.lastActiveCombinedQKV
         << " learnedRopeFp32=" << route.lastActiveLearnedRopeFp32
         << " mma=" << route.lastActiveMma
         << " scalar=" << route.lastActiveScalar
         << " exact=" << (route.lastExact ? 1 : 0)
         << " noMask=" << (route.lastMaskNull ? 1 : 0)
         << " officialStage1=" << (official ? 1 : 0)
         << " officialV105Fp16=" << (officialV105 ? 1 : 0)
         << endl;
  }
  const bool deviceFinite =
    device.outputsFinite && device.outputsNonzero && device.outputsNonconstant &&
    validRate(device.combinedNNEvalsPerSec) &&
    device.combinedWallSeconds > 0.0 && std::isfinite(device.combinedWallSeconds);
  cout << "BENCHMARKNN_DEVICE_ONLY_RESULT"
       << " combinedWallSeconds=" << setprecision(12) << device.combinedWallSeconds
       << " combinedNNEvalsPerSec=" << setprecision(12) << device.combinedNNEvalsPerSec
       << " finite=" << (deviceFinite ? 1 : 0)
       << " inputChecksum=" << checksumString(device.inputChecksum)
       << " outputChecksum=" << checksumString(device.outputChecksum)
       << endl;

  bool perLaneProofMatch = full.numThreads == device.numThreads;
  if(perLaneProofMatch) {
    for(int lane = 0; lane < full.numThreads; lane++) {
      const NeuralNet::BenchmarkRouteProof& fullRoute = full.perThreadProofs[lane].routeAfter;
      const NeuralNet::BenchmarkRouteProof& deviceRoute = device.perThreadProofs[lane].routeAfter;
      perLaneProofMatch = perLaneProofMatch &&
        sameRawProof(full.perThreadProofs[lane].rawIO,device.perThreadProofs[lane].rawIO) &&
        sameRouteFamily(fullRoute,deviceRoute);
    }
  }
  const bool inputChecksumMatch = full.inputChecksum == device.inputChecksum;
  const bool outputChecksumMatch = full.outputChecksum == device.outputChecksum;
  const bool stage1RoutePass = !expectedOfficialStage1 || allLanesOfficialStage1;
  const bool v105RoutePass = !expectedV105Fp16 || allLanesOfficialV105Fp16;
  const bool requestedRoutePass = stage1RoutePass && v105RoutePass;
  // Retain the original field name as a schema-v1 compatibility alias.
  const bool officialRoutePass = requestedRoutePass;
  const bool integrityPass =
    fullFinite && deviceFinite && inputChecksumMatch && outputChecksumMatch &&
    perLaneProofMatch && officialRoutePass;
  cout << "BENCHMARKNN_INTEGRITY"
       << " finite=" << (fullFinite && deviceFinite ? 1 : 0)
       << " inputChecksumMatch=" << (inputChecksumMatch ? 1 : 0)
       << " outputChecksumMatch=" << (outputChecksumMatch ? 1 : 0)
       << " perLaneProofMatch=" << (perLaneProofMatch ? 1 : 0)
       << " expectedOfficialStage1=" << (expectedOfficialStage1 ? 1 : 0)
       << " allLanesOfficialStage1=" << (allLanesOfficialStage1 ? 1 : 0)
       << " expectedV105Fp16=" << (expectedV105Fp16 ? 1 : 0)
       << " allLanesOfficialV105Fp16=" << (allLanesOfficialV105Fp16 ? 1 : 0)
       << " stage1RoutePass=" << (stage1RoutePass ? 1 : 0)
       << " v105RoutePass=" << (v105RoutePass ? 1 : 0)
       << " requestedRoutePass=" << (requestedRoutePass ? 1 : 0)
       << " officialRoutePass=" << (officialRoutePass ? 1 : 0)
       << endl;

  ostringstream json;
  json << setprecision(17);
  json << "{"
       << "\"schema\":\"katago.benchmarknn.v1\""
       << ",\"status\":\"" << (integrityPass ? "pass" : "fail") << "\""
       << ",\"config\":{"
       << "\"batchSize\":" << batchSize
       << ",\"serverThreads\":" << serverThreads
       << ",\"warmupIterations\":" << numWarmups
       << ",\"timedIterations\":" << numIterations
       << ",\"eventPairBudget\":16384"
       << ",\"board\":{"
       << "\"x\":" << boardXSize
       << ",\"y\":" << boardYSize
       << ",\"exact\":true"
       << ",\"noMask\":true"
       << "}"
       << ",\"policyShape\":\"spatialAreaPlusPass\""
       << ",\"policyFloatsPerRow\":" << boardXSize * boardYSize + 1
       << ",\"expectedOfficialStage1\":" << jsonBool(expectedOfficialStage1)
       << ",\"expectedV105Fp16\":" << jsonBool(expectedV105Fp16)
       << "}"
       << ",\"model\":{"
       << "\"modelVersion\":" << actualModelVersion
       << ",\"inputsVersion\":" << actualInputsVersion
       << ",\"spatialFeatures\":" << actualSpatialFeatures
       << ",\"globalFeatures\":" << actualGlobalFeatures
       << ",\"officialStage1InputAbi\":" << jsonBool(officialStage1InputAbi)
       << ",\"v105InputAbi\":" << jsonBool(v105InputAbi)
       << "}"
       << ",\"fullIO\":{"
       << "\"wallMetric\":\"actualWallNNEvalsPerSec\""
       << ",\"actualWallSeconds\":" << full.actualWallSeconds
       << ",\"actualWallNNEvalsPerSec\":" << full.actualWallNNEvalsPerSec
       << ",\"serialExpectedDelta\":" << (uint64_t)numWarmups + (uint64_t)numIterations
       << ",\"inputChecksum\":\"" << checksumString(full.inputChecksum) << "\""
       << ",\"outputChecksum\":\"" << checksumString(full.outputChecksum) << "\""
       << ",\"finite\":" << jsonBool(full.outputsFinite)
       << ",\"nonzero\":" << jsonBool(full.outputsNonzero)
       << ",\"nonconstant\":" << jsonBool(full.outputsNonconstant)
       << ",\"lanes\":[";
  for(int lane = 0; lane < full.numThreads; lane++) {
    if(lane > 0)
      json << ",";
    appendBenchmarkLaneJson(
      json,lane,full.perThreadMedianSeconds[lane],full.perThreadNNEvalsPerSec[lane],
      full.perThreadProofs[lane],"per-getOutput"
    );
  }
  json << "]}"
       << ",\"deviceOnly\":{"
       << "\"wallMetric\":\"combinedNNEvalsPerSec\""
       << ",\"combinedWallSeconds\":" << device.combinedWallSeconds
       << ",\"combinedNNEvalsPerSec\":" << device.combinedNNEvalsPerSec
       << ",\"serialExpectedDelta\":1"
       << ",\"inputChecksum\":\"" << checksumString(device.inputChecksum) << "\""
       << ",\"outputChecksum\":\"" << checksumString(device.outputChecksum) << "\""
       << ",\"finite\":" << jsonBool(device.outputsFinite)
       << ",\"nonzero\":" << jsonBool(device.outputsNonzero)
       << ",\"nonconstant\":" << jsonBool(device.outputsNonconstant)
       << ",\"lanes\":[";
  for(int lane = 0; lane < device.numThreads; lane++) {
    if(lane > 0)
      json << ",";
    appendBenchmarkLaneJson(
      json,lane,device.perThreadMedianSeconds[lane],device.perThreadNNEvalsPerSec[lane],
      device.perThreadProofs[lane],"one-device-session"
    );
  }
  json << "]}"
       << ",\"integrity\":{"
       << "\"finite\":" << jsonBool(fullFinite && deviceFinite)
       << ",\"inputChecksumMatch\":" << jsonBool(inputChecksumMatch)
       << ",\"outputChecksumMatch\":" << jsonBool(outputChecksumMatch)
       << ",\"perLaneProofMatch\":" << jsonBool(perLaneProofMatch)
       << ",\"officialRouteRequired\":" << jsonBool(expectedOfficialStage1 || expectedV105Fp16)
       << ",\"expectedOfficialStage1\":" << jsonBool(expectedOfficialStage1)
       << ",\"allLanesOfficialStage1\":" << jsonBool(allLanesOfficialStage1)
       << ",\"stage1RoutePass\":" << jsonBool(stage1RoutePass)
       << ",\"expectedV105Fp16\":" << jsonBool(expectedV105Fp16)
       << ",\"allLanesOfficialV105Fp16\":" << jsonBool(allLanesOfficialV105Fp16)
       << ",\"v105RoutePass\":" << jsonBool(v105RoutePass)
       << ",\"requestedRoutePass\":" << jsonBool(requestedRoutePass)
       << ",\"officialRoutePass\":" << jsonBool(officialRoutePass)
       << "}"
       << "}";
  cout << "BENCHMARKNN_JSON " << json.str() << endl;

  if(!integrityPass)
    throw StringError("benchmarknn: input/output/route integrity contract failed");
  cout << "BENCHMARKNN_PASS"
       << " finite=1"
       << " inputChecksumMatch=1"
       << " outputChecksumMatch=1"
       << " perLaneProofMatch=1"
       << " stage1RoutePass=1"
       << " v105RoutePass=1"
       << " requestedRoutePass=1"
       << " officialRoutePass=1"
       << endl;
  return 0;
}

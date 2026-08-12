#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/logger.h"
#include "../core/rand.h"
#include "../dataio/sgf.h"
#include "../neuralnet/nneval.h"
#include "../program/setup.h"
#include "../command/commandline.h"
#include "../main.h"

#include <iomanip>
#include <sstream>

using namespace std;

static string jsonEscape(const string& s) {
  ostringstream out;
  for(char c : s) {
    if(c == '"' || c == '\\')
      out << '\\' << c;
    else if(c == '\n')
      out << "\\n";
    else if(c == '\t')
      out << "\\t";
    else if(c == '\r')
      out << "\\r";
    else
      out << c;
  }
  return out.str();
}

int MainCmds::benchmarknn(const vector<string>& args) {
  Board::initHash();
  Rand seedRand;

  ConfigParser cfg;
  string modelFile;
  int numIterations = 100;
  int numWarmups = 10;
  int batchSizeOverride = -1;
  int boardSize = 15;
  bool jsonOut = false;
  bool forceMaskAllOnes = false;

  try {
    KataGoCommandLine cmd(
      "Benchmark pure neural-net forward throughput. Honors config settings for batch size, "
      "one NN server thread and its GPU assignment. Excludes feature generation, "
      "postprocessing, H2D/D2H, and search."
    );
    cmd.addConfigFileArg(KataGoCommandLine::defaultGtpConfigFileName(),"gtp_example.cfg");
    cmd.addModelFileArg();
    TCLAP::ValueArg<int> iterationsArg(
      "","iterations","Number of timed forward passes per server thread (default 100)",
      false,100,"N"
    );
    TCLAP::ValueArg<int> warmupArg(
      "","warmup","Warmup forward passes per server thread before timing (default 10)",
      false,10,"N"
    );
    TCLAP::ValueArg<int> batchSizeArg(
      "","batch-size","Override nnMaxBatchSize from config (default: use config, or 16)",
      false,-1,"N"
    );
    TCLAP::ValueArg<int> boardSizeArg(
      "","boardsize","NN board size from 2 through the compiled maximum (default: compiled maximum)",
      false,Board::MAX_LEN,"N"
    );
    TCLAP::SwitchArg jsonArg("","json","Print results as JSON",false);
    TCLAP::SwitchArg forceMaskAllOnesArg(
      "","force-mask-all-ones",
      "On an exact board, force the CUDA backend to receive a non-null all-ones mask",
      false
    );
    cmd.add(iterationsArg);
    cmd.add(warmupArg);
    cmd.add(batchSizeArg);
    cmd.add(boardSizeArg);
    cmd.add(jsonArg);
    cmd.add(forceMaskAllOnesArg);
    cmd.setShortUsageArgLimit();
    cmd.addOverrideConfigArg();

    cmd.parseArgs(args);

    modelFile = cmd.getModelFile();
    numIterations = iterationsArg.getValue();
    numWarmups = warmupArg.getValue();
    batchSizeOverride = batchSizeArg.getValue();
    boardSize = boardSizeArg.getValue();
    jsonOut = jsonArg.getValue();
    forceMaskAllOnes = forceMaskAllOnesArg.getValue();
    cmd.getConfig(cfg);

    if(numIterations <= 0)
      throw StringError("benchmarknn: iterations must be > 0");
    if(numWarmups < 0)
      throw StringError("benchmarknn: warmup must be >= 0");
    if(boardSize < 2 || boardSize > Board::MAX_LEN)
      throw StringError(
        "benchmarknn: boardsize must be between 2 and " +
        Global::intToString(Board::MAX_LEN)
      );
  }
  catch(TCLAP::ArgException& e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  const bool logToStdoutDefault = true;
  const bool logToStderrDefault = false;
  const bool logTimeDefault = false;
  Logger logger(NULL, logToStdoutDefault, logToStderrDefault, logTimeDefault);
  logger.write("Version " + Version::getGitRevisionWithBackend());
  logger.write("benchmarknn model " + modelFile);
  logger.write("benchmarknn board size " + Global::intToString(boardSize));

  const string expectedSha256 = "";
  int maxBatchSize =
    batchSizeOverride > 0 ? batchSizeOverride :
    cfg.contains("nnMaxBatchSize") ? cfg.getInt("nnMaxBatchSize",1,65536) :
    16;
  logger.write("Using batch size " + Global::intToString(maxBatchSize));

  const int expectedConcurrentEvals = maxBatchSize;
  const int maxConcurrentEvals = maxBatchSize * 64;
  const bool defaultRequireExactNNLen = true;
  const bool disableFP16 = false;

  NNEvaluator* nnEval = NULL;
  try {
    nnEval = Setup::initializeNNEvaluator(
      modelFile,modelFile,expectedSha256,cfg,logger,seedRand,maxConcurrentEvals,expectedConcurrentEvals,
      boardSize,boardSize,maxBatchSize,defaultRequireExactNNLen,disableFP16,
      Setup::SETUP_FOR_BENCHMARK
    );
    // The setup path spawns the normal evaluation server thread; benchmarknn creates a
    // fresh handle after stopping it so the timed loop is pure forward-only.
    nnEval->killServerThreads();

    NNEvalBenchmarkResult result = nnEval->benchmarkPureForward(
      numWarmups,numIterations,forceMaskAllOnes
    );

    if(jsonOut) {
      cout << "{";
      cout << "\"modelFile\":\"" << jsonEscape(nnEval->getModelFileName()) << "\",";
      cout << "\"modelName\":\"" << jsonEscape(nnEval->getInternalModelName()) << "\",";
      cout << "\"revision\":\"" << jsonEscape(Version::getGitRevisionWithBackend()) << "\",";
      cout << "\"batchSize\":" << result.batchSize << ",";
      cout << "\"numServerThreads\":" << result.numServerThreads << ",";
      cout << "\"numIterations\":" << result.numIterations << ",";
      cout << "\"forceMaskAllOnes\":" << (result.forcedMaskAllOnes ? "true" : "false") << ",";
      cout << "\"gpuIdxs\":[";
      bool first = true;
      for(int g : nnEval->getGpuIdxs()) {
        if(!first)
          cout << ",";
        first = false;
        cout << g;
      }
      cout << "],";
      cout << "\"perServerMedianMs\":[";
      for(int i = 0; i < result.numServerThreads; i++) {
        if(i > 0)
          cout << ",";
        cout << setprecision(10) << result.perServerMedianSeconds[i] * 1000.0;
      }
      cout << "],";
      cout << "\"perServerNNEvalsPerSec\":[";
      for(int i = 0; i < result.numServerThreads; i++) {
        if(i > 0)
          cout << ",";
        cout << setprecision(10) << result.perServerNNEvalsPerSec[i];
      }
      cout << "],";
      cout << "\"combinedWallSeconds\":" << setprecision(10) << result.combinedWallSeconds << ",";
      cout << "\"combinedNNEvalsPerSec\":" << setprecision(10) << result.combinedNNEvalsPerSec;
      cout << "}" << endl;
    }
    else {
      cout << "=== benchmarknn ===" << endl;
      cout << "model: " << nnEval->getModelFileName() << endl;
      cout << "internal model: " << nnEval->getInternalModelName() << endl;
      cout << "revision/backend: " << Version::getGitRevisionWithBackend() << endl;
      cout << "batch size per server: " << result.batchSize << endl;
      cout << "NN server threads: " << result.numServerThreads << endl;
      cout << "GPU indices:";
      for(int g : nnEval->getGpuIdxs())
        cout << " " << g;
      cout << endl;
      cout << "timed iterations per server: " << result.numIterations << endl;
      cout << "attention mask: "
           << (result.forcedMaskAllOnes ? "forced non-null all-ones" : "default exact-board behavior")
           << endl;
      cout << "combined concurrent evaluations: " << result.numServerThreads * result.batchSize << endl;
      for(int i = 0; i < result.numServerThreads; i++) {
        cout << "server " << i << ": "
             << setprecision(6) << result.perServerMedianSeconds[i] * 1000.0
             << " ms/batch, " << setprecision(10) << result.perServerNNEvalsPerSec[i]
             << " nnEval/s" << endl;
      }
      cout << "combined wall time: " << setprecision(6) << result.combinedWallSeconds << " s" << endl;
      cout << "combined throughput: " << setprecision(10) << result.combinedNNEvalsPerSec
           << " nnEval/s" << endl;
    }
  }
  catch(...) {
    delete nnEval;
    NeuralNet::globalCleanup();
    throw;
  }

  delete nnEval;
  NeuralNet::globalCleanup();
  return 0;
}

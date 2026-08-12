
#include <sstream>
#include "../core/global.h"
#include "../core/bsearch.h"
#include "../core/rand.h"
#include "../core/elo.h"
#include "../core/fancymath.h"
#include "../core/config_parser.h"
#include "../core/fileutils.h"
#include "../core/base64.h"
#include "../core/timer.h"
#include "../core/threadtest.h"
#include "../game/board.h"
#include "../game/rules.h"
#include "../game/boardhistory.h"
#include "../neuralnet/nninputs.h"
#include "../program/gtpconfig.h"
#include "../program/setup.h"
#include "../tests/tests.h"
#include "../command/commandline.h"
#include "../main.h"

using namespace std;

int MainCmds::testarchitecturedesc(const vector<string>& args) {
  if(args.size() != 1 || args[0] != "testarchitecturedesc")
    throw StringError("testarchitecturedesc takes no arguments");
  Tests::runArchitectureDescTests();
  return 0;
}

int MainCmds::testtransformerproductionplan(const vector<string>& args) {
  if(args.size() != 1 || args[0] != "testtransformerproductionplan")
    throw StringError("testtransformerproductionplan takes no arguments");
  Tests::runTransformerProductionPlanTests();
  return 0;
}

int MainCmds::testbatchawaredispatch(const vector<string>& args) {
  if(args.size() != 1 || args[0] != "testbatchawaredispatch")
    throw StringError("testbatchawaredispatch takes no arguments");
  Tests::runBatchAwareDispatchTests();
  return 0;
}

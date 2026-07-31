
//-------------------------------------------------------------------------------------
//This file contains the main core logic of the search.
//-------------------------------------------------------------------------------------

#include "../search/search.h"

#include <algorithm>
#include <numeric>

#include "../core/fancymath.h"
#include "../core/timer.h"
#include "../game/graphhash.h"
#include "../search/distributiontable.h"
#include "../search/patternbonustable.h"
#include "../search/searchnode.h"
#include "../search/searchnodetable.h"
#include "../search/subtreevaluebiastable.h"

using namespace std;

//-----------------------------------------------------------------------------------------

static string makeSeed(const Search& search, int threadIdx) {
  stringstream ss;
  ss << search.randSeed;
  ss << "$searchThread$";
  ss << threadIdx;
  ss << "$";
  ss << search.rootBoard.pos_hash;
  ss << "$";
  ss << search.rootHistory.moveHistory.size();
  ss << "$";
  ss << search.numSearchesBegun;
  return ss.str();
}

SearchThread::SearchThread(int tIdx, const Search& search)
  :threadIdx(tIdx),
   pla(search.rootPla),board(search.rootBoard),
   history(search.rootHistory),
   normalRulesBoard(search.rootBoard),
   normalRulesHistory(search.rootHistory),
   graphHash(search.rootGraphHash),
   graphPath(),
   rand(makeSeed(search,tIdx)),
   nnResultBuf(),
   statsBuf(),
   upperBoundVisitsLeft(1e30),
   vctAttacker(C_EMPTY),
   normalObjective(C_EMPTY),
   policyGuidanceMode(POLICY_GUIDANCE_NONE),
   policyGuidanceAttacker(C_EMPTY),
   rootVctNormalVerificationMoveLoc(Board::NULL_LOC),
   rootForcedReplySidecarMoveLoc(Board::NULL_LOC),
   oldNNOutputsToCleanUp(),
   illegalMoveHashes()
{
  statsBuf.resize(NNPos::MAX_NN_POLICY_SIZE);
  graphPath.reserve(256);

  //Reserving even this many is almost certainly overkill but should guarantee that we never have hit allocation here.
  oldNNOutputsToCleanUp.reserve(8);
}
SearchThread::~SearchThread() {
  for(size_t i = 0; i<oldNNOutputsToCleanUp.size(); i++)
    delete oldNNOutputsToCleanUp[i];
  oldNNOutputsToCleanUp.resize(0);
}

//-----------------------------------------------------------------------------------------

static const double VALUE_WEIGHT_DEGREES_OF_FREEDOM = 3.0;

static void failIfInvalidMultiValueHeadParams(const SearchParams& params) {
  if(params.multiValueHeadUtilityMix != 0.0 && params.noResultUtilityReduce != 0.0)
    throw StringError("multiValueHeadUtilityMix requires noResultUtilityReduce to be 0");
  if(params.multiValueHeadSelectionBias < -1.0 || params.multiValueHeadSelectionBias > 1.0)
    throw StringError("multiValueHeadSelectionBias must be between -1 and 1");
  if(params.multiHeadNormalPolicyHead1Mix < 0.0 || params.multiHeadNormalPolicyHead1Mix > 1.0)
    throw StringError("multiHeadNormalPolicyHead1Mix must be between 0 and 1");
  if(params.multiHeadObjectiveSearchStrength < 0.0 || params.multiHeadObjectiveSearchStrength > 1.0)
    throw StringError("multiHeadObjectiveSearchStrength must be between 0 and 1");
  if(params.multiHeadObjectiveSearchStrength > 0.0 && params.multiValueHeadUtilityMix == 0.0)
    throw StringError("multiHeadObjectiveSearchStrength requires multiValueHeadUtilityMix to be nonzero");
  if(params.multiHeadObjectiveSelectionPower <= 0.0)
    throw StringError("multiHeadObjectiveSelectionPower must be positive");
  if(
    params.multiHeadObjectiveSelectionSharpness < 0.0 ||
    params.multiHeadObjectiveSelectionSharpness > 16.0
  )
    throw StringError("multiHeadObjectiveSelectionSharpness must be between 0 and 16");
  if(params.multiHeadObjectivePolicyMix < 0.0 || params.multiHeadObjectivePolicyMix > 1.0)
    throw StringError("multiHeadObjectivePolicyMix must be between 0 and 1");
  if(params.multiHeadObjectivePolicyMix > 0.0 && params.multiValueHeadUtilityMix == 0.0)
    throw StringError("multiHeadObjectivePolicyMix requires multiValueHeadUtilityMix to be nonzero");
  if(params.multiHeadDrawPolicyFlattening < 0.0 || params.multiHeadDrawPolicyFlattening > 1.0)
    throw StringError("multiHeadDrawPolicyFlattening must be between 0 and 1");
  if(params.multiHeadDrawPolicyFlattening > 0.0 && params.multiValueHeadUtilityMix == 0.0)
    throw StringError("multiHeadDrawPolicyFlattening requires multiValueHeadUtilityMix to be nonzero");
  if(params.multiHeadDrawRootMinVisitsCoeff < 0.0)
    throw StringError("multiHeadDrawRootMinVisitsCoeff must be nonnegative");
  if(params.multiHeadDrawRootMinVisitsCoeff > 0.0 && params.multiValueHeadUtilityMix == 0.0)
    throw StringError("multiHeadDrawRootMinVisitsCoeff requires multiValueHeadUtilityMix to be nonzero");
  if(params.multiHeadDrawAuxRootVisits > 0.0 && params.multiValueHeadUtilityMix == 0.0)
    throw StringError("multiHeadDrawAuxRootVisits requires multiValueHeadUtilityMix to be nonzero");
  if(params.multiHeadTacticalDisproofStrength < 0.0)
    throw StringError("multiHeadTacticalDisproofStrength must be nonnegative");
  if(params.multiHeadTacticalPolicyPower <= 0.0)
    throw StringError("multiHeadTacticalPolicyPower must be positive");
  if(params.multiHeadDrawForcedReplyRootVisits < 0.0)
    throw StringError("multiHeadDrawForcedReplyRootVisits must be nonnegative");
  if(params.multiHeadDrawForcedReplyRootVisits > 0.0 && params.multiValueHeadUtilityMix == 0.0)
    throw StringError("multiHeadDrawForcedReplyRootVisits requires multiValueHeadUtilityMix to be nonzero");
  if(
    params.multiHeadDrawForcedReplySidecarVisits > 0.0 &&
    params.multiHeadDrawForcedReplySidecarMaxProp <= 0.0
  )
    throw StringError("multiHeadDrawForcedReplySidecarVisits requires multiHeadDrawForcedReplySidecarMaxProp");
  if(
    params.multiHeadDrawForcedReplySidecarMaxProp > 0.0 &&
    params.multiHeadDrawForcedReplySidecarVisits <= 0.0
  )
    throw StringError("multiHeadDrawForcedReplySidecarMaxProp requires multiHeadDrawForcedReplySidecarVisits");
  if(
    params.multiHeadDrawForcedReplySidecarVisits > 0.0 &&
    !params.multiHeadVctUseNormalRules
  )
    throw StringError("forced-reply sidecar search requires multiHeadVctUseNormalRules");
  if(
    params.multiHeadDrawForcedReplyPolicyThreshold < 0.0 ||
    params.multiHeadDrawForcedReplyPolicyThreshold >= 1.0
  )
    throw StringError("multiHeadDrawForcedReplyPolicyThreshold must be between 0 and 1");
  if(params.multiHeadObjectiveSeparatePlayouts && params.multiValueHeadUtilityMix == 0.0)
    throw StringError("multiHeadObjectiveSeparatePlayouts requires multiValueHeadUtilityMix to be nonzero");
  if(params.multiHeadObjectiveCrossWeight < 0.0 || params.multiHeadObjectiveCrossWeight > 1.0)
    throw StringError("multiHeadObjectiveCrossWeight must be between 0 and 1");
  if(params.multiHeadObjectiveCrossWeight > 0.0 && !params.multiHeadObjectiveSeparatePlayouts)
    throw StringError("multiHeadObjectiveCrossWeight requires multiHeadObjectiveSeparatePlayouts");
  if(params.multiHeadObjectiveValueWeightExponent < 0.0)
    throw StringError("multiHeadObjectiveValueWeightExponent must be nonnegative");
  if(params.multiHeadObjectiveVctWeight < 0.0)
    throw StringError("multiHeadObjectiveVctWeight must be nonnegative");
  if(
    params.multiHeadVctObjectiveBudgetMix < 0.0 ||
    params.multiHeadVctObjectiveBudgetMix > 1.0
  )
    throw StringError("multiHeadVctObjectiveBudgetMix must be between 0 and 1");
  if(
    params.multiHeadVctProbeStartFraction < 0.0 ||
    params.multiHeadVctProbeStartFraction > 1.0 ||
    params.multiHeadVctProbeEndFraction < 0.0 ||
    params.multiHeadVctProbeEndFraction > 1.0 ||
    params.multiHeadVctProbeRampFraction < 0.0 ||
    params.multiHeadVctProbeRampFraction > 1.0
  )
    throw StringError("multi-head VCT probe fractions must be between 0 and 1");
  if(params.multiHeadVctProbeEndFraction < params.multiHeadVctProbeStartFraction)
    throw StringError("multiHeadVctProbeEndFraction must be at least multiHeadVctProbeStartFraction");
  if(params.multiHeadVctPriorVisits < 0.0)
    throw StringError("multiHeadVctPriorVisits must be nonnegative");
  if(
    params.multiHeadVctNormalVerificationProp < 0.0 ||
    params.multiHeadVctNormalVerificationProp > 0.9
  )
    throw StringError("multiHeadVctNormalVerificationProp must be between 0 and 0.9");
}

Search::Search(SearchParams params, NNEvaluator* nnEval, Logger* lg, const string& rSeed)
  :rootPla(P_BLACK),
   rootBoard(),
   rootHistory(),
   rootGraphHash(),
   rootHintLoc(Board::NULL_LOC),
   avoidMoveUntilByLocBlack(),avoidMoveUntilByLocWhite(),avoidMoveUntilRescaleRoot(false),
   rootSymmetries(),
   rootPruneOnlySymmetries(),
   searchParams(params),numSearchesBegun(0),searchNodeAge(0),
   plaThatSearchIsFor(C_EMPTY),plaThatSearchIsForLastSearch(C_EMPTY),
   lastSearchNumPlayouts(0),
   effectiveSearchTimeCarriedOver(0.0),
   rootNormalVisitsAtSearchStart(0),
   rootWhiteWinEdgeVisitsAtSearchStart(0),
   rootBlackWinEdgeVisitsAtSearchStart(0),
   rootWhiteVctVisitsAtSearchStart(0),
   rootBlackVctVisitsAtSearchStart(0),
   rootVctNormalVerificationPlayouts(0),
   currentSearchPlayoutBudget(0),
   randSeed(rSeed),
   valueWeightDistribution(NULL),
   patternBonusTable(NULL),
   externalPatternBonusTable(nullptr),
   nonSearchRand(rSeed + string("$nonSearchRand")),
   logger(lg),
   nnEvaluator(nnEval),
   nnXLen(),
   nnYLen(),
   policySize(),
   rootNode(NULL),
   nodeTable(NULL),
   mutexPool(NULL),
   subtreeValueBiasTable(NULL),
   numThreadsSpawned(0),
   threads(NULL),
   threadTasks(NULL),
   threadTasksRemaining(NULL),
   oldNNOutputsToCleanUpMutex(),
   oldNNOutputsToCleanUp()
{
  assert(logger != NULL);
  failIfInvalidMultiValueHeadParams(searchParams);
  nnXLen = nnEval->getNNXLen();
  nnYLen = nnEval->getNNYLen();
  assert(nnXLen > 0 && nnXLen <= NNPos::MAX_BOARD_LEN);
  assert(nnYLen > 0 && nnYLen <= NNPos::MAX_BOARD_LEN);
  policySize = NNPos::getPolicySize(nnXLen,nnYLen);


  valueWeightDistribution = new DistributionTable(
    [](double z) { return FancyMath::tdistpdf(z,VALUE_WEIGHT_DEGREES_OF_FREEDOM); },
    [](double z) { return FancyMath::tdistcdf(z,VALUE_WEIGHT_DEGREES_OF_FREEDOM); },
    -50.0,
    50.0,
    2000
  );

  rootNode = NULL;
  nodeTable = new SearchNodeTable(params.nodeTableShardsPowerOfTwo);
  mutexPool = new MutexPool(nodeTable->mutexPool->getNumMutexes());

  rootHistory.clear(rootBoard,rootPla,Rules());
}

Search::~Search() {
  clearSearch();

  delete valueWeightDistribution;

  delete nodeTable;
  delete mutexPool;
  delete subtreeValueBiasTable;
  delete patternBonusTable;
  killThreads();
}

const Board& Search::getRootBoard() const {
  return rootBoard;
}
const BoardHistory& Search::getRootHist() const {
  return rootHistory;
}
Player Search::getRootPla() const {
  return rootPla;
}

Player Search::getPlayoutDoublingAdvantagePla() const {
  return searchParams.playoutDoublingAdvantagePla == C_EMPTY ? plaThatSearchIsFor : searchParams.playoutDoublingAdvantagePla;
}

int Search::getPos(Loc moveLoc) const {
  return NNPos::locToPos(moveLoc,rootBoard.x_size,nnXLen,nnYLen);
}

void Search::setPosition(Player pla, const Board& board, const BoardHistory& history) {
  clearSearch();
  rootPla = pla;
  plaThatSearchIsFor = C_EMPTY;
  rootBoard = board;
  rootHistory = history;
  avoidMoveUntilByLocBlack.clear();
  avoidMoveUntilByLocWhite.clear();
}

void Search::setPlayerAndClearHistory(Player pla) {
  clearSearch();
  rootPla = pla;
  plaThatSearchIsFor = C_EMPTY;
  Rules rules = rootHistory.rules;
  rootHistory.clear(rootBoard,rootPla,rules);


  //If changing the player alone, don't clear these, leave the user's setting - the user may have tried
  //to adjust the player or will be calling runWholeSearchAndGetMove with a different player and will
  //still want avoid moves to apply.
  //avoidMoveUntilByLocBlack.clear();
  //avoidMoveUntilByLocWhite.clear();
}

void Search::setPlayerIfNew(Player pla) {
  if(pla != rootPla)
    setPlayerAndClearHistory(pla);
}


void Search::setAvoidMoveUntilByLoc(const std::vector<int>& bVec, const std::vector<int>& wVec) {
  if(avoidMoveUntilByLocBlack == bVec && avoidMoveUntilByLocWhite == wVec)
    return;
  clearSearch();
  avoidMoveUntilByLocBlack = bVec;
  avoidMoveUntilByLocWhite = wVec;
}

void Search::setAvoidMoveUntilRescaleRoot(bool b) {
  avoidMoveUntilRescaleRoot = b;
}

void Search::setRootHintLoc(Loc loc) {
  //When we positively change the hint loc, we clear the search to make absolutely sure
  //that the hintloc takes effect, and that all nnevals (including the root noise that adds the hintloc) has a chance to happen
  if(loc != Board::NULL_LOC && rootHintLoc != loc)
    clearSearch();
  rootHintLoc = loc;
}


void Search::setRootSymmetryPruningOnly(const std::vector<int>& v) {
  if(rootPruneOnlySymmetries == v)
    return;
  clearSearch();
  rootPruneOnlySymmetries = v;
}


void Search::setParams(SearchParams params) {
  failIfInvalidMultiValueHeadParams(params);
  clearSearch();
  searchParams = params;
}

void Search::setParamsNoClearing(SearchParams params) {
  failIfInvalidMultiValueHeadParams(params);
  searchParams = params;
}

void Search::setExternalPatternBonusTable(std::unique_ptr<PatternBonusTable>&& table) {
  if(table == externalPatternBonusTable)
    return;
  //Probably not actually needed so long as we do a fresh search to refresh and use the new table
  //but this makes behavior consistent with all the other setters.
  clearSearch();
  externalPatternBonusTable = std::move(table);
}

void Search::setCopyOfExternalPatternBonusTable(const std::unique_ptr<PatternBonusTable>& table) {
  setExternalPatternBonusTable(table == nullptr ? nullptr : std::make_unique<PatternBonusTable>(*table));
}

void Search::setNNEval(NNEvaluator* nnEval) {
  clearSearch();
  nnEvaluator = nnEval;
  nnXLen = nnEval->getNNXLen();
  nnYLen = nnEval->getNNYLen();
  assert(nnXLen > 0 && nnXLen <= NNPos::MAX_BOARD_LEN);
  assert(nnYLen > 0 && nnYLen <= NNPos::MAX_BOARD_LEN);
  policySize = NNPos::getPolicySize(nnXLen,nnYLen);
}

void Search::clearSearch() {
  effectiveSearchTimeCarriedOver = 0.0;
  if(rootNode != NULL) {
    deleteAllTableNodesMulithreaded();
    //Root is not stored in node table
    if(rootNode != NULL) {
      delete rootNode;
      rootNode = NULL;
    }
  }
  clearOldNNOutputs();
  searchNodeAge = 0;
}

bool Search::isLegalTolerant(Loc moveLoc, Player movePla) const {
  //If we somehow have the same player making multiple moves in a row (possible in GTP or an sgf file),
  //clear the ko loc - the simple ko loc of a player should not prohibit the opponent playing there!
  if(movePla != rootPla) {
    Board copy = rootBoard;
    return copy.isLegal(moveLoc,movePla);
  }
  else {
    return rootHistory.isLegalTolerant(rootBoard,moveLoc,movePla);
  }
}

bool Search::isLegalStrict(Loc moveLoc, Player movePla) const {
  return movePla == rootPla && rootHistory.isLegal(rootBoard,moveLoc,movePla);
}


bool Search::makeMove(Loc moveLoc, Player movePla) {
  if(!isLegalTolerant(moveLoc,movePla))
    return false;

  if(movePla != rootPla)
    setPlayerAndClearHistory(movePla);

  if(rootNode != NULL) {
    bool foundChild = false;
    int foundChildIdx = -1;

    int childrenCapacity;
    SearchChildPointer* children = rootNode->getChildren(childrenCapacity);
    int numChildren = 0;
    for(int i = 0; i<childrenCapacity; i++) {
      SearchNode* child = children[i].getIfAllocated();
      if(child == NULL)
        break;
      numChildren++;
      if(!foundChild && children[i].getMoveLocRelaxed() == moveLoc) {
        foundChild = true;
        foundChildIdx = i;
      }
    }

    //Just in case, make sure the child has an nnOutput, otherwise no point keeping it.
    //This is a safeguard against any oddity involving node preservation into states that
    //were considered terminal.
    if(foundChild) {
      SearchNode* child = children[foundChildIdx].getIfAllocated();
      assert(child != NULL);
      NNOutput* nnOutput = child->getNNOutput();
      if(nnOutput == NULL)
        foundChild = false;
    }

    if(foundChild) {
      SearchNode* child = children[foundChildIdx].getIfAllocated();
      assert(child != NULL);

      //Account for time carried over
      {
        int64_t rootVisits = rootNode->stats.visits.load(std::memory_order_acquire);
        int64_t childVisits = child->stats.visits.load(std::memory_order_acquire);
        double visitProportion = (double)childVisits / (double)rootVisits;
        if(visitProportion > 1)
          visitProportion = 1;
        effectiveSearchTimeCarriedOver = effectiveSearchTimeCarriedOver * visitProportion * searchParams.treeReuseCarryOverTimeFactor;
      }

      //Okay, this is now our new root! Create a copy so as to keep the root out of the node table.
      const bool copySubtreeValueBias = false;
      const bool forceNonTerminal = true;
      rootNode = new SearchNode(*child, forceNonTerminal, copySubtreeValueBias);
      //Sweep over the new root marking it as good (calling NULL function), and then delete anything unmarked.
      //This will include the old root node and the old copy of the child that we promoted to root.
      applyRecursivelyAnyOrderMulithreaded({rootNode}, NULL);
      bool old = true;
      deleteAllOldOrAllNewTableNodesAndSubtreeValueBiasMulithreaded(old);
    }
    else {
      clearSearch();
    }
  }


  rootHistory.makeBoardMoveAssumeLegal(rootBoard,moveLoc,rootPla);
  rootPla = getOpp(rootPla);

  //Explicitly clear avoid move arrays when we play a move - user needs to respecify them if they want them.
  avoidMoveUntilByLocBlack.clear();
  avoidMoveUntilByLocWhite.clear();




  return true;
}


Loc Search::runWholeSearchAndGetMove(Player movePla) {
  return runWholeSearchAndGetMove(movePla,false);
}

Loc Search::runWholeSearchAndGetMove(Player movePla, bool pondering) {
  runWholeSearch(movePla,pondering);
  return getChosenMoveLoc();
}

void Search::runWholeSearch(Player movePla) {
  runWholeSearch(movePla,false);
}

void Search::runWholeSearch(Player movePla, bool pondering) {
  if(movePla != rootPla)
    setPlayerAndClearHistory(movePla);
  std::atomic<bool> shouldStopNow(false);
  runWholeSearch(shouldStopNow,pondering);
}

void Search::runWholeSearch(std::atomic<bool>& shouldStopNow) {
  runWholeSearch(shouldStopNow, false);
}

void Search::runWholeSearch(std::atomic<bool>& shouldStopNow, bool pondering) {
  std::function<void()>* searchBegun = NULL;
  runWholeSearch(shouldStopNow,searchBegun,pondering,TimeControls(),1.0);
}

void Search::runWholeSearch(
  std::atomic<bool>& shouldStopNow,
  std::function<void()>* searchBegun,
  bool pondering,
  const TimeControls& tc,
  double searchFactor
) {

  ClockTimer timer;
  atomic<int64_t> numPlayoutsShared(0);

  if(!std::atomic_is_lock_free(&numPlayoutsShared))
    logger->write("Warning: int64_t atomic numPlayoutsShared is not lock free");
  if(!std::atomic_is_lock_free(&shouldStopNow))
    logger->write("Warning: bool atomic shouldStopNow is not lock free");

  //Do this first, just in case this causes us to clear things and have 0 effective time carried over
  beginSearch(pondering);
  if(searchBegun != NULL)
    (*searchBegun)();
  const int64_t numNonPlayoutVisits = getRootVisits();

  //Compute caps on search
  int64_t maxVisits = pondering ? searchParams.maxVisitsPondering : searchParams.maxVisits;
  int64_t maxPlayouts = pondering ? searchParams.maxPlayoutsPondering : searchParams.maxPlayouts;
  double maxTime = pondering ? searchParams.maxTimePondering : searchParams.maxTime;

  {
    //Possibly reduce computation time, for human friendliness

    if(searchFactor != 1.0) {
      double cap = (double)((int64_t)1L << 62);
      maxVisits = (int64_t)ceil(std::min(cap, maxVisits * searchFactor));
      maxPlayouts = (int64_t)ceil(std::min(cap, maxPlayouts * searchFactor));
      maxTime = maxTime * searchFactor;
    }
  }
  currentSearchPlayoutBudget = std::max<int64_t>(
    0,
    std::min(maxPlayouts,maxVisits - numNonPlayoutVisits)
  );

  //Apply time controls. These two don't particularly need to be synchronized with each other so its fine to have two separate atomics.
  std::atomic<double> tcMaxTime(1e30);
  std::atomic<double> upperBoundVisitsLeftDueToTime(1e30);
  const bool hasMaxTime = maxTime < 1.0e12;
  const bool hasTc = !pondering && !tc.isEffectivelyUnlimitedTime();
  if(!pondering && (hasTc || hasMaxTime)) {
    int64_t rootVisits = numPlayoutsShared.load(std::memory_order_relaxed) + numNonPlayoutVisits;
    double timeUsed = timer.getSeconds();
    double tcLimit = 1e30;
    if(hasTc) {
      tcLimit = recomputeSearchTimeLimit(tc, timeUsed, searchFactor, rootVisits);
      tcMaxTime.store(tcLimit, std::memory_order_release);
    }
    double upperBoundVisits = computeUpperBoundVisitsLeftDueToTime(rootVisits, timeUsed, std::min(tcLimit,maxTime));
    upperBoundVisitsLeftDueToTime.store(upperBoundVisits, std::memory_order_release);
  }

  std::function<void(int)> searchLoop = [
    this,&timer,&numPlayoutsShared,numNonPlayoutVisits,&tcMaxTime,&upperBoundVisitsLeftDueToTime,&tc,
    &hasMaxTime,&hasTc,
    &shouldStopNow,maxVisits,maxPlayouts,maxTime,pondering,searchFactor
  ](int threadIdx) {
    SearchThread* stbuf = new SearchThread(threadIdx,*this);

    int64_t numPlayouts = numPlayoutsShared.load(std::memory_order_relaxed);
    try {
      double lastTimeUsedRecomputingTcLimit = 0.0;
      while(true) {
        double timeUsed = 0.0;
        if(hasTc || hasMaxTime)
          timeUsed = timer.getSeconds();

        double tcMaxTimeLimit = 0.0;
        if(hasTc)
          tcMaxTimeLimit = tcMaxTime.load(std::memory_order_acquire);

        bool shouldStop =
          (numPlayouts >= maxPlayouts) ||
          (numPlayouts + numNonPlayoutVisits >= maxVisits);

        if(hasMaxTime && numPlayouts >= 2 && timeUsed >= maxTime)
          shouldStop = true;
        if(hasTc && numPlayouts >= 2 && timeUsed >= tcMaxTimeLimit)
          shouldStop = true;

        if(shouldStop || shouldStopNow.load(std::memory_order_relaxed)) {
          shouldStopNow.store(true,std::memory_order_relaxed);
          break;
        }

        //Thread 0 alone is responsible for recomputing time limits every once in a while
        //Cap of 10 times per second.
        if(!pondering && (hasTc || hasMaxTime) && threadIdx == 0 && timeUsed >= lastTimeUsedRecomputingTcLimit + 0.1) {
          int64_t rootVisits = numPlayouts + numNonPlayoutVisits;
          double tcLimit = 1e30;
          if(hasTc) {
            tcLimit = recomputeSearchTimeLimit(tc, timeUsed, searchFactor, rootVisits);
            tcMaxTime.store(tcLimit, std::memory_order_release);
          }
          double upperBoundVisits = computeUpperBoundVisitsLeftDueToTime(rootVisits, timeUsed, std::min(tcLimit,maxTime));
          upperBoundVisitsLeftDueToTime.store(upperBoundVisits, std::memory_order_release);
        }

        double upperBoundVisitsLeft = 1e30;
        if(hasTc)
          upperBoundVisitsLeft = upperBoundVisitsLeftDueToTime.load(std::memory_order_acquire);
        upperBoundVisitsLeft = std::min(upperBoundVisitsLeft, (double)maxPlayouts - numPlayouts);
        upperBoundVisitsLeft = std::min(upperBoundVisitsLeft, (double)maxVisits - numPlayouts - numNonPlayoutVisits);

        bool finishedPlayout = runSinglePlayout(*stbuf, upperBoundVisitsLeft);
        if(finishedPlayout) {
          numPlayouts = numPlayoutsShared.fetch_add((int64_t)1, std::memory_order_relaxed);
          numPlayouts += 1;
        }
        else {
          //In the case that we didn't finish a playout, give other threads a chance to run before we try again
          //so that it's more likely we become unstuck.
          std::this_thread::yield();
        }
      }
    }
    catch(...) {
      transferOldNNOutputs(*stbuf);
      delete stbuf;
      throw;
    }

    transferOldNNOutputs(*stbuf);
    delete stbuf;
  };

  double actualSearchStartTime = timer.getSeconds();
  performTaskWithThreads(&searchLoop);

  //Relaxed load is fine since numPlayoutsShared should be synchronized already due to the joins
  lastSearchNumPlayouts = numPlayoutsShared.load(std::memory_order_relaxed);
  effectiveSearchTimeCarriedOver += timer.getSeconds() - actualSearchStartTime;
}

//If we're being asked to search from a position where the game is over, this is fine. Just keep going, the boardhistory
//should reasonably tolerate just continuing. We do NOT want to clear history because we could inadvertently make a move
//that an external ruleset COULD think violated superko.
void Search::beginSearch(bool pondering) {
  if(rootBoard.x_size > nnXLen || rootBoard.y_size > nnYLen)
    throw StringError("Search got from NNEval nnXLen = " + Global::intToString(nnXLen) +
                      " nnYLen = " + Global::intToString(nnYLen) + " but was asked to search board with larger x or y size");

  rootBoard.checkConsistency();

  numSearchesBegun++;

  //Avoid any issues in principle from rolling over
  if(searchNodeAge > 0x3FFFFFFF)
    clearSearch();

  if(!pondering)
    plaThatSearchIsFor = rootPla;
  //If we begin the game with a ponder, then assume that "we" are the opposing side until we see otherwise.
  if(plaThatSearchIsFor == C_EMPTY)
    plaThatSearchIsFor = getOpp(rootPla);

  if(plaThatSearchIsForLastSearch != plaThatSearchIsFor) {
    //In the case we are doing playoutDoublingAdvantage without a specific player (so, doing the root player)
    //and the player that the search is for changes, we need to clear the tree since we need new evals for the new way around
    if(searchParams.playoutDoublingAdvantage != 0 && searchParams.playoutDoublingAdvantagePla == C_EMPTY)
      clearSearch();
    //If we are doing pattern bonus and the player the search is for changes, clear the search. Recomputing the search tree
    //recursively *would* fix all our utilities, but the problem is the playout distribution will still be matching the
    //old probabilities without a lot of new search, so clearing ensures a better distribution.
    if(searchParams.avoidRepeatedPatternUtility != 0 || externalPatternBonusTable != nullptr)
      clearSearch();
  }
  plaThatSearchIsForLastSearch = plaThatSearchIsFor;
  //cout << "BEGINSEARCH " << PlayerIO::playerToString(rootPla) << " " << PlayerIO::playerToString(plaThatSearchIsFor) << endl;

  clearOldNNOutputs();

  //Prepare value bias table if we need it
  if(searchParams.subtreeValueBiasFactor != 0 && subtreeValueBiasTable == NULL)
    subtreeValueBiasTable = new SubtreeValueBiasTable(searchParams.subtreeValueBiasTableNumShards);

  //Refresh pattern bonuses if needed
  if(patternBonusTable != NULL) {
    delete patternBonusTable;
    patternBonusTable = NULL;
  }
  if(searchParams.avoidRepeatedPatternUtility != 0 || externalPatternBonusTable != nullptr) {
    if(externalPatternBonusTable != nullptr)
      patternBonusTable = new PatternBonusTable(*externalPatternBonusTable);
    else
      patternBonusTable = new PatternBonusTable();
    if(searchParams.avoidRepeatedPatternUtility != 0) {
      double bonus = plaThatSearchIsFor == P_WHITE ? -searchParams.avoidRepeatedPatternUtility : searchParams.avoidRepeatedPatternUtility;
      patternBonusTable->addBonusForGameMoves(rootHistory,bonus,plaThatSearchIsFor);
    }
    //Clear any pattern bonus on the root node itself
    if(rootNode != NULL)
      rootNode->patternBonusHash = Hash128();
  }

  if(searchParams.rootSymmetryPruning) {
    const std::vector<int>& avoidMoveUntilByLoc = rootPla == P_BLACK ? avoidMoveUntilByLocBlack : avoidMoveUntilByLocWhite;
    if(rootPruneOnlySymmetries.size() > 0)
      SymmetryHelpers::markDuplicateMoveLocs(rootBoard,rootHistory,&rootPruneOnlySymmetries,avoidMoveUntilByLoc,rootSymDupLoc,rootSymmetries);
    else
      SymmetryHelpers::markDuplicateMoveLocs(rootBoard,rootHistory,NULL,avoidMoveUntilByLoc,rootSymDupLoc,rootSymmetries);
  }
  else {
    //Just in case, don't leave the values undefined.
    std::fill(rootSymDupLoc,rootSymDupLoc+Board::MAX_ARR_SIZE, false);
    rootSymmetries.clear();
    rootSymmetries.push_back(0);
  }

  SearchThread dummyThread(-1, *this);

  if(rootNode == NULL) {
    //Avoid storing the root node in the nodeTable, guarantee that it never is part of a cycle, allocate it directly.
    //Also force that it is non-terminal.
    const bool forceNonTerminal = true;
    rootNode = new SearchNode(rootPla, forceNonTerminal, createMutexIdxForNode(dummyThread));
  }
  else {
    //If the root node has any existing children, then prune things down if there are moves that should not be allowed at the root.
    SearchNode& node = *rootNode;
    int childrenCapacity;
    SearchChildPointer* children = node.getChildren(childrenCapacity);
    bool anyFiltered = false;
    if(childrenCapacity > 0 && children != NULL) {

      //This filtering, by deleting children, doesn't conform to the normal invariants that hold during search.
      //However nothing else should be running at this time and the search hasn't actually started yet, so this is okay.
      //Also we can't be affecting the tree since the root node isn't in the table and can't be transposed to.
      int numGoodChildren = 0;
      vector<SearchNode*> filteredNodes;
      {
        int i = 0;
        for(; i<childrenCapacity; i++) {
          SearchNode* child = children[i].getIfAllocated();
          int64_t edgeVisits = children[i].getEdgeVisits();
          int64_t whiteWinEdgeVisits =
            children[i].getObjectiveEdgeVisits(P_WHITE);
          int64_t blackWinEdgeVisits =
            children[i].getObjectiveEdgeVisits(P_BLACK);
          int64_t whiteVctEdgeVisits =
            children[i].getVctEdgeVisits(P_WHITE);
          int64_t blackVctEdgeVisits =
            children[i].getVctEdgeVisits(P_BLACK);
          Loc moveLoc = children[i].getMoveLoc();
          if(child == NULL)
            break;
          //Remove the child from its current spot
          children[i].store(NULL);
          children[i].setEdgeVisits(0);
          children[i].setObjectiveEdgeVisitsRelaxed(P_WHITE,0);
          children[i].setObjectiveEdgeVisitsRelaxed(P_BLACK,0);
          children[i].setVctEdgeVisitsRelaxed(P_WHITE,0);
          children[i].setVctEdgeVisitsRelaxed(P_BLACK,0);
          children[i].setMoveLoc(Board::NULL_LOC);
          //Maybe add it back. Specifically check for legality just in case weird graph interaction in the
          //tree gives wrong legality - ensure that once we are the root, we are strict on legality.
          if(rootHistory.isLegal(rootBoard,moveLoc,rootPla) && isAllowedRootMove(moveLoc)) {
            children[numGoodChildren].store(child);
            children[numGoodChildren].setEdgeVisits(edgeVisits);
            children[numGoodChildren].setObjectiveEdgeVisitsRelaxed(
              P_WHITE,whiteWinEdgeVisits
            );
            children[numGoodChildren].setObjectiveEdgeVisitsRelaxed(
              P_BLACK,blackWinEdgeVisits
            );
            children[numGoodChildren].setVctEdgeVisitsRelaxed(
              P_WHITE,whiteVctEdgeVisits
            );
            children[numGoodChildren].setVctEdgeVisitsRelaxed(
              P_BLACK,blackVctEdgeVisits
            );
            children[numGoodChildren].setMoveLoc(moveLoc);
            numGoodChildren++;
          }
          else {
            anyFiltered = true;
            filteredNodes.push_back(child);
          }
        }
        for(; i<childrenCapacity; i++) {
          SearchNode* child = children[i].getIfAllocated();
          (void)child;
          assert(child == NULL);
        }
      }

      if(anyFiltered) {
        //Fix up the number of visits of the root node after doing this filtering
        int64_t newNumVisits = 0;
        for(int i = 0; i<childrenCapacity; i++) {
          const SearchNode* child = children[i].getIfAllocated();
          if(child == NULL)
            break;
          int64_t edgeVisits = children[i].getEdgeVisits();
          newNumVisits += edgeVisits;
        }

        //Just for cleanliness after filtering - delete the smaller children arrays.
        //They should never be accessed in the upcoming search because all threads
        //spawned will of course be synchronized with any writes we make here,
        //including the current state of the node, so if we've moved on to a
        //higher-capacity array the lower ones will never be accessed.
        if(children == node.children2) {
          delete[] node.children1;
          node.children1 = NULL;
          delete[] node.children0;
          node.children0 = NULL;
        }
        else if(children == node.children1) {
          delete[] node.children0;
          node.children0 = NULL;
        }
        else {
          assert(children == node.children0);
        }

        //For the node's own visit itself
        newNumVisits += 1;

        //Set the visits in place
        while(node.statsLock.test_and_set(std::memory_order_acquire));
        node.stats.visits.store(newNumVisits,std::memory_order_release);
        node.statsLock.clear(std::memory_order_release);

        //Update all other stats
        recomputeNodeStats(node, dummyThread, 0, true);
      }
    }

    //Recursively update all stats in the tree if we have dynamic score values
    //And also to clear out lastResponseBiasDeltaSum and lastResponseBiasWeight
    if(patternBonusTable != NULL) {
      recursivelyRecomputeStats(node);
      if(anyFiltered) {
        //Recursive stats recomputation resulted in us marking all nodes we have. Anything filtered is old now, delete it.
        bool old = true;
        deleteAllOldOrAllNewTableNodesAndSubtreeValueBiasMulithreaded(old);
      }
    }
    else {
      if(anyFiltered) {
        //Sweep over the entire child marking it as good (calling NULL function), and then delete anything unmarked.
        applyRecursivelyAnyOrderMulithreaded({rootNode}, NULL);
        bool old = true;
        deleteAllOldOrAllNewTableNodesAndSubtreeValueBiasMulithreaded(old);
      }
    }
  }

  //Clear unused stuff in value bias table since we may have pruned rootNode stuff
  if(searchParams.subtreeValueBiasFactor != 0 && subtreeValueBiasTable != NULL)
    subtreeValueBiasTable->clearUnusedSynchronous();

  //Mark all nodes old for the purposes of updating old nnoutputs
  searchNodeAge++;

  rootNormalVisitsAtSearchStart =
    rootNode->stats.visits.load(std::memory_order_acquire);
  rootWhiteWinEdgeVisitsAtSearchStart =
    getRootObjectiveEdgeVisits(P_WHITE);
  rootBlackWinEdgeVisitsAtSearchStart =
    getRootObjectiveEdgeVisits(P_BLACK);
  rootWhiteVctVisitsAtSearchStart =
    rootNode->whiteVctStats.visits.load(std::memory_order_acquire);
  rootBlackVctVisitsAtSearchStart =
    rootNode->blackVctStats.visits.load(std::memory_order_acquire);
  rootVctNormalVerificationPlayouts.store(0,std::memory_order_release);
}

uint32_t Search::createMutexIdxForNode(SearchThread& thread) const {
  return thread.rand.nextUInt() & (mutexPool->getNumMutexes()-1);
}

//Based on sha256 of "search.cpp FORCE_NON_TERMINAL_HASH"
static const Hash128 FORCE_NON_TERMINAL_HASH = Hash128(0xd4c31800cb8809e2ULL,0xf75f9d2083f2ffcaULL);

//Must be called AFTER making the bestChildMoveLoc in the thread board and hist.
SearchNode* Search::allocateOrFindNode(SearchThread& thread, Player nextPla, Loc bestChildMoveLoc, bool forceNonTerminal, Hash128 graphHash) {
  //Hash to use as a unique id for this node in the table, for transposition detection.
  //If this collides, we will be sad, but it should be astronomically rare since our hash is 128 bits.
  Hash128 childHash;
  if(searchParams.useGraphSearch) {
    childHash = graphHash;
    if(forceNonTerminal)
      childHash ^= FORCE_NON_TERMINAL_HASH;
  }
  else {
    childHash = thread.board.pos_hash ^ Hash128(thread.rand.nextUInt64(),thread.rand.nextUInt64());
  }

  uint32_t nodeTableIdx = nodeTable->getIndex(childHash.hash0);
  std::mutex& mutex = nodeTable->mutexPool->getMutex(nodeTableIdx);
  std::lock_guard<std::mutex> lock(mutex);

  SearchNode* child = NULL;
  std::map<Hash128,SearchNode*>& nodeMap = nodeTable->entries[nodeTableIdx];

  while(true) {
    auto insertLoc = nodeMap.lower_bound(childHash);

    if(insertLoc != nodeMap.end() && insertLoc->first == childHash) {
      //Attempt to transpose to invalid node - rerandomize hash and just store this node somewhere arbitrary.
      if(insertLoc->second->nextPla != nextPla) {
        childHash = thread.board.pos_hash ^ Hash128(thread.rand.nextUInt64(),thread.rand.nextUInt64());
        continue;
      }
      child = insertLoc->second;
    }
    else {
      child = new SearchNode(nextPla, forceNonTerminal, createMutexIdxForNode(thread));

      //Also perform subtree value bias and pattern bonus handling under the mutex. These parameters are no atomic, so
      //if the node is accessed concurrently by other nodes through the table, we need to make sure these parameters are fully
      //fully-formed before we make the node accessible to anyone.

      if(searchParams.subtreeValueBiasFactor != 0 && subtreeValueBiasTable != NULL) {
        //TODO can we make subtree value bias not depend on prev move loc?
        if(thread.history.moveHistory.size() >= 2) {
          Loc prevMoveLoc = thread.history.moveHistory[thread.history.moveHistory.size()-2].loc;
          if(prevMoveLoc != Board::NULL_LOC) {
            child->subtreeValueBiasTableEntry = subtreeValueBiasTable->get(getOpp(thread.pla), prevMoveLoc, bestChildMoveLoc, thread.history.getRecentBoard(1));
          }
        }
      }

      if(patternBonusTable != NULL)
        child->patternBonusHash = patternBonusTable->getHash(getOpp(thread.pla), bestChildMoveLoc, thread.history.getRecentBoard(1));

      //Insert into map! Use insertLoc as hint.
      nodeMap.insert(insertLoc, std::make_pair(childHash,child));
    }
    break;
  }
  return child;
}

void Search::clearOldNNOutputs() {
  for(size_t i = 0; i<oldNNOutputsToCleanUp.size(); i++)
    delete oldNNOutputsToCleanUp[i];
  oldNNOutputsToCleanUp.resize(0);
}
void Search::transferOldNNOutputs(SearchThread& thread) {
  std::lock_guard<std::mutex> lock(oldNNOutputsToCleanUpMutex);
  for(size_t i = 0; i<thread.oldNNOutputsToCleanUp.size(); i++)
    oldNNOutputsToCleanUp.push_back(thread.oldNNOutputsToCleanUp[i]);
  thread.oldNNOutputsToCleanUp.resize(0);
}

void Search::removeSubtreeValueBias(SearchNode* node) {
  if(node->subtreeValueBiasTableEntry != nullptr) {
    double deltaUtilitySumToSubtract = node->lastSubtreeValueBiasDeltaSum * searchParams.subtreeValueBiasFreeProp;
    double weightSumToSubtract = node->lastSubtreeValueBiasWeight * searchParams.subtreeValueBiasFreeProp;

    SubtreeValueBiasEntry& entry = *(node->subtreeValueBiasTableEntry);
    while(entry.entryLock.test_and_set(std::memory_order_acquire));
    entry.deltaUtilitySum -= deltaUtilitySumToSubtract;
    entry.weightSum -= weightSumToSubtract;
    entry.entryLock.clear(std::memory_order_release);
    node->subtreeValueBiasTableEntry = nullptr;
  }
}

//Delete ALL nodes where nodeAge < searchNodeAge if old is true, else all nodes where nodeAge >= searchNodeAge
//Also clears subtreevaluebias for deleted nodes.
void Search::deleteAllOldOrAllNewTableNodesAndSubtreeValueBiasMulithreaded(bool old) {
  int numAdditionalThreads = numAdditionalThreadsToUseForTasks();
  assert(numAdditionalThreads >= 0);
  std::function<void(int)> g = [&](int threadIdx) {
    size_t idx0 = (size_t)((uint64_t)(threadIdx) * nodeTable->entries.size() / (numAdditionalThreads+1));
    size_t idx1 = (size_t)((uint64_t)(threadIdx+1) * nodeTable->entries.size() / (numAdditionalThreads+1));
    for(size_t i = idx0; i<idx1; i++) {
      std::map<Hash128,SearchNode*>& nodeMap = nodeTable->entries[i];
      for(auto it = nodeMap.cbegin(); it != nodeMap.cend();) {
        SearchNode* node = it->second;
        if(old == (node->nodeAge.load(std::memory_order_acquire) < searchNodeAge)) {
          removeSubtreeValueBias(node);
          delete node;
          it = nodeMap.erase(it);
        }
        else
          ++it;
      }
    }
  };
  performTaskWithThreads(&g);
}

//Delete ALL nodes. More efficient than deleteAllOldOrAllNewTableNodesAndSubtreeValueBiasMulithreaded if deleting everything.
//Doesn't clear subtree value bias.
void Search::deleteAllTableNodesMulithreaded() {
  int numAdditionalThreads = numAdditionalThreadsToUseForTasks();
  assert(numAdditionalThreads >= 0);
  std::function<void(int)> g = [&](int threadIdx) {
    size_t idx0 = (size_t)((uint64_t)(threadIdx) * nodeTable->entries.size() / (numAdditionalThreads+1));
    size_t idx1 = (size_t)((uint64_t)(threadIdx+1) * nodeTable->entries.size() / (numAdditionalThreads+1));
    for(size_t i = idx0; i<idx1; i++) {
      std::map<Hash128,SearchNode*>& nodeMap = nodeTable->entries[i];
      for(auto it = nodeMap.cbegin(); it != nodeMap.cend(); ++it) {
        delete it->second;
      }
      nodeMap.clear();
    }
  };
  performTaskWithThreads(&g);
}

//This function should NOT ever be called concurrently with any other threads modifying the search tree.
//However, it does thread-safely modify things itself, so can safely in theory run concurrently with things
//like ownership computation or analysis that simply read the tree.
void Search::recursivelyRecomputeStats(SearchNode& n) {
  int numAdditionalThreads = numAdditionalThreadsToUseForTasks();
  std::vector<SearchThread*> dummyThreads(numAdditionalThreads+1, NULL);
  for(int threadIdx = 0; threadIdx<numAdditionalThreads+1; threadIdx++)
    dummyThreads[threadIdx] = new SearchThread(threadIdx, *this);

  std::function<void(SearchNode*,int)> f = [&](SearchNode* node, int threadIdx) {
    assert(threadIdx >= 0 && threadIdx < dummyThreads.size());
    SearchThread& thread = *(dummyThreads[threadIdx]);

    bool foundAnyChildren = false;
    int childrenCapacity;
    SearchChildPointer* children = node->getChildren(childrenCapacity);
    int i = 0;
    for(; i<childrenCapacity; i++) {
      SearchNode* child = children[i].getIfAllocated();
      if(child == NULL)
        break;
      foundAnyChildren = true;
    }
    for(; i<childrenCapacity; i++) {
      SearchNode* child = children[i].getIfAllocated();
      (void)child;
      assert(child == NULL);
    }

    //If this node has children, it MUST also have an nnOutput.
    if(foundAnyChildren) {
      NNOutput* nnOutput = node->getNNOutput();
      (void)nnOutput; //avoid warning when we have no asserts
      assert(nnOutput != NULL);
    }

    //Also, something is wrong if we have virtual losses at this point
    int32_t numVirtualLosses = node->virtualLosses.load(std::memory_order_acquire);
    (void)numVirtualLosses;
    assert(numVirtualLosses == 0);
    assert(node->whiteVctVirtualLosses.load(std::memory_order_acquire) == 0);
    assert(node->blackVctVirtualLosses.load(std::memory_order_acquire) == 0);

    bool isRoot = (node == rootNode);

    //If the node has no children, then just update its utility directly
    //Again, this would be a little wrong if this function were running concurrently with anything else in the
    //case that new children were added in the meantime. Although maybe it would be okay.
    if(!foundAnyChildren) {
      int64_t numVisits = node->stats.visits.load(std::memory_order_acquire);
      double weightSum = node->stats.weightSum.load(std::memory_order_acquire);
      double winLossValueAvg = node->stats.winLossValueAvg.load(std::memory_order_acquire);
      double noResultValueAvg = node->stats.noResultValueAvg.load(std::memory_order_acquire);

      //It's possible that this node has 0 weight in the case where it's the root node
      //and has 0 visits because we began a search and then stopped it before any playouts happened.
      //In that case, there's not much to recompute.
      if(weightSum <= 0.0) {
        assert(numVisits == 0);
        assert(isRoot);
      }
      else {
        double resultUtility = getResultUtility(winLossValueAvg, noResultValueAvg);
        double whiteWinProbAvg = node->stats.whiteWinProbAvg.load(std::memory_order_acquire);
        double blackWinProbAvg = node->stats.blackWinProbAvg.load(std::memory_order_acquire);
        double newWhiteWinUtilityAvg = getWhiteWinUtility(resultUtility, whiteWinProbAvg);
        double newBlackWinUtilityInvAvg = getBlackWinUtilityInv(resultUtility, blackWinProbAvg);
        double patternBonus = getPatternBonus(node->patternBonusHash,getOpp(node->nextPla));
        newWhiteWinUtilityAvg += patternBonus;
        newBlackWinUtilityInvAvg += patternBonus;
        double newUtilityAvg = 0.5 * (newWhiteWinUtilityAvg + newBlackWinUtilityInvAvg);
        double newUtilitySqAvg = newUtilityAvg * newUtilityAvg;

        while(node->statsLock.test_and_set(std::memory_order_acquire));
        node->stats.utilityAvg.store(newUtilityAvg,std::memory_order_release);
        node->stats.utilitySqAvg.store(newUtilitySqAvg,std::memory_order_release);
        node->stats.whiteWinUtilityAvg.store(newWhiteWinUtilityAvg,std::memory_order_release);
        node->stats.whiteWinUtilitySqAvg.store(newWhiteWinUtilityAvg * newWhiteWinUtilityAvg,std::memory_order_release);
        node->stats.blackWinUtilityInvAvg.store(newBlackWinUtilityInvAvg,std::memory_order_release);
        node->stats.blackWinUtilityInvSqAvg.store(newBlackWinUtilityInvAvg * newBlackWinUtilityInvAvg,std::memory_order_release);
        node->statsLock.clear(std::memory_order_release);
      }
    }
    else {
      //Otherwise recompute it using the usual method
      recomputeNodeStats(*node, thread, 0, isRoot);
    }
  };

  vector<SearchNode*> nodes;
  nodes.push_back(&n);
  applyRecursivelyPostOrderMulithreaded(nodes,&f);

  for(int threadIdx = 0; threadIdx<numAdditionalThreads+1; threadIdx++)
    delete dummyThreads[threadIdx];
}

double Search::getCurrentSearchProgress(const SearchThread& thread) const {
  if(currentSearchPlayoutBudget <= 0)
    return 0.0;
  return std::clamp(
    1.0 - thread.upperBoundVisitsLeft / (double)currentSearchPlayoutBudget,
    0.0,1.0
  );
}

double Search::getVctProbePhaseScale(const SearchThread& thread) const {
  double start = searchParams.multiHeadVctProbeStartFraction;
  double end = searchParams.multiHeadVctProbeEndFraction;
  if(start <= 0.0 && end >= 1.0)
    return 1.0;
  if(end <= start)
    return 0.0;

  double progress = getCurrentSearchProgress(thread);
  if(progress < start || progress >= end)
    return 0.0;

  double ramp = std::min(
    searchParams.multiHeadVctProbeRampFraction,
    0.5 * (end - start)
  );
  if(ramp <= 0.0)
    return 1.0;

  auto smoothStep = [](double x) {
    x = std::clamp(x,0.0,1.0);
    return x * x * (3.0 - 2.0 * x);
  };
  double startScale = smoothStep((progress - start) / ramp);
  double endScale = smoothStep((end - progress) / ramp);
  return std::min(startScale,endScale);
}

double Search::getVctValidationPhaseScale(const SearchThread& thread) const {
  double end = searchParams.multiHeadVctProbeEndFraction;
  //The default window preserves the pre-phased validation behavior.
  if(end >= 1.0)
    return 1.0;

  double progress = getCurrentSearchProgress(thread);
  if(progress < end)
    return 0.0;
  double ramp = searchParams.multiHeadVctProbeRampFraction;
  if(ramp <= 0.0)
    return 1.0;

  double x = std::clamp((progress - end) / ramp,0.0,1.0);
  return x * x * (3.0 - 2.0 * x);
}

Player Search::chooseVctPlayoutAttacker(const SearchThread& thread) const {
  if(searchParams.multiHeadVctMaxAttackProp <= 0.0 && searchParams.multiHeadVctMaxDefenseProp <= 0.0)
    return C_EMPTY;
  double probePhaseScale = getVctProbePhaseScale(thread);
  if(probePhaseScale <= 0.0)
    return C_EMPTY;

  const NNOutput* nnOutput = rootNode->getNNOutput();
  if(nnOutput == NULL)
    return C_EMPTY;
  if(!nnOutput->hasPolicyByHead())
    throw StringError("multi-head VCT search requires a v112 model with six policy heads");

  int policySize = NNPos::getPolicySize(nnOutput->nnXLen,nnOutput->nnYLen);
  auto getPolicyGate = [&](Player attacker) {
    int vctHead = attacker == rootPla ? 4 : 5;
    int outcomeHead = attacker == rootPla ? 3 : 2;
    double consensusMix = searchParams.multiHeadVctPolicyConsensusMix;
    double consensusMass = 0.0;
    if(consensusMix > 0.0) {
      for(int movePos = 0; movePos<policySize; movePos++) {
        if(nnOutput->getPolicyProbMaybeNoised(movePos) < 0.0)
          continue;
        double vctPolicy = nnOutput->getPolicyProbByHead(vctHead,movePos);
        double outcomePolicy = nnOutput->getPolicyProbByHead(outcomeHead,movePos);
        if(vctPolicy > 0.0 && outcomePolicy > 0.0)
          consensusMass += sqrt(vctPolicy * outcomePolicy);
      }
    }
    double peakPolicy = 0.0;
    double normalPolicyAtPeak = 0.0;
    for(int movePos = 0; movePos<policySize; movePos++) {
      double normalPolicy = nnOutput->getPolicyProbMaybeNoised(movePos);
      if(normalPolicy < 0.0)
        continue;
      double vctPolicy = nnOutput->getPolicyProbByHead(vctHead,movePos);
      double auxiliaryPolicy = vctPolicy;
      if(consensusMix > 0.0 && consensusMass > 0.0) {
        double outcomePolicy = nnOutput->getPolicyProbByHead(outcomeHead,movePos);
        double consensusPolicy =
          vctPolicy > 0.0 && outcomePolicy > 0.0 ?
          sqrt(vctPolicy * outcomePolicy) / consensusMass :
          0.0;
        auxiliaryPolicy =
          (1.0 - consensusMix) * vctPolicy +
          consensusMix * consensusPolicy;
      }
      if(auxiliaryPolicy > peakPolicy) {
        peakPolicy = auxiliaryPolicy;
        normalPolicyAtPeak = normalPolicy;
      }
    }
    if(peakPolicy <= 0.0)
      return 0.0;

    //A diffuse auxiliary policy gets a small but nonzero budget. Concentrated
    //policies get more, particularly when their top move is novel versus head0.
    double concentration =
      peakPolicy /
      (peakPolicy + searchParams.multiHeadAuxPolicyConcentrationScale);
    double novelty = std::clamp(
      1.0 - normalPolicyAtPeak / peakPolicy,
      0.0,1.0
    );
    return concentration * (0.1 + 0.9 * novelty);
  };
  auto getPriorSuccessProb = [&](Player attacker) {
    int head = attacker == rootPla ? 4 : 5;
    double successProb = attacker == P_WHITE ?
      nnOutput->whiteWinProbByHead[head] :
      nnOutput->whiteLossProbByHead[head];
    return std::clamp(successProb,1e-4,1.0);
  };
  auto getMainWinNeed = [&](Player attacker) {
    double normalWinProb = attacker == P_WHITE ?
      nnOutput->whiteWinProbByHead[0] :
      nnOutput->whiteLossProbByHead[0];
    double need = std::clamp(1.0 - normalWinProb,0.0,1.0);
    return pow(need,searchParams.multiHeadVctMainWinSuppression);
  };
  auto getSmoothBudget = [&](double probability) {
    double poweredProbability = pow(
      std::clamp(probability,0.0,1.0),
      searchParams.multiHeadVctProbPower
    );
    double poweredScale = pow(
      searchParams.multiHeadVctProbScale,
      searchParams.multiHeadVctProbPower
    );
    return poweredProbability / (poweredProbability + poweredScale);
  };
  auto getUrgency = [&](Player attacker) {
    double policyGate = getPolicyGate(attacker);
    double mainWinNeed = getMainWinNeed(attacker);
    VctStats stats(rootNode->getVctStats(attacker));
    double priorSuccessProb = getPriorSuccessProb(attacker);
    //Every expanded node starts with one direct NN sample in each isolated
    //stats plane. Until there is a real tactical playout, budget from the
    //auxiliary prior rather than treating that seed as validation evidence.
    if(stats.visits <= 1 || stats.weightSum <= 0.0)
      return
        policyGate *
        mainWinNeed *
        getSmoothBudget(priorSuccessProb);

    double successProb = attacker == P_WHITE ?
      0.5 * (stats.utilityAvg + 1.0) :
      0.5 * (1.0 - stats.utilityAvg);
    successProb = std::clamp(successProb,0.0,1.0);

    double variance = std::max(0.0,stats.utilitySqAvg - stats.utilityAvg * stats.utilityAvg);
    double effectiveSamples = stats.weightSqSum > 0.0 ?
      stats.weightSum * stats.weightSum / stats.weightSqSum :
      (double)stats.visits;
    effectiveSamples = std::max(1.0,effectiveSamples);
    double empiricalStdErr = 0.5 * sqrt(variance / effectiveSamples);
    if(searchParams.multiHeadVctUseNormalRules) {
      double sidecarSelfUtility =
        attacker == P_WHITE ? stats.utilityAvg : -stats.utilityAvg;
      double mainUtility =
        rootNode->stats.utilityAvg.load(std::memory_order_acquire);
      double mainSelfUtility =
        attacker == P_WHITE ? mainUtility : -mainUtility;
      double relativeGainProb =
        0.5 * (sidecarSelfUtility - mainSelfUtility);
      double priorVisits = searchParams.multiHeadVctPriorVisits;
      double explorationStdErr = 0.5 / sqrt(effectiveSamples + priorVisits);
      double uncertainty = std::max(empiricalStdErr,explorationStdErr);
      double priorCarry = priorVisits > 0.0 ?
        priorSuccessProb * priorVisits / (effectiveSamples + priorVisits) :
        0.0;
      double optimisticOpportunity = std::clamp(
        std::max(
          priorCarry,
          relativeGainProb + searchParams.multiHeadVctUcbCoeff * uncertainty
        ),
        0.0,1.0
      );
      return
        policyGate *
        mainWinNeed *
        getSmoothBudget(optimisticOpportunity);
    }

    double explorationStdErr = sqrt(
      std::max(1e-4,successProb * (1.0 - successProb)) /
      (effectiveSamples + 4.0)
    );
    double uncertainty = std::max(empiricalStdErr,explorationStdErr);
    double optimisticProb = std::clamp(
      successProb + searchParams.multiHeadVctUcbCoeff * uncertainty,
      0.0,1.0
    );
    return
      policyGate *
      mainWinNeed *
      getSmoothBudget(optimisticProb);
  };

  Player attackAttacker = rootPla;
  Player defenseAttacker = getOpp(rootPla);
  double attackUrgency = getUrgency(attackAttacker);
  double defenseUrgency = getUrgency(defenseAttacker);
  double attackProp =
    probePhaseScale *
    searchParams.multiHeadVctMaxAttackProp *
    attackUrgency;
  double defenseProp =
    probePhaseScale *
    searchParams.multiHeadVctMaxDefenseProp *
    defenseUrgency;
  double objectiveBudgetStrength =
    searchParams.multiHeadObjectiveSearchStrength *
    searchParams.multiHeadVctObjectiveBudgetMix;
  if(objectiveBudgetStrength > 0.0) {
    double normalWinWorth;
    double normalNonLossWorth;
    getNormalObjectiveWorths(*rootNode,normalWinWorth,normalNonLossWorth);
    double normalWorth = normalWinWorth + normalNonLossWorth;
    double attackWorth =
      searchParams.multiHeadObjectiveVctWeight * attackUrgency;
    double defenseWorth =
      searchParams.multiHeadObjectiveVctWeight * defenseUrgency;
    double totalWorth = normalWorth + attackWorth + defenseWorth;
    double objectiveAttackProp = totalWorth > 0.0 ?
      std::min(searchParams.multiHeadVctMaxAttackProp,attackWorth / totalWorth) :
      0.0;
    double objectiveDefenseProp = totalWorth > 0.0 ?
      std::min(searchParams.multiHeadVctMaxDefenseProp,defenseWorth / totalWorth) :
      0.0;
    double strength = objectiveBudgetStrength;
    attackProp =
      (1.0 - strength) * attackProp +
      strength * probePhaseScale * objectiveAttackProp;
    defenseProp =
      (1.0 - strength) * defenseProp +
      strength * probePhaseScale * objectiveDefenseProp;
  }
  double tacticalProp = attackProp + defenseProp;
  constexpr double MAX_TACTICAL_PROP = 0.85;
  if(tacticalProp > MAX_TACTICAL_PROP) {
    double scale = MAX_TACTICAL_PROP / tacticalProp;
    attackProp *= scale;
    defenseProp *= scale;
    tacticalProp = MAX_TACTICAL_PROP;
  }
  double normalProp = 1.0 - tacticalProp;

  auto getCurrentSearchCount = [](int64_t currentVisits, int64_t initialVisits) {
    int64_t baseline = initialVisits > 0 ? initialVisits : 1;
    return std::max<int64_t>(0,currentVisits - baseline);
  };
  int64_t normalCount = getCurrentSearchCount(
    rootNode->stats.visits.load(std::memory_order_acquire),
    rootNormalVisitsAtSearchStart
  );
  int64_t whiteVctCount = getCurrentSearchCount(
    rootNode->whiteVctStats.visits.load(std::memory_order_acquire),
    rootWhiteVctVisitsAtSearchStart
  );
  int64_t blackVctCount = getCurrentSearchCount(
    rootNode->blackVctStats.visits.load(std::memory_order_acquire),
    rootBlackVctVisitsAtSearchStart
  );
  int64_t attackCount = attackAttacker == P_WHITE ? whiteVctCount : blackVctCount;
  int64_t defenseCount = defenseAttacker == P_WHITE ? whiteVctCount : blackVctCount;
  double nextTotal = (double)(normalCount + whiteVctCount + blackVctCount + 1);

  double bestDeficit = normalProp * nextTotal - normalCount;
  Player selectedAttacker = C_EMPTY;
  double attackDeficit = attackProp * nextTotal - attackCount;
  if(attackDeficit > bestDeficit) {
    bestDeficit = attackDeficit;
    selectedAttacker = attackAttacker;
  }
  double defenseDeficit = defenseProp * nextTotal - defenseCount;
  if(defenseDeficit > bestDeficit)
    selectedAttacker = defenseAttacker;
  return selectedAttacker;
}

Loc Search::chooseForcedReplySidecarMove(const SearchThread& thread) const {
  double targetVisits = searchParams.multiHeadDrawForcedReplySidecarVisits;
  double maxProp = searchParams.multiHeadDrawForcedReplySidecarMaxProp;
  if(
    targetVisits <= 0.0 ||
    maxProp <= 0.0 ||
    rootNode == NULL
  )
    return Board::NULL_LOC;

  double phaseScale = getVctProbePhaseScale(thread);
  if(phaseScale <= 0.0)
    return Board::NULL_LOC;
  double targetProp = maxProp * phaseScale;

  auto getCurrentSearchCount = [](int64_t currentVisits, int64_t initialVisits) {
    int64_t baseline = initialVisits > 0 ? initialVisits : 1;
    return std::max<int64_t>(0,currentVisits - baseline);
  };
  int64_t normalCount = getCurrentSearchCount(
    rootNode->stats.visits.load(std::memory_order_acquire),
    rootNormalVisitsAtSearchStart
  );
  int64_t whiteVctCount = getCurrentSearchCount(
    rootNode->whiteVctStats.visits.load(std::memory_order_acquire),
    rootWhiteVctVisitsAtSearchStart
  );
  int64_t blackVctCount = getCurrentSearchCount(
    rootNode->blackVctStats.visits.load(std::memory_order_acquire),
    rootBlackVctVisitsAtSearchStart
  );
  int64_t sidecarCount =
    rootPla == P_WHITE ? whiteVctCount : blackVctCount;
  double nextTotal =
    (double)(normalCount + whiteVctCount + blackVctCount + 1);
  double sidecarDeficit = targetProp * nextTotal - sidecarCount;
  double normalDeficit = (1.0 - targetProp) * nextTotal - normalCount;
  if(sidecarDeficit <= normalDeficit)
    return Board::NULL_LOC;

  int childrenCapacity;
  const SearchChildPointer* children = rootNode->getChildren(childrenCapacity);
  if(childrenCapacity <= 0)
    return Board::NULL_LOC;

  double concentrationScale =
    searchParams.multiHeadDrawForcedReplyPolicyThreshold;
  double concentrationNorm =
    1.0 + concentrationScale * concentrationScale;
  constexpr double OPPORTUNITY_SCALE = 0.35;
  constexpr double OPPORTUNITY_NORM =
    1.0 + OPPORTUNITY_SCALE * OPPORTUNITY_SCALE;

  Loc bestMoveLoc = Board::NULL_LOC;
  double bestDeficit = 0.0;
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchNode* child = children[i].getIfAllocated();
    if(child == NULL)
      break;
    const NNOutput* childNNOutput = child->getNNOutput();
    if(childNNOutput == NULL)
      continue;
    if(!childNNOutput->hasPolicyByHead())
      throw StringError("forced-reply sidecar search requires a v112 model");

    double peakReplyPolicy = 0.0;
    for(int movePos = 0; movePos<policySize; movePos++) {
      peakReplyPolicy = std::max(
        peakReplyPolicy,
        (double)childNNOutput->getPolicyProbMaybeNoised(movePos)
      );
    }
    double peakSq = peakReplyPolicy * peakReplyPolicy;
    double scaleSq = concentrationScale * concentrationScale;
    double concentration = peakSq > 0.0 ?
      concentrationNorm * peakSq / (peakSq + scaleSq) :
      0.0;
    concentration = std::clamp(concentration,0.0,1.0);

    Player childPla = child->nextPla;
    double childMustWinProb = childPla == P_WHITE ?
      childNNOutput->whiteWinProbByHead[3] :
      childNNOutput->whiteLossProbByHead[3];
    double childLossProbWhenDrawCountsAsWin = childPla == P_WHITE ?
      childNNOutput->whiteLossProbByHead[2] :
      childNNOutput->whiteWinProbByHead[2];
    double childNonLossProb =
      1.0 - childLossProbWhenDrawCountsAsWin;
    double childOutcomeInterval = std::clamp(
      childNonLossProb - childMustWinProb,
      0.0,1.0
    );
    double childDrawSignal = sqrt(
      std::clamp(
        (double)childNNOutput->whiteNoResultProb,
        0.0,1.0
      ) *
      childOutcomeInterval
    );

    int64_t edgeVisits = children[i].getVctEdgeVisits(rootPla);
    double opportunity = 1.0;
    if(edgeVisits > 0) {
      VctStats stats(child->getVctStats(rootPla));
      if(stats.visits > 0 && stats.weightSum > 0.0) {
        double selfUtility =
          rootPla == P_WHITE ? stats.utilityAvg : -stats.utilityAvg;
        double successProb =
          std::clamp(0.5 * (selfUtility + 1.0),0.0,1.0);
        double variance = std::max(
          0.0,
          stats.utilitySqAvg - stats.utilityAvg * stats.utilityAvg
        );
        double effectiveSamples = stats.weightSqSum > 0.0 ?
          stats.weightSum * stats.weightSum / stats.weightSqSum :
          (double)stats.visits;
        effectiveSamples = std::max(1.0,effectiveSamples);
        double empiricalStdErr =
          0.5 * sqrt(variance / effectiveSamples);
        double explorationStdErr = 0.25 / sqrt(effectiveSamples);
        double uncertainty =
          std::max(empiricalStdErr,explorationStdErr);
        double optimisticProb = std::clamp(
          successProb +
            searchParams.multiHeadVctUcbCoeff * uncertainty,
          0.0,1.0
        );
        double optimisticSq = optimisticProb * optimisticProb;
        opportunity =
          OPPORTUNITY_NORM * optimisticSq /
          (optimisticSq + OPPORTUNITY_SCALE * OPPORTUNITY_SCALE);
        opportunity = std::clamp(opportunity,0.0,1.0);
      }
    }

    double desiredVisits =
      targetVisits *
      concentration *
      childDrawSignal *
      opportunity;
    double deficit = desiredVisits - edgeVisits;
    if(deficit > bestDeficit) {
      bestDeficit = deficit;
      bestMoveLoc = children[i].getMoveLocRelaxed();
    }
  }
  return bestMoveLoc;
}

Loc Search::chooseVctNormalVerificationMove(const SearchThread& thread) {
  if(
    searchParams.multiHeadVctNormalVerificationProp <= 0.0 ||
    (
      searchParams.multiHeadVctMaxAttackProp <= 0.0 &&
      searchParams.multiHeadDrawForcedReplySidecarVisits <= 0.0
    ) ||
    rootNode == NULL
  )
    return Board::NULL_LOC;

  int childrenCapacity;
  const SearchChildPointer* children = rootNode->getChildren(childrenCapacity);
  if(childrenCapacity <= 0)
    return Board::NULL_LOC;

  double bestNormalSelfUtility = -1.0;
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchNode* child = children[i].getIfAllocated();
    if(child == NULL)
      break;
    int64_t edgeVisits = children[i].getEdgeVisits();
    if(edgeVisits <= 0)
      continue;
    double utility = child->stats.utilityAvg.load(std::memory_order_acquire);
    double selfUtility = rootPla == P_WHITE ? utility : -utility;
    bestNormalSelfUtility = std::max(bestNormalSelfUtility,selfUtility);
  }

  Loc bestMoveLoc = Board::NULL_LOC;
  double bestSignal = 0.0;
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchNode* child = children[i].getIfAllocated();
    if(child == NULL)
      break;
    int64_t vctEdgeVisits = children[i].getVctEdgeVisits(rootPla);
    if(vctEdgeVisits <= 0)
      continue;
    VctStats stats(child->getVctStats(rootPla));
    if(stats.visits <= 0 || stats.weightSum <= 0.0)
      continue;

    double successProb = rootPla == P_WHITE ?
      0.5 * (stats.utilityAvg + 1.0) :
      0.5 * (1.0 - stats.utilityAvg);
    successProb = std::clamp(successProb,0.0,1.0);
    double variance = std::max(
      0.0,
      stats.utilitySqAvg - stats.utilityAvg * stats.utilityAvg
    );
    double effectiveSamples = stats.weightSqSum > 0.0 ?
      stats.weightSum * stats.weightSum / stats.weightSqSum :
      (double)stats.visits;
    effectiveSamples = std::max(1.0,effectiveSamples);
    double empiricalStdErr = 0.5 * sqrt(variance / effectiveSamples);
    double explorationStdErr = sqrt(
      std::max(1e-4,successProb * (1.0 - successProb)) /
      (effectiveSamples + 4.0)
    );
    double uncertainty = std::max(empiricalStdErr,explorationStdErr);
    double optimisticProb = std::clamp(
      successProb + searchParams.multiHeadVctUcbCoeff * uncertainty,
      0.0,1.0
    );
    double poweredProbability = pow(
      optimisticProb,
      searchParams.multiHeadVctProbPower
    );
    double poweredScale = pow(
      searchParams.multiHeadVctProbScale,
      searchParams.multiHeadVctProbPower
    );
    double probabilityGate =
      poweredProbability / (poweredProbability + poweredScale);
    double visitConfidence = sqrt(
      (double)vctEdgeVisits /
      (vctEdgeVisits + searchParams.multiHeadVctMoveSelectionVisitScale)
    );

    int64_t normalEdgeVisits = children[i].getEdgeVisits();
    double normalDisproofGate = 1.0;
    if(normalEdgeVisits > 0 && bestNormalSelfUtility > -1.0) {
      double normalUtility =
        child->stats.utilityAvg.load(std::memory_order_acquire);
      double normalSelfUtility =
        rootPla == P_WHITE ? normalUtility : -normalUtility;
      double utilityGap = std::max(
        0.0,
        bestNormalSelfUtility - normalSelfUtility - 0.10
      );
      double normalConfidence =
        (double)normalEdgeVisits / (normalEdgeVisits + 8.0);
      normalDisproofGate = exp(-4.0 * utilityGap * normalConfidence);
    }

    double signal =
      probabilityGate * visitConfidence * normalDisproofGate;
    if(signal > bestSignal) {
      bestSignal = signal;
      bestMoveLoc = children[i].getMoveLocRelaxed();
    }
  }
  if(bestMoveLoc == Board::NULL_LOC || bestSignal <= 0.0)
    return Board::NULL_LOC;

  auto getCurrentSearchCount = [](int64_t currentVisits, int64_t initialVisits) {
    int64_t baseline = initialVisits > 0 ? initialVisits : 1;
    return std::max<int64_t>(0,currentVisits - baseline);
  };
  int64_t normalCount = getCurrentSearchCount(
    rootNode->stats.visits.load(std::memory_order_acquire),
    rootNormalVisitsAtSearchStart
  );
  int64_t whiteVctCount = getCurrentSearchCount(
    rootNode->whiteVctStats.visits.load(std::memory_order_acquire),
    rootWhiteVctVisitsAtSearchStart
  );
  int64_t blackVctCount = getCurrentSearchCount(
    rootNode->blackVctStats.visits.load(std::memory_order_acquire),
    rootBlackVctVisitsAtSearchStart
  );
  int64_t nextTotal = normalCount + whiteVctCount + blackVctCount + 1;
  double targetProp =
    searchParams.multiHeadVctNormalVerificationProp * bestSignal;
  int64_t targetCount = (int64_t)floor(targetProp * nextTotal);
  int64_t currentCount =
    rootVctNormalVerificationPlayouts.load(std::memory_order_acquire);
  while(currentCount < targetCount) {
    if(rootVctNormalVerificationPlayouts.compare_exchange_weak(
      currentCount,currentCount + 1,
      std::memory_order_acq_rel,std::memory_order_acquire
    ))
      return bestMoveLoc;
  }
  return Board::NULL_LOC;
}

int64_t Search::getRootObjectiveEdgeVisits(Player objective) const {
  assert(objective == P_WHITE || objective == P_BLACK);
  if(rootNode == NULL)
    return 0;
  int childrenCapacity;
  const SearchChildPointer* children = rootNode->getChildren(childrenCapacity);
  int64_t visits = 0;
  for(int i = 0; i<childrenCapacity; i++) {
    if(children[i].getIfAllocated() == NULL)
      break;
    visits += children[i].getObjectiveEdgeVisits(objective);
  }
  return visits;
}

Player Search::chooseNormalPlayoutObjective() const {
  if(!searchParams.multiHeadObjectiveSeparatePlayouts)
    return C_EMPTY;
  assert(rootNode != NULL);

  double sideWinProp = getSideToMoveWinObjectiveWeight(*rootNode);
  double whiteWinProp =
    rootPla == P_WHITE ? sideWinProp : 1.0 - sideWinProp;
  double blackWinProp = 1.0 - whiteWinProp;
  int64_t whiteCount = std::max<int64_t>(
    0,
    getRootObjectiveEdgeVisits(P_WHITE) -
      rootWhiteWinEdgeVisitsAtSearchStart
  );
  int64_t blackCount = std::max<int64_t>(
    0,
    getRootObjectiveEdgeVisits(P_BLACK) -
      rootBlackWinEdgeVisitsAtSearchStart
  );
  double nextTotal = (double)(whiteCount + blackCount + 1);
  double whiteDeficit = whiteWinProp * nextTotal - whiteCount;
  double blackDeficit = blackWinProp * nextTotal - blackCount;
  return whiteDeficit > blackDeficit ? P_WHITE : P_BLACK;
}

void Search::addNormalEdgeVisit(
  SearchChildPointer& child, Player normalObjective, int64_t delta
) const {
  child.addEdgeVisits(delta);
  if(normalObjective != C_EMPTY)
    child.addObjectiveEdgeVisits(normalObjective,delta);
}

void Search::choosePolicyGuidance(SearchThread& thread) const {
  thread.policyGuidanceMode = SearchThread::POLICY_GUIDANCE_NONE;
  thread.policyGuidanceAttacker = C_EMPTY;
  if(
    searchParams.multiHeadVctGuidedPlayoutProp <= 0.0 &&
    searchParams.multiHeadDrawGuidedPlayoutProp <= 0.0
  )
    return;

  const NNOutput* nnOutput = rootNode->getNNOutput();
  if(nnOutput == NULL)
    return;
  if(!nnOutput->hasPolicyByHead())
    throw StringError("multi-head guided playouts require a v112 model with six policy heads");

  int policySize = NNPos::getPolicySize(nnOutput->nnXLen,nnOutput->nnYLen);
  int legalPolicyCount = 0;
  for(int movePos = 0; movePos<policySize; movePos++) {
    if(nnOutput->getPolicyProbMaybeNoised(movePos) >= 0.0f)
      legalPolicyCount++;
  }
  double twiceUniformPolicy =
    legalPolicyCount > 0 ? 2.0 / legalPolicyCount : 1.0;
  auto getPolicyConcentration = [&](int head) {
    double peakPolicy = 0.0;
    for(int movePos = 0; movePos<policySize; movePos++) {
      float policy = nnOutput->getPolicyProbByHead(head,movePos);
      if(policy > peakPolicy)
        peakPolicy = policy;
    }
    double peakExcess = std::max(0.0,peakPolicy - twiceUniformPolicy);
    return
      peakExcess /
      (peakExcess + searchParams.multiHeadAuxPolicyConcentrationScale);
  };
  auto getWinProb = [&](int head) {
    return rootPla == P_WHITE ?
      (double)nnOutput->whiteWinProbByHead[head] :
      (double)nnOutput->whiteLossProbByHead[head];
  };

  double vctProp =
    searchParams.multiHeadVctGuidedPlayoutProp *
    std::clamp(getWinProb(4),0.0,1.0) *
    getPolicyConcentration(4);
  double mustWinProp =
    searchParams.multiHeadDrawGuidedPlayoutProp *
    std::clamp(getWinProb(3),0.0,1.0) *
    getPolicyConcentration(3);
  if(searchParams.multiHeadGuidedStartFraction > 0.0) {
    double startFraction = searchParams.multiHeadGuidedStartFraction;
    if(startFraction >= 1.0)
      return;
    int64_t plannedPlayouts = std::min(
      searchParams.maxPlayouts,
      searchParams.maxVisits
    );
    if(plannedPlayouts <= 0)
      return;
    double searchProgress = std::clamp(
      1.0 - thread.upperBoundVisitsLeft / plannedPlayouts,
      0.0,1.0
    );
    double ramp = std::clamp(
      (searchProgress - startFraction) / (1.0 - startFraction),
      0.0,1.0
    );
    vctProp *= ramp;
    mustWinProp *= ramp;
  }
  constexpr double MAX_GUIDED_PROP = 0.85;
  double guidedProp = vctProp + mustWinProp;
  if(guidedProp > MAX_GUIDED_PROP) {
    double scale = MAX_GUIDED_PROP / guidedProp;
    vctProp *= scale;
    mustWinProp *= scale;
  }

  double choice = thread.rand.nextDouble();
  if(choice < vctProp)
    thread.policyGuidanceMode = SearchThread::POLICY_GUIDANCE_VCT;
  else if(choice < vctProp + mustWinProp)
    thread.policyGuidanceMode = SearchThread::POLICY_GUIDANCE_MUST_WIN;
  else
    return;
  thread.policyGuidanceAttacker = rootPla;
}

bool Search::runSinglePlayout(SearchThread& thread, double upperBoundVisitsLeft) {
  //Store this value, used for futile-visit pruning this thread's root children selections.
  thread.upperBoundVisitsLeft = upperBoundVisitsLeft;
  thread.rootForcedReplySidecarMoveLoc =
    chooseForcedReplySidecarMove(thread);
  thread.vctAttacker =
    thread.rootForcedReplySidecarMoveLoc != Board::NULL_LOC ?
      rootPla :
      chooseVctPlayoutAttacker(thread);
  thread.normalObjective =
    thread.vctAttacker == C_EMPTY ?
      chooseNormalPlayoutObjective() :
      C_EMPTY;
  thread.rootVctNormalVerificationMoveLoc =
    thread.vctAttacker == C_EMPTY ?
      chooseVctNormalVerificationMove(thread) :
      Board::NULL_LOC;
  thread.policyGuidanceMode = SearchThread::POLICY_GUIDANCE_NONE;
  thread.policyGuidanceAttacker = C_EMPTY;
  if(
    thread.vctAttacker == C_EMPTY &&
    thread.rootVctNormalVerificationMoveLoc == Board::NULL_LOC
  )
    choosePolicyGuidance(thread);
  if(thread.vctAttacker != C_EMPTY) {
    thread.normalRulesBoard = thread.board;
    thread.normalRulesHistory = thread.history;
    if(!searchParams.multiHeadVctUseNormalRules) {
      thread.history.rules.VCNRule =
        thread.vctAttacker == P_BLACK ? Rules::VCNRULE_VC3_B : Rules::VCNRULE_VC3_W;
      thread.history.rules.firstPassWin = false;
      thread.history.rules.maxMoves = 0;
    }
  }

  bool posesWithChildBuf[NNPos::MAX_NN_POLICY_SIZE];
  bool finishedPlayout = playoutDescend(thread,*rootNode,posesWithChildBuf,true);

  //Restore thread state back to the root state
  thread.pla = rootPla;
  thread.board = rootBoard;
  thread.history = rootHistory;
  thread.normalRulesBoard = rootBoard;
  thread.normalRulesHistory = rootHistory;
  thread.graphHash = rootGraphHash;
  thread.graphPath.clear();
  thread.vctAttacker = C_EMPTY;
  thread.normalObjective = C_EMPTY;
  thread.policyGuidanceMode = SearchThread::POLICY_GUIDANCE_NONE;
  thread.policyGuidanceAttacker = C_EMPTY;
  thread.rootVctNormalVerificationMoveLoc = Board::NULL_LOC;
  thread.rootForcedReplySidecarMoveLoc = Board::NULL_LOC;

  return finishedPlayout;
}

void Search::makeMoveForPlayout(SearchThread& thread, Loc moveLoc) {
  Player movePla = thread.pla;
  thread.history.makeBoardMoveAssumeLegal(thread.board,moveLoc,movePla);
  if(thread.vctAttacker != C_EMPTY) {
    thread.normalRulesHistory.makeBoardMoveAssumeLegal(
      thread.normalRulesBoard,moveLoc,movePla
    );
    assert(thread.normalRulesBoard.pos_hash == thread.board.pos_hash);
  }
  thread.pla = getOpp(movePla);

  if(searchParams.useGraphSearch) {
    const BoardHistory& graphHistory =
      thread.vctAttacker == C_EMPTY ? thread.history : thread.normalRulesHistory;
    thread.graphHash = GraphHash::getGraphHash(graphHistory,thread.pla);
  }
}

bool Search::playoutDescend(
  SearchThread& thread, SearchNode& node,
  bool posesWithChildBuf[NNPos::MAX_NN_POLICY_SIZE],
  bool isRoot
) {
  //Hit terminal node, finish
  //forceNonTerminal marks special nodes where we cannot end the game. This includes the root, since if we are searching a position
  //we presumably want to actually explore deeper and get a result. Also it includes the node following a pass from the root in
  //the case where we are conservativePass.
  //Note that we also carefully clear the search when a pass from the root would be terminal, so nodes should never need to switch
  //status after tree reuse in the latter case.
  if(thread.history.isGameFinished && !node.forceNonTerminal) {
    //Avoid running "too fast", by making sure that a leaf evaluation takes roughly the same time as a genuine nn eval
    //This stops a thread from building a silly number of visits to distort MCTS statistics while other threads are stuck on the GPU.

    if(searchParams.finishGameSearchDelayMicroseconds > 0) {
      uint64_t usToDelay = thread.rand.nextUInt64(searchParams.finishGameSearchDelayMicroseconds * 2);
      std::this_thread::sleep_for(std::chrono::microseconds(usToDelay));
    }
    
    nnEvaluator->waitForNextNNEvalIfAny();
   
      double weight = (searchParams.useUncertainty && nnEvaluator->supportsShorttermError()) ? searchParams.uncertaintyMaxWeight : 1.0;
      if(thread.vctAttacker == C_EMPTY) {
        double whiteWinProb = thread.history.winner == C_WHITE ? 1.0 : 0.0;
        double blackWinProb = thread.history.winner == C_BLACK ? 1.0 : 0.0;
        double noResultValue = thread.history.winner == C_EMPTY ? 1.0 : 0.0;
        double legacyUtility = getResultUtility(whiteWinProb - blackWinProb, noResultValue);
        double whiteWinUtility = getWhiteWinUtility(legacyUtility, whiteWinProb);
        double blackWinUtilityInv = getBlackWinUtilityInv(legacyUtility, blackWinProb);
        addLeafValue(
          node,whiteWinProb,blackWinProb,noResultValue,legacyUtility,
          whiteWinUtility,blackWinUtilityInv,weight,weight,weight,
          true,false,thread.normalObjective
        );
      }
      else {
        double utility;
        if(searchParams.multiHeadVctUseNormalRules) {
          double whiteWinProb = thread.history.winner == C_WHITE ? 1.0 : 0.0;
          double blackWinProb = thread.history.winner == C_BLACK ? 1.0 : 0.0;
          double noResultValue = thread.history.winner == C_EMPTY ? 1.0 : 0.0;
          utility = getResultUtility(whiteWinProb - blackWinProb,noResultValue);
        }
        else {
          bool attackerWon = thread.history.winner == thread.vctAttacker;
          if(thread.vctAttacker == P_WHITE)
            utility = attackerWon ? 1.0 : -1.0;
          else
            utility = attackerWon ? -1.0 : 1.0;
        }
        addVctLeafValue(node,thread.vctAttacker,utility,weight,false);
      }
      return true;
    
  }

  int nodeState = node.state.load(std::memory_order_acquire);
  if(nodeState == SearchNode::STATE_UNEVALUATED) {
    //Always attempt to set a new nnOutput. That way, if some GPU is slow and malfunctioning, we don't get blocked by it.
    {
      bool suc = initNodeNNOutput(thread,node,isRoot,false,false);
      //Leave the node as unevaluated - only the thread that first actually set the nnOutput into the node
      //gets to update the state, to avoid races where we update the state while the node stats aren't updated yet.
      if(!suc)
        return false;
    }

    bool suc = node.state.compare_exchange_strong(nodeState, SearchNode::STATE_EVALUATING, std::memory_order_seq_cst);
    if(!suc) {
      //Presumably someone else got there first.
      //Just give up on this playout and try again from the start.
      return false;
    }
    else {
      //Perform the nn evaluation and finish!
      node.initializeChildren();
      node.state.store(SearchNode::STATE_EXPANDED0, std::memory_order_seq_cst);
      return true;
    }
  }
  else if(nodeState == SearchNode::STATE_EVALUATING) {
    //Just give up on this playout and try again from the start.
    return false;
  }

  assert(nodeState >= SearchNode::STATE_EXPANDED0);
  maybeRecomputeExistingNNOutput(thread,node,isRoot);

  //Find the best child to descend down
  int numChildrenFound;
  int bestChildIdx;
  Loc bestChildMoveLoc;

  SearchNode* child = NULL;
  while(true) {
    selectBestChildToDescend(thread,node,nodeState,numChildrenFound,bestChildIdx,bestChildMoveLoc,posesWithChildBuf,isRoot);

    //The absurdly rare case that the move chosen is not legal
    //(this should only happen either on a bug or where the nnHash doesn't have full legality information or when there's an actual hash collision).
    //Regenerate the neural net call and continue
    //Could also be true if we have an illegal move due to graph search and we had a cycle and superko interaction, or a true collision
    //on an older path that results in bad transposition between positions that don't transpose.
    if(bestChildIdx >= 0 && !thread.history.isLegal(thread.board,bestChildMoveLoc,thread.pla)) {
      bool isReInit = true;
      initNodeNNOutput(thread,node,isRoot,true,isReInit);

      {
        NNOutput* nnOutput = node.getNNOutput();
        assert(nnOutput != NULL);
        Hash128 nnHash = nnOutput->nnHash;
        //In case of a cycle or bad transposition, this will fire a lot, so limit it to once per thread per search.
        if(thread.illegalMoveHashes.find(nnHash) == thread.illegalMoveHashes.end()) {
          thread.illegalMoveHashes.insert(nnHash);
          logger->write("WARNING: Chosen move not legal so regenerated nn output, nnhash=" + nnHash.toString());
          ostringstream out;
          thread.history.printBasicInfo(out,thread.board);
          thread.history.printDebugInfo(out, thread.board);
          out << Location::toString(bestChildMoveLoc, thread.board) << endl;
          cout << Location::toString(bestChildMoveLoc, thread.board) << endl;
          logger->write(out.str());
        }
      }

      //As isReInit is true, we don't return, just keep going, since we didn't count this as a true visit in the node stats
      nodeState = node.state.load(std::memory_order_acquire);
      selectBestChildToDescend(thread,node,nodeState,numChildrenFound,bestChildIdx,bestChildMoveLoc,posesWithChildBuf,isRoot);

      if(bestChildIdx >= 0) {
        //New child
        if(bestChildIdx >= numChildrenFound) {
          //In THEORY it might still be illegal this time! This would be the case if when we initialized the NN output, we raced
          //against someone reInitializing the output to add dirichlet noise or something, who was doing so based on an older cached
          //nnOutput that still had the illegal move. If so, then just fail this playout and try again.
          if(!thread.history.isLegal(thread.board,bestChildMoveLoc,thread.pla))
            return false;
        }
        //Existing child
        else {
          //An illegal move should make it into the tree only in case of cycle or bad transposition
          //We want the search to continue as best it can, so we increment visits so other search branches will still make progress.
          int childrenCapacity;
          SearchChildPointer* children = node.getChildren(nodeState,childrenCapacity);
          assert(childrenCapacity > bestChildIdx);
          if(thread.vctAttacker == C_EMPTY)
            addNormalEdgeVisit(
              children[bestChildIdx],thread.normalObjective,1
            );
          else
            children[bestChildIdx].addVctEdgeVisits(thread.vctAttacker,1);
          return true;
        }
      }
    }

    if(bestChildIdx <= -1) {
      //This might happen if all moves have been forbidden. The node will just get stuck counting visits without expanding
      //and we won't do any search.
      if(thread.vctAttacker == C_EMPTY)
        addCurrentNNOutputAsLeafValue(node,false,thread.normalObjective);
      else
        addCurrentNNOutputAsVctLeafValue(node,thread.vctAttacker,false);
      return true;
    }

    //Do we think we are searching a new child for the first time?
    if(bestChildIdx >= numChildrenFound) {
      assert(bestChildIdx == numChildrenFound);
      assert(bestChildIdx < NNPos::MAX_NN_POLICY_SIZE);
      bool suc = node.maybeExpandChildrenCapacityForNewChild(nodeState, numChildrenFound+1);
      //Someone else is expanding. Loop again trying to select the best child to explore.
      if(!suc) {
        std::this_thread::yield();
        nodeState = node.state.load(std::memory_order_acquire);
        continue;
      }

      int childrenCapacity;
      SearchChildPointer* children = node.getChildren(nodeState,childrenCapacity);
      assert(childrenCapacity > bestChildIdx);

      //Make the move! We need to make the move before we create the node so we can see the new state and get the right graphHash.
      makeMoveForPlayout(thread,bestChildMoveLoc);

      //If conservative pass, passing from the root is always non-terminal
      const bool forceNonTerminal = false;
      child = allocateOrFindNode(thread, thread.pla, bestChildMoveLoc, forceNonTerminal, thread.graphHash);
      child->getVirtualLosses(thread.vctAttacker).fetch_add(1,std::memory_order_release);

      {
        //Lock mutex to store child and move loc in a synchronized way
        std::lock_guard<std::mutex> lock(mutexPool->getMutex(node.mutexIdx));
        SearchNode* existingChild = children[bestChildIdx].getIfAllocated();
        if(existingChild == NULL) {
          //Set relaxed *first*, then release this value via storing the child. Anyone who load-acquires the child
          //is guaranteed by release semantics to see the move as well.
          children[bestChildIdx].setMoveLocRelaxed(bestChildMoveLoc);
          children[bestChildIdx].store(child);
        }
        else {
          //Someone got there ahead of us. We already made a move so we can't just loop again. Instead just fail this playout and try again.
          //Even if the node was newly allocated, no need to delete the node, it will get cleaned up next time we mark and sweep the node table later.
          //Clean up virtual losses in case the node is a transposition and is being used.
          child->getVirtualLosses(thread.vctAttacker).fetch_add(-1,std::memory_order_release);
          return false;
        }
      }

      //If edge visits is too much smaller than the child's visits, we can avoid descending.
      //Instead just add edge visits and treat that as a visit.
      if(maybeCatchUpEdgeVisits(thread, node, child, nodeState, bestChildIdx)) {
        updateStatsAfterPlayout(node,thread,isRoot);
        child->getVirtualLosses(thread.vctAttacker).fetch_add(-1,std::memory_order_release);
        return true;
      }
    }
    //Searching an existing child
    else {
      int childrenCapacity;
      SearchChildPointer* children = node.getChildren(nodeState,childrenCapacity);
      child = children[bestChildIdx].getIfAllocated();
      assert(child != NULL);

      child->getVirtualLosses(thread.vctAttacker).fetch_add(1,std::memory_order_release);

      //If edge visits is too much smaller than the child's visits, we can avoid descending.
      //Instead just add edge visits and treat that as a visit.
      if(maybeCatchUpEdgeVisits(thread, node, child, nodeState, bestChildIdx)) {
        updateStatsAfterPlayout(node,thread,isRoot);
        child->getVirtualLosses(thread.vctAttacker).fetch_add(-1,std::memory_order_release);
        return true;
      }

      //Make the move!
      makeMoveForPlayout(thread,bestChildMoveLoc);
    }

    break;
  }

  //If somehow we find ourselves in a cycle, increment edge visits and terminate the playout.
  //Basically if the search likes a cycle... just reinforce playing around the cycle and hope we return something
  //reasonable in the end of the search.
  {
    std::pair<std::unordered_set<SearchNode*>::iterator,bool> result = thread.graphPath.insert(child);
    //No insertion, child was already there
    if(!result.second) {
      int childrenCapacity;
      SearchChildPointer* children = node.getChildren(nodeState,childrenCapacity);
      if(thread.vctAttacker == C_EMPTY)
        addNormalEdgeVisit(
          children[bestChildIdx],thread.normalObjective,1
        );
      else
        children[bestChildIdx].addVctEdgeVisits(thread.vctAttacker,1);
      updateStatsAfterPlayout(node,thread,isRoot);
      child->getVirtualLosses(thread.vctAttacker).fetch_add(-1,std::memory_order_release);
      return true;
    }
  }

  //Recurse!
  bool finishedPlayout = playoutDescend(thread,*child,posesWithChildBuf,false);
  //Update this node stats
  if(finishedPlayout) {
    nodeState = node.state.load(std::memory_order_acquire);
    int childrenCapacity;
    SearchChildPointer* children = node.getChildren(nodeState,childrenCapacity);
    if(thread.vctAttacker == C_EMPTY)
      addNormalEdgeVisit(
        children[bestChildIdx],thread.normalObjective,1
      );
    else
      children[bestChildIdx].addVctEdgeVisits(thread.vctAttacker,1);
    updateStatsAfterPlayout(node,thread,isRoot);
  }
  child->getVirtualLosses(thread.vctAttacker).fetch_add(-1,std::memory_order_release);

  return finishedPlayout;
}


//If edge visits is too much smaller than the child's visits, we can avoid descending.
//Instead just add edge visits and return immediately.
bool Search::maybeCatchUpEdgeVisits(SearchThread& thread, SearchNode& node, SearchNode* child, const int& nodeState, const int bestChildIdx) {
  //Don't need to do this since we already are pretty recent as of finding the best child.
  //nodeState = node.state.load(std::memory_order_acquire);
  int childrenCapacity;
  SearchChildPointer* children = node.getChildren(nodeState,childrenCapacity);

  if(
    thread.vctAttacker == C_EMPTY &&
    thread.normalObjective != C_EMPTY
  ) {
    double objectiveWeightSum =
      thread.normalObjective == P_WHITE ?
        child->stats.whiteWinWeightSum.load(std::memory_order_acquire) :
        child->stats.blackWinWeightSum.load(std::memory_order_acquire);
    double childVisits = std::max(1.0,ceil(objectiveWeightSum));
    Player otherObjective = getOpp(thread.normalObjective);
    double crossWeight = searchParams.multiHeadObjectiveCrossWeight;
    while(true) {
      int64_t selectedEdgeVisits =
        children[bestChildIdx].getObjectiveEdgeVisits(
          thread.normalObjective
        );
      int64_t otherEdgeVisits =
        children[bestChildIdx].getObjectiveEdgeVisits(otherObjective);
      double creditedEdgeVisits =
        selectedEdgeVisits + crossWeight * otherEdgeVisits;
      if(creditedEdgeVisits >= childVisits)
        return false;
      if(
        searchParams.graphSearchCatchUpLeakProb > 0.0 &&
        thread.rand.nextBool(searchParams.graphSearchCatchUpLeakProb)
      )
        return false;
      int64_t expected = selectedEdgeVisits;
      if(children[bestChildIdx].compexweakObjectiveEdgeVisits(
        thread.normalObjective,expected,selectedEdgeVisits + 1
      )) {
        children[bestChildIdx].addEdgeVisits(1);
        return true;
      }
    }
  }

  // int64_t maxNumToAdd = 1;
  // if(searchParams.graphSearchCatchUpProp > 0.0) {
  //   int64_t parentVisits = node.stats.visits.load(std::memory_order_acquire);
  //   //Truncate down
  //   maxNumToAdd = 1 + (int64_t)(searchParams.graphSearchCatchUpProp * parentVisits);
  // }
  int64_t childVisits;
  int64_t edgeVisits;
  if(thread.vctAttacker == C_EMPTY) {
    childVisits = child->stats.visits.load(std::memory_order_acquire);
    edgeVisits = children[bestChildIdx].getEdgeVisits();
  }
  else {
    childVisits = child->getVctStats(thread.vctAttacker).visits.load(std::memory_order_acquire);
    edgeVisits = children[bestChildIdx].getVctEdgeVisits(thread.vctAttacker);
  }

  //If we want to leak through some of the time, then we keep searching the transposition node even if we'd be happy to stop here with
  //how many visits it has
  if(searchParams.graphSearchCatchUpLeakProb > 0.0 && edgeVisits < childVisits && thread.rand.nextBool(searchParams.graphSearchCatchUpLeakProb))
    return false;

  //If the edge visits exceeds the child then we need to search the child more, but as long as that's not the case,
  //we can add more edge visits.
  constexpr int64_t numToAdd = 1;
  // int64_t numToAdd;
  do {
    if(edgeVisits >= childVisits)
      return false;
    // numToAdd = std::min((childVisits - edgeVisits + 3) / 4, maxNumToAdd);
  } while(
    thread.vctAttacker == C_EMPTY ?
      !children[bestChildIdx].compexweakEdgeVisits(
        edgeVisits,edgeVisits + numToAdd
      ) :
      !children[bestChildIdx].compexweakVctEdgeVisits(
        thread.vctAttacker,edgeVisits,edgeVisits + numToAdd
      )
  );

  return true;
}

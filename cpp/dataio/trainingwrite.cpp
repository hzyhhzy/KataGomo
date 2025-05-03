#include "../dataio/trainingwrite.h"

#include "../core/fileutils.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nneval.h"

using namespace std;

ValueTargets::ValueTargets()
  :win(0),
   loss(0),
   noResult(0)
{}
ValueTargets::~ValueTargets()
{}

//-------------------------------------------------------------------------------------

SidePosition::SidePosition()
  :board(),
   hist(),
   pla(P_BLACK),
   unreducedNumVisits(),
   policyTarget(),
   policySurprise(),
   policyEntropy(),
   searchEntropy(),
   whiteValueTargets(),
   targetWeight(),
   targetWeightUnrounded(),
   numNeuralNetChangesSoFar()
{}

SidePosition::SidePosition(const Board& b, const BoardHistory& h, Player p, int numNNChangesSoFar)
  :board(b),
   hist(h),
   pla(p),
   unreducedNumVisits(),
   policyTarget(),
   policySurprise(),
   policyEntropy(),
   searchEntropy(),
   whiteValueTargets(),
   targetWeight(1.0f),
   targetWeightUnrounded(1.0f),
   numNeuralNetChangesSoFar(numNNChangesSoFar)
{}

SidePosition::~SidePosition()
{}

//-------------------------------------------------------------------------------------

FinishedGameData::FinishedGameData()
  :bName(),
   wName(),
   bIdx(0),
   wIdx(0),

   startBoard(),
   startHist(),
   endHist(),
   startPla(P_BLACK),
   gameHash(),

   noResultUtilityForWhite(0.0),
   playoutDoublingAdvantagePla(P_BLACK),
   playoutDoublingAdvantage(0.0),

   hitTurnLimit(false),

   mode(0),
   usedInitialPosition(0),

   hasFullData(false),
   targetWeightByTurn(),
   targetWeightByTurnUnrounded(),
   policyTargetsByTurn(),
   whiteValueTargetsByTurn(),
   nnRawStatsByTurn(),

   sidePositions(),
   changedNeuralNets(),
   bTimeUsed(0.0),
   wTimeUsed(0.0),
   bMoveCount(0),
   wMoveCount(0)
{
}

FinishedGameData::~FinishedGameData() {
  for(size_t i = 0; i<policyTargetsByTurn.size(); i++)
    delete policyTargetsByTurn[i].policyTargets;


  for(size_t i = 0; i<sidePositions.size(); i++)
    delete sidePositions[i];

  for(size_t i = 0; i<changedNeuralNets.size(); i++)
    delete changedNeuralNets[i];
}

void FinishedGameData::printDebug(ostream& out) const {
  out << "bName " << bName << endl;
  out << "wName " << wName << endl;
  out << "bIdx " << bIdx << endl;
  out << "wIdx " << wIdx << endl;
  out << "startPla " << PlayerIO::colorToChar(startPla) << endl;
  out << "start" << endl;
  startHist.printDebugInfo(out,startBoard);
  out << "end" << endl;
  endHist.printDebugInfo(out,endHist.getRecentBoard(0));
  out << "gameHash " << gameHash << endl;
  out << "hitTurnLimit " << hitTurnLimit << endl;
  out << "mode " << mode << endl;
  out << "usedInitialPosition " << usedInitialPosition << endl;
  out << "hasFullData " << hasFullData << endl;
  for(int i = 0; i<targetWeightByTurn.size(); i++)
    out << "targetWeightByTurn " << i << " " << targetWeightByTurn[i] << " " << "unrounded" << " " << targetWeightByTurnUnrounded[i] << endl;
  for(int i = 0; i<policyTargetsByTurn.size(); i++) {
    out << "policyTargetsByTurn " << i << " ";
    out << "unreducedNumVisits " << policyTargetsByTurn[i].unreducedNumVisits << " ";
    if(policyTargetsByTurn[i].policyTargets != NULL) {
      const vector<PolicyTargetMove>& target = *(policyTargetsByTurn[i].policyTargets);
      for(int j = 0; j<target.size(); j++)
        out << Location::toString(target[j].loc,startBoard) << " " << target[j].policyTarget << " ";
    }
    out << endl;
  }
  for (int i = 0; i < policySurpriseByTurn.size(); i++)
    out << "policySurpriseByTurn " << i << " " << policySurpriseByTurn[i] << endl;
  for (int i = 0; i < policyEntropyByTurn.size(); i++)
    out << "policyEntropyByTurn " << i << " " << policyEntropyByTurn[i] << endl;
  for (int i = 0; i < searchEntropyByTurn.size(); i++)
    out << "searchEntropyByTurn " << i << " " << searchEntropyByTurn[i] << endl;

  for(int i = 0; i<whiteValueTargetsByTurn.size(); i++) {
    out << "whiteValueTargetsByTurn " << i << " ";
    out << whiteValueTargetsByTurn[i].win << " ";
    out << whiteValueTargetsByTurn[i].loss << " ";
    out << whiteValueTargetsByTurn[i].noResult << " ";
    out << endl;
  }
  for(int i = 0; i<nnRawStatsByTurn.size(); i++) {
    out << "Raw Stats " << nnRawStatsByTurn[i].whiteWinLoss << " " << nnRawStatsByTurn[i].policyEntropy << endl;
  }
  for(int i = 0; i<sidePositions.size(); i++) {
    SidePosition* sp = sidePositions[i];
    out << "Side position " << i << endl;
    out << "targetWeight " << sp->targetWeight << " " << "unrounded" << " " << sp->targetWeightUnrounded << endl;
    sp->hist.printDebugInfo(out,sp->board);
  }
}

//-------------------------------------------------------------------------------------


//Don't forget to update everything else in the header file and the code below too if changing any of these
//And update the python code
static const int POLICY_TARGET_NUM_CHANNELS = 2;
static const int GLOBAL_TARGET_NUM_CHANNELS = 64;
static const int VALUE_SPATIAL_TARGET_NUM_CHANNELS = 5;

TrainingWriteBuffers::TrainingWriteBuffers(int iVersion, int maxRws, int numBChannels, int numFChannels, int xLen, int yLen)
  :inputsVersion(iVersion),
   maxRows(maxRws),
   numBinaryChannels(numBChannels),
   numGlobalChannels(numFChannels),
   dataXLen(xLen),
   dataYLen(yLen),
   packedBoardArea((xLen*yLen + 7)/8),
   curRows(0),
   binaryInputNCHWUnpacked(NULL),
   binaryInputNCHWPacked({maxRws, numBChannels, packedBoardArea}),
   globalInputNC({maxRws, numFChannels}),
   policyTargetsNCMove({maxRws, POLICY_TARGET_NUM_CHANNELS, NNPos::getPolicySize(xLen,yLen)}),
   globalTargetsNC({maxRws, GLOBAL_TARGET_NUM_CHANNELS}),
   scoreDistrN({maxRws, xLen * yLen * 2 + NNPos::EXTRA_SCORE_DISTR_RADIUS * 2}),
   valueTargetsNCHW({maxRws, VALUE_SPATIAL_TARGET_NUM_CHANNELS, yLen, xLen})
{
  binaryInputNCHWUnpacked = new float[numBChannels * xLen * yLen];
}

TrainingWriteBuffers::~TrainingWriteBuffers()
{
  delete[] binaryInputNCHWUnpacked;
}

void TrainingWriteBuffers::clear() {
  curRows = 0;
}

//Copy floats that are all 0-1 into bits, packing 8 to a byte, big-endian-style within each byte.
static void packBits(const float* binaryFloats, int len, uint8_t* bits) {
  for(int i = 0; i < len; i += 8) {
    if(i + 8 <= len) {
      bits[i >> 3] =
        ((uint8_t)binaryFloats[i + 0] << 7) |
        ((uint8_t)binaryFloats[i + 1] << 6) |
        ((uint8_t)binaryFloats[i + 2] << 5) |
        ((uint8_t)binaryFloats[i + 3] << 4) |
        ((uint8_t)binaryFloats[i + 4] << 3) |
        ((uint8_t)binaryFloats[i + 5] << 2) |
        ((uint8_t)binaryFloats[i + 6] << 1) |
        ((uint8_t)binaryFloats[i + 7] << 0);
    }
    else {
      bits[i >> 3] = 0;
      for(int di = 0; i + di < len; di++) {
        bits[i >> 3] |= ((uint8_t)binaryFloats[i + di] << (7-di));
      }
    }
  }
}

static void zeroPolicyTarget(int policySize, int16_t* target) {
  for(int pos = 0; pos<policySize; pos++)
    target[pos] = 0;
}

static void uniformPolicyTarget(int policySize, int16_t* target) {
  for(int pos = 0; pos<policySize; pos++)
    target[pos] = 1;
}

//Copy playouts into target, expanding out the sparse representation into a full plane.
static void fillPolicyTarget(const vector<PolicyTargetMove>& policyTargetMoves, int policySize, int dataXLen, int dataYLen, int boardXSize, int16_t* target) {
  zeroPolicyTarget(policySize,target);
  size_t size = policyTargetMoves.size();
  for(size_t i = 0; i<size; i++) {
    const PolicyTargetMove& move = policyTargetMoves[i];
    int pos = NNPos::locToPos(move.loc, boardXSize, dataXLen, dataYLen);
    assert(pos >= 0 && pos < policySize);
    target[pos] = move.policyTarget;
  }
}




static void fillValueTDTargets(const vector<ValueTargets>& whiteValueTargetsByTurn, int idx, Player nextPlayer, double nowFactor, float* buf) {
  double winValue = 0.0;
  double lossValue = 0.0;
  double noResultValue = 0.0;

  double weightLeft = 1.0;
  for(int i = idx; i<whiteValueTargetsByTurn.size(); i++) {
    double weightNow;
    if(i == whiteValueTargetsByTurn.size() - 1) {
      weightNow = weightLeft;
      weightLeft = 0.0;
    }
    else {
      weightNow = weightLeft * nowFactor;
      weightLeft *= (1.0 - nowFactor);
    }

    //Training rows need things from the perspective of the player to move, so we flip as appropriate.
    const ValueTargets& targets = whiteValueTargetsByTurn[i];
    winValue += weightNow * (nextPlayer == P_WHITE ? targets.win : targets.loss);
    lossValue += weightNow * (nextPlayer == P_WHITE ? targets.loss : targets.win);
    noResultValue += weightNow * targets.noResult;
  }
  buf[0] = (float)winValue;
  buf[1] = (float)lossValue;
  buf[2] = (float)noResultValue;
  buf[3] = 0.0f;
}

void TrainingWriteBuffers::addRow(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  int turnIdx,
  float targetWeight,
  int64_t unreducedNumVisits,
  const vector<PolicyTargetMove>* policyTarget0, //can be null
  const vector<PolicyTargetMove>* policyTarget1, //can be null
  double policySurprise,
  double policyEntropy,
  double searchEntropy,
  const vector<ValueTargets>& whiteValueTargets,
  int whiteValueTargetsIdx, //index in whiteValueTargets corresponding to this turn.
  const NNRawStats& nnRawStats,
  const Board* finalBoard,
  const vector<Board>* posHistForFutureBoards, //can be null
  bool isSidePosition,
  int numNeuralNetsBehindLatest,
  const FinishedGameData& data,
  Rand& rand,
  NNEvaluator* distillEval

) {
  (void)finalBoard;
  static_assert(NNModelVersion::latestInputsVersionImplemented == 102, "");
  if(inputsVersion < 3 || inputsVersion > 102)
    throw StringError("Training write buffers: Does not support input version: " + Global::intToString(inputsVersion));

  int posArea = dataXLen*dataYLen;
  assert(data.hasFullData);
  assert(curRows < maxRows);

  MiscNNInputParams nnInputParams;
  {
    nnInputParams.noResultUtilityForWhite = data.noResultUtilityForWhite;
    //Note: this is coordinated with the fact that selfplay does not use this feature on side positions
    if(!isSidePosition)
      nnInputParams.playoutDoublingAdvantage = getOpp(nextPlayer) == data.playoutDoublingAdvantagePla ? -data.playoutDoublingAdvantage : data.playoutDoublingAdvantage;

    nnInputParams.resultsBeforeNN.init(board, hist, nextPlayer);

    bool inputsUseNHWC = false;
    float* rowBin = binaryInputNCHWUnpacked;
    float* rowGlobal = globalInputNC.data + curRows * numGlobalChannels;
    static_assert(NNModelVersion::latestInputsVersionImplemented == 102, "");
    if(inputsVersion == 7) {
      assert(NNInputs::NUM_FEATURES_SPATIAL_V7 == numBinaryChannels);
      assert(NNInputs::NUM_FEATURES_GLOBAL_V7 == numGlobalChannels);
      NNInputs::fillRowV7(board, hist, nextPlayer, nnInputParams, dataXLen, dataYLen, inputsUseNHWC, rowBin, rowGlobal);
    } 
    else if(inputsVersion == 101) {
      assert(NNInputs::NUM_FEATURES_SPATIAL_V101 == numBinaryChannels);
      assert(NNInputs::NUM_FEATURES_GLOBAL_V101 == numGlobalChannels);
      NNInputs::fillRowV101(
        board, hist, nextPlayer, nnInputParams, dataXLen, dataYLen, inputsUseNHWC, rowBin, rowGlobal);
    } 
    else if(inputsVersion == 102) {
      assert(NNInputs::NUM_FEATURES_SPATIAL_V102 == numBinaryChannels);
      assert(NNInputs::NUM_FEATURES_GLOBAL_V102 == numGlobalChannels);
      NNInputs::fillRowV102(
        board, hist, nextPlayer, nnInputParams, dataXLen, dataYLen, inputsUseNHWC, rowBin, rowGlobal);
    }
    else
      ASSERT_UNREACHABLE;

    //Pack bools bitwise into uint8_t
    uint8_t* rowBinPacked = binaryInputNCHWPacked.data + curRows * numBinaryChannels * packedBoardArea;
    for(int c = 0; c<numBinaryChannels; c++)
      packBits(rowBin + c * posArea, posArea, rowBinPacked + c * packedBoardArea);
  }

  if(distillEval == NULL) {
    // Vector for global targets and metadata
    float* rowGlobal = globalTargetsNC.data + curRows * GLOBAL_TARGET_NUM_CHANNELS;

    // Target weight for the whole row
    rowGlobal[25] = targetWeight;

    // Fill policy
    int policySize = NNPos::getPolicySize(dataXLen, dataYLen);
    int16_t* rowPolicy = policyTargetsNCMove.data + curRows * POLICY_TARGET_NUM_CHANNELS * policySize;

    if(policyTarget0 != NULL) {
      fillPolicyTarget(*policyTarget0, policySize, dataXLen, dataYLen, board.x_size, rowPolicy + 0 * policySize);
      rowGlobal[26] = 1.0f;
    } else {
      uniformPolicyTarget(policySize, rowPolicy + 0 * policySize);
      rowGlobal[26] = 0.0f;
    }

    if(policyTarget1 != NULL) {
      fillPolicyTarget(*policyTarget1, policySize, dataXLen, dataYLen, board.x_size, rowPolicy + 1 * policySize);
      rowGlobal[28] = 1.0f;
    } else {
      uniformPolicyTarget(policySize, rowPolicy + 1 * policySize);
      rowGlobal[28] = 0.0f;
    }

    // Fill td-like value targets
    int boardArea = board.x_size * board.y_size;
    assert(whiteValueTargetsIdx >= 0 && whiteValueTargetsIdx < whiteValueTargets.size());
    fillValueTDTargets(whiteValueTargets, whiteValueTargetsIdx, nextPlayer, 0.0, rowGlobal);
    // These three constants used to be 'nicer' numbers 0.18, 0.06, 0.02, but we screwed up the functional form
    // by omitting the "1.0 +" at the front (breaks scaling to small board sizes), so when we fixed this we also
    // decreased the other numbers slightly to try to maximally limit the impact of the fix on the numerical values
    // on the actual board sizes 9-19, since it would be costly to retest.
    fillValueTDTargets(
      whiteValueTargets, whiteValueTargetsIdx, nextPlayer, 1.0 / (1.0 + boardArea * 0.176), rowGlobal + 4);
    fillValueTDTargets(
      whiteValueTargets, whiteValueTargetsIdx, nextPlayer, 1.0 / (1.0 + boardArea * 0.056), rowGlobal + 8);
    fillValueTDTargets(
      whiteValueTargets, whiteValueTargetsIdx, nextPlayer, 1.0 / (1.0 + boardArea * 0.016), rowGlobal + 12);
    fillValueTDTargets(whiteValueTargets, whiteValueTargetsIdx, nextPlayer, 1.0, rowGlobal + 16);

    // Lead
    rowGlobal[21] = 0.0f;
    rowGlobal[29] = 0.0f;

    // Expected time of arrival of winloss variance, in turns
    {
      double sum = 0.0;
      for(int i = whiteValueTargetsIdx + 1; i < whiteValueTargets.size(); i++) {
        int turnsFromNow = i - whiteValueTargetsIdx;
        const ValueTargets& prevTargets = whiteValueTargets[i - 1];
        const ValueTargets& targets = whiteValueTargets[i];
        double prevWL = prevTargets.win - prevTargets.loss;
        double nextWL = targets.win - targets.loss;
        double variance = (nextWL - prevWL) * (nextWL - prevWL);
        sum += turnsFromNow * variance;
      }
      rowGlobal[22] = (float)sum;
    }

    // Unused
    rowGlobal[23] = 0.0f;
    rowGlobal[24] = 0.0f;
    rowGlobal[30] = (float)policySurprise;
    rowGlobal[31] = (float)policyEntropy;
    rowGlobal[32] = (float)searchEntropy;
    rowGlobal[35] = 0.0f;

    // Fill in whether we should use history or not
    bool useHist0 = rand.nextDouble() < 0.98;
    bool useHist1 = useHist0 && rand.nextDouble() < 0.98;
    bool useHist2 = useHist1 && rand.nextDouble() < 0.98;
    bool useHist3 = useHist2 && rand.nextDouble() < 0.98;
    bool useHist4 = useHist3 && rand.nextDouble() < 0.98;
    rowGlobal[36] = useHist0 ? 1.0f : 0.0f;
    rowGlobal[37] = useHist1 ? 1.0f : 0.0f;
    rowGlobal[38] = useHist2 ? 1.0f : 0.0f;
    rowGlobal[39] = useHist3 ? 1.0f : 0.0f;
    rowGlobal[40] = useHist4 ? 1.0f : 0.0f;

    // Fill in hash of game
    Hash128 gameHash = data.gameHash;
    rowGlobal[41] = (float)(gameHash.hash0 & 0x3FFFFF);
    rowGlobal[42] = (float)((gameHash.hash0 >> 22) & 0x3FFFFF);
    rowGlobal[43] = (float)((gameHash.hash0 >> 44) & 0xFFFFF);
    rowGlobal[44] = (float)(gameHash.hash1 & 0x3FFFFF);
    rowGlobal[45] = (float)((gameHash.hash1 >> 22) & 0x3FFFFF);
    rowGlobal[46] = (float)((gameHash.hash1 >> 44) & 0xFFFFF);

    // Various other data
    rowGlobal[47] = 0.0f;
    rowGlobal[48] = 1.0f;

    // Earlier neural net metadata
    rowGlobal[49] = data.changedNeuralNets.size() > 0 ? 1.0f : 0.0f;
    rowGlobal[50] = (float)numNeuralNetsBehindLatest;

    // Some misc metadata
    rowGlobal[51] = (float)turnIdx;
    rowGlobal[52] = data.hitTurnLimit ? 1.0f : 0.0f;
    rowGlobal[53] = (float)data.startHist.moveHistory.size();
    rowGlobal[54] = 0.0f;

    // Metadata about how the game was initialized
    rowGlobal[55] = (float)data.mode;
    rowGlobal[56] = (float)hist.initialTurnNumber;

    // Some stats
    rowGlobal[57] = (float)(nextPlayer == P_WHITE ? nnRawStats.whiteWinLoss : -nnRawStats.whiteWinLoss);
    rowGlobal[58] = 0.0f;
    rowGlobal[59] = (float)nnRawStats.policyEntropy;

    // Original number of visits
    rowGlobal[60] = (float)unreducedNumVisits;

    // Bonus points
    rowGlobal[61] = 0.0f;

    // Unused
    rowGlobal[62] = 0.0f;

    // Version
    rowGlobal[63] = 1.0f;

    assert(64 == GLOBAL_TARGET_NUM_CHANNELS);

    int scoreDistrLen = posArea * 2 + NNPos::EXTRA_SCORE_DISTR_RADIUS * 2;
    // int scoreDistrMid = posArea + NNPos::EXTRA_SCORE_DISTR_RADIUS;
    int8_t* rowScoreDistr = scoreDistrN.data + curRows * scoreDistrLen;
    int8_t* rowOwnership = valueTargetsNCHW.data + curRows * VALUE_SPATIAL_TARGET_NUM_CHANNELS * posArea;

    rowGlobal[27] = 0.0f;
    rowGlobal[20] = 0.0f;

    // Fill with zeros in case the buffers differ in size
    for(int i = 0; i < posArea * 2; i++)
      rowOwnership[i] = 0;

    // Fill score vector "onehot"-like
    for(int i = 0; i < scoreDistrLen; i++)
      rowScoreDistr[i] = 0;

    if(posHistForFutureBoards == NULL) {
      rowGlobal[33] = 0.0f;
      for(int i = 0; i < posArea; i++) {
        rowOwnership[i + posArea * 2] = 0;
        rowOwnership[i + posArea * 3] = 0;
      }
    } else {
      const vector<Board>& boards = *posHistForFutureBoards;
      assert(boards.size() == whiteValueTargets.size());
      assert(boards.size() > 0);

      rowGlobal[33] = 1.0f;
      int endIdx = (int)boards.size() - 1;
      const Board& board2 = boards[std::min(whiteValueTargetsIdx + 8, endIdx)];
      const Board& board3 = boards[std::min(whiteValueTargetsIdx + 32, endIdx)];
      assert(board2.y_size == board.y_size && board2.x_size == board.x_size);
      assert(board3.y_size == board.y_size && board3.x_size == board.x_size);

      for(int i = 0; i < posArea; i++) {
        rowOwnership[i + posArea * 2] = 0;
        rowOwnership[i + posArea * 3] = 0;
      }
      Player pla = nextPlayer;
      Player opp = getOpp(nextPlayer);
      for(int y = 0; y < board.y_size; y++) {
        for(int x = 0; x < board.x_size; x++) {
          int pos = NNPos::xyToPos(x, y, dataXLen);
          Loc loc = Location::getLoc(x, y, board.x_size);
          if(board2.colors[loc] == pla)
            rowOwnership[pos + posArea * 2] = 1;
          else if(board2.colors[loc] == opp)
            rowOwnership[pos + posArea * 2] = -1;
          if(board3.colors[loc] == pla)
            rowOwnership[pos + posArea * 3] = 1;
          else if(board3.colors[loc] == opp)
            rowOwnership[pos + posArea * 3] = -1;
        }
      }
    }

    rowGlobal[34] = 0.0f;
    for(int i = 0; i < posArea; i++) {
      rowOwnership[i + posArea * 4] = 0;
    }
  } 
  else  // Distill
  {

    NNResultBuf nnResultBuf;
    Board boardCopy = board;
    distillEval->evaluate(boardCopy, hist, nextPlayer, nnInputParams, nnResultBuf, false);
    NNOutput& nnResult = *(nnResultBuf.result);

    // Vector for global targets and metadata
    float* rowGlobal = globalTargetsNC.data + curRows * GLOBAL_TARGET_NUM_CHANNELS;

    // Target weight for the whole row
    float targetWeight1 = 1.0;
    rowGlobal[25] = targetWeight1;

    // Fill policy
    int policySize = NNPos::getPolicySize(dataXLen, dataYLen);
    int16_t* rowPolicy = policyTargetsNCMove.data + curRows * POLICY_TARGET_NUM_CHANNELS * policySize;

    {
      double maxPolicy = -1;
      for(int i = 0; i < policySize; i++) {
        double p = nnResult.getPolicyProb(i);
        if(p > maxPolicy)
          maxPolicy = p;
      }
      assert(maxPolicy > 0);
      assert(maxPolicy < 1.001);

      double fac = 30000 / maxPolicy;

      int16_t* target = rowPolicy + 0 * policySize;
      zeroPolicyTarget(policySize, target);
      for(size_t i = 0; i < policySize; i++) {
        double p = nnResult.getPolicyProb(i) * fac;
        if(p > 0)
          target[i] = int16_t(p);
        else  // notice that illegal moves are -1
          target[i] = 0;
      }

      rowGlobal[26] = 1.0f;
    }

    {
      uniformPolicyTarget(policySize, rowPolicy + 1 * policySize);
      rowGlobal[28] = 0.0f;
    }

    {
      double winValue = nextPlayer == C_WHITE ? nnResult.whiteWinProb : nnResult.whiteLossProb;
      double lossValue = nextPlayer == C_WHITE ? nnResult.whiteLossProb : nnResult.whiteWinProb;
      double noResultValue = nnResult.whiteNoResultProb;
      double score = 0;

      // float originWR[4];
      // fillValueTDTargets(whiteValueTargets, whiteValueTargetsIdx, nextPlayer, 1.0, originWR);
      // cout << "Distill: " << winValue << " " << score << "  Origin: " << originWR[0] << " " << originWR[3] << endl;

      for(int t = 0; t < 5; t++) {
        float* buf = rowGlobal + 4 * t;
        buf[0] = (float)winValue;
        buf[1] = (float)lossValue;
        buf[2] = (float)noResultValue;
        buf[3] = (float)score;
      }
    }
    // Lead
    rowGlobal[21] = 0.0f;
    rowGlobal[29] = 0.0f;

    // Expected time of arrival of winloss variance, in turns
    { rowGlobal[22] = nnResult.varTimeLeft; }

    // Unused
    rowGlobal[23] = 0.0f;
    rowGlobal[24] = (float)(0.0f);
    rowGlobal[30] = (float)policySurprise;
    rowGlobal[31] = (float)policyEntropy;
    rowGlobal[32] = (float)searchEntropy;
    // Value weight
    rowGlobal[35] = (float)(0.0f);

    // Fill in whether we should use history or not
    bool useHist0 = rand.nextDouble() < 0.98;
    bool useHist1 = useHist0 && rand.nextDouble() < 0.98;
    bool useHist2 = useHist1 && rand.nextDouble() < 0.98;
    bool useHist3 = useHist2 && rand.nextDouble() < 0.98;
    bool useHist4 = useHist3 && rand.nextDouble() < 0.98;
    rowGlobal[36] = useHist0 ? 1.0f : 0.0f;
    rowGlobal[37] = useHist1 ? 1.0f : 0.0f;
    rowGlobal[38] = useHist2 ? 1.0f : 0.0f;
    rowGlobal[39] = useHist3 ? 1.0f : 0.0f;
    rowGlobal[40] = useHist4 ? 1.0f : 0.0f;

    // Fill in hash of game
    Hash128 gameHash = data.gameHash;
    rowGlobal[41] = (float)(gameHash.hash0 & 0x3FFFFF);
    rowGlobal[42] = (float)((gameHash.hash0 >> 22) & 0x3FFFFF);
    rowGlobal[43] = (float)((gameHash.hash0 >> 44) & 0xFFFFF);
    rowGlobal[44] = (float)(gameHash.hash1 & 0x3FFFFF);
    rowGlobal[45] = (float)((gameHash.hash1 >> 22) & 0x3FFFFF);
    rowGlobal[46] = (float)((gameHash.hash1 >> 44) & 0xFFFFF);

    // Various other data
    rowGlobal[47] = 0.0f;
    rowGlobal[48] = 0.0f;

    // Earlier neural net metadata
    rowGlobal[49] = data.changedNeuralNets.size() > 0 ? 1.0f : 0.0f;
    rowGlobal[50] = (float)numNeuralNetsBehindLatest;

    // Some misc metadata
    rowGlobal[51] = (float)turnIdx;
    rowGlobal[52] = data.hitTurnLimit ? 1.0f : 0.0f;
    rowGlobal[53] = (float)data.startHist.moveHistory.size();
    rowGlobal[54] = 0.0f;

    // Metadata about how the game was initialized
    rowGlobal[55] = (float)data.mode;
    rowGlobal[56] = (float)hist.initialTurnNumber;

    // Some stats
    rowGlobal[57] = (float)(nextPlayer == C_WHITE ? nnResult.whiteWinProb - nnResult.whiteLossProb
                                                  : -(nnResult.whiteWinProb - nnResult.whiteLossProb));
    rowGlobal[58] = 0.0f;
    rowGlobal[59] = (float)nnRawStats.policyEntropy;

    // Original number of visits
    rowGlobal[60] = (float)unreducedNumVisits;

    rowGlobal[61] = 0.0f;

    // Game finished
    rowGlobal[62] = 1.0f;

    // Version
    rowGlobal[63] = 2.0f;

    assert(64 == GLOBAL_TARGET_NUM_CHANNELS);

    int scoreDistrLen = posArea * 2 + NNPos::EXTRA_SCORE_DISTR_RADIUS * 2;
    int scoreDistrMid = posArea + NNPos::EXTRA_SCORE_DISTR_RADIUS;
    int8_t* rowScoreDistr = scoreDistrN.data + curRows * scoreDistrLen;
    int8_t* rowOwnership = valueTargetsNCHW.data + curRows * VALUE_SPATIAL_TARGET_NUM_CHANNELS * posArea;

    {
      rowGlobal[27] = 0.0f;
      rowGlobal[20] = 0.0f;
      for(int i = 0; i < posArea * 2; i++)
        rowOwnership[i] = 0;
      for(int i = 0; i < scoreDistrLen; i++)
        rowScoreDistr[i] = 0;
      // Dummy value, to make sure it still sums to 100
      rowScoreDistr[scoreDistrMid - 1] = 50;
      rowScoreDistr[scoreDistrMid] = 50;
    }

    {
      rowGlobal[33] = 0.0f;
      for(int i = 0; i < posArea; i++) {
        rowOwnership[i + posArea * 2] = 0;
        rowOwnership[i + posArea * 3] = 0;
      }
    }

    {
      rowGlobal[34] = 0.0f;
      for(int i = 0; i < posArea; i++) {
        rowOwnership[i + posArea * 4] = 0;
      }
    }
  }

  curRows++;
}

void TrainingWriteBuffers::writeToZipFile(const string& fileName) {
  ZipFile zipFile(fileName);

  uint64_t numBytes;

  numBytes = binaryInputNCHWPacked.prepareHeaderWithNumRows(curRows);
  zipFile.writeBuffer("binaryInputNCHWPacked", binaryInputNCHWPacked.dataIncludingHeader, numBytes);

  numBytes = globalInputNC.prepareHeaderWithNumRows(curRows);
  zipFile.writeBuffer("globalInputNC", globalInputNC.dataIncludingHeader, numBytes);

  numBytes = policyTargetsNCMove.prepareHeaderWithNumRows(curRows);
  zipFile.writeBuffer("policyTargetsNCMove", policyTargetsNCMove.dataIncludingHeader, numBytes);

  numBytes = globalTargetsNC.prepareHeaderWithNumRows(curRows);
  zipFile.writeBuffer("globalTargetsNC", globalTargetsNC.dataIncludingHeader, numBytes);

  numBytes = scoreDistrN.prepareHeaderWithNumRows(curRows);
  zipFile.writeBuffer("scoreDistrN", scoreDistrN.dataIncludingHeader, numBytes);

  numBytes = valueTargetsNCHW.prepareHeaderWithNumRows(curRows);
  zipFile.writeBuffer("valueTargetsNCHW", valueTargetsNCHW.dataIncludingHeader, numBytes);

  zipFile.close();
}

void TrainingWriteBuffers::writeToTextOstream(ostream& out) {
  int64_t len;

  auto printHeader = [&out](const char* dataIncludingHeader) {
    //In actuality our headers aren't that long, so we cut it off at half the total header bytes
    for(int i = 0; i<10; i++)
      out << (int)dataIncludingHeader[i] << " ";
    for(int i = 10; i<NumpyBuffer<int>::TOTAL_HEADER_BYTES/2; i++)
      out << dataIncludingHeader[i];
    out << endl;
  };

  out << "binaryInputNCHWPacked" << endl;
  binaryInputNCHWPacked.prepareHeaderWithNumRows(curRows);
  char buf[32];
  printHeader((const char*)binaryInputNCHWPacked.dataIncludingHeader);
  len = binaryInputNCHWPacked.getActualDataLen(curRows);
  for(int i = 0; i<len; i++) {
    sprintf(buf,"%02X",binaryInputNCHWPacked.data[i]);
    out << buf;
    if((i+1) % (len/curRows) == 0) out << endl;
  }
  out << endl;

  out << "globalInputNC" << endl;
  globalInputNC.prepareHeaderWithNumRows(curRows);
  printHeader((const char*)globalInputNC.dataIncludingHeader);
  len = globalInputNC.getActualDataLen(curRows);
  for(int i = 0; i<len; i++) {
    out << globalInputNC.data[i] << " ";
    if((i+1) % (len/curRows) == 0) out << endl;
  }
  out << endl;

  out << "policyTargetsNCMove" << endl;
  policyTargetsNCMove.prepareHeaderWithNumRows(curRows);
  printHeader((const char*)policyTargetsNCMove.dataIncludingHeader);
  len = policyTargetsNCMove.getActualDataLen(curRows);
  for(int i = 0; i<len; i++) {
    out << policyTargetsNCMove.data[i] << " ";
    if((i+1) % (len/curRows) == 0) out << endl;
  }
  out << endl;

  out << "globalTargetsNC" << endl;
  globalTargetsNC.prepareHeaderWithNumRows(curRows);
  printHeader((const char*)globalTargetsNC.dataIncludingHeader);
  len = globalTargetsNC.getActualDataLen(curRows);
  for(int i = 0; i<len; i++) {
    out << globalTargetsNC.data[i] << " ";
    if((i+1) % (len/curRows) == 0) out << endl;
  }
  out << endl;

  out << "scoreDistrN" << endl;
  scoreDistrN.prepareHeaderWithNumRows(curRows);
  printHeader((const char*)scoreDistrN.dataIncludingHeader);
  len = scoreDistrN.getActualDataLen(curRows);
  for(int i = 0; i<len; i++) {
    out << (int)scoreDistrN.data[i] << " ";
    if((i+1) % (len/curRows) == 0) out << endl;
  }
  out << endl;

  out << "valueTargetsNCHW" << endl;
  valueTargetsNCHW.prepareHeaderWithNumRows(curRows);
  printHeader((const char*)valueTargetsNCHW.dataIncludingHeader);
  len = valueTargetsNCHW.getActualDataLen(curRows);
  for(int i = 0; i<len; i++) {
    out << (int)valueTargetsNCHW.data[i] << " ";
    if((i+1) % (len/curRows) == 0) out << endl;
  }
  out << endl;
}

//-------------------------------------------------------------------------------------

TrainingDataWriter::TrainingDataWriter(const string& outDir, int iVersion, int maxRowsPerFile, double firstFileMinRandProp, int dataXLen, int dataYLen, const string& randSeed, NNEvaluator* dEval)
  : TrainingDataWriter(outDir,NULL,iVersion,maxRowsPerFile,firstFileMinRandProp,dataXLen,dataYLen,1,randSeed,dEval)
{}
TrainingDataWriter::TrainingDataWriter(ostream* dbgOut, int iVersion, int maxRowsPerFile, double firstFileMinRandProp, int dataXLen, int dataYLen, int onlyEvery, const string& randSeed, NNEvaluator* dEval)
  : TrainingDataWriter(string(),dbgOut,iVersion,maxRowsPerFile,firstFileMinRandProp,dataXLen,dataYLen,onlyEvery,randSeed,dEval)
{}

TrainingDataWriter::TrainingDataWriter(const string& outDir, ostream* dbgOut, int iVersion, int maxRowsPerFile, double firstFileMinRandProp, int dataXLen, int dataYLen, int onlyEvery, const string& randSeed, NNEvaluator* dEval)
  :outputDir(outDir),inputsVersion(iVersion),rand(randSeed),writeBuffers(NULL),debugOut(dbgOut),debugOnlyWriteEvery(onlyEvery),rowCount(0),distillEval(dEval)
{
  int numBinaryChannels;
  int numGlobalChannels;
  //Note that this inputsVersion is for data writing, it might be different than the inputsVersion used
  //to feed into a model during selfplay
  static_assert(NNModelVersion::latestInputsVersionImplemented == 102, "");
  if(inputsVersion == 7) {
    numBinaryChannels = NNInputs::NUM_FEATURES_SPATIAL_V7;
    numGlobalChannels = NNInputs::NUM_FEATURES_GLOBAL_V7;
  }
  else if(inputsVersion == 101) {
    numBinaryChannels = NNInputs::NUM_FEATURES_SPATIAL_V101;
    numGlobalChannels = NNInputs::NUM_FEATURES_GLOBAL_V101;
  }
  else if(inputsVersion == 102) {
    numBinaryChannels = NNInputs::NUM_FEATURES_SPATIAL_V102;
    numGlobalChannels = NNInputs::NUM_FEATURES_GLOBAL_V102;
  }
  else {
    throw StringError("TrainingDataWriter: Unsupported inputs version: " + Global::intToString(inputsVersion));
  }

  writeBuffers = new TrainingWriteBuffers(inputsVersion, maxRowsPerFile, numBinaryChannels, numGlobalChannels, dataXLen, dataYLen);

  if(firstFileMinRandProp < 0 || firstFileMinRandProp > 1)
    throw StringError("TrainingDataWriter: firstFileMinRandProp not in [0,1]: " + Global::doubleToString(firstFileMinRandProp));
  isFirstFile = true;
  if(firstFileMinRandProp >= 1.0)
    firstFileMaxRows = maxRowsPerFile;
  else
    firstFileMaxRows = maxRowsPerFile - (int)(maxRowsPerFile * (1.0-firstFileMinRandProp) * rand.nextDouble());
}



TrainingDataWriter::~TrainingDataWriter()
{
  delete writeBuffers;
}

bool TrainingDataWriter::isEmpty() const {
  return writeBuffers->curRows <= 0;
}
int64_t TrainingDataWriter::numRowsInBuffer() const {
  return writeBuffers->curRows;
}

void TrainingDataWriter::writeAndClearIfFull() {
  if(writeBuffers->curRows >= writeBuffers->maxRows || (isFirstFile && writeBuffers->curRows >= firstFileMaxRows)) {
    flushIfNonempty();
  }
}



void TrainingDataWriter::flushIfNonempty() {
  string resultingFilename;
  flushIfNonempty(resultingFilename);
}

bool TrainingDataWriter::flushIfNonempty(string& resultingFilename) {
  if(writeBuffers->curRows <= 0)
    return false;

  isFirstFile = false;

  if(debugOut != NULL) {
    writeBuffers->writeToTextOstream(*debugOut);
    writeBuffers->clear();
    resultingFilename = "";
  }
  else {
    resultingFilename = outputDir + "/" + Global::uint64ToHexString(rand.nextUInt64()) + ".npz";
    string tmpFilename = resultingFilename + ".tmp";
    writeBuffers->writeToZipFile(tmpFilename);
    writeBuffers->clear();
    FileUtils::rename(tmpFilename,resultingFilename);
  }
  return true;
}

void TrainingDataWriter::writeGame(const FinishedGameData& data) {
  int numMoves = (int)(data.endHist.moveHistory.size() - data.startHist.moveHistory.size());
  assert(numMoves >= 0);
  assert(data.startHist.moveHistory.size() <= data.endHist.moveHistory.size());
  assert(data.endHist.moveHistory.size() <= 100000000);
  assert(data.targetWeightByTurn.size() == numMoves);
  assert(data.targetWeightByTurnUnrounded.size() == numMoves);
  assert(data.policyTargetsByTurn.size() == numMoves);
  assert(data.policySurpriseByTurn.size() == numMoves);
  assert(data.policyEntropyByTurn.size() == numMoves);
  assert(data.searchEntropyByTurn.size() == numMoves);
  assert(data.whiteValueTargetsByTurn.size() == numMoves+1);
  assert(data.nnRawStatsByTurn.size() == numMoves);

  //Some sanity checks
  #ifndef NDEBUG
  {
    const ValueTargets& lastTargets = data.whiteValueTargetsByTurn[data.whiteValueTargetsByTurn.size()-1];
    if(!data.endHist.isGameFinished)
      assert(data.hitTurnLimit);
    else if(data.endHist.isNoResult)
      assert(lastTargets.win == 0.0f && lastTargets.loss == 0.0f && lastTargets.noResult == 1.0f);
    else if(data.endHist.winner == P_BLACK)
      assert(lastTargets.win == 0.0f && lastTargets.loss == 1.0f && lastTargets.noResult == 0.0f);
    else if(data.endHist.winner == P_WHITE)
      assert(lastTargets.win == 1.0f && lastTargets.loss == 0.0f && lastTargets.noResult == 0.0f);
    else
      assert(lastTargets.noResult == 1.0f);

    assert(!data.endHist.isResignation);
  }
  #endif

  //Play out all the moves in a single pass first to compute all the future board states
  vector<Board> posHistForFutureBoards;
  {
    Board board(data.startBoard);
    BoardHistory hist(data.startHist);
    Player nextPlayer = data.startPla;
    posHistForFutureBoards.push_back(board);

    int startTurnIdx = (int)data.startHist.moveHistory.size();
    for(int turnAfterStart = 0; turnAfterStart<numMoves; turnAfterStart++) {
      int turnIdx = turnAfterStart + startTurnIdx;

      Move move = data.endHist.moveHistory[turnIdx];
      assert(move.pla == nextPlayer);
      assert(hist.isLegal(board,move.loc,move.pla));
      hist.makeBoardMoveAssumeLegal(board, move.loc, move.pla);
      nextPlayer = getOpp(nextPlayer);

      posHistForFutureBoards.push_back(board);
    }
  }

  Board board(data.startBoard);
  BoardHistory hist(data.startHist);
  Player nextPlayer = data.startPla;

  //Write main game rows
  int startTurnIdx = (int)data.startHist.moveHistory.size();
  for(int turnAfterStart = 0; turnAfterStart<numMoves; turnAfterStart++) {
    double targetWeight = data.targetWeightByTurn[turnAfterStart];
    int turnIdx = turnAfterStart + startTurnIdx;

    int64_t unreducedNumVisits = data.policyTargetsByTurn[turnAfterStart].unreducedNumVisits;
    const vector<PolicyTargetMove>* policyTarget0 = data.policyTargetsByTurn[turnAfterStart].policyTargets;
    const vector<PolicyTargetMove>* policyTarget1 = (turnAfterStart + 1 < numMoves) ? data.policyTargetsByTurn[turnAfterStart+1].policyTargets : NULL;
    bool isSidePosition = false;

    int numNeuralNetsBehindLatest = 0;
    for(int i = 0; i<data.changedNeuralNets.size(); i++) {
      if(data.changedNeuralNets[i]->turnIdx > turnIdx) {
        numNeuralNetsBehindLatest = (int)data.changedNeuralNets.size()-i;
        break;
      }
    }

    while(targetWeight > 0.0) {
      if(targetWeight >= 1.0 || rand.nextBool(targetWeight)) {
        if(debugOut == NULL || rowCount % debugOnlyWriteEvery == 0) {
          writeBuffers->addRow(
            board,hist,nextPlayer,
            turnIdx,
            1.0,
            unreducedNumVisits,
            policyTarget0,
            policyTarget1,
            data.policySurpriseByTurn[turnAfterStart],
            data.policyEntropyByTurn[turnAfterStart],
            data.searchEntropyByTurn[turnAfterStart],
            data.whiteValueTargetsByTurn,
            turnAfterStart,
            data.nnRawStatsByTurn[turnAfterStart],
            &(data.endHist.getRecentBoard(0)),
            &posHistForFutureBoards,
            isSidePosition,
            numNeuralNetsBehindLatest,
            data,
            rand,
            distillEval
          );
          writeAndClearIfFull();
        }
        rowCount++;
      }
      targetWeight -= 1.0;
    }


    Move move = data.endHist.moveHistory[turnIdx];
    assert(move.pla == nextPlayer);
    assert(hist.isLegal(board,move.loc,move.pla));
    hist.makeBoardMoveAssumeLegal(board, move.loc, move.pla);
    nextPlayer = getOpp(nextPlayer);
  }

  //Write side rows
  vector<ValueTargets> whiteValueTargetsBuf(1);
  for(int i = 0; i<data.sidePositions.size(); i++) {
    SidePosition* sp = data.sidePositions[i];

    double targetWeight = sp->targetWeight;
    while(targetWeight > 0.0) {
      if(targetWeight >= 1.0 || rand.nextBool(targetWeight)) {
        if(debugOut == NULL || rowCount % debugOnlyWriteEvery == 0) {

          int turnIdx = (int)sp->hist.moveHistory.size();
          assert(turnIdx >= data.startHist.moveHistory.size());
          whiteValueTargetsBuf[0] = sp->whiteValueTargets;
          bool isSidePosition = true;
          int numNeuralNetsBehindLatest = (int)data.changedNeuralNets.size() - sp->numNeuralNetChangesSoFar;

          writeBuffers->addRow(
            sp->board,sp->hist,sp->pla,
            turnIdx,
            1.0,
            sp->unreducedNumVisits,
            &(sp->policyTarget),
            NULL,
            sp->policySurprise,
            sp->policyEntropy,
            sp->searchEntropy,
            whiteValueTargetsBuf,
            0,
            sp->nnRawStats,
            NULL,
            NULL,
            isSidePosition,
            numNeuralNetsBehindLatest,
            data,
            rand,
            distillEval
          );
          writeAndClearIfFull();
        }
        rowCount++;
      }
      targetWeight -= 1.0;
    }

  }

}

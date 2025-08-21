#include "NNUEBoardHistory.h"
#include "../game/gamelogic.h"
#include "../neuralnet/nninputs.h"
#include <algorithm>
#include <cmath>
#include <cstring>

using namespace std;
using namespace NNUE;
using namespace NNUEV2;

NNUEBoardHistory::NNUEBoardHistory(const ModelWeight* weights, const MiscNNInputParams& nnInputParams, bool skipResultsBeforeNN)
    : BoardHistory(),
      blackEvaluator(weights),
      whiteEvaluator(weights),
      nnInputParams(nnInputParams),
      skipResultsBeforeNN(skipResultsBeforeNN),
      gfInputBuf(),
      illegalMapBuf(),
      historicalBoards(),
      moveCacheB(),
      moveCacheW(),
      moveCacheBlength(0),
      moveCacheWlength(0)
{
    // Initialize stoneTupleCountCache
    memset(stoneTupleCountCache, 0, sizeof(stoneTupleCountCache));
    //computeStoneTupleCountCache(board);
}

NNUEBoardHistory::NNUEBoardHistory(const Board& board, Player pla, const Rules& rules, const ModelWeight* weights, const MiscNNInputParams& nnInputParams, bool skipResultsBeforeNN)
    : BoardHistory(board, pla, rules),
      blackEvaluator(weights),
      whiteEvaluator(weights),
      nnInputParams(nnInputParams),
      skipResultsBeforeNN(skipResultsBeforeNN),
      gfInputBuf(),
      illegalMapBuf(),
      historicalBoards(),
      moveCacheB(),
      moveCacheW(),
      moveCacheBlength(0),
      moveCacheWlength(0)
{
    historicalBoards.push_back(board);
    syncNNUEWithBoard(board);
    // Initialize and compute stoneTupleCountCache
    memset(stoneTupleCountCache, 0, sizeof(stoneTupleCountCache));
    computeStoneTupleCountCache(board);
}

NNUEBoardHistory::NNUEBoardHistory(const NNUEBoardHistory& other)
    : BoardHistory(other),
      blackEvaluator(other.blackEvaluator),
      whiteEvaluator(other.whiteEvaluator),
      nnInputParams(other.nnInputParams),
      gfInputBuf(),
      illegalMapBuf(),
      historicalBoards(other.historicalBoards),
      moveCacheB(),
      moveCacheW(),
      moveCacheBlength(other.moveCacheBlength),
      moveCacheWlength(other.moveCacheWlength)
{
    std::copy(other.gfInputBuf, other.gfInputBuf + NNUEV2::globalFeatureNum, gfInputBuf);
    std::copy(other.illegalMapBuf, other.illegalMapBuf + MaxBS * MaxBS, illegalMapBuf);
    std::copy(other.moveCacheB, other.moveCacheB + MaxBS * MaxBS, moveCacheB);
    std::copy(other.moveCacheW, other.moveCacheW + MaxBS * MaxBS, moveCacheW);
    memcpy(stoneTupleCountCache, other.stoneTupleCountCache, sizeof(stoneTupleCountCache));
}

NNUEBoardHistory& NNUEBoardHistory::operator=(const NNUEBoardHistory& other)
{
    if (this == &other)
        return *this;
    
    BoardHistory::operator=(other);
    skipResultsBeforeNN = other.skipResultsBeforeNN;
    blackEvaluator = other.blackEvaluator;
    whiteEvaluator = other.whiteEvaluator;
    nnInputParams = other.nnInputParams;
    historicalBoards = other.historicalBoards;
    moveCacheBlength = other.moveCacheBlength;
    moveCacheWlength = other.moveCacheWlength;
    
    std::copy(other.gfInputBuf, other.gfInputBuf + NNUEV2::globalFeatureNum, gfInputBuf);
    std::copy(other.illegalMapBuf, other.illegalMapBuf + MaxBS * MaxBS, illegalMapBuf);
    std::copy(other.moveCacheB, other.moveCacheB + MaxBS * MaxBS, moveCacheB);
    std::copy(other.moveCacheW, other.moveCacheW + MaxBS * MaxBS, moveCacheW);
    memcpy(stoneTupleCountCache, other.stoneTupleCountCache, sizeof(stoneTupleCountCache));
    
    return *this;
}

NNUEBoardHistory::NNUEBoardHistory(NNUEBoardHistory&& other) noexcept
    : BoardHistory(std::move(other)),
      blackEvaluator(std::move(other.blackEvaluator)),
      whiteEvaluator(std::move(other.whiteEvaluator)),
      nnInputParams(other.nnInputParams),
      skipResultsBeforeNN(other.skipResultsBeforeNN),
      gfInputBuf(),
      illegalMapBuf(),
      historicalBoards(std::move(other.historicalBoards)),
      moveCacheB(),
      moveCacheW(),
      moveCacheBlength(other.moveCacheBlength),
      moveCacheWlength(other.moveCacheWlength)
{
    std::copy(other.gfInputBuf, other.gfInputBuf + NNUEV2::globalFeatureNum, gfInputBuf);
    std::copy(other.illegalMapBuf, other.illegalMapBuf + MaxBS * MaxBS, illegalMapBuf);
    std::copy(other.moveCacheB, other.moveCacheB + MaxBS * MaxBS, moveCacheB);
    std::copy(other.moveCacheW, other.moveCacheW + MaxBS * MaxBS, moveCacheW);
    memcpy(stoneTupleCountCache, other.stoneTupleCountCache, sizeof(stoneTupleCountCache));
}

NNUEBoardHistory& NNUEBoardHistory::operator=(NNUEBoardHistory&& other) noexcept
{
    if (this == &other)
        return *this;
    
    BoardHistory::operator=(std::move(other));
    blackEvaluator = std::move(other.blackEvaluator);
    whiteEvaluator = std::move(other.whiteEvaluator);
    skipResultsBeforeNN = other.skipResultsBeforeNN;
    nnInputParams = other.nnInputParams;
    historicalBoards = std::move(other.historicalBoards);
    moveCacheBlength = other.moveCacheBlength;
    moveCacheWlength = other.moveCacheWlength;
    
    std::copy(other.gfInputBuf, other.gfInputBuf + NNUEV2::globalFeatureNum, gfInputBuf);
    std::copy(other.illegalMapBuf, other.illegalMapBuf + MaxBS * MaxBS, illegalMapBuf);
    std::copy(other.moveCacheB, other.moveCacheB + MaxBS * MaxBS, moveCacheB);
    std::copy(other.moveCacheW, other.moveCacheW + MaxBS * MaxBS, moveCacheW);
    memcpy(stoneTupleCountCache, other.stoneTupleCountCache, sizeof(stoneTupleCountCache));
    
    return *this;
}

NNUEBoardHistory::~NNUEBoardHistory()
{
}

void NNUEBoardHistory::initializeEvaluators(const ModelWeight* weights)
{
    blackEvaluator = Eva_nnuev2(weights);
    whiteEvaluator = Eva_nnuev2(weights);
}

void NNUEBoardHistory::clear(const Board& board, Player pla, const Rules& rules)
{
    BoardHistory::clear(board, pla, rules);
    assert(board.x_size == MaxBS);
    assert(board.y_size == MaxBS);
    
    // Reset NNUE evaluators
    blackEvaluator.clear();
    whiteEvaluator.clear();
    
    // Reset move caches
    moveCacheBlength = 0;
    moveCacheWlength = 0;
    
    // Clear and initialize board history
    historicalBoards.clear();
    historicalBoards.push_back(board);
    
    // Sync NNUE state with the board
    syncNNUEWithBoard(board);
    
    // Initialize and compute stoneTupleCountCache
    memset(stoneTupleCountCache, 0, sizeof(stoneTupleCountCache));
    computeStoneTupleCountCache(board);
}


void NNUEBoardHistory::updateInputBuf(Color nextPlayer)
{
    if (historicalBoards.empty()) {
        ASSERT_UNREACHABLE;
    }
    const Board& currentBoard = historicalBoards.back();
    Player pla = nextPlayer;
    assert(pla == currentBoard.nextPla);
    Player opp = getOpp(pla);
    nnInputParams.resultsBeforeNN = GameLogic::ResultsBeforeNN();
    if (!skipResultsBeforeNN)
    {
      nnInputParams.resultsBeforeNN.init(currentBoard, *this, nextPlayer);
    }
    else
    {
    }

    if (!skipResultsBeforeNN && nnInputParams.resultsBeforeNN.myOnlyLoc != Board::NULL_LOC && nnInputParams.resultsBeforeNN.winner != C_WALL)
    {
      //no need to call NN
      return;
    }


    // Initialize input buffers
    std::fill(gfInputBuf, gfInputBuf + NNUEV2::globalFeatureNum, 0.0f);
    std::fill(illegalMapBuf, illegalMapBuf + MaxBS * MaxBS, false);
    float* gfInputBufp3 = gfInputBuf + 3;
    
    

    // Update illegal map based on current board state
    for (int i = 0; i < MaxBS * MaxBS; i++) {
        illegalMapBuf[i] = false;
    }
    
    // Fill global features exactly matching fillRowV101's rowGlobal assignments
    if (currentBoard.stage == 0) {
        // Priority value input
        if (currentBoard.numStones == 0) {
            gfInputBufp3[1] = 1.0f; // Priority value is always 0

        }
    } 
    else 
    {
        gfInputBufp3[0] = 1.0f; // Stage 1 indicator

        
        if (currentBoard.numStones == 0 || currentBoard.firstLoc == Board::NULL_LOC || currentBoard.firstLoc == Board::PASS_LOC) {
            gfInputBufp3[2] = 1.0f; // Everywhere is ok
            //fill illegal map
            for (int i = 0; i < MaxBS * MaxBS; i++)
            {
              illegalMapBuf[i] = true;
            }
        }
        
        if (currentBoard.firstLoc == Board::PASS_LOC) {
            gfInputBufp3[3] = 1.0f; // First move was pass

            //fill illegal map
            for(int i = 0; i < MaxBS * MaxBS; i++)
            {
              illegalMapBuf[i] = true;
            }
        }
        else {
            //all illegal second moves

          for(int y = 0; y < MaxBS; y++)
          {
            for(int x = 0; x < MaxBS; x++)
            {
              Loc loc = Location::getLoc(x, y, currentBoard.x_size);
              double priority = currentBoard.getLocationPriority(x, y);
              if(
                  currentBoard.colors[loc] == C_EMPTY && currentBoard.firstLoc != loc &&
                  priority + Board::PRIOR_EPS >= currentBoard.firstLocPriority)  // legal
              {
              }
              else
              {
                int nuloc=y*MaxBS+x;
                illegalMapBuf[nuloc]=true;
              }
            }
          }

        }
      
    }
    
    // Basic rule features
    if (rules.basicRule == Rules::BASICRULE_FREESTYLE) {
        // Freestyle rule specific features can be added here
    }
    else
      ASSERT_UNREACHABLE;
    
    if(!skipResultsBeforeNN) {
        if(nnInputParams.resultsBeforeNN.myOnlyLoc == Board::PASS_LOC)
          gfInputBufp3[38] = 1.0;

        if(nnInputParams.resultsBeforeNN.winner == nextPlayer)
          gfInputBufp3[11] = 1.0;  // can win by five/lifeFour/vcf

    }
    // Pass number features
    int myPassNum = nextPlayer == C_BLACK ? currentBoard.blackPassNum : currentBoard.whitePassNum;
    int oppPassNum = nextPlayer == C_WHITE ? currentBoard.blackPassNum : currentBoard.whitePassNum;
    
    // Win condition features (index 11)
    // Note: This would require GameLogic::ResultsBeforeNN analysis which is complex
    // For now, leaving as 0.0f
    
    // VCN and pass features (indices 12-14)
    if (!rules.firstPassWin && rules.VCNRule == Rules::VCNRULE_NOVC) {
        // Note: noResultUtilityForWhite would need to be passed as parameter
        // For now using 0.0f as placeholder
        gfInputBufp3[12] = nextPlayer == P_BLACK ? -nnInputParams.noResultUtilityForWhite : nnInputParams.noResultUtilityForWhite;
        gfInputBufp3[13] = myPassNum > 0 ? 1.0f : 0.0f;
        gfInputBufp3[14] = oppPassNum > 0 ? 1.0f : 0.0f;
    } else {
        gfInputBufp3[12] = 0.0f;
        gfInputBufp3[13] = 0.0f;
        gfInputBufp3[14] = 0.0f;

    }

     if(nnInputParams.playoutDoublingAdvantage != 0) {
        gfInputBufp3[15] = 1.0;
        gfInputBufp3[16] = (float)(0.5 * nnInputParams.playoutDoublingAdvantage);
    }
    
    // First pass win features (indices 17-19)
    if (rules.firstPassWin) {
        gfInputBufp3[17] = 1.0f;
        gfInputBufp3[18] = myPassNum > 0 ? 1.0f : 0.0f;
        gfInputBufp3[19] = oppPassNum > 0 ? 1.0f : 0.0f;
    }
    
    // VCN rule features (indices 20-29)
    if (rules.VCNRule != Rules::VCNRULE_NOVC) {
        Color VCside = rules.vcSide();
        int VClevel = rules.vcLevel();
        int realVClevel = VClevel + myPassNum + oppPassNum;
        if (VClevel == 4)realVClevel = 4;//vc5/6 are handled outside
        if (realVClevel == 6)
            realVClevel = 5; // vc6 is the same as vc5
        if (realVClevel >= 1 && realVClevel <= 5) {
            if (VCside == nextPlayer)
                gfInputBufp3[19 + realVClevel] = 1.0f;
            else if (VCside == opp)
                gfInputBufp3[24 + realVClevel] = 1.0f;
        }
    }
    
    // Max moves features (indices 30-37)
    if (rules.maxMoves != 0) {
        gfInputBufp3[30] = 1.0f;

        double boardArea = currentBoard.x_size * currentBoard.y_size;
        double movenum = currentBoard.movenum;
        int maxmovesInt = currentBoard.calculateRealMaxmove(rules.maxMoves);
        double maxmoves = maxmovesInt;
        gfInputBufp3[31] = static_cast<float>(maxmoves / boardArea);
        gfInputBufp3[32] = static_cast<float>(movenum / boardArea);
        gfInputBufp3[33] = static_cast<float>(exp(-(maxmoves - movenum) / 70.0));
        gfInputBufp3[34] = static_cast<float>(exp(-(maxmoves - movenum) / 20.0));
        gfInputBufp3[35] = static_cast<float>(exp(-(maxmoves - movenum) / 7.0));
        gfInputBufp3[36] = static_cast<float>(exp(-(maxmoves - movenum) / 2.0));
        int remainFullMoves = maxmoves - movenum + currentBoard.stage;
        remainFullMoves /= 2;
        gfInputBufp3[37] = static_cast<float>(2 * (remainFullMoves % 2) - 1); // final move is pla or opp
    }
    
    // Win by pass feature (index 38)
    // Note: This would require GameLogic::ResultsBeforeNN analysis
    // For now leaving as 0.0f
    
    // Board area input features (indices 39-41) - matching PyTorch boardAreaInput definition
    float boardArea = static_cast<float>(currentBoard.x_size * currentBoard.y_size);
    float boardHs = static_cast<float>(currentBoard.y_size); // board height
    float boardWs = static_cast<float>(currentBoard.x_size);  // board width
    
    gfInputBuf[0] = boardArea / 225.0f - 1.0f;  // boardArea/225-1
    gfInputBuf[1] = sqrt(boardArea / 225.0f) - 1.0f;  // sqrt(boardArea/225)-1
    gfInputBuf[2] = (boardHs - boardWs) * (boardHs - boardWs) / boardArea;  // (boardHs-boardWs)^2/boardArea
}

NNUE::ValueType NNUEBoardHistory::evaluateFull(Color color, NNUE::PolicyType* policy)
{
  updateInputBuf(color);
  bool noPolicy = policy == nullptr;
  if (!skipResultsBeforeNN && nnInputParams.resultsBeforeNN.myOnlyLoc != Board::NULL_LOC && nnInputParams.resultsBeforeNN.winner != C_WALL)
  {
    //no need to call NN
    // Set all policies to MIN_POLICY
    if (!noPolicy)
    {
      for (int i = 0; i < MaxBS * MaxBS + 1; i++) {
        policy[i] = MIN_POLICY;
      }
    
      // Set myOnlyLoc policy to MYFOUR_POLICY
      Loc myOnlyLoc = nnInputParams.resultsBeforeNN.myOnlyLoc;
      if (myOnlyLoc == Board::PASS_LOC) {
        policy[MaxBS * MaxBS] = MYFOUR_POLICY;
      } else {
        int x = Location::getX(myOnlyLoc, MaxBS);
        int y = Location::getY(myOnlyLoc, MaxBS);
        int nu_loc = y * MaxBS + x;
        policy[nu_loc] = MYFOUR_POLICY;
      }
    }
    
    // Return a default value (can be adjusted based on requirements)
    NNUE::ValueType result;
    if(nnInputParams.resultsBeforeNN.winner==color)
    {
        result.win = 1.0f;
        result.loss = 0.0f;
        result.draw = 0.0f;
    }
    else if(nnInputParams.resultsBeforeNN.winner==getOpp(color))
    {
        result.win = 0.0f;
        result.loss = 1.0f;
        result.draw = 0.0f;
    }
    else if(nnInputParams.resultsBeforeNN.winner==C_EMPTY)
    {
        result.win = 0.0f;
        result.loss = 0.0f;
        result.draw = 1.0f;
    }
    return result;
  }
  else
  {
    clearCache(color);
    if (noPolicy)
    {
      if (color == C_BLACK)
        return blackEvaluator.evaluateValue(gfInputBuf, illegalMapBuf);
      else
        return whiteEvaluator.evaluateValue(gfInputBuf, illegalMapBuf);
    }
    else
    {
      if (color == C_BLACK)
        return blackEvaluator.evaluateFull(gfInputBuf, illegalMapBuf, policy);
      else
        return whiteEvaluator.evaluateFull(gfInputBuf, illegalMapBuf, policy);
    }
  }
}

//void NNUEBoardHistory::evaluatePolicy(Color color, NNUE::PolicyType* policy)
//{
//    updateInputBuf(color);
//    clearCache(color);
//    if (color == C_BLACK)
//        blackEvaluator.evaluatePolicy(gfInputBuf, illegalMapBuf, policy);
//    else
//        whiteEvaluator.evaluatePolicy(gfInputBuf, illegalMapBuf, policy);
//}

//NNUE::ValueType NNUEBoardHistory::evaluateValue(Color color)
//{
//    updateInputBuf(color);
//    clearCache(color);
//    if (color == C_BLACK)
//        return blackEvaluator.evaluateValue(gfInputBuf, illegalMapBuf);
//    else
//        return whiteEvaluator.evaluateValue(gfInputBuf, illegalMapBuf);
//}

void NNUEBoardHistory::addCache(bool isUndo, Color color, Loc loc)
{
    if (loc == Board::PASS_LOC || loc == Board::NULL_LOC)
        return;

    int x = Location::getX(loc, MaxBS);
    int y = Location::getY(loc, MaxBS);
    Loc pos = y * MaxBS + x;

    MoveCache newcache(isUndo, color, pos);
    
    if (moveCacheBlength == 0 || !isContraryMove(moveCacheB[moveCacheBlength-1], newcache)) {
        moveCacheB[moveCacheBlength] = newcache;
        moveCacheBlength++;
    } else {
        // Cancel out the previous move
        moveCacheBlength--;
    }
    
    if (moveCacheWlength == 0 || !isContraryMove(moveCacheW[moveCacheWlength-1], newcache)) {
        moveCacheW[moveCacheWlength] = newcache;
        moveCacheWlength++;
    } else {
        // Cancel out the previous move
        moveCacheWlength--;
    }

}

// NNUEBoard replacement methods
void NNUEBoardHistory::play(Color color, Loc loc)
{
    if (color != presumedNextMovePla)
        throw "wrong next player";
    
    // Save current board state to history
    if (!historicalBoards.empty()) {
      historicalBoards.push_back(historicalBoards.back());
    }
    else
      ASSERT_UNREACHABLE;

    // Add to cache for NNUE evaluator
    addCache(false, color, loc);
    
    // Make the move on the current board
    Board& currentBoard = historicalBoards.back();
    //currentBoard.playMoveAssumeLegal(loc, color);
    
    // Update BoardHistory state
    BoardHistory::makeBoardMoveAssumeLegal(currentBoard, loc, color);
    
    // Update stoneTupleCountCache incrementally
    updateStoneTupleCountCache(loc, color, false);

    //assert(checkStoneTupleCountCacheConsistency(currentBoard));

    //assert(checkEvaluatorBoardConsistency());
}

void NNUEBoardHistory::undo()
{
    
    // Remove the last board state from history
    if (historicalBoards.size() > 1) {
        historicalBoards.pop_back();
    }
    else {
      ASSERT_UNREACHABLE;
    }

    // Update BoardHistory state for undo
    if (!moveHistory.empty()) {
        // Get the move to undo
        Move lastMove = moveHistory[moveHistory.size()-1];
        
        // Update stoneTupleCountCache incrementally for undo
        updateStoneTupleCountCache(lastMove.loc, lastMove.pla, true);
        
        // Add undo to cache
        addCache(true, historicalBoards.back().nextPla, lastMove.loc);
        moveHistory.pop_back();
        //assert(checkEvaluatorBoardConsistency());
    }
    else {
      ASSERT_UNREACHABLE;
    }

    assert(!historicalBoards.empty());
    //recentBoards[currentRecentBoardIdx] = historicalBoards[historicalBoards.size() -NUM_RECENT_BOARDS+1];
    //this will not be used, save time
    currentRecentBoardIdx = (currentRecentBoardIdx - 1 + NUM_RECENT_BOARDS) % NUM_RECENT_BOARDS;
    
    presumedNextMovePla = historicalBoards.back().nextPla;
    
    isGameFinished = false;
    winner = C_EMPTY;
    isNoResult = false;
    isResignation = false;
}

const Board& NNUEBoardHistory::getBoard() const
{
    return historicalBoards.back();
}

void NNUEBoardHistory::clearCache(Color color)
{
    if (color == C_BLACK) {
        for (int i = 0; i < moveCacheBlength; i++) {
            MoveCache move = moveCacheB[i];
            if (move.isUndo)
                blackEvaluator.undo(move.loc);
            else
                blackEvaluator.play(move.color, move.loc);
        }
        moveCacheBlength = 0;
    } else if (color == C_WHITE) {
        for (int i = 0; i < moveCacheWlength; i++) {
            MoveCache move = moveCacheW[i];
            if (move.isUndo)
                whiteEvaluator.undo(move.loc);
            else
                whiteEvaluator.play(getOpp(move.color), move.loc);
        }
        moveCacheWlength = 0;
    }
}

bool NNUEBoardHistory::isContraryMove(MoveCache a, MoveCache b)
{
    if (a.isUndo == b.isUndo)
        return false;
    else {
      if (a.loc != b.loc)
      {
        if (b.isUndo)
          std::cout << "NNUEBoardHistory::isContraryMove strange bugs";
        return false;
      }
      if (a.color != b.color)
          std::cout << "NNUEBoardHistory::isContraryMove strange bugs";
      return true;
    }
}

void NNUEBoardHistory::syncNNUEWithBoard(const Board& board)
{
    // Use Eva_nnuev2's syncWithBoard method
    // blackEvaluator uses normal colors
    blackEvaluator.syncWithBoard(board, false);
    
    // whiteEvaluator uses inverted colors (all colors are reversed)
    whiteEvaluator.syncWithBoard(board, true);
}

bool NNUEBoardHistory::checkEvaluatorBoardConsistency()
{
  clearCache(C_BLACK);
  clearCache(C_WHITE);
  const Board& currentBoard = historicalBoards.back();
    
  // Check blackEvaluator consistency
  for (int y = 0; y < currentBoard.y_size; y++) {
    for (int x = 0; x < currentBoard.x_size; x++) {
      Loc loc = Location::getLoc(x, y, currentBoard.x_size);
      NU_Loc nu_loc = y * MaxBS + x;
            
      Color historyColor = currentBoard.colors[loc];
      if (currentBoard.stage == 1 && currentBoard.firstLoc == loc)
      {
        assert(historyColor == C_EMPTY);
        historyColor = currentBoard.nextPla;
      }
      Color blackEvalColor = blackEvaluator.board[nu_loc];
            
      // For blackEvaluator, colors should match exactly
      if (historyColor != blackEvalColor) {
          return false;
      }
            
      // For whiteEvaluator, colors should be inverted
      Color whiteEvalColor = whiteEvaluator.board[nu_loc];
      Color expectedWhiteColor = historyColor;
      if (historyColor == C_BLACK) {
          expectedWhiteColor = C_WHITE;
      } else if (historyColor == C_WHITE) {
          expectedWhiteColor = C_BLACK;
      }
            
      if (expectedWhiteColor != whiteEvalColor) {
          return false;
      }
    }
  }
  
  // Check stoneTupleCountCache consistency
  if (!checkStoneTupleCountCacheConsistency(currentBoard)) {
    return false;
  }
    
  return true;
}

void NNUEBoardHistory::computeStoneTupleCountCache(const Board& board)
{
    // Clear the cache
    memset(stoneTupleCountCache, 0, sizeof(stoneTupleCountCache));
    
    // Direction offsets: [0] +x, [1] +y, [2] +x+y, [3] -x+y
    int16_t directions[4] = {1, board.x_size + 1, board.x_size + 1 + 1, board.x_size + 1 - 1};
    
    for (int dir = 0; dir < 4; dir++) {
        int16_t adj = directions[dir];
        
        // Iterate through all possible 6-tuples in this direction
        for (int y = 0; y < board.y_size; y++) {
            for (int x = 0; x < board.x_size; x++) {
                Loc startLoc = Location::getLoc(x, y, board.x_size);
                
                // Check if this 6-tuple is valid (all positions on board)
                bool validTuple = true;
                for (int i = 0; i < 6; i++) {
                    Loc loc = startLoc + i * adj;
                    if (!board.isOnBoard(loc)) {
                        validTuple = false;
                        break;
                    }
                }
                
                if (!validTuple) {
                    // Mark as out of board
                    stoneTupleCountCache[dir][startLoc] |= 0x80;
                    continue;
                }
                
                // Count black and white stones in this 6-tuple
                int blackCount = 0;
                int whiteCount = 0;
                
                for (int i = 0; i < 6; i++) {
                    Loc loc = startLoc + i * adj;
                    Color c = board.colors[loc];
                    
                    // Handle stage 1 special case
                    if (board.stage == 1 && loc == board.firstLoc) {
                        c = board.nextPla;
                    }
                    
                    if (c == C_BLACK) {
                        blackCount++;
                    } else if (c == C_WHITE) {
                        whiteCount++;
                    }
                }
                
                // Store counts: bits 0-2 for black, bits 3-5 for white
                stoneTupleCountCache[dir][startLoc] = (blackCount & 0x07) | ((whiteCount & 0x07) << 3);
             }
         }
     }
 }

void NNUEBoardHistory::updateStoneTupleCountCache(Loc loc, Color color, bool isUndo)
{
    if (loc == Board::PASS_LOC || loc == Board::NULL_LOC) {
        return;
    }
    
    const Board& board = historicalBoards.back();
    
    // Direction offsets: [0] +x, [1] +y, [2] +x+y, [3] -x+y
    int16_t directions[4] = {1, board.x_size + 1, board.x_size + 1 + 1, board.x_size + 1 - 1};
    
    for (int dir = 0; dir < 4; dir++) {
        int16_t adj = directions[dir];
        
        // Update all 6-tuples that contain this location
        for (int offset = 0; offset < 6; offset++) {
            Loc tupleStart = loc - offset * adj;
            
            // Check if this tuple start is valid
            if (!board.isOnBoard(tupleStart)) {
                break;
            }
            
            // Check if the entire 6-tuple is on board
            //bool validTuple = true;
            //for (int i = 0; i < 6; i++) {
            //    Loc tupleLoc = tupleStart + i * adj;
            //    if (!board.isOnBoard(tupleLoc)) {
            //        validTuple = false;
            //        break;
            //    }
            //}
            
            //if (!validTuple) {
            //    assert(stoneTupleCountCache[dir][tupleStart] & 0x80);
            //    continue;
            //}
            
            // Extract current counts
            uint8_t& cache = stoneTupleCountCache[dir][tupleStart];
            if (cache & 0x80) { // Skip if marked as out of board
                continue;
            }

            int8_t delta=color==C_BLACK?1:8;
            if (isUndo) {
                delta = -delta;
            }

            
            // Store updated counts
            cache += delta;

         }
     }
 }

bool NNUEBoardHistory::checkStoneTupleCountCacheConsistency(const Board& board)
{
    // Direction offsets: [0] +x, [1] +y, [2] +x+y, [3] -x+y
    int16_t directions[4] = {1, board.x_size + 1, board.x_size + 1 + 1, board.x_size + 1 - 1};
    
    for (int dir = 0; dir < 4; dir++) {
        int16_t adj = directions[dir];
        
        // Check all possible 6-tuples in this direction
        for (int y = 0; y < board.y_size; y++) {
            for (int x = 0; x < board.x_size; x++) {
                Loc startLoc = Location::getLoc(x, y, board.x_size);
                
                // Check if this 6-tuple is valid (all positions on board)
                bool validTuple = true;
                for (int i = 0; i < 6; i++) {
                    Loc loc = startLoc + i * adj;
                    if (!board.isOnBoard(loc)) {
                        validTuple = false;
                        break;
                    }
                }
                
                uint8_t cachedValue = stoneTupleCountCache[dir][startLoc];
                
                if (!validTuple) {
                    // Should be marked as out of board
                    if (!(cachedValue & 0x80)) {
                        return false;
                    }
                    continue;
                }
                
                // Should not be marked as out of board
                if (cachedValue & 0x80) {
                    return false;
                }
                
                // Count actual black and white stones in this 6-tuple
                int actualBlackCount = 0;
                int actualWhiteCount = 0;
                
                for (int i = 0; i < 6; i++) {
                    Loc loc = startLoc + i * adj;
                    Color c = board.colors[loc];
                    
                    // Handle stage 1 special case
                    if (board.stage == 1 && loc == board.firstLoc) {
                        c = board.nextPla;
                    }
                    
                    if (c == C_BLACK) {
                        actualBlackCount++;
                    } else if (c == C_WHITE) {
                        actualWhiteCount++;
                    }
                }
                
                // Extract cached counts
                int cachedBlackCount = cachedValue & 0x07;
                int cachedWhiteCount = (cachedValue >> 3) & 0x07;
                
                // Compare with actual counts
                if (cachedBlackCount != actualBlackCount || cachedWhiteCount != actualWhiteCount) {
                    return false;
                }
            }
        }
    }
    
    return true;
}

#include "../game/gamelogic.h"

/*
 * gamelogic.cpp
 * Logics of game rules
 * Some other game logics are in board.h/cpp
 *
 * Gomoku as a representive
 */

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

using namespace std;

namespace {

static const int DIRECTIONS[][3] = {
  {1,0,0},
  {0,1,0},
  {0,0,1},
  {1,1,0},
  {1,-1,0},
  {1,0,1},
  {1,0,-1},
  {0,1,1},
  {0,1,-1},
  {1,1,1},
  {1,1,-1},
  {1,-1,1},
  {1,-1,-1},
};

static const int AXIAL_DIRECTIONS[][3] = {
  {1,0,0},
  {0,1,0},
  {0,0,1},
};

struct LineInfo {
  int total;
  bool openNeg;
  bool openPos;
};

static void countDirection(
  const Board& board,
  Loc loc,
  int x,
  int y,
  int z,
  Player pla,
  int dx,
  int dy,
  int dz,
  int& count,
  bool& openEnd
) {
  int xyStride = board.x_size * board.y_size;
  int locDelta = dx + dy * board.x_size + dz * xyStride;
  count = 0;
  int x1 = x + dx;
  int y1 = y + dy;
  int z1 = z + dz;
  Loc loc1 = loc + locDelta;
  while(x1 >= 0 && x1 < board.x_size && y1 >= 0 && y1 < board.y_size && z1 >= 0 && z1 < board.z_size) {
    if(board.colors[loc1] != pla)
      break;
    count += 1;
    x1 += dx;
    y1 += dy;
    z1 += dz;
    loc1 += locDelta;
  }
  openEnd =
    x1 >= 0 && x1 < board.x_size &&
    y1 >= 0 && y1 < board.y_size &&
    z1 >= 0 && z1 < board.z_size &&
    board.colors[loc1] == C_EMPTY;
}

static LineInfo getLineInfo(const Board& board, Loc loc, int x, int y, int z, Player pla, int dx, int dy, int dz) {
  int negCount;
  int posCount;
  bool openNeg;
  bool openPos;
  countDirection(board, loc, x, y, z, pla, -dx, -dy, -dz, negCount, openNeg);
  countDirection(board, loc, x, y, z, pla, dx, dy, dz, posCount, openPos);
  LineInfo info;
  info.total = negCount + posCount + 1;
  info.openNeg = openNeg;
  info.openPos = openPos;
  return info;
}

static int getWinningLineLength(const Rules& rules) {
  if(rules.basicRule == Rules::BASICRULE_CON7)
    return 7;
  if(rules.basicRule == Rules::BASICRULE_DCON5)
    return 5;
  return 6;
}

static bool usesAxialDirectionsOnly(const Rules& rules) {
  return rules.basicRule == Rules::BASICRULE_DCON5;
}

static bool isWinningLine(const LineInfo& info, const Rules& rules) {
  int winningLineLength = getWinningLineLength(rules);
  if(rules.basicRule == Rules::BASICRULE_STANDARD)
    return info.total == winningLineLength;
  return info.total >= winningLineLength;
}

struct MovePatternInfo {
  bool isWinning = false;
  bool isLiveFour = false;
};

static MovePatternInfo analyzeMovePatterns(const Board& board, const Rules& rules, Player pla, Loc loc, int x, int y, int z) {
  MovePatternInfo result;
  if(loc == Board::PASS_LOC || !board.isOnBoard(loc))
    return result;
  if(board.colors[loc] != C_EMPTY && board.colors[loc] != pla)
    return result;

  const int (*directions)[3] = usesAxialDirectionsOnly(rules) ? AXIAL_DIRECTIONS : DIRECTIONS;
  int numDirections = usesAxialDirectionsOnly(rules) ? 3 : 13;
  int liveFourLength = getWinningLineLength(rules) - 1;
  for(int i = 0; i < numDirections; i++) {
    const auto& direction = directions[i];
    LineInfo info = getLineInfo(board, loc, x, y, z, pla, direction[0], direction[1], direction[2]);
    if(!result.isWinning && isWinningLine(info, rules))
      result.isWinning = true;
    if(!result.isLiveFour && info.total == liveFourLength && info.openNeg && info.openPos)
      result.isLiveFour = true;
    if(result.isWinning && result.isLiveFour)
      break;
  }
  return result;
}

static bool isWinningMove(const Board& board, const Rules& rules, Player pla, Loc loc) {
  int x = Location::getX(loc, board.x_size);
  int y = Location::getY(loc, board.x_size, board.y_size);
  int z = Location::getZ(loc, board.x_size, board.y_size);
  return analyzeMovePatterns(board, rules, pla, loc, x, y, z).isWinning;
}

static bool isLiveFourMove(const Board& board, const Rules& rules, Player pla, Loc loc) {
  if(loc == Board::PASS_LOC || !board.isOnBoard(loc) || board.colors[loc] != C_EMPTY)
    return false;
  int x = Location::getX(loc, board.x_size);
  int y = Location::getY(loc, board.x_size, board.y_size);
  int z = Location::getZ(loc, board.x_size, board.y_size);
  return analyzeMovePatterns(board, rules, pla, loc, x, y, z).isLiveFour;
}

}


bool GameLogic::isLegal(const Board& board, Player pla, Loc loc) {
  if(pla != board.nextPla) {
    std::cout << "Error next player ";
    return false;
  }

  if(loc != Board::PASS_LOC && (!board.isOnBoard(loc)))
    return false;

  if(board.stage == 0)  // choose or copy a piece
  {
    if(loc == Board::PASS_LOC)
      return true;
    return board.colors[loc] == C_EMPTY;
  } 
  else if(board.stage == 1)  // place the piece
  {
    return false;
  }
  ASSERT_UNREACHABLE;
  return false;
}

GameLogic::MovePriority GameLogic::getMovePriorityAssumeLegal(const Board& board, const BoardHistory& hist, Player pla, Loc loc) {
  return MP_NORMAL;
}

GameLogic::MovePriority GameLogic::getMovePriority(const Board& board, const BoardHistory& hist, Player pla, Loc loc) {
  if(loc == Board::PASS_LOC)
    return MP_NORMAL;
  if(!board.isLegal(loc, pla))
    return MP_ILLEGAL;
  MovePriority MP = getMovePriorityAssumeLegal(board, hist, pla, loc);
  return MP;
}

bool GameLogic::hasLegalMoveAssumeStage0(const Board& board) {
  if(board.stage != 0)
    return false;
  for(int loc = 0; loc < board.boardVolume(); loc++) {
    if(board.colors[loc] == C_EMPTY)
      return true;
  }
  return false;
}

bool GameLogic::hasLegalMoveAssumeStage1(const Board& board, Loc chosenLoc) {
  (void)board;
  (void)chosenLoc;
  return false;
}




Color GameLogic::checkWinnerAfterPlayed(
  const Board& board,
  const BoardHistory& hist,
  Player pla,
  Loc loc,
  bool isLegalPass) {

  Player opp = getOpp(pla);
  if(isLegalPass) {
    assert(loc == Board::PASS_LOC);
    assert(board.stage == 0);
    if(hasLegalMoveAssumeStage0(board))
      return opp;
    return C_EMPTY;
  } else if(loc == Board::PASS_LOC)
    return getOpp(pla);//illegal pass
  
  if(isWinningMove(board, hist.rules, pla, loc))
    return pla;
  int playedMoves = hist.initialTurnNumber + (int)hist.moveHistory.size();
  if(hist.rules.maxMoves != 0 && playedMoves >= hist.rules.maxMoves)
    return C_EMPTY;
  if(!hasLegalMoveAssumeStage0(board))
    return C_EMPTY;
  return C_WALL;
}

GameLogic::ResultsBeforeNN::ResultsBeforeNN() {
  inited = false;
  winner = C_WALL;
  myOnlyLoc = Board::NULL_LOC;
}

void GameLogic::ResultsBeforeNN::init(const Board& board, const BoardHistory& hist, Color nextPlayer) {
  if(inited)
    return;
  inited = true;

  if(board.stage != 0)
    return;

  Color opp = getOpp(nextPlayer);
  vector<Loc> winningMoves;
  vector<Loc> blockFourMoves;
  vector<Loc> liveFourMoves;
  for(int loc = 0; loc < board.boardVolume(); loc++) {
    if(board.colors[loc] != C_EMPTY)
      continue;

    int x = Location::getX((Loc)loc, board.x_size);
    int y = Location::getY((Loc)loc, board.x_size, board.y_size);
    int z = Location::getZ((Loc)loc, board.x_size, board.y_size);
    MovePatternInfo myPatterns = analyzeMovePatterns(board, hist.rules, nextPlayer, (Loc)loc, x, y, z);
    if(myPatterns.isWinning)
      winningMoves.push_back((Loc)loc);

    MovePatternInfo oppPatterns = analyzeMovePatterns(board, hist.rules, opp, (Loc)loc, x, y, z);
    if(oppPatterns.isWinning)
      blockFourMoves.push_back((Loc)loc);
    if(myPatterns.isLiveFour)
      liveFourMoves.push_back((Loc)loc);
  }

  if(!winningMoves.empty()) {
    winner = nextPlayer;
    myOnlyLoc = winningMoves[0];
    return;
  }

  if(!blockFourMoves.empty()) {
    myOnlyLoc = blockFourMoves[0];
    return;
  }

  if(!liveFourMoves.empty()) {
    myOnlyLoc = liveFourMoves[0];
    int remainMoves = hist.rules.maxMoves == 0 ? 1000000000 : hist.rules.maxMoves - hist.initialTurnNumber - (int)hist.moveHistory.size();
    if(remainMoves >= 3)
      winner = nextPlayer;
  }
}

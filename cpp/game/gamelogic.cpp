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

struct LineInfo {
  int total;
  bool openNeg;
  bool openPos;
};

static void countDirection(
  const Board& board,
  int x,
  int y,
  int z,
  Player pla,
  int dx,
  int dy,
  int dz,
  int sign,
  int& count,
  bool& openEnd
) {
  count = 0;
  int x1 = x + dx * sign;
  int y1 = y + dy * sign;
  int z1 = z + dz * sign;
  while(x1 >= 0 && x1 < board.x_size && y1 >= 0 && y1 < board.y_size && z1 >= 0 && z1 < board.z_size) {
    Loc loc1 = Location::getLoc(x1,y1,z1,board.x_size,board.y_size);
    if(board.colors[loc1] != pla)
      break;
    count += 1;
    x1 += dx * sign;
    y1 += dy * sign;
    z1 += dz * sign;
  }
  openEnd =
    x1 >= 0 && x1 < board.x_size &&
    y1 >= 0 && y1 < board.y_size &&
    z1 >= 0 && z1 < board.z_size &&
    board.colors[Location::getLoc(x1,y1,z1,board.x_size,board.y_size)] == C_EMPTY;
}

static LineInfo getLineInfo(const Board& board, Loc loc, Player pla, int dx, int dy, int dz) {
  int x = Location::getX(loc, board.x_size);
  int y = Location::getY(loc, board.x_size, board.y_size);
  int z = Location::getZ(loc, board.x_size, board.y_size);
  int negCount;
  int posCount;
  bool openNeg;
  bool openPos;
  countDirection(board, x, y, z, pla, dx, dy, dz, -1, negCount, openNeg);
  countDirection(board, x, y, z, pla, dx, dy, dz, 1, posCount, openPos);
  LineInfo info;
  info.total = negCount + posCount + 1;
  info.openNeg = openNeg;
  info.openPos = openPos;
  return info;
}

static bool isWinningLine(const LineInfo& info, const Rules& rules) {
  if(rules.basicRule == Rules::BASICRULE_STANDARD)
    return info.total == 6;
  return info.total >= 6;
}

static bool isWinningMove(const Board& board, const Rules& rules, Player pla, Loc loc) {
  if(loc == Board::PASS_LOC || !board.isOnBoard(loc))
    return false;
  if(board.colors[loc] != C_EMPTY && board.colors[loc] != pla)
    return false;
  for(const auto& direction: DIRECTIONS) {
    if(isWinningLine(getLineInfo(board, loc, pla, direction[0], direction[1], direction[2]), rules))
      return true;
  }
  return false;
}

static bool isLiveFourMove(const Board& board, const Rules& rules, Player pla, Loc loc) {
  if(loc == Board::PASS_LOC || !board.isOnBoard(loc) || board.colors[loc] != C_EMPTY)
    return false;
  for(const auto& direction: DIRECTIONS) {
    LineInfo info = getLineInfo(board, loc, pla, direction[0], direction[1], direction[2]);
    if(info.total == 5 && info.openNeg && info.openPos)
      return true;
    if(rules.basicRule == Rules::BASICRULE_STANDARD)
      continue;
  }
  return false;
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

    if(isWinningMove(board, hist.rules, nextPlayer, loc))
      winningMoves.push_back((Loc)loc);
    if(isWinningMove(board, hist.rules, opp, loc))
      blockFourMoves.push_back((Loc)loc);
    if(isLiveFourMove(board, hist.rules, nextPlayer, loc))
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

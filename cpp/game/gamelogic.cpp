#include "../game/gamelogic.h"

#include <cassert>
#include <iostream>

using namespace std;

namespace {

void nextLocOfCircle(int boardSize, int& x, int& y, int& dx, int& dy, bool& traversedLoop) {
  x += dx;
  y += dy;
  if(x < 0 || y < 0 || x >= boardSize || y >= boardSize)
    traversedLoop = true;

  if(x < 0) {
    assert(dx == -1 && dy == 0);
    if(y < boardSize / 2) {
      x = y;
      y = 0;
      dx = 0;
      dy = 1;
    }
    else {
      x = boardSize - y - 1;
      y = boardSize - 1;
      dx = 0;
      dy = -1;
    }
  }
  else if(y < 0) {
    assert(dx == 0 && dy == -1);
    if(x < boardSize / 2) {
      y = x;
      x = 0;
      dx = 1;
      dy = 0;
    }
    else {
      y = boardSize - x - 1;
      x = boardSize - 1;
      dx = -1;
      dy = 0;
    }
  }
  else if(x >= boardSize) {
    assert(dx == 1 && dy == 0);
    if(y < boardSize / 2) {
      x = boardSize - y - 1;
      y = 0;
      dx = 0;
      dy = 1;
    }
    else {
      x = y;
      y = boardSize - 1;
      dx = 0;
      dy = -1;
    }
  }
  else if(y >= boardSize) {
    assert(dx == 0 && dy == 1);
    if(x < boardSize / 2) {
      y = boardSize - x - 1;
      x = 0;
      dx = 1;
      dy = 0;
    }
    else {
      y = x;
      x = boardSize - 1;
      dx = -1;
      dy = 0;
    }
  }
}

Loc findCaptureLocOneDirection(const Board& board, Player pla, Loc startLoc, int dx, int dy) {
  assert(board.x_size == 6 && board.y_size == 6);
  const int boardSize = board.x_size;
  int x = Location::getX(startLoc, boardSize);
  int y = Location::getY(startLoc, boardSize);
  bool traversedLoop = false;

  // The four literal grid corners are not part of either capture circuit.
  if((x == 0 || x == boardSize - 1) && (y == 0 || y == boardSize - 1))
    return Board::NULL_LOC;

  // A directed circuit on a 6x6 Surakarta board has at most 24 states.
  for(int step = 0; step < 4 * boardSize; step++) {
    nextLocOfCircle(boardSize, x, y, dx, dy, traversedLoop);

    if((x == 0 || x == boardSize - 1) && (y == 0 || y == boardSize - 1))
      return Board::NULL_LOC;

    Loc loc = Location::getLoc(x, y, boardSize);
    assert(board.isOnBoard(loc));
    if(loc != startLoc && board.colors[loc] != C_EMPTY) {
      if(!traversedLoop || board.colors[loc] == pla)
        return Board::NULL_LOC;
      return loc;
    }
  }
  return Board::NULL_LOC;
}

bool isLegalCapture(const Board& board, Player pla, Loc startLoc, Loc targetLoc) {
  static const int dxs[4] = {1, -1, 0, 0};
  static const int dys[4] = {0, 0, 1, -1};
  for(int i = 0; i < 4; i++) {
    if(findCaptureLocOneDirection(board, pla, startLoc, dxs[i], dys[i]) == targetLoc)
      return true;
  }
  return false;
}

bool hasLegalDestination(const Board& board, Player pla, Loc startLoc) {
  int startX = Location::getX(startLoc, board.x_size);
  int startY = Location::getY(startLoc, board.x_size);
  for(int dy = -1; dy <= 1; dy++) {
    for(int dx = -1; dx <= 1; dx++) {
      if(dx == 0 && dy == 0)
        continue;
      int x = startX + dx;
      int y = startY + dy;
      if(x < 0 || x >= board.x_size || y < 0 || y >= board.y_size)
        continue;
      if(board.colors[Location::getLoc(x, y, board.x_size)] == C_EMPTY)
        return true;
    }
  }

  static const int dxs[4] = {1, -1, 0, 0};
  static const int dys[4] = {0, 0, 1, -1};
  for(int i = 0; i < 4; i++) {
    if(findCaptureLocOneDirection(board, pla, startLoc, dxs[i], dys[i]) != Board::NULL_LOC)
      return true;
  }
  return false;
}

Color getWinnerByStoneCount(const Board& board) {
  int blackCount = board.numPlaStonesOnBoard(C_BLACK);
  int whiteCount = board.numPlaStonesOnBoard(C_WHITE);
  if(blackCount > whiteCount)
    return C_BLACK;
  if(whiteCount > blackCount)
    return C_WHITE;
  return C_EMPTY;
}

} // namespace

bool GameLogic::isLegal(const Board& board, Player pla, Loc loc) {
  if(pla != board.nextPla)
    return false;

  // Pass is deliberately always pseudolegal. NN postprocessing masks it out
  // whenever any non-pass move exists, and playing it immediately loses.
  if(loc == Board::PASS_LOC)
    return true;
  if(!board.isOnBoard(loc))
    return false;

  if(board.stage == 0) {
    return board.colors[loc] == pla && hasLegalDestination(board, pla, loc);
  }
  if(board.stage == 1) {
    Loc startLoc = board.midLocs[0];
    Color target = board.colors[loc];
    if(target == C_EMPTY)
      return Location::euclideanDistanceSquared(startLoc, loc, board.x_size) <= 2;
    if(target == getOpp(pla))
      return isLegalCapture(board, pla, startLoc, loc);
    return false;
  }
  ASSERT_UNREACHABLE;
  return false;
}

bool GameLogic::isLegalStrict(const Board& board, Player pla, Loc loc) {
  return isLegal(board, pla, loc);
}

GameLogic::MovePriority GameLogic::getMovePriorityAssumeLegal(
  const Board& board, const BoardHistory& hist, Player pla, Loc loc
) {
  (void)board;
  (void)hist;
  (void)pla;
  (void)loc;
  return MP_NORMAL;
}

GameLogic::MovePriority GameLogic::getMovePriority(
  const Board& board, const BoardHistory& hist, Player pla, Loc loc
) {
  if(!board.isLegal(loc, pla))
    return MP_ILLEGAL;
  return getMovePriorityAssumeLegal(board, hist, pla, loc);
}

bool GameLogic::hasLegalMoveAssumeStage0(const Board& board) {
  return hasLegalMoveAssumeStage0(board, board.nextPla);
}

bool GameLogic::hasLegalMoveAssumeStage0(const Board& board, Player pla) {
  if(board.stage == 1)
    return pla == board.nextPla && hasLegalMoveAssumeStage1(board, board.midLocs[0]);
  assert(board.stage == 0);
  for(int y = 0; y < board.y_size; y++) {
    for(int x = 0; x < board.x_size; x++) {
      Loc loc = Location::getLoc(x, y, board.x_size);
      if(board.colors[loc] == pla && hasLegalDestination(board, pla, loc))
        return true;
    }
  }
  return false;
}

bool GameLogic::hasLegalMoveAssumeStage1(const Board& board, Loc chosenLoc) {
  if(!board.isOnBoard(chosenLoc))
    return false;
  Player pla = board.colors[chosenLoc];
  if(pla != C_BLACK && pla != C_WHITE)
    return false;
  return hasLegalDestination(board, pla, chosenLoc);
}

Color GameLogic::getWinnerForNoLegalMoves(const Board& board, const Rules& rules, Player stuckPla) {
  if(rules.noLegalMoveRule == Rules::NO_LEGAL_MOVE_LOSE)
    return getOpp(stuckPla);
  if(rules.noLegalMoveRule == Rules::NO_LEGAL_MOVE_DRAW)
    return C_EMPTY;
  if(rules.noLegalMoveRule == Rules::NO_LEGAL_MOVE_COUNT)
    return getWinnerByStoneCount(board);
  ASSERT_UNREACHABLE;
  return C_EMPTY;
}

Color GameLogic::checkWinnerAfterPlayed(
  const Board& board,
  const BoardHistory& hist,
  Player pla,
  Loc loc
) {
  if(loc == Board::PASS_LOC)
    return getOpp(pla);

  // Selecting a source is only the first half of a move.
  if(board.stage != 0)
    return C_WALL;

  if(board.numPlaStonesOnBoard(getOpp(pla)) == 0)
    return pla;

  Player nextPla = board.nextPla;
  if(!hasLegalMoveAssumeStage0(board, nextPla))
    return getWinnerForNoLegalMoves(board, hist.rules, nextPla);

  auto repeatIter = hist.posHashHistoryCount.find(board.pos_hash);
  assert(repeatIter != hist.posHashHistoryCount.end());
  if(repeatIter != hist.posHashHistoryCount.end() && repeatIter->second >= hist.rules.repetitionCount)
    return getWinnerByStoneCount(board);

  if(hist.rules.maxMoves > 0 && board.movenum >= hist.rules.maxMoves)
    return getWinnerByStoneCount(board);

  return C_WALL;
}

GameLogic::ResultsBeforeNN::ResultsBeforeNN()
  : inited(false), winner(C_WALL), myOnlyLoc(Board::NULL_LOC) {}

void GameLogic::ResultsBeforeNN::init(
  const Board& board, const BoardHistory& hist, Color nextPlayer
) {
  if(inited)
    return;
  inited = true;

  if(board.stage == 0 && !hasLegalMoveAssumeStage0(board, nextPlayer))
    winner = getWinnerForNoLegalMoves(board, hist.rules, nextPlayer);
}

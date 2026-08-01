#include "../game/gamelogic.h"

/*
 * gamelogic.cpp
 * Logics of game rules
 * Some other game logics are in board.h/cpp
 */

#include <algorithm>
#include <cassert>

using namespace std;

static bool checkConnectionHelper(int8_t* buf, int xs, int ys, int x0, int y0) {
  if(y0 == ys - 1)
    return true;

  buf[x0 + y0 * xs] = 2;

  static const int dxs[6] = {0, 1, 1, 0, -1, -1};
  static const int dys[6] = {-1, -1, 0, 1, 1, 0};
  for(int d = 0; d < 6; d++) {
    int x = x0 + dxs[d];
    int y = y0 + dys[d];
    if(x >= 0 && x < xs && y >= 0 && y < ys && buf[x + y * xs] == 1) {
      if(checkConnectionHelper(buf, xs, ys, x, y))
        return true;
    }
  }
  return false;
}

bool Board::checkConnection(int8_t* buf, Player pla) const {
  int xs = x_size;
  int ys = y_size;
  Player opp = getOpp(pla);

  // Always search along the y axis. White's board is transposed because
  // black connects top-to-bottom and white connects left-to-right.
  if(pla == C_BLACK) {
    for(int y = 0; y < ys; y++) {
      for(int x = 0; x < xs; x++) {
        Loc loc = Location::getLoc(x, y, xs);
        Color c = colors[loc];
        buf[x + y * xs] = c == pla ? 1 : c == opp ? 2 : 0;
      }
    }
  }
  else {
    swap(xs, ys);
    for(int y = 0; y < ys; y++) {
      for(int x = 0; x < xs; x++) {
        Loc loc = Location::getLoc(y, x, ys);
        Color c = colors[loc];
        buf[x + y * xs] = c == pla ? 1 : c == opp ? 2 : 0;
      }
    }
  }

  for(int x = 0; x < xs; x++) {
    if(buf[x] == 1 && checkConnectionHelper(buf, xs, ys, x, 0))
      return true;
  }
  return false;
}

Color GameLogic::checkWinnerAfterPlayed(
  const Board& board,
  const BoardHistory& hist,
  Player pla,
  Loc loc,
  int8_t* bufferForCheckingWinner) {
  (void)hist;

  // AntiHex: making your own real stone chain connect your two sides loses.
  if(board.checkConnection(bufferForCheckingWinner, pla))
    return getOpp(pla);

  if(loc == Board::PASS_LOC)
    return getOpp(pla);  // Pass is not allowed.

  return C_WALL;
}

GameLogic::ResultsBeforeNN::ResultsBeforeNN() {
  inited = false;
  winner = C_WALL;
  myOnlyLoc = Board::NULL_LOC;
}

void GameLogic::ResultsBeforeNN::init(const Board& board, const BoardHistory& hist, Color nextPlayer) {
  (void)board;
  (void)hist;
  (void)nextPlayer;
  if(inited)
    return;
  inited = true;
}

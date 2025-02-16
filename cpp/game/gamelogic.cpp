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
static bool checkConnection(const Board& board, Player pla) {
  int directions[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};  // 四个方向：水平、垂直、对角线（正、反）
  int x_size = board.x_size;
  int y_size = board.y_size;

  for(int x = 0; x < x_size; ++x) {
    for(int y = 0; y < y_size; ++y) {
      if(board.colors[Location::getLoc(x, y, x_size)] == pla) {
        for(auto dir: directions) {
          int dx = dir[0];
          int dy = dir[1];
          int count = 1;
          int nx = x + dx;
          int ny = y + dy;

          while(nx >= 0 && nx < x_size && ny >= 0 && ny < y_size &&
                board.colors[Location::getLoc(nx, ny, x_size)] == pla) {
            count++;
            nx += dx;
            ny += dy;
          }

          if(count >= 4) {
            return true;
          }
        }
      }
    }
  }

  return false;
}

static int connectionLengthOneDirection(const Board& board, Player pla, bool isSixWin, Loc loc, short adj) {
  Loc tmploc = loc;
  int conNum = 0;
  while(1) {
    tmploc += adj;
    if(!board.isOnBoard(tmploc))
      break;
    if(board.colors[tmploc] == pla)
      conNum++;
    else if(board.colors[tmploc] == C_EMPTY) {

      break;
    } else
      break;
  }
  return conNum;
}

static bool isWinOneDirectionAssumeLegal(
  const Board& board,
  Player pla,
  bool isSixWin,
  Loc loc,
  int adj) {
  Player opp = getOpp(pla);
  int myConNum = connectionLengthOneDirection(board, pla, isSixWin, loc, adj) +
                 connectionLengthOneDirection(board, pla, isSixWin, loc, -adj) + 1;

  if(myConNum == 4 || (myConNum > 4 && isSixWin))
    return true;


  return false;
}

bool GameLogic::isWinAssumeLegal(const Board& board, const BoardHistory& hist, Player pla, Loc loc) {
  if(loc == Board::PASS_LOC)
    return false;

  bool isSixWin = true;

  int adjs[4] = {1, (board.x_size + 1), (board.x_size + 1) + 1, (board.x_size + 1) - 1};  // +x +y +x+y -x+y
  for(int i = 0; i < 4; i++) {
    if(isWinOneDirectionAssumeLegal(board, pla, isSixWin, loc, adjs[i]))
      return true;
  }

  return false;
}

Color GameLogic::checkWinnerAfterPlayed(
  const Board& board,
  const BoardHistory& hist,
  Player pla,
  Loc loc) {


  Player opp = getOpp(pla);
  
  if(hist.rules.VCNRule != Rules::VCNRULE_NOVC || hist.rules.firstPassWin || hist.rules.wallBlock) {
    throw StringError("GameLogic::checkWinnerAfterPlayed This rule is not supported for ZhenQi");
  }

  if(!board.isOnBoard(loc))  // probably pass
    if(loc == Board::PASS_LOC)
      return opp;
    else
      ASSERT_UNREACHABLE;

  // 连四判定
  bool plaConnect = checkConnection(board, pla);
  bool oppConnect = checkConnection(board, opp);
  if(plaConnect && oppConnect) {
    if(hist.rules.sameTimeWinRule == Rules::SAMETIMEWIN_BLACK)
      return C_BLACK;
    else if(hist.rules.sameTimeWinRule == Rules::SAMETIMEWIN_SELF)
      return pla;
    else if(hist.rules.sameTimeWinRule == Rules::SAMETIMEWIN_OPP)
      return opp;
    else
      ASSERT_UNREACHABLE;
  } else if(plaConnect)
    return pla;
  else if(oppConnect)
    return opp;

  // maxmoves判定
  if(hist.rules.maxMoves != 0 && board.movenum >= hist.rules.maxMoves) {
     return C_EMPTY;
  }

  if (board.movenum >= 10 * board.x_size * board.y_size)
  {
    //if(board.x_size * board.y_size >= 40)
    //  cout << "Long game >= 10 * board.x_size * board.y_size" << endl;
    return C_EMPTY;
  }

  if(board.numPlaStonesOnBoard(C_EMPTY) == 0)
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
  return;

}

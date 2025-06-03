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

static int connectionLengthOneDirection(
  const Board& board,
  Player pla,
  Loc loc,
  short adj) {
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

static bool isConnected(const Board& board, Player pla, Loc loc) {
  int adjs[4] = {1, (board.x_size + 1), (board.x_size + 1) + 1, (board.x_size + 1) - 1};  // +x +y +x+y -x+y
  for(int i = 0; i < 4; i++) {
    int adj = adjs[i];
    int myConNum =
      connectionLengthOneDirection(board, pla, loc, adj) + connectionLengthOneDirection(board, pla, loc, -adj) + 1;
    if(myConNum >= CON_LEN)
      return true;
  }

  return false;
}

bool Board::isPruned(Loc loc, Player pla) const {
  if(!isLegal(loc, pla))
    return true;
  if(loc == PASS_LOC)
    return true;

  if(isConnected(*this, pla, loc))
    return true;

  return false;
}


Color GameLogic::checkWinnerAfterPlayed(
  const Board& board,
  const BoardHistory& hist,
  Player pla,
  Loc loc) {

  if(loc == Board::PASS_LOC)
    return getOpp(pla);  //pass is not allowed
  
  if(isConnected(board, pla, loc))
    return getOpp(pla);  //connected



  if(board.stonenum >= board.x_size * board.y_size)
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
  //write your own logic

  return;
}

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


bool GameLogic::isLegal(const Board& board, Player pla, Loc loc) {
  
  if(pla != board.nextPla) {
    std::cout << "Error next player ";
    return false;
  }

  if(loc != Board::PASS_LOC && (!board.isOnBoard(loc)))
    return false;

  if(board.stage == 0)  // move the piece
  {
    assert(board.legalMapUpToDate);
    if(loc==Board::PASS_LOC) {
      return true;
    }
    return board.legalMap[loc] == 1;
  } 
  else if(board.stage == 1)  // place a ban loc
  {
    if(loc == Board::PASS_LOC) {
      return true;
    }
    
    return board.colors[loc] == C_EMPTY;
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




Color GameLogic::checkWinnerAfterPlayed(
  const Board& board,
  const BoardHistory& hist,
  Player pla,
  Loc loc) {

  if(loc == Board::PASS_LOC) {
    return getOpp(pla);
  }

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

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


bool GameLogic::isLegal(const Board& board, Player pla, Loc loc, const Rules& rule) {
  
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
    return board.colors[loc] == C_EMPTY;
  } 
  else if(board.stage == 1)  // place a ban loc
  {
    if(loc == Board::PASS_LOC) {
      return true;
    }
    int neutral_stones_played = pla == P_BLACK ? board.neutral_stones_b : board.neutral_stones_w;
    int neutral_stones_remain = rule.komi - neutral_stones_played;
    if(neutral_stones_remain <= 0)
      return false;
    return board.colors[loc] == C_EMPTY;
  }
  ASSERT_UNREACHABLE;
  return false;
}

int Board::calculateFinalScore(Player pla) const {
  bool c[16] = {false}; 
  for(Loc loc = 0; loc < Board::MAX_ARR_SIZE; loc++)
  {
    if(colors[loc] == pla) // check which type
    {
      int res=0;
      for(int adji=0;adji<4;adji++)
      {
        bool hasMyStone=false;
        Loc adj=adj_offsets[adji];
        Loc l=loc+adj;
        while(l>=0 && l<Board::MAX_ARR_SIZE && colors[l]!=C_WALL)
        {
          if(colors[l]==pla)
          {
            hasMyStone=true;
            break;
          }
          l+=adj;
        }
        if(hasMyStone)
          res+=1<<adji;
      }

      c[res] = true;
    }
  }
  //sum c
  int sc=0;
  for(int i=0;i<16;i++)
    if(c[i])
      sc++;
  return sc;
}



GameLogic::MovePriority GameLogic::getMovePriorityAssumeLegal(const Board& board, const BoardHistory& hist, Player pla, Loc loc) {
  return MP_NORMAL;
}

GameLogic::MovePriority GameLogic::getMovePriority(const Board& board, const BoardHistory& hist, Player pla, Loc loc) {
  if(loc == Board::PASS_LOC)
    return MP_NORMAL;
  if(!board.isLegal(loc, pla, hist.rules))
    return MP_ILLEGAL;
  MovePriority MP = getMovePriorityAssumeLegal(board, hist, pla, loc);
  return MP;
}





Color GameLogic::checkWinnerAfterPlayed(
  const Board& board,
  const BoardHistory& hist,
  Player pla,
  Loc loc) {
  if(board.stage == 1 && loc == Board::PASS_LOC) { //pass is not allowed in stage0
    return getOpp(pla);
  }
  if(board.stage == 0) {
    int sb = board.numPlaStonesOnBoard(C_BLACK);
    int sw = board.numPlaStonesOnBoard(C_WHITE);
    if(sb >= Rules::STONE_NUM_LIMIT && sw >= Rules::STONE_NUM_LIMIT)  // game ends, check winner
    {
      int scoreb = board.calculateFinalScore(C_BLACK);
      int scorew = board.calculateFinalScore(C_WHITE);
      if(scoreb > scorew)
        return C_BLACK;
      else if(scoreb < scorew)
        return C_WHITE;
      else
      {
        if(board.neutral_stones_b < board.neutral_stones_w)
          return C_BLACK;
        else if(board.neutral_stones_b > board.neutral_stones_w)
          return C_WHITE;
        else
          return C_EMPTY;
      }
    }
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

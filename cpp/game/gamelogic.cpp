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
  const Rules& rules,
  Player pla,
  Loc loc,
  short adj) 
{
  Loc tmploc = loc;
  int conNum = 0;
  Color nextColor;
  while(1) {
    tmploc += adj;
    nextColor = board.isOnBoard(tmploc) ? board.colors[tmploc] : C_WALL;
    if(nextColor != pla)
      break;
    conNum++;
  }
  return conNum;
}




static bool isFive_oneLine(
  const Board& board,
  const Rules& rules,
  Player pla,
  Loc loc,
  int adj)  {
  int myConNum = connectionLengthOneDirection(board, rules, pla, loc, adj) +
                 connectionLengthOneDirection(board, rules, pla, loc, -adj) + 1;
  return myConNum >= 5;
}

static int captureNumAfterMove(const Board& board, const Rules& rules, Player pla, Loc loc) {
  if(!board.isOnBoard(loc))
    return 0;
  int cap = 0;
  Color opp = getOpp(pla);
  for(int i = 0; i < 8; i++) {
    int adj = board.adj_offsets[i];
    cap += board.captureNumAfterMoveOneDirection(rules, pla, loc, adj);
  }

  return cap;
}

MovePriority GameLogic::getMovePriorityAssumeLegal(const Board& board, const BoardHistory& hist, Player pla, Loc loc) {
  if(loc == Board::PASS_LOC)
    return MP_NORMAL;

  int adjs[4] = {1, (board.x_size + 1), (board.x_size + 1) + 1, (board.x_size + 1) - 1};// +x +y +x+y -x+y
  for(int i = 0; i < 4; i++) {
    MovePriority tmpMP = MP_NORMAL;
    if(isFive_oneLine(board, hist.rules, pla, loc, adjs[i]))
      return MP_FIVE;
  }

  int cap = captureNumAfterMove(board, hist.rules, pla, loc);
  int myRemain =
    pla == C_BLACK ? hist.rules.blackTargetCap - board.blackCapNum : hist.rules.whiteTargetCap - board.whiteCapNum;
  if(cap>=myRemain)
    return MP_FIVE;


  return MP_NORMAL;
}

MovePriority GameLogic::getMovePriority(const Board& board, const BoardHistory& hist, Player pla, Loc loc) {
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



  
  if((hist.rules.maxMoves != 0 || hist.rules.VCNRule != Rules::VCNRULE_NOVC) && hist.rules.firstPassWin) {
    throw StringError("GameLogic::checkWinnerAfterPlayed This rule is not supported");
  }

  Player opp = getOpp(pla);
  int myPassNum = pla == C_BLACK ? board.blackPassNum : board.whitePassNum;
  int oppPassNum = pla == C_WHITE ? board.blackPassNum : board.whitePassNum;

  if(loc == Board::PASS_LOC) {
    if(hist.rules.VCNRule == Rules::VCNRULE_NOVC) {
      if(oppPassNum > 0) {
        if(!hist.rules.firstPassWin)  // 常规和棋
        {
          return C_EMPTY;
        } else  // 对方先pass
        {
          return opp;
        }
      }
    } else {
      Color VCside = hist.rules.vcSide();
      int VClevel = hist.rules.vcLevel();

      if(VCside == pla)  // VCN不允许己方pass
      {
        return opp;
      } else  // pass次数足够则判胜
      {
        if(myPassNum >= 6 - VClevel) {
          return pla;
        }
      }
    }
  }

  // 连五判定

  int adjs[4] = {1, (board.x_size + 1), (board.x_size + 1) + 1, (board.x_size + 1) - 1};  // +x +y +x+y -x+y
  for(int i = 0; i < 4; i++) {
    if(isFive_oneLine(board, hist.rules, pla, loc, adjs[i]))
      return pla;
  }
  // 提子数判定
  int myRemain =
    pla == C_BLACK ? hist.rules.blackTargetCap - board.blackCapNum : hist.rules.whiteTargetCap - board.whiteCapNum;
  if(myRemain <= 0)
    return pla;

  // maxmoves判定
  if(hist.rules.maxMoves != 0 && board.movenum >= hist.rules.maxMoves) {
    if(hist.rules.VCNRule == Rules::VCNRULE_NOVC) {
      return C_EMPTY;
    } else  // 和棋判进攻方负
    {
      static_assert(Rules::VCNRULE_VC1_W == Rules::VCNRULE_VC1_B + 10, "Ensure VCNRule%10==N, VCNRule/10+1==color");
      Color VCside = hist.rules.vcSide();
      return getOpp(VCside);
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
  if(hist.rules.VCNRule != Rules::VCNRULE_NOVC && hist.rules.maxMoves != 0)
    throw StringError("ResultBeforeNN::init() can not support VCN and maxMoves simutaneously");
  if(inited)
    return;
  inited = true;

  Color opp = getOpp(nextPlayer);

  // check five and four
  for(int x = 0; x < board.x_size; x++)
    for(int y = 0; y < board.y_size; y++) {
      Loc loc = Location::getLoc(x, y, board.x_size);
      MovePriority mp = getMovePriority(board, hist, nextPlayer, loc);
      if(mp == MP_FIVE) {
        winner = nextPlayer;
        myOnlyLoc = loc;
        return;
      } 
    }

  if(hist.rules.VCNRule != Rules::VCNRULE_NOVC) {
    int vcLevel = hist.rules.vcLevel() + board.blackPassNum + board.whitePassNum;

    Color vcSide = hist.rules.vcSide();
    if(vcSide == nextPlayer) {
      if(vcLevel == 5) {
        winner = opp;
        myOnlyLoc = Board::NULL_LOC;
        return;
      }
    } else if(vcSide == opp) {
      if(vcLevel == 5) {
        winner = nextPlayer;
        myOnlyLoc = Board::PASS_LOC;
        return;
      } 
    }
  }


}

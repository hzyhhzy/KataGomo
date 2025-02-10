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

static bool isInTrap(Loc loc, Player pla)
{
  if(pla == C_BLACK)
    return loc == Location::getLocConst(3, 7) || 
           loc == Location::getLocConst(2, 8) ||
           loc == Location::getLocConst(4, 8);
  else if(pla == C_WHITE)
    return loc == Location::getLocConst(3, 1) || 
           loc == Location::getLocConst(2, 0) ||
           loc == Location::getLocConst(4, 0);
  else
    return false;
}
static bool isInRiver(Loc loc)
{
  return 
    loc == Location::getLocConst(1, 3) ||
    loc == Location::getLocConst(1, 4) ||
    loc == Location::getLocConst(1, 5) ||
    loc == Location::getLocConst(2, 3) ||
    loc == Location::getLocConst(2, 4) ||
    loc == Location::getLocConst(2, 5) ||
    loc == Location::getLocConst(4, 3) ||
    loc == Location::getLocConst(4, 4) ||
    loc == Location::getLocConst(4, 5) ||
    loc == Location::getLocConst(5, 3) ||
    loc == Location::getLocConst(5, 4) ||
    loc == Location::getLocConst(5, 5) ;
}

static Loc getHomeLoc(Player pla) {
  if(pla == C_BLACK)
    return Location::getLocConst(3, 8);
  else if(pla == C_WHITE)
    return Location::getLocConst(3, 0);
  else
    return Board::NULL_LOC;
}

bool Board::maybeCrossRiver(Loc loc0, Loc loc1) const {
  if (loc0 == Location::getLocConst(0, 3))
  {
    if(loc1 != Location::getLocConst(3, 3))
      return false;
    if(colors[Location::getLocConst(1, 3)] != C_EMPTY)
      return false;
    if(colors[Location::getLocConst(2, 3)] != C_EMPTY)
      return false;
    return true;
  }
  else if(loc0 == Location::getLocConst(0, 4)) {
    if(loc1 != Location::getLocConst(3, 4))
      return false;
    if(colors[Location::getLocConst(1, 4)] != C_EMPTY)
      return false;
    if(colors[Location::getLocConst(2, 4)] != C_EMPTY)
      return false;
    return true;
  }
  else if(loc0 == Location::getLocConst(0, 5)) {
    if(loc1 != Location::getLocConst(3, 5))
      return false;
    if(colors[Location::getLocConst(1, 5)] != C_EMPTY)
      return false;
    if(colors[Location::getLocConst(2, 5)] != C_EMPTY)
      return false;
    return true;
  } 
  else if (loc0 == Location::getLocConst(6, 3))
  {
    if(loc1 != Location::getLocConst(3, 3))
      return false;
    if(colors[Location::getLocConst(5, 3)] != C_EMPTY)
      return false;
    if(colors[Location::getLocConst(4, 3)] != C_EMPTY)
      return false;
    return true;
  }
  else if(loc0 == Location::getLocConst(6, 4)) {
    if(loc1 != Location::getLocConst(3, 4))
      return false;
    if(colors[Location::getLocConst(5, 4)] != C_EMPTY)
      return false;
    if(colors[Location::getLocConst(4, 4)] != C_EMPTY)
      return false;
    return true;
  }
  else if(loc0 == Location::getLocConst(6, 5)) {
    if(loc1 != Location::getLocConst(3, 5))
      return false;
    if(colors[Location::getLocConst(5, 5)] != C_EMPTY)
      return false;
    if(colors[Location::getLocConst(4, 5)] != C_EMPTY)
      return false;
    return true;
  }
  else if (loc0 == Location::getLocConst(3, 3))
  {
    if(loc1 == Location::getLocConst(0, 3)) {
      if(colors[Location::getLocConst(1, 3)] != C_EMPTY)
        return false;
      if(colors[Location::getLocConst(2, 3)] != C_EMPTY)
        return false;
      return true;
    }
    else if(loc1 == Location::getLocConst(6, 3)) {
      if(colors[Location::getLocConst(5, 3)] != C_EMPTY)
        return false;
      if(colors[Location::getLocConst(4, 3)] != C_EMPTY)
        return false;
      return true;
    } 
    else
      return false;
  }
  else if (loc0 == Location::getLocConst(3, 4))
  {
    if(loc1 == Location::getLocConst(0, 4)) {
      if(colors[Location::getLocConst(1, 4)] != C_EMPTY)
        return false;
      if(colors[Location::getLocConst(2, 4)] != C_EMPTY)
        return false;
      return true;
    }
    else if(loc1 == Location::getLocConst(6, 4)) {
      if(colors[Location::getLocConst(5, 4)] != C_EMPTY)
        return false;
      if(colors[Location::getLocConst(4, 4)] != C_EMPTY)
        return false;
      return true;
    } 
    else
      return false;
  }
  else if (loc0 == Location::getLocConst(3, 5))
  {
    if(loc1 == Location::getLocConst(0, 5)) {
      if(colors[Location::getLocConst(1, 5)] != C_EMPTY)
        return false;
      if(colors[Location::getLocConst(2, 5)] != C_EMPTY)
        return false;
      return true;
    }
    else if(loc1 == Location::getLocConst(6, 5)) {
      if(colors[Location::getLocConst(5, 5)] != C_EMPTY)
        return false;
      if(colors[Location::getLocConst(4, 5)] != C_EMPTY)
        return false;
      return true;
    } 
    else
      return false;
  }
  else if (loc0 == Location::getLocConst(1, 2))
  {
    if(loc1 != Location::getLocConst(1, 6))
      return false;
    if(colors[Location::getLocConst(1, 3)] != C_EMPTY)
      return false;
    if(colors[Location::getLocConst(1, 4)] != C_EMPTY)
      return false;
    if(colors[Location::getLocConst(1, 5)] != C_EMPTY)
      return false;
    return true;
  }
  else if (loc0 == Location::getLocConst(2, 2))
  {
    if(loc1 != Location::getLocConst(2, 6))
      return false;
    if(colors[Location::getLocConst(2, 3)] != C_EMPTY)
      return false;
    if(colors[Location::getLocConst(2, 4)] != C_EMPTY)
      return false;
    if(colors[Location::getLocConst(2, 5)] != C_EMPTY)
      return false;
    return true;
  }
  else if (loc0 == Location::getLocConst(4, 2))
  {
    if(loc1 != Location::getLocConst(4, 6))
      return false;
    if(colors[Location::getLocConst(4, 3)] != C_EMPTY)
      return false;
    if(colors[Location::getLocConst(4, 4)] != C_EMPTY)
      return false;
    if(colors[Location::getLocConst(4, 5)] != C_EMPTY)
      return false;
    return true;
  }
  else if (loc0 == Location::getLocConst(5, 2))
  {
    if(loc1 != Location::getLocConst(5, 6))
      return false;
    if(colors[Location::getLocConst(5, 3)] != C_EMPTY)
      return false;
    if(colors[Location::getLocConst(5, 4)] != C_EMPTY)
      return false;
    if(colors[Location::getLocConst(5, 5)] != C_EMPTY)
      return false;
    return true;
  }
  else if (loc0 == Location::getLocConst(1, 6))
  {
    if(loc1 != Location::getLocConst(1, 2))
      return false;
    if(colors[Location::getLocConst(1, 3)] != C_EMPTY)
      return false;
    if(colors[Location::getLocConst(1, 4)] != C_EMPTY)
      return false;
    if(colors[Location::getLocConst(1, 5)] != C_EMPTY)
      return false;
    return true;
  }
  else if (loc0 == Location::getLocConst(2, 6))
  {
    if(loc1 != Location::getLocConst(2, 2))
      return false;
    if(colors[Location::getLocConst(2, 3)] != C_EMPTY)
      return false;
    if(colors[Location::getLocConst(2, 4)] != C_EMPTY)
      return false;
    if(colors[Location::getLocConst(2, 5)] != C_EMPTY)
      return false;
    return true;
  }
  else if (loc0 == Location::getLocConst(4, 6))
  {
    if(loc1 != Location::getLocConst(4, 2))
      return false;
    if(colors[Location::getLocConst(4, 3)] != C_EMPTY)
      return false;
    if(colors[Location::getLocConst(4, 4)] != C_EMPTY)
      return false;
    if(colors[Location::getLocConst(4, 5)] != C_EMPTY)
      return false;
    return true;
  }
  else if (loc0 == Location::getLocConst(5, 6))
  {
    if(loc1 != Location::getLocConst(5, 2))
      return false;
    if(colors[Location::getLocConst(5, 3)] != C_EMPTY)
      return false;
    if(colors[Location::getLocConst(5, 4)] != C_EMPTY)
      return false;
    if(colors[Location::getLocConst(5, 5)] != C_EMPTY)
      return false;
    return true;
  }

  return false;


}


bool GameLogic::isLegal(const Board& board, Player pla, Loc loc) {
  if(pla != board.nextPla) {
    std::cout << "Error next player ";
    return false;
  }

  if(loc == Board::PASS_LOC)  // pass is lose, but not illegal
    return true;

  if(!board.isOnBoard(loc))
    return false;

  if(board.stage == 0)  // choose a piece
  {
    return getPiecePla(board.colors[loc]) == pla;
  } 
  else if(board.stage == 1)  // place the piece
  {
    Color opp = getOpp(pla);
    Loc chosenMove = board.midLocs[0];
    Color c0 = board.colors[chosenMove];
    Color c1 = board.colors[loc];
    assert(getPiecePla(c0) == pla);
    if(getPiecePla(c1) == pla)
      return false;  // cannot eat my piece
    Color p0 = getPieceType(c0);
    Color p1 = getPieceType(c1);  // opponent piece(1~8) or empty(0)

    int x0 = Location::getX(chosenMove, board.x_size);
    int y0 = Location::getY(chosenMove, board.x_size);
    int x1 = Location::getX(loc, board.x_size);
    int y1 = Location::getY(loc, board.x_size);
    int dy = y1 - y0;
    int dx = x1 - x0;
    if(dx != 0 && dy != 0)
      return false;
    if(dx * dx + dy * dy != 1 && !((p0 == C_LION || p0 == C_TIGER) && board.maybeCrossRiver(chosenMove, loc)))
      return false;
    if(loc == getHomeLoc(pla))
      return false;

    bool inTrap = isInTrap(loc, pla);  // my trap, can eat opponent's any piece
    bool inRiver = isInRiver(loc);

    if(inRiver && p0 != C_RAT) {
      return false;  // other pieces cannot move into river
    }
    else if(inRiver)
    {
      if(p1 != C_EMPTY && !isInRiver(chosenMove))
        return false; //cannot eat opponent's rat in river
    }

    if(p0 == C_RAT && p1 != C_EMPTY && isInRiver(chosenMove) && !inRiver) {
      return false;  // rat in river cannot eat piece on land
    }

    if(p1 == C_EMPTY) {
      return true;  // can move to empty space
    }
    if(inTrap) {
      return true;  // can eat any piece in my trap
    }
    if(p0 == C_RAT && p1 == C_ELEPHANT) {
      return true;  // rat can eat elephant
    }
    if(p0 >= p1 && (p0 != C_ELEPHANT || p1 != C_RAT)) {
      return true; // can eat smaller piece, but elephant cannot eat rat
    }
    return false; // cannot eat bigger piece

  }
  ASSERT_UNREACHABLE;
  return false;
}

GameLogic::MovePriority GameLogic::getMovePriorityAssumeLegal(const Board& board, const BoardHistory& hist, Player pla, Loc loc) {
  if(loc == Board::PASS_LOC)
    return MP_NORMAL;

  int y = Location::getY(loc, board.x_size);
  if(board.stage == 0) {
    return MP_NORMAL;
  }
  else if(board.stage == 1) {
    if(loc==getHomeLoc(getOpp(pla)))
      return MP_SUDDEN_WIN;
  }


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
  if(loc == Board::PASS_LOC)
    return getOpp(pla);  //pass is not allowed
  
  
  if(getPiecePla(board.colors[getHomeLoc(getOpp(pla))]) == pla)
    return pla;
  //TODO: move limit


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


  for(int x = 0; x < board.x_size; x++)
    for(int y = 0; y < board.y_size; y++) {
      Loc loc = Location::getLoc(x, y, board.x_size);
      MovePriority mp = getMovePriority(board, hist, nextPlayer, loc);
      if(mp == MP_SUDDEN_WIN || mp == MP_WINNING) {
        winner = nextPlayer;
        myOnlyLoc = loc;
        return;
      }
    }



  return;
}

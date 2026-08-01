/*
 * gamelogic.h
 * Logics of game rules
 * Some other game logics are in board.h/cpp
 */

#ifndef GAME_GAMELOGIC_H_
#define GAME_GAMELOGIC_H_

#include "../game/boardhistory.h"

/*
* Other game logics:
* Board::
*/

namespace GameLogic {

  //C_EMPTY = draw, C_WALL = not finished 
  Color checkWinnerAfterPlayed(
    const Board& board,
    const BoardHistory& hist,
    Player pla,
    Loc loc,
    int8_t* bufferForCheckingWinner);


  //some results calculated before calculating NN
  //part of NN input, and then change policy/value according to this
  struct ResultsBeforeNN {
    bool inited;
    Color winner;
    Loc myOnlyLoc;
    ResultsBeforeNN();
    void init(const Board& board, const BoardHistory& hist, Color nextPlayer);
  };
}




#endif // GAME_RULELOGIC_H_

#include "../game/randomopening.h"
#include "../game/gamelogic.h"
#include "../core/rand.h"
#include "../search/asyncbot.h"
using namespace RandomOpening;
//disabled

void RandomOpening::initializeBalancedRandomOpening(
  Board& board,
  BoardHistory& hist,
  Player& nextPlayer,
  Rand& gameRand,
  bool forSelfplay) {
    
  if(gameRand.nextBool(playSettings.randomInitPieceProb)) {
    double changeProb = playSettings.randomInitPieceDensity * gameRand.nextExponential();
    double pieceDensity = pow(gameRand.nextDouble(), 4) * 0.5;
    for(int x = 0; x < board.x_size; x++)
      for(int y = 0; y < board.y_size; y++) {
        if(gameRand.nextDouble() > changeProb)
          continue;
        Loc loc = Location::getLoc(x, y, board.x_size);
        if(loc == GameLogic::getHomeLoc(C_BLACK) || loc == GameLogic::getHomeLoc(C_WHITE) || GameLogic::isInRiver(loc))
          continue;

        Color c = C_EMPTY;

        // will set a random piece
        if (gameRand.nextBool(pieceDensity))
        {
          double blackProb = ((tanh(double(2 * y + 1 - board.y_size) / double(board.y_size)) / tanh(1)) + 1)/2;
          Player side = gameRand.nextBool(blackProb) ? C_BLACK : C_WHITE;  // the more near the home, the higher prob is my piece
          c = getPiece(side, 1 + gameRand.nextUInt(8));
        }

        board.setStone(loc, c);

      }
  }


  }



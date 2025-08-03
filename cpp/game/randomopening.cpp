#include "../game/randomopening.h"
#include "../game/gamelogic.h"
#include "../core/rand.h"
#include "../search/asyncbot.h"
#include <vector>
using namespace RandomOpening;
//disabled

void RandomOpening::initializeRandomOpening(
  Board& board,
  BoardHistory& hist,
  Player& nextPlayer,
  Rand& gameRand,
  const PlaySettings& playSettings) {
    
  if(gameRand.nextBool(playSettings.randomInitPieceProb)) {

    //random change some pieces
    double changeProb = playSettings.randomInitPieceDensity * gameRand.nextExponential();
    double pieceDensity = pow(gameRand.nextDouble(), 4.0) * 0.5;
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

    //random move some pieces
    int stoneNum=0;
    for(int x = 0; x < board.x_size; x++)
      for(int y = 0; y < board.y_size; y++) {
        Loc loc = Location::getLoc(x, y, board.x_size);
        if(board.colors[loc] != C_EMPTY)
          stoneNum++;
      }
    int moveStoneNum = int(playSettings.randomMovePieceRate * stoneNum * gameRand.nextExponential());
    for(int i = 0; i < moveStoneNum; i++) {
      //random select a piece on board
      std::vector<Loc> allLocs;
      //find all non-empty locations
      for(int x = 0; x < board.x_size; x++)
        for(int y = 0; y < board.y_size; y++) {
          Loc loc = Location::getLoc(x, y, board.x_size);
          if(board.colors[loc] != C_EMPTY)
            allLocs.push_back(loc);
        }
      if(allLocs.size() == 0)
        break;
      Loc loc_from = allLocs[gameRand.nextUInt(allLocs.size())];
      Color c=board.colors[loc_from];
      Color pside=getPiecePla(c);
      Color ptype=getPieceType(c);
      int randX=gameRand.nextUInt(board.x_size);
      int randY=gameRand.nextUInt(board.y_size);
      Loc loc_to=Location::getLoc(randX,randY,board.x_size);
      if(loc_to == loc_from)
        continue;
      if(loc_to == GameLogic::getHomeLoc(C_BLACK) || loc_to == GameLogic::getHomeLoc(C_WHITE))
        continue;
      if(GameLogic::isInRiver(loc_to) && ptype != C_RAT)
        continue;
      if(GameLogic::isInTrap(loc_to,getOpp(pside)))
        continue;
      if((randY>=6&&pside==C_WHITE) || (randY<=2&&pside==C_BLACK))//on opponent's side
        if(gameRand.nextBool(0.75))
          continue;
      board.setStone(loc_to,c);
      board.setStone(loc_from,C_EMPTY);
    }

    nextPlayer=board.nextPla;
    Rules rules=hist.rules;
    hist.clear(board,nextPlayer,rules);
  }



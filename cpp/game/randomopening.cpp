#include "../game/randomopening.h"
#include "../game/gamelogic.h"
#include "../core/rand.h"
#include "../search/asyncbot.h"
using namespace RandomOpening;

void RandomOpening::initializeBalancedRandomOpening(
  Search* botB,
  Search* botW,
  Board& board,
  BoardHistory& hist,
  Player& nextPlayer,
  Rand& gameRand,
  double makeOpeningFairRate,
  double dropPow,
  double minAcceptRate) {

  if(gameRand.nextBool(makeOpeningFairRate))  // make game fair
  {
    Loc firstMove;
    Loc secondMove;
    while(1) {
      int boardArea = board.x_size * board.y_size;
      assert(boardArea >= 2);
      int firstPos = gameRand.nextUInt(boardArea);
      int secondPos = gameRand.nextUInt(boardArea - 1);
      if(secondPos >= firstPos)
        secondPos += 1;
      firstMove = Location::getLoc(firstPos % board.x_size, firstPos / board.x_size, board.x_size);
      secondMove = Location::getLoc(secondPos % board.x_size, secondPos / board.x_size, board.x_size);

      Board boardCopy(board);
      BoardHistory histCopy(hist);
      Player firstPlayer = nextPlayer;
      Player secondPlayer = getOpp(firstPlayer);
      histCopy.makeBoardMoveAssumeLegal(boardCopy, firstMove, firstPlayer);
      histCopy.makeBoardMoveAssumeLegal(boardCopy, secondMove, secondPlayer);

      NNResultBuf nnbuf;
      MiscNNInputParams nnInputParams;
      Search* nextBot = firstPlayer == C_BLACK ? botB : botW;
      nextBot->nnEvaluator->evaluate(boardCopy, histCopy, firstPlayer, nnInputParams, nnbuf, false);
      std::shared_ptr<NNOutput> nnOutput = std::move(nnbuf.result);

      double maxOutcome = std::max(
        nnOutput->whiteWinProb,
        std::max(nnOutput->whiteLossProb, nnOutput->whiteNoResultProb));
      double acceptRate = pow(1 - maxOutcome, dropPow);
      acceptRate = std::max(acceptRate, minAcceptRate);
      if(gameRand.nextBool(acceptRate))
        break;
    }

    hist.makeBoardMoveAssumeLegal(board, firstMove, nextPlayer);
    nextPlayer = getOpp(nextPlayer);
    hist.makeBoardMoveAssumeLegal(board, secondMove, nextPlayer);
    nextPlayer = getOpp(nextPlayer);
  }
}

void RandomOpening::initializeSpecialOpening(
  Search* botB,
  Search* botW,
  Board& board,
  BoardHistory& hist,
  Player& nextPlayer,
  Rand& gameRand) {
  (void)botB;
  (void)botW;
  (void)board;
  (void)hist;
  (void)nextPlayer;
  (void)gameRand;
}

void RandomOpening::initializeCompletelyRandomOpening(
  Board& board,
  BoardHistory& hist,
  Player& nextPlayer,
  Rand& gameRand,
  double areaPropAvg) {
  double fillProb = gameRand.nextExponential() * areaPropAvg;
  randomFillBoard(board, gameRand, fillProb, fillProb);
  nextPlayer = gameRand.nextBool(0.5) ? C_BLACK : C_WHITE;
  auto rules = hist.rules;
  hist.clear(board, nextPlayer, rules);
}

void RandomOpening::randomFillBoard(Board& board, Rand& gameRand, double bProb, double wProb) {
  if(bProb > 0.5)
    bProb = 0.5;
  if(wProb + bProb > 1)
    wProb = 1 - bProb;

  for(int x = 0; x < board.x_size; x++)
    for(int y = 0; y < board.y_size; y++) {
      Loc loc = Location::getLoc(x, y, board.x_size);
      if(board.colors[loc] == C_EMPTY)
      {
        Color c = C_EMPTY;
        double r = gameRand.nextDouble();
        if(r < bProb)
          c = C_BLACK;
        else if(r < bProb + wProb)
          c = C_WHITE;
        board.setStone(loc, c);
      }
    }
}

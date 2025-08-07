#include "../game/randomopening.h"
#include "../game/gamelogic.h"
#include "../core/rand.h"
#include "../search/asyncbot.h"
#include <vector>
using namespace RandomOpening;

std::atomic<int64_t> triedCount(0);
std::atomic<int64_t> succeedCount(0);
std::atomic<int64_t> evalCount(0);

//win loss draw 3 values
struct BoardValue {
  double win;
  double loss;
  double draw;
};


static BoardValue getBoardValue(Search* bot, const Board& board, const BoardHistory& hist, Player nextPlayer) {
  evalCount++;
  NNEvaluator* nnEval = bot->nnEvaluator;
  MiscNNInputParams nnInputParams;
  NNResultBuf buf;
  nnEval->evaluate(board, hist, nextPlayer, nnInputParams, buf, false);
  std::shared_ptr<NNOutput> nnOutput = std::move(buf.result);
  BoardValue value;
  if(nextPlayer == C_BLACK) {
    value.win = nnOutput->whiteLossProb;
    value.loss = nnOutput->whiteWinProb;
    value.draw = nnOutput->whiteNoResultProb;
  } else {
    value.win = nnOutput->whiteWinProb;
    value.loss = nnOutput->whiteLossProb;
    value.draw = nnOutput->whiteNoResultProb;
  }
  return value;
}

static bool tryInitializeRandomOpening(
  Search* botB,
  Search* botW,
  Board& board0,
  BoardHistory& hist0,
  Player& nextPlayer0,
  Rand& gameRand,
  const PlaySettings& playSettings) {

  Board board(board0);
  BoardHistory hist(hist0);
  Player nextPlayer(nextPlayer0);
    

  //random change some pieces
  double changeProb = playSettings.randomInitPieceDensity * gameRand.nextExponential();
  double pieceDensity = pow(gameRand.nextDouble(), 4.0) * 0.5;
  for(int x = 0; x < board.x_size; x++)
  {
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
    // random select a piece on board
    std::vector<Loc> allLocs;
    // find all non-empty locations
    for(int x = 0; x < board.x_size; x++)
      for(int y = 0; y < board.y_size; y++) {
        Loc loc = Location::getLoc(x, y, board.x_size);
        if(board.colors[loc] != C_EMPTY)
          allLocs.push_back(loc);
      }
    if(allLocs.size() == 0)
      break;
    Loc loc_from = allLocs[gameRand.nextUInt(allLocs.size())];
    Color c = board.colors[loc_from];
    Color pside = getPiecePla(c);
    Color ptype = getPieceType(c);
    int randX = gameRand.nextUInt(board.x_size);
    int randY = gameRand.nextUInt(board.y_size);
    Loc loc_to = Location::getLoc(randX, randY, board.x_size);
    if(loc_to == loc_from)
      continue;
    if(loc_to == GameLogic::getHomeLoc(C_BLACK) || loc_to == GameLogic::getHomeLoc(C_WHITE))
      continue;
    if(GameLogic::isInRiver(loc_to) && ptype != C_RAT)
      continue;
    if(GameLogic::isInTrap(loc_to, getOpp(pside)))
      continue;
    if((randY >= 6 && pside == C_WHITE) || (randY <= 2 && pside == C_BLACK))  // on opponent's side
      if(gameRand.nextBool(0.75))
        continue;
    board.setStone(loc_to, c);
    board.setStone(loc_from, C_EMPTY);
  }
      
  hist.clear(board,nextPlayer,hist0.rules);

  //check winrate
  BoardValue value = getBoardValue(botB, board, hist, nextPlayer);
  double extremeValue = std::max(value.win, std::max(value.loss, value.draw));
  double rejectRate=pow(extremeValue,4)*0.97;
  if(gameRand.nextBool(rejectRate))
    return false;
    
  board0=board;
  nextPlayer0=board.nextPla;
  hist0=hist;
  return true;
}

static bool tryInitializeRandomOpeningForMatch(
  Search* botB,
  Search* botW,
  Board& board0,
  BoardHistory& hist0,
  Player& nextPlayer0,
  Rand& gameRand,
  const PlaySettings& playSettings) {

  Board board(board0);
  BoardHistory hist(hist0);
  Player nextPlayer(nextPlayer0);

  if (gameRand.nextBool(0.5)) {
    board.setStone(Location::getLoc(0, 8, board.x_size), C_EMPTY);//remove a tiger
  }
  else {
    board.setStone(Location::getLoc(5, 7, board.x_size), C_EMPTY);//remove a dog,cat,and wolf
    board.setStone(Location::getLoc(1, 7, board.x_size), C_EMPTY);
    board.setStone(Location::getLoc(2, 6, board.x_size), C_EMPTY);
  }


  hist.clear(board, nextPlayer, hist0.rules);
  board0 = board;
  nextPlayer0 = board.nextPla;
  hist0 = hist;
  return true;
}

void RandomOpening::initializeRandomOpening(
    Search* botB,
    Search* botW,
    Board& board,
    BoardHistory& hist,
    Player& nextPlayer,
    Rand& gameRand,
    const PlaySettings& playSettings)
{
  triedCount++;
  succeedCount++;
  int count=0;
  while(
    playSettings.forSelfPlay?
    (!tryInitializeRandomOpening(botB,botW,board,hist,nextPlayer,gameRand,playSettings)):
    (!tryInitializeRandomOpeningForMatch(botB,botW,board,hist,nextPlayer,gameRand,playSettings))
    ) {
    triedCount++;
    count++;
    if(count>500) //exp(-500*0.03)
    {
      succeedCount -= 1;
      std::cout<<"RandomOpening failed"<<std::endl;
      break;
    }
  }
  if(gameRand.nextBool(0.001))
    std::cout << "Random opening Tried=" << triedCount << "   Succeed=" << succeedCount << std::endl;
}
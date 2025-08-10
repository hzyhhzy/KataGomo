#include "../game/randomopening.h"
#include "../core/rand.h"
#include "../game/gamelogic.h"
#include "../search/asyncbot.h"
#include "../program/play.h"
using namespace RandomOpening;
using namespace std;
std::atomic<int64_t> triedCount(0);
std::atomic<int64_t> succeedCount(0);
std::atomic<int64_t> evalCount(0);

static Loc getRandomNearbyMove(Board& board, Rand& gameRand, double avgDist) {
  int xsize = board.x_size, ysize = board.y_size;
  if(board.isEmpty()) {
    double middleBonusFactor = 1.5;
    double xd = gameRand.nextGaussianTruncated(middleBonusFactor * 0.999) / (2 * middleBonusFactor);
    double yd = gameRand.nextGaussianTruncated(middleBonusFactor * 0.999) / (2 * middleBonusFactor);
    int x = round(xd * xsize + 0.5 * (xsize - 1)), y = round(yd * ysize + 0.5 * (ysize - 1));
    if(x < 0 || x >= xsize || y < 0 || y >= ysize) {
      cout << x << xsize << y << ysize;
      ASSERT_UNREACHABLE;
    }
    Loc loc = Location::getLoc(x, y, xsize);
    return loc;
  }
  std::vector<double> prob(xsize * ysize, 0);
  for(int x1 = 0; x1 < xsize; x1++)
    for(int y1 = 0; y1 < ysize; y1++) {
      Loc loc = Location::getLoc(x1, y1, xsize);
      if(board.colors[loc] == C_EMPTY && loc != board.firstLoc)
        continue;
      for(int x2 = 0; x2 < xsize; x2++)
        for(int y2 = 0; y2 < ysize; y2++) {
          Loc loc2 = Location::getLoc(x2, y2, xsize);
          if(board.colors[loc2] != C_EMPTY || loc2 == board.firstLoc)
            continue;
          double middleBonusFactor = 1.5;
          double halfBoardLen = std::max(0.5 * (xsize - 1), 0.5 * (ysize - 1));
          double distFromCenter = std::max(std::abs(x2 - 0.5 * (xsize - 1)), std::abs(y2 - 0.5 * (ysize - 1)));
          double middleBonus = middleBonusFactor * (halfBoardLen - distFromCenter) / halfBoardLen;
          double prob_increase =
            (1 + middleBonus) * pow((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + avgDist * avgDist, -2);
          prob[y2 * xsize + x2] += prob_increase;
        }
    }

  double totalProb = 0;
  for(int x = 0; x < xsize; x++)
    for(int y = 0; y < ysize; y++) {
      totalProb += prob[y * xsize + x];
    }

  for(int x = 0; x < xsize; x++)
    for(int y = 0; y < ysize; y++) {
      prob[y * xsize + x] /= totalProb;
    }

  double randomDouble = gameRand.nextDouble() - 1e-8;

  double probSum = 0;
  for(int x = 0; x < xsize; x++)
    for(int y = 0; y < ysize; y++) {
      probSum += prob[y * xsize + x];
      if(probSum >= randomDouble) {
        return Location::getLoc(x, y, xsize);
      }
    }

  ASSERT_UNREACHABLE;
}
static double getBoardValue(Search* bot, const Board& board, const BoardHistory& hist, Player nextPlayer) {
  evalCount++;
  NNEvaluator* nnEval = bot->nnEvaluator;
  MiscNNInputParams nnInputParams;
  NNResultBuf buf;
  nnEval->evaluate(board, hist, nextPlayer, nnInputParams, buf, false);
  std::shared_ptr<NNOutput> nnOutput = std::move(buf.result);
  double value = nnOutput->whiteWinProb - nnOutput->whiteLossProb +
                 bot->searchParams.noResultUtilityForWhite * nnOutput->whiteNoResultProb;
  if(nextPlayer == C_BLACK)
    return -value;
  else
    return value;
}
static Loc getBalanceMove(
  Search* botB,
  Search* botW,
  const Board& board,
  const BoardHistory& hist,
  Player nextPlayer,
  Rand& gameRand,
  bool forSelfplay,
  double rejectProb) {
  int xsize = board.x_size, ysize = board.y_size;

  Search* bot = gameRand.nextBool(0.5) ? botB : botW;
  double maxProb = 0;

  double rootValuePla = getBoardValue(bot, board, hist, nextPlayer);
  if(rootValuePla < 0) {  // probably all moves are losing
    double rejectFactor = 1 - exp(-3 * rootValuePla * rootValuePla);
    if(gameRand.nextBool(rejectFactor) && gameRand.nextBool(rejectProb)) {
      return Board::NULL_LOC;
    }
  }

  std::vector<double> prob(xsize * ysize, 0);
  for(int x = 0; x < xsize; x++)
    for(int y = 0; y < ysize; y++) {
      Loc loc = Location::getLoc(x, y, xsize);

      if(!board.isLegal(loc, nextPlayer))
        continue;


      Board boardCopy(board);
      BoardHistory histCopy(hist);

      histCopy.makeBoardMoveAssumeLegal(boardCopy, loc, nextPlayer);
      if(histCopy.isGameFinished)
        continue;

      double value = getBoardValue(bot, boardCopy, histCopy, boardCopy.nextPla);

      double p = forSelfplay ? pow(1 - value * value, 4) : pow(1 - value * value, 10);
      maxProb = std::max(maxProb, p);
      prob[y * xsize + x] = p;
    }
  if(gameRand.nextBool(1 - maxProb) && gameRand.nextBool(rejectProb)) {
    return Board::NULL_LOC;
  }

  double totalProb = 0;
  for(int x = 0; x < xsize; x++)
    for(int y = 0; y < ysize; y++) {
      totalProb += prob[y * xsize + x];
    }
  for(int x = 0; x < xsize; x++)
    for(int y = 0; y < ysize; y++) {
      prob[y * xsize + x] /= totalProb;
    }

  double randomDouble = gameRand.nextDouble() - 1e-8;

  double probSum = 0;
  for(int x = 0; x < xsize; x++)
    for(int y = 0; y < ysize; y++) {
      probSum += prob[y * xsize + x];
      if(probSum >= randomDouble) {
        return Location::getLoc(x, y, xsize);
      }
    }

  // some rare conditions, return NULL_LOC.
  std::cout << "totalProb=" << totalProb << ", probSum=" << probSum
            << " in getBalanceMove(), Rule=" << hist.rules.toString() << std::endl;
  return Board::NULL_LOC;
  /*while (1)
  {
    int x = gameRand.nextUInt(xsize);
    int y = gameRand.nextUInt(ysize);
    Loc loc=Location::getLoc(x, y, xsize);
    if (board.isLegal(loc, nextPlayer, true))
      return loc;
  }
  ASSERT_UNREACHABLE;*/
}
static bool tryInitializeBalancedRandomOpening(
  Search* botB,
  Search* botW,
  Board& board,
  BoardHistory& hist,
  Player& nextPlayer,
  Rand& gameRand,
  bool forSelfplay,
  double rejectProb) {
  Board boardCopy(board);
  BoardHistory histCopy(hist);
  Player nextPlayerCopy = nextPlayer;

  triedCount++;

  std::vector<float> randomMoveNumProb;

  if(hist.rules.VCNRule == Rules::VCNRULE_NOVC)
    randomMoveNumProb = vector<float>{10, 30, 50, 80, 60, 40, 20, 10, 5, 1, 0, 0};
  else if(hist.rules.VCNRule == Rules::VCNRULE_VC2_B)
    randomMoveNumProb = vector<float>{0.01, 0.01, 10, 20, 35, 30, 25, 20, 15, 10, 5, 1};
  else if(hist.rules.VCNRule == Rules::VCNRULE_VC2_W)
    randomMoveNumProb = vector<float>{0.01, 0.01, 0.03, 0.03, 15, 30, 25, 20, 15, 10, 5, 1};
  else
    cout << Rules::writeVCNRule(hist.rules.VCNRule) << " does not support balanced openings init" << endl;
  int maxRandomMoveNum = randomMoveNumProb.size();

  static const double avgRandomDistFactor = 0.8;

  double randomMoveNumProbTotal = 0;
  for(int i = 0; i < maxRandomMoveNum; i++)
    randomMoveNumProbTotal += randomMoveNumProb[i];
  double randomMoveNumProbSum = 0;
  double randomMoveNumProbRandomDouble = gameRand.nextDouble() * randomMoveNumProbTotal - 1e-7;
  int randomMoveNum = -1;
  for(int i = 0; i < maxRandomMoveNum; i++) {
    randomMoveNumProbSum += randomMoveNumProb[i];
    if(randomMoveNumProbSum >= randomMoveNumProbRandomDouble) {
      randomMoveNum = i;
      break;
    }
  }
  if(randomMoveNum == -1)
    ASSERT_UNREACHABLE;

  double avgDist = gameRand.nextExponential() * avgRandomDistFactor;
  for(int i = 0; i < randomMoveNum; i++) {
    Loc randomLoc = getRandomNearbyMove(boardCopy, gameRand, avgDist);
    assert(boardCopy.isLegal(randomLoc, nextPlayerCopy));
    histCopy.makeBoardMoveAssumeLegal(boardCopy, randomLoc, nextPlayerCopy);
    if(histCopy.isGameFinished)
      return false;
    nextPlayerCopy = boardCopy.nextPla;
  }
  Loc balancedMove = getBalanceMove(botB, botW, boardCopy, histCopy, nextPlayerCopy, gameRand, forSelfplay, rejectProb);
  if(balancedMove == Board::NULL_LOC)
    return false;
  histCopy.makeBoardMoveAssumeLegal(boardCopy, balancedMove, nextPlayerCopy);
  if(histCopy.isGameFinished)
    return false;
  nextPlayerCopy = boardCopy.nextPla;

  board = boardCopy;
  hist = histCopy;
  nextPlayer = nextPlayerCopy;
  return true;
}
void RandomOpening::initializeBalancedRandomOpening(
  Search* botB,
  Search* botW,
  Board& board,
  BoardHistory& hist,
  Player& nextPlayer,
  Rand& gameRand,
  bool forSelfplay) {
  static const int maxTryTimes = 20;
  int tryTimes = 0;
  double rejectProb = 0.995;
  while(!tryInitializeBalancedRandomOpening(botB, botW, board, hist, nextPlayer, gameRand, forSelfplay, rejectProb)) {
    tryTimes++;
    if(tryTimes > maxTryTimes) {
      tryTimes = 0;
      std::cout << "Reached max trying times for finding balanced openings, Rule=" << hist.rules.toString()
                << std::endl;
      rejectProb = 0.8;
    }
  }
  succeedCount++;
  // if(succeedCount%500==0) {
  //   cout << "Generated " << succeedCount << " openings,"
  //        << " tried " << triedCount << ", nneval " << evalCount << endl;
  //  }
}
void RandomOpening::initializeSpecialOpening(
  Board& board,
  BoardHistory& hist,
  Rules& rules,
  Player& nextPlayer,
  OtherGameProperties& otherGameProps,
  Rand& gameRand) {
  if(board.numStonesOnBoard() != 0)
    throw StringError("Board should be empty before calling RandomOpening::initializeBalancedRandomOpening");
  if(board.x_size != 19 || board.y_size != 19)
    return;
  rules = Rules();
  otherGameProps.isOpeningPos = true;

  if(gameRand.nextBool(0.8)) {
    int t = (109 - 1) / 4 + int(2.0 * gameRand.nextGaussianTruncated(6) + 0.5);
    rules.maxMoves = t * 4 + 1;  // mostly 105,109 and 113
  }

  int r = gameRand.nextUInt(100);
  if(r < 8)  // main branch of J10 K11 J8
  {
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(9, 9, board.x_size), board.nextPla);

    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(10, 10, board.x_size), board.nextPla);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(9, 7, board.x_size), board.nextPla);

    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(9, 11, board.x_size), board.nextPla);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(10, 11, board.x_size), board.nextPla);

    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(8, 11, board.x_size), board.nextPla);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(9, 12, board.x_size), board.nextPla);

    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(8, 9, board.x_size), board.nextPla);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(7, 10, board.x_size), board.nextPla);

    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(7, 9, board.x_size), board.nextPla);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(10, 8, board.x_size), board.nextPla);

    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(8, 6, board.x_size), board.nextPla);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(10, 6, board.x_size), board.nextPla);
    
  }
  else if(r < 23)  // main branch of J10 K11 J8 (shorter)
  {
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(9, 9, board.x_size), board.nextPla);

    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(10, 10, board.x_size), board.nextPla);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(9, 7, board.x_size), board.nextPla);

    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(9, 11, board.x_size), board.nextPla);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(10, 11, board.x_size), board.nextPla);

    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(8, 11, board.x_size), board.nextPla);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(9, 12, board.x_size), board.nextPla);

    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(8, 9, board.x_size), board.nextPla);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(7, 10, board.x_size), board.nextPla);

  } 
  else if(r < 31)  // main branch of J10 K11 I11
  {
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(9, 9, board.x_size), board.nextPla);

    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(10, 10, board.x_size), board.nextPla);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(8, 10, board.x_size), board.nextPla);

    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(9, 10, board.x_size), board.nextPla);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(8, 12, board.x_size), board.nextPla);

    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(9, 12, board.x_size), board.nextPla);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(7, 9, board.x_size), board.nextPla);

    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(9, 11, board.x_size), board.nextPla);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(7, 13, board.x_size), board.nextPla);

    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(8, 11, board.x_size), board.nextPla);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(10, 13, board.x_size), board.nextPla);

    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(10, 14, board.x_size), board.nextPla);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(11, 14, board.x_size), board.nextPla);

    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(9, 14, board.x_size), board.nextPla);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(6, 10, board.x_size), board.nextPla);

    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(7, 10, board.x_size), board.nextPla);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(8, 8, board.x_size), board.nextPla);

  } 
  else if(r < 33)  // a blind spot branch of J10 K11 I11 
  {
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(9, 9, board.x_size), board.nextPla);

    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(10, 10, board.x_size), board.nextPla);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(8, 10, board.x_size), board.nextPla);

    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(9, 10, board.x_size), board.nextPla);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(8, 12, board.x_size), board.nextPla);

    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(9, 12, board.x_size), board.nextPla);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(9, 8, board.x_size), board.nextPla);

    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(8, 14, board.x_size), board.nextPla);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(10, 14, board.x_size), board.nextPla);

    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(9, 13, board.x_size), board.nextPla);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(8, 13, board.x_size), board.nextPla);

    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(9, 14, board.x_size), board.nextPla);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(7, 14, board.x_size), board.nextPla);

    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(11, 14, board.x_size), board.nextPla);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(5, 14, board.x_size), board.nextPla);

    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(10, 13, board.x_size), board.nextPla);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(10, 9, board.x_size), board.nextPla);

    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(11, 12, board.x_size), board.nextPla);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(12, 9, board.x_size), board.nextPla);

    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(7, 12, board.x_size), board.nextPla);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(7, 13, board.x_size), board.nextPla);

    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(6, 12, board.x_size), board.nextPla);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(7, 15, board.x_size), board.nextPla);

    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(10, 16, board.x_size), board.nextPla);
    if(gameRand.nextBool(0.5)) {
      hist.makeBoardMoveAssumeLegal(board, Location::getLoc(11, 8, board.x_size), board.nextPla);

      hist.makeBoardMoveAssumeLegal(board, Location::getLoc(10, 12, board.x_size), board.nextPla);
      hist.makeBoardMoveAssumeLegal(board, Location::getLoc(8, 11, board.x_size), board.nextPla);

      hist.makeBoardMoveAssumeLegal(board, Location::getLoc(11, 13, board.x_size), board.nextPla);
      if(gameRand.nextBool(0.5)) {
        hist.makeBoardMoveAssumeLegal(board, Location::getLoc(11, 7, board.x_size), board.nextPla);
      }
    }



  } 
  else if(r < 50)  // J10 K11 I11 J11 I13
  {
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(9, 9, board.x_size), board.nextPla);

    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(10, 10, board.x_size), board.nextPla);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(8, 10, board.x_size), board.nextPla);

    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(9, 10, board.x_size), board.nextPla);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(8, 12, board.x_size), board.nextPla);

  } 
  else   // Only first move on J10
  {
    hist.makeBoardMoveAssumeLegal(board,Location::getLoc(9, 9, board.x_size), board.nextPla);
  }
  nextPlayer = board.nextPla;
  hist.clear(board, nextPlayer, rules);
}

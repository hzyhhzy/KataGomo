#include "../game/boardhistory.h"
#include "../game/gamelogic.h"
#include <algorithm>

using namespace std;



BoardHistory::BoardHistory()
  :rules(),
   moveHistory(),
   initialBoard(),
   initialPla(P_BLACK),
   initialTurnNumber(0),
   recentBoards(),
   currentRecentBoardIdx(0),
   presumedNextMovePla(P_BLACK),
    isGameFinished(false),
    winner(C_EMPTY),
    finalScoreBlack(0.0),
   isNoResult(false),isResignation(false)
{
}

BoardHistory::~BoardHistory()
{}

BoardHistory::BoardHistory(const Board& board, Player pla, const Rules& r)
  :rules(r),
   moveHistory(),
   initialBoard(),
   initialPla(),
   initialTurnNumber(0),
   recentBoards(),
   currentRecentBoardIdx(0),
   presumedNextMovePla(pla),
    isGameFinished(false),
    winner(C_EMPTY),
    finalScoreBlack(0.0),
   isNoResult(false),isResignation(false)
{

  clear(board,pla,rules);
}

BoardHistory::BoardHistory(const BoardHistory& other)
  :rules(other.rules),
   moveHistory(other.moveHistory),
   initialBoard(other.initialBoard),
   initialPla(other.initialPla),
   initialTurnNumber(other.initialTurnNumber),
   recentBoards(),
   currentRecentBoardIdx(other.currentRecentBoardIdx),
   presumedNextMovePla(other.presumedNextMovePla),
   isGameFinished(other.isGameFinished),winner(other.winner),finalScoreBlack(other.finalScoreBlack),
   isNoResult(other.isNoResult),isResignation(other.isResignation)
{
  std::copy(other.recentBoards, other.recentBoards+NUM_RECENT_BOARDS, recentBoards);
}


BoardHistory& BoardHistory::operator=(const BoardHistory& other)
{
  if(this == &other)
    return *this;
  rules = other.rules;
  moveHistory = other.moveHistory;
  initialBoard = other.initialBoard;
  initialPla = other.initialPla;
  initialTurnNumber = other.initialTurnNumber;
  std::copy(other.recentBoards, other.recentBoards+NUM_RECENT_BOARDS, recentBoards);
  currentRecentBoardIdx = other.currentRecentBoardIdx;
  presumedNextMovePla = other.presumedNextMovePla;
  isGameFinished = other.isGameFinished;
  winner = other.winner;
  finalScoreBlack = other.finalScoreBlack;
  isNoResult = other.isNoResult;
  isResignation = other.isResignation;

  return *this;
}

BoardHistory::BoardHistory(BoardHistory&& other) noexcept
 :rules(other.rules),
  moveHistory(std::move(other.moveHistory)),
  initialBoard(other.initialBoard),
  initialPla(other.initialPla),
  initialTurnNumber(other.initialTurnNumber),
  recentBoards(),
  currentRecentBoardIdx(other.currentRecentBoardIdx),
  presumedNextMovePla(other.presumedNextMovePla),
    isGameFinished(other.isGameFinished),
    winner(other.winner),
    finalScoreBlack(other.finalScoreBlack),
  isNoResult(other.isNoResult),isResignation(other.isResignation)
{
  std::copy(other.recentBoards, other.recentBoards+NUM_RECENT_BOARDS, recentBoards);
}

BoardHistory& BoardHistory::operator=(BoardHistory&& other) noexcept
{
  rules = other.rules;
  moveHistory = std::move(other.moveHistory);
  initialBoard = other.initialBoard;
  initialPla = other.initialPla;
  initialTurnNumber = other.initialTurnNumber;
  std::copy(other.recentBoards, other.recentBoards+NUM_RECENT_BOARDS, recentBoards);
  currentRecentBoardIdx = other.currentRecentBoardIdx;
  presumedNextMovePla = other.presumedNextMovePla;
  isGameFinished = other.isGameFinished;
  winner = other.winner;
  finalScoreBlack = other.finalScoreBlack;
  isNoResult = other.isNoResult;
  isResignation = other.isResignation;

  return *this;
}

void BoardHistory::clear(const Board& board, Player pla, const Rules& r) {
  rules = r;
  moveHistory.clear();

  initialBoard = board;
  initialPla = pla;
  initialTurnNumber = 0;

  //This makes it so that if we ask for recent boards with a lookback beyond what we have a history for,
  //we simply return copies of the starting board.
  for(int i = 0; i<NUM_RECENT_BOARDS; i++)
    recentBoards[i] = board;
  currentRecentBoardIdx = 0;

  presumedNextMovePla = pla;


  isGameFinished = false;
  winner = C_EMPTY;
  finalScoreBlack = 0;
  isNoResult = false;
  isResignation = false;

}

BoardHistory BoardHistory::copyToInitial() const {
  BoardHistory hist(initialBoard, initialPla, rules);
  hist.setInitialTurnNumber(initialTurnNumber);
  return hist;
}

void BoardHistory::setInitialTurnNumber(int n) {
  initialTurnNumber = n;
}

void BoardHistory::printBasicInfo(ostream& out, const Board& board) const {
  Board::printBoard(out, board, Board::NULL_LOC, &moveHistory);
  out << "Next player: " << PlayerIO::playerToString(presumedNextMovePla) << endl;
  out << "Rules: " << rules.toJsonString() << endl;
  out << "White 7-3 rule history: ";
  auto w73 = get73ruleHistory(board, P_WHITE);
  for(int i = 0; i < w73.size(); i++)
    out << Location::toString(w73[i], board) << " ";
  out << endl;
  out << "Black 7-3 rule history: ";
  auto b73 = get73ruleHistory(board, P_BLACK);
  for(int i = 0; i < b73.size(); i++)
    out << Location::toString(b73[i], board) << " ";
  out << endl;
}

void BoardHistory::printDebugInfo(ostream& out, const Board& board) const {
  out << board << endl;
  out << "Initial pla " << PlayerIO::playerToString(initialPla) << endl;
  out << "Rules " << rules << endl;
  out << "Presumed next pla " << PlayerIO::playerToString(presumedNextMovePla) << endl;
  out << "Game result " << isGameFinished << " " << PlayerIO::playerToString(winner) << " " << finalScoreBlack << " "
      << isNoResult << " " << isResignation << endl;
  out << "Last moves ";
  for(int i = 0; i<moveHistory.size(); i++)
    out << Location::toString(moveHistory[i].loc,board) << " ";
  out << endl;
}

std::vector<Loc> BoardHistory::get73ruleHistory(const Board& board, Player pla, int maxLen) const {
  std::vector<Loc> h;
  int t = moveHistory.size() - 1;  // which move is pla's last move
  if(pla == board.nextPla && board.stage == 1) {
    t = moveHistory.size() - 4;
    if(t < 0)
      return h;
    assert(moveHistory[t].pla == pla);
    if(moveHistory[t].loc != board.midLocs[0])
      return h;//chosen a different stone, ignore the history
  }
  else if(pla != board.nextPla && board.stage == 0)
    t = moveHistory.size() - 1;
  else if(pla != board.nextPla && board.stage == 1)
    t = moveHistory.size() - 2;
  else if(pla == board.nextPla && board.stage == 0)
    t = moveHistory.size() - 3;

  if(t < 0)
    return h;

  Loc nowloc = moveHistory[t].loc;

  while (h.size() < maxLen && t >= 0) {
    if(!board.isOnBoard(nowloc))
      return h;
    if(GameLogic::isInTrap(nowloc, pla))
      return h;
    assert(moveHistory[t].pla == pla);
    assert(t - 1 >= 0 && moveHistory[t - 1].pla == pla);
    if(nowloc == moveHistory[t].loc) {
      h.push_back(moveHistory[t].loc);
      nowloc = moveHistory[t - 1].loc;
    } else
      break;
    t -= 4;
  }

  

  return h;
}

std::vector<Loc> BoardHistory::get73ruleHistory(const Board& board, Player pla) const {
  const int maxLen = 7;
  return get73ruleHistory(board, pla, maxLen);
}

double BoardHistory::calculateScoreBlackWhenDraw(const Board& board) const {
  if(rules.drawJudgeRule == rules.DRAWJUDGE_DRAW)
    return 0;
  else if(rules.drawJudgeRule == rules.DRAWJUDGE_COUNT) {
    double s = 0;
    for(int y = 0; y < board.y_size; y++)
      for(int x = 0; x < board.x_size; x++) {
        Loc loc = Location::getLoc(x, y, board.x_size);
        Color c = board.colors[loc];
        if(c == C_EMPTY)
          continue;
        Color cp = getPiecePla(c);
        Color ct = getPieceType(c);
        if(cp == C_BLACK)
          s += 1;
        else
          s -= 1;
      }
    return 0.3 * s;
  } 
  else if(rules.drawJudgeRule == rules.DRAWJUDGE_WEIGHT) {
    double s = 0;
    for(int y = 0; y < board.y_size; y++)
      for(int x = 0; x < board.x_size; x++) {
        Loc loc = Location::getLoc(x, y, board.x_size);
        Color c = board.colors[loc];
        if(c == C_EMPTY)
          continue;
        Color cp = getPiecePla(c);
        Color ct = getPieceType(c);

        double w = 1;
        if(ct == C_ELEPHANT)
          w = 4;
        else if(ct == C_LION)
          w = 4;
        else if(ct == C_RAT)
          w = 3;
        else if(ct == C_TIGER)
          w = 2;

        if(cp == C_BLACK)
          s += w;
        else
          s -= w;
      }
    return 0.15 * s;
  } 
  else
    ASSERT_UNREACHABLE;
  return 0;
}


const Board& BoardHistory::getRecentBoard(int numMovesAgo) const {
  assert(numMovesAgo >= 0 && numMovesAgo < NUM_RECENT_BOARDS);
  int idx = (currentRecentBoardIdx - numMovesAgo + NUM_RECENT_BOARDS) % NUM_RECENT_BOARDS;
  return recentBoards[idx];
}





void BoardHistory::setWinnerByResignation(Player pla) {
  isGameFinished = true;
  isNoResult = false;
  isResignation = true;
  winner = pla;
  finalScoreBlack = 0.0;
}

void BoardHistory::setWinner(Player pla, double bscore) {
  isGameFinished = true;
  isNoResult = false;
  isResignation = false;
  winner = pla;
  if(winner == C_EMPTY)
    finalScoreBlack = bscore;
  else
    finalScoreBlack = 0;
}

bool BoardHistory::isLegal(const Board& board, Loc moveLoc, Player movePla) const {
  if(!board.isLegal(moveLoc,movePla,rules))
    return false;

  return true;
}




bool BoardHistory::isLegalTolerant(const Board& board, Loc moveLoc, Player movePla) const {
  return board.isLegal(moveLoc, movePla, rules);
}
bool BoardHistory::makeBoardMoveTolerant(Board& board, Loc moveLoc, Player movePla) {
  if(!board.isLegal(moveLoc, movePla, rules))
    return false;
  makeBoardMoveAssumeLegal(board,moveLoc,movePla);
  return true;
}


void BoardHistory::makeBoardMoveAssumeLegal(Board& board, Loc moveLoc, Player movePla) {

  //If somehow we're making a move after the game was ended, just clear those values and continue
  isGameFinished = false;
  winner = C_EMPTY;
  finalScoreBlack = 0.0;
  isNoResult = false;
  isResignation = false;

  board.playMoveAssumeLegal(moveLoc,movePla);

  

  //Update recent boards
  currentRecentBoardIdx = (currentRecentBoardIdx + 1) % NUM_RECENT_BOARDS;
  recentBoards[currentRecentBoardIdx] = board;

  moveHistory.push_back(Move(moveLoc,movePla));
  presumedNextMovePla = board.nextPla;
  Color maybeWinner = GameLogic::checkWinnerAfterPlayed(board, *this, movePla, moveLoc);
  if(maybeWinner!=C_WALL) { //game finished
    double s = 0;
    if (maybeWinner == C_EMPTY)
    {
      //drawJudge rule
      s = calculateScoreBlackWhenDraw(board);
    }
    setWinner(maybeWinner, s);
  }

}



Hash128 BoardHistory::getSituationRulesHash(const Board& board, const BoardHistory& hist, Player nextPlayer) {
 //Note that board.pos_hash also incorporates the size of the board.
  Hash128 hash = board.pos_hash;
  hash ^= Board::ZOBRIST_PLAYER_HASH[nextPlayer];

  //Fold in the ko, scoring, and suicide rules
  hash ^= Rules::ZOBRIST_SCORING_RULE_HASH[hist.rules.scoringRule];
  hash ^= Rules::ZOBRIST_DRAWJUDGE_RULE_HASH[hist.rules.drawJudgeRule];
  hash ^= Board::ZOBRIST_MM_RULE_HASH[hist.rules.maxmoves];
  hash ^= Board::ZOBRIST_MC_RULE_HASH[hist.rules.maxmovesNoCapture];

  auto h1 = hist.get73ruleHistory(board, C_BLACK);
  auto h2 = hist.get73ruleHistory(board, C_WHITE);
  for (int i = 0; i < h1.size(); i++)
  {
    hash ^= Board::ZOBRIST_73RULE_HISTORY_HASH[h1[i]][i][C_BLACK];
  }
  for(int i = 0; i < h2.size(); i++) {
    hash ^= Board::ZOBRIST_73RULE_HISTORY_HASH[h2[i]][i][C_WHITE];
  }


  return hash;
}



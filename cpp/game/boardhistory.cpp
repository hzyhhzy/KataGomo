#include "../game/boardhistory.h"
#include "../game/gamelogic.h"
#include <algorithm>

using namespace std;

static Hash128 repetitionCountContribution(Hash128 posHash, int count) {
  if(count <= 0)
    return Hash128();
  uint64_t countBits = static_cast<uint64_t>(count) * 0x9e3779b97f4a7c15ULL;
  return Hash128(
    Hash::murmurMix(posHash.hash0 ^ countBits),
    Hash::nasam(posHash.hash1 + countBits)
  );
}

static void clearRepetitionHistory(BoardHistory& hist) {
  hist.posHashHistoryCount.clear();
  hist.repetitionHistoryHash = Hash128();
}

static void incrementRepetitionHistory(BoardHistory& hist, Hash128 posHash) {
  int oldCount = 0;
  auto iter = hist.posHashHistoryCount.find(posHash);
  if(iter != hist.posHashHistoryCount.end())
    oldCount = iter->second;
  hist.repetitionHistoryHash ^= repetitionCountContribution(posHash, oldCount);
  int newCount = oldCount + 1;
  hist.posHashHistoryCount[posHash] = newCount;
  hist.repetitionHistoryHash ^= repetitionCountContribution(posHash, newCount);
}



BoardHistory::BoardHistory()
  :rules(),
   moveHistory(),
   posHashHistoryCount(),
   repetitionHistoryHash(),
   initialBoard(),
   initialPla(P_BLACK),
   initialTurnNumber(0),
   recentBoards(),
   currentRecentBoardIdx(0),
   presumedNextMovePla(P_BLACK),
   isGameFinished(false),winner(C_EMPTY),
   isNoResult(false),isResignation(false)
{
}

BoardHistory::~BoardHistory()
{}

BoardHistory::BoardHistory(const Board& board, Player pla, const Rules& r)
  :rules(r),
   moveHistory(),
   posHashHistoryCount(),
   repetitionHistoryHash(),
   initialBoard(),
   initialPla(),
   initialTurnNumber(0),
   recentBoards(),
   currentRecentBoardIdx(0),
   presumedNextMovePla(pla),
   isGameFinished(false),winner(C_EMPTY),
   isNoResult(false),isResignation(false)
{

  clear(board,pla,rules);
}

BoardHistory::BoardHistory(const BoardHistory& other)
  :rules(other.rules),
   moveHistory(other.moveHistory),
   posHashHistoryCount(other.posHashHistoryCount),
   repetitionHistoryHash(other.repetitionHistoryHash),
   initialBoard(other.initialBoard),
   initialPla(other.initialPla),
   initialTurnNumber(other.initialTurnNumber),
   recentBoards(),
   currentRecentBoardIdx(other.currentRecentBoardIdx),
   presumedNextMovePla(other.presumedNextMovePla),
   isGameFinished(other.isGameFinished),winner(other.winner),
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
  posHashHistoryCount = other.posHashHistoryCount;
  repetitionHistoryHash = other.repetitionHistoryHash;
  initialBoard = other.initialBoard;
  initialPla = other.initialPla;
  initialTurnNumber = other.initialTurnNumber;
  std::copy(other.recentBoards, other.recentBoards+NUM_RECENT_BOARDS, recentBoards);
  currentRecentBoardIdx = other.currentRecentBoardIdx;
  presumedNextMovePla = other.presumedNextMovePla;
  isGameFinished = other.isGameFinished;
  winner = other.winner;
  isNoResult = other.isNoResult;
  isResignation = other.isResignation;

  return *this;
}

BoardHistory::BoardHistory(BoardHistory&& other) noexcept
 :rules(other.rules),
  moveHistory(std::move(other.moveHistory)),
  posHashHistoryCount(std::move(other.posHashHistoryCount)),
  repetitionHistoryHash(other.repetitionHistoryHash),
  initialBoard(other.initialBoard),
  initialPla(other.initialPla),
  initialTurnNumber(other.initialTurnNumber),
  recentBoards(),
  currentRecentBoardIdx(other.currentRecentBoardIdx),
  presumedNextMovePla(other.presumedNextMovePla),
  isGameFinished(other.isGameFinished),winner(other.winner),
  isNoResult(other.isNoResult),isResignation(other.isResignation)
{
  std::copy(other.recentBoards, other.recentBoards+NUM_RECENT_BOARDS, recentBoards);
}

BoardHistory& BoardHistory::operator=(BoardHistory&& other) noexcept
{
  rules = other.rules;
  moveHistory = std::move(other.moveHistory);
  posHashHistoryCount = std::move(other.posHashHistoryCount);
  repetitionHistoryHash = other.repetitionHistoryHash;
  initialBoard = other.initialBoard;
  initialPla = other.initialPla;
  initialTurnNumber = other.initialTurnNumber;
  std::copy(other.recentBoards, other.recentBoards+NUM_RECENT_BOARDS, recentBoards);
  currentRecentBoardIdx = other.currentRecentBoardIdx;
  presumedNextMovePla = other.presumedNextMovePla;
  isGameFinished = other.isGameFinished;
  winner = other.winner;
  isNoResult = other.isNoResult;
  isResignation = other.isResignation;

  return *this;
}

void BoardHistory::clear(const Board& board, Player pla, const Rules& r) {
  rules = r;
  moveHistory.clear();
  clearRepetitionHistory(*this);
  if(board.stage == 0)
    incrementRepetitionHistory(*this, board.pos_hash);

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
  isNoResult = false;
  isResignation = false;

  if(board.stage == 0 && !GameLogic::hasLegalMoveAssumeStage0(board, pla)) {
    isGameFinished = true;
    winner = GameLogic::getWinnerForNoLegalMoves(board, rules, pla);
  }

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
}

void BoardHistory::printDebugInfo(ostream& out, const Board& board) const {
  out << board << endl;
  out << "Initial pla " << PlayerIO::playerToString(initialPla) << endl;
  out << "Rules " << rules << endl;
  out << "Presumed next pla " << PlayerIO::playerToString(presumedNextMovePla) << endl;
  out << "Game result " << isGameFinished << " " << PlayerIO::playerToString(winner) << " "
      << isNoResult << " " << isResignation << endl;
  out << "Last moves ";
  for(int i = 0; i<moveHistory.size(); i++)
    out << Location::toString(moveHistory[i].loc,board) << " ";
  out << endl;
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
}

void BoardHistory::setWinner(Player pla) {
  isGameFinished = true;
  isNoResult = false;
  isResignation = false;
  winner = pla;
}

bool BoardHistory::isLegal(const Board& board, Loc moveLoc, Player movePla) const {
  if(!board.isLegal(moveLoc,movePla))
    return false;

  return true;
}




bool BoardHistory::isLegalTolerant(const Board& board, Loc moveLoc, Player movePla) const {
  return board.isLegal(moveLoc, movePla);
}
bool BoardHistory::makeBoardMoveTolerant(Board& board, Loc moveLoc, Player movePla) {
  if(!board.isLegal(moveLoc,movePla))
    return false;
  makeBoardMoveAssumeLegal(board,moveLoc,movePla);
  return true;
}

void BoardHistory::makeBoardMoveAssumeLegal(Board& board, Loc moveLoc, Player movePla) {

  //If somehow we're making a move after the game was ended, just clear those values and continue
  isGameFinished = false;
  winner = C_EMPTY;
  isNoResult = false;
  isResignation = false;

  bool completesMove = moveLoc == Board::PASS_LOC || board.stage == 1;
  bool isCapture =
    board.stage == 1 &&
    moveLoc != Board::PASS_LOC &&
    board.isOnBoard(moveLoc) &&
    board.colors[moveLoc] == getOpp(movePla);

  board.playMoveAssumeLegal(moveLoc,movePla);

  if(completesMove) {
    if(isCapture)
      clearRepetitionHistory(*this);
    incrementRepetitionHistory(*this, board.pos_hash);
  }

  //Update recent boards
  currentRecentBoardIdx = (currentRecentBoardIdx + 1) % NUM_RECENT_BOARDS;
  recentBoards[currentRecentBoardIdx] = board;

  moveHistory.push_back(Move(moveLoc,movePla));
  presumedNextMovePla = board.nextPla;

  Color maybeWinner = GameLogic::checkWinnerAfterPlayed(board, *this, movePla, moveLoc);
  if(maybeWinner!=C_WALL) { //game finished
    setWinner(maybeWinner);
  }

}



Hash128 BoardHistory::getSituationRulesHash(const Board& board, const BoardHistory& hist, Player nextPlayer) {
  // board.pos_hash incorporates board size, stones, stage, selected source,
  // and the side to move. Move count and repetition history are deliberately
  // kept outside board.pos_hash so that board.pos_hash remains suitable as a
  // repetition key.
  Hash128 hash = board.pos_hash;
  hash ^= Board::ZOBRIST_PLAYER_HASH[nextPlayer];
  hash ^= Rules::ZOBRIST_NO_LEGAL_MOVE_RULE_HASH[hist.rules.noLegalMoveRule];
  hash ^= Rules::ZOBRIST_REPETITION_COUNT_HASH[hist.rules.repetitionCount];
  hash ^= hist.repetitionHistoryHash;

  uint64_t moveBits = static_cast<uint64_t>(board.movenum);
  uint64_t maxMoveBits = static_cast<uint64_t>(hist.rules.maxMoves);
  hash.hash0 ^= Hash::murmurMix(moveBits ^ (maxMoveBits << 32));
  hash.hash1 ^= Hash::nasam((moveBits << 32) ^ maxMoveBits);
  return hash;
}



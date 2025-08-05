#include "../game/boardhistory.h"
#include "../game/gamelogic.h"
#include <algorithm>

using namespace std;



BoardHistory::BoardHistory()
  :rules(),
   moveHistory(),
   hashHistory(),
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
   hashHistory(),
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
   hashHistory(other.hashHistory),
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
  hashHistory = other.hashHistory;
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
  hashHistory(std::move(other.hashHistory)),
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
  hashHistory = std::move(other.hashHistory);
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
  hashHistory.clear();


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
  out << "White loop rule history: ";
  auto w73 = getLoopRuleHistory(board, P_WHITE);
  for(int i = 0; i < w73.size(); i++)
    out << Location::toString(w73[i], board) << " ";
  out << endl;
  out << "Black loop rule history: ";
  auto b73 = getLoopRuleHistory(board, P_BLACK);
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
  out << "Hash history ";
  for(int i = 0; i<hashHistory.size(); i++)
    out << hashHistory[i] << " ";
  out << endl;
}

std::vector<Loc> BoardHistory::get73RuleHistory(const Board& board, Player pla, int maxLen) const {
  if(maxLen==0)
    return std::vector<Loc>();
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

std::vector<Loc> BoardHistory::getLoopRuleHistory(const Board& board, Player pla) const {
  if(rules.loopRule == rules.LOOPRULE_SEVENTHREE)
    return get73RuleHistory(board, pla, 7);
  else if(rules.loopRule == rules.LOOPRULE_TWOONE)
    return get73RuleHistory(board, pla, 2);
  else if(rules.loopRule == rules.LOOPRULE_FIVETWO)
    return get73RuleHistory(board, pla, 5);
  else if(rules.loopRule == rules.LOOPRULE_NONE)
    return std::vector<Loc>();
  else if(rules.loopRule == rules.LOOPRULE_REPEATEND)
  {
    //last 20 pla's locs of movehistory
    std::vector<Loc> h;
    for(int i=moveHistory.size()-1;i>=0;i-=1)
    {
      if(moveHistory[i].pla!=pla)
        continue;
      h.push_back(moveHistory[i].loc);
      if(h.size()>=20)
        break;
    }
    return h;

  }
  else
    ASSERT_UNREACHABLE;
}
/*
std::vector<Loc> BoardHistory::getLastPieceMoveHistory(const Board& board, int maxTurn) const {
  if(maxTurn<2)
    assert(false);
  std::vector<Loc> h;
  if(moveHistory.size()<4)
    return h;
  Loc myPickedPiece=Board::NULL_LOC;
  Loc oppPickedPiece=Board::NULL_LOC;

  int t = moveHistory.size() - 1;  // which move is pla's last move
  if(board.stage==1)//my side only have half move
  {
    myPickedPiece=moveHistory[t].loc;
    t-=1;
    oppPickedPiece=moveHistory[t].loc;
    h.push_back(oppPickedPiece);
    t-=1;
    oppPickedPiece=moveHistory[t].loc;
    t-=1;//now movehistory[t] is the my last full move
    h.push_back(myPickedPiece);
    h.push_back(oppPickedPiece);
  }
  else
  {
    myPickedPiece=moveHistory[t].loc;
    h.push_back(myPickedPiece);
    t-=1;
    myPickedPiece=moveHistory[t].loc;
    t-=1;

    oppPickedPiece=moveHistory[t].loc;
    h.push_back(oppPickedPiece);
    t-=1;
    oppPickedPiece=moveHistory[t].loc;
    t-=1;//now movehistory[t] is the my last full move
    h.push_back(myPickedPiece);
    h.push_back(oppPickedPiece);

  }
  maxTurn-=2;
  
  while(maxTurn>0 && t>=3)
  {
    maxTurn-=1;
    assert(moveHistory[t].pla == pla);
    if(myPickedPiece!=moveHistory[t].loc)//changed piece
      break;
    t-=1;
    assert(moveHistory[t].pla == pla);
    myPickedPiece=moveHistory[t].loc;
    h.push_back(myPickedPiece);
    t-=1;

    assert(moveHistory[t].pla == opp);
    if(oppPickedPiece!=moveHistory[t].loc)//changed piece
      break;
    t-=1;
    assert(moveHistory[t].pla == opp);
    oppPickedPiece=moveHistory[t].loc;
    h.push_back(oppPickedPiece);
    t-=1;//now movehistory[t] is the my last full move
  }



  return h;
}
*/
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
  if(board.stage == 0)
    hashHistory.push_back(board.getSitHashNoStage(board.nextPla));
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
  hash ^= Rules::ZOBRIST_LOOP_RULE_HASH[hist.rules.loopRule];
  hash ^= Board::ZOBRIST_MM_RULE_HASH[hist.rules.maxmoves];
  hash ^= Board::ZOBRIST_MC_RULE_HASH[hist.rules.maxmovesNoCapture];

  auto h1 = hist.getLoopRuleHistory(board, C_BLACK);
  auto h2 = hist.getLoopRuleHistory(board, C_WHITE);
  for (int i = 0; i < h1.size(); i++)
  {
    hash ^= Board::ZOBRIST_73RULE_HISTORY_HASH[h1[i]][i][C_BLACK];
  }
  for(int i = 0; i < h2.size(); i++) {
    hash ^= Board::ZOBRIST_73RULE_HISTORY_HASH[h2[i]][i][C_WHITE];
  }


  return hash;
}
//check past 16 turns whether repeats
//if repeat:
//  I have been in traps: I win
//  opp have been in traps: opp win
//  both: draw
//  neither:  if move-t1 repeat move-t2
//      if(moveHistory[t1-1].loc == moveHistory[t2-1].loc) //target location of the last move
//        return board.nextPla;//who made the game repeats lose
//      else
//        return getOpp(board.nextPla);//who made the game repeats win

Color BoardHistory::checkRepeatEndWinner(const Board& board) const {
  assert(board.stage==0);
  assert(board.getSitHashNoStage(board.nextPla) == hashHistory[hashHistory.size()-1]);

  Hash128 lastHash = hashHistory[hashHistory.size()-1];
  int repeatLength = 0;
  for(int turn=hashHistory.size()-3; turn>=0; turn-=2) {
    if(hashHistory.size() - turn - 1 > 2 * 9) // 9 turns
      break;
    if(hashHistory[turn] == lastHash)
    {
      repeatLength = hashHistory.size() - turn - 1;
      break;
    }
  }
  if(repeatLength == 0)
    return C_WALL;
  assert(repeatLength % 2 == 0);
  int repeatTurn = repeatLength / 2;

  //检查是否有pass
  for(int i=0; i<repeatTurn*4; i++) {
    assert(moveHistory.size()-1-i >= 0);
    if(!board.isOnBoard(moveHistory[moveHistory.size()-1-i].loc))
      return C_WALL;
  }

  bool hasBeenInTrapMe = false;
  bool hasBeenInTrapOpp = false;
  for(int turn=0; turn<repeatTurn; turn++) {
    int moveHistIdx0=moveHistory.size()-1-4*turn;
    assert(moveHistIdx0 >= 3);
    assert(moveHistory[moveHistIdx0].pla == getOpp(board.nextPla));
    assert(moveHistory[moveHistIdx0-1].pla == getOpp(board.nextPla));
    assert(moveHistory[moveHistIdx0-2].pla == board.nextPla);
    assert(moveHistory[moveHistIdx0-3].pla == board.nextPla);
    if(GameLogic::isInTrap(moveHistory[moveHistIdx0].loc,getOpp(board.nextPla)))
      hasBeenInTrapOpp = true;
    if(GameLogic::isInTrap(moveHistory[moveHistIdx0-2].loc,board.nextPla))
      hasBeenInTrapMe = true;
  }
  if(hasBeenInTrapMe && hasBeenInTrapOpp)
    return C_EMPTY;
  else if(hasBeenInTrapMe)
    return board.nextPla;
  else if(hasBeenInTrapOpp)
    return getOpp(board.nextPla);
  else 
  {
    //检查是否一方为老鼠，另一方为狮虎。允许阻渡
    Color colorOpp=board.colors[moveHistory[moveHistory.size()-1].loc];
    Color colorMe=board.colors[moveHistory[moveHistory.size()-3].loc];
    assert(getPiecePla(colorOpp) == getOpp(board.nextPla));
    assert(getPiecePla(colorMe) == board.nextPla);
    colorOpp=getPieceType(colorOpp);
    colorMe=getPieceType(colorMe);
    if(colorOpp==C_RAT&&(colorMe==C_TIGER||colorMe==C_LION))
      return getOpp(board.nextPla);
    else if(colorMe==C_RAT&&(colorOpp==C_TIGER||colorOpp==C_LION))
      return board.nextPla;

    //重复局面的上一步是否为同一个棋子，判定长捉
    int moveHistIdx0=moveHistory.size()-1;
    int moveHistIdx1=moveHistory.size()-1-4*repeatTurn;
    assert(moveHistIdx1>=0);
    assert(moveHistory[moveHistIdx0].pla == getOpp(board.nextPla));
    if(moveHistory[moveHistIdx0].loc == moveHistory[moveHistIdx1].loc)
      return board.nextPla;
    else
      return getOpp(board.nextPla);
    
  }
  ASSERT_UNREACHABLE;
}


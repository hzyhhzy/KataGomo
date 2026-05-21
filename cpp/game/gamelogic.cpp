#include "../game/gamelogic.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

using namespace std;

bool GameLogic::isLegal(const Board& board, Player pla, Loc loc) {
  if(pla != board.nextPla) {
    std::cout << "Error next player ";
    return false;
  }

  if(loc == Board::PASS_LOC)
    return board.stage == 0;
  if(!board.isOnBoard(loc))
    return false;
  if(board.stage != 0)
    return false;
  if(board.colors[loc] != C_EMPTY)
    return false;
  if(board.isSingleStoneSuicide(loc, pla))
    return false;
  return true;
}

bool GameLogic::isLegalStrict(const Board& board, Player pla, Loc loc) {
  return isLegal(board, pla, loc);
}

GameLogic::MovePriority GameLogic::getMovePriorityAssumeLegal(const Board& board, const BoardHistory& hist, Player pla, Loc loc) {
  (void)board;
  (void)hist;
  (void)pla;
  (void)loc;
  return MP_NORMAL;
}

GameLogic::MovePriority GameLogic::getMovePriority(const Board& board, const BoardHistory& hist, Player pla, Loc loc) {
  if(!hist.isLegal(board, loc, pla))
    return MP_ILLEGAL;
  return getMovePriorityAssumeLegal(board, hist, pla, loc);
}

bool GameLogic::hasLegalMoveAssumeStage0(const Board& board) {
  if(board.stage != 0)
    return false;
  for(int loc = 0; loc < board.boardVolume(); loc++) {
    if(board.colors[loc] == C_EMPTY && !board.isKoBanned((Loc)loc) && !board.isSingleStoneSuicide((Loc)loc, board.nextPla))
      return true;
  }
  return false;
}

bool GameLogic::hasLegalMoveAssumeStage1(const Board& board, Loc chosenLoc) {
  (void)board;
  (void)chosenLoc;
  return false;
}

Color GameLogic::checkWinnerAfterPlayed(
  const Board& board,
  const BoardHistory& hist,
  Player pla,
  Loc loc,
  bool isLegalPass) {
  (void)pla;
  if(loc != Board::PASS_LOC || !isLegalPass)
    return C_WALL;

  if(hist.moveHistory.size() < 2)
    return C_WALL;
  const Move& prev = hist.moveHistory[hist.moveHistory.size()-2];
  if(prev.loc != Board::PASS_LOC)
    return C_WALL;

  double score = board.calculateAreaScoreWhiteMinusBlack(hist.rules.komi);
  if(score > 0.0)
    return C_WHITE;
  if(score < 0.0)
    return C_BLACK;
  return C_EMPTY;
}

GameLogic::ResultsBeforeNN::ResultsBeforeNN() {
  inited = false;
  winner = C_WALL;
  myOnlyLoc = Board::NULL_LOC;
}

void GameLogic::ResultsBeforeNN::init(const Board& board, const BoardHistory& hist, Color nextPlayer) {
  (void)board;
  (void)hist;
  (void)nextPlayer;
  if(inited)
    return;
  inited = true;
  winner = C_WALL;
  myOnlyLoc = Board::NULL_LOC;
}

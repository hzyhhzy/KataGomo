
#include <sstream>
#include "../core/global.h"
#include "../core/bsearch.h"
#include "../core/rand.h"
#include "../core/elo.h"
#include "../core/fancymath.h"
#include "../core/config_parser.h"
#include "../core/fileutils.h"
#include "../core/base64.h"
#include "../core/timer.h"
#include "../core/threadtest.h"
#include "../game/board.h"
#include "../game/rules.h"
#include "../game/boardhistory.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nneval.h"
#include "../program/gtpconfig.h"
#include "../program/play.h"
#include "../program/setup.h"
#include "../tests/tests.h"
#include "../command/commandline.h"
#include "../main.h"

using namespace std;

namespace {

Loc xy(const Board& board, int x, int y) {
  return Location::getLoc(x, y, board.x_size);
}

Board emptyBoard() {
  Board board;
  for(int y = 0; y < board.y_size; y++) {
    for(int x = 0; x < board.x_size; x++)
      testAssert(board.setStone(xy(board, x, y), C_EMPTY));
  }
  return board;
}

void playTurn(Board& board, BoardHistory& hist, int fromX, int fromY, int toX, int toY) {
  Player pla = board.nextPla;
  Loc from = xy(board, fromX, fromY);
  Loc to = xy(board, toX, toY);
  testAssert(hist.isLegal(board, from, pla));
  hist.makeBoardMoveAssumeLegal(board, from, pla);
  testAssert(board.stage == 1);
  testAssert(hist.isLegal(board, to, pla));
  hist.makeBoardMoveAssumeLegal(board, to, pla);
  testAssert(board.stage == 0);
}

void testRulesConfiguration() {
  Rules defaults;
  testAssert(defaults.maxMoves == 200);
  testAssert(defaults.repetitionCount == 2);
  testAssert(defaults.noLegalMoveRule == Rules::NO_LEGAL_MOVE_COUNT);

  Rules parsed = Rules::parseRules(
    "{\"maxmoves\":137,\"repetition\":3,\"nolegal\":\"DRAW\"}"
  );
  testAssert(parsed.maxMoves == 137);
  testAssert(parsed.repetitionCount == 3);
  testAssert(parsed.noLegalMoveRule == Rules::NO_LEGAL_MOVE_DRAW);
  testAssert(Rules::parseRules(parsed.toJsonString()) == parsed);

  string generatedConfig = GTPConfig::makeConfig(
    parsed, 100, 100, 10.0, 0.0, vector<int>(), 16, 12, 1
  );
  testAssert(generatedConfig.find("$$") == string::npos);
  testAssert(generatedConfig.find("maxMoves = 137") != string::npos);
  testAssert(generatedConfig.find("repetitionCount = 3") != string::npos);
  testAssert(generatedConfig.find("noLegalMoveRule = DRAW") != string::npos);
}

void testInitialSetupAndOrdinaryMoves() {
  Board board;
  testAssert(board.x_size == 6 && board.y_size == 6);
  testAssert(board.nextPla == P_BLACK);
  testAssert(board.stage == 0);
  testAssert(board.movenum == 0);
  testAssert(board.numPlaStonesOnBoard(P_BLACK) == 12);
  testAssert(board.numPlaStonesOnBoard(P_WHITE) == 12);
  for(int x = 0; x < 6; x++) {
    testAssert(board.colors[xy(board, x, 0)] == P_BLACK);
    testAssert(board.colors[xy(board, x, 1)] == P_BLACK);
    testAssert(board.colors[xy(board, x, 4)] == P_WHITE);
    testAssert(board.colors[xy(board, x, 5)] == P_WHITE);
  }

  Loc source = xy(board, 0, 1);
  Loc adjacent = xy(board, 0, 2);
  testAssert(board.isLegal(source, P_BLACK));
  board.playMoveAssumeLegal(source, P_BLACK);
  testAssert(board.stage == 1 && board.movenum == 0);
  testAssert(board.isLegal(adjacent, P_BLACK));
  testAssert(!board.isLegal(xy(board, 2, 2), P_BLACK));
  board.playMoveAssumeLegal(adjacent, P_BLACK);
  testAssert(board.stage == 0 && board.movenum == 1);
  testAssert(board.nextPla == P_WHITE);
  board.checkConsistency();
}

void testLoopCaptures() {
  Board board = emptyBoard();
  board.setStone(xy(board, 1, 2), P_BLACK);
  board.setStone(xy(board, 2, 3), P_WHITE);
  BoardHistory hist(board, P_BLACK, Rules());
  Loc source = xy(board, 1, 2);
  Loc target = xy(board, 2, 3);
  testAssert(hist.isLegal(board, source, P_BLACK));
  hist.makeBoardMoveAssumeLegal(board, source, P_BLACK);
  testAssert(hist.isLegal(board, target, P_BLACK));
  hist.makeBoardMoveAssumeLegal(board, target, P_BLACK);
  testAssert(board.colors[target] == P_BLACK);
  testAssert(board.colors[source] == C_EMPTY);
  testAssert(hist.isGameFinished && hist.winner == P_BLACK);
  testAssert(hist.posHashHistoryCount.size() == 1);
  testAssert(hist.posHashHistoryCount.find(board.pos_hash)->second == 1);
  board.checkConsistency();

  Board blocked = emptyBoard();
  blocked.setStone(xy(blocked, 1, 2), P_BLACK);
  blocked.setStone(xy(blocked, 2, 1), P_BLACK);
  blocked.setStone(xy(blocked, 2, 2), P_BLACK);
  blocked.setStone(xy(blocked, 1, 1), P_BLACK);
  blocked.setStone(xy(blocked, 1, 3), P_BLACK);
  blocked.setStone(xy(blocked, 2, 3), P_WHITE);
  blocked.playMoveAssumeLegal(xy(blocked, 1, 2), P_BLACK);
  testAssert(!blocked.isLegal(xy(blocked, 2, 3), P_BLACK));

  Board straight = emptyBoard();
  straight.setStone(xy(straight, 1, 2), P_BLACK);
  straight.setStone(xy(straight, 2, 2), P_BLACK);
  straight.setStone(xy(straight, 1, 1), P_BLACK);
  straight.setStone(xy(straight, 1, 3), P_BLACK);
  straight.setStone(xy(straight, 0, 2), P_WHITE);
  straight.playMoveAssumeLegal(xy(straight, 1, 2), P_BLACK);
  testAssert(!straight.isLegal(xy(straight, 0, 2), P_BLACK));

  Board corner = emptyBoard();
  corner.setStone(xy(corner, 0, 0), P_BLACK);
  corner.setStone(xy(corner, 2, 3), P_WHITE);
  corner.playMoveAssumeLegal(xy(corner, 0, 0), P_BLACK);
  testAssert(!corner.isLegal(xy(corner, 2, 3), P_BLACK));
}

void testRepetition(int repetitionCount, int cyclesUntilEnd) {
  Board board = emptyBoard();
  board.setStone(xy(board, 0, 0), P_BLACK);
  board.setStone(xy(board, 1, 1), P_BLACK);
  board.setStone(xy(board, 4, 4), P_WHITE);
  Rules rules(0, repetitionCount, Rules::NO_LEGAL_MOVE_COUNT);
  BoardHistory hist(board, P_BLACK, rules);
  Hash128 initialPosHash = board.pos_hash;
  testAssert(hist.posHashHistoryCount.find(initialPosHash)->second == 1);

  for(int cycle = 0; cycle < cyclesUntilEnd; cycle++) {
    playTurn(board, hist, 1, 1, 1, 2);
    playTurn(board, hist, 4, 4, 4, 3);
    playTurn(board, hist, 1, 2, 1, 1);
    playTurn(board, hist, 4, 3, 4, 4);
    if(cycle + 1 < cyclesUntilEnd)
      testAssert(!hist.isGameFinished);
  }

  testAssert(board.pos_hash == initialPosHash);
  testAssert(hist.posHashHistoryCount.find(initialPosHash)->second == repetitionCount);
  testAssert(hist.isGameFinished && hist.winner == P_BLACK);
  testAssert(board.movenum == cyclesUntilEnd * 4);
}

void testMoveLimitAndNoLegalRules() {
  Board board = emptyBoard();
  board.setStone(xy(board, 0, 0), P_BLACK);
  board.setStone(xy(board, 1, 1), P_BLACK);
  board.setStone(xy(board, 4, 4), P_WHITE);
  Rules limitRules(1, 3, Rules::NO_LEGAL_MOVE_COUNT);
  BoardHistory limitHist(board, P_BLACK, limitRules);
  playTurn(board, limitHist, 1, 1, 1, 2);
  testAssert(board.movenum == 1);
  testAssert(limitHist.isGameFinished && limitHist.winner == P_BLACK);

  Board stuck = emptyBoard();
  stuck.setStone(xy(stuck, 0, 0), P_BLACK);
  stuck.setStone(xy(stuck, 0, 1), P_WHITE);
  stuck.setStone(xy(stuck, 1, 0), P_WHITE);
  stuck.setStone(xy(stuck, 1, 1), P_WHITE);
  testAssert(!GameLogic::hasLegalMoveAssumeStage0(stuck, P_BLACK));
  testAssert(stuck.isLegal(Board::PASS_LOC, P_BLACK));

  BoardHistory loseHist(stuck, P_BLACK, Rules(200, 2, Rules::NO_LEGAL_MOVE_LOSE));
  testAssert(loseHist.isGameFinished && loseHist.winner == P_WHITE);
  BoardHistory drawHist(stuck, P_BLACK, Rules(200, 2, Rules::NO_LEGAL_MOVE_DRAW));
  testAssert(drawHist.isGameFinished && drawHist.winner == C_EMPTY);
  BoardHistory countHist(stuck, P_BLACK, Rules(200, 2, Rules::NO_LEGAL_MOVE_COUNT));
  testAssert(countHist.isGameFinished && countHist.winner == P_WHITE);
}

void testDisabledRepetitionNNFeatures() {
  Board board = emptyBoard();
  board.setStone(xy(board, 0, 0), P_BLACK);
  board.setStone(xy(board, 1, 1), P_BLACK);
  board.setStone(xy(board, 4, 4), P_WHITE);
  BoardHistory hist(board, P_BLACK, Rules(0, 2, Rules::NO_LEGAL_MOVE_COUNT));
  playTurn(board, hist, 1, 1, 1, 2);
  playTurn(board, hist, 4, 4, 4, 3);
  playTurn(board, hist, 1, 2, 1, 1);

  Loc source = xy(board, 4, 3);
  testAssert(hist.isLegal(board, source, P_WHITE));
  hist.makeBoardMoveAssumeLegal(board, source, P_WHITE);
  testAssert(board.stage == 1);

  float spatial[NNInputs::NUM_FEATURES_SPATIAL_V7 * 6 * 6];
  float global[NNInputs::NUM_FEATURES_GLOBAL_V7];
  MiscNNInputParams nnInputParams;
  NNInputs::fillRowV7(
    board, hist, P_WHITE, nnInputParams,
    6, 6, false, spatial, global
  );
  int destinationPos = NNPos::xyToPos(4, 4, 6);
  testAssert(spatial[4 * 36 + destinationPos] == 1.0f);
  testAssert(spatial[5 * 36 + destinationPos] == 0.0f);
  testAssert(spatial[6 * 36 + destinationPos] == 0.0f);

  Board limitedBoard = emptyBoard();
  limitedBoard.movenum = 50;
  BoardHistory limitedHist(
    limitedBoard, P_BLACK, Rules(200, 3, Rules::NO_LEGAL_MOVE_DRAW)
  );
  NNInputs::fillRowV7(
    limitedBoard, limitedHist, P_BLACK, nnInputParams,
    6, 6, false, spatial, global
  );
  testAssert(global[2] == 0.0f);
  testAssert(global[3] == 0.0f);
  testAssert(global[4] == 1.0f);
  testAssert(global[5] == 0.0f);
  testAssert(global[6] == 1.0f);
  testAssert(std::abs(global[7] - std::exp(-1.0f)) < 1e-6f);
  testAssert(std::abs(global[8] - std::exp(-3.0f)) < 1e-6f);
  testAssert(global[10] == -1.0f);
  testAssert(std::abs(global[11] - std::exp(-10.0f)) < 1e-6f);
  testAssert(global[9] == 0.0f);

  BoardHistory legacyLoseHist(
    limitedBoard, P_BLACK, Rules(200, 3, Rules::NO_LEGAL_MOVE_LOSE)
  );
  NNInputs::fillRowV7(
    limitedBoard, legacyLoseHist, P_BLACK, nnInputParams,
    6, 6, false, spatial, global
  );
  testAssert(global[3] == 0.0f && global[4] == 0.0f && global[5] == 0.0f);

  BoardHistory countRuleHist(
    limitedBoard, P_BLACK, Rules(200, 3, Rules::NO_LEGAL_MOVE_COUNT)
  );
  NNInputs::fillRowV7(
    limitedBoard, countRuleHist, P_BLACK, nnInputParams,
    6, 6, false, spatial, global
  );
  testAssert(global[3] == 0.0f && global[4] == 0.0f && global[5] == 1.0f);
}

void testPassAndHashes() {
  Board board;
  Rules rules;
  BoardHistory hist(board, P_BLACK, rules);
  testAssert(hist.isLegal(board, Board::PASS_LOC, P_BLACK));
  hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_BLACK);
  testAssert(hist.isGameFinished && hist.winner == P_WHITE);
  testAssert(board.nextPla == P_WHITE && board.stage == 0 && board.movenum == 1);
  board.checkConsistency();

  Board stageOne;
  BoardHistory stageOneHist(stageOne, P_BLACK, rules);
  Loc source = xy(stageOne, 0, 1);
  stageOneHist.makeBoardMoveAssumeLegal(stageOne, source, P_BLACK);
  testAssert(stageOneHist.isLegal(stageOne, Board::PASS_LOC, P_BLACK));
  stageOneHist.makeBoardMoveAssumeLegal(stageOne, Board::PASS_LOC, P_BLACK);
  testAssert(stageOneHist.isGameFinished && stageOneHist.winner == P_WHITE);
  testAssert(stageOne.stage == 0 && stageOne.midLocs[0] == Board::NULL_LOC);
  stageOne.checkConsistency();

  Board hashBoard;
  Hash128 posHash = hashBoard.pos_hash;
  Board moveCountBoard = hashBoard;
  moveCountBoard.movenum = 17;
  testAssert(moveCountBoard.pos_hash == posHash);

  Rules baseRules(200, 2, Rules::NO_LEGAL_MOVE_COUNT);
  Rules maxRules(201, 2, Rules::NO_LEGAL_MOVE_COUNT);
  Rules repRules(200, 3, Rules::NO_LEGAL_MOVE_COUNT);
  Rules stuckRules(200, 2, Rules::NO_LEGAL_MOVE_DRAW);
  BoardHistory baseHist(hashBoard, P_BLACK, baseRules);
  BoardHistory maxHist(hashBoard, P_BLACK, maxRules);
  BoardHistory repHist(hashBoard, P_BLACK, repRules);
  BoardHistory stuckHist(hashBoard, P_BLACK, stuckRules);
  Hash128 baseHash = BoardHistory::getSituationRulesHash(hashBoard, baseHist, P_BLACK);
  testAssert(baseHash != BoardHistory::getSituationRulesHash(hashBoard, maxHist, P_BLACK));
  testAssert(baseHash != BoardHistory::getSituationRulesHash(hashBoard, repHist, P_BLACK));
  testAssert(baseHash != BoardHistory::getSituationRulesHash(hashBoard, stuckHist, P_BLACK));
  testAssert(baseHash != BoardHistory::getSituationRulesHash(moveCountBoard, baseHist, P_BLACK));
}

void testPassPolicyMasking() {
  NNEvaluator nnEval(
    "surakarta-rules-test", "", "", nullptr,
    1, 1, 6, 6, true, false,
    -1, 0, true, "", "", false,
    enabled_t::False, enabled_t::False,
    1, vector<int>{0}, "surakarta-rules-test-seed", false, 0, 1
  );
  nnEval.spawnServerThreads();
  MiscNNInputParams nnInputParams;
  int passPos = NNPos::locToPos(Board::PASS_LOC, 6, 6, 6);

  Board ordinaryBoard;
  BoardHistory ordinaryHist(ordinaryBoard, P_BLACK, Rules());
  NNResultBuf ordinaryResult;
  nnEval.evaluate(ordinaryBoard, ordinaryHist, P_BLACK, nnInputParams, ordinaryResult, true);
  testAssert(ordinaryResult.result->policyProbs[passPos] == -1.0f);
  int sourcePos = NNPos::locToPos(xy(ordinaryBoard, 0, 1), 6, 6, 6);
  testAssert(ordinaryResult.result->policyProbs[sourcePos] >= 0.0f);

  Board stuck = emptyBoard();
  stuck.setStone(xy(stuck, 0, 0), P_BLACK);
  stuck.setStone(xy(stuck, 0, 1), P_WHITE);
  stuck.setStone(xy(stuck, 1, 0), P_WHITE);
  stuck.setStone(xy(stuck, 1, 1), P_WHITE);
  BoardHistory stuckHist(stuck, P_BLACK, Rules());
  NNResultBuf stuckResult;
  nnEval.evaluate(stuck, stuckHist, P_BLACK, nnInputParams, stuckResult, true);
  testAssert(stuckResult.result->policyProbs[passPos] == 1.0f);
  for(int pos = 0; pos < 36; pos++)
    testAssert(stuckResult.result->policyProbs[pos] == -1.0f);
  nnEval.killServerThreads();
}

void testSelfplayRuleSampling() {
  Logger logger;
  logger.setDisabled(true);

  ConfigParser weightedCfg(map<string,string>{
    {"repetitionRules", "2,3,3"},
    {"noLegalMoveRules", "LOSE,DRAW,DRAW,COUNT"},
    {"bSizes", "6"},
    {"bSizeRelProbs", "1"},
  });
  GameInitializer weightedInitializer(weightedCfg, logger, "weighted-rule-test");
  int repetition2 = 0;
  int repetition3 = 0;
  int noLegalLose = 0;
  int noLegalDraw = 0;
  int noLegalCount = 0;
  for(int i = 0; i < 2000; i++) {
    Rules rules = weightedInitializer.createRules();
    repetition2 += rules.repetitionCount == 2 ? 1 : 0;
    repetition3 += rules.repetitionCount == 3 ? 1 : 0;
    testAssert(rules.maxMoves == 200);
    noLegalLose += rules.noLegalMoveRule == Rules::NO_LEGAL_MOVE_LOSE ? 1 : 0;
    noLegalDraw += rules.noLegalMoveRule == Rules::NO_LEGAL_MOVE_DRAW ? 1 : 0;
    noLegalCount += rules.noLegalMoveRule == Rules::NO_LEGAL_MOVE_COUNT ? 1 : 0;
  }
  testAssert(repetition3 > repetition2);
  testAssert(repetition2 + repetition3 == 2000);
  testAssert(noLegalDraw > noLegalLose && noLegalDraw > noLegalCount);
  testAssert(noLegalLose + noLegalDraw + noLegalCount == 2000);

  ConfigParser randomizedCfg(map<string,string>{
    {"repetitionRules", "2"},
    {"noLegalMoveRules", "COUNT"},
    {"bSizes", "6"},
    {"bSizeRelProbs", "1"},
  });
  GameInitializer randomizedInitializer(randomizedCfg, logger, "continuous-max-moves-test");
  PlaySettings playSettings;
  set<int> sampledMaxMoves;
  for(int i = 0; i < 100; i++) {
    Board board;
    BoardHistory hist;
    Player pla = C_EMPTY;
    OtherGameProperties otherGameProps;
    randomizedInitializer.createGame(
      board, pla, hist, nullptr, playSettings, otherGameProps, nullptr
    );
    testAssert(!otherGameProps.isRandomInitialBoard);
    testAssert(hist.rules.maxMoves >= 10 && hist.rules.maxMoves <= 700);
    sampledMaxMoves.insert(hist.rules.maxMoves);
  }
  testAssert(sampledMaxMoves.size() > 1);

  ConfigParser randomBoardCfg(map<string,string>{
    {"repetitionRules", "2"},
    {"noLegalMoveRules", "COUNT"},
    {"maxMovesRandomBase", "123"},
    {"maxMovesRandomPositiveScale", "0"},
    {"maxMovesRandomNegativeScale", "0"},
    {"maxMovesRandomMin", "10"},
    {"maxMovesRandomMax", "700"},
    {"randomInitialBoardProb", "1"},
    {"bSizes", "6"},
    {"bSizeRelProbs", "1"},
  });
  GameInitializer randomBoardInitializer(randomBoardCfg, logger, "random-board-test");
  int emptyCount = 0;
  int blackCount = 0;
  int whiteCount = 0;
  for(int i = 0; i < 100; i++) {
    Board board;
    BoardHistory hist;
    Player pla = C_EMPTY;
    OtherGameProperties otherGameProps;
    randomBoardInitializer.createGame(
      board, pla, hist, nullptr, playSettings, otherGameProps, nullptr
    );
    testAssert(otherGameProps.isRandomInitialBoard);
    testAssert(hist.rules.maxMoves == 123);
    for(int y = 0; y < board.y_size; y++) {
      for(int x = 0; x < board.x_size; x++) {
        Color color = board.colors[xy(board, x, y)];
        emptyCount += color == C_EMPTY ? 1 : 0;
        blackCount += color == C_BLACK ? 1 : 0;
        whiteCount += color == C_WHITE ? 1 : 0;
      }
    }
  }
  testAssert(emptyCount + blackCount + whiteCount == 3600);
  testAssert(emptyCount > 900 && emptyCount < 1500);
  testAssert(blackCount > 900 && blackCount < 1500);
  testAssert(whiteCount > 900 && whiteCount < 1500);
}

} // namespace

int MainCmds::runtests(const vector<string>& args) {
  if(args.size() != 1 || args[0] != "runtests")
    throw StringError("runtests takes no arguments");
  Board::initHash();
  testRulesConfiguration();
  testInitialSetupAndOrdinaryMoves();
  testLoopCaptures();
  testRepetition(2, 1);
  testRepetition(3, 2);
  testMoveLimitAndNoLegalRules();
  testDisabledRepetitionNNFeatures();
  testPassAndHashes();
  testPassPolicyMasking();
  testSelfplayRuleSampling();
  cout << "Surakarta rules tests passed" << endl;
  return 0;
}

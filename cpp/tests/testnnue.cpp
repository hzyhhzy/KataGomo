#include "../tests/tests.h"

#include "../program/playutils.h"
#include "../main.h"

#include "../nnue/Eva_nnuev2.h"
#include "../nnue/NNUEBoardHistory.h"
#include "../nnue/NNUE_VCF_ABSearch.h"
#include "../nnue_search/NNUE_VCF_MCTSsearch.h"
#include "../game/gamelogic.h"
#include "../neuralnet/nninputs.h"
#include "../nnue_search/VCFLogic.h"
#include "../nnue_search/VCFCalculator.h"

//------------------------
#include "../core/using.h"
//------------------------
#include <chrono>

using namespace std;
using namespace NNUE;
using namespace NNUEV2;

// Function declarations
void testAutoPlay(const ModelWeight* weights);
void testABSearch(const ModelWeight* weights);
void testMCTSSearch(const ModelWeight* weights);
void testMCTSSearch2(const ModelWeight* weights);
void testMCTSSearch3(const ModelWeight* weights);
void testVCFPrune1(const ModelWeight* weights);
void testVCFPrune2(const ModelWeight* weights);


int MainCmds::testnnue() {
  Board::initHash();
  Rand seedRand;

  ConfigParser cfg;
  string nnueModelFile;
  /*
  try {
    KataGoCommandLine cmd("Test NNUE.");
    cmd.addConfigFileArg(KataGoCommandLine::defaultGtpConfigFileName(), "nnue_example.cfg");
    TCLAP::ValueArg<int> boardSizeArg(
      "", "boardsize", "Size of board, default 19", false, 19, "SIZE");
    cmd.add(boardSizeArg);

    cmd.setShortUsageArgLimit();
    cmd.addOverrideConfigArg();

    cmd.parseArgs(args);

    boardSize = boardSizeArg.getValue();
    cmd.getConfig(cfg);

  } catch(TCLAP::ArgException& e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }*/

  const bool logToStdoutDefault = true;
  const bool logToStderrDefault = false;
  const bool logTimeDefault = false;
  Logger logger(NULL, logToStdoutDefault, logToStderrDefault, logTimeDefault);

  //string modelpath = "H:/gomtrain2024/connectsix/export/v2_64_highwd2.txt"; 
  string modelpath = "C:/gomtrain2025/connect6nnue/export/v2_c16_vcfonly_3.txt";

  NNUEV2::ModelWeight* nnueWeight = new NNUEV2::ModelWeight();
  nnueWeight->loadParam(modelpath);

  Eva_nnuev2* eva = new Eva_nnuev2(nnueWeight);

  Board board(19, 19);
  string rootBoardSequence = "j10i8l10";
  vector<Loc> rootBoardLocSeq = Location::parseSequenceGom(rootBoardSequence, board);
  PlayUtils::playMoveLocSequence(board, board.nextPla, rootBoardLocSeq);
  //eva->debug_print();
  
  // Test auto-play function
  //testAutoPlay(nnueWeight);
  
  // Test AB search with specific initial position
  //testABSearch(nnueWeight);
  
  // Test MCTS search with same initial position
  //testMCTSSearch2(nnueWeight);
  //

  //testVCFPrune1(nnueWeight);

  testVCFPrune2(nnueWeight);

  delete nnueWeight;
  return 0;
}

// Test function: auto-play from empty 19x19 board using highest policy moves
void testAutoPlay(const ModelWeight* weights) {
  cout << "Starting auto-play test from empty 19x19 board..." << endl;
  
  // Create board and rules
  Board board(19, 19);
  Rules rules;
  string rootBoardSequence = "j10i9i11";
  vector<Loc> rootBoardLocSeq = Location::parseSequenceGom(rootBoardSequence, board);
  PlayUtils::playMoveLocSequence(board, board.nextPla, rootBoardLocSeq);
  
  // Create NNUE input parameters
  MiscNNInputParams nnInputParams;
  
  // Create NNUEBoardHistory
  NNUEBoardHistory nnueHistory(weights, nnInputParams, false);
  nnueHistory.clear(board, board.nextPla, rules);
  
  
  cout << "Starting game with empty board..." << endl;
  
  while (true) {
    Color pla = board.nextPla;
    // Get policy from NNUE
    PolicyType policy[MaxBS * MaxBS + 1];
    auto value = nnueHistory.evaluateFull(pla, policy);
    bool cons= nnueHistory.checkEvaluatorBoardConsistency();
    if (!cons)
      throw StringError("nnueHistory not consist");
    
    // Find the move with highest policy value among legal moves
    Loc bestMove = Board::NULL_LOC;
    PolicyType bestPolicy = MIN_POLICY;
    
    for (int nu = 0; nu < MaxBS * MaxBS + 1; nu++) {
      Loc loc = nu < MaxBS * MaxBS ? Location::getLoc(nu % MaxBS, nu / MaxBS, board.x_size) : Board::PASS_LOC;
      if(!board.isLegal( loc, pla))
        continue;
      if(board.stage == 1 && loc != Board::PASS_LOC) { // check move priority
        if(board.getLocationPriority(loc) + Board::PRIOR_EPS < board.firstLocPriority)
          continue;
      }

      if (policy[nu] > bestPolicy) {
        bestPolicy = policy[nu];
        bestMove = loc;
      }
        
    }
    
    // If no legal move found, try pass
    if (bestMove == Board::NULL_LOC) {
      ASSERT_UNREACHABLE;
      bestMove = Board::PASS_LOC;
    }
    
    
    // Make the move
    nnueHistory.play(pla, bestMove);
    board.playMoveAssumeLegal(bestMove, pla);
    
    // Print move information
    string moveStr = Location::toString(bestMove, board);
    cout << "Move " << board.movenum << ": " << (pla == C_BLACK ? "Black" : "White") 
         << " plays " << moveStr << " (policy: " << bestPolicy << ")" << endl;
    cout << "Win=" << value.win << " Lose=" << value.loss <<" Draw="<< value.draw << endl;
    
    
    
    // Optional: print board state every 10 moves
    cout << "Board state after " << board.movenum << " moves:" << endl;
    Board::printBoard(cout, board, board.firstLoc, NULL);
    cout << endl;

    // Check for game end conditions
    if (nnueHistory.isGameFinished) {
      cout << "Game finished - winner detected after " << board.movenum << " moves." << endl;
      cout<<"winner="<<(nnueHistory.winner==C_BLACK?"Black":nnueHistory.winner==C_WHITE?"White":"Draw")<<endl;

      break;
    }
  }
  
}

// Test function: AB search with specific initial position
void testABSearch(const ModelWeight* weights) {
  cout << "Starting AB search test with specific initial position..." << endl;
  
  // Create board and rules
  Board board(19, 19);
  Rules rules;
  rules.maxMoves = 109;
  //rules.VCNRule = Rules::VCNRULE_VC4_B;
  
  // Set up the specific initial position
  //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15g11h11i9i14g12h13j9l13j7h15h16h12h17f12i16f13j16m12j17g9n13g14g8e11k18f8f7h9h8f10f9f11f5i8h6h7k8";//can vcf
  //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15g11h11i9i14g12h13j9l13j7h15h16h12h17f12i16f13j16m12j17g9n13g14g8e11k18f8f7h9h8f10f9f11f5i8h6h7g7";//cannot vcf
  string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15g11h11i9i14g12h13j9l13j7h15h16h12h17f12i16f13j16m12j17g9n13g14g8e11k18f8f7h9h8f10f9f11f5i8h6g7i6k8l7m8m6o13o15n14n15m14n12l12m11n10o11m9d11m10c11l11l10l9n8m13o12m7p10n9o9l6k5m5j4";
  //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15i14h12";
  vector<Loc> initialLocSeq = Location::parseSequenceGom(initialSequence, board);
  PlayUtils::playMoveLocSequence(board, board.nextPla, initialLocSeq);
  
  // Create NNUE input parameters
  MiscNNInputParams nnInputParams;
  
  // Create NNUEBoardHistory
  NNUEBoardHistory nnueHistory(weights, nnInputParams, true);
  nnueHistory.clear(board, board.nextPla, rules);
 // Board::printBoard(cout, board, Board::NULL_LOC, NULL);
 // int p = GameLogic::checkTwoFourThreats(board, C_BLACK);
  
  cout << "Initial board position:" << endl;
  Board::printBoard(cout, board, board.firstLoc, NULL);
  cout << endl;
  
  cout << "Move history: " << initialSequence << endl;
  cout << "Total moves: " << board.movenum << endl;
  cout << "Next player: " << (board.nextPla == C_BLACK ? "Black" : "White") << endl;
  cout << endl;
  
  // Create cache table with size 2^25 and mutex pool size 2^11
  ABSearch_CacheTable cacheTable(25, 11);
  
  // Create AB search instance with cache table
  NNUE::VCF_ABSearch abSearch(&nnueHistory, &cacheTable, board.nextPla);
  
  // Test different search depths
  vector<double> testDepths = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28 };
  
  for (double depth : testDepths) {
    cout << "Testing AB search with depth " << depth << "..." << endl;
    
    auto startTime = chrono::high_resolution_clock::now();
    double result = abSearch.search(depth);
    auto endTime = chrono::high_resolution_clock::now();
    
    auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime);
    
    cout << "Search depth: " << depth << endl;
    cout << "Search result: " << result << endl;
    cout << "Search time: " << duration.count() << " ms" << endl;
    cout << "NNEval: " << abSearch.nnevalCount << "  " << "Nodes: " << abSearch.nodeCount << "  " << endl;
    
    // Interpret the result
    if (result > 1.0) {
      cout << "Result interpretation: Attacking player has a winning strategy" << endl;
    } else if (result < -1.0) {
      cout << "Result interpretation: Defending player can defend successfully" << endl;
    } else {
      cout << "Result interpretation: Uncertain outcome (value: " << result << ")" << endl;
    }
    cout << "----------------------------------------" << endl;
  }
  
  cout << "AB search test completed." << endl;
}

void testMCTSSearch(const ModelWeight* weights) {
  cout << "Starting MCTS search test with specific initial position..." << endl;
  
  // Create board and rules
  Board board(19, 19);
  Rules rules;
  rules.VCNRule = Rules::VCNRULE_VC4_B;
  rules.maxMoves = 109;
  
  // Set up the same initial position as AB search test
  //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15g11h11i9i14g12h13j9l13j7h15h16h12h17f12i16f13j16m12j17g9n13g14g8e11k18f8f7h9h8f10f9f11f5i8h6g7i6k8l7m8m6o13o15n14n15m14n12l12m11n10o11m9d11m10c11l11l10l9n8m13o12m7p10n9o9l6k5m5j4"; //~2 seconds or longer, win in 101 moves
  //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15g11h11i9i14g12h13j9l13j7h15h16h12h17f12i16f13j16m12j17g9n13g14g8e11k18f8f7h9h8f10f9f11f5i8h6h7g7";//cannot vcf
  //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15g11h11i9i14g12h13j9l13j7h15h16h12h17f12i16f13j16m12j17g9n13g14g8e11k18f8f7h9h8f10f9f11f5i8h6h7k8";//can vcf
  //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15i14h12i15g15j15f16j16k17g13l18l14i17m13h18j17l17h17m17";//2 moves win
  //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15g11h11i9i14g12h13j9l13j7h15h16h12h17f12i16f13j16m12j17g9n13g14g8e11k18f8f7h9h8f10f9f11f5i8h6h7k8g7i5e9j4";
  //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15k12";
  //string initialSequence = "j10i11h10j9k7g9g11m6m5k8l6m4a1m8l8";//white has a four
  //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15n12";//even a bit difficult for katago
  //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15g11h11i9g12l13i14i16f12h16e13d12j14s19";//require 20s (5e5 nodes), even a bit difficult for katago
  //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15g11h11i9g12l13i14i16f12h16e13d12j14i15g15g17f14f16g9m12j9m14f13m11g13n15n14o13e15d16h12c17g14n12e16f9e9o14g8p13h7p12i8h8f8i7j7g7h9f6g6g5g10g4k6l6k12l8l5o16i6m4k5m5j17j5k17j18k9m15k10k4m9k7n10k3o9k2n9n6n7l4"; //109 moves to win
  string initialSequence = "j10c2p5k11l12q15d15";//4 useless white stones
  //string initialSequence = "j10c2p5k11j12k12i9";//4 useless white stones
  //string initialSequence = "j10f5d3k11l12d4e4";//4 useless white stones
  //string initialSequence = "j10d4e3k9i11e5g5f2j9a1c3";//first move must be purely defense
  //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15g11h11i9i14g12h13j9l13j7h15h16h12h17f12i16f13j16m12j17g9n13g14g8e11k18f8f7h9h8f10f9f11f5i8h6h7j8g7i5e9j4i7i6i10i4j6g6k6f6h4g3g5f2";//1 move win
  vector<Loc> initialLocSeq = Location::parseSequenceGom(initialSequence, board);
  PlayUtils::playMoveLocSequence(board, board.nextPla, initialLocSeq);
  
  // Create NNUE input parameters
  MiscNNInputParams nnInputParams;
  
  // Create NNUEBoardHistory
  NNUEBoardHistory nnueHistory(weights, nnInputParams, true);
  nnueHistory.clear(board, board.nextPla, rules);
  
  cout << "Initial board position:" << endl;
  Board::printBoard(cout, board, board.firstLoc, NULL);
  cout << endl;
  
  cout << "Move history: " << initialSequence << endl;
  cout << "Total moves: " << board.movenum << endl;
  cout << "Next player: " << (board.nextPla == C_BLACK ? "Black" : "White") << endl;
  cout << endl;
  
  // Initialize MCTS cache
  NNUE_VCF_MCTSsearch::MCTS_CacheTable cachetable(25, 11);
  
  // Create MCTS search instance
  NNUE_VCF_MCTSsearch::MCTSsearch mcts(&cachetable ,&nnueHistory, board.nextPla);
  //NNUE_VCF_MCTSsearch::MCTSsearch mcts(nullptr, &nnueHistory, board.nextPla);
  
  // Set MCTS parameters
  mcts.params.puct = 0.7;
  mcts.params.expandFactor = 0.2;
  mcts.params.policyTemp = 1.0;
  mcts.params.puctPow = 0.75;
  mcts.params.fpuReductionConst = 0.0;
  mcts.params.fpuReductionPolicy = 0.1;
  mcts.params.localPolicyBonusStage1 = 0.0;
  
  // Test different search factors (visits = factor * 1000)
  vector<int64_t> testFactors = {1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536};
  for (int i = 0; i < 10000; i++)
    testFactors.push_back(65536);



  auto startTime = chrono::high_resolution_clock::now();
  for (double factor : testFactors) {
    cout << "Testing MCTS search with factor " << factor << " (" << (int)(factor) << " visits)..." << endl;
    
    Loc bestMove;
    double result = mcts.fullsearch(board.nextPla, factor, bestMove);
    auto endTime = chrono::high_resolution_clock::now();
    
    auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime);
    
    cout << "Search visits: " << factor << endl;
    cout << "Best move: " << Location::toString(bestMove, board) << endl;
    cout << "Search result: " << result << endl;
    cout << "Search time: " << duration.count() << " ms" << endl;
    cout << "Root visits: " << mcts.getRootVisit() << endl;
    
    // Get and display principal variation
    vector<pair<Loc, uint64_t>> pv = mcts.getPV();
    cout << "Principal Variation (" << pv.size() << " moves): ";
    for (size_t i = 0; i < pv.size(); i++) {
      if (i > 0) cout << " ";
      cout << Location::toString(pv[i].first, board) << "(" << pv[i].second << ")";
    }
    cout << endl;
    
    // Check if win/loss is determined
    if (mcts.rootNode && mcts.rootNode->isWinDetermined) {
      cout << "Result interpretation: " << (mcts.rootNode->winner == board.nextPla ? "Win" : "Loss")
           << " determined in " << abs(mcts.rootNode->stepsToWin) << " steps" << endl;
      
      // Calculate winning dependency tree size
      auto calcStartTime = chrono::high_resolution_clock::now();
      int64_t treeSize = mcts.calculateWinningDependencyTreeSize();
      auto calcEndTime = chrono::high_resolution_clock::now();
      auto calcDuration = chrono::duration_cast<chrono::microseconds>(calcEndTime - calcStartTime);
      
      cout << "Winning dependency tree size: " << treeSize << " nodes" << endl;
      cout << "Tree calculation time: " << calcDuration.count() << " microseconds" << endl;
      
      cout << "Win/Loss determined, stopping further searches." << endl;
      cout << "----------------------------------------" << endl;
      break; // 检测到必胜/必败时直接退出循环
    } else {
      cout << "Result interpretation: Uncertain outcome (value: " << result << ")" << endl;
    }
    cout << "----------------------------------------" << endl;
  }

  if (mcts.rootNode->winner == board.nextPla)
  {
    // Calculate defense dependency map
    auto mapStartTime = chrono::high_resolution_clock::now();
    vector<int8_t> dependMap = mcts.calculateDefenseDependencyMap();
    auto mapEndTime = chrono::high_resolution_clock::now();
    auto mapDuration = chrono::duration_cast<chrono::microseconds>(mapEndTime - mapStartTime);

    cout << "Defense dependency map calculation time: " << mapDuration.count() << " microseconds" << endl;
    VCFLogic::printBoardWithDependencyMap(board, dependMap);
  }
  cout << "MCTS search test completed." << endl;
}

void testMCTSSearch2(const ModelWeight* weights) {
  cout << "Starting MCTS search test with specific initial position..." << endl;
  
  // Create board and rules
  Board board(19, 19);
  Rules rules;
  rules.VCNRule = Rules::VCNRULE_VC4_B;
  rules.maxMoves = 25;
  
  // Set up the same initial position as AB search test
  //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15g11h11i9i14g12h13j9l13j7h15h16h12h17f12i16f13j16m12j17g9n13g14g8e11k18f8f7h9h8f10f9f11f5i8h6g7i6k8l7m8m6o13o15n14n15m14n12l12m11n10o11m9d11m10c11l11l10l9n8m13o12m7p10n9o9l6k5m5j4"; //~2 seconds or longer, win in 101 moves
  //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15g11h11i9i14g12h13j9l13j7h15h16h12h17f12i16f13j16m12j17g9n13g14g8e11k18f8f7h9h8f10f9f11f5i8h6h7g7";//cannot vcf
  //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15g11h11i9i14g12h13j9l13j7h15h16h12h17f12i16f13j16m12j17g9n13g14g8e11k18f8f7h9h8f10f9f11f5i8h6h7k8";//can vcf
  //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15i14h12i15g15j15f16j16k17g13l18l14i17m13h18j17l17h17m17";//2 moves win
  //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15g11h11i9i14g12h13j9l13j7h15h16h12h17f12i16f13j16m12j17g9n13g14g8e11k18f8f7h9h8f10f9f11f5i8h6h7k8g7i5e9j4";
  //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15k12";
  //string initialSequence = "j10i11h10j9k7g9g11m6m5k8l6m4a1m8l8";//white has a four
  //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15n12";//even a bit difficult for katago
  //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15g11h11i9g12l13i14i16f12h16e13d12j14s19";//require 20s (5e5 nodes), even a bit difficult for katago
  //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15g11h11i9g12l13i14i16f12h16e13d12j14i15g15g17f14f16g9m12j9m14f13m11g13n15n14o13e15d16h12c17g14n12e16f9e9o14g8p13h7p12i8h8f8i7j7g7h9f6g6g5g10g4k6l6k12l8l5o16i6m4k5m5j17j5k17j18k9m15k10k4m9k7n10k3o9k2n9n6n7l4"; //109 moves to win
  string initialSequence = "j10c2p5k11l12q15d15";//4 useless white stones
  //string initialSequence = "j10c2p5k11j12k12i9";//4 useless white stones
  //string initialSequence = "j10f5d3k11l12d4e4";//4 useless white stones
  //string initialSequence = "j10d4e3k9i11e5g5f2j9a1c3";//first move must be purely defense
  //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15g11h11i9i14g12h13j9l13j7h15h16h12h17f12i16f13j16m12j17g9n13g14g8e11k18f8f7h9h8f10f9f11f5i8h6h7j8g7i5e9j4i7i6i10i4j6g6k6f6h4g3g5f2";//1 move win




  //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15g11h11i9g12l13i14i16f12h16e13d12j14pass";



  vector<Loc> initialLocSeq = Location::parseSequenceGom(initialSequence, board);
  PlayUtils::playMoveLocSequence(board, board.nextPla, initialLocSeq);
  
  // Create NNUE input parameters
  MiscNNInputParams nnInputParams;
  
  // Create NNUEBoardHistory
  NNUEBoardHistory nnueHistory(weights, nnInputParams, true);
  nnueHistory.clear(board, board.nextPla, rules);
  
  cout << "Initial board position:" << endl;
  Board::printBoard(cout, board, board.firstLoc, NULL);
  cout << endl;
  
  cout << "Move history: " << initialSequence << endl;
  cout << "Total moves: " << board.movenum << endl;
  cout << "Next player: " << (board.nextPla == C_BLACK ? "Black" : "White") << endl;
  cout << endl;
  
  // Initialize MCTS cache
  NNUE_VCF_MCTSsearch::MCTS_CacheTable cachetable(25, 11);
  
  // Create MCTS search instance
  NNUE_VCF_MCTSsearch::MCTSsearch mcts(&cachetable ,&nnueHistory, board.nextPla);
  //NNUE_VCF_MCTSsearch::MCTSsearch mcts(nullptr, &nnueHistory, board.nextPla);
  
  // Set MCTS parameters
  mcts.params.puct = 0.3;
  mcts.params.expandFactor = 0.2;
  mcts.params.policyTemp = 1.1;
  mcts.params.puctPow = 0.75;
  mcts.params.fpuReductionConst = 0.0;
  mcts.params.fpuReductionPolicy = 0.2;
  mcts.params.localPolicyBonusStage1 = 0.0;
  
  // Test different search factors (visits = factor * 1000)
  vector<double> testFactors = {};
  for (int i = 0; i < 40; i++)
    testFactors.push_back(pow(2.0,i));



  auto startTime = chrono::high_resolution_clock::now();
  for (double factor : testFactors) {
    cout << "Testing MCTS search with factor " << factor << " (" << (int)(factor) << " visits)..." << endl;
    
    bool result = mcts.vcfSearchAutoStop(&nnueHistory, board.nextPla, factor);
    auto endTime = chrono::high_resolution_clock::now();
    
    auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime);
    
    cout << "Search factor: " << factor << endl;
    cout << "Search result: " << result << endl;
    cout << "Search time: " << duration.count() << " ms" << endl;
    cout << "Root visits: " << mcts.getRootVisit() << endl;
    
    // Get and display principal variation
    vector<pair<Loc, uint64_t>> pv = mcts.getPV();
    cout << "Principal Variation (" << pv.size() << " moves): ";
    for (size_t i = 0; i < pv.size(); i++) {
      if (i > 0) cout << " ";
      cout << Location::toString(pv[i].first, board) << "(" << pv[i].second << ")";
    }
    cout << endl;
    
    // Check if win/loss is determined
    if (mcts.rootNode && mcts.rootNode->isWinDetermined) {
      cout << "Result interpretation: " << (mcts.rootNode->winner == board.nextPla ? "Win" : "Loss")
           << " determined in " << abs(mcts.rootNode->stepsToWin) << " steps" << endl;
      
      // Calculate winning dependency tree size
      auto calcStartTime = chrono::high_resolution_clock::now();
      int64_t treeSize = mcts.calculateWinningDependencyTreeSize();
      auto calcEndTime = chrono::high_resolution_clock::now();
      auto calcDuration = chrono::duration_cast<chrono::microseconds>(calcEndTime - calcStartTime);
      
      cout << "Winning dependency tree size: " << treeSize << " nodes" << endl;
      cout << "Tree calculation time: " << calcDuration.count() << " microseconds" << endl;
      
      cout << "Win/Loss determined, stopping further searches." << endl;
      cout << "----------------------------------------" << endl;
      break; // 检测到必胜/必败时直接退出循环
    } else {
      cout << "Result interpretation: Uncertain outcome (value: " << mcts.getRootValue() << ")" << endl;
    }
    cout << "----------------------------------------" << endl;
  }

  if (mcts.rootNode->winner == board.nextPla)
  {
    // Calculate defense dependency map
    auto mapStartTime = chrono::high_resolution_clock::now();
    vector<int8_t> dependMap = mcts.calculateDefenseDependencyMap();
    auto mapEndTime = chrono::high_resolution_clock::now();
    auto mapDuration = chrono::duration_cast<chrono::microseconds>(mapEndTime - mapStartTime);

    cout << "Defense dependency map calculation time: " << mapDuration.count() << " microseconds" << endl;
    VCFLogic::printBoardWithDependencyMap(board, dependMap);
  }
  cout << "MCTS search test completed." << endl;
}



void testMCTSSearch3(const ModelWeight* weights) {
  cout << "Starting MCTS search test with specific initial position..." << endl;
  
  // Create board and rules
  Board board(19, 19);
  
  // Set up the same initial position as AB search test
  //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15g11h11i9i14g12h13j9l13j7h15h16h12h17f12i16f13j16m12j17g9n13g14g8e11k18f8f7h9h8f10f9f11f5i8h6g7i6k8l7m8m6o13o15n14n15m14n12l12m11n10o11m9d11m10c11l11l10l9n8m13o12m7p10n9o9l6k5m5j4"; //~2 seconds or longer, win in 101 moves
  //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15g11h11i9i14g12h13j9l13j7h15h16h12h17f12i16f13j16m12j17g9n13g14g8e11k18f8f7h9h8f10f9f11f5i8h6h7g7";//cannot vcf
  //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15g11h11i9i14g12h13j9l13j7h15h16h12h17f12i16f13j16m12j17g9n13g14g8e11k18f8f7h9h8f10f9f11f5i8h6h7k8";//can vcf
  //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15i14h12i15g15j15f16j16k17g13l18l14i17m13h18j17l17h17m17";//2 moves win
  //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15g11h11i9i14g12h13j9l13j7h15h16h12h17f12i16f13j16m12j17g9n13g14g8e11k18f8f7h9h8f10f9f11f5i8h6h7k8g7i5e9j4";
  //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15k12";
  //string initialSequence = "j10i11h10j9k7g9g11m6m5k8l6m4a1m8l8";//white has a four
  //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15n12";//even a bit difficult for katago
  string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15g11h11i9g12l13i14i16f12h16e13d12j14s19";//require 20s (5e5 nodes), even a bit difficult for katago
  //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15g11h11i9g12l13i14i16f12h16e13d12j14i15g15g17f14f16g9m12j9m14f13m11g13n15n14o13e15d16h12c17g14n12e16f9e9o14g8p13h7p12i8h8f8i7j7g7h9f6g6g5g10g4k6l6k12l8l5o16i6m4k5m5j17j5k17j18k9m15k10k4m9k7n10k3o9k2n9n6n7l4"; //109 moves to win
  //string initialSequence = "j10c2p5k11l12q15d15";//4 useless white stones
  //string initialSequence = "j10c2p5k11j12k12i9";//4 useless white stones
  //string initialSequence = "j10f5d3k11l12d4e4";//4 useless white stones
  //string initialSequence = "j10d4e3k9i11e5g5f2j9a1c3";//first move must be purely defense
  //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15g11h11i9i14g12h13j9l13j7h15h16h12h17f12i16f13j16m12j17g9n13g14g8e11k18f8f7h9h8f10f9f11f5i8h6h7j8g7i5e9j4i7i6i10i4j6g6k6f6";//1 move win
  //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15g11h11i9i14g12h13j9l13j7h15h16h12h17f12i16f13j16m12j17g9n13g14g8e11k18f8f7h9h8f10f9f11f5i8h6h7j8g7i5e9j4i7i6i10i4j6g6k6f6h4g3g5f2";//0 move win




  //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15g11h11i9g12l13i14i16f12h16e13d12j14pass";



  vector<Loc> initialLocSeq = Location::parseSequenceGom(initialSequence, board);
  PlayUtils::playMoveLocSequence(board, board.nextPla, initialLocSeq);
  
  // Initialize MCTS cache
  NNUE_VCF_MCTSsearch::MCTS_CacheTable cachetable(25, 11);
  VCFCalculator vcfCalculator(&cachetable, weights);


  
  cout << "Initial board position:" << endl;
  Board::printBoard(cout, board, board.firstLoc, NULL);
  cout << endl;
  
  cout << "Move history: " << initialSequence << endl;
  cout << "Total moves: " << board.movenum << endl;
  cout << "Next player: " << (board.nextPla == C_BLACK ? "Black" : "White") << endl;
  cout << endl;
  
  
  
  
  // Test different search factors (visits = factor * 1000)
  double searchFactor=1e6;



  auto startTime = chrono::high_resolution_clock::now();

  std::vector<int8_t> dependMap;
  int vcfmovenum = vcfCalculator.calculateShortestVCFAndDependMap(board, board.nextPla, 109, 0, 0, searchFactor, dependMap, false);


  auto endTime = chrono::high_resolution_clock::now();
  
  auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime);
  
  cout << "Search factor: " << searchFactor << endl;
  cout << "Search time: " << duration.count() << " ms" << endl;
  VCFLogic::printBoardWithDependencyMap(board,dependMap);
  cout<<"vcfmove:"<<vcfmovenum<<endl;

}

void testVCFPrune1(const ModelWeight* weights) {
  cout << "Starting VCF Prune test..." << endl;
  
  // Create board and rules
  Board board(19, 19);
  
  // Set up a test position for stage 1 (defender's turn)
  //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15g11h11i9g12l13i14i16f12h16e13d12j14"; 
  string initialSequence = "j10c2p5k11k9k10";//2 useless white stones
  
  vector<Loc> initialLocSeq = Location::parseSequenceGom(initialSequence, board);
  PlayUtils::playMoveLocSequence(board, board.nextPla, initialLocSeq);
  
  // Initialize VCF calculator
  NNUE_VCF_MCTSsearch::MCTS_CacheTable cachetable(25, 11);
  VCFCalculator vcfCalculator(&cachetable, weights);
  
  cout << "Initial board position:" << endl;
  Board::printBoard(cout, board, board.firstLoc, NULL);
  cout << endl;
  
  cout << "Move history: " << initialSequence << endl;
  cout << "Total moves: " << board.movenum << endl;
  cout << "Next player: " << (board.nextPla == C_BLACK ? "Black" : "White") << endl;
  cout << "Board stage: " << board.stage << endl;
  cout << endl;
  
  // Calculate VCF prune results for stage 1
  auto startTime = chrono::high_resolution_clock::now();
  
  Color attackPlayer = getOpp(board.nextPla); // Attack player is opposite of current player
  int maxMove = 109;
  double searchFactor = 1e5;
  
  std::vector<VCFPrunedInfo> pruneResults = vcfCalculator.CalculateAllVCFDefendResults(
    board, attackPlayer, maxMove, searchFactor);
  
  auto endTime = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime);
  
  cout << "VCF Prune calculation time: " << duration.count() << " ms" << endl;
  cout << "Found " << pruneResults.size() << " prune results" << endl;
  cout << endl;
  
  // Display the prune information on the board
  VCFLogic::printBoardWithPruneInfo(board, pruneResults);
  
  // Print detailed prune information
  cout << "Detailed prune information:" << endl;
  for (const auto& info : pruneResults) {
    string locStr = Location::toString(info.loc, board);
    cout << "  " << locStr << ": ";
    if (info.isPruned) {
      cout << "PRUNED (opponent wins in " << info.moveNum << " moves)";
    } else if (info.notPruned) {
      cout << "SAFE (opponent cannot VCF)";
    } else {
      cout << "UNKNOWN";
    }
    cout << endl;
  }
  
  cout << "VCF Prune test completed." << endl;
}

void testVCFPrune2(const ModelWeight* weights) {
    cout << "Starting VCF Prune test..." << endl;

    // Create board and rules
    Board board(19, 19);

    // Set up a test position for stage 1 (defender's turn)
    //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15g11h11i9g12l13i14i16f12h16e13d12j14"; //stage1
    //string initialSequence = "j10c2p5k11k9";//stage0, 2 useless white stones
    //string initialSequence = "j10k11i11j11i13";//stage0,normal 5 moves
    //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15g11h11i9";//stage0, normal moves
    //string initialSequence = "j10a1s1k11k9j6";//stage1, 2 useless white stones
    //string initialSequence = "j10c2d18k11k9";//stage0, 2 useless white stones
    string initialSequence = "j10h8l8j9k10";//stage0,2 weak 2nd moves
    //string initialSequence = "j10n15o14i9h8o16m16g7g9";//stage0, black has a four now
    //string initialSequence = "j10n15o14i9h8o16m16g7g9l10";//stage1, black has a four now
    //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15g11h11i9g12l13i14i16f12h16e13d12j14i15g15g17f14f16g9m12g13k9f13m11e15d16h12c17g14";//stage1, black have a long vcf if white pass
    //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15g11h11i9g12l13i14i16f12h16e13d12j14i15g15g17f14f16g9m12g13k9f13m11e15d16h12c17g14m10e16c14c13j7k10g7l11";//stage1, long vcf if white play at l9
    //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15g11h11i9g12l13i14i16f12h16e13d12j14i15g15g17f14f16g9m12g13k9f13m11e15d16h12c17g14m10e16c14c13j7k10g7l11h8l9n9i8b12h9d14g8f8k8e8f9j17j16n7";//stage0
    //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15g11h11i9g12l13i14i16f12h16e13d12j14i15g15g17f14f16g9m12g13k9f13m11e15d16h12c17g14m10e16c14c13j7k10g7l11h8l9n9i8b12h9d14g8f8k8e8f9j17j16n7f7m8f11d13f10e12d10f5d11";//stage1
    //string initialSequence = "j10k11i11j11i13j13h10j12h14i12p3g15g14";//stage0, the strongest 9 moves and then white play 3 useless moves
    //string initialSequence = "j10k11i11j11i13j13h10j12h14i12k14k15l15j15g11h11i9g12l13i14i16f12h16e13d12j14i15g15g17f14f16g9m12g14j9f15m11m13n14o13o15o14p13m14p14m16n15l17q12n12n16n13j17k16n17e14n18e15d13g16o9o12d16g13n11h17o16l16p9n10n9f10o8o10c17e11b18h8c13p10n7";//stage0
    vector<Loc> initialLocSeq = Location::parseSequenceGom(initialSequence, board);
    PlayUtils::playMoveLocSequence(board, board.nextPla, initialLocSeq);

    // Initialize VCF calculator
    NNUE_VCF_MCTSsearch::MCTS_CacheTable cachetable(20, 5);
    VCFCalculator vcfCalculator(&cachetable, weights);

    cout << "Initial board position:" << endl;
    Board::printBoard(cout, board, board.firstLoc, NULL);
    cout << endl;

    cout << "Move history: " << initialSequence << endl;
    cout << "Total moves: " << board.movenum << endl;
    cout << "Next player: " << (board.nextPla == C_BLACK ? "Black" : "White") << endl;
    cout << "Board stage: " << board.stage << endl;
    cout << endl;

    // Calculate VCF prune results for stage 1
    auto startTime = chrono::high_resolution_clock::now();

    Color attackPlayer = getOpp(board.nextPla); // Attack player is opposite of current player
    int maxMove = 109;
    double searchFactor = 1e6;

    std::vector<VCFPrunedInfo> pruneResults = vcfCalculator.CalculateAllVCFDefendResultsV2(
        board, attackPlayer, maxMove, searchFactor);

    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime);

    cout << "VCF Prune calculation time: " << duration.count() << " ms" << endl;
    cout << "Found " << pruneResults.size() << " prune results" << endl;
    cout << endl;

    // Display the prune information on the board
    VCFLogic::printBoardWithPruneInfo(board, pruneResults);


    cout << "VCF Prune test completed." << endl;
}
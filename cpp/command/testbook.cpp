#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/fileutils.h"
#include "../book/book.h"
#include "../program/setup.h"
#include "../command/commandline.h"
#include "../main.h"

#include "../core/using.h"

using namespace std;

static bool shouldSelectNode(ConstSymBookNode node) {
  // 检查是否轮到黑棋走
  if(node.pla() != C_BLACK) {
    return false;
  }
  
  
  vector<BookMove> moves = node.getUniqueMovesInBook();
  int winningChildCount = 0;
  
  for(const BookMove& move : moves) {
    ConstSymBookNode child = node.follow(move.move);
    if(!child.isNull() ) {
      
      // check winrate
      double winLossValue = child.recursiveValues().winLossValue;
      if(node.pla() == C_BLACK) {
        winLossValue = -winLossValue;
      }
      double winrate = 0.5 + 0.5 * winLossValue;
      if(winrate > 0.7) {
        winningChildCount++;
      }
      if(winningChildCount >= 2) {
        return true;
      }
    }
  }
  
  return false;
}

// sort
static bool compareNodes(const ConstSymBookNode& a, const ConstSymBookNode& b) {
  return a.recursiveValues().adjustedVisits > b.recursiveValues().adjustedVisits; 
}

int MainCmds::testbook(const vector<string>& args) {
  Board::initHash();
  
  ConfigParser cfg;
  string bookFile;
  string outputFile = "testbook.txt";
  
  try {
    KataGoCommandLine cmd("Test book analysis");
    cmd.addConfigFileArg("","",false);
    cmd.addOverrideConfigArg();
    
    TCLAP::ValueArg<string> bookFileArg("","book-file","Book file to analyze",true,string(),"FILE");
    TCLAP::ValueArg<string> outputFileArg("","output","Output file name",false,"testbook.txt","FILE");
    
    cmd.add(bookFileArg);
    cmd.add(outputFileArg);
    
    cmd.parseArgs(args);
    
    cmd.getConfigAllowEmpty(cfg);
    bookFile = bookFileArg.getValue();
    outputFile = outputFileArg.getValue();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }
  
  const bool logToStdoutDefault = true;
  Logger logger(&cfg, logToStdoutDefault);
  
  logger.write("Loading book from: " + bookFile);
  Book* book = Book::loadFromFile(bookFile);
  
  logger.write("Analyzing book with " + Global::uint64ToString(book->size()) + " nodes");
  
  vector<SymBookNode> allNodes = book->getAllNodes();
  
  vector<ConstSymBookNode> selectedNodes;
  
  for(SymBookNode& node : allNodes) {
    ConstSymBookNode constNode = node;
    if(shouldSelectNode(constNode)) {
      selectedNodes.push_back(constNode);
    }
  }
  
  logger.write("Found " + Global::uint64ToString(selectedNodes.size()) + " nodes matching criteria");
  
  sort(selectedNodes.begin(), selectedNodes.end(), compareNodes);
  
  ofstream outFile;
  FileUtils::open(outFile, outputFile);
  
  outFile << "TestBook Analysis Results" << endl;
  outFile << "========================" << endl;
  outFile << "Total nodes analyzed: " << book->size() << endl;
  outFile << "Nodes matching criteria: " << selectedNodes.size() << endl;
  outFile << "Criteria: Black to play, winrate > 70%, >= 2 black child nodes" << endl;
  outFile << "Sorted by: current sorting criteria (descending)" << endl;
  outFile << endl;
  
  for(size_t i = 0; i < selectedNodes.size(); i++) {
    ConstSymBookNode node = selectedNodes[i];
    
    outFile << "=== Node " << (i+1) << " ===" << endl;
    outFile << "Hash: " << node.hash().toString() << endl;
    outFile << "Player to move: " << (node.pla() == C_BLACK ? "Black" : "White") << endl;
    outFile << "WinLossValue: " << node.recursiveValues().winLossValue << endl;
    outFile << "AdjustedVisits: " << node.recursiveValues().adjustedVisits << endl;
    outFile << "Visits: " << node.recursiveValues().visits << endl;
    outFile << "Weight: " << node.recursiveValues().weight << endl;
    outFile << "MinDepthFromRoot: " << node.minDepthFromRoot() << endl;
    outFile << "MinCostFromRoot: " << node.minCostFromRoot() << endl;
    
    BoardHistory hist;
    vector<Loc> moveHistory;
    if(node.getBoardHistoryReachingHere(hist, moveHistory)) {
      Board board = hist.getRecentBoard(0);
      outFile << "Board position:" << endl;
      Board::printBoard(outFile, board, Board::NULL_LOC, nullptr);
      
      outFile << "Move sequence: ";
      for(size_t j = 0; j < moveHistory.size(); j++) {
        if(j > 0) outFile << " ";
        outFile << Location::toString(moveHistory[j], board);
      }
      outFile << endl;
    }
    
    vector<BookMove> moves = node.getUniqueMovesInBook();
    outFile << "Child moves (" << moves.size() << "):" << endl;
    for(const BookMove& move : moves) {
      ConstSymBookNode child = node.follow(move.move);
      if(!child.isNull()) {
        outFile << "  " << Location::toString(move.move, book->initialBoard) 
                << " -> Player: " << (child.pla() == C_BLACK ? "Black" : "White")
                << ", Winrate: " << (0.5 + 0.5 * child.recursiveValues().winLossValue)
                << ", Visits: " << child.recursiveValues().visits << endl;
      }
    }
    
    outFile << endl;
  }
  
  outFile.close();
  
  logger.write("Analysis complete. Results written to: " + outputFile);
  
  delete book;
  return 0;
}
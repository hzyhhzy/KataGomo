#include "../program/matchopenings.h"
#include "../main.h"

#include <atomic>
#include <sstream>
#include <thread>

namespace {
void require(bool value, const std::string& message) {
  if(!value)
    throw StringError("match openings test failed: " + message);
}
std::vector<MatchOpening> parse(const std::string& text) {
  std::istringstream in(text);
  return MatchOpenings::parse(in,"test.txt");
}
template<typename F> void fails(F f, const std::string& expected) {
  try { f(); }
  catch(const std::exception& e) {
    require(std::string(e.what()).find(expected) != std::string::npos, e.what());
    return;
  }
  throw StringError("Expected failure: " + expected);
}
}

int MainCmds::testmatchopenings(const std::vector<std::string>& args) {
  if(args.size() < 1 || args.size() > 2)
    throw StringError("Usage: katago testmatchopenings [opening-file]");
  Board::initHash();
  for(const std::string& text : {"matchOpeningFile =\n", "matchOpeningFile =   # disabled\n",
       "matchOpeningFile = \"\" # disabled\n", "matchOpeningFile = \"   \"\n"}) {
    std::istringstream in(text);
    ConfigParser cfg(in);
    require(Global::trim(cfg.getString("matchOpeningFile")).empty(),"empty config value");
  }
  for(const std::string& text : {"matchOpeningFile\n", "matchOpeningFile = \"\n", "otherKey =\n"}) {
    fails([&] { std::istringstream in(text); ConfigParser cfg(in); },"Could not parse");
  }
  Rules renju(Rules::BASICRULE_RENJU,Rules::VCNRULE_NOVC,false,0);
  auto openings = parse("\xEF\xBB\xBF# UTF8\r\n\n15 h4 J3 L5 J5 H5 # first\r\n15 A1 B1\n");
  require(openings.size() == 2,"comments and CRLF");
  auto pos = openings[0].createPosition(renju);
  require(pos.board.numStonesOnBoard() == 5 && pos.pla == P_WHITE,"five stones and white to move");
  require(pos.hist.moveHistory.size() == 5 && pos.hist.initialBoard.numStonesOnBoard() == 0,"full opening history retained");
  require(openings[1].createPosition(renju).pla == P_BLACK,"even move count");
  for(const std::string& bad : {"", "# only comment\n", "15\n", "0 A1\n", "100 A1\n",
       "x A1\n", "15 A1 A1\n", "15 Q1\n", "15 A0\n", "15 I1\n", "15 pass\n", "15 (0,0)\n"})
    fails([&] { parse(bad); },"matchOpeningFile");
  fails([&] { MatchOpenings::load("/__nonexistent_match_openings__/missing.txt"); },"Could not open");
  fails([&] { parse("15 A1 A2 B1 B2 C1 C2 D1 D2 E1\n")[0].createPosition(renju); },"finished");
  fails([&] { parse("15 A1 A2 B1 B2 C1 C2 D1 D2 E1 F2\n")[0].createPosition(renju); },"illegal move");
  // H8 creates a black double-three. White's remote stones do not terminate.
  fails([&] { parse("15 G8 A1 J8 C1 H7 E1 H9 G1 H8\n")[0].createPosition(renju); },"illegal move");
  fails([&] { MatchOpenings::assignmentForGame(-1,2); },"schedule");
  fails([&] { MatchOpenings::assignmentForGame(0,0); },"schedule");
  for(int n : {1,2,3,4,5,7,1000,1965}) {
    std::atomic<int64_t> next(0);
    std::vector<MatchOpenings::Assignment> assignments(n);
    std::vector<std::thread> threads;
    for(int t = 0; t < 8; t++) {
      threads.emplace_back([&] {
        while(true) {
          int64_t i = next.fetch_add(1);
          if(i >= n) break;
          assignments[i] = MatchOpenings::assignmentForGame(i,3);
          std::this_thread::yield();
        }
      });
    }
    for(auto& thread : threads) thread.join();
    for(int i = 0; i < n; i++) {
      const auto& a = assignments[i];
      require(a.openingIndex == static_cast<size_t>((i/2)%3),"ordered cyclic openings");
      require(a.blackBotIndex == i%2 && a.whiteBotIndex == 1-i%2,"alternating bot colors");
    }
  }
  if(args.size() == 2) {
    auto library = MatchOpenings::load(args[1]);
    for(const auto& opening : library) opening.createPosition(renju);
    std::cout << "Validated " << library.size() << " Renju openings" << std::endl;
  }
  std::cout << "match openings tests PASS" << std::endl;
  return 0;
}

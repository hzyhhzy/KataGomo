#include "../program/matchopenings.h"
#include "../core/fileutils.h"

#include <sstream>

namespace {
std::string location(const std::string& source, size_t line) {
  return "matchOpeningFile " + source + ":" + std::to_string(line) + ": ";
}
}

std::vector<MatchOpening> MatchOpenings::load(const std::string& path) {
  std::ifstream in;
  if(!FileUtils::tryOpen(in,path))
    throw StringError("Could not open matchOpeningFile: " + path);
  return parse(in,path);
}

std::vector<MatchOpening> MatchOpenings::parse(std::istream& in, const std::string& source) {
  std::vector<MatchOpening> openings;
  std::string line;
  size_t lineNumber = 0;
  while(std::getline(in,line)) {
    lineNumber++;
    // Accept UTF-8 BOM, CRLF, and inline comments.
    if(lineNumber == 1 && line.compare(0,3,"\xEF\xBB\xBF") == 0)
      line.erase(0,3);
    line = line.substr(0,line.find('#'));
    std::istringstream fields(line);
    std::string token;
    if(!(fields >> token))
      continue;
    MatchOpening opening;
    opening.source = source;
    opening.lineNumber = lineNumber;
    if(!Global::tryStringToInt(token,opening.boardSize) || opening.boardSize < 1 || opening.boardSize > Board::MAX_LEN)
      throw StringError(location(source,lineNumber) + "invalid board size: " + token);
    Board board(opening.boardSize,opening.boardSize);
    Player pla = P_BLACK;
    while(fields >> token) {
      Loc loc;
      // Restrict to human-readable GTP coordinates; Location also accepts raw (x,y).
      bool coordinate = token.size() >= 2 &&
        ((token[0] >= 'A' && token[0] <= 'Z') || (token[0] >= 'a' && token[0] <= 'z'));
      for(size_t i = 1; i < token.size(); i++)
        coordinate = coordinate && token[i] >= '0' && token[i] <= '9';
      if(!coordinate || !Location::tryOfString(token,board,loc) || loc == Board::PASS_LOC || !board.isLegal(loc,pla))
        throw StringError(location(source,lineNumber) + "invalid or occupied coordinate: " + token);
      opening.moves.push_back(loc);
      board.playMoveAssumeLegal(loc,pla);
      pla = getOpp(pla);
    }
    if(opening.moves.empty())
      throw StringError(location(source,lineNumber) + "opening must contain at least one move");
    openings.push_back(std::move(opening));
  }
  if(in.bad() || (!in.eof() && in.fail()))
    throw StringError("Error reading matchOpeningFile: " + source);
  if(openings.empty())
    throw StringError("matchOpeningFile contains no openings: " + source);
  return openings;
}

InitialPosition MatchOpening::createPosition(const Rules& rules) const {
  Board board(boardSize,boardSize);
  Player pla = P_BLACK;
  BoardHistory hist(board,pla,rules);
  for(Loc loc : moves) {
    if(hist.isGameFinished || !hist.isLegal(board,loc,pla) ||
       (rules.basicRule == Rules::BASICRULE_RENJU && pla == P_BLACK && board.isForbidden(loc)))
      throw StringError(location(source,lineNumber) + "illegal move under " + rules.toString() + ": " + Location::toString(loc,board));
    hist.makeBoardMoveAssumeLegal(board,loc,pla);
    pla = getOpp(pla);
  }
  if(hist.isGameFinished)
    throw StringError(location(source,lineNumber) + "opening is already a finished game");
  return InitialPosition(board,hist,pla);
}

MatchOpenings::Assignment MatchOpenings::assignmentForGame(int64_t gameIndex, size_t numOpenings) {
  if(gameIndex < 0 || numOpenings == 0)
    throw StringError("Invalid match opening schedule");
  int black = static_cast<int>(gameIndex % 2);
  return {static_cast<size_t>((static_cast<uint64_t>(gameIndex) / 2) % numOpenings),black,1-black};
}

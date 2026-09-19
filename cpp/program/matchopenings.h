#ifndef PROGRAM_MATCHOPENINGS_H_
#define PROGRAM_MATCHOPENINGS_H_

#include "../program/play.h"

// One line: square board size, then alternating B/W GTP coordinates (black first).
// Blank lines and # comments are ignored. No passes or already-finished games.
struct MatchOpening {
  int boardSize;
  std::vector<Loc> moves;
  std::string source;
  size_t lineNumber;

  InitialPosition createPosition(const Rules& rules) const;
};

namespace MatchOpenings {
  std::vector<MatchOpening> load(const std::string& path);
  std::vector<MatchOpening> parse(std::istream& in, const std::string& source);

  struct Assignment {
    size_t openingIndex;
    int blackBotIndex;
    int whiteBotIndex;
  };
  // gameIndex is zero-based and assigned before dispatch, NOT at game completion.
  Assignment assignmentForGame(int64_t gameIndex, size_t numOpenings);
}

#endif

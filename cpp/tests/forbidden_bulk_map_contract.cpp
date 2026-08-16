#include "../forbiddenPoint/ForbiddenPointFinder.h"
#include "../forbiddenPoint/ForbiddenPointFinderBulkTest.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int BOARD_LEN = 15;
constexpr int BOARD_AREA = BOARD_LEN * BOARD_LEN;

using Position = std::array<uint8_t, BOARD_AREA>;
using ForbiddenMap = std::array<uint8_t, BOARD_AREA>;

struct DeterministicRng {
  uint64_t state;

  explicit DeterministicRng(uint64_t seed) : state(seed) {}

  uint64_t next() {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return state * UINT64_C(2685821657736338717);
  }
};

void require(bool condition, const std::string& message) {
  if(!condition)
    throw std::runtime_error(message);
}

void loadFinder(CForbiddenPointFinder& finder, const Position& position) {
  for(int y = 0; y < BOARD_LEN; y++)
    for(int x = 0; x < BOARD_LEN; x++)
      finder.SetStone(x, y, static_cast<char>(position[y * BOARD_LEN + x]));
}

// Frozen copy of the pre-bulk nearby gate. This intentionally does not call
// isForbidden(), so the differential oracle cannot accidentally share the new
// nearby-count implementation.
bool legacyIsForbidden(CForbiddenPointFinder& finder, int x, int y) {
  int nearbyBlack = 0;
  x++;
  y++;
  if(finder.cBoard[x][y] != C_EMPTY)
    return false;
  if(x >= 2 && x <= finder.f_boardsize - 1 && y >= 2 && y <= finder.f_boardsize - 1) {
    if(finder.cBoard[x + 2][y + 2] == C_BLACK) nearbyBlack++;
    if(finder.cBoard[x + 2][y] == C_BLACK) nearbyBlack++;
    if(finder.cBoard[x + 2][y - 2] == C_BLACK) nearbyBlack++;
    if(finder.cBoard[x][y + 2] == C_BLACK) nearbyBlack++;
    if(finder.cBoard[x][y - 2] == C_BLACK) nearbyBlack++;
    if(finder.cBoard[x - 2][y - 2] == C_BLACK) nearbyBlack++;
    if(finder.cBoard[x - 2][y] == C_BLACK) nearbyBlack++;
    if(finder.cBoard[x - 2][y + 2] == C_BLACK) nearbyBlack++;
    if(finder.cBoard[x + 1][y - 1] == C_BLACK) nearbyBlack++;
    if(finder.cBoard[x + 1][y] == C_BLACK) nearbyBlack++;
    if(finder.cBoard[x + 1][y + 1] == C_BLACK) nearbyBlack++;
    if(finder.cBoard[x][y - 1] == C_BLACK) nearbyBlack++;
    if(finder.cBoard[x][y + 1] == C_BLACK) nearbyBlack++;
    if(finder.cBoard[x - 1][y - 1] == C_BLACK) nearbyBlack++;
    if(finder.cBoard[x - 1][y] == C_BLACK) nearbyBlack++;
    if(finder.cBoard[x - 1][y + 1] == C_BLACK) nearbyBlack++;
  }
  else {
    for(int i = std::max(x - 2, 1); i <= std::min(x + 2, finder.f_boardsize); i++)
      for(int j = std::max(y - 2, 1); j <= std::min(y + 2, finder.f_boardsize); j++) {
        int xd = i - x;
        int yd = j - y;
        xd = xd > 0 ? xd : -xd;
        yd = yd > 0 ? yd : -yd;
        if((xd + yd) != 3 && finder.cBoard[i][j] == C_BLACK)
          nearbyBlack++;
      }
  }
  if(nearbyBlack < 2)
    return false;
  x--;
  y--;
  return finder.isForbiddenNoNearbyCheck(x, y);
}

void legacyFillForbiddenMap(CForbiddenPointFinder& finder, ForbiddenMap& map) {
  map.fill(0);
  for(int y = 0; y < BOARD_LEN; y++)
    for(int x = 0; x < BOARD_LEN; x++)
      map[y * BOARD_LEN + x] = legacyIsForbidden(finder, x, y) ? uint8_t(1) : uint8_t(0);
}

int legacyNearbyBlackCount(const Position& position, int x, int y) {
  static constexpr int offsets[16][2] = {
    {-2, -2}, {-2,  0}, {-2,  2},
    { 0, -2},           { 0,  2},
    { 2, -2}, { 2,  0}, { 2,  2},
    {-1, -1}, {-1,  0}, {-1,  1},
    { 0, -1},           { 0,  1},
    { 1, -1}, { 1,  0}, { 1,  1},
  };
  int count = 0;
  for(const auto& offset : offsets) {
    const int px = x + offset[0];
    const int py = y + offset[1];
    if(px >= 0 && px < BOARD_LEN && py >= 0 && py < BOARD_LEN &&
       position[py * BOARD_LEN + px] == C_BLACK)
      count++;
  }
  return count;
}

bool fullLineNecessary(const Position& position, int x, int y) {
  int lineBlack[4] = {};
  for(int i = 0; i < BOARD_LEN; i++) {
    lineBlack[0] += position[y * BOARD_LEN + i] == C_BLACK ? 1 : 0;
    lineBlack[1] += position[i * BOARD_LEN + x] == C_BLACK ? 1 : 0;

    const int sameY = i - x + y;
    if(sameY >= 0 && sameY < BOARD_LEN)
      lineBlack[2] += position[sameY * BOARD_LEN + i] == C_BLACK ? 1 : 0;

    const int oppositeY = x + y - i;
    if(oppositeY >= 0 && oppositeY < BOARD_LEN)
      lineBlack[3] += position[oppositeY * BOARD_LEN + i] == C_BLACK ? 1 : 0;
  }

  bool hasThreeOnOneLine = false;
  int linesWithTwo = 0;
  for(int count : lineBlack) {
    hasThreeOnOneLine |= count >= 3;
    linesWithTwo += count >= 2 ? 1 : 0;
  }
  return hasThreeOnOneLine || linesWithTwo >= 2;
}

bool looseWindowNecessary(const Position& position, int x, int y) {
  static constexpr int directions[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
  int threeDirections = 0;
  for(const auto& direction : directions) {
    bool threePotential = false;
    for(int candidateOffset = 0; candidateOffset < 6; candidateOffset++) {
      const int startX = x - candidateOffset * direction[0];
      const int startY = y - candidateOffset * direction[1];
      const int endX = startX + 5 * direction[0];
      const int endY = startY + 5 * direction[1];
      if(startX < 0 || startX >= BOARD_LEN || startY < 0 || startY >= BOARD_LEN ||
         endX < 0 || endX >= BOARD_LEN || endY < 0 || endY >= BOARD_LEN)
        continue;
      int blackCount = 0;
      bool blocked = false;
      for(int i = 0; i < 6; i++) {
        const uint8_t stone = position[
          (startY + i * direction[1]) * BOARD_LEN + startX + i * direction[0]];
        blackCount += stone == C_BLACK ? 1 : 0;
        blocked |= stone != C_EMPTY && stone != C_BLACK;
      }
      if(blocked)
        continue;
      if(blackCount >= 5)
        return true;
      threePotential |= blackCount >= 2;
    }

    for(int candidateOffset = 0; candidateOffset < 5; candidateOffset++) {
      const int startX = x - candidateOffset * direction[0];
      const int startY = y - candidateOffset * direction[1];
      const int endX = startX + 4 * direction[0];
      const int endY = startY + 4 * direction[1];
      if(startX < 0 || startX >= BOARD_LEN || startY < 0 || startY >= BOARD_LEN ||
         endX < 0 || endX >= BOARD_LEN || endY < 0 || endY >= BOARD_LEN)
        continue;
      int blackCount = 0;
      bool blocked = false;
      for(int i = 0; i < 5; i++) {
        const uint8_t stone = position[
          (startY + i * direction[1]) * BOARD_LEN + startX + i * direction[0]];
        blackCount += stone == C_BLACK ? 1 : 0;
        blocked |= stone != C_EMPTY && stone != C_BLACK;
      }
      if(!blocked && blackCount >= 3)
        return true;
    }

    if(threePotential && ++threeDirections >= 2)
      return true;
  }
  return false;
}

bool strictWindowNecessary(const Position& position, int x, int y) {
  static constexpr int directions[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
  int threeDirections = 0;
  int fourContributions = 0;
  for(const auto& direction : directions) {
    bool threePotential = false;
    for(int candidateOffset = 0; candidateOffset < 6; candidateOffset++) {
      const int startX = x - candidateOffset * direction[0];
      const int startY = y - candidateOffset * direction[1];
      const int endX = startX + 5 * direction[0];
      const int endY = startY + 5 * direction[1];
      if(startX < 0 || startX >= BOARD_LEN || startY < 0 || startY >= BOARD_LEN ||
         endX < 0 || endX >= BOARD_LEN || endY < 0 || endY >= BOARD_LEN)
        continue;
      int blackCount = 0;
      bool blocked = false;
      for(int i = 0; i < 6; i++) {
        const uint8_t stone = position[
          (startY + i * direction[1]) * BOARD_LEN + startX + i * direction[0]];
        blackCount += stone == C_BLACK ? 1 : 0;
        blocked |= stone != C_EMPTY && stone != C_BLACK;
      }
      if(blocked)
        continue;
      if(blackCount >= 5)
        return true;
      threePotential |=
        candidateOffset >= 1 && candidateOffset <= 4 && blackCount == 2 &&
        position[startY * BOARD_LEN + startX] == C_EMPTY &&
        position[endY * BOARD_LEN + endX] == C_EMPTY;
    }

    std::array<bool, BOARD_AREA> winningCompletions{};
    for(int candidateOffset = 0; candidateOffset < 5; candidateOffset++) {
      const int startX = x - candidateOffset * direction[0];
      const int startY = y - candidateOffset * direction[1];
      const int endX = startX + 4 * direction[0];
      const int endY = startY + 4 * direction[1];
      if(startX < 0 || startX >= BOARD_LEN || startY < 0 || startY >= BOARD_LEN ||
         endX < 0 || endX >= BOARD_LEN || endY < 0 || endY >= BOARD_LEN)
        continue;
      int blackCount = 0;
      bool blocked = false;
      int completion = -1;
      for(int i = 0; i < 5; i++) {
        const int pointX = startX + i * direction[0];
        const int pointY = startY + i * direction[1];
        const int point = pointY * BOARD_LEN + pointX;
        const uint8_t stone = position[point];
        blackCount += stone == C_BLACK ? 1 : 0;
        blocked |= stone != C_EMPTY && stone != C_BLACK;
        if(stone == C_EMPTY && (pointX != x || pointY != y))
          completion = point;
      }
      if(!blocked && blackCount == 3 && completion >= 0)
        winningCompletions[completion] = true;
    }
    const int completionCount = static_cast<int>(
      std::count(winningCompletions.begin(), winningCompletions.end(), true));
    fourContributions += std::min(2, completionCount);
    if(fourContributions >= 2)
      return true;
    if(threePotential && ++threeDirections >= 2)
      return true;
  }
  return false;
}

void bulkNearbyOnlyFillForbiddenMap(CForbiddenPointFinder& finder, ForbiddenMap& map) {
  map.fill(0);
  uint8_t nearbyBlack[COMPILE_MAX_BOARD_LEN + 4][COMPILE_MAX_BOARD_LEN + 4] = {};
  static constexpr int offsets[16][2] = {
    {-2, -2}, {-2,  0}, {-2,  2},
    { 0, -2},           { 0,  2},
    { 2, -2}, { 2,  0}, { 2,  2},
    {-1, -1}, {-1,  0}, {-1,  1},
    { 0, -1},           { 0,  1},
    { 1, -1}, { 1,  0}, { 1,  1},
  };
  for(int x = 0; x < BOARD_LEN; x++)
    for(int y = 0; y < BOARD_LEN; y++) {
      if(finder.cBoard[x + 1][y + 1] != C_BLACK)
        continue;
      for(const auto& offset : offsets)
        nearbyBlack[x + 2 + offset[0]][y + 2 + offset[1]]++;
    }
  for(int y = 0; y < BOARD_LEN; y++)
    for(int x = 0; x < BOARD_LEN; x++) {
      if(finder.cBoard[x + 1][y + 1] != C_EMPTY || nearbyBlack[x + 2][y + 2] < 2)
        continue;
      map[y * BOARD_LEN + x] = finder.isForbiddenNoNearbyCheck(x, y) ? uint8_t(1) : uint8_t(0);
    }
}

void fullLineFillForbiddenMap(CForbiddenPointFinder& finder, ForbiddenMap& map) {
  map.fill(0);
  uint8_t nearbyBlack[COMPILE_MAX_BOARD_LEN + 4][COMPILE_MAX_BOARD_LEN + 4] = {};
  uint8_t rowBlack[BOARD_LEN] = {};
  uint8_t columnBlack[BOARD_LEN] = {};
  uint8_t diagonalSameBlack[BOARD_LEN * 2 - 1] = {};
  uint8_t diagonalOppositeBlack[BOARD_LEN * 2 - 1] = {};
  static constexpr int offsets[16][2] = {
    {-2, -2}, {-2,  0}, {-2,  2},
    { 0, -2},           { 0,  2},
    { 2, -2}, { 2,  0}, { 2,  2},
    {-1, -1}, {-1,  0}, {-1,  1},
    { 0, -1},           { 0,  1},
    { 1, -1}, { 1,  0}, { 1,  1},
  };
  for(int x = 0; x < BOARD_LEN; x++)
    for(int y = 0; y < BOARD_LEN; y++) {
      if(finder.cBoard[x + 1][y + 1] != C_BLACK)
        continue;
      rowBlack[y]++;
      columnBlack[x]++;
      diagonalSameBlack[x - y + BOARD_LEN - 1]++;
      diagonalOppositeBlack[x + y]++;
      for(const auto& offset : offsets)
        nearbyBlack[x + 2 + offset[0]][y + 2 + offset[1]]++;
    }
  for(int y = 0; y < BOARD_LEN; y++)
    for(int x = 0; x < BOARD_LEN; x++) {
      if(finder.cBoard[x + 1][y + 1] != C_EMPTY || nearbyBlack[x + 2][y + 2] < 2)
        continue;
      const uint8_t lineBlack[4] = {
        rowBlack[y], columnBlack[x],
        diagonalSameBlack[x - y + BOARD_LEN - 1], diagonalOppositeBlack[x + y],
      };
      bool hasThreeOnOneLine = false;
      int linesWithTwo = 0;
      for(uint8_t count : lineBlack) {
        hasThreeOnOneLine |= count >= 3;
        linesWithTwo += count >= 2 ? 1 : 0;
      }
      if(!hasThreeOnOneLine && linesWithTwo < 2)
        continue;
      map[y * BOARD_LEN + x] = finder.isForbiddenNoNearbyCheck(x, y) ? uint8_t(1) : uint8_t(0);
    }
}

void looseWindowFillForbiddenMap(CForbiddenPointFinder& finder, ForbiddenMap& map) {
  map.fill(0);
  constexpr int lineMaskCount = 88;
  uint8_t nearbyBlack[COMPILE_MAX_BOARD_LEN + 4][COMPILE_MAX_BOARD_LEN + 4] = {};
  uint16_t blackLineMasks[lineMaskCount] = {};
  uint16_t blockedLineMasks[lineMaskCount] = {};
  uint16_t validLineMasks[lineMaskCount] = {};
  static constexpr int offsets[16][2] = {
    {-2, -2}, {-2,  0}, {-2,  2},
    { 0, -2},           { 0,  2},
    { 2, -2}, { 2,  0}, { 2,  2},
    {-1, -1}, {-1,  0}, {-1,  1},
    { 0, -1},           { 0,  1},
    { 1, -1}, { 1,  0}, { 1,  1},
  };
  auto lineIndex = [](int x, int y, int direction) {
    if(direction == 0) return y;
    if(direction == 1) return 15 + x;
    if(direction == 2) return 30 + (x - y + 14);
    return 59 + x + y;
  };
  auto lineBit = [](int x, int y, int direction) {
    return direction == 1 ? y : x;
  };
  auto popcount = [](uint16_t value) {
    value = static_cast<uint16_t>(value - ((value >> 1) & 0x5555));
    value = static_cast<uint16_t>((value & 0x3333) + ((value >> 2) & 0x3333));
    value = static_cast<uint16_t>((value + (value >> 4)) & 0x0f0f);
    return static_cast<uint16_t>(value * 0x0101) >> 8;
  };

  for(int x = 0; x < BOARD_LEN; x++)
    for(int y = 0; y < BOARD_LEN; y++) {
      const char stone = finder.cBoard[x + 1][y + 1];
      for(int direction = 0; direction < 4; direction++) {
        const int index = lineIndex(x, y, direction);
        const uint16_t bit = static_cast<uint16_t>(uint16_t(1) << lineBit(x, y, direction));
        validLineMasks[index] |= bit;
        if(stone == C_BLACK)
          blackLineMasks[index] |= bit;
        else if(stone != C_EMPTY)
          blockedLineMasks[index] |= bit;
      }
      if(stone == C_BLACK)
        for(const auto& offset : offsets)
          nearbyBlack[x + 2 + offset[0]][y + 2 + offset[1]]++;
    }

  for(int y = 0; y < BOARD_LEN; y++)
    for(int x = 0; x < BOARD_LEN; x++) {
      if(finder.cBoard[x + 1][y + 1] != C_EMPTY || nearbyBlack[x + 2][y + 2] < 2)
        continue;
      bool potential = false;
      int threeDirections = 0;
      for(int direction = 0; direction < 4 && !potential; direction++) {
        const int index = lineIndex(x, y, direction);
        const int candidateBit = lineBit(x, y, direction);
        const uint16_t black = blackLineMasks[index];
        const uint16_t blocked = blockedLineMasks[index];
        const uint16_t valid = validLineMasks[index];
        bool threePotential = false;
        for(int start = std::max(0, candidateBit - 5); start <= std::min(candidateBit, 9); start++) {
          const uint16_t window = static_cast<uint16_t>(uint16_t(0x3f) << start);
          if((window & valid) != window || (window & blocked) != 0)
            continue;
          const int blackCount = popcount(static_cast<uint16_t>(window & black));
          potential |= blackCount >= 5;
          threePotential |= blackCount >= 2;
        }
        for(int start = std::max(0, candidateBit - 4);
            start <= std::min(candidateBit, 10) && !potential; start++) {
          const uint16_t window = static_cast<uint16_t>(uint16_t(0x1f) << start);
          if((window & valid) == window && (window & blocked) == 0 &&
             popcount(static_cast<uint16_t>(window & black)) >= 3)
            potential = true;
        }
        potential |= threePotential && ++threeDirections >= 2;
      }
      if(potential)
        map[y * BOARD_LEN + x] = finder.isForbiddenNoNearbyCheck(x, y) ? uint8_t(1) : uint8_t(0);
    }
}

Position makeAlternatingPosition(DeterministicRng& rng, int moveCount) {
  Position position{};
  std::array<int, BOARD_AREA> order{};
  for(int i = 0; i < BOARD_AREA; i++)
    order[i] = i;
  for(int i = BOARD_AREA - 1; i > 0; i--) {
    const int j = static_cast<int>(rng.next() % static_cast<uint64_t>(i + 1));
    std::swap(order[i], order[j]);
  }
  for(int i = 0; i < moveCount; i++)
    position[order[i]] = (i & 1) == 0 ? C_BLACK : C_WHITE;
  return position;
}

std::vector<Position> makeCorrectnessCorpus(size_t randomPositions) {
  std::vector<Position> corpus;
  corpus.reserve(randomPositions + 40);

  Position empty{};
  corpus.push_back(empty);

  Position allBlack{};
  allBlack.fill(C_BLACK);
  corpus.push_back(allBlack);

  Position allWhite{};
  allWhite.fill(C_WHITE);
  corpus.push_back(allWhite);

  Position checker{};
  for(int y = 0; y < BOARD_LEN; y++)
    for(int x = 0; x < BOARD_LEN; x++)
      checker[y * BOARD_LEN + x] = ((x + y) & 1) == 0 ? C_BLACK : C_WHITE;
  corpus.push_back(checker);

  Position lines{};
  for(int x = 3; x <= 7; x++)
    lines[7 * BOARD_LEN + x] = C_BLACK;
  for(int y = 3; y <= 7; y++)
    lines[y * BOARD_LEN + 11] = C_BLACK;
  corpus.push_back(lines);

  Position overline{};
  for(int x : {3, 4, 5, 6, 8})
    overline[7 * BOARD_LEN + x] = C_BLACK;
  corpus.push_back(overline);

  Position doubleFour{};
  for(int distance : {-3, -2, -1}) {
    doubleFour[7 * BOARD_LEN + 7 + distance] = C_BLACK;
    doubleFour[(7 + distance) * BOARD_LEN + 7] = C_BLACK;
  }
  corpus.push_back(doubleFour);

  Position doubleThree{};
  for(int distance : {-1, 1}) {
    doubleThree[7 * BOARD_LEN + 7 + distance] = C_BLACK;
    doubleThree[(7 + distance) * BOARD_LEN + 7] = C_BLACK;
  }
  corpus.push_back(doubleThree);

  Position singleThree{};
  singleThree[7 * BOARD_LEN + 6] = C_BLACK;
  singleThree[7 * BOARD_LEN + 8] = C_BLACK;
  corpus.push_back(singleThree);

  // After the center move this becomes XXX.X.XXX: the two gaps are distinct
  // winning completions in one direction, so legacy counts two fours.
  Position brokenDoubleFour{};
  for(int x : {3, 4, 5, 9, 10, 11})
    brokenDoubleFour[7 * BOARD_LEN + x] = C_BLACK;
  corpus.push_back(brokenDoubleFour);

  // Two overlapping five-cell windows point at the same completion. The
  // strict filter must deduplicate that point rather than count two windows.
  Position sharedFourCompletion{};
  for(int x : {4, 5, 6, 9})
    sharedFourCompletion[7 * BOARD_LEN + x] = C_BLACK;
  sharedFourCompletion[7 * BOARD_LEN + 3] = C_WHITE;
  corpus.push_back(sharedFourCompletion);

  // Legacy overline has the unusual rule that an exact five on any line
  // suppresses an overline on another line. Exercise every ordered pair of
  // directions and both reflected placements of the sixth stone.
  static constexpr int directions[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
  for(int exactDirection = 0; exactDirection < 4; exactDirection++)
    for(int overlineDirection = 0; overlineDirection < 4; overlineDirection++) {
      if(exactDirection == overlineDirection)
        continue;
      for(int reflected : {-1, 1}) {
        Position exactFiveCrossOverline{};
        for(int distance : {-2, -1, 1, 2}) {
          const int x = 7 + distance * directions[exactDirection][0];
          const int y = 7 + distance * directions[exactDirection][1];
          exactFiveCrossOverline[y * BOARD_LEN + x] = C_BLACK;
        }
        for(int distance : {-2, -1, 1, 2, 3}) {
          const int signedDistance = distance == 3 ? 3 * reflected : distance;
          const int x = 7 + signedDistance * directions[overlineDirection][0];
          const int y = 7 + signedDistance * directions[overlineDirection][1];
          exactFiveCrossOverline[y * BOARD_LEN + x] = C_BLACK;
        }
        corpus.push_back(exactFiveCrossOverline);
      }
    }

  DeterministicRng rng(UINT64_C(0x6d2b79f5a4c31e07));
  static constexpr int moveCounts[] = {0, 1, 2, 3, 8, 16, 32, 64, 96, 128, 160, 192, 224};
  for(size_t i = 0; i < randomPositions; i++) {
    int moveCount = moveCounts[i % (sizeof(moveCounts) / sizeof(moveCounts[0]))];
    if((i & 15) == 15)
      moveCount = static_cast<int>(rng.next() % BOARD_AREA);
    Position position = makeAlternatingPosition(rng, moveCount);
    if((i & 31) == 28) {
      for(uint8_t& stone : position) {
        const uint64_t value = rng.next() % 10;
        stone = value < 5 ? C_EMPTY : (value < 8 ? C_BLACK : C_WHITE);
      }
    }
    else if((i & 31) == 29) {
      // Adversarial black imbalance, still using only valid board colors.
      for(uint8_t& stone : position)
        if(stone == C_WHITE && (rng.next() & 3) != 0)
          stone = C_BLACK;
    }
    else if((i & 31) == 30) {
      // Adversarial white blockers exercise the no-blocker window condition.
      for(uint8_t& stone : position)
        if(stone == C_BLACK && (rng.next() & 3) != 0)
          stone = C_WHITE;
    }
    else if((i & 31) == 31) {
      for(uint8_t& stone : position) {
        const uint64_t value = rng.next() % 16;
        stone = value < 3 ? C_EMPTY : (value < 13 ? C_BLACK : C_WHITE);
      }
    }
    corpus.push_back(position);
  }
  return corpus;
}

uint64_t mapChecksum(const ForbiddenMap& map) {
  uint64_t checksum = 0;
  for(size_t i = 0; i < map.size(); i++)
    checksum = checksum * UINT64_C(0x100000001b3) + map[i] + i;
  return checksum;
}

void runDifferentialContract(const std::vector<Position>& corpus) {
  uint64_t checksum = 0;
  uint64_t forbiddenCount = 0;
  uint64_t nearbyCandidates = 0;
  uint64_t fullLineCandidates = 0;
  uint64_t looseWindowCandidates = 0;
  uint64_t strictWindowCandidates = 0;
  for(size_t boardIndex = 0; boardIndex < corpus.size(); boardIndex++) {
    CForbiddenPointFinder legacyFinder(BOARD_LEN);
    CForbiddenPointFinder bulkFinder(BOARD_LEN);
    loadFinder(legacyFinder, corpus[boardIndex]);
    loadFinder(bulkFinder, corpus[boardIndex]);

    decltype(legacyFinder.cBoard) legacyBefore;
    decltype(bulkFinder.cBoard) bulkBefore;
    std::memcpy(legacyBefore, legacyFinder.cBoard, sizeof(legacyBefore));
    std::memcpy(bulkBefore, bulkFinder.cBoard, sizeof(bulkBefore));

    ForbiddenMap expected{};
    ForbiddenMap actual{};
    ForbiddenMap secondActual{};
    ForbiddenMap freshActual{};
    ForbiddenMap productionCandidates{};
    legacyFillForbiddenMap(legacyFinder, expected);
    ForbiddenPointFinderBulkTest::fillCandidateMap(bulkFinder, productionCandidates.data());
    bulkFinder.fillForbiddenMap(actual.data());
    bulkFinder.fillForbiddenMap(secondActual.data());
    CForbiddenPointFinder freshFinder(BOARD_LEN);
    loadFinder(freshFinder, corpus[boardIndex]);
    freshFinder.fillForbiddenMap(freshActual.data());

    require(
      std::memcmp(legacyFinder.cBoard, legacyBefore, sizeof(legacyBefore)) == 0,
      "legacy oracle left the finder board mutated at corpus board " + std::to_string(boardIndex));
    require(
      std::memcmp(bulkFinder.cBoard, bulkBefore, sizeof(bulkBefore)) == 0,
      "bulk implementation left the finder board mutated at corpus board " + std::to_string(boardIndex));
    require(
      std::memcmp(expected.data(), actual.data(), BOARD_AREA) == 0,
      "bulk map differs from the frozen legacy oracle at corpus board " + std::to_string(boardIndex));
    require(
      std::memcmp(expected.data(), secondActual.data(), BOARD_AREA) == 0,
      "second consecutive bulk map differs at corpus board " + std::to_string(boardIndex));
    require(
      std::memcmp(expected.data(), freshActual.data(), BOARD_AREA) == 0,
      "reused finder differs from a fresh reload at corpus board " + std::to_string(boardIndex));

    for(int y = 0; y < BOARD_LEN; y++)
      for(int x = 0; x < BOARD_LEN; x++) {
        const int pos = y * BOARD_LEN + x;
        require(actual[pos] == 0 || actual[pos] == 1, "bulk map emitted a non-boolean byte");
        const bool passesNearby =
          corpus[boardIndex][pos] == C_EMPTY && legacyNearbyBlackCount(corpus[boardIndex], x, y) >= 2;
        const bool passesFullLine = passesNearby && fullLineNecessary(corpus[boardIndex], x, y);
        const bool passesLooseWindow = passesNearby && looseWindowNecessary(corpus[boardIndex], x, y);
        const bool passesStrictWindow = passesNearby && strictWindowNecessary(corpus[boardIndex], x, y);
        require(
          productionCandidates[pos] == static_cast<uint8_t>(passesStrictWindow),
          "production candidate bitmap differs from coordinate oracle at board=" +
            std::to_string(boardIndex) + " x=" + std::to_string(x) + " y=" + std::to_string(y));
        nearbyCandidates += passesNearby ? 1 : 0;
        fullLineCandidates += passesFullLine ? 1 : 0;
        looseWindowCandidates += passesLooseWindow ? 1 : 0;
        strictWindowCandidates += passesStrictWindow ? 1 : 0;
        if(actual[pos] != 0) {
          require(passesNearby, "bulk map bypassed the legacy nearby gate");
          require(passesStrictWindow, "strict window necessary filter rejected a forbidden point");
          forbiddenCount++;
        }
      }
    checksum ^= mapChecksum(actual) + boardIndex;
  }

  std::cout << "FORBIDDEN_BULK_MAP_DIFFERENTIAL_PASS"
            << " boards=" << corpus.size()
            << " points=" << corpus.size() * BOARD_AREA
            << " forbidden=" << forbiddenCount
            << " nearby_candidates=" << nearbyCandidates
            << " full_line_candidates=" << fullLineCandidates
            << " loose_window_candidates=" << looseWindowCandidates
            << " strict_window_candidates=" << strictWindowCandidates
            << " strict_window_reduction_pct=" << std::fixed << std::setprecision(3)
            << (1.0 - static_cast<double>(strictWindowCandidates) /
                        static_cast<double>(nearbyCandidates)) * 100.0
            << " checksum=" << checksum << std::endl;
}

void runTacticalContract(const std::vector<Position>& corpus) {
  static constexpr size_t overlineIndex = 5;
  static constexpr size_t doubleFourIndex = 6;
  static constexpr size_t doubleThreeIndex = 7;
  static constexpr size_t singleThreeIndex = 8;
  static constexpr size_t brokenDoubleFourIndex = 9;
  static constexpr size_t sharedCompletionIndex = 10;
  for(const auto& expected : {
        std::pair<size_t, bool>(overlineIndex, true),
        std::pair<size_t, bool>(doubleFourIndex, true),
        std::pair<size_t, bool>(doubleThreeIndex, true),
        std::pair<size_t, bool>(singleThreeIndex, false),
        std::pair<size_t, bool>(brokenDoubleFourIndex, true),
        std::pair<size_t, bool>(sharedCompletionIndex, false),
      }) {
    CForbiddenPointFinder legacyFinder(BOARD_LEN);
    CForbiddenPointFinder candidateFinder(BOARD_LEN);
    loadFinder(legacyFinder, corpus[expected.first]);
    loadFinder(candidateFinder, corpus[expected.first]);
    ForbiddenMap candidateMap{};
    candidateFinder.fillForbiddenMap(candidateMap.data());
    const bool legacy = legacyIsForbidden(legacyFinder, 7, 7);
    require(legacy == expected.second, "unexpected frozen-legacy tactical result");
    require(candidateMap[7 * BOARD_LEN + 7] == static_cast<uint8_t>(legacy),
            "directional filter differs on named tactical position");
  }
  require(
    !fullLineNecessary(corpus[singleThreeIndex], 7, 7),
    "single-three rejection fixture unexpectedly passes the directional filter");
  require(
    !looseWindowNecessary(corpus[singleThreeIndex], 7, 7),
    "single-three rejection fixture unexpectedly passes the loose window filter");
  require(
    !strictWindowNecessary(corpus[singleThreeIndex], 7, 7),
    "single-three rejection fixture unexpectedly passes the strict window filter");
  require(
    strictWindowNecessary(corpus[brokenDoubleFourIndex], 7, 7),
    "broken double-four fixture lost its two unique completions");
  require(
    !strictWindowNecessary(corpus[sharedCompletionIndex], 7, 7),
    "overlapping windows incorrectly double-counted a shared completion");

  for(size_t boardIndex : {brokenDoubleFourIndex, sharedCompletionIndex}) {
    CForbiddenPointFinder finder(BOARD_LEN);
    loadFinder(finder, corpus[boardIndex]);
    ForbiddenMap productionCandidates{};
    ForbiddenPointFinderBulkTest::fillCandidateMap(finder, productionCandidates.data());
    require(
      productionCandidates[7 * BOARD_LEN + 7] ==
        static_cast<uint8_t>(boardIndex == brokenDoubleFourIndex),
      "production candidate map did not deduplicate winning completions");
  }

  // The next 24 boards are all exact-five/overline crosses and therefore
  // must retain the legacy exact-five suppression at their center.
  for(size_t boardIndex = 11; boardIndex < 35; boardIndex++) {
    CForbiddenPointFinder legacyFinder(BOARD_LEN);
    CForbiddenPointFinder candidateFinder(BOARD_LEN);
    loadFinder(legacyFinder, corpus[boardIndex]);
    loadFinder(candidateFinder, corpus[boardIndex]);
    ForbiddenMap candidateMap{};
    candidateFinder.fillForbiddenMap(candidateMap.data());
    require(!legacyIsForbidden(legacyFinder, 7, 7), "exact-five cross is forbidden in frozen legacy");
    require(candidateMap[7 * BOARD_LEN + 7] == 0, "exact-five cross is forbidden after filtering");
  }
  std::cout << "FORBIDDEN_WINDOW_TACTICAL_PASS"
            << " named=6 exact_five_overline_crosses=24" << std::endl;
}

void runWindowGeometryFixtures() {
  static constexpr int directions[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
  auto addThreeWindow = [](Position& position, int startX, int startY, int direction, int candidateOffset) {
    int added = 0;
    for(int offset = 1; offset <= 4 && added < 2; offset++) {
      if(offset == candidateOffset)
        continue;
      const int x = startX + offset * directions[direction][0];
      const int y = startY + offset * directions[direction][1];
      position[y * BOARD_LEN + x] = C_BLACK;
      added++;
    }
  };
  auto requireProductionCandidate = [](const Position& position, int x, int y, bool expected, const char* label) {
    CForbiddenPointFinder finder(BOARD_LEN);
    loadFinder(finder, position);
    ForbiddenMap candidates{};
    ForbiddenPointFinderBulkTest::fillCandidateMap(finder, candidates.data());
    require(
      candidates[y * BOARD_LEN + x] == static_cast<uint8_t>(expected),
      std::string("production candidate fixture differs: ") + label);
  };

  uint64_t offsetFixtures = 0;
  for(int direction = 0; direction < 4; direction++)
    for(int candidateOffset = 1; candidateOffset <= 4; candidateOffset++) {
      Position position{};
      int startX = 0;
      int startY = 7;
      if(direction == 1) {
        startX = 7;
        startY = 0;
      }
      else if(direction == 2) {
        startX = 0;
        startY = 0;
      }
      else if(direction == 3) {
        startX = 0;
        startY = 14;
      }
      const int candidateX = startX + candidateOffset * directions[direction][0];
      const int candidateY = startY + candidateOffset * directions[direction][1];
      addThreeWindow(position, startX, startY, direction, candidateOffset);

      const int secondDirection = direction == 0 ? 1 : 0;
      const int secondOffset = secondDirection == 0 ? std::min(4, std::max(1, candidateX)) : 2;
      const int secondStartX = secondDirection == 0 ? candidateX - secondOffset : candidateX;
      const int secondStartY = secondDirection == 0 ? candidateY : candidateY - secondOffset;
      addThreeWindow(position, secondStartX, secondStartY, secondDirection, secondOffset);
      require(strictWindowNecessary(position, candidateX, candidateY),
              "legal inner-offset fixture was rejected by coordinate oracle");
      requireProductionCandidate(position, candidateX, candidateY, true, "inner-offset/edge");
      offsetFixtures++;
    }

  Position endpointBlockers{};
  endpointBlockers[7 * BOARD_LEN + 6] = C_BLACK;
  endpointBlockers[7 * BOARD_LEN + 8] = C_BLACK;
  endpointBlockers[7 * BOARD_LEN + 5] = C_WHITE;
  endpointBlockers[6 * BOARD_LEN + 7] = C_BLACK;
  endpointBlockers[8 * BOARD_LEN + 7] = C_BLACK;
  endpointBlockers[5 * BOARD_LEN + 7] = C_BLACK;
  endpointBlockers[4 * BOARD_LEN + 7] = C_WHITE;
  require(!strictWindowNecessary(endpointBlockers, 7, 7),
          "black/white endpoint blockers incorrectly form two open threes");
  requireProductionCandidate(endpointBlockers, 7, 7, false, "strict empty endpoints");

  Position shortDiagonal{};
  for(const auto& point : {std::pair<int, int>(10, 0), {11, 1}, {13, 3}})
    shortDiagonal[point.second * BOARD_LEN + point.first] = C_BLACK;
  require(!strictWindowNecessary(shortDiagonal, 12, 2),
          "short diagonal incorrectly provides two four contributions");
  requireProductionCandidate(shortDiagonal, 12, 2, false, "short diagonal without six-window");

  Position asymmetricBroken{};
  for(int x : {3, 4, 5, 9, 10, 11})
    asymmetricBroken[7 * BOARD_LEN + x] = C_BLACK;
  asymmetricBroken[2 * BOARD_LEN + 1] = C_WHITE;
  asymmetricBroken[4 * BOARD_LEN + 12] = C_BLACK;
  for(int symmetry = 0; symmetry < 8; symmetry++) {
    Position transformed{};
    for(int y = 0; y < BOARD_LEN; y++)
      for(int x = 0; x < BOARD_LEN; x++) {
        int tx = x;
        int ty = y;
        if(symmetry >= 4)
          tx = BOARD_LEN - 1 - tx;
        for(int rotation = 0; rotation < (symmetry & 3); rotation++) {
          const int nextX = BOARD_LEN - 1 - ty;
          ty = tx;
          tx = nextX;
        }
        transformed[ty * BOARD_LEN + tx] = asymmetricBroken[y * BOARD_LEN + x];
      }
    CForbiddenPointFinder legacyFinder(BOARD_LEN);
    loadFinder(legacyFinder, transformed);
    require(legacyIsForbidden(legacyFinder, 7, 7), "D4 broken-four legacy fixture is not forbidden");
    requireProductionCandidate(transformed, 7, 7, true, "D4 symmetry");
  }

  std::cout << "FORBIDDEN_WINDOW_GEOMETRY_FIXTURES_PASS"
            << " inner_offsets=" << offsetFixtures
            << " endpoint_blockers=2 short_diagonal=1 symmetries=8" << std::endl;
}

void runNon15DifferentialContract() {
  std::vector<int> boardSizes = {1, 5, 14};
#if COMPILE_MAX_BOARD_LEN >= 19
  boardSizes.push_back(19);
#endif
  DeterministicRng rng(UINT64_C(0xd2f0741b3cae9658));
  uint64_t boards = 0;
  uint64_t points = 0;
  for(int boardSize : boardSizes)
    for(int boardIndex = 0; boardIndex < 64; boardIndex++) {
      CForbiddenPointFinder legacyFinder(boardSize);
      CForbiddenPointFinder candidateFinder(boardSize);
      for(int y = 0; y < boardSize; y++)
        for(int x = 0; x < boardSize; x++) {
          const uint64_t value = rng.next() % 16;
          const char stone = value < 8 ? C_EMPTY : (value < 12 ? C_BLACK : C_WHITE);
          legacyFinder.SetStone(x, y, stone);
          candidateFinder.SetStone(x, y, stone);
        }

      decltype(candidateFinder.cBoard) legacyBefore;
      decltype(candidateFinder.cBoard) candidateBefore;
      std::memcpy(legacyBefore, legacyFinder.cBoard, sizeof(legacyBefore));
      std::memcpy(candidateBefore, candidateFinder.cBoard, sizeof(candidateBefore));
      std::vector<uint8_t> expected(boardSize * boardSize, 0);
      std::vector<uint8_t> actual(boardSize * boardSize, 0);
      for(int y = 0; y < boardSize; y++)
        for(int x = 0; x < boardSize; x++)
          expected[y * boardSize + x] = legacyIsForbidden(legacyFinder, x, y) ? uint8_t(1) : uint8_t(0);
      candidateFinder.fillForbiddenMap(actual.data());
      require(
        std::memcmp(expected.data(), actual.data(), expected.size()) == 0,
        "non15 directional map differs at size=" + std::to_string(boardSize) +
          " board=" + std::to_string(boardIndex));
      require(
        std::memcmp(legacyBefore, legacyFinder.cBoard, sizeof(legacyBefore)) == 0,
        "non15 legacy finder board was mutated");
      require(
        std::memcmp(candidateBefore, candidateFinder.cBoard, sizeof(candidateBefore)) == 0,
        "non15 candidate finder board was mutated");
      boards++;
      points += static_cast<uint64_t>(boardSize * boardSize);
    }
  std::cout << "FORBIDDEN_WINDOW_NON15_DIFFERENTIAL_PASS"
            << " sizes=1,5,14"
#if COMPILE_MAX_BOARD_LEN >= 19
            << ",19"
#endif
            << " boards=" << boards
            << " points=" << points << std::endl;
}

void runMutationDifferentialContract() {
  DeterministicRng rng(UINT64_C(0x6f8d2b4193cae507));
  CForbiddenPointFinder legacyFinder(BOARD_LEN);
  CForbiddenPointFinder candidateFinder(BOARD_LEN);
  static constexpr char transitionValues[] = {C_EMPTY, C_BLACK, C_WHITE, C_WALL, '$', char(-1)};
  constexpr int mutations = 20000;
  uint64_t fullMaps = 0;
  for(int mutation = 0; mutation < mutations; mutation++) {
    const int x = static_cast<int>(rng.next() % BOARD_LEN);
    const int y = static_cast<int>(rng.next() % BOARD_LEN);
    const char stone = transitionValues[rng.next() %
      (sizeof(transitionValues) / sizeof(transitionValues[0]))];
    legacyFinder.SetStone(x, y, stone);
    candidateFinder.SetStone(x, y, stone);

    if((mutation & 63) != 63)
      continue;
    Position position{};
    for(int py = 0; py < BOARD_LEN; py++)
      for(int px = 0; px < BOARD_LEN; px++)
        position[py * BOARD_LEN + px] =
          static_cast<uint8_t>(candidateFinder.cBoard[px + 1][py + 1]);

    ForbiddenMap expected{};
    ForbiddenMap actual{};
    ForbiddenMap secondActual{};
    ForbiddenMap productionCandidates{};
    legacyFillForbiddenMap(legacyFinder, expected);
    candidateFinder.fillForbiddenMap(actual.data());
    candidateFinder.fillForbiddenMap(secondActual.data());
    ForbiddenPointFinderBulkTest::fillCandidateMap(candidateFinder, productionCandidates.data());
    require(std::memcmp(expected.data(), actual.data(), BOARD_AREA) == 0,
            "mutation map differs from frozen legacy at mutation=" + std::to_string(mutation));
    require(std::memcmp(expected.data(), secondActual.data(), BOARD_AREA) == 0,
            "second mutation map differs at mutation=" + std::to_string(mutation));
    for(int py = 0; py < BOARD_LEN; py++)
      for(int px = 0; px < BOARD_LEN; px++) {
        const int pos = py * BOARD_LEN + px;
        const bool expectedCandidate =
          position[pos] == C_EMPTY && legacyNearbyBlackCount(position, px, py) >= 2 &&
          strictWindowNecessary(position, px, py);
        require(productionCandidates[pos] == static_cast<uint8_t>(expectedCandidate),
                "mutation candidate bitmap differs from coordinate oracle");
      }
    fullMaps++;
  }

  for(int x = 0; x < BOARD_LEN + 2; x++)
    for(int y = 0; y < BOARD_LEN + 2; y++)
      require(legacyFinder.cBoard[x][y] == candidateFinder.cBoard[x][y],
              "mutation candidate and legacy active boards differ");
  std::cout << "FORBIDDEN_WINDOW_MUTATION_DIFFERENTIAL_PASS"
            << " mutations=" << mutations
            << " full_maps=" << fullMaps
            << " transitions=empty,black,white,wall,invalid" << std::endl;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

template<typename Func>
double timeBoards(Func&& func, const std::vector<CForbiddenPointFinder>& finders, int repeats, uint64_t& checksum) {
  ForbiddenMap map{};
  const auto start = std::chrono::steady_clock::now();
  for(int repeat = 0; repeat < repeats; repeat++)
    for(size_t i = 0; i < finders.size(); i++) {
      CForbiddenPointFinder finder = finders[i];
      func(finder, map);
      checksum += map[(i * 37 + static_cast<size_t>(repeat) * 17) % BOARD_AREA];
    }
  const auto stop = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::nano>(stop - start).count() /
         static_cast<double>(finders.size() * static_cast<size_t>(repeats));
}

void runMicrobenchmarkCorpus(const char* label, int minMoves, int maxMoves, uint64_t seed) {
  DeterministicRng rng(seed);
  std::vector<CForbiddenPointFinder> finders;
  std::vector<Position> positions;
  constexpr size_t corpusSize = 128;
  finders.reserve(corpusSize);
  positions.reserve(corpusSize);
  for(size_t i = 0; i < corpusSize; i++) {
    const int moveCount = minMoves + static_cast<int>((i * (maxMoves - minMoves)) / (corpusSize - 1));
    Position position = makeAlternatingPosition(rng, moveCount);
    CForbiddenPointFinder finder(BOARD_LEN);
    loadFinder(finder, position);
    finders.push_back(finder);
    positions.push_back(position);
  }

  auto legacy = [](CForbiddenPointFinder& finder, ForbiddenMap& map) {
    legacyFillForbiddenMap(finder, map);
  };
  auto bulkOnly = [](CForbiddenPointFinder& finder, ForbiddenMap& map) {
    bulkNearbyOnlyFillForbiddenMap(finder, map);
  };
  auto fullLine = [](CForbiddenPointFinder& finder, ForbiddenMap& map) {
    fullLineFillForbiddenMap(finder, map);
  };
  auto looseWindow = [](CForbiddenPointFinder& finder, ForbiddenMap& map) {
    looseWindowFillForbiddenMap(finder, map);
  };
  auto strictWindow = [](CForbiddenPointFinder& finder, ForbiddenMap& map) {
    finder.fillForbiddenMap(map.data());
  };

  uint64_t checksum = 0;
  timeBoards(legacy, finders, 1, checksum);
  timeBoards(bulkOnly, finders, 1, checksum);
  timeBoards(fullLine, finders, 1, checksum);
  timeBoards(looseWindow, finders, 1, checksum);
  timeBoards(strictWindow, finders, 1, checksum);

  std::vector<double> legacyTrials;
  std::vector<double> bulkOnlyTrials;
  std::vector<double> fullLineTrials;
  std::vector<double> looseWindowTrials;
  std::vector<double> strictWindowTrials;
  constexpr int trials = 7;
  constexpr int repeats = 2;
  for(int trial = 0; trial < trials; trial++) {
    for(int offset = 0; offset < 5; offset++) {
      switch((trial + offset) % 5) {
      case 0:
        legacyTrials.push_back(timeBoards(legacy, finders, repeats, checksum));
        break;
      case 1:
        bulkOnlyTrials.push_back(timeBoards(bulkOnly, finders, repeats, checksum));
        break;
      case 2:
        fullLineTrials.push_back(timeBoards(fullLine, finders, repeats, checksum));
        break;
      case 3:
        looseWindowTrials.push_back(timeBoards(looseWindow, finders, repeats, checksum));
        break;
      default:
        strictWindowTrials.push_back(timeBoards(strictWindow, finders, repeats, checksum));
        break;
      }
    }
  }

  const double legacyNs = median(legacyTrials);
  const double bulkOnlyNs = median(bulkOnlyTrials);
  const double fullLineNs = median(fullLineTrials);
  const double looseWindowNs = median(looseWindowTrials);
  const double strictWindowNs = median(strictWindowTrials);
  uint64_t nearbyCandidates = 0;
  uint64_t fullLineCandidates = 0;
  uint64_t looseWindowCandidates = 0;
  uint64_t strictWindowCandidates = 0;
  for(const Position& position : positions)
    for(int y = 0; y < BOARD_LEN; y++)
      for(int x = 0; x < BOARD_LEN; x++) {
        const int pos = y * BOARD_LEN + x;
        if(position[pos] != C_EMPTY || legacyNearbyBlackCount(position, x, y) < 2)
          continue;
        nearbyCandidates++;
        fullLineCandidates += fullLineNecessary(position, x, y) ? 1 : 0;
        looseWindowCandidates += looseWindowNecessary(position, x, y) ? 1 : 0;
        strictWindowCandidates += strictWindowNecessary(position, x, y) ? 1 : 0;
      }
  std::cout << std::fixed << std::setprecision(3)
            << "FORBIDDEN_WINDOW_FIVE_WAY_MICROBENCH"
            << " label=" << label
            << " move_range=" << minMoves << "-" << maxMoves
            << " corpus=" << corpusSize
            << " trials=" << trials
            << " repeats=" << repeats
            << " legacy_us_per_board=" << legacyNs / 1000.0
            << " bulk_only_us_per_board=" << bulkOnlyNs / 1000.0
            << " full_line_us_per_board=" << fullLineNs / 1000.0
            << " loose_window_us_per_board=" << looseWindowNs / 1000.0
            << " strict_window_us_per_board=" << strictWindowNs / 1000.0
            << " bulk_only_speedup=" << legacyNs / bulkOnlyNs
            << " full_line_speedup=" << legacyNs / fullLineNs
            << " loose_window_speedup=" << legacyNs / looseWindowNs
            << " strict_window_speedup=" << legacyNs / strictWindowNs
            << " strict_vs_loose_speedup=" << looseWindowNs / strictWindowNs
            << " nearby_candidates=" << nearbyCandidates
            << " full_line_candidates=" << fullLineCandidates
            << " loose_window_candidates=" << looseWindowCandidates
            << " strict_window_candidates=" << strictWindowCandidates
            << " strict_window_reduction_pct="
            << (1.0 - static_cast<double>(strictWindowCandidates) /
                        static_cast<double>(nearbyCandidates)) * 100.0
            << " checksum=" << checksum << std::endl;
}

void runMicrobenchmark() {
  runMicrobenchmarkCorpus("early", 0, 32, UINT64_C(0x72b3b4f18c029a57));
  runMicrobenchmarkCorpus("mid", 33, 96, UINT64_C(0x46ef53cf1d4b8607));
  runMicrobenchmarkCorpus("late", 97, 160, UINT64_C(0xe1b20d1f0c3ea2c9));
  runMicrobenchmarkCorpus("mixed", 8, 160, UINT64_C(0xb5ad4eceda1ce2a9));
}

}  // namespace

int main(int argc, const char* const* argv) {
  try {
    const size_t randomPositions = argc >= 2 ? static_cast<size_t>(std::stoull(argv[1])) : 4096;
    const std::vector<Position> corpus = makeCorrectnessCorpus(randomPositions);
    runDifferentialContract(corpus);
    runTacticalContract(corpus);
    runWindowGeometryFixtures();
    runNon15DifferentialContract();
    runMutationDifferentialContract();
    runMicrobenchmark();
    return 0;
  }
  catch(const std::exception& error) {
    std::cerr << "FORBIDDEN_BULK_MAP_CONTRACT_FAIL: " << error.what() << std::endl;
    return 1;
  }
}

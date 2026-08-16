#include "../forbiddenPoint/ForbiddenPointFinder.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
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
  corpus.reserve(randomPositions + 8);

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

  DeterministicRng rng(UINT64_C(0x6d2b79f5a4c31e07));
  static constexpr int moveCounts[] = {0, 1, 2, 3, 8, 16, 32, 64, 96, 128, 160, 192, 224};
  for(size_t i = 0; i < randomPositions; i++) {
    int moveCount = moveCounts[i % (sizeof(moveCounts) / sizeof(moveCounts[0]))];
    if((i & 15) == 15)
      moveCount = static_cast<int>(rng.next() % BOARD_AREA);
    Position position = makeAlternatingPosition(rng, moveCount);
    if((i & 31) == 31) {
      // Adversarial color imbalance, still using only valid board colors.
      for(uint8_t& stone : position)
        if(stone == C_WHITE && (rng.next() & 3) != 0)
          stone = C_BLACK;
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
    legacyFillForbiddenMap(legacyFinder, expected);
    bulkFinder.fillForbiddenMap(actual.data());

    require(
      std::memcmp(legacyFinder.cBoard, legacyBefore, sizeof(legacyBefore)) == 0,
      "legacy oracle left the finder board mutated at corpus board " + std::to_string(boardIndex));
    require(
      std::memcmp(bulkFinder.cBoard, bulkBefore, sizeof(bulkBefore)) == 0,
      "bulk implementation left the finder board mutated at corpus board " + std::to_string(boardIndex));
    require(
      std::memcmp(expected.data(), actual.data(), BOARD_AREA) == 0,
      "bulk map differs from the frozen legacy oracle at corpus board " + std::to_string(boardIndex));

    for(int y = 0; y < BOARD_LEN; y++)
      for(int x = 0; x < BOARD_LEN; x++) {
        const int pos = y * BOARD_LEN + x;
        require(actual[pos] == 0 || actual[pos] == 1, "bulk map emitted a non-boolean byte");
        if(actual[pos] != 0) {
          require(corpus[boardIndex][pos] == C_EMPTY, "bulk map marked an occupied point");
          require(legacyNearbyBlackCount(corpus[boardIndex], x, y) >= 2, "bulk map bypassed the legacy nearby gate");
          forbiddenCount++;
        }
      }
    checksum ^= mapChecksum(actual) + boardIndex;
  }

  std::cout << "FORBIDDEN_BULK_MAP_DIFFERENTIAL_PASS"
            << " boards=" << corpus.size()
            << " points=" << corpus.size() * BOARD_AREA
            << " forbidden=" << forbiddenCount
            << " checksum=" << checksum << std::endl;
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
  constexpr size_t corpusSize = 128;
  finders.reserve(corpusSize);
  for(size_t i = 0; i < corpusSize; i++) {
    const int moveCount = minMoves + static_cast<int>((i * (maxMoves - minMoves)) / (corpusSize - 1));
    Position position = makeAlternatingPosition(rng, moveCount);
    CForbiddenPointFinder finder(BOARD_LEN);
    loadFinder(finder, position);
    finders.push_back(finder);
  }

  auto legacy = [](CForbiddenPointFinder& finder, ForbiddenMap& map) {
    legacyFillForbiddenMap(finder, map);
  };
  auto bulk = [](CForbiddenPointFinder& finder, ForbiddenMap& map) {
    finder.fillForbiddenMap(map.data());
  };

  uint64_t checksum = 0;
  timeBoards(legacy, finders, 1, checksum);
  timeBoards(bulk, finders, 1, checksum);

  std::vector<double> legacyTrials;
  std::vector<double> bulkTrials;
  constexpr int trials = 7;
  constexpr int repeats = 2;
  for(int trial = 0; trial < trials; trial++) {
    if((trial & 1) == 0) {
      legacyTrials.push_back(timeBoards(legacy, finders, repeats, checksum));
      bulkTrials.push_back(timeBoards(bulk, finders, repeats, checksum));
    }
    else {
      bulkTrials.push_back(timeBoards(bulk, finders, repeats, checksum));
      legacyTrials.push_back(timeBoards(legacy, finders, repeats, checksum));
    }
  }

  const double legacyNs = median(legacyTrials);
  const double bulkNs = median(bulkTrials);
  const double speedup = legacyNs / bulkNs;
  const double reduction = (1.0 - bulkNs / legacyNs) * 100.0;
  std::cout << std::fixed << std::setprecision(3)
            << "FORBIDDEN_BULK_MAP_MICROBENCH"
            << " label=" << label
            << " move_range=" << minMoves << "-" << maxMoves
            << " corpus=" << corpusSize
            << " trials=" << trials
            << " repeats=" << repeats
            << " legacy_us_per_board=" << legacyNs / 1000.0
            << " bulk_us_per_board=" << bulkNs / 1000.0
            << " speedup=" << speedup
            << " reduction_pct=" << reduction
            << " checksum=" << checksum << std::endl;
}

void runMicrobenchmark() {
  runMicrobenchmarkCorpus("early", 0, 32, UINT64_C(0x72b3b4f18c029a57));
  runMicrobenchmarkCorpus("mid", 33, 96, UINT64_C(0x46ef53cf1d4b8607));
  runMicrobenchmarkCorpus("late", 97, 160, UINT64_C(0xe1b20d1f0c3ea2c9));
  runMicrobenchmarkCorpus("mixed", 8, 160, UINT64_C(0xb5ad4eceda1ce2a9));
}

}  // namespace

int main() {
  try {
    const std::vector<Position> corpus = makeCorrectnessCorpus(4096);
    runDifferentialContract(corpus);
    runMicrobenchmark();
    return 0;
  }
  catch(const std::exception& error) {
    std::cerr << "FORBIDDEN_BULK_MAP_CONTRACT_FAIL: " << error.what() << std::endl;
    return 1;
  }
}

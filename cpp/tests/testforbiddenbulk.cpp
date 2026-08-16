#include "../tests/tests.h"

#include "../forbiddenPoint/ForbiddenPointFinder.h"
#include "../neuralnet/nninputs.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int BOARD_LEN = 15;
constexpr int BOARD_AREA = BOARD_LEN * BOARD_LEN;
using Position = std::array<uint8_t, BOARD_AREA>;

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

using FillRow = void(*)(
  const Board&, const BoardHistory&, Player, const MiscNNInputParams&,
  int, int, bool, float*, float*);

void require(bool condition, const std::string& message) {
  if(!condition)
    throw std::runtime_error(message);
}

Position makePosition(DeterministicRng& rng, int moveCount) {
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

Board makeBoard(const Position& position) {
  Board board(BOARD_LEN, BOARD_LEN);
  for(int y = 0; y < BOARD_LEN; y++)
    for(int x = 0; x < BOARD_LEN; x++) {
      const uint8_t stone = position[y * BOARD_LEN + x];
      if(stone != C_EMPTY)
        require(
          board.setStone(Location::getLoc(x, y, BOARD_LEN), static_cast<Color>(stone)),
          "failed to populate feature differential board");
    }
  return board;
}

void loadLegacyFinder(CForbiddenPointFinder& finder, const Board& board) {
  for(int x = 0; x < BOARD_LEN; x++)
    for(int y = 0; y < BOARD_LEN; y++)
      finder.SetStone(x, y, board.colors[Location::getLoc(x, y, BOARD_LEN)]);
}

void runFeatureCase(
  const Board& board,
  Player nextPlayer,
  bool useNHWC,
  FillRow fillRow,
  int spatialFeatures,
  int globalFeatures,
  const char* version,
  size_t boardIndex,
  uint64_t& comparedBytes) {

  Rules rules(Rules::BASICRULE_RENJU, Rules::VCNRULE_NOVC, false, 0);
  BoardHistory hist(board, nextPlayer, rules);

  MiscNNInputParams candidateParams;
  candidateParams.useVCFInput = false;
  candidateParams.useForbiddenInput = true;
  candidateParams.useHistoryInput = false;
  candidateParams.resultsBeforeNN.inited = true;

  MiscNNInputParams referenceParams = candidateParams;
  referenceParams.useForbiddenInput = false;

  std::vector<float> actualSpatial(spatialFeatures * BOARD_AREA);
  std::vector<float> actualGlobal(globalFeatures);
  std::vector<float> expectedSpatial(spatialFeatures * BOARD_AREA);
  std::vector<float> expectedGlobal(globalFeatures);

  fillRow(
    board, hist, nextPlayer, candidateParams,
    BOARD_LEN, BOARD_LEN, useNHWC, actualSpatial.data(), actualGlobal.data());
  fillRow(
    board, hist, nextPlayer, referenceParams,
    BOARD_LEN, BOARD_LEN, useNHWC, expectedSpatial.data(), expectedGlobal.data());

  // useForbiddenInput only affects global feature 6 and spatial feature 3 or 4.
  // Rebuild those bytes using the untouched scalar legacy API.
  expectedGlobal[6] = 1.0f;
  CForbiddenPointFinder legacyFinder(BOARD_LEN);
  loadLegacyFinder(legacyFinder, board);
  const int forbiddenFeature = nextPlayer == C_BLACK ? 3 : 4;
  for(int y = 0; y < BOARD_LEN; y++)
    for(int x = 0; x < BOARD_LEN; x++) {
      if(!legacyFinder.isForbidden(x, y))
        continue;
      const int pos = y * BOARD_LEN + x;
      const int index = useNHWC ? pos * spatialFeatures + forbiddenFeature : forbiddenFeature * BOARD_AREA + pos;
      expectedSpatial[index] = 1.0f;
    }

  require(
    std::memcmp(actualSpatial.data(), expectedSpatial.data(), actualSpatial.size() * sizeof(float)) == 0,
    std::string(version) + " spatial bytes differ at board=" + std::to_string(boardIndex) +
      " player=" + std::to_string(nextPlayer) + " nhwc=" + std::to_string(useNHWC));
  require(
    std::memcmp(actualGlobal.data(), expectedGlobal.data(), actualGlobal.size() * sizeof(float)) == 0,
    std::string(version) + " global bytes differ at board=" + std::to_string(boardIndex) +
      " player=" + std::to_string(nextPlayer) + " nhwc=" + std::to_string(useNHWC));
  comparedBytes += (actualSpatial.size() + actualGlobal.size()) * sizeof(float);
}

}  // namespace

void Tests::runForbiddenBulkTests() {
  Board::initHash();

  std::vector<Position> corpus;
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

  DeterministicRng rng(UINT64_C(0x16f11fe89b0d677c));
  static constexpr int moveCounts[] = {0, 1, 2, 3, 8, 16, 32, 64, 96, 128, 160, 192, 224};
  constexpr size_t randomBoards = 512;
  for(size_t i = 0; i < randomBoards; i++) {
    int moveCount = moveCounts[i % (sizeof(moveCounts) / sizeof(moveCounts[0]))];
    if((i & 15) == 15)
      moveCount = static_cast<int>(rng.next() % BOARD_AREA);
    corpus.push_back(makePosition(rng, moveCount));
  }

  uint64_t comparedBytes = 0;
  uint64_t rows = 0;
  for(size_t boardIndex = 0; boardIndex < corpus.size(); boardIndex++) {
    const Board board = makeBoard(corpus[boardIndex]);
    for(Player nextPlayer : {C_BLACK, C_WHITE})
      for(bool useNHWC : {false, true}) {
        runFeatureCase(
          board, nextPlayer, useNHWC,
          NNInputs::fillRowV7,
          NNInputs::NUM_FEATURES_SPATIAL_V7,
          NNInputs::NUM_FEATURES_GLOBAL_V7,
          "V7", boardIndex, comparedBytes);
        rows++;
        runFeatureCase(
          board, nextPlayer, useNHWC,
          NNInputs::fillRowV101,
          NNInputs::NUM_FEATURES_SPATIAL_V101,
          NNInputs::NUM_FEATURES_GLOBAL_V101,
          "V101", boardIndex, comparedBytes);
        rows++;
      }
  }

  std::cout << "FORBIDDEN_BULK_FEATURE_DIFFERENTIAL_PASS"
            << " boards=" << corpus.size()
            << " rows=" << rows
            << " bytes=" << comparedBytes
            << " versions=V7,V101 players=black,white layouts=NCHW,NHWC" << std::endl;
}

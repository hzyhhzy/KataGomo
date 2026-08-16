#include "ForbiddenPointFinder.h"
#ifdef KATAGO_FORBIDDEN_BULK_TEST_HELPERS
#include "ForbiddenPointFinderBulkTest.h"
#endif

#include <algorithm>

namespace {

constexpr int BOARD_LEN_15 = 15;
constexpr int BOARD_AREA_15 = BOARD_LEN_15 * BOARD_LEN_15;
constexpr int LINE_MASK_COUNT_15 = 88;

int lineIndex15(int x, int y, int direction) {
	switch (direction) {
	case 0: return y;
	case 1: return 15 + x;
	case 2: return 30 + (x - y + 14);
	default: return 59 + x + y;
	}
}

int lineBit15(int x, int y, int direction) {
	return direction == 1 ? y : x;
}

int popcount16(uint16_t value) {
	value = static_cast<uint16_t>(value - ((value >> 1) & 0x5555));
	value = static_cast<uint16_t>((value & 0x3333) + ((value >> 2) & 0x3333));
	value = static_cast<uint16_t>((value + (value >> 4)) & 0x0f0f);
	return static_cast<uint16_t>(value * 0x0101) >> 8;
}

struct DirectionWindows15 {
	uint16_t six[6] = {};
	uint16_t sixEndpoints[6] = {};
	uint16_t five[5] = {};
	uint16_t candidate = 0;
	uint8_t sixCount = 0;
	uint8_t fiveCount = 0;
};

struct WindowGeometry15 {
	DirectionWindows15 windows[BOARD_AREA_15][4];

	WindowGeometry15() {
		uint16_t validLineMasks[LINE_MASK_COUNT_15] = {};
		for (int y = 0; y < BOARD_LEN_15; y++)
			for (int x = 0; x < BOARD_LEN_15; x++)
				for (int direction = 0; direction < 4; direction++)
					validLineMasks[lineIndex15(x, y, direction)] |=
						static_cast<uint16_t>(uint16_t(1) << lineBit15(x, y, direction));

		for (int y = 0; y < BOARD_LEN_15; y++)
			for (int x = 0; x < BOARD_LEN_15; x++)
				for (int direction = 0; direction < 4; direction++) {
					DirectionWindows15& geometry = windows[y * BOARD_LEN_15 + x][direction];
					const int lineIndex = lineIndex15(x, y, direction);
					const int candidateBit = lineBit15(x, y, direction);
					geometry.candidate = static_cast<uint16_t>(uint16_t(1) << candidateBit);
					for (int start = std::max(0, candidateBit - 5); start <= std::min(candidateBit, 9); start++) {
						const uint16_t window = static_cast<uint16_t>(uint16_t(0x3f) << start);
						if ((window & validLineMasks[lineIndex]) != window)
							continue;
						const int index = geometry.sixCount++;
						geometry.six[index] = window;
						geometry.sixEndpoints[index] = static_cast<uint16_t>(
							(uint16_t(1) << start) | (uint16_t(1) << (start + 5)));
					}
					for (int start = std::max(0, candidateBit - 4); start <= std::min(candidateBit, 10); start++) {
						const uint16_t window = static_cast<uint16_t>(uint16_t(0x1f) << start);
						if ((window & validLineMasks[lineIndex]) == window)
							geometry.five[geometry.fiveCount++] = window;
					}
				}
	}
};

const WindowGeometry15& windowGeometry15() {
	static const WindowGeometry15 geometry;
	return geometry;
}

bool hasStrictWindowPotential15(
	int x,
	int y,
	const uint16_t* blackLineMasks,
	const uint16_t* blockedLineMasks)
{
	const WindowGeometry15& allGeometry = windowGeometry15();
	int threeDirections = 0;
	int fourContributions = 0;
	for (int direction = 0; direction < 4; direction++) {
		const int lineIndex = lineIndex15(x, y, direction);
		const uint16_t black = blackLineMasks[lineIndex];
		const uint16_t blocked = blockedLineMasks[lineIndex];
		const DirectionWindows15& geometry = allGeometry.windows[y * BOARD_LEN_15 + x][direction];

		bool threePotential = false;
		for (int i = 0; i < geometry.sixCount; i++) {
			const uint16_t window = geometry.six[i];
			if ((window & blocked) != 0)
				continue;
			const int blackCount = popcount16(static_cast<uint16_t>(window & black));
			if (blackCount >= 5)
				return true;

			// A successful legacy open-three extension creates .BBBB. in
			// this direction. Before playing the candidate and extension, the
			// candidate is one of the inner four, both endpoints are strictly
			// empty, and exactly two of the other inner points are black.
			threePotential |=
				(geometry.candidate & geometry.sixEndpoints[i]) == 0 &&
				blackCount == 2 && ((black | blocked) & geometry.sixEndpoints[i]) == 0;
		}

		uint16_t winningCompletions = 0;
		for (int i = 0; i < geometry.fiveCount; i++) {
			const uint16_t window = geometry.five[i];
			if ((window & blocked) != 0 ||
				popcount16(static_cast<uint16_t>(window & black)) != 3)
				continue;
			// Exactly three black stones, no blocker, and the empty candidate
			// leave one unique other empty completion in this five-cell window.
			winningCompletions |= static_cast<uint16_t>(window & ~(black | geometry.candidate));
		}

		fourContributions += std::min(2, popcount16(winningCompletions));
		if (fourContributions >= 2)
			return true;
		if (threePotential && ++threeDirections >= 2)
			return true;
	}
	return false;
}

void fillCandidateMapImpl(const CForbiddenPointFinder& finder, uint8_t* candidateMap) {
	const int boardSize = finder.f_boardsize;
	const int boardArea = boardSize * boardSize;
	std::fill(candidateMap, candidateMap + boardArea, uint8_t(0));

	// These are exactly the 16 offsets counted by isForbidden. Scatter from
	// each black stone into a two-cell padded accumulator so that edge stones
	// need no special-case branches.
	uint8_t nearbyBlack[COMPILE_MAX_BOARD_LEN + 4][COMPILE_MAX_BOARD_LEN + 4] = {};
	uint16_t blackLineMasks15[LINE_MASK_COUNT_15] = {};
	uint16_t blockedLineMasks15[LINE_MASK_COUNT_15] = {};
	static constexpr int nearbyOffsets[16][2] = {
		{-2, -2}, {-2,  0}, {-2,  2},
		{ 0, -2},           { 0,  2},
		{ 2, -2}, { 2,  0}, { 2,  2},
		{-1, -1}, {-1,  0}, {-1,  1},
		{ 0, -1},           { 0,  1},
		{ 1, -1}, { 1,  0}, { 1,  1},
	};

	for (int x = 0; x < boardSize; x++)
		for (int y = 0; y < boardSize; y++) {
			const char stone = finder.cBoard[x + 1][y + 1];
			if (boardSize == BOARD_LEN_15 && stone != C_EMPTY) {
				uint16_t* lineMasks = stone == C_BLACK ? blackLineMasks15 : blockedLineMasks15;
				const uint16_t xBit = static_cast<uint16_t>(uint16_t(1) << x);
				const uint16_t yBit = static_cast<uint16_t>(uint16_t(1) << y);
				lineMasks[y] |= xBit;
				lineMasks[15 + x] |= yBit;
				lineMasks[30 + (x - y + 14)] |= xBit;
				lineMasks[59 + x + y] |= xBit;
			}
			if (stone == C_BLACK) {
				const int paddedX = x + 2;
				const int paddedY = y + 2;
				for (const auto& offset : nearbyOffsets)
					nearbyBlack[paddedX + offset[0]][paddedY + offset[1]]++;
			}
		}

	for (int y = 0; y < boardSize; y++)
		for (int x = 0; x < boardSize; x++) {
			if (finder.cBoard[x + 1][y + 1] != C_EMPTY || nearbyBlack[x + 2][y + 2] < 2)
				continue;
			if (boardSize == BOARD_LEN_15 &&
				!hasStrictWindowPotential15(x, y, blackLineMasks15, blockedLineMasks15))
				continue;
			candidateMap[y * boardSize + x] = 1;
		}
}

}  // namespace

void CForbiddenPointFinder::fillForbiddenMap(uint8_t* forbiddenMap)
{
	fillCandidateMapImpl(*this, forbiddenMap);
	for (int y = 0; y < f_boardsize; y++)
		for (int x = 0; x < f_boardsize; x++) {
			const int pos = y * f_boardsize + x;
			if (forbiddenMap[pos] != 0)
				forbiddenMap[pos] = isForbiddenNoNearbyCheck(x, y) ? uint8_t(1) : uint8_t(0);
		}
}

#ifdef KATAGO_FORBIDDEN_BULK_TEST_HELPERS
namespace ForbiddenPointFinderBulkTest {

void fillCandidateMap(const CForbiddenPointFinder& finder, uint8_t* candidateMap) {
	fillCandidateMapImpl(finder, candidateMap);
}

}
#endif

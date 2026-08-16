#include "ForbiddenPointFinder.h"

#include <algorithm>

void CForbiddenPointFinder::fillForbiddenMap(uint8_t* forbiddenMap)
{
	const int boardArea = f_boardsize * f_boardsize;
	std::fill(forbiddenMap, forbiddenMap + boardArea, uint8_t(0));

	// These are exactly the 16 offsets counted by isForbidden: the eight
	// adjacent points, the four distance-two cardinals, and the four
	// distance-two diagonals. Scatter from each black stone into a two-cell
	// padded accumulator so that edge stones need no special-case branches.
	uint8_t nearbyBlack[COMPILE_MAX_BOARD_LEN + 4][COMPILE_MAX_BOARD_LEN + 4] = {};
	static constexpr int nearbyOffsets[16][2] = {
		{-2, -2}, {-2,  0}, {-2,  2},
		{ 0, -2},           { 0,  2},
		{ 2, -2}, { 2,  0}, { 2,  2},
		{-1, -1}, {-1,  0}, {-1,  1},
		{ 0, -1},           { 0,  1},
		{ 1, -1}, { 1,  0}, { 1,  1},
	};

	for (int x = 0; x < f_boardsize; x++)
		for (int y = 0; y < f_boardsize; y++)
		{
			if (cBoard[x + 1][y + 1] != C_BLACK)
				continue;
			const int paddedX = x + 2;
			const int paddedY = y + 2;
			for (const auto& offset : nearbyOffsets)
				nearbyBlack[paddedX + offset[0]][paddedY + offset[1]]++;
		}

	for (int y = 0; y < f_boardsize; y++)
		for (int x = 0; x < f_boardsize; x++)
		{
			if (cBoard[x + 1][y + 1] != C_EMPTY || nearbyBlack[x + 2][y + 2] < 2)
				continue;
			forbiddenMap[y * f_boardsize + x] = isForbiddenNoNearbyCheck(x, y) ? uint8_t(1) : uint8_t(0);
		}
}

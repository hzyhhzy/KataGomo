#include "../game/gamelogic.h"

/*
 * gamelogic.cpp
 * Logics of game rules
 * Some other game logics are in board.h/cpp
 *
 * Gomoku as a representive
 */

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>
#include <numeric>    // For std::iota (potentially, or manual loops)
#include <set>        // Though not strictly needed for the final version of isSymmetric

using namespace std;


// --- Assume previous definitions for Board, Loc, Player, Color, Location namespace ---
// Minimal forward declarations if not fully included:
// typedef short Loc;
// typedef int8_t Color;
// typedef int8_t Player;
// static constexpr Color C_EMPTY = 0; // Example
// static constexpr int MAX_ARR_SIZE = 20*20; // Example, ensure it's large enough
// struct Board {
//     int x_size; int y_size; Color colors[MAX_ARR_SIZE];
//     Board(int x=0, int y=0): x_size(x), y_size(y) {}
//     bool isOnBoard(Loc l) const { /* ... */ return false; }
// };
// namespace Location {
//   int getX(Loc l, int xs) { return 0; }
//   int getY(Loc l, int xs) { return 0; }
//   Loc getLoc(int x, int y, int xs) { return 0; }
//   void getAdjacentOffsets(short adj_offsets[8], int x_size) { /* ... */ }
// }
// --- End of assumed definitions ---

namespace SymmetryLogic {

  class ConnectedBlock {
   public:
    int local_xsize;             // Width of this block's bounding box
    int local_ysize;             // Height of this block's bounding box
    std::vector<bool> is_stone;  // Flattened grid: true if stone of the component is present

    ConnectedBlock(int w = 0, int h = 0) : local_xsize(w), local_ysize(h) {
      if(w < 0 || h < 0) {
        local_xsize = 0;
        local_ysize = 0;
      }
      is_stone.assign(local_xsize * local_ysize, false);
    }

    void setStoneRelativeToOrigin(int local_x, int local_y) {
      if(local_x >= 0 && local_x < local_xsize && local_y >= 0 && local_y < local_ysize) {
        is_stone[local_y * local_xsize + local_x] = true;
      }
    }

    // "isStoneRelativeToOrigin没必要判断是否越界" - the current implementation does this correctly.
    // If local_x or local_y are out of bounds [0, size-1], it returns false.
    // This is desired because a reflection landing outside the bounding box means no symmetric partner *within the
    // block*.
    bool isStoneRelativeToOrigin(int local_x, int local_y) const {
      if(local_x >= 0 && local_x < local_xsize && local_y >= 0 && local_y < local_ysize) {
        return is_stone[local_y * local_xsize + local_x];
      }
      return false;
    }

    int stoneNum() const {
      int count = 0;
      for(bool s: is_stone) {
        if(s) {
          count++;
        }
      }
      return count;
    }

   private:
    // Helper to get all actual stone points in local coordinates for iteration
    // This is still useful to avoid iterating over all cells in is_stone multiple times.
    std::vector<std::pair<int, int>> getLocalStoneCoordinates() const {
      std::vector<std::pair<int, int>> coords;
      coords.reserve(stoneNum());  // Optional optimization
      for(int y = 0; y < local_ysize; ++y) {
        for(int x = 0; x < local_xsize; ++x) {
          if(isStoneRelativeToOrigin(x, y)) {  // Check internal vector directly
            coords.push_back({x, y});
          }
        }
      }
      return coords;
    }

   public:
    bool isSymmetric() const {
      if(local_xsize == 0 || local_ysize == 0)
        return true;

      std::vector<std::pair<int, int>> stone_coords = getLocalStoneCoordinates();
      if(stone_coords.empty()) {
        ASSERT_UNREACHABLE;
      } 
      else if(stone_coords.size() <= 2) {
        return true;
      }

      // Check 1: Symmetry about the central vertical line: x_reflected = local_xsize - 1 - x
      bool vertical_symmetry = true;
      for(const auto& p: stone_coords) {
        if(!isStoneRelativeToOrigin(local_xsize - 1 - p.first, p.second)) {
          vertical_symmetry = false;
          break;
        }
      }
      if(vertical_symmetry)
        return true;

      // Check 2: Symmetry about the central horizontal line: y_reflected = local_ysize - 1 - y
      bool horizontal_symmetry = true;
      for(const auto& p: stone_coords) {
        if(!isStoneRelativeToOrigin(p.first, local_ysize - 1 - p.second)) {
          horizontal_symmetry = false;
          break;
        }
      }
      if(horizontal_symmetry)
        return true;

      // Diagonal symmetries only if the bounding box is square
      if(local_xsize == local_ysize) {
        int size = local_xsize;  // or local_ysize

        // Check 3: Symmetry about the main diagonal (y=x): reflected = (y, x)
        bool main_diagonal_symmetry = true;
        for(const auto& p: stone_coords) {
          if(!isStoneRelativeToOrigin(p.second, p.first)) {
            main_diagonal_symmetry = false;
            break;
          }
        }
        if(main_diagonal_symmetry)
          return true;

        // Check 4: Symmetry about the anti-diagonal (y = -x + size-1): reflected = (size-1-y, size-1-x)
        bool anti_diagonal_symmetry = true;
        for(const auto& p: stone_coords) {
          if(!isStoneRelativeToOrigin(size - 1 - p.second, size - 1 - p.first)) {
            anti_diagonal_symmetry = false;
            break;
          }
        }
        if(anti_diagonal_symmetry)
          return true;
      }

      return false;  // No symmetry found among the four canonical axes
    }
  };

  // The getConnectedBlocks function from the previous response remains largely the same,
  // as its job is to find components and map them into the ConnectedBlock's bounding box representation.
  // For completeness, here it is again (assuming Board, Loc, Player, Color, Location namespace are defined):

  std::vector<ConnectedBlock> getConnectedBlocks(const Board& main_board, Player pla) {
    std::vector<ConnectedBlock> all_extracted_blocks;
    // Ensure MAX_ARR_SIZE is appropriate for main_board's potential size
    // If MAX_ARR_SIZE is defined in board.h, use Board::MAX_ARR_SIZE
    std::vector<bool> visited_on_main_board(main_board.x_size * (main_board.y_size + 2) + main_board.x_size + 2, false);
    // A more robust sizing for visited_on_main_board would be to use the actual max possible Loc value + 1,
    // or Board::MAX_ARR_SIZE if that's what colors array uses. The formula above is an attempt if MAX_ARR_SIZE is not
    // known. Better: Assuming Loc is an index into main_board.colors, and main_board.colors has Board::MAX_ARR_SIZE
    // elements. std::vector<bool> visited_on_main_board(Board::MAX_ARR_SIZE, false);

    short adjOffsets[8];
    Location::getAdjacentOffsets(adjOffsets, main_board.x_size);

    for(int r_board = 0; r_board < main_board.y_size; ++r_board) {
      for(int c_board = 0; c_board < main_board.x_size; ++c_board) {
        Loc start_loc_on_main = Location::getLoc(c_board, r_board, main_board.x_size);

        // Check start_loc_on_main validity before array access
        bool loc_is_valid_for_visit_array =
          (start_loc_on_main >= 0 && start_loc_on_main < visited_on_main_board.size());

        if(
          main_board.isOnBoard(start_loc_on_main) &&
          main_board.colors[start_loc_on_main] == pla &&  // Ensure color matches player
          loc_is_valid_for_visit_array && !visited_on_main_board[start_loc_on_main]) {
          std::vector<std::pair<int, int>> component_coords_global;  // Global (x,y) coords

          std::vector<Loc> q;  // Queue for BFS
          q.push_back(start_loc_on_main);
          if(loc_is_valid_for_visit_array)
            visited_on_main_board[start_loc_on_main] = true;

          int head = 0;
          while(head < q.size()) {
            Loc current_board_loc = q[head++];
            component_coords_global.push_back(
              {Location::getX(current_board_loc, main_board.x_size),
               Location::getY(current_board_loc, main_board.x_size)});

            for(int i = 0; i < 8; ++i) {  // 8-way adjacency
              Loc neighbor_loc_on_main = current_board_loc + adjOffsets[i];

              bool neighbor_valid_for_visit_array =
                (neighbor_loc_on_main >= 0 && neighbor_loc_on_main < visited_on_main_board.size());

              if(
                main_board.isOnBoard(neighbor_loc_on_main) && main_board.colors[neighbor_loc_on_main] == pla &&
                neighbor_valid_for_visit_array && !visited_on_main_board[neighbor_loc_on_main]) {
                visited_on_main_board[neighbor_loc_on_main] = true;
                q.push_back(neighbor_loc_on_main);
              }
            }
          }

          if(!component_coords_global.empty()) {
            int min_gx = component_coords_global[0].first;
            int max_gx = component_coords_global[0].first;
            int min_gy = component_coords_global[0].second;
            int max_gy = component_coords_global[0].second;

            for(size_t i = 1; i < component_coords_global.size(); ++i) {
              min_gx = std::min(min_gx, component_coords_global[i].first);
              max_gx = std::max(max_gx, component_coords_global[i].first);
              min_gy = std::min(min_gy, component_coords_global[i].second);
              max_gy = std::max(max_gy, component_coords_global[i].second);
            }

            int block_width = max_gx - min_gx + 1;
            int block_height = max_gy - min_gy + 1;

            ConnectedBlock extracted_block(block_width, block_height);

            for(const auto& global_coord: component_coords_global) {
              int local_x = global_coord.first - min_gx;
              int local_y = global_coord.second - min_gy;
              extracted_block.setStoneRelativeToOrigin(local_x, local_y);
            }
            all_extracted_blocks.push_back(extracted_block);
          }
        }
      }
    }
    return all_extracted_blocks;
  }

}  // namespace SymmetryLogic




Loc GameLogic::nearestJumpTarget(const Board& board, Loc lsrc, Loc ldst) {
  int x0 = Location::getX(lsrc, board.x_size);
  int y0 = Location::getY(lsrc, board.x_size);
  int x1 = Location::getX(ldst, board.x_size);
  int y1 = Location::getY(ldst, board.x_size);
  int dy = y1 - y0;
  int dx = x1 - x0;
  if(dx == 0 && dy == 0)
    return Board::NULL_LOC;
  if(dx != 0 && dy != 0 && dx != dy && dx != -dy)
    return Board::NULL_LOC;
  int dx0 = dx > 0 ? 1 : dx < 0 ? -1 : 0;
  int dy0 = dy > 0 ? 1 : dy < 0 ? -1 : 0;
  int x = x0 + dx0;
  int y = y0 + dy0;
  while (x != x1 || y != y1)
  {
    Loc loc = Location::getLoc(x, y, board.x_size);
    if(board.colors[loc] == C_EMPTY)
      return loc;
    x += dx0;
    y += dy0;
  }
  return Board::NULL_LOC;
}

bool isOnJumpPath(const Board& board, Loc lsrc, Loc ldst, Loc lmid) {
  if(board.colors[lmid] != C_EMPTY)
    return false;


  int x0 = Location::getX(lsrc, board.x_size);
  int y0 = Location::getY(lsrc, board.x_size);
  int x1 = Location::getX(ldst, board.x_size);
  int y1 = Location::getY(ldst, board.x_size);
  int x2 = Location::getX(lmid, board.x_size);
  int y2 = Location::getY(lmid, board.x_size);
  int dy1 = y2 - y1;
  int dx1 = x2 - x1;
  int dy2 = y2 - y0;
  int dx2 = x2 - x0;
  int t = dx1 * dx2 + dy1 * dy2;
  if(t >= 0)
    return false;
  if((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) != t * t)
    return false;
  return true;

}

bool GameLogic::isLegal(const Board& board, Player pla, Loc loc) {
  if(pla != board.nextPla) {
    std::cout << "Error next player ";
    return false;
  }

  if(loc == Board::PASS_LOC)  // pass is lose, but not illegal
    return true;

  if(!board.isOnBoard(loc))
    return false;

  if(board.stage == 0)  // place a stone
  {
    return board.colors[loc] == C_EMPTY;
  } 
  else if(board.stage == 1)  // choose a stone to move
  {
    if(board.colors[loc] == C_EMPTY)
      return false;
    Loc chosenMove = board.midLocs[0];
    return nearestJumpTarget(board, loc, chosenMove) != Board::NULL_LOC;
  }
  else if(board.stage == 2)  
  {
    return isOnJumpPath(board,board.midLocs[0], board.midLocs[1], loc);
  }
  ASSERT_UNREACHABLE;
  return false;
}

GameLogic::MovePriority GameLogic::getMovePriorityAssumeLegal(const Board& board, const BoardHistory& hist, Player pla, Loc loc) {
  

  return MP_NORMAL;
}

GameLogic::MovePriority GameLogic::getMovePriority(const Board& board, const BoardHistory& hist, Player pla, Loc loc) {
  if(loc == Board::PASS_LOC)
    return MP_NORMAL;
  if(!board.isLegal(loc, pla))
    return MP_ILLEGAL;
  MovePriority MP = getMovePriorityAssumeLegal(board, hist, pla, loc);
  return MP;
}




Color GameLogic::checkWinnerAfterPlayed(
  const Board& board,
  const BoardHistory& hist,
  Player pla,
  Loc loc,
  int oldStage) {
  if(loc == Board::PASS_LOC && oldStage != 1)
    return getOpp(pla);  // pass is not allowed
  if(board.stage == 0) {
    auto myConBlocks = SymmetryLogic::getConnectedBlocks(board, pla);
    for(int i = 0; i < myConBlocks.size(); i++) {
      auto b = myConBlocks[i];
      if(!b.isSymmetric())
        return getOpp(pla);
    }
    if (myConBlocks.size() >= 9) {
      auto oppConBlocks = SymmetryLogic::getConnectedBlocks(board, getOpp(pla));
      if(myConBlocks.size() > oppConBlocks.size())
        return getOpp(pla);
    }
  }


  return C_WALL;
}

GameLogic::ResultsBeforeNN::ResultsBeforeNN() {
  inited = false;
  winner = C_WALL;
  myOnlyLoc = Board::NULL_LOC;
}

void GameLogic::ResultsBeforeNN::init(const Board& board, const BoardHistory& hist, Color nextPlayer) {
  if(inited)
    return;
  inited = true;




  return;
}

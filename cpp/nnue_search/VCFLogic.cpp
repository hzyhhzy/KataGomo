#include "VCFLogic.h"
#include "VCFCalculator.h"
#include <vector>
#include <map>

// VCF Logic implementation
using namespace std;
int VCFLogic::checkTwoFourThreats(const Board& board, Player pla) {
  Player opp = getOpp(pla);
  vector<vector<Loc>> emptyPositionsInTuples; // Record empty positions in tuples with 4-5 pla pieces and no opp pieces
  
  // Record the status of all tuples
  bool hasPlaWin = false;     // Whether there is a tuple with all pla pieces
  
  auto checkTuple = [&](Loc loc0, int16_t adj) -> void {
     int plaCount = 0;
     int oppCount = 0;
     
     for (int i = 0; i < 6; i++) {
       Loc loc = loc0 + i * adj;
       if (!board.isOnBoard(loc))
       {
         ASSERT_UNREACHABLE;//Boundary check should be done outside this function
         return; // Invalid tuple, skip
       }
       
       Color c = board.colors[loc];
       if (board.stage == 1 && loc == board.firstLoc)
         c = board.nextPla;
       if (c == pla) {
         plaCount++;
       } else if (c == opp) {
         oppCount++;
       }
     }
     
     // Record various states
     if (plaCount == 6) {
       hasPlaWin = true;
     }
     
     // Record empty positions in tuples with 4-5 pla pieces and no opp pieces
     if (plaCount >= 4 && plaCount <= 5 && oppCount == 0) {
       vector<Loc> emptyLocs;
       for (int i = 0; i < 6; i++) {
         Loc loc = loc0 + i * adj;
         Color c = board.colors[loc];
         if (c == C_EMPTY && board.firstLoc != loc) {
           emptyLocs.push_back(loc);
         }
       }
       emptyPositionsInTuples.push_back(emptyLocs);
     }
   };
  
  // Traverse all tuples in all directions
  // +x direction (horizontal)
  for (int y = 0; y < board.y_size; y++) {
    for (int x = 0; x < board.x_size - 5; x++) {
      Loc loc0 = Location::getLoc(x, y, board.x_size);
      checkTuple(loc0, 1);
    }
  }
  
  // +y direction (vertical)
  for (int y = 0; y < board.y_size - 5; y++) {
    for (int x = 0; x < board.x_size; x++) {
      Loc loc0 = Location::getLoc(x, y, board.x_size);
      checkTuple(loc0, board.x_size + 1);
    }
  }
  
  // +x+y direction (positive diagonal)
  for (int y = 0; y < board.y_size - 5; y++) {
    for (int x = 0; x < board.x_size - 5; x++) {
      Loc loc0 = Location::getLoc(x, y, board.x_size);
      checkTuple(loc0, board.x_size + 1 + 1);
    }
  }
  
  // -x+y direction (negative diagonal)
  for (int y = 0; y < board.y_size - 5; y++) {
    for (int x = 5; x < board.x_size; x++) {
      Loc loc0 = Location::getLoc(x, y, board.x_size);
      checkTuple(loc0, board.x_size + 1 - 1);
    }
  }
  
  // Return results by priority
  if (hasPlaWin) {
    return 4; // All pla pieces
  }
  
  // If no threatening tuples, return 0
  if (emptyPositionsInTuples.empty()) {
    return 0;
  }
  
  // Check if all threats can be blocked with 1 or 2 pieces
  // Find all empty positions and remove duplicates
  set<Loc> uniqueEmptyPositions;
  for (const auto& emptyLocs : emptyPositionsInTuples) {
    for (Loc loc : emptyLocs) {
      uniqueEmptyPositions.insert(loc);
    }
  }
  vector<Loc> allEmptyPositions(uniqueEmptyPositions.begin(), uniqueEmptyPositions.end());
  
  // Check if one position can block all tuples
  for (Loc candidateLoc : allEmptyPositions) {
    bool canBlockAll = true;
    for (const auto& emptyLocs : emptyPositionsInTuples) {
      bool foundInThisTuple = false;
      for (Loc loc : emptyLocs) {
        if (loc == candidateLoc) {
          foundInThisTuple = true;
          break;
        }
      }
      if (!foundInThisTuple) {
        canBlockAll = false;
        break;
      }
    }
    if (canBlockAll) {
      return 1; // One piece can block all
    }
  }
  
  // Check if 2 pieces can block all tuples
  for (size_t i = 0; i < allEmptyPositions.size(); i++) {
    for (size_t j = i + 1; j < allEmptyPositions.size(); j++) {
      Loc loc1 = allEmptyPositions[i];
      Loc loc2 = allEmptyPositions[j];
      
      bool canBlockAll = true;
      for (const auto& emptyLocs : emptyPositionsInTuples) {
        bool foundInThisTuple = false;
        for (Loc loc : emptyLocs) {
          if (loc == loc1 || loc == loc2) {
            foundInThisTuple = true;
            break;
          }
        }
        if (!foundInThisTuple) {
          canBlockAll = false;
          break;
        }
      }
      if (canBlockAll) {
          return 2; // Two pieces can block all
        }
      }
    }
    
    return 3; // Two pieces cannot block all
}

int VCFLogic::checkMaxConnectLen(const Board& board, Player pla) {
  //if (board.stage != 0)
  //  ASSERT_UNREACHABLE;
  int maxLen = 0;
  
  auto checkDirection = [&](Loc startLoc, int16_t adj) -> int {
    int len = 0;
    Loc loc = startLoc;
    for (int i = 0; i < 6; i++)
    {
      if (!board.isOnBoard(loc))
        return 0;
      if (board.colors[loc] == getOpp(pla))
        return 0;
      if (board.colors[loc] == pla)
      {
        len++;
      }
      loc += adj;
    }
    return len;
  };
  
  // Check all positions in four directions
  for (int y = 0; y < board.y_size; y++) {
    for (int x = 0; x < board.x_size; x++) {
      Loc loc = Location::getLoc(x, y, board.x_size);
      if (board.colors[loc] == pla) {
        // Horizontal (+x direction)
        int len = checkDirection(loc, 1);
        maxLen = max(maxLen, len);
        
        // Vertical (+y direction)
        len = checkDirection(loc, board.x_size + 1);
        maxLen = max(maxLen, len);
        
        // Positive diagonal (+x+y direction)
        len = checkDirection(loc, board.x_size + 1 + 1);
        maxLen = max(maxLen, len);
        
        // Negative diagonal (-x+y direction)
        len = checkDirection(loc, board.x_size + 1 - 1);
        maxLen = max(maxLen, len);
      }
    }
  }
  
  return maxLen;
}

vector<Loc> VCFLogic::getAllVCFAttackOrDefenseLocs(const Board& board, Player attackPla, Color& winner, int& gameEndMovenum) {
  winner = C_WALL;
  gameEndMovenum = 0;
  Player defendPla = getOpp(attackPla);
  vector<Loc> locs;
  
  if (board.nextPla == attackPla) {
    // Attack player logic
    int maybeLegalMap[Board::MAX_ARR_SIZE] = {0};
    int maxAttackCount = 0;
    int maxDefendCount = 0;
    int fourCount = 0;


    
    auto checkTuple = [&](Loc loc0, int16_t adj) -> void {
      int attackCount = 0;
      int defendCount = 0;
      
      for (int i = 0; i < 6; i++) {
        Loc loc = loc0 + i * adj;
        //if (!board.isOnBoard(loc)) {
        //  ASSERT_UNREACHABLE;
        //  return; // 无效六元组，跳过
        //}
        
        Color c = board.colors[loc];
        if (board.stage == 1 && loc == board.firstLoc)
          c = board.nextPla;
        if (c == attackPla) {
          attackCount++;
        } else if (c == defendPla) {
          defendCount++;
        }
      }
      
      // Update maximum counts
        if (defendCount == 6) {
          maxDefendCount = max(maxDefendCount, defendCount);
        }
        
        if (board.stage == 0) {
          if (attackCount >= 4 && attackCount <= 6 && defendCount == 0) {
            maxAttackCount = max(maxAttackCount, attackCount);
          }
          // Record possible legal positions (2-3 attack pieces and no defend pieces)
          if (attackCount >= 2 && attackCount <= 3 && defendCount == 0) {
            for (int i = 0; i < 6; i++) {
              Loc loc = loc0 + i * adj;
              if (board.colors[loc] == C_EMPTY) {
                maybeLegalMap[loc] += 1;
              }
            }
          }
        } else { // stage == 1
          if (attackCount >= 5 && attackCount <= 6 && defendCount == 0) {
            maxAttackCount = max(maxAttackCount, attackCount);
          }
          if(attackCount == 4 && defendCount == 0) {
            fourCount++;
          }
          // Record possible legal positions (3-4 attack pieces and no defend pieces)
          if (attackCount >= 3 && attackCount <= 4 && defendCount == 0) {
            for (int i = 0; i < 6; i++) {
              Loc loc = loc0 + i * adj;
              if (board.colors[loc] == C_EMPTY && board.firstLoc != loc) {
                maybeLegalMap[loc] += 1;
              }
            }
          }
        }
    };
    
    // Traverse all tuples in all directions
    // +x direction (horizontal)
    for (int y = 0; y < board.y_size; y++) {
      for (int x = 0; x < board.x_size - 5; x++) {
        Loc loc0 = Location::getLoc(x, y, board.x_size);
        checkTuple(loc0, 1);
      }
    }
    
    // +y direction (vertical)
    for (int y = 0; y < board.y_size - 5; y++) {
      for (int x = 0; x < board.x_size; x++) {
        Loc loc0 = Location::getLoc(x, y, board.x_size);
        checkTuple(loc0, board.x_size + 1);
      }
    }
    
    // +x+y direction (positive diagonal)
    for (int y = 0; y < board.y_size - 5; y++) {
      for (int x = 0; x < board.x_size - 5; x++) {
        Loc loc0 = Location::getLoc(x, y, board.x_size);
        checkTuple(loc0, board.x_size + 1 + 1);
      }
    }
    
    // -x+y direction (negative diagonal)
    for (int y = 0; y < board.y_size - 5; y++) {
      for (int x = 5; x < board.x_size; x++) {
        Loc loc0 = Location::getLoc(x, y, board.x_size);
        checkTuple(loc0, board.x_size + 1 - 1);
      }
    }
    
    // Check win/loss conditions
    if (maxDefendCount == 6) {
      winner = defendPla;
      gameEndMovenum = board.movenum;
      return locs;
    }
    
    if (board.stage == 0) {
      if (maxAttackCount >= 4) {
        winner = attackPla;
        gameEndMovenum = board.movenum + 2;
        return locs;
      }
    } else { // stage == 1
      if (maxAttackCount >= 5) {
        winner = attackPla;
        gameEndMovenum = board.movenum + 1;
        return locs;
      }
    }

    int maybeLegalThreshold = 1;
    if (board.stage == 0) {
      maybeLegalThreshold = 0;//maybe there are enough fours
    }

    if(board.stage == 1 && fourCount == 0) {
      maybeLegalThreshold = 2;//must create two fours at once
    }

    if (board.stage == 1 && fourCount >= 2) {
      maybeLegalThreshold = 0;//maybe there are enough fours
    }

    // Collect possible legal positions
    for (int y = 0; y < board.y_size; y++) {
      for (int x = 0; x < board.x_size; x++) {
        Loc loc = Location::getLoc(x, y, board.x_size);
        if (maybeLegalMap[loc] >= maybeLegalThreshold && board.colors[loc] == C_EMPTY && loc != board.firstLoc) {
          if (board.stage == 0 || (board.getLocationPriority(loc) + Board::PRIOR_EPS >= board.firstLocPriority)) {
            locs.push_back(loc);
          }
        }
      }
    }
    
    // If no feasible positions, defend player wins
    if (locs.empty()) {
      winner = defendPla;
      gameEndMovenum = board.movenum;
    }
    
  } else {
    // Defend player logic
    vector<vector<Loc>> emptyPositionsInTuples; // Record empty positions in tuples with 4-5 attackPla pieces and no defendPla pieces
    int maxDefendCount = 0;
    int maxAttackCount = 0;
    
    auto checkTuple = [&](Loc loc0, int16_t adj) -> void {
      int attackCount = 0;
      int defendCount = 0;
      
      for (int i = 0; i < 6; i++) {
        Loc loc = loc0 + i * adj;
        //if (!board.isOnBoard(loc)) {
        //  ASSERT_UNREACHABLE;
        //  return; // 无效六元组，跳过
        //}
        
        Color c = board.colors[loc];
        if (board.stage == 1 && loc == board.firstLoc)
          c = board.nextPla;
        if (c == attackPla) {
          attackCount++;
        } else if (c == defendPla) {
          defendCount++;
        }
      }
      
      if (board.stage == 0) {
        // Update maximum counts
        if (attackCount == 6) {
          maxAttackCount = max(maxAttackCount, attackCount);
        }
        
        // Update defend player maximum counts
        if (defendCount >= 4 && defendCount <= 6 && attackCount == 0) {
          maxDefendCount = max(maxDefendCount, defendCount);
        }
        
        // Record empty positions in tuples with 4-5 attackPla pieces and no defendPla pieces
        if (attackCount >= 4 && attackCount <= 5 && defendCount == 0) {
          vector<Loc> emptyLocs;
          for (int i = 0; i < 6; i++) {
            Loc loc = loc0 + i * adj;
            if (board.colors[loc] == C_EMPTY) {
              emptyLocs.push_back(loc);
            }
          }
          emptyPositionsInTuples.push_back(emptyLocs);
        }
      } else { // stage == 1
        // Update maximum counts
        if (attackCount == 6) {
          maxAttackCount = max(maxAttackCount, attackCount);
        }
        
        // Update defend player maximum counts
        if (defendCount >= 5 && defendCount <= 6 && attackCount == 0) {
          maxDefendCount = max(maxDefendCount, defendCount);
        }
        
        // Record empty positions in tuples with 4-5 attackPla pieces and no defendPla pieces
        if (attackCount >= 4 && attackCount <= 5 && defendCount == 0) {
          vector<Loc> emptyLocs;
          for (int i = 0; i < 6; i++) {
            Loc loc = loc0 + i * adj;
            if (board.colors[loc] == C_EMPTY && board.firstLoc != loc) {
              emptyLocs.push_back(loc);
            }
          }
          emptyPositionsInTuples.push_back(emptyLocs);
        }
      }
    };
    
    // 遍历所有方向的六元组
    // +x direction (横向)
    for (int y = 0; y < board.y_size; y++) {
      for (int x = 0; x < board.x_size - 5; x++) {
        Loc loc0 = Location::getLoc(x, y, board.x_size);
        checkTuple(loc0, 1);
      }
    }
    
    // +y direction (竖向)
    for (int y = 0; y < board.y_size - 5; y++) {
      for (int x = 0; x < board.x_size; x++) {
        Loc loc0 = Location::getLoc(x, y, board.x_size);
        checkTuple(loc0, board.x_size + 1);
      }
    }
    
    // +x+y direction (正斜向)
    for (int y = 0; y < board.y_size - 5; y++) {
      for (int x = 0; x < board.x_size - 5; x++) {
        Loc loc0 = Location::getLoc(x, y, board.x_size);
        checkTuple(loc0, board.x_size + 1 + 1);
      }
    }
    
    // -x+y direction (反斜向)
    for (int y = 0; y < board.y_size - 5; y++) {
      for (int x = 5; x < board.x_size; x++) {
        Loc loc0 = Location::getLoc(x, y, board.x_size);
        checkTuple(loc0, board.x_size + 1 - 1);
      }
    }
    
    // Check win/loss conditions
    if (board.stage == 0) {
      if (maxAttackCount == 6) {
        winner = attackPla;
        gameEndMovenum = board.movenum;
        return locs;
      }
      if (maxDefendCount >= 4) {
        winner = defendPla;
        gameEndMovenum = board.movenum + 2;
        return locs;
      }
    } else { // stage == 1
      if (maxAttackCount == 6) {
        assert(false); // Should not have 6 attack pieces in stage==1
      }
      if (maxDefendCount >= 5) {
        winner = defendPla;
        gameEndMovenum = board.movenum + 1;
        return locs;
      }
    }
    
    // If no threatening tuples, return empty
    if (emptyPositionsInTuples.empty()) {
      winner = defendPla;
      gameEndMovenum = board.movenum;

      return locs;
    }

    if (emptyPositionsInTuples.size() == 1 && board.stage == 0) {
      winner = defendPla;
      gameEndMovenum = board.movenum + 1;

      return locs;
    }
    
    // Calculate positions that can block all fours
    set<Loc> uniqueEmptyPositions;
    for (const auto& emptyLocs : emptyPositionsInTuples) {
      for (Loc loc : emptyLocs) {
        uniqueEmptyPositions.insert(loc);
      }
    }
    vector<Loc> allEmptyPositions(uniqueEmptyPositions.begin(), uniqueEmptyPositions.end());
    
    if (board.stage == 0) {
      // Check if 2 pieces can block all tuples
      for (size_t i = 0; i < allEmptyPositions.size(); i++) {
        Loc loc1 = allEmptyPositions[i];
        bool canBlock = false;
        for (size_t j = 0; j < allEmptyPositions.size(); j++) {
          Loc loc2 = allEmptyPositions[j];
          if (loc2 == loc1)
            continue;
          
          // Check if second move meets priority requirements
          if (!(board.getLocationPriority(loc2) + Board::PRIOR_EPS >= board.getLocationPriority(loc1))) {
            continue;
          }
          
          bool canBlockAll = true;
          for (const auto& emptyLocs : emptyPositionsInTuples) {
            bool foundInThisTuple = false;
            for (Loc loc : emptyLocs) {
              if (loc == loc1 || loc == loc2) {
                foundInThisTuple = true;
                break;
              }
            }
            if (!foundInThisTuple) {
              canBlockAll = false;
              break;
            }
          }
          if (canBlockAll) {
            canBlock = true;
            break;
          }
        }
        if (canBlock) {
          locs.push_back(loc1);
        }
      }
      
      // If no position found that can block in two moves, attack player wins
      if (locs.empty()) {
        winner = attackPla;
        gameEndMovenum = board.movenum + 4;
      }
    } else { // stage == 1
      // Check if one position can block all tuples and meets priority requirements
      for (Loc candidateLoc : allEmptyPositions) {
        if (!(board.getLocationPriority(candidateLoc) + Board::PRIOR_EPS >= board.firstLocPriority)) {
          continue;
        }
        
        bool canBlockAll = true;
        for (const auto& emptyLocs : emptyPositionsInTuples) {
          bool foundInThisTuple = false;
          for (Loc loc : emptyLocs) {
            if (loc == candidateLoc) {
              foundInThisTuple = true;
              break;
            }
          }
          if (!foundInThisTuple) {
            canBlockAll = false;
            break;
          }
        }
        if (canBlockAll) {
          locs.push_back(candidateLoc);
        }
      }
      
      // If no position found that can block in one move, attack player wins
      if (locs.empty()) {
        winner = attackPla;
        gameEndMovenum = board.movenum + 3;
      }
    }
  }
  if (winner != C_WALL)
    assert(gameEndMovenum >= board.movenum);
  return locs;
}

vector<Loc> VCFLogic::getAllVCFAttackOrDefenseLocsWithCache(const Board& board, Player attackPla, Color& winner, int& gameEndMovenum, uint8_t (&stoneTupleCache)[4][Board::MAX_ARR_SIZE]) {
    if(attackPla!=C_BLACK)
    {
        //this never considered white VCF. if white vcf is considered, attackCount and defendCount in checkTuple should swapped

        ASSERT_UNREACHABLE;
    }
    winner = C_WALL;
    gameEndMovenum = 0;
    Player defendPla = getOpp(attackPla);
    int adjs[4] = { 1,board.x_size + 1,board.x_size + 1 + 1,board.x_size + 1 - 1 };
    vector<Loc> locs;

    if (board.nextPla == attackPla) {
        // Attack player logic
        int maybeLegalMap[Board::MAX_ARR_SIZE] = { 0 };
        int maxAttackCount = 0;
        int maxDefendCount = 0;
        int fourCount = 0;


        auto checkTuple = [&](Loc loc0, int16_t adjidx) -> void {
            uint8_t c=stoneTupleCache[adjidx][loc0];
            assert(!(c&0x80));

            int attackCount = c&0x07;
            int defendCount = (c>>3)&0x07;

            

            // Update maximum counts
            if (defendCount == 6) {
                maxDefendCount = max(maxDefendCount, defendCount);
            }

            if (board.stage == 0) {
                if (attackCount >= 4 && attackCount <= 6 && defendCount == 0) {
                    maxAttackCount = max(maxAttackCount, attackCount);
                }
                // Record possible legal positions (2-3 attack pieces and no defend pieces)
                if (attackCount >= 2 && attackCount <= 3 && defendCount == 0) {
                    for (int i = 0; i < 6; i++) {
                        Loc loc = loc0 + i * adjs[adjidx];
                        if (board.colors[loc] == C_EMPTY) {
                            maybeLegalMap[loc] += 1;
                        }
                    }
                }
            }
            else { // stage == 1
                if (attackCount >= 5 && attackCount <= 6 && defendCount == 0) {
                    maxAttackCount = max(maxAttackCount, attackCount);
                }
                if (attackCount == 4 && defendCount == 0) {
                    fourCount++;
                }
                // Record possible legal positions (3-4 attack pieces and no defend pieces)
                if (attackCount >= 3 && attackCount <= 4 && defendCount == 0) {
                    for (int i = 0; i < 6; i++) {
                        Loc loc = loc0 + i * adjs[adjidx];
                        if (board.colors[loc] == C_EMPTY && board.firstLoc != loc) {
                            maybeLegalMap[loc] += 1;
                        }
                    }
                }
            }
            };

        // Traverse all tuples in all directions
        // +x direction (horizontal)
        for (int y = 0; y < board.y_size; y++) {
            for (int x = 0; x < board.x_size - 5; x++) {
                Loc loc0 = Location::getLoc(x, y, board.x_size);
                checkTuple(loc0, 0);
            }
        }

        // +y direction (vertical)
        for (int y = 0; y < board.y_size - 5; y++) {
            for (int x = 0; x < board.x_size; x++) {
                Loc loc0 = Location::getLoc(x, y, board.x_size);
                checkTuple(loc0, 1);
            }
        }

        // +x+y direction (positive diagonal)
        for (int y = 0; y < board.y_size - 5; y++) {
            for (int x = 0; x < board.x_size - 5; x++) {
                Loc loc0 = Location::getLoc(x, y, board.x_size);
                checkTuple(loc0, 2);
            }
        }

        // -x+y direction (negative diagonal)
        for (int y = 0; y < board.y_size - 5; y++) {
            for (int x = 5; x < board.x_size; x++) {
                Loc loc0 = Location::getLoc(x, y, board.x_size);
                checkTuple(loc0, 3);
            }
        }

        // Check win/loss conditions
        if (maxDefendCount == 6) {
            winner = defendPla;
            gameEndMovenum = board.movenum;
            return locs;
        }

        if (board.stage == 0) {
            if (maxAttackCount >= 4) {
                winner = attackPla;
                gameEndMovenum = board.movenum + 2;
                return locs;
            }
        }
        else { // stage == 1
            if (maxAttackCount >= 5) {
                winner = attackPla;
                gameEndMovenum = board.movenum + 1;
                return locs;
            }
        }

        int maybeLegalThreshold = 1;
        if (board.stage == 0) {
            maybeLegalThreshold = 0;//maybe there are enough fours
        }

        if (board.stage == 1 && fourCount == 0) {
            maybeLegalThreshold = 2;//must create two fours at once
        }

        if (board.stage == 1 && fourCount >= 2) {
            maybeLegalThreshold = 0;//maybe there are enough fours
        }

        // Collect possible legal positions
        for (int y = 0; y < board.y_size; y++) {
            for (int x = 0; x < board.x_size; x++) {
                Loc loc = Location::getLoc(x, y, board.x_size);
                if (maybeLegalMap[loc] >= maybeLegalThreshold && board.colors[loc] == C_EMPTY && loc != board.firstLoc) {
                    if (board.stage == 0 || (board.getLocationPriority(loc) + Board::PRIOR_EPS >= board.firstLocPriority)) {
                        locs.push_back(loc);
                    }
                }
            }
        }

        // If no feasible positions, defend player wins
        if (locs.empty()) {
            winner = defendPla;
            gameEndMovenum = board.movenum;
        }

    }
    else {
        // Defend player logic
        vector<vector<Loc>> emptyPositionsInTuples; // Record empty positions in tuples with 4-5 attackPla pieces and no defendPla pieces
        int maxDefendCount = 0;
        int maxAttackCount = 0;


        auto checkTuple = [&](Loc loc0, int16_t adjidx) -> void {
            uint8_t c=stoneTupleCache[adjidx][loc0];
            assert(!(c&0x80));

            int attackCount = c&0x07;
            int defendCount = (c>>3)&0x07;

            if (board.stage == 0) {
                // Update maximum counts
                if (attackCount == 6) {
                    maxAttackCount = max(maxAttackCount, attackCount);
                }

                // Update defend player maximum counts
                if (defendCount >= 4 && defendCount <= 6 && attackCount == 0) {
                    maxDefendCount = max(maxDefendCount, defendCount);
                }

                // Record empty positions in tuples with 4-5 attackPla pieces and no defendPla pieces
                if (attackCount >= 4 && attackCount <= 5 && defendCount == 0) {
                    vector<Loc> emptyLocs;
                    for (int i = 0; i < 6; i++) {
                        Loc loc = loc0 + i * adjs[adjidx];
                        if (board.colors[loc] == C_EMPTY) {
                            emptyLocs.push_back(loc);
                        }
                    }
                    emptyPositionsInTuples.push_back(emptyLocs);
                }
            }
            else { // stage == 1
                // Update maximum counts
                if (attackCount == 6) {
                    maxAttackCount = max(maxAttackCount, attackCount);
                }

                // Update defend player maximum counts
                if (defendCount >= 5 && defendCount <= 6 && attackCount == 0) {
                    maxDefendCount = max(maxDefendCount, defendCount);
                }

                // Record empty positions in tuples with 4-5 attackPla pieces and no defendPla pieces
                if (attackCount >= 4 && attackCount <= 5 && defendCount == 0) {
                    vector<Loc> emptyLocs;
                    for (int i = 0; i < 6; i++) {
                        Loc loc = loc0 + i * adjs[adjidx];
                        if (board.colors[loc] == C_EMPTY && board.firstLoc != loc) {
                            emptyLocs.push_back(loc);
                        }
                    }
                    emptyPositionsInTuples.push_back(emptyLocs);
                }
            }
            };

        // 遍历所有方向的六元组
        // +x direction (横向)
        for (int y = 0; y < board.y_size; y++) {
            for (int x = 0; x < board.x_size - 5; x++) {
                Loc loc0 = Location::getLoc(x, y, board.x_size);
                checkTuple(loc0, 0);
            }
        }

        // +y direction (竖向)
        for (int y = 0; y < board.y_size - 5; y++) {
            for (int x = 0; x < board.x_size; x++) {
                Loc loc0 = Location::getLoc(x, y, board.x_size);
                checkTuple(loc0, 1);
            }
        }

        // +x+y direction (正斜向)
        for (int y = 0; y < board.y_size - 5; y++) {
            for (int x = 0; x < board.x_size - 5; x++) {
                Loc loc0 = Location::getLoc(x, y, board.x_size);
                checkTuple(loc0, 2);
            }
        }

        // -x+y direction (反斜向)
        for (int y = 0; y < board.y_size - 5; y++) {
            for (int x = 5; x < board.x_size; x++) {
                Loc loc0 = Location::getLoc(x, y, board.x_size);
                checkTuple(loc0, 3);
            }
        }

        // Check win/loss conditions
        if (board.stage == 0) {
            if (maxAttackCount == 6) {
                winner = attackPla;
                gameEndMovenum = board.movenum;
                return locs;
            }
            if (maxDefendCount >= 4) {
                winner = defendPla;
                gameEndMovenum = board.movenum + 2;
                return locs;
            }
        }
        else { // stage == 1
            if (maxAttackCount == 6) {
                assert(false); // Should not have 6 attack pieces in stage==1
            }
            if (maxDefendCount >= 5) {
                winner = defendPla;
                gameEndMovenum = board.movenum + 1;
                return locs;
            }
        }

        // If no threatening tuples, return empty
        if (emptyPositionsInTuples.empty()) {
            winner = defendPla;
            gameEndMovenum = board.movenum;

            return locs;
        }

        if (emptyPositionsInTuples.size() == 1 && board.stage == 0) {
            winner = defendPla;
            gameEndMovenum = board.movenum + 1;

            return locs;
        }

        // Calculate positions that can block all fours
        set<Loc> uniqueEmptyPositions;
        for (const auto& emptyLocs : emptyPositionsInTuples) {
            for (Loc loc : emptyLocs) {
                uniqueEmptyPositions.insert(loc);
            }
        }
        vector<Loc> allEmptyPositions(uniqueEmptyPositions.begin(), uniqueEmptyPositions.end());

        if (board.stage == 0) {
            // Check if 2 pieces can block all tuples
            for (size_t i = 0; i < allEmptyPositions.size(); i++) {
                Loc loc1 = allEmptyPositions[i];
                bool canBlock = false;
                for (size_t j = 0; j < allEmptyPositions.size(); j++) {
                    Loc loc2 = allEmptyPositions[j];
                    if (loc2 == loc1)
                        continue;

                    // Check if second move meets priority requirements
                    if (!(board.getLocationPriority(loc2) + Board::PRIOR_EPS >= board.getLocationPriority(loc1))) {
                        continue;
                    }

                    bool canBlockAll = true;
                    for (const auto& emptyLocs : emptyPositionsInTuples) {
                        bool foundInThisTuple = false;
                        for (Loc loc : emptyLocs) {
                            if (loc == loc1 || loc == loc2) {
                                foundInThisTuple = true;
                                break;
                            }
                        }
                        if (!foundInThisTuple) {
                            canBlockAll = false;
                            break;
                        }
                    }
                    if (canBlockAll) {
                        canBlock = true;
                        break;
                    }
                }
                if (canBlock) {
                    locs.push_back(loc1);
                }
            }

            // If no position found that can block in two moves, attack player wins
            if (locs.empty()) {
                winner = attackPla;
                gameEndMovenum = board.movenum + 4;
            }
        }
        else { // stage == 1
            // Check if one position can block all tuples and meets priority requirements
            for (Loc candidateLoc : allEmptyPositions) {
                if (!(board.getLocationPriority(candidateLoc) + Board::PRIOR_EPS >= board.firstLocPriority)) {
                    continue;
                }

                bool canBlockAll = true;
                for (const auto& emptyLocs : emptyPositionsInTuples) {
                    bool foundInThisTuple = false;
                    for (Loc loc : emptyLocs) {
                        if (loc == candidateLoc) {
                            foundInThisTuple = true;
                            break;
                        }
                    }
                    if (!foundInThisTuple) {
                        canBlockAll = false;
                        break;
                    }
                }
                if (canBlockAll) {
                    locs.push_back(candidateLoc);
                }
            }

            // If no position found that can block in one move, attack player wins
            if (locs.empty()) {
                winner = attackPla;
                gameEndMovenum = board.movenum + 3;
            }
        }
    }
    if (winner != C_WALL)
        assert(gameEndMovenum >= board.movenum);
    return locs;
}

vector<Loc> VCFLogic::getAllDefenseFourLocs(const Board& board, Player attackPla) {
    Player defendPla = getOpp(attackPla);
    assert(board.stage==0);
    assert(board.nextPla==attackPla);
    vector<Loc> locs;
  
    
    auto checkTuple = [&](Loc loc0, int16_t adj) -> void {
        int attackCount = 0;
        int defendCount = 0;
        
        for (int i = 0; i < 6; i++) {
            Loc loc = loc0 + i * adj;
            //if (!board.isOnBoard(loc)) {
            //  ASSERT_UNREACHABLE;
            //  return; // 无效六元组，跳过
            //}
            
            Color c = board.colors[loc];
            if (board.stage == 1 && loc == board.firstLoc)
                c = board.nextPla;
            if (c == attackPla) {
                attackCount++;
            } else if (c == defendPla) {
                defendCount++;
            }
        }

        assert(!(defendCount==6));//defender can directly win
        //assert(!(defendCount-board.stage>=4&&attackCount==0));//defender can directly win
        assert(attackCount<=5);

        // Record empty positions in tuples with 4-5 attackPla pieces and no defendPla pieces
        if (attackCount >= 4 && attackCount <= 5 && defendCount == 0) {
            for (int i = 0; i < 6; i++) {
                Loc loc = loc0 + i * adj;
                if (board.colors[loc] == C_EMPTY) {
                    locs.push_back(loc);
                }
            }
        }
        
    };
    
    // 遍历所有方向的六元组
    // +x direction (横向)
    for (int y = 0; y < board.y_size; y++) {
      for (int x = 0; x < board.x_size - 5; x++) {
        Loc loc0 = Location::getLoc(x, y, board.x_size);
        checkTuple(loc0, 1);
      }
    }
    
    // +y direction (竖向)
    for (int y = 0; y < board.y_size - 5; y++) {
      for (int x = 0; x < board.x_size; x++) {
        Loc loc0 = Location::getLoc(x, y, board.x_size);
        checkTuple(loc0, board.x_size + 1);
      }
    }
    
    // +x+y direction (正斜向)
    for (int y = 0; y < board.y_size - 5; y++) {
      for (int x = 0; x < board.x_size - 5; x++) {
        Loc loc0 = Location::getLoc(x, y, board.x_size);
        checkTuple(loc0, board.x_size + 1 + 1);
      }
    }
    
    // -x+y direction (反斜向)
    for (int y = 0; y < board.y_size - 5; y++) {
      for (int x = 5; x < board.x_size; x++) {
        Loc loc0 = Location::getLoc(x, y, board.x_size);
        checkTuple(loc0, board.x_size + 1 - 1);
      }
    }
    
    return locs;
}

void VCFLogic::markAllDefenseDependedLocs(const Board& board, Player attackPla, std::vector<int8_t>& dependMap)
{
    assert(dependMap.size()==Board::MAX_ARR_SIZE);
    Player defendPla = getOpp(attackPla);
    
    //mark all existing stones as 2
    for(int i=0;i<Board::MAX_ARR_SIZE;i++)
    {
      if (board.colors[i] != C_EMPTY)
        dependMap[i] = 2;
    }
    dependMap[board.firstLoc]=2;



    auto checkTuple = [&](Loc loc0, int16_t adj) -> void {
        int attackCount = 0;
        int defendCount = 0;

        for (int i = 0; i < 6; i++) {
            Loc loc = loc0 + i * adj;
            //if (!board.isOnBoard(loc)) {
            //  ASSERT_UNREACHABLE;
            //  return; // 无效六元组，跳过
            //}

            Color c = board.colors[loc];
            if (board.stage == 1 && loc == board.firstLoc)
            c = board.nextPla;
            if (c == attackPla) {
            attackCount++;
            }
            else if (c == defendPla) {
            defendCount++;
            }
        }

        if (attackCount >= 4 && defendCount == 0) {
            if (board.nextPla == attackPla && attackCount - board.stage >= 4)
              assert(false);
            for (int i = 0; i < 6; i++) {
                Loc loc = loc0 + i * adj;
                if (board.colors[loc] == C_EMPTY) {
                    dependMap[loc] = 2;
                }
            }
        }

        if (attackCount == 0 && defendCount >= 3) {
            for (int i = 0; i < 6; i++) {
                Loc loc = loc0 + i * adj;
                if (board.colors[loc] == C_EMPTY) {
                    dependMap[loc] = 2;
                }
            }
        }
        else if (attackCount == 0 && defendCount >= 2) {
            for (int i = 0; i < 6; i++) {
                Loc loc = loc0 + i * adj;
                if (board.colors[loc] == C_EMPTY) {
                    dependMap[loc] = max(dependMap[loc], int8_t(1));
                }
            }
        }
        
    };

    // Traverse all tuples in all directions
    // +x direction (horizontal)
    for (int y = 0; y < board.y_size; y++) {
        for (int x = 0; x < board.x_size - 5; x++) {
        Loc loc0 = Location::getLoc(x, y, board.x_size);
        checkTuple(loc0, 1);
        }
    }

    // +y direction (vertical)
    for (int y = 0; y < board.y_size - 5; y++) {
        for (int x = 0; x < board.x_size; x++) {
        Loc loc0 = Location::getLoc(x, y, board.x_size);
        checkTuple(loc0, board.x_size + 1);
        }
    }

    // +x+y direction (positive diagonal)
    for (int y = 0; y < board.y_size - 5; y++) {
        for (int x = 0; x < board.x_size - 5; x++) {
        Loc loc0 = Location::getLoc(x, y, board.x_size);
        checkTuple(loc0, board.x_size + 1 + 1);
        }
    }

    // -x+y direction (negative diagonal)
    for (int y = 0; y < board.y_size - 5; y++) {
        for (int x = 5; x < board.x_size; x++) {
        Loc loc0 = Location::getLoc(x, y, board.x_size);
        checkTuple(loc0, board.x_size + 1 - 1);
        }
    }

}

Loc VCFLogic::findImmediateWinInVCFAttack(const Board& board, Player attackPla) {
  // Only search for stage==1 and nextPla==attackPla cases
  if (board.stage != 1 || board.nextPla != attackPla) {
    ASSERT_UNREACHABLE;
  }
  
  Color winner = C_WALL;
  int gameEndMovenum = 0;
  
  // Get all VCF attack positions
  vector<Loc> attackLocs = getAllVCFAttackOrDefenseLocs(board, attackPla, winner, gameEndMovenum);
  
  // If there's already a determined win/loss result, return directly
  if (winner != C_WALL) {

    if(winner==attackPla)
      ASSERT_UNREACHABLE;
    else
      return Board::NULL_LOC;
  }
  
  // Search one layer for each attack position
  for (Loc attackLoc : attackLocs) {
    // Create temporary board for testing
    Board tempBoard = board;
    tempBoard.playMoveAssumeLegal(attackLoc, attackPla);
    
    // Check the position after this move
    Color tempWinner;
    int tempGameEndMovenum;
    vector<Loc> defenseLocs = getAllVCFAttackOrDefenseLocs(tempBoard, attackPla, tempWinner, tempGameEndMovenum);
    
    // If attack player wins directly, return this position
    if (tempWinner == attackPla) {
      assert(tempGameEndMovenum == board.movenum + 5);
      return attackLoc;
    }
  }
  
  // No immediate winning position found
  return Board::NULL_LOC;
}

Loc VCFLogic::findImmediateWinInVCFAttackLayer2(const Board& board, Player attackPla) {
  // Only search for stage==0 and nextPla==attackPla cases
  if (board.stage != 0 || board.nextPla != attackPla) {
    ASSERT_UNREACHABLE;
  }

  Color winner = C_WALL;
  int gameEndMovenum = 0;

  // Get all VCF attack positions
  vector<Loc> attackLocs = getAllVCFAttackOrDefenseLocs(board, attackPla, winner, gameEndMovenum);

  // If there's already a determined win/loss result, return directly
  if (winner != C_WALL) {
    ASSERT_UNREACHABLE;
  }

  // Search one layer for each attack position
  for (Loc attackLoc : attackLocs) {
    // Create temporary board for testing
    Board tempBoard = board;
    tempBoard.playMoveAssumeLegal(attackLoc, attackPla);

    // Check the position after this move
    Loc winLoc = VCFLogic::findImmediateWinInVCFAttack(tempBoard, attackPla);

    // If attack player wins directly, return this position
    if (winLoc != Board::NULL_LOC) {
      return attackLoc;
    }
  }

  // No immediate winning position found
  return Board::NULL_LOC;
}


void VCFLogic::printBoardWithDependencyMap(const Board& board, const std::vector<int8_t>& dependMap) {
    if (dependMap.empty())
    {
        cout << "Board has no VCF:" << endl;
        Board::printBoard(cout, board, Board::NULL_LOC, nullptr);
        return;
    }
    else
        assert(dependMap.size() == Board::MAX_ARR_SIZE);
    cout << "Defense dependency map (board format):" << endl;

    // Print board with dependency map values (similar to Board::printBoard)
    bool showCoords = board.x_size <= 50 && board.y_size <= 50;
    if (showCoords) {
        const char* xChar = "ABCDEFGHJKLMNOPQRSTUVWXYZ";
        cout << "  ";
        for (int x = 0; x < board.x_size; x++) {
            if (x <= 24) {
                cout << " ";
                cout << xChar[x];
            }
            else {
                cout << "A" << xChar[x - 25];
            }
        }
        cout << endl;
    }

    // Count dependency values in empty positions
    int count0 = 0, count1 = 0, count2 = 0, countOther = 0;
  
    for (int y = 0; y < board.y_size; y++) {
        if (showCoords) {
            char buf[16];
            sprintf(buf, "%2d", board.y_size - y);
            cout << buf << ' ';
        }
        for (int x = 0; x < board.x_size; x++) {
            Loc loc = Location::getLoc(x, y, board.x_size);
            if (board.colors[loc] != C_EMPTY) {
                // Show existing pieces
                char s = PlayerIO::colorToChar(board.colors[loc]);
                cout << s;
            }
            else {
                // Count all dependency values in empty positions
                if (dependMap[loc] == 0) {
                    count0++;
                    cout << '.';
                }
                else if (dependMap[loc] == 1) {
                    count1++;
                    cout << '1';
                }
                else if (dependMap[loc] == 2) {
                    count2++;
                    cout << '2';
                }
                else {
                    countOther++;
                    cout << (int)dependMap[loc];
                }
            }

            if (x < board.x_size - 1)
                cout << ' ';
        }
        cout << endl;
    }
  
    // Display statistics
    cout << "Dependency statistics (empty positions only):" << endl;
    cout << "  Positions with value 0: " << count0 << endl;
    cout << "  Positions with value 1: " << count1 << endl;
    cout << "  Positions with value 2: " << count2 << endl;
    if (countOther > 0) {
        assert(false);
    }
    cout << "  Total empty positions: " << (count0 + count1 + count2 + countOther) << endl;
    cout << "  Total dependency positions: " << (count1 + count2 + countOther) << endl;
  cout << endl;
}

void VCFLogic::printBoardWithPruneInfo(const Board& board, const std::map<Loc,int16_t>& pruneInfo) {
    cout << "VCF Prune Information (board format):" << endl;

    // pruneInfo is already a map, no need to create another one

    // Print board with prune info (4 characters per cell)
    bool showCoords = board.x_size <= 50 && board.y_size <= 50;
    if (showCoords) {
        const char* xChar = "ABCDEFGHJKLMNOPQRSTUVWXYZ";
        cout << "    "; // 4 spaces for row numbers
        for (int x = 0; x < board.x_size; x++) {
            if (x <= 24) {
                cout << " " << xChar[x] << "  ";
            }
            else {
                cout << "A" << xChar[x - 25] << " ";
            }
        }
        cout << endl;
    }

    for (int y = 0; y < board.y_size; y++) {
        if (showCoords) {
            char buf[16];
            sprintf(buf, "%2d ", board.y_size - y);
            cout << buf << " ";
        }

        for (int x = 0; x < board.x_size; x++) {
            Loc loc = Location::getLoc(x, y, board.x_size);

            if (board.colors[loc] != C_EMPTY) {
                // Show existing pieces
                char s = PlayerIO::colorToChar(board.colors[loc]);
                cout << " " << s << "  ";
            }
            else {
                // Check if this location has prune info
                auto it = pruneInfo.find(loc);
                if (it != pruneInfo.end()) {
                    int16_t moveNum = it->second;
                    if (moveNum < 10)
                        cout << " L" << moveNum << " ";
                    else if (moveNum < 100)
                        cout << "L" << moveNum << " ";
                    else
                        cout << "L" << moveNum;
                }
                else {
                    // Check if position is illegal due to move priority
                    if (board.isLegal(loc, board.nextPla)) {
                        if (board.getLocationPriority(x, y) + Board::PRIOR_EPS < board.firstLocPriority) {
                            cout << " L0 ";
                        }
                        else {
                            cout << " .  ";
                        }
                    }
                    else {
                        if (board.firstLoc == loc)
                            cout << " @  ";
                        else
                            cout << " .  ";
                    }
                }
            }
        }
        cout << endl;
    }
    cout << endl;
}

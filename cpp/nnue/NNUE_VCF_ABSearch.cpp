#include "NNUE_VCF_ABSearch.h"
#include "../nnue_search/VCFLogic.h"
#include "../game/gamelogic.h"
#include "Eva_nnuev2.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <cassert>

using namespace std;
using namespace NNUE;

ABSearch_CacheTable::Entry::Entry()
  :hash(0,0),value(0),depth(-1000),bestloc0(Board::NULL_LOC),bestLoc1(Board::NULL_LOC)
{
}
ABSearch_CacheTable::Entry::~Entry()
{
}

ABSearch_CacheTable::ABSearch_CacheTable(int sizePowerOfTwo, int mutexPoolSizePowerOfTwo) {
  if (sizePowerOfTwo < 0 || sizePowerOfTwo > 63)
    throw StringError("ABSearch_CacheTable: Invalid sizePowerOfTwo: " + Global::intToString(sizePowerOfTwo));
  if (mutexPoolSizePowerOfTwo < 0 || mutexPoolSizePowerOfTwo > 31)
    throw StringError("ABSearch_CacheTable: Invalid mutexPoolSizePowerOfTwo: " + Global::intToString(mutexPoolSizePowerOfTwo));
#if defined(SIMULATE_TRUE_HASH_COLLISIONS)
  sizePowerOfTwo = sizePowerOfTwo > 12 ? 12 : sizePowerOfTwo;
#endif
  if (mutexPoolSizePowerOfTwo > sizePowerOfTwo)
    mutexPoolSizePowerOfTwo = sizePowerOfTwo;

  tableSize = ((uint64_t)1) << sizePowerOfTwo;
  tableMask = tableSize - 1;
  entries = new Entry[tableSize];
  uint32_t mutexPoolSize = ((uint32_t)1) << mutexPoolSizePowerOfTwo;
  mutexPoolMask = mutexPoolSize - 1;
  mutexPool = new MutexPool(mutexPoolSize);
}
ABSearch_CacheTable::~ABSearch_CacheTable() {
  delete[] entries;
  delete mutexPool;
}
bool ABSearch_CacheTable::get(Hash128 nnHash, Entry& ret) {
  //Free ret BEFORE locking, to avoid any expensive operations while locked.
  ret = Entry();

  uint64_t idx = nnHash.hash0 & tableMask;
  uint32_t mutexIdx = (uint32_t)idx & mutexPoolMask;
  Entry& entry = entries[idx];
  std::mutex& mutex = mutexPool->getMutex(mutexIdx);

  std::lock_guard<std::mutex> lock(mutex);
  bool found = false;
#if defined(SIMULATE_TRUE_HASH_COLLISIONS)
  if (entry.hash.hash0 ^ nnHash.hash0) & 0xFFF) == 0) {
    ret = entry;
    found = true;
  }
#else
  if (entry.hash == nnHash) {
    ret = entry;
    found = true;
  }
#endif
  return found;
}

void ABSearch_CacheTable::set(const Entry& ent) {
  //Immediately copy p right now, before locking, to avoid any expensive operations while locked.

  uint64_t idx = ent.hash.hash0 & tableMask;
  uint32_t mutexIdx = (uint32_t)idx & mutexPoolMask;
  Entry& entry = entries[idx];
  std::mutex& mutex = mutexPool->getMutex(mutexIdx);

  {
    std::lock_guard<std::mutex> lock(mutex);
    //Perform a swap, to avoid any expensive free under the mutex.
    entry = ent;
  }

  //No longer locked, allow buf to fall out of scope now, will free whatever used to be present in the table.
}

void ABSearch_CacheTable::clear() {
  for (size_t idx = 0; idx < tableSize; idx++) {
    Entry& entry = entries[idx];
    uint32_t mutexIdx = (uint32_t)idx & mutexPoolMask;
    std::mutex& mutex = mutexPool->getMutex(mutexIdx);
    {
      std::lock_guard<std::mutex> lock(mutex);
      entry = Entry();
    }
  }
}

// VCF_ABSearch class implementation
VCF_ABSearch::VCF_ABSearch(NNUEBoardHistory* hist, ABSearch_CacheTable* cache, Player pla) 
  : boardHistory(hist), attackPlayer(pla), defendPlayer(getOpp(pla)), nodeCount(0), nnevalCount(0), cacheTable(cache) {
  // Check initial state
  assert(boardHistory->getBoard().stage == 0);
  assert(boardHistory->getBoard().nextPla == attackPlayer);
}

VCF_ABSearch::~VCF_ABSearch() {
  // Cache table is managed externally, don't delete it here
}

double VCF_ABSearch::search(double maxDepth) {
  remainingDepth = maxDepth;
  nodeCount = 0;
  nnevalCount = 0;
  return alphaBeta(-1.2, 
                   std::numeric_limits<double>::infinity(), 
                   maxDepth);
}

double VCF_ABSearch::alphaBeta(
    double alpha, double beta, double depth) {

  nodeCount++;
  
  // Check cache first
  const Board& board = boardHistory->getBoard();
  ABSearch_CacheTable::Entry cacheEntry;
  Loc cachedBestMove = Board::NULL_LOC;
  if (cacheTable && cacheTable->get(board.pos_hash, cacheEntry)) {
    // Get cached best move based on current player
    cachedBestMove = cacheEntry.bestloc0; // Attacker's best move
    
    // If cached depth is sufficient (difference < 0.5), use cached result
    if (cacheEntry.depth >= depth-0.01 && (cacheEntry.value>1.1|| cacheEntry.value < -1.1)) {
      //cout << cacheEntry.value << endl;
      return cacheEntry.value;
    }
  }
  
  // Directly determine search direction based on current player (attacker's perspective)
  
  // Check game state
  Color maybeWinner=C_WALL;
  int gameEndMovenum=0;
  std::vector<Loc> allLegalLocs = VCFLogic::getAllVCFAttackOrDefenseLocs(boardHistory->getBoard(), attackPlayer, maybeWinner, gameEndMovenum);


  //int gameState = checkGameState();
  //cout << gameState;
  if (maybeWinner != C_WALL) {

    if (maybeWinner == attackPlayer) { // Attacker wins
      // Attacker wins: board.x_size*board.y_size+100-board.numStones (ensure > 1)
      double winValue = boardHistory->getBoard().x_size * boardHistory->getBoard().y_size + 100 - gameEndMovenum;

      // Store result in cache
      if (cacheTable) {
        ABSearch_CacheTable::Entry newEntry;
        newEntry.hash = board.pos_hash;
        newEntry.value = winValue;
        newEntry.depth = depth;
        newEntry.bestloc0 = Board::NULL_LOC;
        newEntry.bestLoc1 = Board::NULL_LOC;
        cacheTable->set(newEntry);
      }
      
      return winValue;
    } else { // Defender wins
      // Defender wins: -2.0
      
      // Store result in cache
      if (cacheTable) {
        ABSearch_CacheTable::Entry newEntry;
        newEntry.hash = board.pos_hash;
        newEntry.value = -2.0;
        newEntry.depth = depth;
        newEntry.bestloc0 = Board::NULL_LOC;
        newEntry.bestLoc1 = Board::NULL_LOC;
        cacheTable->set(newEntry);
      }
      
      return -2.0;
    }
  }
  
  // Check if reached leaf node (when defender moves with depth<0 and no clear winner)
  if (depth < 0 && boardHistory->getBoard().nextPla == defendPlayer) {
    double leafValue = evaluateLeafAssumeNotEnd();
    
    // Store result in cache
    if (cacheTable) {
      ABSearch_CacheTable::Entry newEntry;
      newEntry.hash = board.pos_hash;
      newEntry.value = leafValue;
      newEntry.depth = depth;
      newEntry.bestloc0 = Board::NULL_LOC;
      newEntry.bestLoc1 = Board::NULL_LOC;
      cacheTable->set(newEntry);
    }
    
    return leafValue;
  }
  
  // Get legal moves
  vector<pair<Loc, double>> moves = getLegalMovesWithPolicy(allLegalLocs, true);
  if (moves.empty()) {
    double emptyValue;
    if(boardHistory->getBoard().numStones == boardHistory->getBoard().x_size * boardHistory->getBoard().y_size)
      emptyValue = -2.0; // Draw,failed to attack
    else
    // play a corner move which cause the second move can only be pass
      if(boardHistory->getBoard().nextPla == attackPlayer)
        emptyValue = -10000; 
      else
        emptyValue = 10000;
    
    // Store result in cache
    if (cacheTable) {
      ABSearch_CacheTable::Entry newEntry;
      newEntry.hash = board.pos_hash;
      newEntry.value = emptyValue;
      newEntry.depth = depth;
      newEntry.bestloc0 = Board::NULL_LOC;
      newEntry.bestLoc1 = Board::NULL_LOC;
      cacheTable->set(newEntry);
    }
    
    return emptyValue;
  }
  
  // Find the move with maximum policy
  double maxPolicy = -1e100;
  for (const auto& move : moves) {
    maxPolicy = max(maxPolicy, move.second);
  }
  
  // Reorder moves to prioritize cached best move
  if (cachedBestMove != Board::NULL_LOC) {
    // Find and modify cached move policy to max+0.01, then sort
    for (auto& move : moves) {
      if (move.first == cachedBestMove) {
        move.second = maxPolicy + 0.01; // Set to max policy + 0.01 for priority
        if(move.second>0)move.second=0;
        maxPolicy = move.second;
        break;
      }
    }
    // Sort moves by policy descending (cached move will be first)
    sort(moves.begin(), moves.end(), [](const pair<Loc, double>& a, const pair<Loc, double>& b) {
      return a.second > b.second;
    });
  }
  
  if (boardHistory->getBoard().nextPla == attackPlayer) { // Attacker maximizes
    double bestValue = -std::numeric_limits<double>::infinity();
    Loc bestMove = Board::NULL_LOC;
    
    int t = 0;
    for (const auto& move : moves) {
      Loc loc = move.first;
      double policy = move.second;
      
      // Check if this move should be excluded
      double policyCost = -policy;
      //if (policy == maxPolicy)policyCost = 0;
      if (policyCost > depth && policy < maxPolicy - 0.01) {
        // Excluded moves are assumed to have value -1.5
        double childValue = -1.5;
        if (childValue > bestValue) {
          bestValue = childValue;
          bestMove = loc;
        }
        continue;
      }
      t += 1;
      // Make move
      boardHistory->play(boardHistory->getBoard().nextPla, loc);
      
      // Calculate new depth
      double newDepth = depth - policyCost;
      
      // Recursive search
      double childValue = alphaBeta(alpha, beta, newDepth);
      
      // Undo move
      boardHistory->undo();
      
      if (childValue > bestValue) {
        bestValue = childValue;
        bestMove = loc;
      }
      
      // Attacker returns immediately once a winning move is found (value > 1)
      //if (bestValue > 1.0) {
      //  return bestValue;
      //}
      
      alpha = max(alpha, bestValue);
      if (beta <= alpha) {
        break; // Beta pruning
      }
    }
    //cout << depth << " " << t << endl;
    
    // Store result in cache
    if (cacheTable) {
      ABSearch_CacheTable::Entry newEntry;
      newEntry.hash = board.pos_hash;
      newEntry.value = bestValue;
      newEntry.depth = depth;
      newEntry.bestloc0 = bestMove; // Store best move for attacker
      newEntry.bestLoc1 = Board::NULL_LOC;
      cacheTable->set(newEntry);
    }
    
     return bestValue;
  } else { // Defender minimizes
    double bestValue = std::numeric_limits<double>::infinity();
    Loc bestMove = Board::NULL_LOC;
    
    for (const auto& move : moves) {
      Loc loc = move.first;
      double policy = move.second;
      
      // Record current player
      Player currentPlayer = boardHistory->getBoard().nextPla;
      
      // Make move
      boardHistory->play(currentPlayer, loc);
      
      // Defender does not deduct depth
      double childValue = alphaBeta(alpha, beta, depth);
      
      // Undo move
      boardHistory->undo();
      
      if (childValue < bestValue) {
        bestValue = childValue;
        bestMove = loc;
      }
      
      beta = min(beta, bestValue);
      if (beta <= alpha) {
        break; // Alpha pruning
      }
    }
    
    // Store result in cache
    if (cacheTable) {
      ABSearch_CacheTable::Entry newEntry;
      newEntry.hash = board.pos_hash;
      newEntry.value = bestValue;
      newEntry.depth = depth;
      newEntry.bestloc0 = bestMove;
      newEntry.bestLoc1 = Board::NULL_LOC;
      cacheTable->set(newEntry);
    }
    
    return bestValue;
  }
}

int VCF_ABSearch::checkGameState() {
  const Board& board = boardHistory->getBoard();
  //cout << (board.nextPla == defendPlayer ? "w" : "b") << board.stage;
  
  if (board.stage == 0 && board.nextPla == defendPlayer) {
    // Attacker just finished placing two pieces
    int twoFourThreats = VCFLogic::checkTwoFourThreats(board, attackPlayer);
    int oppMaxLen = VCFLogic::checkMaxConnectLen(board, defendPlayer);

    //Board::printBoard(cout, board, board.firstLoc, NULL);
    //cout.flush();

    if (twoFourThreats == 4) {
      return 1; // Attacker wins
    }
    if (oppMaxLen >= 4) {
      return -1; // Defender wins
    }
    if (twoFourThreats == 0 || twoFourThreats == 1) {
      return -1; // Defender wins (can block)
    }
    if (twoFourThreats == 3) {
      return 1; // Attacker wins (cannot block)
    }
    // twoFourThreats == 2, continue search
  }
  
  if (board.stage == 1 && board.nextPla == defendPlayer) {
    // Defender placed one piece
    int twoFourThreats = VCFLogic::checkTwoFourThreats(board, attackPlayer);
    if (twoFourThreats == 1) {
      return 0; // Continue
    }
    if (twoFourThreats == 2 || twoFourThreats == 3) {
      //Board::printBoard(cout, board, board.firstLoc, NULL);
      //cout.flush();
      return 1; // Attacker wins
    }
    if (twoFourThreats == 0) {
      return -1; // Defender wins
    }
  }
  
  if (board.stage == 0 && board.nextPla == attackPlayer) {
    // Attacker's turn to place two pieces
    int oppMaxLen = VCFLogic::checkMaxConnectLen(board, defendPlayer);
    if (oppMaxLen >= 6) {
      return -1; // Defender wins
    }
    int plaMaxLen = VCFLogic::checkMaxConnectLen(board, attackPlayer);
    if (plaMaxLen >= 4) {
      return 1; // Attacker wins
    }
  }
  
  return 0; // Continue search
}

vector<pair<Loc, double>> VCF_ABSearch::getLegalMovesWithPolicy(const std::vector<Loc>& legalLocs, bool hasLegalLocs) {
  if(hasLegalLocs)
    assert(legalLocs.size() > 0);


  const Board& board = boardHistory->getBoard();
  // Update input buffer
  //boardHistory->updateInputBuf(board.nextPla);
  
  // Get policy
  NNUE::PolicyType policy[MaxBS * MaxBS + 1];
  boardHistory->evaluateFull(board.nextPla, policy);
  nnevalCount++;
  
  // Collect all legal positions with their policy values
  vector<pair<Loc, double>> locPolicyPairs;
  
  // Find all legal positions and their policy values
  if(hasLegalLocs)
  {
    for (const Loc& loc : legalLocs) {
      assert(board.isLegal(loc, board.nextPla));
      if(board.stage == 1)
        assert(board.getLocationPriority(loc) + Board::PRIOR_EPS >= board.firstLocPriority);
      int nu_loc = Location::getX(loc, board.x_size) + Location::getY(loc, board.x_size) * MaxBS;

      locPolicyPairs.push_back(make_pair(loc, (double)policy[nu_loc]));
    }

  }
  else
  {
    for (int y = 0; y < board.y_size; y++) {
      for (int x = 0; x < board.x_size; x++) {
        Loc loc = Location::getLoc(x, y, board.x_size);
        if (board.isLegal(loc, board.nextPla) && (board.stage==0 || (board.getLocationPriority(loc) + Board::PRIOR_EPS >= board.firstLocPriority)))
        {
          int nu_loc = x + y * MaxBS;
          locPolicyPairs.push_back(make_pair(loc, (double)policy[nu_loc]));
        }
      }
    }
  }
  
  if (locPolicyPairs.empty()) {
    assert(false);
    return locPolicyPairs;
  }
  
  // Sort by policy value from high to low
  sort(locPolicyPairs.begin(), locPolicyPairs.end(), 
       [](const pair<Loc, double>& a, const pair<Loc, double>& b) {
         return a.second > b.second;
       });
  
  // Apply penalty to sorted positions
  double penalty_const1 = 0.0;
  double penalty_const2 = 5.0;
  double penalty_const3 = 0.3;
  
  for (int i = 0; i < locPolicyPairs.size(); i++) {
    double penalty = penalty_const1 * log(i + penalty_const2);
    locPolicyPairs[i].second -= penalty;
  }
  
  // Find maximum policy value after penalty
  double maxPolicy = -std::numeric_limits<double>::infinity();
  for (const auto& pair : locPolicyPairs) {
    maxPolicy = max(maxPolicy, pair.second);
  }
  
  // Calculate softmax denominator
  double ptemp=1.0;
  double sumExp = 0.0;
  for (const auto& pair : locPolicyPairs) {
    sumExp += exp(ptemp * (pair.second - maxPolicy) / NNUE::policyQuantFactor);
  }
  double logSumExp = log(sumExp) / ptemp;
  
  // Collect legal moves and calculate log_softmax
  for (auto& pair : locPolicyPairs) {
    double logSoftmax = (pair.second - maxPolicy) / NNUE::policyQuantFactor - logSumExp;
    pair.second=logSoftmax;
  }
  
  // Sort by policy from high to low
  //sort(moves.begin(), moves.end(), 
  //     [](const pair<Loc, double>& a, const pair<Loc, double>& b) {
  //       return a.second > b.second;
  //     });
  
  return locPolicyPairs;
}

double VCF_ABSearch::evaluateLeafAssumeNotEnd() {
  // First check if the game outcome is already determined
  //int gameState = checkGameState();
  //cout << gameState;
  //if (gameState == 1) {
    // Attacker wins: board.x_size*board.y_size+100-board.numStones (ensure > 1)
  //  return boardHistory->getBoard().x_size * boardHistory->getBoard().y_size + 100 - boardHistory->getBoard().movenum;
 // }
 // if (gameState == -1) {
    // Defender wins: -2.0
  //  return -2.0;
  //}
  
  // Use NNUE evaluation (neural network returns values in [-1, 1] interval)
  //boardHistory->updateInputBuf(boardHistory->getBoard().nextPla);
  NNUE::ValueType value = boardHistory->evaluateFull(boardHistory->getBoard().nextPla, nullptr);
  nnevalCount++;

  //Board::printBoard(cout, boardHistory->getBoard(), boardHistory->getBoard().firstLoc, NULL);
  //cout.flush();
  
   
  // Convert to attacker's perspective win rate
  if (boardHistory->getBoard().nextPla == attackPlayer) {
    return value.win-value.loss-value.draw;
  } else {
    return value.loss- value.draw -value.win;
  }
}
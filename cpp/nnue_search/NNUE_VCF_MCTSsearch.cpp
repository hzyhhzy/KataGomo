#include "NNUE_VCF_MCTSsearch.h"
#include "VCFLogic.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>
#include <iostream>
#include <climits>

using namespace NNUE_VCF_MCTSsearch;
using namespace NNUE;



// Helper functions
// Return attacker's perspective value: win rate minus loss rate and draw rate
inline double sureResultWR(Color winner, Player attackPlayer, int stepsToWin) {
    if (winner == attackPlayer) {
        return 1.0;  // Attacker wins
    } else if (winner != C_WALL) {
        return -1.0; // Attacker loses
    } else {
        return 0.0;  // Undetermined
    }
}

// Convert NNUE's ValueType to attacker's perspective double value
inline double valueTypeToAttackerPerspective(const NNUE::ValueType& value, Player attackPlayer, Color currentPlayer) {
  assert(value.win + value.loss + value.draw < 1.01 && value.win + value.loss + value.draw >0.99);
    
    
  // If current player is not the attacker, need to negate
  if (currentPlayer == attackPlayer) {
    return value.win - value.loss - value.draw;
  }
  else
  {
    return value.loss - value.win - value.draw;
  }
}

inline double MCTSpuctFactor(double totalVisit, double puct, double puctPow, double puctBase) {
    return puct * pow((totalVisit + puctBase) / puctBase, puctPow);
}

inline double MCTSselectionValue(double puctFactor, double winrate, double childVisit, double childPolicy) {
  return winrate + pow(winrate,3) + pow(winrate,7) + puctFactor * childPolicy / (childVisit + 1);
}

// MCTS_CacheTable implementation
MCTS_CacheTable::Entry::Entry()
  :hash(0,0), maybeWinner(C_WALL), gameEndMovenum(0), nnueValue(0,0,0), bestMove(Board::NULL_LOC)
{
}

MCTS_CacheTable::Entry::~Entry()
{
}

MCTS_CacheTable::MCTS_CacheTable(int sizePowerOfTwo, int mutexPoolSizePowerOfTwo) {
  if (sizePowerOfTwo < 0 || sizePowerOfTwo > 63)
    throw StringError("MCTS_CacheTable: Invalid sizePowerOfTwo: " + Global::intToString(sizePowerOfTwo));
  if (mutexPoolSizePowerOfTwo < 0 || mutexPoolSizePowerOfTwo > 31)
    throw StringError("MCTS_CacheTable: Invalid mutexPoolSizePowerOfTwo: " + Global::intToString(mutexPoolSizePowerOfTwo));
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

MCTS_CacheTable::~MCTS_CacheTable() {
  delete[] entries;
  delete mutexPool;
}

bool MCTS_CacheTable::get(Hash128 nnHash, Entry& ret) {
  //Free ret BEFORE locking, to avoid any expensive operations while locked.
  ret = Entry();

  uint64_t idx = nnHash.hash0 & tableMask;
  uint32_t mutexIdx = (uint32_t)idx & mutexPoolMask;
  Entry& entry = entries[idx];
  std::mutex& mutex = mutexPool->getMutex(mutexIdx);

  std::lock_guard<std::mutex> lock(mutex);
  bool found = false;
#if defined(SIMULATE_TRUE_HASH_COLLISIONS)
  if ((entry.hash.hash0 ^ nnHash.hash0) & 0xFFF) == 0) {
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

void MCTS_CacheTable::set(const Entry& ent) {
  //Immediately copy ent right now, before locking, to avoid any expensive operations while locked.

  uint64_t idx = ent.hash.hash0 & tableMask;
  uint32_t mutexIdx = (uint32_t)idx & mutexPoolMask;
  Entry& entry = entries[idx];
  std::mutex& mutex = mutexPool->getMutex(mutexIdx);

  {
    std::lock_guard<std::mutex> lock(mutex);
    //Perform a swap, to avoid any expensive free under the mutex.
    entry = ent;
  }

  //No longer locked, allow ent to fall out of scope now, will free whatever used to be present in the table.
}

void MCTS_CacheTable::clear() {
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

// MCTSnode implementation
MCTSnode::MCTSnode(MCTSsearch* search, Color nextColor, double policyTemp) : nextColor(nextColor) {
    isWinDetermined = false;
    winner = C_WALL;
    stepsToWin = 0;
    childrennum = 0;
    children = nullptr;
    visits = 1;
    
    const Board& board = search->boardHistory->getBoard();
    
    // Get legal moves with policy from cache or calculate
    Color maybeWinner = C_WALL;
    int gameEndMovenum = 0;
    std::vector<std::pair<Loc, float>> moves = search->getLegalMovesAndVCFResultWithPolicy(nextColor, maybeWinner, gameEndMovenum);

    if (MCTSsearch::IMMEDIATE_WIN_SEARCH_LAYERS >= 2 && maybeWinner == C_WALL && board.nextPla == search->attackPlayer && board.stage == 0)//check one move win from stage 0
    {
      Loc winloc = VCFLogic::findImmediateWinInVCFAttackLayer2(board, search->attackPlayer);
      if (winloc != Board::NULL_LOC)
      {
        maybeWinner = search->attackPlayer;
        gameEndMovenum = board.movenum + 6;
      }
    }

    if (MCTSsearch::IMMEDIATE_WIN_SEARCH_LAYERS >= 1 && maybeWinner == C_WALL && board.nextPla == search->attackPlayer && board.stage == 1)//check one move win from stage 1
    {
      Loc winloc = VCFLogic::findImmediateWinInVCFAttack(board, search->attackPlayer);
      if (winloc != Board::NULL_LOC)
      {
        maybeWinner = search->attackPlayer;
        gameEndMovenum = board.movenum + 5;
      }
    }


    if (search->boardHistory->rules.maxMoves > 0)
    {
      //Can attack player win in rules.maxMoves?
      bool attackerCannotWin = false;
      int mm = search->boardHistory->rules.maxMoves;

      if (maybeWinner != C_WALL)
      {
        if (maybeWinner != search->attackPlayer)
          attackerCannotWin = true;
        else if (gameEndMovenum > mm)
          attackerCannotWin = true;
      }
      else
      {
        bool isAttacker = search->attackPlayer == nextColor;
        int stage = search->boardHistory->getBoard().stage;
        int earliestWinUntilNow =
          (isAttacker && stage == 0) ? (MCTSsearch::IMMEDIATE_WIN_SEARCH_LAYERS >= 2 ? 10 : 6) : //attacker has no four now
          (isAttacker && stage == 1) ? (MCTSsearch::IMMEDIATE_WIN_SEARCH_LAYERS >= 1 ? 9 : 5) :
          (!isAttacker && stage == 0) ? 8 : //the defender can block all fours this turn
          7;

        if (search->boardHistory->getBoard().movenum + earliestWinUntilNow > mm)
          attackerCannotWin = true;
      }

      if (attackerCannotWin)
      {
        maybeWinner = getOpp(search->attackPlayer);
        gameEndMovenum = search->boardHistory->rules.maxMoves;
      }

    }


    if(maybeWinner!=C_WALL)
    {
      isWinDetermined = true;
      stepsToWin = gameEndMovenum;
      assert(stepsToWin > 0);
      winner = maybeWinner;
      WRtotal = sureResultWR(winner, search->attackPlayer, stepsToWin);
      return;
    }
    
    assert(!moves.empty());
    
    // Get NNUE evaluation and convert to attacker perspective
    NNUE::ValueType value = search->evaluatePosition(nextColor);
    WRtotal = valueTypeToAttackerPerspective(value, search->attackPlayer, nextColor);
    
    // Set up children - dynamically allocate based on actual number of legal moves
    legalChildrennum = (int)moves.size();
    children = new MCTSchild[legalChildrennum];
    for (int i = 0; i < legalChildrennum; i++) {
        children[i].loc = moves[i].first;
        children[i].policy = uint16_t(moves[i].second * policyQuant) + 1;
        children[i].ptr = nullptr;
    }
}

MCTSnode::~MCTSnode() {
    if (children != nullptr) {
        for (int i = 0; i < childrennum; i++) {
            if (children[i].ptr != nullptr) delete children[i].ptr;
        }
        delete[] children;
    }
}

// MCTSsearch implementation
MCTSsearch::MCTSsearch(MCTS_CacheTable* cacheTable, NNUEBoardHistory* hist, Player attackPla)
    : rootNode(nullptr), boardHistory(hist), attackPlayer(attackPla), cacheTable(cacheTable) {
    terminate.store(false, std::memory_order_relaxed);
}

MCTSsearch::~MCTSsearch() {
    if (rootNode != nullptr) delete rootNode;
}

float MCTSsearch::fullsearch(Color color, int64_t maxVisits, Loc& bestmove) {
    terminate.store(false, std::memory_order_relaxed);

    if (rootNode == nullptr) 
      rootNode = new MCTSnode(this, color, params.policyTemp);

    // If root is already determined, return immediately
    if (rootNode->isWinDetermined) {
        bestmove = Board::NULL_LOC;
        return (rootNode->winner == attackPlayer) ? 1.0f : -1.0f;
    }

    if (option.maxNodes > 0) {
        maxVisits = std::min(maxVisits, option.maxNodes);
    }
    
    search(rootNode, maxVisits, true);
    
    bestmove = bestRootMove();
    return getRootValue();
}

void MCTSsearch::play(Color color, Loc loc) {
    boardHistory->play(color, loc);
    
    // Try to reuse subtree
    if (rootNode != nullptr) {
        MCTSnode* newRoot = nullptr;
        for (int i = 0; i < rootNode->childrennum; i++) {
            if (rootNode->children[i].loc == loc && rootNode->children[i].ptr != nullptr) {
                newRoot = rootNode->children[i].ptr;
                rootNode->children[i].ptr = nullptr; // Prevent deletion
                break;
            }
        }
        delete rootNode;
        rootNode = newRoot;
    }
}

void MCTSsearch::undo() {
    boardHistory->undo();
    // Clear root node as it's no longer valid
    if (rootNode != nullptr) {
        delete rootNode;
        rootNode = nullptr;
    }
}

void MCTSsearch::clearBoard() {
    boardHistory->clear(boardHistory->getBoard(), boardHistory->getBoard().nextPla, boardHistory->rules);
    if (rootNode != nullptr) {
        delete rootNode;
        rootNode = nullptr;
    }
}

bool MCTSsearch::vcfSearchAutoStop(NNUEBoardHistory* hist, Player attackPla, double searchFactor) {
  // Initialize with new board history and attack player
  boardHistory = hist;
  attackPlayer = attackPla;
  assert(attackPlayer == boardHistory->getBoard().nextPla);
  assert(boardHistory->getBoard().stage == 0);
  
  // Clear any existing search tree
  if (rootNode != nullptr) {
    delete rootNode;
    rootNode = nullptr;
  }
  
  // Get current board state
  Color currentColor = boardHistory->getBoard().nextPla;
  
  // Create root node
  rootNode = new MCTSnode(this, currentColor, params.policyTemp);
  
  // Start with initial search visits
  int64_t totalVisits = 0;
  
  while (true) {
    
    // Also stop if we have a determined result
    if (rootNode->isWinDetermined) {
      break;
    }
    
    // Perform a batch of searches
    int visitToSearch = 0.5 * totalVisits + 5;
    SearchResult result = search(rootNode, visitToSearch, true);
    totalVisits += result.newVisits;
    
    // Get current attacker value (win rate from attacker's perspective)
    double attackerWinrate = (getRootValue() + 1)/2;
    if(attackerWinrate<0.0001) {
      attackerWinrate = 0.0001;
    }
    
    // Calculate stopping condition: visits / (attackerValue + 1) > searchFactor
    double ratio = (double)totalVisits / pow(attackerWinrate ,2);
    
    if (ratio > searchFactor) {
      break;
    }
    
  }
  if(rootNode->isWinDetermined 
    && rootNode->winner == attackPlayer) {
    return true;
  }
  return false;
}

MCTSsearch::SearchResult MCTSsearch::search(MCTSnode* node, uint64_t remainVisits, bool isRoot) {
  if (remainVisits == 0)
    ASSERT_UNREACHABLE;
    
    if (!isRoot) remainVisits = std::min(remainVisits, uint64_t(params.expandFactor * double(node->visits)) + 1);
    
    SearchResult SR = {0, 0.0};
    
    // If outcome is determined, just update visits
    if (node->isWinDetermined) {
        node->visits += remainVisits;
        SR.newVisits = remainVisits;
        SR.WRchange = sureResultWR(node->winner, attackPlayer, node->stepsToWin) * remainVisits;
        node->WRtotal += SR.WRchange;
        return SR;
    }
    
    Color color = node->nextColor;
    Color opp = getOpp(color);
    
    while (remainVisits > 0 && !terminate.load(std::memory_order_relaxed)) {
        int nextChildID = selectChildIDToSearch(node);
        Loc nextChildLoc = node->children[nextChildID].loc;
        SearchResult childSR;
        
        if (nextChildID >= node->childrennum) { // New child
            node->childrennum++;
            
            // Make move and check if outcome is determined
            boardHistory->play(color, nextChildLoc);
            
            node->children[nextChildID].ptr = new MCTSnode(this, boardHistory->getBoard().nextPla, params.policyTemp);
            
            
            boardHistory->undo();
            
            childSR.newVisits = 1;
            childSR.WRchange = node->children[nextChildID].ptr->WRtotal;
        } else {
            // Existing child
            boardHistory->play(color, nextChildLoc);
            childSR = search(node->children[nextChildID].ptr, remainVisits, false);
            boardHistory->undo();
        }
        
        // Update stats - both child and parent nodes are from attacker's perspective, directly accumulate
        remainVisits -= childSR.newVisits;
        node->visits += childSR.newVisits;
        node->WRtotal += childSR.WRchange;  // Direct accumulation, both from attacker's perspective
        SR.newVisits += childSR.newVisits;
        SR.WRchange += childSR.WRchange;    // Direct accumulation, both from attacker's perspective
        
        // Check if this node's outcome is now determined
        if (!node->isWinDetermined) {
          auto res = checkWinnerDetermined(node);
          if (res.first != C_WALL)
          {
            node->isWinDetermined = true;
            node->winner = res.first;
            node->stepsToWin = res.second;
            node->visits += remainVisits;
            SR.newVisits += remainVisits;
            remainVisits = 0;
            double oldWRtotal = node->WRtotal;
            node->WRtotal = sureResultWR(node->winner, attackPlayer, node->stepsToWin) * node->visits;
            SR.WRchange += (node->WRtotal - oldWRtotal);
            break;
          }
        }
    }
    
    return SR;
}

std::pair<Color, int64_t> MCTSsearch::checkWinnerDetermined(const MCTSnode* node) const {
    if (node->isWinDetermined) return std::make_pair(C_WALL, 0);
    
    // Check if all children have determined outcomes
    if (node->childrennum == 0) 
    {
        assert(node->visits == 1);
        return std::make_pair(C_WALL, 0);
    }
    
    int bestResult = -2;//-2:lose, 1:win, -1:draw, 0:undetermined
    if(node->childrennum < node->legalChildrennum)//Not all children have been expanded
    {
        bestResult = 0;
    }
    int shortestStepsToWin = INT_MAX;
    int longestStepsToLoss = INT_MIN;
    Loc bestLoc = Board::NULL_LOC;
    
    for (int i = 0; i < node->childrennum; i++) {
        int result = -2;
        const MCTSnode* child = node->children[i].ptr;
        if (child == nullptr || !child->isWinDetermined) {
            result = 0;
        }
        else if (child->winner == node->nextColor) {
            result = 1;
        }
        else if (child->winner == getOpp(node->nextColor)) {
            result = -2;
        }
        else if (child->winner == C_EMPTY) {
            result = -1;
        }
        else assert(false);

        if (result == -2) {
            longestStepsToLoss = std::max(longestStepsToLoss, child->stepsToWin);
        }
        else if (result == 1) {
            shortestStepsToWin = std::min(shortestStepsToWin, child->stepsToWin);
        }
        
        if (result > bestResult) {
            bestResult = result;
            bestLoc = node->children[i].loc;
        }
    }
    
    if (bestResult != 0) {
        Color winner = bestResult==1 ? node->nextColor : bestResult==-2 ? getOpp(node->nextColor) : C_EMPTY;
        int stepsToWin = bestResult==1 ? shortestStepsToWin : bestResult==-2 ? longestStepsToLoss : 0;
        return std::make_pair(winner, stepsToWin);
    }
    
    return std::make_pair(C_WALL, 0);
}

int MCTSsearch::selectChildIDToSearch(MCTSnode* node) {
    int childrennum = node->childrennum;
    if (childrennum == 0) return 0;
    
    double bestSelectionValue = -1e20;
    int bestChildID = -1;
    
    double totalVisit = node->visits;
    double puctFactor = MCTSpuctFactor(totalVisit, params.puct, params.puctPow, params.puctBase);
    double parentValue = node->WRtotal / node->visits;  // Parent node's attacker perspective value
    if (node->nextColor != attackPlayer)
      parentValue = -parentValue;
    
    float totalChildPolicy = 0;
    for (int i = 0; i < childrennum; i++) {
        const MCTSnode* child = node->children[i].ptr;
        double visit = child->visits;
        // Both child and parent nodes are from attacker's perspective, use directly
        double value = child->WRtotal / visit;
        float policy = float(node->children[i].policy) * policyQuantInv;
        totalChildPolicy += policy;
        
        // Prioritize determined winning moves
        if (child->isWinDetermined) {
            if (child->winner == attackPlayer) {
                value = 1.0; // Attacker wins
            } else if (child->winner != C_WALL) {
                value = -1.0; // Attacker loses
            }
        }
        if (node->nextColor != attackPlayer)
          value = -value;
        double selectionValue = MCTSselectionValue(puctFactor, (value+1.0)/2.0, visit, policy);
        if (child->isWinDetermined) {
          if (child->winner == node->nextColor) {
            selectionValue += 10000;
            //assert(false);
          }
          else {
            selectionValue -= 10000;
          }
        }
        if (selectionValue > bestSelectionValue) {
            bestSelectionValue = selectionValue;
            bestChildID = i;
        }
    }
    
    // Check new child
    if (childrennum < node->legalChildrennum) {
        double winrate = (parentValue + 1.0) / 2.0;
        double fpuFactor = 1.0 - sqrt(totalChildPolicy) * params.fpuReductionPolicy - params.fpuReductionConst;
        float policy = float(node->children[childrennum].policy) * policyQuantInv;
        double visit = 0;
        double selectionValue = MCTSselectionValue(puctFactor, winrate*fpuFactor, visit, policy);
        if (selectionValue > bestSelectionValue) bestChildID = childrennum;
    }
    
    return bestChildID;
}

std::vector<std::pair<Loc, float>> MCTSsearch::getLegalMovesAndVCFResultWithPolicy(Color color, Color& maybeWinner, int& gameEndMovenum) {
    const Board& board0 = boardHistory->getBoard();
    Hash128 sitHash = BoardHistory::getSituationRulesHash(board0, *boardHistory, color);
    
    // Try to get from cache first
    if (cacheTable != nullptr) {
        MCTS_CacheTable::Entry entry;
        if (cacheTable->get(sitHash, entry)) {
            if (!entry.legalMovesWithPolicy.empty()) {
                return entry.legalMovesWithPolicy;
            }
        }
    }
    
    // Calculate legal moves with policy
    //std::vector<Loc> legalLocs = VCFLogic::getAllVCFAttackOrDefenseLocs(board0, attackPlayer, maybeWinner, gameEndMovenum);
    std::vector<Loc> legalLocs = VCFLogic::getAllVCFAttackOrDefenseLocsWithCache(board0, attackPlayer, maybeWinner, gameEndMovenum, boardHistory->stoneTupleCountCache);
    if (legalLocs.empty())
      assert(maybeWinner != C_WALL);
    
    std::vector<std::pair<Loc, float>> moves;
    
    if (maybeWinner != C_WALL || legalLocs.empty()) {
        // Game is determined or no legal moves
      if (false) //fast results without NN, no need to save
      {
        if (cacheTable != nullptr) {
          MCTS_CacheTable::Entry entry;
          entry.hash = sitHash;
          entry.maybeWinner = maybeWinner;
          entry.gameEndMovenum = gameEndMovenum;
          entry.legalMovesWithPolicy = moves;
          cacheTable->set(entry);
        }
      }
      return moves;
    }
    
    // Get policy from NNUE
    NNUE::PolicyType policy[MaxBS * MaxBS + 1];
    NNUE::ValueType value = boardHistory->evaluateFull(color, policy);
    
    // Collect moves with policy values
    const Board& board1 = boardHistory->getBoard();
    for (const Loc& loc : legalLocs) {
        if (board1.isLegal(loc, color)) {
            int nu_loc = Location::getX(loc, board1.x_size) + Location::getY(loc, board1.x_size) * MaxBS;
            moves.push_back(std::make_pair(loc, (float)policy[nu_loc]));
        }
    }
    
    // Apply local policy bonus for stage 1 if conditions are met
    if (color == attackPlayer && board1.stage == 1 && params.localPolicyBonusStage1 != 0.0f && board1.isOnBoard(board1.firstLoc)) {
        // Check if current player is the attacking player (first player to move)
        
        for (size_t i = 0; i < moves.size(); i++) {
            Loc move = moves[i].first;
            if (board1.isOnBoard(move)) {
                int d2 = Location::euclideanDistanceSquared(move, board1.firstLoc, board1.x_size);
                float bonus = policyQuantFactor * params.localPolicyBonusStage1 * (-log(d2+36.0));
                moves[i].second += bonus;
            }
        }
        
    }
    
    // Sort by policy (highest first)
    std::sort(moves.begin(), moves.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });
    
    assert(moves.size()>0);
    // Find max policy for numerical stability
    float maxPolicy = moves[0].second; // Already sorted, so first is max
    
    // Calculate exp(policy - maxPolicy) for each move
    float factor = 1.0f / (params.policyTemp * policyQuantFactor);
    float sumExp = 0.0f;
    for (auto& move : moves) {
      float p = std::exp(factor * (move.second - maxPolicy));
      sumExp += p;
    }
    
    float factor2 = 1.0f / sumExp;
    // Normalize to get softmax probabilities
    for (auto& move : moves) {
        move.second = factor2 * std::exp(factor * (move.second - maxPolicy));
        if (move.second < 0.001f)move.second = 0.001f;//avoid too little policy
    }
    
    
    // Cache the result
    if (cacheTable != nullptr) {
        MCTS_CacheTable::Entry entry;
        entry.hash = sitHash;
        entry.maybeWinner = maybeWinner;
        entry.gameEndMovenum = gameEndMovenum;
        entry.nnueValue = value;
        entry.legalMovesWithPolicy = moves;
        cacheTable->set(entry);
    }
    return moves;
}

NNUE::ValueType MCTSsearch::evaluatePosition(Color color) {
    Hash128 sitHash = BoardHistory::getSituationRulesHash(boardHistory->getBoard(), *boardHistory, color);
    
    // Try to get from cache first
    if (cacheTable != nullptr) {
        MCTS_CacheTable::Entry entry;
        if (cacheTable->get(sitHash, entry)) {
            return entry.nnueValue;
        }
    }
    
    // Calculate NNUE evaluation
    NNUE::ValueType value = boardHistory->evaluateFull(color, nullptr);
    
    // Cache the result
    if (cacheTable != nullptr) {
        MCTS_CacheTable::Entry entry;
        entry.hash = sitHash;
        entry.nnueValue = value;
        cacheTable->set(entry);
    }
    return value;
}

Loc MCTSsearch::bestRootMove() const {
    if (rootNode == nullptr || rootNode->childrennum == 0) return Board::NULL_LOC;
    
    int bestChildID = -1;
    uint64_t bestVisits = 0;
    
    for (int i = 0; i < rootNode->childrennum; i++) {
        if (rootNode->children[i].ptr != nullptr) {
            uint64_t visits = rootNode->children[i].ptr->visits;
            // Prioritize determined winning moves
            if (rootNode->children[i].ptr->isWinDetermined && rootNode->children[i].ptr->winner == attackPlayer) {
                return rootNode->children[i].loc;
            }
            if (visits > bestVisits) {
                bestVisits = visits;
                bestChildID = i;
            }
        }
    }
    
    return bestChildID >= 0 ? rootNode->children[bestChildID].loc : Board::NULL_LOC;
}

float MCTSsearch::getRootValue() const {
    if (rootNode == nullptr) return 0.0f;
    if (rootNode->visits == 0) return 0.0f;
    
    if (rootNode->isWinDetermined) {
        return (rootNode->winner == attackPlayer) ? 1.0f : -1.0f;
    }
    
    return float(rootNode->WRtotal / rootNode->visits);
}

int64_t MCTSsearch::getRootVisit() const {
    return rootNode ? rootNode->visits : 0;
}

std::vector<std::pair<Loc, uint64_t>> MCTSsearch::getPV() const {
    std::vector<std::pair<Loc, uint64_t>> pv;
    if (rootNode == nullptr) return pv;
    
    MCTSnode* currentNode = rootNode;
    
    // Follow the path of best children until we reach a leaf
    while (currentNode != nullptr && currentNode->childrennum > 0) {
        int bestChildIndex = -1;
        uint64_t maxVisits = 0;
        int shortestWinSteps = INT_MAX;
        bool hasWinningChild = false;
        
        // First pass: look for winning children for current player
        for (int i = 0; i < currentNode->childrennum; i++) {
            MCTSnode* child = currentNode->children[i].ptr;
            if (child != nullptr && child->isWinDetermined && 
                child->winner == currentNode->nextColor ) {
                assert(child->stepsToWin>0);
                // This is a winning move for current player
                if (!hasWinningChild || child->stepsToWin < shortestWinSteps) {
                    hasWinningChild = true;
                    shortestWinSteps = child->stepsToWin;
                    bestChildIndex = i;
                    maxVisits = child->visits;
                }
            }
        }
        
        // If no winning child found, select the most visited child
        if (!hasWinningChild) {
            for (int i = 0; i < currentNode->childrennum; i++) {
                MCTSnode* child = currentNode->children[i].ptr;
                if (child != nullptr && child->visits > maxVisits) {
                    maxVisits = child->visits;
                    bestChildIndex = i;
                }
            }
        }
        
        // If we found a best child, add its move and visit count to PV and continue
        if (bestChildIndex >= 0) {
            MCTSnode* bestChild = currentNode->children[bestChildIndex].ptr;
            pv.push_back(std::make_pair(currentNode->children[bestChildIndex].loc, bestChild->visits));
            currentNode = bestChild;
        } else {
            break;
        }
    }
    
    return pv;
}

void MCTSsearch::loadParamFile(std::string filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open param file: " << filename << std::endl;
        return;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        size_t pos = line.find('=');
        if (pos == std::string::npos) continue;
        
        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);
        
        if (key == "expandFactor") params.expandFactor = std::stod(value);
        else if (key == "puct") params.puct = std::stod(value);
        else if (key == "puctPow") params.puctPow = std::stod(value);
        else if (key == "puctBase") params.puctBase = std::stod(value);
        else if (key == "fpuReductionPolicy") params.fpuReductionPolicy = std::stod(value);
        else if (key == "fpuReductionConst") params.fpuReductionConst = std::stod(value);
        else if (key == "policyTemp") params.policyTemp = std::stod(value);
        else if (key == "localPolicyBonusStage1") params.localPolicyBonusStage1 = std::stod(value);
    }
}

int64_t MCTSsearch::calculateWinningDependencyTreeSize() const {
    if (rootNode == nullptr) {
        return 0;
    }
    
    // Check if root node has a determined winning outcome
    if (!rootNode->isWinDetermined) {
        // Root node outcome is not determined, return 0
        return 0;
    }
    
    return calculateWinningDependencyTreeSizeRecursive(rootNode);
}

int64_t MCTSsearch::calculateWinningDependencyTreeSizeRecursive(const MCTSnode* node) const {
    if (node == nullptr) {
        return 0;
    }
    
    int64_t count = 1; // Count current node
    
    // If this node has a determined outcome, we need to traverse the dependency tree
    if (node->isWinDetermined) {
        // For a winning node, we need to include at least one winning child path
        // For a losing node, we need to include all children (since opponent can choose any)
        
        if (node->winner == node->nextColor) {
            // This is a winning node for the current player
            // Find the best winning move and include its dependency tree
            int bestSteps = INT_MAX;
            const MCTSnode* bestChild = nullptr;
            
            for (int i = 0; i < node->childrennum; i++) {
                const MCTSnode* child = node->children[i].ptr;
                if (child != nullptr && child->isWinDetermined && 
                    child->winner == node->winner && 
                    child->stepsToWin < bestSteps) {
                    bestSteps = child->stepsToWin;
                    bestChild = child;
                }
            }
            
            if (bestChild != nullptr) {
                count += calculateWinningDependencyTreeSizeRecursive(bestChild);
            }
            else{
              assert(node->childrennum==0);
            }
        } else if (node->winner == getOpp(node->nextColor)) {
            // This is a losing node for the current player
            // Include all children in the dependency tree since opponent controls the choice
            for (int i = 0; i < node->childrennum; i++) {
                const MCTSnode* child = node->children[i].ptr;
                if (child != nullptr) {
                    count += calculateWinningDependencyTreeSizeRecursive(child);
                }
            }
        }
        // For draw nodes (winner == C_EMPTY), we don't need to traverse further
    }
    
    return count;
}

std::vector<int8_t> MCTSsearch::calculateDefenseDependencyMap() {
    static_assert(IMMEDIATE_WIN_SEARCH_LAYERS == 0);
    Hash128 hash_init = boardHistory->getBoard().pos_hash;
    std::vector<int8_t> dependMap(Board::MAX_ARR_SIZE, 0);
    
    if (rootNode == nullptr || rootNode->winner != attackPlayer) {
        assert(false);
    }
    
    
    calculateDefenseDependencyMapRecursive(rootNode, dependMap);
    Hash128 hash_end = boardHistory->getBoard().pos_hash;
    assert(hash_init == hash_end);
    return dependMap;
}

void MCTSsearch::calculateDefenseDependencyMapRecursive(const MCTSnode* node, std::vector<int8_t>& dependMap) {
    if (node == nullptr) {
        return;
    }
    
    // Get current board state by reconstructing from boardHistory
    // Note: This is a simplified approach - in practice, we'd need to track the board state
    // through the search tree path. For now, we'll use the current board state.
    
    // Execute markAllDefenseDependedLocs for current node
    VCFLogic::markAllDefenseDependedLocs(boardHistory->getBoard(), attackPlayer, dependMap);
    
    // If this node has a determined outcome, recursively process relevant children
    if (node->isWinDetermined) {
        if (node->winner == node->nextColor) {
            // This is a winning node - find the best winning child
            int bestSteps = INT_MAX;
            const MCTSnode* bestChild = nullptr;
            Loc bestMove = Board::NULL_LOC;
            
            for (int i = 0; i < node->childrennum; i++) {
                const MCTSnode* child = node->children[i].ptr;
                if (child != nullptr && child->isWinDetermined && 
                    child->winner == node->winner && 
                    child->stepsToWin < bestSteps) {
                    bestSteps = child->stepsToWin;
                    bestChild = child;
                    bestMove = node->children[i].loc;
                }
            }
            
            if (bestChild != nullptr && bestMove != Board::NULL_LOC) {
                // Play the move to update board state
                boardHistory->play(node->nextColor, bestMove);
                
                // Recursively process the child
                calculateDefenseDependencyMapRecursive(bestChild, dependMap);
                
                // Undo the move to restore board state
                boardHistory->undo();
            }
        } else if (node->winner == getOpp(node->nextColor)) {
            // This is a losing node - process all children
            for (int i = 0; i < node->childrennum; i++) {
                const MCTSnode* child = node->children[i].ptr;
                if (child != nullptr) {
                    Loc move = node->children[i].loc;
                    if (move != Board::NULL_LOC) {
                        // Play the move to update board state
                        boardHistory->play(node->nextColor, move);
                        
                        // Recursively process the child
                        calculateDefenseDependencyMapRecursive(child, dependMap);
                        
                        // Undo the move to restore board state
                        boardHistory->undo();
                    }
                }
            }
        }
    }
}

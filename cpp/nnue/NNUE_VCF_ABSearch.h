#ifndef NNUE_NNUE_VCF_ABSEARCH_H_
#define NNUE_NNUE_VCF_ABSEARCH_H_

#include "../game/board.h"
#include "../game/boardhistory.h"
#include "../search/mutexpool.h"
#include "NNUEBoardHistory.h"
#include <vector>
#include <utility>

namespace NNUE {

class ABSearch_CacheTable {
public:
  struct Entry {
    Hash128 hash;
    double value;
    float depth;
    Loc bestloc0;
    Loc bestLoc1;//reserve
    Entry();
    ~Entry();
  };
private:
  Entry* entries;
  MutexPool* mutexPool;
  uint64_t tableSize;
  uint64_t tableMask;
  uint32_t mutexPoolMask;

public:
  ABSearch_CacheTable(int sizePowerOfTwo, int mutexPoolSizePowerOfTwo);
  ~ABSearch_CacheTable();

  ABSearch_CacheTable(const ABSearch_CacheTable& other) = delete;
  ABSearch_CacheTable& operator=(const ABSearch_CacheTable& other) = delete;

  //These are thread-safe. For get, ret will be set to nullptr upon a failure to find.
  bool get(Hash128 nnHash, Entry& ret);
  void set(const Entry& entry);
  void clear();
};

// AB search class for checking consecutive four-threat wins in Connect6 with one player making two moves
class VCF_ABSearch {
public:
  NNUEBoardHistory* boardHistory;
  Player attackPlayer;   // Attacking player
  Player defendPlayer;   // Defending player
  double remainingDepth; // Remaining search depth (equivalent layers)
  int64_t nodeCount;
  ABSearch_CacheTable* cacheTable;
  int64_t nnevalCount;

  VCF_ABSearch(NNUEBoardHistory* hist, ABSearch_CacheTable* cache, Player pla);
  ~VCF_ABSearch();
  
  // Main search function
  double search(double maxDepth);
  
private:
  // AB search core function
  double alphaBeta(double alpha, double beta, double depth);
  
  // Check game state
  int checkGameState();
  
  // Get legal moves and sort by policy
  std::vector<std::pair<Loc, double>> getLegalMovesWithPolicy(const std::vector<Loc>& legalLocs, bool hasLegalLocs);
  
  // Calculate leaf node evaluation
  double evaluateLeafAssumeNotEnd();
};

} // namespace NNUE

#endif // NNUE_NNUE_VCF_ABSEARCH_H_
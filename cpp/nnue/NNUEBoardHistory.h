#ifndef NNUE_NNUEBOARDHISTORY_H_
#define NNUE_NNUEBOARDHISTORY_H_

#include "../game/boardhistory.h"
#include "../game/board.h"
#include "Eva_nnuev2.h"
#include "../neuralnet/nninputs.h"

#include <vector>

using namespace NNUE;
using namespace NNUEV2;

//A BoardHistory that maintains NNUE evaluators and replaces NNUEBoard functionality
class NNUEBoardHistory : public BoardHistory {
public:

    //skip "resultsBeforeNN" in nninput. for VCF this can be "true" to reduce cost
    bool skipResultsBeforeNN = false;

    // NNUE evaluators for both players
    Eva_nnuev2 blackEvaluator;
    Eva_nnuev2 whiteEvaluator;
    
    // Input buffers for neural network
    MiscNNInputParams nnInputParams;
    float gfInputBuf[NNUEV2::globalFeatureNum];
    bool illegalMapBuf[MaxBS * MaxBS];
    
    // Historical boards for undo operations
    std::vector<Board> historicalBoards;
    
    //Cache for number of black and white stone nums for every 6-location tuple
    //[0] +x direction, [1] +y direction, [2] +x+y direction, [3] -x+y direction
    //uint8_t[0:2] for black, uint8_t[3:5] for white, uint8_t[7] for illegal(out of board)
    uint8_t stoneTupleCountCache[4][Board::MAX_ARR_SIZE];
    
    // Move cache for efficient undo operations
    struct MoveCache {
        bool isUndo;
        Color color;
        Loc loc;
        MoveCache() : isUndo(false), color(C_EMPTY), loc(Board::NULL_LOC) {}
        MoveCache(bool isUndo, Color color, Loc loc) : isUndo(isUndo), color(color), loc(loc) {}
    };
    
    MoveCache moveCacheB[MaxBS * MaxBS], moveCacheW[MaxBS * MaxBS];
    int moveCacheBlength, moveCacheWlength;

    // Constructors
    NNUEBoardHistory(const ModelWeight* weights, const MiscNNInputParams& nnInputParams, bool skipResultsBeforeNN);
    NNUEBoardHistory(const Board& board, Player pla, const Rules& rules, const ModelWeight* weights, const MiscNNInputParams& nnInputParams, bool skipResultsBeforeNN);
    
    // Copy and move constructors
    NNUEBoardHistory(const NNUEBoardHistory& other);
    NNUEBoardHistory& operator=(const NNUEBoardHistory& other);
    NNUEBoardHistory(NNUEBoardHistory&& other) noexcept;
    NNUEBoardHistory& operator=(NNUEBoardHistory&& other) noexcept;
    
    // Destructor
    ~NNUEBoardHistory();
    
    // Initialize with model weights
    void initializeEvaluators(const ModelWeight* weights);
    
    // Clear and reset the board history with NNUE state
    void clear(const Board& board, Player pla, const Rules& rules);
    
    // NNUEBoard replacement methods
    void play(Color color, Loc loc);
    void undo();
    
    // Get current board state
    const Board& getBoard() const;
    

    
    
    // Evaluation methods
    NNUE::ValueType evaluateFull(Color color, NNUE::PolicyType* policy);
    //void evaluatePolicy(Color color, NNUE::PolicyType* policy);
    //NNUE::ValueType evaluateValue(Color color);
    
    // Validation method
    bool checkEvaluatorBoardConsistency();
    
    // Get stone tuple count cache for VCF optimization
    // Returns counts for 6-tuple starting at loc in direction dir
    // dir: 0=+x, 1=+y, 2=+x+y, 3=-x+y
    // Returns: bits 0-2 for black count, bits 3-5 for white count, bit 7 for out-of-board
    uint8_t getStoneTupleCount(int dir, Loc loc) const {
        return stoneTupleCountCache[dir][loc];
    }
    
    // Get black and white counts separately
    void getStoneTupleCounts(int dir, Loc loc, int& blackCount, int& whiteCount, bool& outOfBoard) const {
        uint8_t cache = stoneTupleCountCache[dir][loc];
        outOfBoard = (cache & 0x80) != 0;
        if (!outOfBoard) {
            blackCount = cache & 0x07;
            whiteCount = (cache >> 3) & 0x07;
        } else {
            blackCount = whiteCount = 0;
        }
    }
    
private:
    // Update input buffers for neural network evaluation
    void updateInputBuf(Color nextPlayer);

    // Add move to cache for efficient undo
    void addCache(bool isUndo, Color color, Loc loc);
    
    // Clear cache for specific color
    void clearCache(Color color);
    
    // Check if two moves are contrary (undo each other)
    bool isContraryMove(MoveCache a, MoveCache b);
    
    // Internal helper to sync NNUE state with board state
    void syncNNUEWithBoard(const Board& board);
    
    // Stone tuple count cache management
    void computeStoneTupleCountCache(const Board& board);
    void updateStoneTupleCountCache(Loc loc, Color color, bool isUndo);
    bool checkStoneTupleCountCacheConsistency(const Board& board);
};

#endif // NNUE_NNUEBOARDHISTORY_H_
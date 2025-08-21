#pragma once

#include "NNUE_VCF_MCTSsearch.h"
#include "../nnue/NNUEBoardHistory.h"
#include "../nnue/Eva_nnuev2.h"
#include "../game/board.h"
#include <vector>



// Forward declaration - VCFresults definition to be provided by user
struct VCFPrunedInfo{
    Loc loc;
    bool isPruned; //if pruned, if play here, the opponent will win by VCF
    int16_t moveNum; //opponent will win by VCF in moveNum moves

    bool notPruned; //if not pruned, the opponent will surely not win by VCF
    float calculateFactor; //how many visits have VCF calculated
    float value; //the value of VCF side

    VCFPrunedInfo();
    VCFPrunedInfo(Loc loc, bool isPruned, int16_t moveNum, bool notPruned, float calculateFactor, float value);
};

class VCFCalculator {
public:
    // Constructor
    VCFCalculator(NNUE_VCF_MCTSsearch::MCTS_CacheTable* cacheTable, const NNUEV2::ModelWeight* weights);
    
    // Destructor
    ~VCFCalculator();
    
    // Calculate all VCF defend results
    std::vector<VCFPrunedInfo> CalculateAllVCFDefendResults(
        const Board& board, 
        Color attackPlayer, 
        int maxMove, 
        double searchFactor
    );

    // Calculate all VCF defend results
    // Version 2: Prune locations during calculating other defenses
    std::vector<VCFPrunedInfo> CalculateAllVCFDefendResultsV2(
        const Board& board,
        Color attackPlayer,
        int maxMove,
        double searchFactor
    );
    
    // Calculate shortest VCF steps and return dependency map
    int calculateShortestVCFAndDependMap(
        const Board& board,
        Color attackPlayer,
        int initialMaxMove,
        int minMaxMove,
        int recommendedMaxMove,
        double searchFactor,
        std::vector<int8_t>& dependMap,
        bool noOptimize //if true, return immediately when find a vcf, not optimize the VCF steps
    );
    
private:
    NNUE_VCF_MCTSsearch::MCTS_CacheTable* cacheTable;
    const NNUEV2::ModelWeight* weights;

    NNUEBoardHistory nnueHistory1;
    
    // Calculate all VCF defend results
    std::vector<VCFPrunedInfo> CalculateAllVCFDefendResults_stage0(
        const Board& board, 
        Color attackPlayer, 
        int maxMove, 
        double searchFactor
    );
    
    // Calculate all VCF defend results
    std::vector<VCFPrunedInfo> CalculateAllVCFDefendResults_stage1(
        const Board& board, 
        Color attackPlayer, 
        int maxMove, 
        double searchFactor
    );
    // Calculate all VCF defend results
    std::vector<VCFPrunedInfo> CalculateAllVCFDefendResultsV2_stage0(
        const Board& board,
        Color attackPlayer,
        int maxMove,
        double searchFactor
    );

    // Calculate all VCF defend results
    std::vector<VCFPrunedInfo> CalculateAllVCFDefendResultsV2_stage1(
        const Board& board,
        Color attackPlayer,
        int maxMove,
        double searchFactor
    );
};

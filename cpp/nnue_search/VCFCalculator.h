#pragma once

#include "NNUE_VCF_MCTSsearch.h"
#include "../nnue/NNUEBoardHistory.h"
#include "../nnue/Eva_nnuev2.h"
#include "../game/board.h"
#include <vector>
#include <map>

class VCFCalculator {
public:
    // Constructor
    VCFCalculator(NNUE_VCF_MCTSsearch::MCTS_CacheTable* cacheTable, const NNUEV2::ModelWeight* weights);
    
    // Destructor
    ~VCFCalculator();
    
    // Calculate all VCF defend results
    std::map<Loc,int16_t> CalculateAllVCFDefendResults(
        const Board& board, 
        Color attackPlayer, 
        int maxMove, 
        double searchFactor
    );

    // Calculate all VCF defend results
    // Version 2: Prune locations during calculating other defenses
    std::map<Loc,int16_t> CalculateAllVCFDefendResultsV2(
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
        Loc& winLoc,
        bool noOptimize //if true, return immediately when find a vcf, not optimize the VCF steps
    );
    // Calculate shortest VCF steps
    int calculateShortestVCF(
        const Board& board,
        Loc& winLoc,
        Color attackPlayer,
        int initialMaxMove,
        double searchFactor
    );
    
private:
    NNUE_VCF_MCTSsearch::MCTS_CacheTable* cacheTable;
    const NNUEV2::ModelWeight* weights;

    NNUEBoardHistory nnueHistory1;
    
    // Calculate all VCF defend results
    std::map<Loc,int16_t> CalculateAllVCFDefendResults_stage0(
        const Board& board, 
        Color attackPlayer, 
        int maxMove, 
        double searchFactor
    );
    
    // Calculate all VCF defend results
    std::map<Loc,int16_t> CalculateAllVCFDefendResults_stage1(
        const Board& board, 
        Color attackPlayer, 
        int maxMove, 
        double searchFactor
    );
    // Calculate all VCF defend results
    std::map<Loc,int16_t> CalculateAllVCFDefendResultsV2_stage0(
        const Board& board,
        Color attackPlayer,
        int maxMove,
        double searchFactor
    );

    // Calculate all VCF defend results
    std::map<Loc,int16_t> CalculateAllVCFDefendResultsV2_stage1(
        const Board& board,
        Color attackPlayer,
        int maxMove,
        double searchFactor
    );
};

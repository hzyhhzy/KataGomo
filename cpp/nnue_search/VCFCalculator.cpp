#include "VCFCalculator.h"
#include "../neuralnet/nninputs.h"
#include "../game/rules.h"
#include "../nnue_search/VCFLogic.h"
#include <algorithm>
#include <set>

using namespace NNUE_VCF_MCTSsearch;
using namespace NNUE;
using namespace NNUEV2;

static const bool VCFCalculator_debug_print = true;

VCFPrunedInfo::VCFPrunedInfo()
    : loc(Board::NULL_LOC), isPruned(false), moveNum(0), notPruned(false), calculateFactor(0), value(0) {
}

VCFPrunedInfo::VCFPrunedInfo(Loc loc, bool isPruned, int16_t moveNum, bool notPruned, float calculateFactor, float value)
    : loc(loc), isPruned(isPruned), moveNum(moveNum), notPruned(notPruned), calculateFactor(calculateFactor), value(value) {
}

VCFCalculator::VCFCalculator(MCTS_CacheTable* cacheTable, const ModelWeight* weights)
    : cacheTable(cacheTable), weights(weights), nnueHistory1(weights, MiscNNInputParams(), true) {
}

VCFCalculator::~VCFCalculator() {
    // No cleanup needed for pointer members (not owned by this class)
}

std::vector<VCFPrunedInfo> VCFCalculator::CalculateAllVCFDefendResults(
    const Board& board, 
    Color attackPlayer, 
    int maxMove, 
    double searchFactor) {
    
    // Route to appropriate stage-specific function based on board stage
    if (board.stage == 1) {
        return CalculateAllVCFDefendResults_stage1(board, attackPlayer, maxMove, searchFactor);
    } else {
        assert(board.stage == 0);
        return CalculateAllVCFDefendResults_stage0(board, attackPlayer, maxMove, searchFactor);
    }
}

std::vector<VCFPrunedInfo> VCFCalculator::CalculateAllVCFDefendResultsV2(
    const Board& board,
    Color attackPlayer,
    int maxMove,
    double searchFactor) {

    // Route to appropriate stage-specific function based on board stage
    if (board.stage == 1) {
        return CalculateAllVCFDefendResultsV2_stage1(board, attackPlayer, maxMove, searchFactor);
    }
    else {
        assert(board.stage == 0);
        return CalculateAllVCFDefendResultsV2_stage0(board, attackPlayer, maxMove, searchFactor);
    }
}

std::vector<VCFPrunedInfo> VCFCalculator::CalculateAllVCFDefendResults_stage0(
    const Board& board,
    Color attackPlayer,
    int maxMove,
    double searchFactor) {

    Color defendPlayer = getOpp(attackPlayer);
    assert(board.stage == 0);
    assert(board.nextPla == defendPlayer);

    //check whether defend player can directly win
    {
        int defendMaxLenNow = VCFLogic::checkMaxConnectLen(board, defendPlayer);
        assert(defendMaxLenNow < 6);
        if (defendMaxLenNow - board.stage >= 4)//defend pla can win
        {
            return std::vector<VCFPrunedInfo>();
        }
    }


    Board boardWith2pass = board;
    boardWith2pass.playMoveAssumeLegal(Board::PASS_LOC, board.nextPla);
    boardWith2pass.playMoveAssumeLegal(Board::PASS_LOC, board.nextPla);

    // Calculate shortest VCF and dependency map for boardWith1pass
    std::vector<int8_t> dependMap1;
    int vcfSteps1 = calculateShortestVCFAndDependMap(boardWith2pass, attackPlayer, maxMove, 0, 0, searchFactor, dependMap1, false);

    std::vector<VCFPrunedInfo> results;
    
    if (vcfSteps1 <= 0) { //can't VCF even with 2 pass
        return results;
    }
    
    // Record positions where VCF is not possible (vcfSteps2 <= 0)
    std::set<Loc> noVcfPositions;

    // If VCF found, analyze all legal positions for pruning
    // First, collect all legal positions
    std::vector<Loc> legalPositions;
    for (int y1 = 0; y1 < board.y_size; y1++) {
        for (int x1 = 0; x1 < board.x_size; x1++) {
            Loc loc1 = Location::getLoc(x1, y1, board.x_size);
            if (board.isLegal(loc1, board.nextPla)) {
                legalPositions.push_back(loc1);
            }
        }
    }

    // Sort positions by dependMap value (2, 1, 0) and then by priority
    std::sort(legalPositions.begin(), legalPositions.end(), [&](Loc a, Loc b) {
        int8_t dependA = dependMap1[a];
        int8_t dependB = dependMap1[b];

        // First sort by dependMap value (descending: 2, 1, 0)
        if (dependA != dependB) {
            return dependA > dependB;
        }

        // Then sort by priority (descending)
        int xa = Location::getX(a, board.x_size);
        int ya = Location::getY(a, board.y_size);
        int xb = Location::getX(b, board.x_size);
        int yb = Location::getY(b, board.y_size);

        double priorityA = board.getLocationPriority(xa, ya);
        double priorityB = board.getLocationPriority(xb, yb);

        return priorityA < priorityB;
        });

    // Process positions in sorted order
    for (size_t i = 0; i < legalPositions.size(); i++) {
        Loc loc1 = legalPositions[i];
        Board board1 = board;
        board1.playMoveAssumeLegal(loc1, board1.nextPla);

        int8_t dependValue1 = dependMap1[loc1]; 
        //std::cout << i << " " << dependValue1 << " " << Location::toString(loc1, board1) << "\n";
        //std::cout.flush();

        std::vector<int8_t> dependMap2;
        int vcfSteps2 = 0;
        //if dependValue==1, the old vcf is not changed, but dependMap changed
        //if dependValue==2, the old vcf is changed, recalculate VCF for this position
        if (dependValue1 >= 1) {
            // Calculate VCF for the test board
            Board board1With1pass = board1;
            board1With1pass.playMoveAssumeLegal(Board::PASS_LOC, board1With1pass.nextPla);
            vcfSteps2 = calculateShortestVCFAndDependMap(board1With1pass, attackPlayer, maxMove, vcfSteps1, vcfSteps1, searchFactor, dependMap2, false);
            //std::cout << i <<" "<< vcfSteps2 << " " << Location::toString(loc1, board1) << "\n";
            //std::cout.flush();
            if(vcfSteps2<=0)
            {
                //VCF not possible after playing this move - position is not pruned
                // Record this position as having no VCF
                noVcfPositions.insert(loc1);
                continue;
            }
        }
        else { //the old vcf and dependMap is completely not influenced
            dependMap2 = dependMap1;
            vcfSteps2 = vcfSteps1;
        }


        bool canDefend = false;
        int longestDefense = vcfSteps2; //initially the vcf win move of pass

        //search all children
        // Collect all legal second move positions
        std::vector<Loc> legalPositions2;
        for (int y2 = 0; y2 < board.y_size; y2++) {
            for (int x2 = 0; x2 < board.x_size; x2++) {
                Loc loc2 = Location::getLoc(x2, y2, board.x_size);

                // Check if position is legal and meets priority requirements
                if (board1.isLegal(loc2, board1.nextPla) &&
                    board1.getLocationPriority(x2, y2) + Board::PRIOR_EPS >= board1.firstLocPriority) {
                    legalPositions2.push_back(loc2);
                }
            }
        }

        // Sort second move positions by dependMap value (2, 1, 0) and then by priority
        std::sort(legalPositions2.begin(), legalPositions2.end(), [&](Loc a, Loc b) {
            int8_t dependA = dependMap2[a];
            int8_t dependB = dependMap2[b];

            // First sort by dependMap value (descending: 2, 1, 0)
            if (dependA != dependB) {
                return dependA > dependB;
            }

            // Then sort by priority (descending)
            int xa = Location::getX(a, board1.x_size);
            int ya = Location::getY(a, board1.y_size);
            int xb = Location::getX(b, board1.x_size);
            int yb = Location::getY(b, board1.y_size);

            double priorityA = board1.getLocationPriority(xa, ya);
            double priorityB = board1.getLocationPriority(xb, yb);

            return priorityA < priorityB;
            });

        // Check if any second move position is in noVcfPositions (early defense detection)
         for (size_t j = 0; j < legalPositions2.size(); j++) {
             Loc loc2 = legalPositions2[j];
             if (noVcfPositions.count(loc2) > 0) {
                 canDefend = true;
                 break;
             }
         }
        
        // Process second move positions in sorted order
         for (size_t j = 0; j < legalPositions2.size(); j++) {
             Loc loc2 = legalPositions2[j];
             if(canDefend)
                 break;
                 
             int8_t dependValue2 = dependMap2[loc2];
            if (dependValue2 < 2)
            {
                continue;
            }

            // Need to test this position by actually playing it
            Board testBoard = board1;
            testBoard.playMoveAssumeLegal(loc2, board1.nextPla);

            // Calculate VCF for the test board
            std::vector<int8_t> testDependMap;
            int testVcfSteps = calculateShortestVCFAndDependMap(testBoard, attackPlayer, maxMove, vcfSteps2, vcfSteps2, searchFactor, testDependMap, false);

            if (testVcfSteps > 0) {
                longestDefense = std::max(longestDefense, testVcfSteps);
            }
            else {
                canDefend = true;
                break;
            }


        }

        if (!canDefend) //all 2nd move are losing
        {
            VCFPrunedInfo info(loc1, true, longestDefense, false, 0.0f, 0.0f);
            results.push_back(info);
        }


    }


    return results;



}


std::vector<VCFPrunedInfo> VCFCalculator::CalculateAllVCFDefendResultsV2_stage0(
    const Board& board,
    Color attackPlayer,
    int maxMove,
    double searchFactor) {

    const bool enable_loc3_prune = true;
    const int initialRecommendedMovenumExtra = 18;
    const int stage1RecommendedMovenumExtra = 4; //moves not pruned in stage1 often affects the VCF length, not try to make vcfStep2 equal to vcfStep1 because a failed VCF waste a lot of time
    const int stage2RecommendedMovenumExtra = 0; 



    Color defendPlayer = getOpp(attackPlayer);
    assert(board.stage == 0);
    assert(board.nextPla == defendPlayer);

    //check whether defend player can directly win
    {
        int defendMaxLenNow = VCFLogic::checkMaxConnectLen(board, defendPlayer);
        assert(defendMaxLenNow < 6);
        if (defendMaxLenNow - board.stage >= 4)//defend pla can win
        {
            return std::vector<VCFPrunedInfo>();
        }
    }


    Board boardWith2pass = board;
    boardWith2pass.playMoveAssumeLegal(Board::PASS_LOC, board.nextPla);
    boardWith2pass.playMoveAssumeLegal(Board::PASS_LOC, board.nextPla);

    // Calculate shortest VCF and dependency map for boardWith1pass
    std::vector<int8_t> dependMap1;
    int vcfSteps1 = calculateShortestVCFAndDependMap(boardWith2pass, attackPlayer, maxMove, 0, boardWith2pass.movenum + initialRecommendedMovenumExtra, searchFactor, dependMap1, false);

    std::vector<VCFPrunedInfo> results;
    
    if (vcfSteps1 <= 0) { //can't VCF even with 2 pass
        return results;
    }
    
    // Record positions where VCF is not possible (vcfSteps2 <= 0)
    //std::set<Loc> noVcfPositions;
    
    //16384 is not sure, 32767 is sure no VCF, minWinStepsMap[Loc1][Loc2] is the min win steps  Loc1+Loc2
    const int16_t NOT_SURE = 16384;
    const int16_t SURE_NO_VCF = 32767;
    int16_t minWinStepsMap[Board::MAX_ARR_SIZE][Board::MAX_ARR_SIZE];
    for(int i=0;i<Board::MAX_ARR_SIZE;i++)
    {
        for(int j=0;j<Board::MAX_ARR_SIZE;j++)
        {
            minWinStepsMap[i][j] = NOT_SURE;
        }
    }

    
    for (int y1 = 0; y1 < board.y_size; y1++) {
        for (int x1 = 0; x1 < board.x_size; x1++) {
            Loc loc1 = Location::getLoc(x1, y1, board.x_size);
            if(!board.isLegal(loc1, board.nextPla))
                continue;
            int8_t dependValue1 = dependMap1[loc1];

            if(dependValue1 == 2)
            {
                Board board1 = board;
                board1.playMoveAssumeLegal(loc1, board1.nextPla);
                board1.playMoveAssumeLegal(Board::PASS_LOC, board1.nextPla);
                

                std::vector<int8_t> dependMap2;
                int vcfSteps2 = 0;
                int recommendedMaxMoveStage1 = vcfSteps1 + stage1RecommendedMovenumExtra;
                vcfSteps2 = calculateShortestVCFAndDependMap(board1, attackPlayer, maxMove, recommendedMaxMoveStage1, recommendedMaxMoveStage1, searchFactor, dependMap2, false);
                if(vcfSteps2 > 0)
                {
                    //all locs with loc1 has no vcf
                    for(Loc loc2=0;loc2<Board::MAX_ARR_SIZE;loc2++)
                    {
                        if(dependMap2[loc2] < 2)
                        {
                            if(minWinStepsMap[loc1][loc2] > NOT_SURE)
                            {
                                //very rare case, may have bug
                                //if no bug, there is a very difficult VCF that loc1-pass found it and marked loc2 as not-depended loc, but loc2-pass didn't find a VCF
                                std::cout<<"Warning: minWinStepsMap[loc1][loc2] > NOT_SURE but dependMap2[loc2] < 2, "<<minWinStepsMap[loc1][loc2]<<" "<<vcfSteps2<<std::endl;
                                std::cout<<Location::toString(loc1,board)<<" "<<Location::toString(loc2, board)<<std::endl;
                                Board::printBoard(std::cout,board,loc1,nullptr);
                            }
                            minWinStepsMap[loc1][loc2] = std::min(minWinStepsMap[loc1][loc2], int16_t(vcfSteps2));
                            minWinStepsMap[loc2][loc1] = std::min(minWinStepsMap[loc2][loc1], int16_t(vcfSteps2));
                        }
                    }
                }
                else{
                    //all locs with loc1 has no vcf
                    for(Loc loc2=0;loc2<Board::MAX_ARR_SIZE;loc2++)
                    {
                        if(minWinStepsMap[loc1][loc2] < NOT_SURE)
                        {
                            //very rare case, may have bug
                            //if no bug, there is a very difficult VCF that loc2 found it and marked loc2 as not-depended loc but loc1-pass didn't find a VCF
                            std::cout<<"Warning: minWinStepsMap[loc1][loc2] < NOT_SURE but vcfSteps2==0, "<<minWinStepsMap[loc1][loc2]<<" "<<vcfSteps2<<std::endl;
                            std::cout<<Location::toString(loc1, board)<<" "<<Location::toString(loc2, board)<<std::endl;
                            Board::printBoard(std::cout,board,loc1,nullptr);
                        }
                        else
                        {
                            minWinStepsMap[loc1][loc2] = SURE_NO_VCF;
                            minWinStepsMap[loc2][loc1] = SURE_NO_VCF;
                        }
                    }
                }
            }
            else if(dependValue1 == 1){
                //mark all dependvalue2 < 1 as lose
                for(Loc loc2=0;loc2<Board::MAX_ARR_SIZE;loc2++)
                {
                    if(dependMap1[loc2] < 1)
                    {
                        if(minWinStepsMap[loc1][loc2] > NOT_SURE)
                        {
                            //very rare case, may have bug
                            //if no bug, there is a very difficult VCF that loc1-pass found it and marked loc2 as not-depended loc, but loc2-pass didn't find a VCF
                            std::cout<<"Warning: minWinStepsMap[loc1][loc2] > NOT_SURE but dependMap1[loc1] == 1 and dependMap1[loc2] < 1, "<<minWinStepsMap[loc1][loc2]<<" "<<vcfSteps1<<std::endl;
                            std::cout<<Location::toString(loc1, board)<<" "<<Location::toString(loc2, board)<<std::endl;
                            Board::printBoard(std::cout,board,loc1,nullptr);
                        }
                        minWinStepsMap[loc1][loc2] = vcfSteps1;
                        minWinStepsMap[loc2][loc1] = vcfSteps1;
                    }
                }

                //the depend map may change, recalculate
                
                Board board1 = board;
                board1.playMoveAssumeLegal(loc1, board1.nextPla);
                board1.playMoveAssumeLegal(Board::PASS_LOC, board1.nextPla);
                

                std::vector<int8_t> dependMap2;
                int vcfSteps2 = 0;
                int recommendedMaxMoveStage1 = vcfSteps1 + 0;
                vcfSteps2 = calculateShortestVCFAndDependMap(board1, attackPlayer, maxMove, recommendedMaxMoveStage1, recommendedMaxMoveStage1, searchFactor, dependMap2, false);
                if(vcfSteps2 != vcfSteps1)
                {
                    //very rare case, may have bug
                    //if no bug, there is a very difficult VCF that pass-pass found it and marked loc1 as not-depended loc but loc1-pass didn't find a VCF
                    std::cout<<"Warning: find a vcf of pass-pass in "<<vcfSteps1<<" moves, but recalculate with loc1 as "<<Location::toString(loc1, board)<<" the vcf is "<<vcfSteps2<<std::endl;
                    
                    Board::printBoard(std::cout,board,loc1,nullptr);
                }
                if(vcfSteps2 > 0)
                {
                    //all locs with loc1 has vcf in vcfSteps2 moves
                    for(Loc loc2=0;loc2<Board::MAX_ARR_SIZE;loc2++)
                    {
                        if(dependMap2[loc2] < 2)
                        {
                            if(minWinStepsMap[loc1][loc2] > NOT_SURE)
                            {
                                //very rare case, may have bug
                                //if no bug, there is a very difficult VCF that loc1-pass found it and marked loc2 as not-depended loc, but loc2-pass didn't find a VCF
                                std::cout<<"Warning: minWinStepsMap[loc1][loc2] > NOT_SURE but after loc1 played dependMap2[loc2] < 2, "<<minWinStepsMap[loc1][loc2]<<" "<<vcfSteps1<<" "<<vcfSteps2<<std::endl;
                                std::cout<<Location::toString(loc1, board)<<" "<<Location::toString(loc2, board)<<std::endl;
                                Board::printBoard(std::cout,board,loc1,nullptr);
                            }
                            minWinStepsMap[loc1][loc2] = std::min(minWinStepsMap[loc1][loc2], int16_t(vcfSteps2));
                            minWinStepsMap[loc2][loc1] = std::min(minWinStepsMap[loc2][loc1], int16_t(vcfSteps2));
                        }
                    }
                }
            }
            else if(dependValue1 == 0){
                //mark all dependvalue2 < 2 as lose
                for(Loc loc2=0;loc2<Board::MAX_ARR_SIZE;loc2++)
                {
                    if(dependMap1[loc2] < 2)
                    {
                        minWinStepsMap[loc1][loc2] = vcfSteps1;
                        minWinStepsMap[loc2][loc1] = vcfSteps1;
                    }
                }
            }


        }
    }

    // If VCF found, analyze all legal positions for pruning
    // First, collect all legal positions
    std::vector<Loc> legalPositions;
    for (int y1 = 0; y1 < board.y_size; y1++) {
        for (int x1 = 0; x1 < board.x_size; x1++) {
            Loc loc1 = Location::getLoc(x1, y1, board.x_size);
            if (board.isLegal(loc1, board.nextPla)) {
                legalPositions.push_back(loc1);
            }
        }
    }

    // Sort positions by dependMap value (2, 1, 0) and then by priority
    std::sort(legalPositions.begin(), legalPositions.end(), [&](Loc a, Loc b) {
        int8_t dependA = dependMap1[a];
        int8_t dependB = dependMap1[b];

        // First sort by dependMap value (descending: 2, 1, 0)
        if (dependA != dependB) {
            return dependA > dependB;
        }

        // Then sort by priority (descending)
        int xa = Location::getX(a, board.x_size);
        int ya = Location::getY(a, board.y_size);
        int xb = Location::getX(b, board.x_size);
        int yb = Location::getY(b, board.y_size);

        double priorityA = board.getLocationPriority(xa, ya);
        double priorityB = board.getLocationPriority(xb, yb);

        return priorityA < priorityB;
        });

    // Process positions in sorted order
    int64_t loc2countTotal = 0;
    for (size_t i = 0; i < legalPositions.size(); i++) {
        Loc loc1 = legalPositions[i];
        Board board1 = board;
        board1.playMoveAssumeLegal(loc1, board1.nextPla);

        int16_t longestDefense = vcfSteps1;
        int16_t longestKnownDefense = vcfSteps1;
        //search all children
        // Collect all legal second move positions
        std::vector<Loc> nonLosingPositions2;
        for (int y2 = 0; y2 < board.y_size; y2++) {
            for (int x2 = 0; x2 < board.x_size; x2++) {
                Loc loc2 = Location::getLoc(x2, y2, board.x_size);

                // Check if position is legal and meets priority requirements
                if (board1.isLegal(loc2, board1.nextPla) &&
                    board1.getLocationPriority(x2, y2) + Board::PRIOR_EPS >= board1.firstLocPriority) {
                    assert(minWinStepsMap[loc1][loc2] > 0);
                    assert(minWinStepsMap[loc1][loc2] == minWinStepsMap[loc2][loc1]);
                    longestDefense = std::max(longestDefense, minWinStepsMap[loc1][loc2]);
                    if(minWinStepsMap[loc1][loc2]==NOT_SURE)
                        nonLosingPositions2.push_back(loc2);
                    else{
                        longestKnownDefense = std::max(longestKnownDefense, minWinStepsMap[loc1][loc2]);
                    }
                }
            }
        }
        if(longestDefense == SURE_NO_VCF)
        {
            continue;
        }
        if(longestDefense < NOT_SURE) //all loc2 has vcf
        {
            assert(longestDefense > 0); 
            VCFPrunedInfo info(loc1, true, longestDefense, false, 0.0f, 0.0f);
            results.push_back(info);
            continue;
        }

        assert(longestDefense==NOT_SURE);
        assert(longestKnownDefense<NOT_SURE);
        if(VCFCalculator_debug_print)
            std::cout << std::endl << "loc1idx " << i << " "<<Location::toString(loc1,board) << " nonLosingPositions2.size()=" << nonLosingPositions2.size() << "  ";


        bool canDefend = false;
        longestDefense = longestKnownDefense; //initially the vcf win move of pass

        // Process second move positions in sorted order
         for (size_t j = 0; j < nonLosingPositions2.size(); j++) {
             Loc loc2 = nonLosingPositions2[j];
             if(canDefend)
                 break;
                 

            // Need to test this position by actually playing it
            Board testBoard = board1;
            testBoard.playMoveAssumeLegal(loc2, testBoard.nextPla);

            // Calculate VCF for the test board
            std::vector<int8_t> testDependMap;
            int recommendedMaxMoveStage2 = longestKnownDefense + stage2RecommendedMovenumExtra;
            int testVcfSteps = calculateShortestVCFAndDependMap(testBoard, attackPlayer, maxMove, recommendedMaxMoveStage2, recommendedMaxMoveStage2, searchFactor, testDependMap, false);
            loc2countTotal += 1;
            if (testVcfSteps > 0) {
                longestDefense = std::max(longestDefense, int16_t(testVcfSteps));

                //loc1+loc2 loses
                // if dependMap2[loc3]<2, then loc1+loc2+loc3 loses
                // so loc1+loc3, loc2+loc3 loses
                if(enable_loc3_prune)
                {
                    //all locs with loc1 has vcf in vcfSteps2 moves
                    for (Loc loc3 = 0; loc3 < Board::MAX_ARR_SIZE; loc3++)
                    {
                        if (testDependMap[loc3] < 2)
                        {
                            if (minWinStepsMap[loc1][loc3] > NOT_SURE)
                            {
                                //very rare case, may have bug
                                //if no bug, there is a very difficult VCF that loc1-pass found it and marked loc2 as not-depended loc, but loc2-pass didn't find a VCF
                                std::cout << "Warning: minWinStepsMap[loc1][loc3] > NOT_SURE but after loc1+loc2 played dependMap2[loc3] < 2, " << minWinStepsMap[loc1][loc3] << " " << testVcfSteps << " " << std::endl;
                                std::cout << Location::toString(loc1, board) << " " << Location::toString(loc2, board) << " " << Location::toString(loc3, board) << std::endl;
                                Board::printBoard(std::cout, board, loc1, nullptr);
                            }
                            minWinStepsMap[loc1][loc3] = std::min(minWinStepsMap[loc1][loc3], int16_t(testVcfSteps));
                            minWinStepsMap[loc3][loc1] = std::min(minWinStepsMap[loc3][loc1], int16_t(testVcfSteps));

                            if (minWinStepsMap[loc2][loc3] > NOT_SURE)
                            {
                                //very rare case, may have bug
                                //if no bug, there is a very difficult VCF that loc1-pass found it and marked loc2 as not-depended loc, but loc2-pass didn't find a VCF
                                std::cout << "Warning: minWinStepsMap[loc2][loc3] > NOT_SURE but after loc1+loc2 played dependMap2[loc3] < 2, " << minWinStepsMap[loc2][loc3] << " " << testVcfSteps << " " << std::endl;
                                std::cout << Location::toString(loc1, board) << " " << Location::toString(loc2, board) << " " << Location::toString(loc3, board) << std::endl;
                                Board::printBoard(std::cout, board, loc2, nullptr);
                            }
                            minWinStepsMap[loc2][loc3] = std::min(minWinStepsMap[loc2][loc3], int16_t(testVcfSteps));
                            minWinStepsMap[loc3][loc2] = std::min(minWinStepsMap[loc3][loc2], int16_t(testVcfSteps));
                        }
                    }
                    
                }
            }
            else {
                canDefend = true;
                break;
            }


        }

        if (!canDefend) //all 2nd move are losing
        {
            VCFPrunedInfo info(loc1, true, longestDefense, false, 0.0f, 0.0f);
            results.push_back(info);
        }


    }


    if (VCFCalculator_debug_print)
        std::cout << "  \n" << "Loc2 vcf calculation count: " << loc2countTotal << "  \n";
    return results;



}
std::vector<VCFPrunedInfo> VCFCalculator::CalculateAllVCFDefendResults_stage1(
    const Board& board, 
    Color attackPlayer, 
    int maxMove, 
    double searchFactor) {
    
    Color defendPlayer = getOpp(attackPlayer);
    assert(board.stage == 1);
    assert(board.nextPla == defendPlayer);
    
    //check whether defend player can directly win
    {
        int defendMaxLenNow = VCFLogic::checkMaxConnectLen(board, defendPlayer);
        assert(defendMaxLenNow < 6);
        if (defendMaxLenNow - board.stage >= 4)//defend pla can win
        {
            return std::vector<VCFPrunedInfo>();
        }
    }


    Board boardWith1pass=board;
    boardWith1pass.playMoveAssumeLegal(Board::PASS_LOC,board.nextPla);

    // Calculate shortest VCF and dependency map for boardWith1pass
    std::vector<int8_t> dependMap;
    int vcfSteps = calculateShortestVCFAndDependMap(boardWith1pass, attackPlayer, maxMove, 0, 0, searchFactor, dependMap, false);
    
    std::vector<VCFPrunedInfo> results;
    
    if (vcfSteps <= 0) { //can't VCF even with 1 pass
        return results;
    }

    // If VCF found, analyze all legal positions for pruning
    for (int y = 0; y < board.y_size; y++) {
        for (int x = 0; x < board.x_size; x++) {
            Loc loc = Location::getLoc(x, y, board.x_size);
            
            // Check if position is legal and meets priority requirements
            if (!(board.isLegal(loc, board.nextPla) && 
                board.getLocationPriority(x, y) + Board::PRIOR_EPS >= board.firstLocPriority)) {
                continue;
            }
                
            int8_t dependValue = dependMap[loc];
            
            if (dependValue == 0 || dependValue == 1) {
                // Position is pruned - opponent will win by VCF
                VCFPrunedInfo info(loc, true, vcfSteps, false, 0.0f, 0.0f);
                results.push_back(info);
            }
            else if (dependValue == 2) {
                // Need to test this position by actually playing it
                Board testBoard = board;
                testBoard.playMoveAssumeLegal(loc, board.nextPla);
                
                // Calculate VCF for the test board
                std::vector<int8_t> testDependMap;
                int testVcfSteps = calculateShortestVCFAndDependMap(testBoard, attackPlayer, maxMove, vcfSteps, vcfSteps, searchFactor, testDependMap, false);
                
                if (testVcfSteps > 0) {
                    // VCF still possible after playing this move - position is pruned
                    VCFPrunedInfo info(loc, true, testVcfSteps, false, 0.0f, 0.0f);
                    results.push_back(info);
                } else {
                    // VCF not possible after playing this move - position is not pruned
                    //VCFPrunedInfo info(loc, false, 0, true, 0.0f, 0.0f);
                    //results.push_back(info);
                }
            }
            else 
                assert(false);
            
        }
    }

    
    return results;
}
std::vector<VCFPrunedInfo> VCFCalculator::CalculateAllVCFDefendResultsV2_stage1(
    const Board& board,
    Color attackPlayer,
    int maxMove,
    double searchFactor) {

    Color defendPlayer = getOpp(attackPlayer);
    assert(board.stage == 1);
    assert(board.nextPla == defendPlayer);

    //check whether defend player can directly win
    {
        int defendMaxLenNow = VCFLogic::checkMaxConnectLen(board, defendPlayer);
        assert(defendMaxLenNow < 6);
        if (defendMaxLenNow - board.stage >= 4)//defend pla can win
        {
            return std::vector<VCFPrunedInfo>();
        }
    }


    Board boardWith1pass = board;
    boardWith1pass.playMoveAssumeLegal(Board::PASS_LOC, board.nextPla);

    // Calculate shortest VCF and dependency map for boardWith1pass
    std::vector<int8_t> dependMap;
    int vcfSteps = calculateShortestVCFAndDependMap(boardWith1pass, attackPlayer, maxMove, 0, 0, searchFactor, dependMap, false);

    std::vector<VCFPrunedInfo> results;

    if (vcfSteps <= 0) { //can't VCF even with 1 pass
        return results;
    }

    int16_t minWinStepsMap[Board::MAX_ARR_SIZE];
    for(int i=0;i<Board::MAX_ARR_SIZE;i++)
    {
        minWinStepsMap[i] = 32767;
    }
    //dependmap < 2 then minstep is vcfSteps
    for (int y = 0; y < board.y_size; y++) {
        for (int x = 0; x < board.x_size; x++) {
            Loc loc = Location::getLoc(x, y, board.x_size);
            int8_t dependValue = dependMap[loc];
            if (dependValue < 2) {
                minWinStepsMap[loc] = std::min(minWinStepsMap[loc], int16_t(vcfSteps));
            }
        }
    }

    // If VCF found, analyze all legal positions for pruning
    for (int y = 0; y < board.y_size; y++) {
        for (int x = 0; x < board.x_size; x++) {
            Loc loc = Location::getLoc(x, y, board.x_size);

            // Check if position is legal and meets priority requirements
            if (!(board.isLegal(loc, board.nextPla) &&
                board.getLocationPriority(x, y) + Board::PRIOR_EPS >= board.firstLocPriority)) {
                continue;
            }


            if (minWinStepsMap[loc] < 32767) {
                //if (dependMap[loc] == 2)
                //    std::cout << 2;
                continue;
                // Position is pruned - opponent will win by VCF
                //VCFPrunedInfo info(loc, true, minWinStepsMap[loc], false, 0.0f, 0.0f);
                //results.push_back(info);
            }
            else {
                assert(dependMap[loc] == 2);
                // Need to test this position by actually playing it
                Board testBoard = board;
                testBoard.playMoveAssumeLegal(loc, board.nextPla);

                // Calculate VCF for the test board
                std::vector<int8_t> testDependMap;
                int testVcfSteps = calculateShortestVCFAndDependMap(testBoard, attackPlayer, maxMove, vcfSteps, vcfSteps, searchFactor, testDependMap, false);

                if (testVcfSteps > 0) {
                    // VCF still possible after playing this move - position is pruned
                    //VCFPrunedInfo info(loc, true, testVcfSteps, false, 0.0f, 0.0f);
                    //results.push_back(info);

                    //update minWinStepsMap
                    minWinStepsMap[loc] = std::min(minWinStepsMap[loc], int16_t(testVcfSteps));
                    for(int y1=0;y1<board.y_size;y1++){
                        for(int x1=0;x1<board.x_size;x1++){
                            Loc loc2 = Location::getLoc(x1, y1, board.x_size);
                            // Check if position is legal and meets priority requirements
                            if (!(board.isLegal(loc2, board.nextPla) &&
                                board.getLocationPriority(x1, y1) + Board::PRIOR_EPS >= board.firstLocPriority)) {
                                continue;
                            }
                            if(loc2==loc)
                                continue;

                            //this means, even give the defend player one more stone at loc1, loc2 is also losing
                            if(testDependMap[loc2] < 2){
                                minWinStepsMap[loc2] = std::min(minWinStepsMap[loc2], int16_t(testVcfSteps));
                            }
                        }
                    }
                }
                else {
                    // VCF not possible after playing this move - position is not pruned
                    //VCFPrunedInfo info(loc, false, 0, true, 0.0f, 0.0f);
                    //results.push_back(info);
                }
            }

        }
    }

    // If VCF found, analyze all legal positions for pruning
    for (int y = 0; y < board.y_size; y++) {
        for (int x = 0; x < board.x_size; x++) {
            Loc loc = Location::getLoc(x, y, board.x_size);

            // Check if position is legal and meets priority requirements
            if (!(board.isLegal(loc, board.nextPla) &&
                board.getLocationPriority(x, y) + Board::PRIOR_EPS >= board.firstLocPriority)) {
                continue;
            }
            if(minWinStepsMap[loc] < 32767)
            {
                assert(minWinStepsMap[loc]>board.movenum&&minWinStepsMap[loc]<board.x_size*board.y_size+100);
                // Position is pruned - opponent will win by VCF
                VCFPrunedInfo info(loc, true, minWinStepsMap[loc], false, 0.0f, 0.0f);
                results.push_back(info);
            }
        }
    }

    return results;
}
int VCFCalculator::calculateShortestVCFAndDependMap(
    const Board& board,
    Color attackPlayer,
    int initialMaxMove,
    int minMaxMove,
    int recommendedMaxMove,
    double searchFactor,
    std::vector<int8_t>& dependMap,
    bool noOptimize) {
    if (VCFCalculator_debug_print)
    {
        std::cout << 1;
        std::cout.flush();
    }
    dependMap.clear();

    assert(board.nextPla == attackPlayer);
    assert(board.stage == 0);
    //Board::printBoard(std::cout, board, Board::NULL_LOC, nullptr);
    //std::cout.flush();
    
    //check whether attackPlayer has fours
    {
      std::vector<Loc> defenseFourLocs = VCFLogic::getAllDefenseFourLocs(board,attackPlayer);

      if(defenseFourLocs.size()>0){ //if attackPlayer has fours, then attackPlayer will win in 2 moves
        //mark all defenseFourLocs
        dependMap = std::vector<int8_t>(Board::MAX_ARR_SIZE, 0);
        for(Loc loc:defenseFourLocs){
          dependMap[loc] = 2;
        }
        return board.movenum + 2;
      }
    }


    // Initialize with the current board state
    Rules rules; // Use default rules, it will be overrided during the VCF calculating
    nnueHistory1.clear(board, board.nextPla, rules);

    if (initialMaxMove != 0)
    {
        if (minMaxMove > initialMaxMove)
            minMaxMove = initialMaxMove;
        if (recommendedMaxMove > initialMaxMove)
            recommendedMaxMove = initialMaxMove;
    }
    
    int currentMaxMove = initialMaxMove;
    if (minMaxMove < board.movenum + 6)
        minMaxMove = board.movenum + 6;
    if (recommendedMaxMove > 0 && recommendedMaxMove < minMaxMove)
        recommendedMaxMove = minMaxMove;

    bool isTestingRecommendedMaxMove = false;
    if (recommendedMaxMove > 0 && recommendedMaxMove >= minMaxMove && recommendedMaxMove < initialMaxMove) //try recommendedMaxMove first
    {
        isTestingRecommendedMaxMove = true;
        currentMaxMove = recommendedMaxMove;
    }

    int lastSuccessfulVCFSteps = -1;
    std::vector<int8_t> lastSuccessfulDependMap;
    
    while (true) {
        if (currentMaxMove < board.movenum + 6 || currentMaxMove < minMaxMove)
            break;
        // Modify the rules to set new maxMoves
        Rules rules=Rules();
        rules.maxMoves = currentMaxMove;
        if(attackPlayer==C_BLACK)
          rules.VCNRule=Rules::VCNRULE_VC4_B;
        else
          rules.VCNRule=Rules::VCNRULE_VC4_W;

        nnueHistory1.rules = rules;
        
        // Create MCTS search instance
        MCTSsearch mcts(cacheTable, &nnueHistory1, attackPlayer);
        
        // Perform VCF search
        Loc bestMove;
        bool canWin = mcts.vcfSearchAutoStop(&nnueHistory1, attackPlayer, searchFactor);
        //std::cout << currentMaxMove <<" "<<canWin<< std::endl;
        
        if (canWin) {
            isTestingRecommendedMaxMove = false;
            // Calculate dependency map for successful VCF
            int winMoveNum = mcts.rootNode->stepsToWin; 
            assert(winMoveNum >= board.movenum + 6 && (currentMaxMove==0||winMoveNum<=currentMaxMove));
            assert((winMoveNum-board.movenum)%4 == 2);
            std::vector<int8_t> currentDependMap = mcts.calculateDefenseDependencyMap();
            
            // Store the successful result
            lastSuccessfulVCFSteps = winMoveNum;
            lastSuccessfulDependMap = currentDependMap;
            
            // Try with fewer moves (reduce by 4 as specified)
            currentMaxMove = winMoveNum - 4;
            if(noOptimize)
              break;
        } else {
            if (isTestingRecommendedMaxMove)
            {
                //recommendedMaxMove failed, at least recommendedMaxMove+4
                isTestingRecommendedMaxMove = false;
                currentMaxMove = initialMaxMove;
                minMaxMove = recommendedMaxMove + 4;
            }
            else
                break;// VCF failed, stop the search
        }
        
    }
    
    // Return the dependency map from the last successful VCF
    if (lastSuccessfulVCFSteps != -1) {
        dependMap = lastSuccessfulDependMap;
        return lastSuccessfulVCFSteps;
    } else {
        // No successful VCF found
        dependMap.clear();
        return -1;
    }
}
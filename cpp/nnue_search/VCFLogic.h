#pragma once
// VCF Logic header file
#include "../game/board.h"
#include <vector>

// Forward declaration
struct VCFPrunedInfo;

namespace VCFLogic {
  //Check if player has two four-in-a-row threats
  int checkTwoFourThreats(const Board& board, Player pla);
  
  //Check maximum consecutive length for a player
  int checkMaxConnectLen(const Board& board, Player pla);
  
  //Get all possible VCF attack or defense locations
  //if board.nextPla==attackPla, return all attack locations
  //else return all defense locations
  
  std::vector<Loc> getAllVCFAttackOrDefenseLocs(const Board& board, Player attackPla, Color& winner, int& gameEndMovenum);
  std::vector<Loc> getAllVCFAttackOrDefenseLocsWithCache(const Board& board, Player attackPla, Color& winner, int& gameEndMovenum, uint8_t (&stoneTupleCache)[4][Board::MAX_ARR_SIZE]);



  //Get all possible defense locations of attackPla's four, empty if no four
  std::vector<Loc> getAllDefenseFourLocs(const Board& board, Player attackPla);

  //mark all locations may influence the vcf
  //dependMap==2: one defendPla's stone will make the vcf failed. Block one of attackPla's four or create a defendPla's four
  //dependMap==1: two defendPla's stone will make the vcf failed. create a defendPla's four with 2 stones
  void markAllDefenseDependedLocs(const Board& board, Player attackPla,std::vector<int8_t>& dependMap);
  
  //One-layer brute force search for stage==1 and nextPla==attackPla
  //Check if any VCF attack move leads to immediate win
  Loc findImmediateWinInVCFAttack(const Board& board, Player attackPla);
  Loc findImmediateWinInVCFAttackLayer2(const Board& board, Player attackPla);


  void printBoardWithDependencyMap(const Board& board, const std::vector<int8_t>& dependMap);
  void printBoardWithPruneInfo(const Board& board, const std::vector<VCFPrunedInfo>& pruneInfo);
}

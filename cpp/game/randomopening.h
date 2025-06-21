#pragma once
#include "../search/asyncbot.h"

class Search;
class OtherGameProperties;
namespace RandomOpening {
  void initializeSpecialOpening(
    Board& board,
    BoardHistory& hist,
    Rules& rules,
    Player& nextPlayer,
    OtherGameProperties& otherGameProps,
    Rand& gameRand);
  void initializeBalancedRandomOpening(
    Search* botB,
    Search* botW,
    Board& board,
    BoardHistory& hist,
    Player& nextPlayer,
    Rand& gameRand,
    bool forSelfplay);

}

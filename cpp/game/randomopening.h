#pragma once
#include "../search/asyncbot.h"

class Search;

namespace RandomOpening {
  //disabled
  void initializeRandomOpening(
    Board& board,
    BoardHistory& hist,
    Player& nextPlayer,
    Rand& gameRand,
    bool forSelfplay);

}

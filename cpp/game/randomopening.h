#pragma once
#include "../search/asyncbot.h"
#include "../program/playsettings.h"

class Search;

namespace RandomOpening {
  //disabled
  void initializeRandomOpening(
    Search* botB,
    Search* botW,
    Board& board,
    BoardHistory& hist,
    Player& nextPlayer,
    Rand& gameRand,
    const PlaySettings& playSettings);

}

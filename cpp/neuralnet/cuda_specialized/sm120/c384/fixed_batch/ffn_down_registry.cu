#include "ffn_down.h"

namespace C384ExactFfnDownAot {

PreparedSelection prepareSelection(
  const RuntimeShape& shape,
  const char* requestedId
) {
  std::size_t count = 0;
  const Tactic* tactics = generatedTactics(count);
  return prepareSelectionFromRegistry(shape,requestedId,tactics,count);
}

}  // namespace C384ExactFfnDownAot

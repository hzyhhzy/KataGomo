#include "../neuralnet/c384_exact_ffn_down_aot.h"

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

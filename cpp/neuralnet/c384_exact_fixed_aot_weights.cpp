#include "../neuralnet/c384_exact_fixed_aot_weights.h"

#include <stdexcept>

namespace C384ExactFixedAot {

namespace {

void requireIndex(bool condition, const char* message) {
  if(!condition)
    throw std::out_of_range(message);
}

}  // namespace

std::size_t packedQkvWeightIndex(
  int inputChannel,
  int plane,
  int outputChannel
) {
  requireIndex(inputChannel >= 0 && inputChannel < kChannels,
    "QKV input channel out of range");
  requireIndex(plane >= 0 && plane < kQkvPlanes,"QKV plane out of range");
  requireIndex(outputChannel >= 0 && outputChannel < kChannels,
    "QKV output channel out of range");
  return static_cast<std::size_t>(inputChannel) * kQkvPackedColumns +
    plane * kChannels + outputChannel;
}

std::vector<float> packQkvWeights(
  const std::vector<float>& q,
  const std::vector<float>& k,
  const std::vector<float>& v
) {
  const std::size_t matrixElements =
    static_cast<std::size_t>(kChannels) * kChannels;
  if(q.size() != matrixElements || k.size() != matrixElements ||
     v.size() != matrixElements)
    throw std::invalid_argument("C384 Q/K/V weight matrix size mismatch");
  std::vector<float> packed(
    static_cast<std::size_t>(kChannels) * kQkvPackedColumns);
  for(int input = 0; input < kChannels; input++) {
    for(int output = 0; output < kChannels; output++) {
      const std::size_t source =
        static_cast<std::size_t>(input) * kChannels + output;
      packed[packedQkvWeightIndex(input,0,output)] = q[source];
      packed[packedQkvWeightIndex(input,1,output)] = k[source];
      packed[packedQkvWeightIndex(input,2,output)] = v[source];
    }
  }
  return packed;
}

std::size_t packedDualFfnWeightIndex(
  int inputChannel,
  bool gate,
  int outputChannel
) {
  requireIndex(inputChannel >= 0 && inputChannel < kChannels,
    "dual-FFN input channel out of range");
  requireIndex(outputChannel >= 0 && outputChannel < kFfnChannels,
    "dual-FFN output channel out of range");
  const int block = outputChannel / kDualFfnPairColumns;
  const int lane = outputChannel % kDualFfnPairColumns;
  return static_cast<std::size_t>(inputChannel) * kDualFfnPackedColumns +
    block * (2 * kDualFfnPairColumns) +
    (gate ? kDualFfnPairColumns : 0) + lane;
}

std::vector<float> packDualFfnWeights(
  const std::vector<float>& up,
  const std::vector<float>& gate
) {
  const std::size_t matrixElements =
    static_cast<std::size_t>(kChannels) * kFfnChannels;
  if(up.size() != matrixElements || gate.size() != matrixElements)
    throw std::invalid_argument("C384 F1024 up/gate weight matrix size mismatch");
  std::vector<float> packed(
    static_cast<std::size_t>(kChannels) * kDualFfnPackedColumns);
  for(int input = 0; input < kChannels; input++) {
    for(int output = 0; output < kFfnChannels; output++) {
      const std::size_t source =
        static_cast<std::size_t>(input) * kFfnChannels + output;
      packed[packedDualFfnWeightIndex(input,false,output)] = up[source];
      packed[packedDualFfnWeightIndex(input,true,output)] = gate[source];
    }
  }
  return packed;
}

}  // namespace C384ExactFixedAot

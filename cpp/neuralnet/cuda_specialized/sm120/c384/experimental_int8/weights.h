#ifndef KATAGO_C384_EXPERIMENTAL_INT8_WEIGHTS_H_
#define KATAGO_C384_EXPERIMENTAL_INT8_WEIGHTS_H_

#include "policy.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace C384Int8Experiment {

struct PackedWeights {
  std::vector<int8_t> values;
  float scale = 0.0f;
  int inputChannels = 0;
  int outputChannels = 0;
};

inline int8_t quantizeWeight(float value, float scale) {
  if(!std::isfinite(value) || !(scale > 0.0f) || !std::isfinite(scale))
    throw std::invalid_argument("C384 INT8 weight quantization is non-finite");
  const float scaled = value / scale;
  if(scaled <= -127.0f) return int8_t(-127);
  if(scaled >= 127.0f) return int8_t(127);
  const float lowerFloat = std::floor(scaled);
  const int lower = static_cast<int>(lowerFloat);
  const float fraction = scaled - lowerFloat;
  int rounded = fraction < 0.5f ? lower :
    (fraction > 0.5f ? lower + 1 : ((lower & 1) == 0 ? lower : lower + 1));
  rounded = std::max(-127,std::min(127,rounded));
  return static_cast<int8_t>(rounded);
}

inline float commonWeightScale(
  const std::vector<const std::vector<float>*>& matrices
) {
  if(matrices.empty())
    throw std::invalid_argument("C384 INT8 matrix group is empty");
  float maxAbs = 0.0f;
  for(const std::vector<float>* matrix: matrices) {
    if(matrix == nullptr || matrix->empty())
      throw std::invalid_argument("C384 INT8 matrix is empty");
    for(float value: *matrix) {
      if(!std::isfinite(value))
        throw std::invalid_argument("C384 INT8 matrix contains NaN or infinity");
      maxAbs = std::max(maxAbs,std::fabs(value));
    }
  }
  const float scale = std::max(
    maxAbs / 127.0f,std::numeric_limits<float>::min());
  if(!(scale > 0.0f) || !std::isfinite(scale))
    throw std::invalid_argument("C384 INT8 matrix scale is invalid");
  return scale;
}

// Source is the backend MatMul host layout [K,N]. CUTLASS B is column-major,
// represented by output-major K-contiguous bytes [N,K].
inline PackedWeights packMatrix(
  const std::vector<float>& source,
  int inputChannels,
  int outputChannels
) {
  if(inputChannels <= 0 || outputChannels <= 0 ||
     source.size() != std::size_t(inputChannels) * outputChannels)
    throw std::invalid_argument("C384 INT8 matrix shape mismatch");
  PackedWeights result;
  result.scale = commonWeightScale({&source});
  result.inputChannels = inputChannels;
  result.outputChannels = outputChannels;
  result.values.resize(source.size());
  for(int k = 0; k < inputChannels; k++)
    for(int n = 0; n < outputChannels; n++)
      result.values[std::size_t(n) * inputChannels + k] =
        quantizeWeight(source[std::size_t(k) * outputChannels + n],result.scale);
  return result;
}

inline PackedWeights packProjection(
  const std::vector<float>& q,
  const std::vector<float>& k,
  const std::vector<float>& v,
  EngineMode mode
) {
  constexpr int channels = 384;
  const std::size_t matrixElements = std::size_t(channels) * channels;
  if(q.size() != matrixElements || k.size() != matrixElements ||
     v.size() != matrixElements ||
     (mode != EngineMode::Conservative && mode != EngineMode::Aggressive))
    throw std::invalid_argument("C384 INT8 QKV projection contract mismatch");
  const bool aggressive = mode == EngineMode::Aggressive;
  PackedWeights result;
  result.scale = aggressive ? commonWeightScale({&q,&k,&v}) :
    commonWeightScale({&q,&k});
  result.inputChannels = channels;
  result.outputChannels = aggressive ? 3 * channels : 2 * channels;
  result.values.resize(std::size_t(result.outputChannels) * channels);
  const std::vector<float>* matrices[3] = {&q,&k,&v};
  const int planes = aggressive ? 3 : 2;
  for(int plane = 0; plane < planes; plane++)
    for(int input = 0; input < channels; input++)
      for(int output = 0; output < channels; output++) {
        const int n = plane * channels + output;
        result.values[std::size_t(n) * channels + input] = quantizeWeight(
          (*matrices[plane])[std::size_t(input) * channels + output],
          result.scale);
      }
  return result;
}

}  // namespace C384Int8Experiment

#endif  // KATAGO_C384_EXPERIMENTAL_INT8_WEIGHTS_H_

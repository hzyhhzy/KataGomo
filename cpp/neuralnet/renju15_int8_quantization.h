#ifndef KATAGO_RENJU15_INT8_QUANTIZATION_H_
#define KATAGO_RENJU15_INT8_QUANTIZATION_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace Renju15Int8Quantization {

inline int8_t quantizeWeight(float value, float scale) {
  if(!std::isfinite(value) || !(scale > 0.0f) || !std::isfinite(scale))
    throw std::invalid_argument(
      "INT8 weight quantization received a non-finite value");
  const float scaled = value / scale;
  if(scaled <= -127.0f)
    return static_cast<int8_t>(-127);
  if(scaled >= 127.0f)
    return static_cast<int8_t>(127);
  const float lowerFloat = std::floor(scaled);
  const int lower = static_cast<int>(lowerFloat);
  const float fraction = scaled - lowerFloat;
  int quantized;
  if(fraction < 0.5f)
    quantized = lower;
  else if(fraction > 0.5f)
    quantized = lower + 1;
  else
    quantized = (lower % 2 == 0) ? lower : lower + 1;
  quantized = std::max(-127,std::min(127,quantized));
  return static_cast<int8_t>(quantized);
}

inline void validateContract() {
  struct TestCase { float input; int expected; };
  constexpr TestCase cases[] = {
    {-1000.0f,-127},{-127.5f,-127},{-2.5f,-2},{-1.5f,-2},
    {-0.5f,0},{0.5f,0},{1.5f,2},{2.5f,2},{126.5f,126},
    {127.5f,127},{1000.0f,127},
  };
  for(const TestCase& test: cases) {
    const int actual = static_cast<int>(quantizeWeight(test.input,1.0f));
    if(actual != test.expected || actual == -128)
      throw std::runtime_error(
        "deterministic INT8 round-ties-even self-test failed");
  }
}

inline float perMatrixScale(const std::vector<float>& weights) {
  if(weights.empty())
    throw std::invalid_argument("INT8 matrix must not be empty");
  float maxAbs = 0.0f;
  for(float value: weights) {
    if(!std::isfinite(value))
      throw std::invalid_argument("INT8 matrix contains NaN or infinity");
    maxAbs = std::max(maxAbs,std::fabs(value));
  }
  const float scale = std::max(
    maxAbs / 127.0f,std::numeric_limits<float>::min());
  if(!(scale > 0.0f) || !std::isfinite(scale))
    throw std::invalid_argument("INT8 matrix produced an invalid scale");
  return scale;
}

inline std::vector<int8_t> quantizeAndPackMatrix(
  const std::vector<float>& weights,
  int inputChannels,
  int outputChannels,
  float scale
) {
  if(inputChannels <= 0 || outputChannels <= 0 ||
     weights.size() != (size_t)inputChannels * outputChannels)
    throw std::invalid_argument("INT8 matrix weight count mismatch");
  validateContract();
  std::vector<int8_t> packed((size_t)inputChannels * outputChannels);
  for(int k = 0; k < inputChannels; k++) {
    for(int n = 0; n < outputChannels; n++) {
      packed[(size_t)n * inputChannels + k] = quantizeWeight(
        weights[(size_t)k * outputChannels + n],scale);
    }
  }
  return packed;
}

} // namespace Renju15Int8Quantization

#endif

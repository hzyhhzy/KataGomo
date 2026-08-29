#include "kernel.h"

#include "../../core/global.h"

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using std::string;
using std::vector;

namespace CpuPtq {
namespace {

constexpr int S = BOARD_AREA;
constexpr int HEAD_DIM = 32;
constexpr int PROJECTION_MR = 14;
constexpr float RMS_EPSILON = 1.0e-6f;
constexpr float ATTENTION_SCALE = 0.1767766952966368811f;

#if defined(__clang__)
#define CPU_PTQ_UNROLL_FULL _Pragma("clang loop unroll(full)")
#elif defined(__GNUC__)
#define CPU_PTQ_UNROLL_FULL _Pragma("GCC unroll 16")
#endif

[[noreturn]] void fail(const string& message) {
  throw StringError("CPU-PTQ Ice Lake kernel: " + message);
}

size_t roundUp(size_t value, size_t multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

inline __m512 exp512_ps(__m512 x) {
#ifdef CPU_PTQ_EXACT_MATH
  alignas(64) float lanes[16];
  _mm512_store_ps(lanes,x);
  for(float& lane: lanes)
    lane = std::exp(lane);
  return _mm512_load_ps(lanes);
#else
  const __m512 maximum = _mm512_set1_ps(88.3762626647949f);
  const __m512 minimum = _mm512_set1_ps(-88.3762626647949f);
  x = _mm512_min_ps(x,maximum);
  x = _mm512_max_ps(x,minimum);

  __m512 fx = _mm512_fmadd_ps(
    x,_mm512_set1_ps(1.44269504088896341f),_mm512_set1_ps(0.5f));
  fx = _mm512_floor_ps(fx);

  x = _mm512_fnmadd_ps(fx,_mm512_set1_ps(0.693359375f),x);
  x = _mm512_fnmadd_ps(fx,_mm512_set1_ps(-2.12194440e-4f),x);
  const __m512 z = _mm512_mul_ps(x,x);

  __m512 y = _mm512_set1_ps(4.1665795894e-2f);
  y = _mm512_fmadd_ps(y,x,_mm512_set1_ps(1.6666665459e-1f));
  y = _mm512_fmadd_ps(y,x,_mm512_set1_ps(5.0000001201e-1f));
  y = _mm512_fmadd_ps(y,z,x);
  y = _mm512_add_ps(y,_mm512_set1_ps(1.0f));

  __m512i exponent = _mm512_cvttps_epi32(fx);
  exponent = _mm512_add_epi32(exponent,_mm512_set1_epi32(127));
  exponent = _mm512_slli_epi32(exponent,23);
  return _mm512_mul_ps(y,_mm512_castsi512_ps(exponent));
#endif
}

inline __m512 silu512_ps(__m512 value) {
  const __m512 denominator = _mm512_add_ps(
    _mm512_set1_ps(1.0f),exp512_ps(_mm512_sub_ps(_mm512_setzero_ps(),value)));
  return _mm512_div_ps(value,denominator);
}

void siluInPlace(float* values, size_t count) {
  size_t index = 0;
  const size_t vectorizedCount = count / 16 * 16;
  for(; index < vectorizedCount; index += 16) {
    _mm512_storeu_ps(values + index,silu512_ps(_mm512_loadu_ps(values + index)));
  }
  for(; index < count; index++)
    values[index] = values[index] / (1.0f + std::exp(-values[index]));
}

float dot(const float* a, const float* b, size_t count) {
  __m512 sum = _mm512_setzero_ps();
  size_t index = 0;
  const size_t vectorizedCount = count / 16 * 16;
  for(; index < vectorizedCount; index += 16)
    sum = _mm512_fmadd_ps(
      _mm512_loadu_ps(a + index),_mm512_loadu_ps(b + index),sum);
  float scalar = _mm512_reduce_add_ps(sum);
  for(; index < count; index++)
    scalar += a[index] * b[index];
  return scalar;
}

vector<float> outputMajorToInputMajor(
  const vector<float>& source,
  size_t outputs,
  size_t inputs
) {
  if(source.size() != outputs * inputs)
    fail("malformed FP32 matrix");
  vector<float> result(inputs * outputs);
  for(size_t output = 0; output < outputs; output++)
    for(size_t input = 0; input < inputs; input++)
      result[input * outputs + output] = source[output * inputs + input];
  return result;
}

void linearInputMajor(
  const float* input,
  size_t rows,
  size_t inputs,
  const float* weights,
  size_t outputs,
  const float* bias,
  float* output
) {
  if(outputs % 16 != 0)
    fail("vector FP32 linear output is not a multiple of 16");
  for(size_t row = 0; row < rows; row++) {
    float* destination = output + row * outputs;
    for(size_t column = 0; column < outputs; column += 16) {
      __m512 value = bias == nullptr ?
        _mm512_setzero_ps() : _mm512_loadu_ps(bias + column);
      for(size_t inputChannel = 0; inputChannel < inputs; inputChannel++) {
        value = _mm512_fmadd_ps(
          _mm512_set1_ps(input[row * inputs + inputChannel]),
          _mm512_loadu_ps(weights + inputChannel * outputs + column),
          value);
      }
      _mm512_storeu_ps(destination + column,value);
    }
  }
}

struct PackedS8Matrix {
  size_t inputs = 0;
  size_t inputsPadded = 0;
  size_t outputs = 0;
  size_t outputsPadded = 0;
  vector<int8_t> packed;
  vector<float> scales;
  vector<int32_t> sums;
};

PackedS8Matrix packOutputMajor(
  const int8_t* codes,
  const float* scales,
  size_t outputs,
  size_t inputs
) {
  PackedS8Matrix result;
  result.inputs = inputs;
  result.inputsPadded = roundUp(inputs,4);
  result.outputs = outputs;
  result.outputsPadded = roundUp(outputs,16);
  result.scales.assign(result.outputsPadded,0.0f);
  result.sums.assign(result.outputsPadded,0);
  std::copy(scales,scales + outputs,result.scales.begin());
  const size_t nBlocks = result.outputsPadded / 16;
  const size_t kGroups = result.inputsPadded / 4;
  result.packed.assign(nBlocks * kGroups * 64,0);
  for(size_t output = 0; output < outputs; output++) {
    int32_t sum = 0;
    const size_t nBlock = output / 16;
    const size_t lane = output % 16;
    for(size_t input = 0; input < inputs; input++) {
      const int8_t code = codes[output * inputs + input];
      sum += static_cast<int32_t>(code);
      const size_t kGroup = input / 4;
      const size_t within = input % 4;
      result.packed[(nBlock * kGroups + kGroup) * 64 + lane * 4 + within] = code;
    }
    result.sums[output] = sum;
  }
  return result;
}

PackedS8Matrix allocateRuntimePacked(size_t outputs, size_t inputs) {
  PackedS8Matrix result;
  result.inputs = inputs;
  result.inputsPadded = roundUp(inputs,4);
  result.outputs = outputs;
  result.outputsPadded = roundUp(outputs,16);
  result.packed.assign(result.outputsPadded * result.inputsPadded,0);
  result.scales.assign(result.outputsPadded,0.0f);
  result.sums.assign(result.outputsPadded,0);
  return result;
}

PackedS8Matrix packTensor(const Tensor& tensor) {
  if(tensor.kind != TensorKind::S8PerOutput || tensor.shape.size() != 2)
    fail("attempted to pack a non-S8 matrix");
  return packOutputMajor(
    tensor.codes.data(),tensor.scales.data(),tensor.shape[0],tensor.shape[1]);
}

PackedS8Matrix concatenateAndPack(
  const vector<const Tensor*>& matrices,
  size_t inputs
) {
  size_t outputs = 0;
  for(const Tensor* tensor: matrices) {
    if(
      tensor == nullptr || tensor->kind != TensorKind::S8PerOutput ||
      tensor->shape.size() != 2 || tensor->shape[1] != inputs
    )
      fail("cannot concatenate malformed S8 matrices");
    outputs += tensor->shape[0];
  }
  vector<int8_t> codes(outputs * inputs);
  vector<float> scales(outputs);
  size_t offset = 0;
  for(const Tensor* tensor: matrices) {
    const size_t count = tensor->shape[0] * inputs;
    std::copy(tensor->codes.begin(),tensor->codes.end(),codes.begin() + offset * inputs);
    std::copy(tensor->scales.begin(),tensor->scales.end(),scales.begin() + offset);
    offset += tensor->shape[0];
    (void)count;
  }
  return packOutputMajor(codes.data(),scales.data(),outputs,inputs);
}

template<bool MaxAbsAlreadyComputed>
void quantizeRowsU8(
  const float* input,
  size_t rows,
  size_t channels,
  size_t stride,
  uint8_t* output,
  float* scales
) {
  const __m512 absMask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
  const __m512i zeroPoint = _mm512_set1_epi32(128);
  const __m512i minimum = _mm512_set1_epi32(-127);
  const __m512i maximum = _mm512_set1_epi32(127);
  for(size_t row = 0; row < rows; row++) {
    const float* source = input + row * channels;
    uint8_t* destination = output + row * stride;
    float maxAbs;
    size_t channel = 0;
    if(MaxAbsAlreadyComputed)
      maxAbs = scales[row];
    else {
      __m512 maximumVector = _mm512_setzero_ps();
      for(; channel + 16 <= channels; channel += 16) {
        maximumVector = _mm512_max_ps(
          maximumVector,
          _mm512_and_ps(_mm512_loadu_ps(source + channel),absMask));
      }
      maxAbs = _mm512_reduce_max_ps(maximumVector);
      for(size_t tail = channel; tail < channels; tail++)
        maxAbs = std::max(maxAbs,std::abs(source[tail]));
    }
    if(!(maxAbs > 0.0f)) {
      scales[row] = 1.0f;
      std::fill(destination,destination + stride,static_cast<uint8_t>(128));
      continue;
    }
    const float scale = maxAbs / 127.0f;
    scales[row] = scale;
    const __m512 inverse = _mm512_set1_ps(127.0f / maxAbs);
    channel = 0;
    for(; channel + 16 <= channels; channel += 16) {
      __m512i codes = _mm512_cvtps_epi32(
        _mm512_mul_ps(_mm512_loadu_ps(source + channel),inverse));
      codes = _mm512_min_epi32(maximum,_mm512_max_epi32(minimum,codes));
      codes = _mm512_add_epi32(codes,zeroPoint);
      _mm_storeu_si128(
        reinterpret_cast<__m128i*>(destination + channel),
        _mm512_cvtusepi32_epi8(codes));
    }
    for(; channel < channels; channel++) {
      float scaled = source[channel] / scale;
      scaled = std::max(-127.0f,std::min(127.0f,scaled));
      destination[channel] = static_cast<uint8_t>(
        static_cast<int>(std::nearbyint(scaled)) + 128);
    }
    std::fill(destination + channels,destination + stride,static_cast<uint8_t>(128));
  }
}

inline void storePacked16Codes(
  PackedS8Matrix& destination,
  size_t output,
  size_t inputBase,
  __m128i codes
) {
  const size_t nBlock = output / 16;
  const size_t lane = output % 16;
  const size_t kGroups = destination.inputsPadded / 4;
  const size_t firstKGroup = inputBase / 4;
  const uint32_t group0 = static_cast<uint32_t>(_mm_extract_epi32(codes,0));
  const uint32_t group1 = static_cast<uint32_t>(_mm_extract_epi32(codes,1));
  const uint32_t group2 = static_cast<uint32_t>(_mm_extract_epi32(codes,2));
  const uint32_t group3 = static_cast<uint32_t>(_mm_extract_epi32(codes,3));
  std::memcpy(
    destination.packed.data() +
      (nBlock * kGroups + firstKGroup + 0) * 64 + lane * 4,
    &group0,sizeof(group0));
  std::memcpy(
    destination.packed.data() +
      (nBlock * kGroups + firstKGroup + 1) * 64 + lane * 4,
    &group1,sizeof(group1));
  std::memcpy(
    destination.packed.data() +
      (nBlock * kGroups + firstKGroup + 2) * 64 + lane * 4,
    &group2,sizeof(group2));
  std::memcpy(
    destination.packed.data() +
      (nBlock * kGroups + firstKGroup + 3) * 64 + lane * 4,
    &group3,sizeof(group3));
}

void quantizeAttentionKeysPacked(
  const float* input,
  PackedS8Matrix& destination
) {
  if(destination.outputs != S || destination.inputs != HEAD_DIM)
    fail("invalid packed attention-key geometry");
  const __m512 absMask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
  const __m512i minimum = _mm512_set1_epi32(-127);
  const __m512i maximum = _mm512_set1_epi32(127);
  for(size_t row = 0; row < S; row++) {
    const float* source = input + row * HEAD_DIM;
    const __m512 value0 = _mm512_loadu_ps(source);
    const __m512 value1 = _mm512_loadu_ps(source + 16);
    const float maxAbs = _mm512_reduce_max_ps(_mm512_max_ps(
      _mm512_and_ps(value0,absMask),_mm512_and_ps(value1,absMask)));
    const float scale = maxAbs > 0.0f ? maxAbs / 127.0f : 1.0f;
    destination.scales[row] = scale;
    const __m512 inverse = _mm512_set1_ps(maxAbs > 0.0f ? 127.0f / maxAbs : 0.0f);
    __m512i codes0 = _mm512_cvtps_epi32(_mm512_mul_ps(value0,inverse));
    __m512i codes1 = _mm512_cvtps_epi32(_mm512_mul_ps(value1,inverse));
    codes0 = _mm512_min_epi32(maximum,_mm512_max_epi32(minimum,codes0));
    codes1 = _mm512_min_epi32(maximum,_mm512_max_epi32(minimum,codes1));
    destination.sums[row] =
      _mm512_reduce_add_epi32(codes0) + _mm512_reduce_add_epi32(codes1);
    storePacked16Codes(
      destination,row,0,_mm512_cvtsepi32_epi8(codes0));
    storePacked16Codes(
      destination,row,16,_mm512_cvtsepi32_epi8(codes1));
  }
}

void quantizeAttentionValuesPacked(
  const float* channelMajorInput,
  PackedS8Matrix& destination
) {
  if(destination.outputs != HEAD_DIM || destination.inputs != S)
    fail("invalid packed attention-value geometry");
  const __m512 absMask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
  const __m512i minimum = _mm512_set1_epi32(-127);
  const __m512i maximum = _mm512_set1_epi32(127);
  const size_t kGroups = destination.inputsPadded / 4;
  for(size_t channel = 0; channel < HEAD_DIM; channel++) {
    const float* source = channelMajorInput + channel * S;
    const __m512 value0 = _mm512_loadu_ps(source);
    const __m512 value1 = _mm512_loadu_ps(source + 16);
    const __m512 value2 = _mm512_loadu_ps(source + 32);
    float maxAbs = _mm512_reduce_max_ps(_mm512_max_ps(
      _mm512_max_ps(_mm512_and_ps(value0,absMask),_mm512_and_ps(value1,absMask)),
      _mm512_and_ps(value2,absMask)));
    maxAbs = std::max(maxAbs,std::abs(source[48]));
    const float scale = maxAbs > 0.0f ? maxAbs / 127.0f : 0.0f;
    destination.scales[channel] = scale;
    const __m512 inverse = _mm512_set1_ps(maxAbs > 0.0f ? 127.0f / maxAbs : 0.0f);
    __m512i codes0 = _mm512_cvtps_epi32(_mm512_mul_ps(value0,inverse));
    __m512i codes1 = _mm512_cvtps_epi32(_mm512_mul_ps(value1,inverse));
    __m512i codes2 = _mm512_cvtps_epi32(_mm512_mul_ps(value2,inverse));
    codes0 = _mm512_min_epi32(maximum,_mm512_max_epi32(minimum,codes0));
    codes1 = _mm512_min_epi32(maximum,_mm512_max_epi32(minimum,codes1));
    codes2 = _mm512_min_epi32(maximum,_mm512_max_epi32(minimum,codes2));
    storePacked16Codes(
      destination,channel,0,_mm512_cvtsepi32_epi8(codes0));
    storePacked16Codes(
      destination,channel,16,_mm512_cvtsepi32_epi8(codes1));
    storePacked16Codes(
      destination,channel,32,_mm512_cvtsepi32_epi8(codes2));
    int tailCode = 0;
    if(maxAbs > 0.0f) {
      tailCode = std::max(-127,std::min(
        127,static_cast<int>(std::nearbyint(source[48] * 127.0f / maxAbs))));
    }
    const size_t nBlock = channel / 16;
    const size_t lane = channel % 16;
    destination.packed[(nBlock * kGroups + 12) * 64 + lane * 4] =
      static_cast<int8_t>(tailCode);
  }
}

template<int TileRows, bool SubtractInputZeroPoint>
inline __attribute__((always_inline)) void gemmPackedPairTile(
  const uint8_t* input,
  size_t inputStride,
  const float* inputScales,
  const PackedS8Matrix& weights,
  size_t nBlock,
  size_t kGroups,
  float* output,
  size_t outputStride,
  size_t rowBase,
  __mmask16 storeMask1
) {
  __m512i accumulators0[TileRows];
  __m512i accumulators1[TileRows];
  CPU_PTQ_UNROLL_FULL
  for(int row = 0; row < TileRows; row++) {
    accumulators0[row] = _mm512_setzero_si512();
    accumulators1[row] = _mm512_setzero_si512();
  }
  for(size_t kGroup = 0; kGroup < kGroups; kGroup++) {
    const __m512i packedWeight0 = _mm512_loadu_si512(
      reinterpret_cast<const __m512i*>(
        weights.packed.data() + (nBlock * kGroups + kGroup) * 64));
    const __m512i packedWeight1 = _mm512_loadu_si512(
      reinterpret_cast<const __m512i*>(
        weights.packed.data() + ((nBlock + 1) * kGroups + kGroup) * 64));
    CPU_PTQ_UNROLL_FULL
    for(int row = 0; row < TileRows; row++) {
      uint32_t fourInputs;
      std::memcpy(
        &fourInputs,
        input + (rowBase + static_cast<size_t>(row)) * inputStride + kGroup * 4,
        sizeof(fourInputs));
      const __m512i broadcast = _mm512_set1_epi32(static_cast<int>(fourInputs));
      accumulators0[row] = _mm512_dpbusd_epi32(
        accumulators0[row],broadcast,packedWeight0);
      accumulators1[row] = _mm512_dpbusd_epi32(
        accumulators1[row],broadcast,packedWeight1);
    }
  }
  const size_t outputBase0 = nBlock * 16;
  const size_t outputBase1 = outputBase0 + 16;
  const __m512 weightScale0 = _mm512_loadu_ps(
    weights.scales.data() + outputBase0);
  __m512i correction0 = _mm512_setzero_si512();
  if(SubtractInputZeroPoint) {
    correction0 = _mm512_slli_epi32(
      _mm512_loadu_si512(
        reinterpret_cast<const __m512i*>(weights.sums.data() + outputBase0)),7);
  }
  CPU_PTQ_UNROLL_FULL
  for(int row = 0; row < TileRows; row++) {
    __m512i accumulator = accumulators0[row];
    if(SubtractInputZeroPoint)
      accumulator = _mm512_sub_epi32(accumulator,correction0);
    const __m512 value = _mm512_mul_ps(
      _mm512_mul_ps(_mm512_cvtepi32_ps(accumulator),weightScale0),
      _mm512_set1_ps(inputScales[rowBase + static_cast<size_t>(row)]));
    _mm512_storeu_ps(
      output + (rowBase + static_cast<size_t>(row)) * outputStride + outputBase0,
      value);
  }
  const __m512 weightScale1 = _mm512_loadu_ps(
    weights.scales.data() + outputBase1);
  __m512i correction1 = _mm512_setzero_si512();
  if(SubtractInputZeroPoint) {
    correction1 = _mm512_slli_epi32(
      _mm512_loadu_si512(
        reinterpret_cast<const __m512i*>(weights.sums.data() + outputBase1)),7);
  }
  CPU_PTQ_UNROLL_FULL
  for(int row = 0; row < TileRows; row++) {
    __m512i accumulator = accumulators1[row];
    if(SubtractInputZeroPoint)
      accumulator = _mm512_sub_epi32(accumulator,correction1);
    const __m512 value = _mm512_mul_ps(
      _mm512_mul_ps(_mm512_cvtepi32_ps(accumulator),weightScale1),
      _mm512_set1_ps(inputScales[rowBase + static_cast<size_t>(row)]));
    _mm512_mask_storeu_ps(
      output + (rowBase + static_cast<size_t>(row)) * outputStride + outputBase1,
      storeMask1,value);
  }
}

template<bool SubtractInputZeroPoint>
void gemmPackedU8S8(
  const uint8_t* input,
  const float* inputScales,
  const PackedS8Matrix& weights,
  float* output,
  size_t outputStride
) {
  const size_t nBlocks = weights.outputsPadded / 16;
  const size_t kGroups = weights.inputsPadded / 4;
  if(nBlocks % 2 != 0)
    fail("packed output count is not a multiple of 32");
  for(size_t nBlock = 0; nBlock < nBlocks; nBlock += 2) {
    const size_t outputBase1 = nBlock * 16 + 16;
    const size_t validColumns1 = std::min<size_t>(
      16,weights.outputs > outputBase1 ? weights.outputs - outputBase1 : 0);
    const __mmask16 storeMask1 = validColumns1 == 16 ?
      static_cast<__mmask16>(0xffffU) :
      static_cast<__mmask16>(validColumns1 == 0 ? 0 : (1U << validColumns1) - 1U);
    size_t rowBase = 0;
    for(; rowBase + PROJECTION_MR <= S; rowBase += PROJECTION_MR) {
      gemmPackedPairTile<PROJECTION_MR,SubtractInputZeroPoint>(
        input,weights.inputsPadded,inputScales,weights,nBlock,kGroups,
        output,outputStride,rowBase,storeMask1);
    }
    static_assert(S % PROJECTION_MR == 7,"unexpected projection tail geometry");
    gemmPackedPairTile<7,SubtractInputZeroPoint>(
      input,weights.inputsPadded,inputScales,weights,nBlock,kGroups,
      output,outputStride,rowBase,storeMask1);
  }
}

void rmsNormAndMaxAbs(
  const float* input,
  size_t rows,
  size_t channels,
  const float* gamma,
  float* output,
  float* maxAbsValues
) {
  const __m512 absMask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
  for(size_t row = 0; row < rows; row++) {
    const float* source = input + row * channels;
    float* destination = output + row * channels;
    __m512 squares = _mm512_setzero_ps();
    for(size_t channel = 0; channel < channels; channel += 16) {
      const __m512 value = _mm512_loadu_ps(source + channel);
      squares = _mm512_fmadd_ps(value,value,squares);
    }
    const float multiplier = 1.0f / std::sqrt(
      _mm512_reduce_add_ps(squares) / static_cast<float>(channels) + RMS_EPSILON);
    const __m512 scale = _mm512_set1_ps(multiplier);
    __m512 maximum = _mm512_setzero_ps();
    for(size_t channel = 0; channel < channels; channel += 16) {
      const __m512 normalized = _mm512_mul_ps(
        _mm512_mul_ps(_mm512_loadu_ps(source + channel),scale),
        _mm512_loadu_ps(gamma + channel));
      _mm512_storeu_ps(
        destination + channel,normalized);
      maximum = _mm512_max_ps(maximum,_mm512_and_ps(normalized,absMask));
    }
    maxAbsValues[row] = _mm512_reduce_max_ps(maximum);
  }
}

void addRmsNormAndMaxAbs(
  float* destination,
  const float* residual,
  size_t rows,
  size_t channels,
  const float* gamma,
  float* output,
  float* maxAbsValues
) {
  const __m512 absMask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
  for(size_t row = 0; row < rows; row++) {
    float* trunkRow = destination + row * channels;
    const float* residualRow = residual + row * channels;
    __m512 squares = _mm512_setzero_ps();
    for(size_t channel = 0; channel < channels; channel += 16) {
      const __m512 value = _mm512_add_ps(
        _mm512_loadu_ps(trunkRow + channel),
        _mm512_loadu_ps(residualRow + channel));
      _mm512_storeu_ps(trunkRow + channel,value);
      squares = _mm512_fmadd_ps(value,value,squares);
    }
    const float multiplier = 1.0f / std::sqrt(
      _mm512_reduce_add_ps(squares) / static_cast<float>(channels) + RMS_EPSILON);
    const __m512 scale = _mm512_set1_ps(multiplier);
    __m512 maximum = _mm512_setzero_ps();
    float* normalizedRow = output + row * channels;
    for(size_t channel = 0; channel < channels; channel += 16) {
      const __m512 normalized = _mm512_mul_ps(
        _mm512_mul_ps(_mm512_loadu_ps(trunkRow + channel),scale),
        _mm512_loadu_ps(gamma + channel));
      _mm512_storeu_ps(normalizedRow + channel,normalized);
      maximum = _mm512_max_ps(maximum,_mm512_and_ps(normalized,absMask));
    }
    maxAbsValues[row] = _mm512_reduce_max_ps(maximum);
  }
}

void addInPlace(float* destination, const float* source, size_t count) {
  size_t index = 0;
  for(; index + 16 <= count; index += 16) {
    _mm512_storeu_ps(
      destination + index,
      _mm512_add_ps(
        _mm512_loadu_ps(destination + index),_mm512_loadu_ps(source + index)));
  }
  for(; index < count; index++)
    destination[index] += source[index];
}

struct BlockWeights {
  bool useQKNorm;
  float swigluClip;
  vector<float> norm1;
  vector<float> norm2;
  vector<float> qNorm;
  vector<float> kNorm;
  PackedS8Matrix qkv;
  PackedS8Matrix attentionOut;
  PackedS8Matrix upGate;
  PackedS8Matrix down;
};

class IceLakeKernel final : public Kernel {
 public:
  explicit IceLakeKernel(const Model& model)
    : profile_(*model.profile),
      c(static_cast<size_t>(profile_.channels)),
      heads(static_cast<size_t>(profile_.heads)),
      ffn(static_cast<size_t>(profile_.ffnChannels)) {
    if(profile_.headDim != HEAD_DIM || c % 16 != 0 || ffn % 16 != 0)
      fail("compiled profile violates vector geometry");
    loadWeights(model);
    allocateWorkspace();
    precomputeRope();
  }

  const ProfileSpec& profile() const override { return profile_; }

  void infer(
    const float* spatialNCHW,
    const float* global,
    float* policy,
    float* value,
    float* miscValue
  ) override {
    inferImpl(spatialNCHW,global,policy,value,miscValue,nullptr);
  }

  void inferWithTraces(
    const float* spatialNCHW,
    const float* global,
    float* policy,
    float* value,
    float* miscValue,
    vector<float>& traces
  ) override {
    traces.clear();
    inferImpl(spatialNCHW,global,policy,value,miscValue,&traces);
  }

 private:
  void inferImpl(
    const float* spatialNCHW,
    const float* global,
    float* policy,
    float* value,
    float* miscValue,
    vector<float>* traces
  ) {
    stem(spatialNCHW,global);
    if(traces != nullptr)
      traces->insert(traces->end(),trunk.begin(),trunk.end());
    for(size_t block = 0; block < blocks.size(); block++) {
      transformerBlock(blocks[block]);
      if(traces != nullptr)
        traces->insert(traces->end(),trunk.begin(),trunk.end());
    }
    headsOutput(policy,value,miscValue);
  }
  const ProfileSpec& profile_;
  size_t c;
  size_t heads;
  size_t ffn;

  vector<float> stemWeights;
  vector<float> globalWeights;
  vector<BlockWeights> blocks;
  vector<float> trunkMul;
  vector<float> trunkBias;

  vector<float> policyConv;
  vector<float> policyBias;
  vector<float> policyLinearG;
  vector<float> policyLinearPass;
  vector<float> policyConv2;

  vector<float> valueConv;
  vector<float> valueBias1;
  vector<float> valueLinear2Effective;
  vector<float> valueBias2;
  vector<float> valueLinear;
  vector<float> valueBias;
  vector<float> miscLinear;
  vector<float> miscBias;
  vector<float> moreLinear;
  vector<float> moreBias;

  vector<float> ropeCos;
  vector<float> ropeSin;

  vector<float> trunk;
  vector<float> normalized;
  vector<float> projection;
  vector<float> qkv;
  vector<float> q;
  vector<float> k;
  vector<float> v;
  vector<float> attentionOutput;
  vector<float> upGate;
  vector<float> product;
  vector<uint8_t> quantizedInput;
  vector<float> quantizedScales;

  vector<uint8_t> attentionQU8;
  vector<float> attentionQScale;
  vector<float> attentionScores;
  vector<uint8_t> attentionProb;
  vector<float> attentionProbScale;
  PackedS8Matrix packedKey;
  PackedS8Matrix packedValue;

  vector<float> headTrunk;
  vector<float> policyBranches;
  vector<float> valueBranch;
  vector<float> globalProjected;

  void loadWeights(const Model& model) {
    const vector<float>& sourceStem = requireFP32(
      model,"stem.weight",{static_cast<uint32_t>(c),SPATIAL_INPUTS,3,3});
    stemWeights.resize(3 * 3 * SPATIAL_INPUTS * c);
    for(size_t output = 0; output < c; output++)
      for(size_t input = 0; input < SPATIAL_INPUTS; input++)
        for(size_t y = 0; y < 3; y++)
          for(size_t x = 0; x < 3; x++)
            stemWeights[((y * 3 + x) * SPATIAL_INPUTS + input) * c + output] =
              sourceStem[((output * SPATIAL_INPUTS + input) * 3 + y) * 3 + x];
    globalWeights = outputMajorToInputMajor(
      requireFP32(model,"global.weight",{static_cast<uint32_t>(c),GLOBAL_INPUTS}),
      c,GLOBAL_INPUTS);

    if(model.transformerBlocks.size() != static_cast<size_t>(profile_.blocks))
      fail("Transformer semantic descriptor count mismatch");
    blocks.reserve(profile_.blocks);
    for(int block = 0; block < profile_.blocks; block++) {
      const string prefix = "blocks." + std::to_string(block) + ".";
      BlockWeights weights;
      const TransformerBlockSemantics& semantics = model.transformerBlocks[block];
      weights.useQKNorm = semantics.useQKNorm;
      weights.swigluClip = semantics.swigluClip;
      weights.norm1 = requireFP32(
        model,prefix + "norm1",{static_cast<uint32_t>(c)});
      weights.norm2 = requireFP32(
        model,prefix + "norm2",{static_cast<uint32_t>(c)});
      if(weights.useQKNorm) {
        weights.qNorm = requireFP32(model,prefix + "q_norm",{HEAD_DIM});
        weights.kNorm = requireFP32(model,prefix + "k_norm",{HEAD_DIM});
      }
      const Tensor& qProjection = requireS8(
        model,prefix + "q_proj",{static_cast<uint32_t>(c),static_cast<uint32_t>(c)});
      const Tensor& kProjection = requireS8(
        model,prefix + "k_proj",{static_cast<uint32_t>(c),static_cast<uint32_t>(c)});
      const Tensor& vProjection = requireS8(
        model,prefix + "v_proj",{static_cast<uint32_t>(c),static_cast<uint32_t>(c)});
      weights.qkv = concatenateAndPack(
        {&qProjection,&kProjection,&vProjection},c);
      weights.attentionOut = packTensor(requireS8(
        model,prefix + "out_proj",{static_cast<uint32_t>(c),static_cast<uint32_t>(c)}));
      const Tensor& up = requireS8(
        model,prefix + "ffn_linear1",{static_cast<uint32_t>(ffn),static_cast<uint32_t>(c)});
      const Tensor& gate = requireS8(
        model,prefix + "ffn_linear_gate",{static_cast<uint32_t>(ffn),static_cast<uint32_t>(c)});
      weights.upGate = concatenateAndPack({&up,&gate},c);
      weights.down = packTensor(requireS8(
        model,prefix + "ffn_linear2",{static_cast<uint32_t>(c),static_cast<uint32_t>(ffn)}));
      blocks.push_back(std::move(weights));
    }

    trunkMul = requireFP32(model,"trunk.mul",{static_cast<uint32_t>(c)});
    trunkBias = requireFP32(model,"trunk.bias",{static_cast<uint32_t>(c)});

    vector<float> policyCombined(64 * c);
    const vector<float>& policyP = requireFP32(
      model,"policy.conv1p",{32,static_cast<uint32_t>(c)});
    const vector<float>& policyG = requireFP32(
      model,"policy.conv1g",{32,static_cast<uint32_t>(c)});
    std::copy(policyP.begin(),policyP.end(),policyCombined.begin());
    std::copy(policyG.begin(),policyG.end(),policyCombined.begin() + 32 * c);
    policyConv = outputMajorToInputMajor(policyCombined,64,c);
    policyBias.assign(64,0.0f);
    const vector<float>& biasG = requireFP32(model,"policy.biasg",{32});
    const vector<float>& biasP = requireFP32(model,"policy.bias2",{32});
    std::copy(biasP.begin(),biasP.end(),policyBias.begin());
    std::copy(biasG.begin(),biasG.end(),policyBias.begin() + 32);
    policyLinearG = outputMajorToInputMajor(
      requireFP32(model,"policy.linear_g",{32,96}),32,96);
    policyLinearPass = requireFP32(model,"policy.linear_pass",{96});
    policyConv2 = requireFP32(model,"policy.conv2p",{32});

    valueConv = outputMajorToInputMajor(
      requireFP32(model,"value.conv1",{32,static_cast<uint32_t>(c)}),32,c);
    valueBias1 = requireFP32(model,"value.bias1",{32});
    const vector<float>& valueLinear2 = requireFP32(model,"value.linear2",{64,96});
    vector<float> valueLinear2Collapsed(64 * 32);
    for(size_t output = 0; output < 64; output++) {
      for(size_t channel = 0; channel < 32; channel++) {
        valueLinear2Collapsed[output * 32 + channel] =
          valueLinear2[output * 96 + channel] -
          0.7f * valueLinear2[output * 96 + 32 + channel] +
          0.39f * valueLinear2[output * 96 + 64 + channel];
      }
    }
    valueLinear2Effective = outputMajorToInputMajor(
      valueLinear2Collapsed,64,32);
    valueBias2 = requireFP32(model,"value.bias2",{64});
    valueLinear = requireFP32(model,"value.linear_value",{3,64});
    valueBias = requireFP32(model,"value.bias_value",{3});
    miscLinear = requireFP32(model,"value.linear_misc",{4,64});
    miscBias = requireFP32(model,"value.bias_misc",{4});
    moreLinear = requireFP32(model,"value.linear_more",{2,64});
    moreBias = requireFP32(model,"value.bias_more",{2});
  }

  void allocateWorkspace() {
    trunk.resize(S * c);
    normalized.resize(S * c);
    projection.resize(S * c);
    qkv.resize(S * 3 * c);
    q.resize(heads * S * HEAD_DIM);
    k.resize(heads * S * HEAD_DIM);
    v.resize(heads * S * HEAD_DIM);
    attentionOutput.resize(S * c);
    upGate.resize(S * 2 * ffn);
    product.resize(S * ffn);
    quantizedInput.resize(S * std::max(c,ffn));
    quantizedScales.resize(S);

    attentionQU8.resize(S * HEAD_DIM);
    attentionQScale.resize(S);
    attentionScores.resize(S * S);
    attentionProb.resize(S * roundUp(S,4));
    attentionProbScale.resize(S);
    packedKey = allocateRuntimePacked(S,HEAD_DIM);
    packedValue = allocateRuntimePacked(HEAD_DIM,S);

    headTrunk.resize(S * c);
    policyBranches.resize(S * 64);
    valueBranch.resize(S * 32);
    globalProjected.resize(c);
  }

  void precomputeRope() {
    ropeCos.resize(S * HEAD_DIM);
    ropeSin.resize(S * HEAD_DIM);
    const float theta = 100.0f;
    float frequencies[8];
    for(size_t index = 0; index < 8; index++)
      frequencies[index] = 1.0f / std::pow(theta,static_cast<float>(2 * index) / 16.0f);
    for(size_t position = 0; position < S; position++) {
      const float y = static_cast<float>(position / BOARD_LEN);
      const float x = static_cast<float>(position % BOARD_LEN);
      for(size_t pair = 0; pair < 16; pair++) {
        const float coordinate = pair < 8 ? y : x;
        const float angle = coordinate * frequencies[pair % 8];
        const float cosine = std::cos(angle);
        const float sine = std::sin(angle);
        ropeCos[position * HEAD_DIM + pair * 2] = cosine;
        ropeCos[position * HEAD_DIM + pair * 2 + 1] = cosine;
        // Store the alternating RoPE sign here so inference can rotate a full
        // ZMM vector with one adjacent-lane permutation and one FMA.
        ropeSin[position * HEAD_DIM + pair * 2] = -sine;
        ropeSin[position * HEAD_DIM + pair * 2 + 1] = sine;
      }
    }
  }

  void stem(const float* spatial, const float* global) {
    std::fill(globalProjected.begin(),globalProjected.end(),0.0f);
    for(size_t input = 0; input < GLOBAL_INPUTS; input++) {
      const __m512 inputValue = _mm512_set1_ps(global[input]);
      for(size_t output = 0; output < c; output += 16) {
        _mm512_storeu_ps(
          globalProjected.data() + output,
          _mm512_fmadd_ps(
            inputValue,
            _mm512_loadu_ps(globalWeights.data() + input * c + output),
            _mm512_loadu_ps(globalProjected.data() + output)));
      }
    }
    for(size_t y = 0; y < BOARD_LEN; y++) {
      for(size_t x = 0; x < BOARD_LEN; x++) {
        float* destination = trunk.data() + (y * BOARD_LEN + x) * c;
        std::copy(globalProjected.begin(),globalProjected.end(),destination);
        for(size_t kernelY = 0; kernelY < 3; kernelY++) {
          const int sourceY = static_cast<int>(y + kernelY) - 1;
          if(sourceY < 0 || sourceY >= BOARD_LEN)
            continue;
          for(size_t kernelX = 0; kernelX < 3; kernelX++) {
            const int sourceX = static_cast<int>(x + kernelX) - 1;
            if(sourceX < 0 || sourceX >= BOARD_LEN)
              continue;
            for(size_t input = 0; input < SPATIAL_INPUTS; input++) {
              const float inputScalar = spatial[
                (input * BOARD_LEN + static_cast<size_t>(sourceY)) * BOARD_LEN +
                static_cast<size_t>(sourceX)];
              if(inputScalar == 0.0f)
                continue;
              if(inputScalar != 1.0f)
                fail("spatial input is not binary");
              const float* weight = stemWeights.data() +
                ((kernelY * 3 + kernelX) * SPATIAL_INPUTS + input) * c;
              for(size_t output = 0; output < c; output += 16) {
                _mm512_storeu_ps(
                  destination + output,
                  _mm512_add_ps(
                    _mm512_loadu_ps(weight + output),
                    _mm512_loadu_ps(destination + output)));
              }
            }
          }
        }
      }
    }
  }

  void transformerBlock(const BlockWeights& weights) {
    rmsNormAndMaxAbs(
      trunk.data(),S,c,weights.norm1.data(),normalized.data(),quantizedScales.data());
    quantizeRowsU8<true>(
      normalized.data(),S,c,weights.qkv.inputsPadded,
      quantizedInput.data(),quantizedScales.data());
    gemmPackedU8S8<true>(
      quantizedInput.data(),quantizedScales.data(),weights.qkv,qkv.data(),3 * c);
    if(weights.useQKNorm)
      applyRopeAndAttention<true>(weights);
    else
      applyRopeAndAttention<false>(weights);

    quantizeRowsU8<false>(
      attentionOutput.data(),S,c,weights.attentionOut.inputsPadded,
      quantizedInput.data(),quantizedScales.data());
    gemmPackedU8S8<true>(
      quantizedInput.data(),quantizedScales.data(),weights.attentionOut,
      projection.data(),c);
    addRmsNormAndMaxAbs(
      trunk.data(),projection.data(),S,c,weights.norm2.data(),
      normalized.data(),quantizedScales.data());
    quantizeRowsU8<true>(
      normalized.data(),S,c,weights.upGate.inputsPadded,
      quantizedInput.data(),quantizedScales.data());
    gemmPackedU8S8<true>(
      quantizedInput.data(),quantizedScales.data(),weights.upGate,
      upGate.data(),2 * ffn);
    if(weights.swigluClip > 0.0f)
      swigluProduct<true>(weights.swigluClip);
    else
      swigluProduct<false>(0.0f);
    quantizeRowsU8<true>(
      product.data(),S,ffn,weights.down.inputsPadded,
      quantizedInput.data(),quantizedScales.data());
    gemmPackedU8S8<true>(
      quantizedInput.data(),quantizedScales.data(),weights.down,
      projection.data(),c);
    addInPlace(trunk.data(),projection.data(),S * c);
  }

  template<bool UseClip>
  void swigluProduct(float clip) {
    const __m512 absMask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
    const __m512 lower = _mm512_set1_ps(-clip);
    const __m512 upper = _mm512_set1_ps(clip);
    for(size_t row = 0; row < S; row++) {
      const float* up = upGate.data() + row * 2 * ffn;
      const float* gate = up + ffn;
      float* destination = product.data() + row * ffn;
      __m512 maximum = _mm512_setzero_ps();
      for(size_t channel = 0; channel < ffn; channel += 16) {
        __m512 activated = silu512_ps(_mm512_loadu_ps(up + channel));
        __m512 gateValues = _mm512_loadu_ps(gate + channel);
        if(UseClip) {
          activated = _mm512_max_ps(lower,_mm512_min_ps(upper,activated));
          gateValues = _mm512_max_ps(lower,_mm512_min_ps(upper,gateValues));
        }
        const __m512 multiplied = _mm512_mul_ps(activated,gateValues);
        _mm512_storeu_ps(destination + channel,multiplied);
        maximum = _mm512_max_ps(maximum,_mm512_and_ps(multiplied,absMask));
      }
      quantizedScales[row] = _mm512_reduce_max_ps(maximum);
    }
  }

  template<bool UseQKNorm>
  void applyRopeAndAttention(const BlockWeights& weights) {
    for(size_t position = 0; position < S; position++) {
      const float* source = qkv.data() + position * 3 * c;
      const float* cosine = ropeCos.data() + position * HEAD_DIM;
      const float* signedSine = ropeSin.data() + position * HEAD_DIM;
      for(size_t head = 0; head < heads; head++) {
        const size_t destinationBase = (head * S + position) * HEAD_DIM;
        const size_t channelBase = head * HEAD_DIM;
        __m512 queryParts[2] = {
          _mm512_loadu_ps(source + channelBase),
          _mm512_loadu_ps(source + channelBase + 16),
        };
        __m512 keyParts[2] = {
          _mm512_loadu_ps(source + c + channelBase),
          _mm512_loadu_ps(source + c + channelBase + 16),
        };
        if(UseQKNorm) {
          const float queryMultiplier = 1.0f / std::sqrt(
            _mm512_reduce_add_ps(_mm512_fmadd_ps(
              queryParts[0],queryParts[0],
              _mm512_mul_ps(queryParts[1],queryParts[1]))) /
              static_cast<float>(HEAD_DIM) + RMS_EPSILON);
          const float keyMultiplier = 1.0f / std::sqrt(
            _mm512_reduce_add_ps(_mm512_fmadd_ps(
              keyParts[0],keyParts[0],
              _mm512_mul_ps(keyParts[1],keyParts[1]))) /
              static_cast<float>(HEAD_DIM) + RMS_EPSILON);
          const __m512 queryScale = _mm512_set1_ps(queryMultiplier);
          const __m512 keyScale = _mm512_set1_ps(keyMultiplier);
          for(size_t part = 0; part < 2; part++) {
            queryParts[part] = _mm512_mul_ps(
              _mm512_mul_ps(queryParts[part],queryScale),
              _mm512_loadu_ps(weights.qNorm.data() + part * 16));
            keyParts[part] = _mm512_mul_ps(
              _mm512_mul_ps(keyParts[part],keyScale),
              _mm512_loadu_ps(weights.kNorm.data() + part * 16));
          }
        }
        for(size_t part = 0; part < 2; part++) {
          const size_t channel = part * 16;
          const __m512 query = queryParts[part];
          const __m512 key = keyParts[part];
          const __m512 cosineVector = _mm512_loadu_ps(cosine + channel);
          const __m512 sineVector = _mm512_loadu_ps(signedSine + channel);
          _mm512_storeu_ps(
            q.data() + destinationBase + channel,
            _mm512_fmadd_ps(
              _mm512_permute_ps(query,_MM_SHUFFLE(2,3,0,1)),sineVector,
              _mm512_mul_ps(query,cosineVector)));
          _mm512_storeu_ps(
            k.data() + destinationBase + channel,
            _mm512_fmadd_ps(
              _mm512_permute_ps(key,_MM_SHUFFLE(2,3,0,1)),sineVector,
              _mm512_mul_ps(key,cosineVector)));
        }
        for(size_t channel = 0; channel < HEAD_DIM; channel++) {
          v[(channelBase + channel) * S + position] =
            source[2 * c + channelBase + channel];
        }
      }
    }
    for(size_t head = 0; head < heads; head++)
      attentionHead(head);
  }

  void attentionHead(size_t head) {
    const float* query = q.data() + head * S * HEAD_DIM;
    const float* key = k.data() + head * S * HEAD_DIM;
    const float* value = v.data() + head * S * HEAD_DIM;
    quantizeRowsU8<false>(
      query,S,HEAD_DIM,HEAD_DIM,attentionQU8.data(),attentionQScale.data());
    quantizeAttentionKeysPacked(key,packedKey);
    gemmPackedU8S8<true>(
      attentionQU8.data(),attentionQScale.data(),packedKey,
      attentionScores.data(),S);

    const size_t probabilityStride = roundUp(S,4);
    alignas(64) float partialSums[S];
    alignas(64) float tailExponentials[S];
    for(size_t row = 0; row < S; row++) {
      float* scores = attentionScores.data() + row * S;
      const __m512 attentionScale = _mm512_set1_ps(ATTENTION_SCALE);
      const __m512 scores0 = _mm512_mul_ps(
        _mm512_loadu_ps(scores),attentionScale);
      const __m512 scores1 = _mm512_mul_ps(
        _mm512_loadu_ps(scores + 16),attentionScale);
      const __m512 scores2 = _mm512_mul_ps(
        _mm512_loadu_ps(scores + 32),attentionScale);
      _mm512_storeu_ps(scores,scores0);
      _mm512_storeu_ps(scores + 16,scores1);
      _mm512_storeu_ps(scores + 32,scores2);
      float maximum = _mm512_reduce_max_ps(
        _mm512_max_ps(scores0,_mm512_max_ps(scores1,scores2)));
      scores[48] *= ATTENTION_SCALE;
      maximum = std::max(maximum,scores[48]);
      uint8_t* probabilities = attentionProb.data() + row * probabilityStride;
      float sum = 0.0f;
      size_t column = 0;
      const __m512 maximumVector = _mm512_set1_ps(maximum);
      for(; column + 16 <= S; column += 16) {
        const __m512 exponential = exp512_ps(
          _mm512_sub_ps(_mm512_loadu_ps(scores + column),maximumVector));
        sum += _mm512_reduce_add_ps(exponential);
        __m512i codes = _mm512_cvtps_epi32(
          _mm512_mul_ps(exponential,_mm512_set1_ps(255.0f)));
        codes = _mm512_min_epi32(_mm512_set1_epi32(255),
          _mm512_max_epi32(_mm512_setzero_si512(),codes));
        _mm_storeu_si128(
          reinterpret_cast<__m128i*>(probabilities + column),
          _mm512_cvtusepi32_epi8(codes));
      }
      static_assert(S == 49,"unexpected attention tail geometry");
      partialSums[row] = sum;
      tailExponentials[row] = scores[48] - maximum;
    }
    size_t row = 0;
    for(; row + 16 <= S; row += 16) {
      _mm512_store_ps(
        tailExponentials + row,
        exp512_ps(_mm512_load_ps(tailExponentials + row)));
    }
    const __mmask16 tailMask = static_cast<__mmask16>((1U << (S - row)) - 1U);
    _mm512_mask_storeu_ps(
      tailExponentials + row,tailMask,
      exp512_ps(_mm512_maskz_loadu_ps(tailMask,tailExponentials + row)));
    for(row = 0; row < S; row++) {
      uint8_t* probabilities = attentionProb.data() + row * probabilityStride;
      const float exponential = tailExponentials[row];
      probabilities[48] = static_cast<uint8_t>(std::max(
        0,std::min(255,static_cast<int>(std::nearbyint(exponential * 255.0f)))));
      std::fill(probabilities + S,probabilities + probabilityStride,0);
      attentionProbScale[row] = 1.0f / (255.0f * (partialSums[row] + exponential));
    }

    quantizeAttentionValuesPacked(value,packedValue);
    gemmPackedU8S8<false>(
      attentionProb.data(),attentionProbScale.data(),packedValue,
      attentionOutput.data() + head * HEAD_DIM,c);
  }

  void headsOutput(float* policy, float* value, float* miscValue) {
    for(size_t row = 0; row < S; row++) {
      for(size_t channel = 0; channel < c; channel += 16) {
        _mm512_storeu_ps(
          headTrunk.data() + row * c + channel,
          silu512_ps(_mm512_fmadd_ps(
            _mm512_loadu_ps(trunk.data() + row * c + channel),
            _mm512_loadu_ps(trunkMul.data() + channel),
            _mm512_loadu_ps(trunkBias.data() + channel))));
      }
    }
    linearInputMajor(
      headTrunk.data(),S,c,policyConv.data(),64,policyBias.data(),
      policyBranches.data());
    for(size_t row = 0; row < S; row++)
      siluInPlace(policyBranches.data() + row * 64 + 32,32);

    float pooled[96];
    for(size_t channel = 0; channel < 32; channel++) {
      float sum = 0.0f;
      float maximum = -std::numeric_limits<float>::infinity();
      for(size_t row = 0; row < S; row++) {
        const float valueAt = policyBranches[row * 64 + 32 + channel];
        sum += valueAt;
        maximum = std::max(maximum,valueAt);
      }
      const float mean = sum / static_cast<float>(S);
      pooled[channel] = mean;
      pooled[32 + channel] = -0.7f * mean;
      pooled[64 + channel] = maximum;
    }
    const float pass = dot(pooled,policyLinearPass.data(),96);
    float policyBiasFromGlobal[32];
    linearInputMajor(
      pooled,1,96,policyLinearG.data(),32,nullptr,policyBiasFromGlobal);
    for(size_t row = 0; row < S; row++) {
      float* spatialPolicy = policyBranches.data() + row * 64;
      for(size_t channel = 0; channel < 32; channel++)
        spatialPolicy[channel] += policyBiasFromGlobal[channel];
      siluInPlace(spatialPolicy,32);
      policy[row] = dot(spatialPolicy,policyConv2.data(),32);
    }
    policy[S] = pass;

    linearInputMajor(
      headTrunk.data(),S,c,valueConv.data(),32,valueBias1.data(),valueBranch.data());
    siluInPlace(valueBranch.data(),valueBranch.size());
    float valueMean[32];
    for(size_t channel = 0; channel < 32; channel++) {
      float sum = 0.0f;
      for(size_t row = 0; row < S; row++)
        sum += valueBranch[row * 32 + channel];
      valueMean[channel] = sum / static_cast<float>(S);
    }
    float valueHidden[64];
    linearInputMajor(
      valueMean,1,32,valueLinear2Effective.data(),64,valueBias2.data(),valueHidden);
    siluInPlace(valueHidden,64);
    for(size_t output = 0; output < 3; output++)
      value[output] = valueBias[output] +
        dot(valueHidden,valueLinear.data() + output * 64,64);
    for(size_t output = 0; output < 4; output++)
      miscValue[output] = miscBias[output] +
        dot(valueHidden,miscLinear.data() + output * 64,64);
    for(size_t output = 0; output < 2; output++)
      miscValue[4 + output] = moreBias[output] +
        dot(valueHidden,moreLinear.data() + output * 64,64);
  }
};

}  // namespace

std::unique_ptr<Kernel> createKernel(const Model& model) {
  if(model.profile == nullptr)
    fail("model has no selected profile");
  return std::unique_ptr<Kernel>(new IceLakeKernel(model));
}

}  // namespace CpuPtq

#include "kernel.h"

#include <xnnpack.h>
#include <mlas.h>

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <malloc.h>
#endif

#ifndef CPU_PTQ_PROFILE_NAMESPACE
#error "CPU_PTQ_PROFILE_NAMESPACE must name this profile's implementation namespace"
#endif
#ifndef CPU_PTQ_PROFILE_KIND
#error "CPU_PTQ_PROFILE_KIND must identify this profile"
#endif
#ifndef CPU_PTQ_TRUNK_CHANNELS
#error "CPU_PTQ_TRUNK_CHANNELS must be defined"
#endif
#ifndef CPU_PTQ_HEADS
#error "CPU_PTQ_HEADS must be defined"
#endif
#ifndef CPU_PTQ_FFN_CHANNELS
#error "CPU_PTQ_FFN_CHANNELS must be defined"
#endif
#ifndef CPU_PTQ_BLOCKS
#error "CPU_PTQ_BLOCKS must be defined"
#endif
#ifndef CPU_PTQ_VALUE_HIDDEN_CHANNELS
#error "CPU_PTQ_VALUE_HIDDEN_CHANNELS must be defined"
#endif
#ifndef CPU_PTQ_PROFILE_FUNCTION
#error "CPU_PTQ_PROFILE_FUNCTION must name this profile descriptor function"
#endif
#ifndef KTCP_ATTN_Q_BLOCK
#define KTCP_ATTN_Q_BLOCK 225
#endif
#ifndef KTCP_ATTN_KV_BLOCK
#define KTCP_ATTN_KV_BLOCK 225
#endif
namespace CpuPtq {
namespace CPU_PTQ_PROFILE_NAMESPACE {

using Clock = std::chrono::steady_clock;

enum ProjectionMode {
  KTCP_MODE_FP32 = 0,
  KTCP_MODE_QD8 = 1,
  KTCP_MODE_MLAS_U8S8 = 2,
};

constexpr size_t KTCP_SPATIAL_INPUTS = SPATIAL_INPUTS;
constexpr size_t KTCP_GLOBAL_INPUTS = GLOBAL_INPUTS;
constexpr size_t KTCP_BOARD_LEN = BOARD_LEN;
constexpr size_t KTCP_POLICY_CHANNELS = 1;
constexpr size_t KTCP_VALUE_SIZE = VALUE_SIZE;
constexpr size_t KTCP_SCORE_VALUE_SIZE = SCORE_VALUE_SIZE;
constexpr size_t KTCP_OWNERSHIP_SIZE = OWNERSHIP_SIZE;

enum ProfileStage : size_t {
  kProfileInputStem = 0,
  kProfileNorm1,
  kProfileQkv,
  kProfileRope,
  kProfileAttentionQk,
  kProfileAttentionSoftmax,
  kProfileAttentionPv,
  kProfileAttentionTranspose,
  kProfileOutProjection,
  kProfileResidualNorm2,
  kProfileUpGate,
  kProfileSwiGlu,
  kProfileDown,
  kProfileResidualAdd,
  kProfileHeads,
};

struct Profile {
  std::array<double, 15> milliseconds{};

  void add(ProfileStage stage, Clock::time_point begin, Clock::time_point end) {
    milliseconds[stage] +=
        std::chrono::duration<double, std::milli>(end - begin).count();
  }
};

constexpr size_t kS = BOARD_AREA;
constexpr size_t kC = CPU_PTQ_TRUNK_CHANNELS;
constexpr size_t kHeads = CPU_PTQ_HEADS;
constexpr size_t kHeadDim = 32;
constexpr size_t kPairs = 16;
constexpr size_t kFfn = CPU_PTQ_FFN_CHANNELS;
constexpr size_t kBlocks = CPU_PTQ_BLOCKS;
constexpr size_t kValueHidden = CPU_PTQ_VALUE_HIDDEN_CHANNELS;
constexpr size_t kAttentionQBlock = KTCP_ATTN_Q_BLOCK;
constexpr size_t kAttentionKvBlock = KTCP_ATTN_KV_BLOCK;
constexpr size_t kAttentionPaddedS = 240;
constexpr float kRmsEpsilon = 1.0e-6f;
constexpr float kSwiGluClip = 4.0f;
constexpr float kAttentionScale = 0.1767766952966369f;  // 1/sqrt(32)
static_assert(kAttentionQBlock >= 1 && kAttentionQBlock <= kS);
static_assert(kAttentionKvBlock >= 1 && kAttentionKvBlock <= kS);

inline __m256 exp256_ps(__m256 x);
inline float horizontal_sum(__m256 value);
#if defined(KTCP_POSTOPS_AVX512)
inline __m512 exp512_post_ps(__m512 x);
#endif

#if defined(KTCP_ATTENTION_VNNI)
inline float quantize_s8x32_avx512(
    __m512 value0,
    __m512 value1,
    int8_t* destination,
    int32_t* quantized_sum = nullptr) {
  const __m512 sign_mask =
      _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
  const float maximum = _mm512_reduce_max_ps(_mm512_max_ps(
      _mm512_and_ps(value0, sign_mask),
      _mm512_and_ps(value1, sign_mask)));
  if (!(maximum > 0.0f)) {
    std::memset(destination, 0, kHeadDim);
    if (quantized_sum != nullptr) *quantized_sum = 0;
    return 1.0f;
  }
  const float scale = maximum / 127.0f;
  const __m512 inverse = _mm512_set1_ps(127.0f / maximum);
  const __m512i minimum = _mm512_set1_epi32(-127);
  const __m512i maximum_i32 = _mm512_set1_epi32(127);
  const __m512i quantized0 = _mm512_min_epi32(
      maximum_i32,
      _mm512_max_epi32(
          minimum, _mm512_cvtps_epi32(_mm512_mul_ps(value0, inverse))));
  const __m512i quantized1 = _mm512_min_epi32(
      maximum_i32,
      _mm512_max_epi32(
          minimum, _mm512_cvtps_epi32(_mm512_mul_ps(value1, inverse))));
  _mm_storeu_si128(
      reinterpret_cast<__m128i*>(destination),
      _mm512_cvtsepi32_epi8(quantized0));
  _mm_storeu_si128(
      reinterpret_cast<__m128i*>(destination + 16),
      _mm512_cvtsepi32_epi8(quantized1));
  if (quantized_sum != nullptr) {
    *quantized_sum = _mm512_reduce_add_epi32(
        _mm512_add_epi32(quantized0, quantized1));
  }
  return scale;
}
#endif

void check_xnn(enum xnn_status status, const char* operation) {
  if (status != xnn_status_success) {
    std::ostringstream message;
    message << operation << " failed with xnn_status=" << static_cast<int>(status);
    throw std::runtime_error(message.str());
  }
}

template <class T>
std::vector<T> padded_vector(size_t elements) {
  const size_t extra = (static_cast<size_t>(XNN_EXTRA_BYTES) + sizeof(T) - 1) / sizeof(T);
  return std::vector<T>(elements + extra);
}

class Operator {
 public:
  Operator() = default;
  explicit Operator(xnn_operator_t value) : value_(value) {}
  ~Operator() {
    if (value_ != nullptr) {
      xnn_delete_operator(value_);
    }
  }
  Operator(const Operator&) = delete;
  Operator& operator=(const Operator&) = delete;
  Operator(Operator&& other) noexcept : value_(std::exchange(other.value_, nullptr)) {}
  Operator& operator=(Operator&& other) noexcept {
    if (this != &other) {
      if (value_ != nullptr) {
        xnn_delete_operator(value_);
      }
      value_ = std::exchange(other.value_, nullptr);
    }
    return *this;
  }
  xnn_operator_t get() const { return value_; }

 private:
  xnn_operator_t value_ = nullptr;
};

const std::vector<float>& require_values(
    const TensorMap& tensors,
    const std::string& name,
    std::initializer_list<uint64_t> dims) {
  return requireTensor(tensors, name, dims);
}

std::vector<float> zero_bias(size_t channels) {
  return std::vector<float>(channels, 0.0f);
}

struct CanonicalS8Weights {
  std::vector<int8_t> values;
  std::vector<float> scales;
};

CanonicalS8Weights copy_s8_weights(
    const TensorMap& tensors,
    const std::string& name,
    std::initializer_list<uint64_t> dims) {
  const Tensor& tensor = requireQuantizedTensor(tensors, name, dims);
  return CanonicalS8Weights{tensor.quantizedValues, tensor.quantizedScales};
}

CanonicalS8Weights concatenate_s8_rows(
    const std::vector<const Tensor*>& branches,
    size_t input_channels) {
  CanonicalS8Weights result;
  size_t elements = 0;
  size_t outputs = 0;
  for (const Tensor* branch : branches) {
    if (branch == nullptr || branch->shape.size() != 2 ||
        branch->shape[1] != input_channels ||
        branch->quantizedValues.size() % input_channels != 0 ||
        branch->quantizedScales.size() != branch->shape[0] ||
        !branch->values.empty()) {
      throw std::runtime_error("Cannot concatenate malformed S8 projection weights");
    }
    elements += branch->quantizedValues.size();
    outputs += branch->quantizedScales.size();
  }
  result.values.reserve(elements);
  result.scales.reserve(outputs);
  for (const Tensor* branch : branches) {
    result.values.insert(
        result.values.end(),
        branch->quantizedValues.begin(), branch->quantizedValues.end());
    result.scales.insert(
        result.scales.end(),
        branch->quantizedScales.begin(), branch->quantizedScales.end());
  }
  return result;
}

struct QkvRopeOutput {
  float* q = nullptr;
  float* k = nullptr;
  float* v = nullptr;
  float* k_transposed = nullptr;
  const float* cosine = nullptr;
  const float* sine = nullptr;
  const float* q_gamma = nullptr;
  const float* k_gamma = nullptr;

  bool enabled() const {
    return q != nullptr && k != nullptr && v != nullptr &&
           cosine != nullptr && sine != nullptr &&
           q_gamma != nullptr && k_gamma != nullptr;
  }
};

struct SwiGluOutput {
  float* values = nullptr;
  bool enabled() const { return values != nullptr; }
};

struct ResidualOutput {
  float* residual = nullptr;
  float* normalized = nullptr;
  const float* gamma = nullptr;

  bool enabled() const { return residual != nullptr; }
  bool normalizes() const { return normalized != nullptr && gamma != nullptr; }
};

class QkvRopeOutputProcessor final : public MLAS_QGEMM_OUTPUT_PROCESSOR {
 public:
  QkvRopeOutputProcessor(const float* scales, QkvRopeOutput output)
      : scales_(scales), output_(output) {}

  void Process(
      const int32_t* accumulators,
      size_t start_row,
      size_t start_column,
      size_t row_count,
      size_t column_count,
      size_t accumulator_stride) const override {
#if defined(KTCP_POSTOPS_AVX512)
    if (start_column == 0 && column_count == 3 * kC) {
      for (size_t row = start_row; row < start_row + row_count; ++row) {
        const int32_t* source = accumulators + row * accumulator_stride;
        for (size_t projection = 0; projection < 2; ++projection) {
          float* projection_output = projection == 0 ? output_.q : output_.k;
          const float* gamma =
              projection == 0 ? output_.q_gamma : output_.k_gamma;
          for (size_t head = 0; head < kHeads; ++head) {
            const size_t channel_base = projection * kC + head * kHeadDim;
            const size_t rope_base = (row * kHeads + head) * kHeadDim;
            float* destination = projection_output +
                (head * kS + row) * kHeadDim;
            const __m512 values0 = _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_loadu_si512(source + channel_base)),
                _mm512_loadu_ps(scales_ + channel_base));
            const __m512 values1 = _mm512_mul_ps(
                _mm512_cvtepi32_ps(
                    _mm512_loadu_si512(source + channel_base + 16)),
                _mm512_loadu_ps(scales_ + channel_base + 16));
            const float inv_rms = 1.0f / std::sqrt(
                _mm512_reduce_add_ps(_mm512_add_ps(
                    _mm512_mul_ps(values0, values0),
                    _mm512_mul_ps(values1, values1))) /
                    static_cast<float>(kHeadDim) +
                kRmsEpsilon);
            const __m512 norm_scale = _mm512_set1_ps(inv_rms);
            for (size_t half = 0; half < kHeadDim; half += 16) {
              const __m512 values = half == 0 ? values0 : values1;
              const __m512 normalized = _mm512_mul_ps(
                  _mm512_mul_ps(values, _mm512_loadu_ps(gamma + half)),
                  norm_scale);
              const __m512 swapped = _mm512_permute_ps(normalized, 0xb1);
              const __m512 rotated = _mm512_fmadd_ps(
                  swapped,
                  _mm512_loadu_ps(output_.sine + rope_base + half),
                  _mm512_mul_ps(
                       normalized,
                       _mm512_loadu_ps(output_.cosine + rope_base + half)));
              _mm512_storeu_ps(destination + half, rotated);
            }
            if (projection == 1 && output_.k_transposed != nullptr) {
              for (size_t channel = 0; channel < kHeadDim; ++channel) {
                output_.k_transposed[
                    (head * kHeadDim + channel) * kAttentionPaddedS + row] =
                    destination[channel];
              }
            }
          }
        }
        for (size_t channel = 0; channel < kC; channel += 16) {
          const size_t column = 2 * kC + channel;
          const __m512 values = _mm512_mul_ps(
              _mm512_cvtepi32_ps(_mm512_loadu_si512(source + column)),
              _mm512_loadu_ps(scales_ + column));
          const size_t head = channel / kHeadDim;
          const size_t within_head = channel % kHeadDim;
          _mm512_storeu_ps(
              output_.v + (head * kS + row) * kHeadDim + within_head,
              values);
        }
      }
      return;
    }
#endif
    const size_t end_column = start_column + column_count;
    for (size_t row = start_row; row < start_row + row_count; ++row) {
      const int32_t* source = accumulators + row * accumulator_stride;
      for (size_t projection = 0; projection < 2; ++projection) {
        const size_t projection_begin = projection * kC;
        const size_t intersection_begin = std::max(start_column, projection_begin);
        const size_t intersection_end =
            std::min(end_column, projection_begin + kC);
        if (intersection_begin >= intersection_end) continue;
        const size_t first_head =
            (intersection_begin - projection_begin) / kHeadDim;
        const size_t last_head =
            (intersection_end - 1 - projection_begin) / kHeadDim;
        const float* gamma =
            projection == 0 ? output_.q_gamma : output_.k_gamma;
        float* projection_output = projection == 0 ? output_.q : output_.k;
        for (size_t head = first_head; head <= last_head; ++head) {
          std::array<float, kHeadDim> values{};
          float sum_squares = 0.0f;
          const size_t channel_base = projection_begin + head * kHeadDim;
          for (size_t channel = 0; channel < kHeadDim; ++channel) {
            const size_t source_column = channel_base + channel;
            const float value = static_cast<float>(source[source_column]) *
                                scales_[source_column];
            values[channel] = value;
            sum_squares += value * value;
          }
          const float inv_rms = 1.0f / std::sqrt(
              sum_squares / static_cast<float>(kHeadDim) + kRmsEpsilon);
          float* destination = projection_output +
              (head * kS + row) * kHeadDim;
          for (size_t pair = 0; pair < kPairs; ++pair) {
            const size_t pair_channel = pair * 2;
            const float first =
                values[pair_channel] * gamma[pair_channel] * inv_rms;
            const float second =
                values[pair_channel + 1] * gamma[pair_channel + 1] * inv_rms;
#if defined(KTCP_POSTOPS_AVX512)
            const size_t table =
                (row * kHeads + head) * kHeadDim + pair * 2;
            const float cosine = output_.cosine[table];
            const float sine = output_.sine[table + 1];
#else
            const size_t table = (row * kHeads + head) * kPairs + pair;
            const float cosine = output_.cosine[table];
            const float sine = output_.sine[table];
#endif
            const float rotated0 = first * cosine - second * sine;
            const float rotated1 = first * sine + second * cosine;
            destination[pair_channel] = rotated0;
            destination[pair_channel + 1] = rotated1;
            if (projection == 1 && output_.k_transposed != nullptr) {
              output_.k_transposed[
                  (head * kHeadDim + pair_channel) * kAttentionPaddedS + row] =
                  rotated0;
              output_.k_transposed[
                  (head * kHeadDim + pair_channel + 1) * kAttentionPaddedS + row] =
                  rotated1;
            }
          }
        }
      }
      const size_t v_begin = std::max(start_column, 2 * kC);
      const size_t v_end = std::min(end_column, 3 * kC);
      for (size_t column = v_begin; column < v_end; ++column) {
        const size_t within = column - 2 * kC;
        const size_t head = within / kHeadDim;
        const size_t head_channel = within % kHeadDim;
        output_.v[(head * kS + row) * kHeadDim + head_channel] =
            static_cast<float>(source[column]) * scales_[column];
      }
    }
  }

 private:
  const float* scales_ = nullptr;
  QkvRopeOutput output_{};
};

class SwiGluOutputProcessor final : public MLAS_QGEMM_OUTPUT_PROCESSOR {
 public:
  SwiGluOutputProcessor(const float* scales, SwiGluOutput output)
      : scales_(scales), output_(output) {}

  void Process(
      const int32_t* accumulators,
      size_t start_row,
      size_t start_column,
      size_t row_count,
      size_t column_count,
      size_t accumulator_stride) const override {
    const size_t end_column = start_column + column_count;
    for (size_t row = start_row; row < start_row + row_count; ++row) {
      const int32_t* source = accumulators + row * accumulator_stride;
      float* destination = output_.values + row * kFfn;

      size_t column = start_column;
      const size_t up_end = std::min(end_column, kFfn);
#if defined(KTCP_POSTOPS_AVX512)
      for (; column + 16 <= up_end; column += 16) {
        const __m512 values = _mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_loadu_si512(source + column)),
            _mm512_loadu_ps(scales_ + column));
        _mm512_storeu_ps(destination + column, values);
      }
#endif
      for (; column + 8 <= up_end; column += 8) {
        const __m256 values = _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(source + column))),
            _mm256_loadu_ps(scales_ + column));
        _mm256_storeu_ps(destination + column, values);
      }
      for (; column < up_end; ++column) {
        destination[column] = static_cast<float>(source[column]) * scales_[column];
      }

      column = std::max(start_column, kFfn);
#if defined(KTCP_POSTOPS_AVX512)
      for (; column + 16 <= end_column; column += 16) {
        const size_t channel = column - kFfn;
        const __m512 up = _mm512_loadu_ps(destination + channel);
        const __m512 gate_unclipped = _mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_loadu_si512(source + column)),
            _mm512_loadu_ps(scales_ + column));
        const __m512 clip_min = _mm512_set1_ps(-kSwiGluClip);
        const __m512 clip_max = _mm512_set1_ps(kSwiGluClip);
        const __m512 gate =
            _mm512_min_ps(clip_max, _mm512_max_ps(clip_min, gate_unclipped));
        const __m512 sigmoid = _mm512_div_ps(
            _mm512_set1_ps(1.0f),
            _mm512_add_ps(
                _mm512_set1_ps(1.0f),
                exp512_post_ps(_mm512_sub_ps(_mm512_setzero_ps(), up))));
        const __m512 activated = _mm512_min_ps(
            clip_max,
            _mm512_max_ps(clip_min, _mm512_mul_ps(up, sigmoid)));
        _mm512_storeu_ps(
            destination + channel,
            _mm512_mul_ps(activated, gate));
      }
#endif
      for (; column + 8 <= end_column; column += 8) {
        const size_t channel = column - kFfn;
        const __m256 up = _mm256_loadu_ps(destination + channel);
        const __m256 gate_unclipped = _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(source + column))),
            _mm256_loadu_ps(scales_ + column));
        const __m256 clip_min = _mm256_set1_ps(-kSwiGluClip);
        const __m256 clip_max = _mm256_set1_ps(kSwiGluClip);
        const __m256 gate =
            _mm256_min_ps(clip_max, _mm256_max_ps(clip_min, gate_unclipped));
        const __m256 sigmoid = _mm256_div_ps(
            _mm256_set1_ps(1.0f),
            _mm256_add_ps(
                _mm256_set1_ps(1.0f),
                exp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), up))));
        const __m256 activated = _mm256_min_ps(
            clip_max,
            _mm256_max_ps(clip_min, _mm256_mul_ps(up, sigmoid)));
        _mm256_storeu_ps(
            destination + channel,
            _mm256_mul_ps(activated, gate));
      }
      for (; column < end_column; ++column) {
        const size_t channel = column - kFfn;
        const float up = destination[channel];
        const float gate = std::clamp(
            static_cast<float>(source[column]) * scales_[column],
            -kSwiGluClip, kSwiGluClip);
        const float activated = std::clamp(
            up / (1.0f + std::exp(-up)), -kSwiGluClip, kSwiGluClip);
        destination[channel] = activated * gate;
      }
    }
  }

 private:
  const float* scales_ = nullptr;
  SwiGluOutput output_{};
};

class ResidualOutputProcessor final : public MLAS_QGEMM_OUTPUT_PROCESSOR {
 public:
  ResidualOutputProcessor(const float* scales, ResidualOutput output)
      : scales_(scales), output_(output) {}

  void Process(
      const int32_t* accumulators,
      size_t start_row,
      size_t start_column,
      size_t row_count,
      size_t column_count,
      size_t accumulator_stride) const override {
    // Both supported residual widths fit inside every qualified packed-N
    // panel. Every callback therefore owns a full row, which permits
    // residual/RMSNorm fusion without cross-tile state.
    if (start_column != 0 || column_count != kC) return;
    for (size_t row = start_row; row < start_row + row_count; ++row) {
      const int32_t* source = accumulators + row * accumulator_stride;
      float* residual = output_.residual + row * kC;
#if defined(KTCP_POSTOPS_AVX512)
      __m512 squares512 = _mm512_setzero_ps();
      for (size_t channel = 0; channel < kC; channel += 16) {
        const __m512 update = _mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_loadu_si512(source + channel)),
            _mm512_loadu_ps(scales_ + channel));
        const __m512 value =
            _mm512_add_ps(_mm512_loadu_ps(residual + channel), update);
        _mm512_storeu_ps(residual + channel, value);
        if (output_.normalizes()) {
          squares512 = _mm512_fmadd_ps(value, value, squares512);
        }
      }
      if (output_.normalizes()) {
        const float inv_rms = 1.0f / std::sqrt(
            _mm512_reduce_add_ps(squares512) / static_cast<float>(kC) +
            kRmsEpsilon);
        const __m512 scale = _mm512_set1_ps(inv_rms);
        float* normalized = output_.normalized + row * kC;
        for (size_t channel = 0; channel < kC; channel += 16) {
          const __m512 value = _mm512_mul_ps(
              _mm512_loadu_ps(residual + channel),
              _mm512_loadu_ps(output_.gamma + channel));
          _mm512_storeu_ps(
              normalized + channel, _mm512_mul_ps(value, scale));
        }
      }
#else
      __m256 squares = _mm256_setzero_ps();
      for (size_t channel = 0; channel < kC; channel += 8) {
        const __m256 update = _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(source + channel))),
            _mm256_loadu_ps(scales_ + channel));
        const __m256 value =
            _mm256_add_ps(_mm256_loadu_ps(residual + channel), update);
        _mm256_storeu_ps(residual + channel, value);
        if (output_.normalizes()) {
          squares = _mm256_fmadd_ps(value, value, squares);
        }
      }
      if (output_.normalizes()) {
        const float inv_rms = 1.0f / std::sqrt(
            horizontal_sum(squares) / static_cast<float>(kC) + kRmsEpsilon);
        const __m256 scale = _mm256_set1_ps(inv_rms);
        float* normalized = output_.normalized + row * kC;
        for (size_t channel = 0; channel < kC; channel += 8) {
          const __m256 value = _mm256_mul_ps(
              _mm256_loadu_ps(residual + channel),
              _mm256_loadu_ps(output_.gamma + channel));
          _mm256_storeu_ps(
              normalized + channel, _mm256_mul_ps(value, scale));
        }
      }
#endif
    }
  }

 private:
  const float* scales_ = nullptr;
  ResidualOutput output_{};
};

class FullyConnected {
 public:
  FullyConnected(
      int mode,
      size_t rows,
      size_t input_channels,
      size_t output_channels,
      std::vector<float> weights,
      std::vector<float> bias,
      const float* input,
      float* output,
      QkvRopeOutput qkv_rope_output = {},
      SwiGluOutput swiglu_output = {},
      ResidualOutput residual_output = {})
      : mode_(mode),
        rows_(rows),
        input_channels_(input_channels),
        output_channels_(output_channels),
        weights_(std::move(weights)),
        bias_(std::move(bias)) {
    if (weights_.size() != input_channels_ * output_channels_ ||
        bias_.size() != output_channels_) {
      throw std::runtime_error("FullyConnected weight/bias size mismatch");
    }

    if (mode_ == KTCP_MODE_FP32) {
      xnn_operator_t raw = nullptr;
      check_xnn(
          xnn_create_fully_connected_nc_f32(
              input_channels_, output_channels_, input_channels_, output_channels_,
              weights_.data(), bias_.data(),
              -std::numeric_limits<float>::infinity(),
              std::numeric_limits<float>::infinity(), 0, nullptr, nullptr, &raw),
          "xnn_create_fully_connected_nc_f32");
      op_ = Operator(raw);
      check_xnn(
          xnn_reshape_fully_connected_nc_f32(op_.get(), rows_, nullptr),
          "xnn_reshape_fully_connected_nc_f32");
      check_xnn(
          xnn_setup_fully_connected_nc_f32(op_.get(), input, output),
          "xnn_setup_fully_connected_nc_f32");
      return;
    }
    if (mode_ == KTCP_MODE_MLAS_U8S8) {
      throw std::runtime_error(
          "MLAS projection requires canonical v106 S8 weights");
    }
    if (mode_ != KTCP_MODE_QD8) {
      throw std::runtime_error("Unsupported projection mode");
    }

    quantized_input_ = padded_vector<int8_t>(rows_ * input_channels_);
    quantization_params_.resize(rows_ + XNN_EXTRA_QUANTIZATION_PARAMS);
    quantized_weights_ = padded_vector<int8_t>(weights_.size());
    weight_scales_.assign(output_channels_ + XNN_EXTRA_QUANTIZATION_PARAMS, 1.0f);
    for (size_t output_channel = 0; output_channel < output_channels_; ++output_channel) {
      float max_abs = 0.0f;
      const float* source = weights_.data() + output_channel * input_channels_;
      for (size_t input_channel = 0; input_channel < input_channels_; ++input_channel) {
        max_abs = std::max(max_abs, std::abs(source[input_channel]));
      }
      const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
      weight_scales_[output_channel] = scale;
      int8_t* destination =
          quantized_weights_.data() + output_channel * input_channels_;
      for (size_t input_channel = 0; input_channel < input_channels_; ++input_channel) {
        const int rounded = static_cast<int>(std::nearbyint(source[input_channel] / scale));
        destination[input_channel] =
            static_cast<int8_t>(std::clamp(rounded, -127, 127));
      }
    }

    xnn_operator_t raw_convert = nullptr;
    check_xnn(
        xnn_create_convert_nc_f32_qd8(0, &raw_convert),
        "xnn_create_convert_nc_f32_qd8");
    convert_ = Operator(raw_convert);
    check_xnn(
        xnn_reshape_convert_nc_f32_qd8(
            convert_.get(), rows_, input_channels_, input_channels_, input_channels_, nullptr),
        "xnn_reshape_convert_nc_f32_qd8");
    check_xnn(
        xnn_setup_convert_nc_f32_qd8(
            convert_.get(), input, quantized_input_.data(), quantization_params_.data()),
        "xnn_setup_convert_nc_f32_qd8");

    xnn_operator_t raw_fc = nullptr;
    check_xnn(
        xnn_create_fully_connected_nc_qd8_f32_qc8w(
            input_channels_, output_channels_, input_channels_, output_channels_,
            weight_scales_.data(), quantized_weights_.data(), bias_.data(),
            -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity(), 0, nullptr, nullptr, &raw_fc),
        "xnn_create_fully_connected_nc_qd8_f32_qc8w");
    op_ = Operator(raw_fc);
    check_xnn(
        xnn_reshape_fully_connected_nc_qd8_f32_qc8w(op_.get(), rows_, nullptr),
        "xnn_reshape_fully_connected_nc_qd8_f32_qc8w");
    check_xnn(
        xnn_setup_fully_connected_nc_qd8_f32_qc8w(
            op_.get(), quantized_input_.data(), output, quantization_params_.data()),
        "xnn_setup_fully_connected_nc_qd8_f32_qc8w");
  }

  FullyConnected(
      int mode,
      size_t rows,
      size_t input_channels,
      size_t output_channels,
      CanonicalS8Weights weights,
      std::vector<float> bias,
      const float* input,
      float* output,
      QkvRopeOutput qkv_rope_output = {},
      SwiGluOutput swiglu_output = {},
      ResidualOutput residual_output = {})
      : mode_(mode),
        rows_(rows),
        input_channels_(input_channels),
        output_channels_(output_channels),
        bias_(std::move(bias)) {
    if (mode_ != KTCP_MODE_MLAS_U8S8) {
      throw std::runtime_error("Canonical S8 weights require the MLAS U8S8 path");
    }
    if (weights.values.size() != input_channels_ * output_channels_ ||
        weights.scales.size() != output_channels_ ||
        bias_.size() != output_channels_) {
      throw std::runtime_error("Canonical S8 weight/bias size mismatch");
    }
    for (float scale : weights.scales) {
      if (!(scale > 0.0f) || !std::isfinite(scale)) {
        throw std::runtime_error("Canonical S8 weight scale is invalid");
      }
    }
    for (int8_t value : weights.values) {
      if (value == std::numeric_limits<int8_t>::min()) {
        throw std::runtime_error("Canonical symmetric S8 weight contains -128");
      }
    }
    initialize_mlas(
        std::move(weights), input, output,
        qkv_rope_output, swiglu_output, residual_output);
  }

  void run() {
    if (mode_ == KTCP_MODE_MLAS_U8S8) {
      float minimum = 0.0f;
      float maximum = 0.0f;
      MlasFindMinMaxElement(
          mlas_input_f32_, &minimum, &maximum, rows_ * input_channels_);
      minimum = std::min(minimum, 0.0f);
      maximum = std::max(maximum, 0.0f);
      float input_scale = (maximum - minimum) / 255.0f;
      if (!(input_scale > 0.0f) || !std::isfinite(input_scale)) {
        input_scale = 1.0f;
      }
      const int zero_point_int = std::clamp(
          static_cast<int>(std::nearbyint(-minimum / input_scale)), 0, 255);
      const uint8_t zero_point = static_cast<uint8_t>(zero_point_int);
      MlasQuantizeLinear<uint8_t>(
          mlas_input_f32_, mlas_input_.data(), rows_ * input_channels_,
          input_scale, zero_point);
      for (size_t channel = 0; channel < output_channels_; ++channel) {
        mlas_output_scales_[channel] = input_scale * weight_scales_[channel];
      }
      mlas_data_.ZeroPointA = zero_point;
      MlasGemm(mlas_shape_, mlas_data_, nullptr);
      return;
    }
    if (mode_ == KTCP_MODE_QD8) {
      check_xnn(xnn_run_operator(convert_.get(), nullptr), "xnn_run_operator(convert)");
    }
    check_xnn(xnn_run_operator(op_.get(), nullptr), "xnn_run_operator(fully_connected)");
  }

 private:
  void initialize_mlas(
      CanonicalS8Weights weights,
      const float* input,
      float* output,
      QkvRopeOutput qkv_rope_output,
      SwiGluOutput swiglu_output,
      ResidualOutput residual_output) {
    mlas_input_f32_ = input;
    mlas_input_ = padded_vector<uint8_t>(rows_ * input_channels_);
    mlas_accumulator_ = padded_vector<int32_t>(rows_ * output_channels_);
    weight_scales_ = std::move(weights.scales);
    std::vector<int8_t> weights_kn(input_channels_ * output_channels_);
    for (size_t output_channel = 0; output_channel < output_channels_; ++output_channel) {
      for (size_t input_channel = 0; input_channel < input_channels_; ++input_channel) {
        weights_kn[input_channel * output_channels_ + output_channel] =
            weights.values[output_channel * input_channels_ + input_channel];
      }
    }

    const size_t packed_size = MlasGemmPackBSize(
        output_channels_, input_channels_, false, true);
    if (packed_size == 0) {
      throw std::runtime_error("MLAS U8S8 packed GEMM is unavailable on this CPU");
    }
    mlas_packed_weights_.resize(packed_size);
    MlasGemmPackB(
        output_channels_, input_channels_,
        reinterpret_cast<const uint8_t*>(weights_kn.data()), output_channels_,
        false, true, mlas_packed_weights_.data());

    mlas_output_scales_.assign(output_channels_, 1.0f);
    if (qkv_rope_output.enabled()) {
      if (rows_ != kS || input_channels_ != kC || output_channels_ != 3 * kC) {
        throw std::runtime_error("QKV/RoPE fusion received an incompatible shape");
      }
      mlas_output_processor_ = std::make_unique<QkvRopeOutputProcessor>(
          mlas_output_scales_.data(), qkv_rope_output);
    } else if (swiglu_output.enabled()) {
      if (rows_ != kS || input_channels_ != kC || output_channels_ != 2 * kFfn) {
        throw std::runtime_error("SwiGLU fusion received an incompatible shape");
      }
      mlas_output_processor_ = std::make_unique<SwiGluOutputProcessor>(
          mlas_output_scales_.data(), swiglu_output);
    } else if (residual_output.enabled()) {
      if (rows_ != kS || output_channels_ != kC) {
        throw std::runtime_error("Residual fusion received an incompatible shape");
      }
      mlas_output_processor_ = std::make_unique<ResidualOutputProcessor>(
          mlas_output_scales_.data(), residual_output);
    } else {
      mlas_output_processor_ =
          std::make_unique<MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR>(
              output, output_channels_, mlas_output_scales_.data(), bias_.data(),
              MLAS_QGEMM_OUTPUT_MODE::ZeroMode,
              MLAS_QUANTIZATION_GRANULARITY::PerColumn);
    }
    mlas_shape_.M = rows_;
    mlas_shape_.N = output_channels_;
    mlas_shape_.K = input_channels_;
    mlas_shape_.AIsSigned = false;
    mlas_shape_.BIsSigned = true;
    mlas_data_.A = mlas_input_.data();
    mlas_data_.lda = input_channels_;
    mlas_data_.B = mlas_packed_weights_.data();
    mlas_data_.ldb = output_channels_;
    mlas_data_.ZeroPointB = &mlas_weight_zero_point_;
    mlas_data_.BIsPacked = true;
    mlas_data_.C = mlas_accumulator_.data();
    mlas_data_.ldc = output_channels_;
    mlas_data_.OutputProcessor = mlas_output_processor_.get();
  }

  int mode_ = KTCP_MODE_FP32;
  size_t rows_ = 0;
  size_t input_channels_ = 0;
  size_t output_channels_ = 0;
  std::vector<float> weights_;
  std::vector<float> bias_;
  std::vector<int8_t> quantized_input_;
  std::vector<xnn_quantization_params> quantization_params_;
  std::vector<int8_t> quantized_weights_;
  std::vector<float> weight_scales_;
  const float* mlas_input_f32_ = nullptr;
  std::vector<uint8_t> mlas_input_;
  std::vector<uint8_t> mlas_packed_weights_;
  std::vector<int32_t> mlas_accumulator_;
  std::vector<float> mlas_output_scales_;
  uint8_t mlas_weight_zero_point_ = 0;
  MLAS_GEMM_QUANT_SHAPE_PARAMS mlas_shape_;
  MLAS_GEMM_QUANT_DATA_PARAMS mlas_data_;
  std::unique_ptr<MLAS_QGEMM_OUTPUT_PROCESSOR> mlas_output_processor_;
  Operator convert_;
  Operator op_;
};

class StemConvolution {
 public:
  StemConvolution(
      const std::vector<float>& pytorch_kernel,
      const float* input,
      float* output) {
    constexpr size_t input_channels = KTCP_SPATIAL_INPUTS;
    constexpr size_t output_channels = kC;
    if (pytorch_kernel.size() != output_channels * input_channels * 3 * 3) {
      throw std::runtime_error("Unexpected stem kernel size");
    }
    // XNNPACK uses [output, kernel_y, kernel_x, input] for an ungrouped NHWC conv.
    kernel_.resize(pytorch_kernel.size());
    for (size_t oc = 0; oc < output_channels; ++oc) {
      for (size_t ky = 0; ky < 3; ++ky) {
        for (size_t kx = 0; kx < 3; ++kx) {
          for (size_t ic = 0; ic < input_channels; ++ic) {
            const size_t source = ((oc * input_channels + ic) * 3 + ky) * 3 + kx;
            const size_t destination = ((oc * 3 + ky) * 3 + kx) * input_channels + ic;
            kernel_[destination] = pytorch_kernel[source];
          }
        }
      }
    }
    bias_.assign(output_channels, 0.0f);
    xnn_operator_t raw = nullptr;
    check_xnn(
        xnn_create_convolution2d_nhwc_f32(
            1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1,
            input_channels, output_channels, input_channels, output_channels,
            kernel_.data(), bias_.data(),
            -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity(), 0, nullptr, nullptr, &raw),
        "xnn_create_convolution2d_nhwc_f32");
    op_ = Operator(raw);
    size_t output_height = 0;
    size_t output_width = 0;
    check_xnn(
        xnn_reshape_convolution2d_nhwc_f32(
            op_.get(), 1, KTCP_BOARD_LEN, KTCP_BOARD_LEN,
            &workspace_size_, &workspace_alignment_, &output_height, &output_width, nullptr),
        "xnn_reshape_convolution2d_nhwc_f32");
    if (output_height != KTCP_BOARD_LEN || output_width != KTCP_BOARD_LEN) {
      throw std::runtime_error("Stem convolution produced an unexpected output shape");
    }
    if (workspace_size_ != 0) {
#ifdef _WIN32
      workspace_ = _aligned_malloc(
          workspace_size_, std::max<size_t>(workspace_alignment_, alignof(void*)));
#else
      workspace_ = std::aligned_alloc(workspace_alignment_, workspace_size_);
#endif
      if (workspace_ == nullptr) {
        throw std::bad_alloc();
      }
    }
    check_xnn(
        xnn_setup_convolution2d_nhwc_f32(op_.get(), workspace_, input, output),
        "xnn_setup_convolution2d_nhwc_f32");
  }

  ~StemConvolution() {
#ifdef _WIN32
    _aligned_free(workspace_);
#else
    std::free(workspace_);
#endif
  }

  void run() const {
    check_xnn(xnn_run_operator(op_.get(), nullptr), "xnn_run_operator(stem_conv)");
  }

 private:
  std::vector<float> kernel_;
  std::vector<float> bias_;
  Operator op_;
  size_t workspace_size_ = 0;
  size_t workspace_alignment_ = 1;
  void* workspace_ = nullptr;
};

inline float horizontal_sum(__m256 value) {
  const __m128 low = _mm256_castps256_ps128(value);
  const __m128 high = _mm256_extractf128_ps(value, 1);
  __m128 sum = _mm_add_ps(low, high);
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  return _mm_cvtss_f32(sum);
}

inline float horizontal_max(__m256 value) {
  const __m128 low = _mm256_castps256_ps128(value);
  const __m128 high = _mm256_extractf128_ps(value, 1);
  __m128 maximum = _mm_max_ps(low, high);
  maximum = _mm_max_ps(maximum, _mm_movehl_ps(maximum, maximum));
  maximum = _mm_max_ss(maximum, _mm_shuffle_ps(maximum, maximum, 1));
  return _mm_cvtss_f32(maximum);
}

// AVX2 Cephes-style exponential. Inputs in this runtime are modest, but the
// clamp also makes it safe for arbitrary SwiGLU activations and softmax tails.
inline __m256 exp256_ps(__m256 x) {
  const __m256 one = _mm256_set1_ps(1.0f);
  x = _mm256_min_ps(x, _mm256_set1_ps(88.3762626647949f));
  x = _mm256_max_ps(x, _mm256_set1_ps(-88.3762626647949f));

  __m256 fx = _mm256_fmadd_ps(x, _mm256_set1_ps(1.44269504088896341f),
                              _mm256_set1_ps(0.5f));
  __m256 tmp = _mm256_cvtepi32_ps(_mm256_cvttps_epi32(fx));
  __m256 mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
  mask = _mm256_and_ps(mask, one);
  fx = _mm256_sub_ps(tmp, mask);

  x = _mm256_sub_ps(x, _mm256_mul_ps(fx, _mm256_set1_ps(0.693359375f)));
  x = _mm256_sub_ps(x, _mm256_mul_ps(fx, _mm256_set1_ps(-2.12194440e-4f)));
  const __m256 z = _mm256_mul_ps(x, x);

  __m256 y = _mm256_set1_ps(1.9875691500e-4f);
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507e-3f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073e-3f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894e-2f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459e-1f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201e-1f));
  y = _mm256_fmadd_ps(y, z, x);
  y = _mm256_add_ps(y, one);

  __m256i exponent = _mm256_cvttps_epi32(fx);
  exponent = _mm256_add_epi32(exponent, _mm256_set1_epi32(0x7f));
  exponent = _mm256_slli_epi32(exponent, 23);
  return _mm256_mul_ps(y, _mm256_castsi256_ps(exponent));
}

#if defined(KTCP_POSTOPS_AVX512)
inline __m512 exp512_post_ps(__m512 x) {
  const __m512 one = _mm512_set1_ps(1.0f);
  x = _mm512_min_ps(x, _mm512_set1_ps(88.3762626647949f));
  x = _mm512_max_ps(x, _mm512_set1_ps(-88.3762626647949f));
  __m512 fx = _mm512_fmadd_ps(
      x, _mm512_set1_ps(1.44269504088896341f), _mm512_set1_ps(0.5f));
  const __m512 truncated = _mm512_cvtepi32_ps(_mm512_cvttps_epi32(fx));
  const __mmask16 correction = _mm512_cmp_ps_mask(truncated, fx, _CMP_GT_OS);
  fx = _mm512_mask_sub_ps(truncated, correction, truncated, one);
  x = _mm512_sub_ps(x, _mm512_mul_ps(fx, _mm512_set1_ps(0.693359375f)));
  x = _mm512_sub_ps(x, _mm512_mul_ps(fx, _mm512_set1_ps(-2.12194440e-4f)));
  const __m512 z = _mm512_mul_ps(x, x);
  __m512 y = _mm512_set1_ps(1.9875691500e-4f);
  y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.3981999507e-3f));
  y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(8.3334519073e-3f));
  y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(4.1665795894e-2f));
  y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.6666665459e-1f));
  y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(5.0000001201e-1f));
  y = _mm512_fmadd_ps(y, z, x);
  y = _mm512_add_ps(y, one);
  __m512i exponent = _mm512_cvttps_epi32(fx);
  exponent = _mm512_add_epi32(exponent, _mm512_set1_epi32(0x7f));
  exponent = _mm512_slli_epi32(exponent, 23);
  return _mm512_mul_ps(y, _mm512_castsi512_ps(exponent));
}
#endif

void rmsnorm(
    const float* input,
    float* output,
    const float* gamma) {
  for (size_t token = 0; token < kS; ++token) {
    const float* source = input + token * kC;
    float* destination = output + token * kC;
    __m256 squares = _mm256_setzero_ps();
    for (size_t channel = 0; channel < kC; channel += 8) {
      const __m256 value = _mm256_loadu_ps(source + channel);
      squares = _mm256_fmadd_ps(value, value, squares);
    }
    const float inv_rms =
        1.0f / std::sqrt(horizontal_sum(squares) / static_cast<float>(kC) + kRmsEpsilon);
    const __m256 scale = _mm256_set1_ps(inv_rms);
    for (size_t channel = 0; channel < kC; channel += 8) {
      __m256 value = _mm256_loadu_ps(source + channel);
      value = _mm256_mul_ps(value, _mm256_loadu_ps(gamma + channel));
      _mm256_storeu_ps(destination + channel, _mm256_mul_ps(value, scale));
    }
  }
}

void residual_rmsnorm(
    float* residual,
    const float* update,
    float* normalized,
    const float* gamma) {
  for (size_t token = 0; token < kS; ++token) {
    float* x = residual + token * kC;
    const float* y = update + token * kC;
    float* destination = normalized + token * kC;
    __m256 squares = _mm256_setzero_ps();
    for (size_t channel = 0; channel < kC; channel += 8) {
      const __m256 value =
          _mm256_add_ps(_mm256_loadu_ps(x + channel), _mm256_loadu_ps(y + channel));
      _mm256_storeu_ps(x + channel, value);
      squares = _mm256_fmadd_ps(value, value, squares);
    }
    const float inv_rms =
        1.0f / std::sqrt(horizontal_sum(squares) / static_cast<float>(kC) + kRmsEpsilon);
    const __m256 scale = _mm256_set1_ps(inv_rms);
    for (size_t channel = 0; channel < kC; channel += 8) {
      __m256 value = _mm256_loadu_ps(x + channel);
      value = _mm256_mul_ps(value, _mm256_loadu_ps(gamma + channel));
      _mm256_storeu_ps(destination + channel, _mm256_mul_ps(value, scale));
    }
  }
}

void add_in_place(float* destination, const float* source, size_t elements) {
  size_t index = 0;
  for (; index + 8 <= elements; index += 8) {
    _mm256_storeu_ps(
        destination + index,
        _mm256_add_ps(
            _mm256_loadu_ps(destination + index),
            _mm256_loadu_ps(source + index)));
  }
  for (; index < elements; ++index) {
    destination[index] += source[index];
  }
}

class BatchedAttention {
 public:
  BatchedAttention(
      const float* q,
      const float* k,
      const float* k_transposed,
      const float* v,
      float* workspace,
      float* output_tokens)
      : q_(q),
        k_(k),
        k_transposed_(k_transposed),
        v_(v),
        workspace_(workspace),
        output_tokens_(output_tokens) {
    flash_.batch_size = 1;
    flash_.num_heads = static_cast<int>(kHeads);
    flash_.q_sequence_length = static_cast<int>(kS);
    flash_.kv_sequence_length = static_cast<int>(kS);
    flash_.qk_head_size = static_cast<int>(kHeadDim);
    flash_.v_head_size = static_cast<int>(kHeadDim);
    flash_.q_block_size = static_cast<int>(kAttentionQBlock);
    flash_.kv_block_size = static_cast<int>(kAttentionKvBlock);
    flash_.scale = kAttentionScale;
    flash_.thread_count = 1;
    flash_.buffer = workspace_;
    flash_.buffer_size_per_thread =
        (2 * kAttentionQBlock + kAttentionQBlock * kAttentionKvBlock +
         kAttentionQBlock * kHeadDim) * sizeof(float);
    flash_.query = q_;
    flash_.key = k_;
    flash_.value = v_;
    flash_.output = output_tokens_;
  }

  void run(Profile* profile = nullptr) const {
#if defined(KTCP_ATTENTION_VNNI)
    run_vnni(profile);
#else
    auto started = Clock::now();
#if defined(KTCP_ATTENTION_AVX512)
    run_avx512();
#else
    MlasFlashAttention(const_cast<MlasFlashAttentionThreadedArgs*>(&flash_), nullptr);
#endif
    auto ended = Clock::now();
    if (profile != nullptr) profile->add(kProfileAttentionQk, started, ended);
#endif
  }

 private:
#if defined(KTCP_ATTENTION_AVX512) || defined(KTCP_ATTENTION_VNNI)
  static __forceinline __m512 exp512_ps(__m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f);
    x = _mm512_min_ps(x, _mm512_set1_ps(88.3762626647949f));
    x = _mm512_max_ps(x, _mm512_set1_ps(-88.3762626647949f));

    __m512 fx = _mm512_fmadd_ps(
        x, _mm512_set1_ps(1.44269504088896341f), _mm512_set1_ps(0.5f));
    const __m512 truncated = _mm512_cvtepi32_ps(_mm512_cvttps_epi32(fx));
    const __mmask16 correction =
        _mm512_cmp_ps_mask(truncated, fx, _CMP_GT_OS);
    fx = _mm512_mask_sub_ps(truncated, correction, truncated, one);

    x = _mm512_sub_ps(x, _mm512_mul_ps(fx, _mm512_set1_ps(0.693359375f)));
    x = _mm512_sub_ps(x, _mm512_mul_ps(fx, _mm512_set1_ps(-2.12194440e-4f)));
    const __m512 z = _mm512_mul_ps(x, x);

    __m512 y = _mm512_set1_ps(1.9875691500e-4f);
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.3981999507e-3f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(8.3334519073e-3f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(4.1665795894e-2f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.6666665459e-1f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(5.0000001201e-1f));
    y = _mm512_fmadd_ps(y, z, x);
    y = _mm512_add_ps(y, one);

    __m512i exponent = _mm512_cvttps_epi32(fx);
    exponent = _mm512_add_epi32(exponent, _mm512_set1_epi32(0x7f));
    exponent = _mm512_slli_epi32(exponent, 23);
    return _mm512_mul_ps(y, _mm512_castsi512_ps(exponent));
  }
#endif

#if defined(KTCP_ATTENTION_AVX512)
  static __forceinline __m512 qk_block(
      const float* query,
      const float* transposed_key,
      size_t key_block) {
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();
    const float* key = transposed_key + key_block * 16;
    for (size_t channel = 0; channel < kHeadDim; channel += 4) {
      sum0 = _mm512_fmadd_ps(
          _mm512_set1_ps(query[channel]),
          _mm512_loadu_ps(key + channel * kAttentionPaddedS), sum0);
      sum1 = _mm512_fmadd_ps(
          _mm512_set1_ps(query[channel + 1]),
          _mm512_loadu_ps(key + (channel + 1) * kAttentionPaddedS), sum1);
      sum2 = _mm512_fmadd_ps(
          _mm512_set1_ps(query[channel + 2]),
          _mm512_loadu_ps(key + (channel + 2) * kAttentionPaddedS), sum2);
      sum3 = _mm512_fmadd_ps(
          _mm512_set1_ps(query[channel + 3]),
          _mm512_loadu_ps(key + (channel + 3) * kAttentionPaddedS), sum3);
    }
    return _mm512_mul_ps(
        _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3)),
        _mm512_set1_ps(kAttentionScale));
  }

  void run_avx512() const {
    constexpr size_t full_key_blocks = kS / 16;
    constexpr size_t tail_keys = kS % 16;
    constexpr __mmask16 tail_mask =
        static_cast<__mmask16>((uint32_t{1} << tail_keys) - 1u);
    alignas(64) float probabilities[kAttentionPaddedS]{};
    const __m512 negative_infinity =
        _mm512_set1_ps(-std::numeric_limits<float>::infinity());

    for (size_t head = 0; head < kHeads; ++head) {
      const float* head_q = q_ + head * kS * kHeadDim;
      const float* head_k =
          k_transposed_ + head * kHeadDim * kAttentionPaddedS;
      const float* head_v = v_ + head * kS * kHeadDim;
      for (size_t token = 0; token < kS; ++token) {
        const float* query = head_q + token * kHeadDim;
        __m512 maximum = negative_infinity;
        for (size_t block = 0; block < full_key_blocks; ++block) {
          const __m512 score = qk_block(query, head_k, block);
          _mm512_store_ps(probabilities + block * 16, score);
          maximum = _mm512_max_ps(maximum, score);
        }
        const __m512 tail_score = qk_block(query, head_k, full_key_blocks);
        _mm512_mask_store_ps(
            probabilities + full_key_blocks * 16, tail_mask, tail_score);
        maximum = _mm512_max_ps(
            maximum,
            _mm512_mask_mov_ps(negative_infinity, tail_mask, tail_score));
        const __m512 row_maximum = _mm512_set1_ps(_mm512_reduce_max_ps(maximum));

        __m512 exponential_sum = _mm512_setzero_ps();
        for (size_t block = 0; block < full_key_blocks; ++block) {
          const __m512 exponentials = exp512_ps(_mm512_sub_ps(
              _mm512_load_ps(probabilities + block * 16), row_maximum));
          _mm512_store_ps(probabilities + block * 16, exponentials);
          exponential_sum = _mm512_add_ps(exponential_sum, exponentials);
        }
        const __m512 tail_exponentials = _mm512_maskz_mov_ps(
            tail_mask,
            exp512_ps(_mm512_sub_ps(tail_score, row_maximum)));
        _mm512_mask_store_ps(
            probabilities + full_key_blocks * 16,
            tail_mask,
            tail_exponentials);
        exponential_sum = _mm512_add_ps(exponential_sum, tail_exponentials);
        const float inverse_sum = 1.0f / _mm512_reduce_add_ps(exponential_sum);

        __m512 output00 = _mm512_setzero_ps();
        __m512 output01 = _mm512_setzero_ps();
        __m512 output10 = _mm512_setzero_ps();
        __m512 output11 = _mm512_setzero_ps();
        __m512 output20 = _mm512_setzero_ps();
        __m512 output21 = _mm512_setzero_ps();
        __m512 output30 = _mm512_setzero_ps();
        __m512 output31 = _mm512_setzero_ps();
        size_t key_token = 0;
        for (; key_token + 4 <= kS; key_token += 4) {
          const float* value = head_v + key_token * kHeadDim;
          output00 = _mm512_fmadd_ps(
              _mm512_set1_ps(probabilities[key_token]),
              _mm512_loadu_ps(value), output00);
          output01 = _mm512_fmadd_ps(
              _mm512_set1_ps(probabilities[key_token]),
              _mm512_loadu_ps(value + 16), output01);
          output10 = _mm512_fmadd_ps(
              _mm512_set1_ps(probabilities[key_token + 1]),
              _mm512_loadu_ps(value + kHeadDim), output10);
          output11 = _mm512_fmadd_ps(
              _mm512_set1_ps(probabilities[key_token + 1]),
              _mm512_loadu_ps(value + kHeadDim + 16), output11);
          output20 = _mm512_fmadd_ps(
              _mm512_set1_ps(probabilities[key_token + 2]),
              _mm512_loadu_ps(value + 2 * kHeadDim), output20);
          output21 = _mm512_fmadd_ps(
              _mm512_set1_ps(probabilities[key_token + 2]),
              _mm512_loadu_ps(value + 2 * kHeadDim + 16), output21);
          output30 = _mm512_fmadd_ps(
              _mm512_set1_ps(probabilities[key_token + 3]),
              _mm512_loadu_ps(value + 3 * kHeadDim), output30);
          output31 = _mm512_fmadd_ps(
              _mm512_set1_ps(probabilities[key_token + 3]),
              _mm512_loadu_ps(value + 3 * kHeadDim + 16), output31);
        }
        __m512 output0 =
            _mm512_add_ps(_mm512_add_ps(output00, output10),
                          _mm512_add_ps(output20, output30));
        __m512 output1 =
            _mm512_add_ps(_mm512_add_ps(output01, output11),
                          _mm512_add_ps(output21, output31));
        for (; key_token < kS; ++key_token) {
          const __m512 probability = _mm512_set1_ps(probabilities[key_token]);
          const float* value = head_v + key_token * kHeadDim;
          output0 = _mm512_fmadd_ps(
              probability, _mm512_loadu_ps(value), output0);
          output1 = _mm512_fmadd_ps(
              probability, _mm512_loadu_ps(value + 16), output1);
        }
        const __m512 normalization = _mm512_set1_ps(inverse_sum);
        float* destination =
            output_tokens_ + (token * kHeads + head) * kHeadDim;
        _mm512_storeu_ps(destination, _mm512_mul_ps(output0, normalization));
        _mm512_storeu_ps(destination + 16, _mm512_mul_ps(output1, normalization));
      }
    }
  }
#endif

#if defined(KTCP_ATTENTION_VNNI)
  static float quantize_s8_32(const float* source, int8_t* destination) {
    return quantize_s8x32_avx512(
        _mm512_loadu_ps(source),
        _mm512_loadu_ps(source + 16),
        destination);
  }

  void pack_vnni_keys() const {
    constexpr size_t key_blocks = (kS + 15) / 16;
    constexpr size_t groups = kHeadDim / 4;
    for (size_t head = 0; head < kHeads; ++head) {
      const float* head_key = k_ + head * kS * kHeadDim;
      for (size_t token = 0; token < kS; ++token) {
        alignas(16) int8_t quantized[kHeadDim];
        const float scale =
            quantize_s8_32(head_key + token * kHeadDim, quantized);
        const size_t padded_token = head * kAttentionPaddedS + token;
        vnni_key_scales_[padded_token] = scale;
        int32_t sum = 0;
        for (size_t channel = 0; channel < kHeadDim; ++channel) {
          sum += quantized[channel];
        }
        vnni_key_sums_[padded_token] = sum;
        const size_t block = token / 16;
        const size_t lane = token % 16;
        int8_t* packed_head = vnni_packed_keys_.data() +
            head * key_blocks * groups * 64;
        for (size_t group = 0; group < groups; ++group) {
          std::memcpy(
              packed_head + (block * groups + group) * 64 + lane * 4,
              quantized + group * 4,
              4);
        }
      }
    }
  }

  static float prepare_vnni_query(
      const float* query,
      __m512i* broadcasts) {
    constexpr size_t groups = kHeadDim / 4;
    alignas(16) int8_t quantized_query[kHeadDim];
    const float scale = quantize_s8_32(query, quantized_query);
    for (size_t group = 0; group < groups; ++group) {
      uint32_t pattern;
      std::memcpy(&pattern, quantized_query + group * 4, 4);
      broadcasts[group] = _mm512_set1_epi32(
          static_cast<int32_t>(pattern ^ 0x80808080u));
    }
    return scale;
  }

  static float exponentiate_vnni_score_row(
      float* scores,
      __m512 maximum,
      __m512 tail_score) {
    constexpr size_t full_key_blocks = kS / 16;
    constexpr size_t tail_keys = kS % 16;
    constexpr __mmask16 tail_mask =
        static_cast<__mmask16>((uint32_t{1} << tail_keys) - 1u);
    const __m512 row_maximum =
        _mm512_set1_ps(_mm512_reduce_max_ps(maximum));
    __m512 exponential_sum = _mm512_setzero_ps();
    for (size_t block = 0; block < full_key_blocks; ++block) {
      const __m512 exponentials = exp512_ps(_mm512_sub_ps(
          _mm512_loadu_ps(scores + block * 16), row_maximum));
      _mm512_storeu_ps(scores + block * 16, exponentials);
      exponential_sum = _mm512_add_ps(exponential_sum, exponentials);
    }
    const __m512 tail_exponentials = _mm512_maskz_mov_ps(
        tail_mask,
        exp512_ps(_mm512_sub_ps(tail_score, row_maximum)));
    _mm512_mask_storeu_ps(
        scores + full_key_blocks * 16, tail_mask, tail_exponentials);
    exponential_sum = _mm512_add_ps(exponential_sum, tail_exponentials);
    return 1.0f / _mm512_reduce_add_ps(exponential_sum);
  }

  static void normalize_vnni_score_row(float* scores, float inverse_sum) {
    constexpr size_t full_key_blocks = kS / 16;
    constexpr size_t tail_keys = kS % 16;
    constexpr __mmask16 tail_mask =
        static_cast<__mmask16>((uint32_t{1} << tail_keys) - 1u);
    const __m512 scale = _mm512_set1_ps(inverse_sum);
    for (size_t block = 0; block < full_key_blocks; ++block) {
      _mm512_storeu_ps(
          scores + block * 16,
          _mm512_mul_ps(_mm512_loadu_ps(scores + block * 16), scale));
    }
    _mm512_mask_storeu_ps(
        scores + full_key_blocks * 16,
        tail_mask,
        _mm512_mul_ps(
            _mm512_maskz_loadu_ps(
                tail_mask, scores + full_key_blocks * 16),
            scale));
  }

#if defined(KTCP_ATTENTION_PV_INT8)
  static __forceinline void pack_four_s8x16_rows(
      __m128i row0,
      __m128i row1,
      __m128i row2,
      __m128i row3,
      int8_t* destination) {
    const __m128i pairs01_low = _mm_unpacklo_epi8(row0, row1);
    const __m128i pairs01_high = _mm_unpackhi_epi8(row0, row1);
    const __m128i pairs23_low = _mm_unpacklo_epi8(row2, row3);
    const __m128i pairs23_high = _mm_unpackhi_epi8(row2, row3);
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(destination),
        _mm_unpacklo_epi16(pairs01_low, pairs23_low));
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(destination + 16),
        _mm_unpackhi_epi16(pairs01_low, pairs23_low));
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(destination + 32),
        _mm_unpacklo_epi16(pairs01_high, pairs23_high));
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(destination + 48),
        _mm_unpackhi_epi16(pairs01_high, pairs23_high));
  }

  void pack_vnni_values(const float* values) const {
    const __m512 absolute_mask =
        _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
    __m512 maximum0 = _mm512_setzero_ps();
    __m512 maximum1 = _mm512_setzero_ps();
    for (size_t token = 0; token < kS; ++token) {
      const float* value = values + token * kHeadDim;
      maximum0 = _mm512_max_ps(
          maximum0,
          _mm512_and_ps(_mm512_loadu_ps(value), absolute_mask));
      maximum1 = _mm512_max_ps(
          maximum1,
          _mm512_and_ps(_mm512_loadu_ps(value + 16), absolute_mask));
    }

    const __m512 zero = _mm512_setzero_ps();
    const __m512 quantized_maximum = _mm512_set1_ps(127.0f);
    const __m512 scale0 = _mm512_div_ps(maximum0, quantized_maximum);
    const __m512 scale1 = _mm512_div_ps(maximum1, quantized_maximum);
    _mm512_storeu_ps(vnni_value_scales_.data(), scale0);
    _mm512_storeu_ps(vnni_value_scales_.data() + 16, scale1);
    const __mmask16 nonzero0 =
        _mm512_cmp_ps_mask(maximum0, zero, _CMP_GT_OQ);
    const __mmask16 nonzero1 =
        _mm512_cmp_ps_mask(maximum1, zero, _CMP_GT_OQ);
    const __m512 inverse0 = _mm512_mask_div_ps(
        zero, nonzero0, quantized_maximum, maximum0);
    const __m512 inverse1 = _mm512_mask_div_ps(
        zero, nonzero1, quantized_maximum, maximum1);
    const __m512i minimum_i32 = _mm512_set1_epi32(-127);
    const __m512i maximum_i32 = _mm512_set1_epi32(127);

    constexpr size_t value_blocks = kHeadDim / 16;
    constexpr size_t key_groups = (kS + 3) / 4;
    for (size_t group = 0; group < key_groups; ++group) {
      __m128i rows[4][value_blocks];
      for (size_t byte = 0; byte < 4; ++byte) {
        const size_t token = group * 4 + byte;
        if (token < kS) {
          const float* value = values + token * kHeadDim;
          const __m512i quantized0 = _mm512_min_epi32(
              maximum_i32,
              _mm512_max_epi32(
                  minimum_i32,
                  _mm512_cvtps_epi32(
                      _mm512_mul_ps(_mm512_loadu_ps(value), inverse0))));
          const __m512i quantized1 = _mm512_min_epi32(
              maximum_i32,
              _mm512_max_epi32(
                  minimum_i32,
                  _mm512_cvtps_epi32(
                      _mm512_mul_ps(_mm512_loadu_ps(value + 16), inverse1))));
          rows[byte][0] = _mm512_cvtsepi32_epi8(quantized0);
          rows[byte][1] = _mm512_cvtsepi32_epi8(quantized1);
        } else {
          rows[byte][0] = _mm_setzero_si128();
          rows[byte][1] = _mm_setzero_si128();
        }
      }
      for (size_t value_block = 0; value_block < value_blocks; ++value_block) {
        pack_four_s8x16_rows(
            rows[0][value_block],
            rows[1][value_block],
            rows[2][value_block],
            rows[3][value_block],
            vnni_packed_values_.data() +
                (group * value_blocks + value_block) * 64);
      }
    }
  }

  static void quantize_vnni_probabilities(
      const float* exponentials,
      uint8_t* quantized_probabilities) {
    constexpr size_t full_key_blocks = kS / 16;
    constexpr size_t tail_keys = kS % 16;
    constexpr __mmask16 tail_mask =
        static_cast<__mmask16>((uint32_t{1} << tail_keys) - 1u);
    const __m512 probability_multiplier = _mm512_set1_ps(255.0f);
    for (size_t block = 0; block < full_key_blocks; ++block) {
      const __m512i quantized = _mm512_cvtps_epi32(_mm512_mul_ps(
          _mm512_loadu_ps(exponentials + block * 16),
          probability_multiplier));
      _mm_store_si128(
          reinterpret_cast<__m128i*>(quantized_probabilities + block * 16),
          _mm512_cvtusepi32_epi8(quantized));
    }
    const __m512 tail = _mm512_maskz_loadu_ps(
        tail_mask, exponentials + full_key_blocks * 16);
    const __m512i quantized_tail =
        _mm512_cvtps_epi32(_mm512_mul_ps(tail, probability_multiplier));
    _mm_store_si128(
        reinterpret_cast<__m128i*>(
            quantized_probabilities + full_key_blocks * 16),
        _mm512_cvtusepi32_epi8(quantized_tail));
  }

  void run_vnni_int8_pv_pair(
      const float* exponentials0,
      float inverse_sum0,
      float* output0,
      const float* exponentials1,
      float inverse_sum1,
      float* output1) const {
    constexpr size_t key_groups = (kS + 3) / 4;
    constexpr size_t value_blocks = kHeadDim / 16;
    alignas(64) uint8_t quantized_probabilities0[kAttentionPaddedS];
    alignas(64) uint8_t quantized_probabilities1[kAttentionPaddedS];
    quantize_vnni_probabilities(
        exponentials0, quantized_probabilities0);
    quantize_vnni_probabilities(
        exponentials1, quantized_probabilities1);

    const int8_t* packed_values = vnni_packed_values_.data();
    for (size_t value_block = 0; value_block < value_blocks; ++value_block) {
      __m512i accumulator00 = _mm512_setzero_si512();
      __m512i accumulator01 = _mm512_setzero_si512();
      __m512i accumulator02 = _mm512_setzero_si512();
      __m512i accumulator03 = _mm512_setzero_si512();
      __m512i accumulator10 = _mm512_setzero_si512();
      __m512i accumulator11 = _mm512_setzero_si512();
      __m512i accumulator12 = _mm512_setzero_si512();
      __m512i accumulator13 = _mm512_setzero_si512();
      size_t group = 0;
      for (; group + 4 <= key_groups; group += 4) {
        uint32_t patterns0[4];
        uint32_t patterns1[4];
        std::memcpy(patterns0, quantized_probabilities0 + group * 4, 16);
        std::memcpy(patterns1, quantized_probabilities1 + group * 4, 16);
        const int8_t* packed_group = packed_values +
            (group * value_blocks + value_block) * 64;
        const __m512i value0 = _mm512_loadu_si512(packed_group);
        accumulator00 = _mm512_dpbusd_epi32(
            accumulator00,
            _mm512_set1_epi32(static_cast<int32_t>(patterns0[0])), value0);
        accumulator10 = _mm512_dpbusd_epi32(
            accumulator10,
            _mm512_set1_epi32(static_cast<int32_t>(patterns1[0])), value0);
        const __m512i value1 =
            _mm512_loadu_si512(packed_group + value_blocks * 64);
        accumulator01 = _mm512_dpbusd_epi32(
            accumulator01,
            _mm512_set1_epi32(static_cast<int32_t>(patterns0[1])), value1);
        accumulator11 = _mm512_dpbusd_epi32(
            accumulator11,
            _mm512_set1_epi32(static_cast<int32_t>(patterns1[1])), value1);
        const __m512i value2 =
            _mm512_loadu_si512(packed_group + 2 * value_blocks * 64);
        accumulator02 = _mm512_dpbusd_epi32(
            accumulator02,
            _mm512_set1_epi32(static_cast<int32_t>(patterns0[2])), value2);
        accumulator12 = _mm512_dpbusd_epi32(
            accumulator12,
            _mm512_set1_epi32(static_cast<int32_t>(patterns1[2])), value2);
        const __m512i value3 =
            _mm512_loadu_si512(packed_group + 3 * value_blocks * 64);
        accumulator03 = _mm512_dpbusd_epi32(
            accumulator03,
            _mm512_set1_epi32(static_cast<int32_t>(patterns0[3])), value3);
        accumulator13 = _mm512_dpbusd_epi32(
            accumulator13,
            _mm512_set1_epi32(static_cast<int32_t>(patterns1[3])), value3);
      }
      if (group < key_groups) {
        uint32_t pattern0;
        uint32_t pattern1;
        std::memcpy(&pattern0, quantized_probabilities0 + group * 4, 4);
        std::memcpy(&pattern1, quantized_probabilities1 + group * 4, 4);
        const __m512i value = _mm512_loadu_si512(
            packed_values + (group * value_blocks + value_block) * 64);
        accumulator00 = _mm512_dpbusd_epi32(
            accumulator00,
            _mm512_set1_epi32(static_cast<int32_t>(pattern0)), value);
        accumulator10 = _mm512_dpbusd_epi32(
            accumulator10,
            _mm512_set1_epi32(static_cast<int32_t>(pattern1)), value);
      }
      const __m512i accumulator0 = _mm512_add_epi32(
          _mm512_add_epi32(accumulator00, accumulator01),
          _mm512_add_epi32(accumulator02, accumulator03));
      const __m512i accumulator1 = _mm512_add_epi32(
          _mm512_add_epi32(accumulator10, accumulator11),
          _mm512_add_epi32(accumulator12, accumulator13));
      const __m512 value_scale =
          _mm512_loadu_ps(vnni_value_scales_.data() + value_block * 16);
      _mm512_storeu_ps(
          output0 + value_block * 16,
          _mm512_mul_ps(
              _mm512_cvtepi32_ps(accumulator0),
              _mm512_mul_ps(
                  _mm512_set1_ps(inverse_sum0 / 255.0f), value_scale)));
      _mm512_storeu_ps(
          output1 + value_block * 16,
          _mm512_mul_ps(
              _mm512_cvtepi32_ps(accumulator1),
              _mm512_mul_ps(
                  _mm512_set1_ps(inverse_sum1 / 255.0f), value_scale)));
    }
  }
#endif

  void run_vnni(Profile* profile) const {
    constexpr size_t full_key_blocks = kS / 16;
    constexpr size_t key_blocks = (kS + 15) / 16;
    constexpr size_t groups = kHeadDim / 4;
    constexpr size_t tail_keys = kS % 16;
    constexpr __mmask16 tail_mask =
        static_cast<__mmask16>((uint32_t{1} << tail_keys) - 1u);
    auto pack_started = Clock::time_point{};
    if (profile != nullptr) pack_started = Clock::now();
    pack_vnni_keys();
    if (profile != nullptr) {
      profile->add(kProfileAttentionTranspose, pack_started, Clock::now());
    }
    float* probabilities = workspace_;
    const __m512 negative_infinity =
        _mm512_set1_ps(-std::numeric_limits<float>::infinity());

    for (size_t head = 0; head < kHeads; ++head) {
      const float* head_query = q_ + head * kS * kHeadDim;
      const float* head_value = v_ + head * kS * kHeadDim;
      const int8_t* packed_head = vnni_packed_keys_.data() +
          head * key_blocks * groups * 64;
      const float* key_scales =
          vnni_key_scales_.data() + head * kAttentionPaddedS;
      const int32_t* key_sums =
          vnni_key_sums_.data() + head * kAttentionPaddedS;
      // Heads are consumed immediately by P*V, so one 225x225 probability
      // matrix is reused instead of retaining every profile-specific head.
      float* head_probabilities = probabilities;
#if defined(KTCP_ATTENTION_PV_INT8)
      alignas(64) float inverse_sums[kS];
      auto value_pack_started = Clock::time_point{};
      if (profile != nullptr) value_pack_started = Clock::now();
      pack_vnni_values(head_value);
      if (profile != nullptr) {
        profile->add(
            kProfileAttentionTranspose, value_pack_started, Clock::now());
      }
#endif
      auto qk_started = Clock::time_point{};
      if (profile != nullptr) qk_started = Clock::now();

      static_assert(groups == 8, "The fixed VNNI QK unroll expects 32 channels");
      for (size_t token = 0; token < kS; token += 2) {
        const bool has_second = token + 1 < kS;
        const size_t second_token = has_second ? token + 1 : token;
        __m512i query_broadcasts0[groups];
        __m512i query_broadcasts1[groups];
        const float query_scale0 = prepare_vnni_query(
            head_query + token * kHeadDim, query_broadcasts0);
        const float query_scale1 = prepare_vnni_query(
            head_query + second_token * kHeadDim, query_broadcasts1);
        const __m512 query_multiplier0 =
            _mm512_set1_ps(query_scale0 * kAttentionScale);
        const __m512 query_multiplier1 =
            _mm512_set1_ps(query_scale1 * kAttentionScale);

        float* scores0 = head_probabilities + token * kS;
        // The VNNI workspace has room beyond the 225x225 score matrix; use one
        // padded row for the unpaired final query so the hot loop stays branchless.
        float* scores1 = has_second
            ? head_probabilities + second_token * kS
            : head_probabilities + kS * kS;
        __m512 maximum0 = negative_infinity;
        __m512 maximum1 = negative_infinity;
        __m512 tail_score0 = negative_infinity;
        __m512 tail_score1 = negative_infinity;
        for (size_t block = 0; block < key_blocks; ++block) {
          const int8_t* packed_block =
              packed_head + block * groups * 64;
          const __m512i key0 = _mm512_loadu_si512(packed_block);
          __m512i accumulator00 = _mm512_dpbusd_epi32(
              _mm512_setzero_si512(), query_broadcasts0[0], key0);
          __m512i accumulator10 = _mm512_dpbusd_epi32(
              _mm512_setzero_si512(), query_broadcasts1[0], key0);
          const __m512i key1 = _mm512_loadu_si512(packed_block + 64);
          __m512i accumulator01 = _mm512_dpbusd_epi32(
              _mm512_setzero_si512(), query_broadcasts0[1], key1);
          __m512i accumulator11 = _mm512_dpbusd_epi32(
              _mm512_setzero_si512(), query_broadcasts1[1], key1);
          const __m512i key2 = _mm512_loadu_si512(packed_block + 128);
          __m512i accumulator02 = _mm512_dpbusd_epi32(
              _mm512_setzero_si512(), query_broadcasts0[2], key2);
          __m512i accumulator12 = _mm512_dpbusd_epi32(
              _mm512_setzero_si512(), query_broadcasts1[2], key2);
          const __m512i key3 = _mm512_loadu_si512(packed_block + 192);
          __m512i accumulator03 = _mm512_dpbusd_epi32(
              _mm512_setzero_si512(), query_broadcasts0[3], key3);
          __m512i accumulator13 = _mm512_dpbusd_epi32(
              _mm512_setzero_si512(), query_broadcasts1[3], key3);
          const __m512i key4 = _mm512_loadu_si512(packed_block + 256);
          accumulator00 = _mm512_dpbusd_epi32(
              accumulator00, query_broadcasts0[4], key4);
          accumulator10 = _mm512_dpbusd_epi32(
              accumulator10, query_broadcasts1[4], key4);
          const __m512i key5 = _mm512_loadu_si512(packed_block + 320);
          accumulator01 = _mm512_dpbusd_epi32(
              accumulator01, query_broadcasts0[5], key5);
          accumulator11 = _mm512_dpbusd_epi32(
              accumulator11, query_broadcasts1[5], key5);
          const __m512i key6 = _mm512_loadu_si512(packed_block + 384);
          accumulator02 = _mm512_dpbusd_epi32(
              accumulator02, query_broadcasts0[6], key6);
          accumulator12 = _mm512_dpbusd_epi32(
              accumulator12, query_broadcasts1[6], key6);
          const __m512i key7 = _mm512_loadu_si512(packed_block + 448);
          accumulator03 = _mm512_dpbusd_epi32(
              accumulator03, query_broadcasts0[7], key7);
          accumulator13 = _mm512_dpbusd_epi32(
              accumulator13, query_broadcasts1[7], key7);
          __m512i accumulator0 = _mm512_add_epi32(
              _mm512_add_epi32(accumulator00, accumulator01),
              _mm512_add_epi32(accumulator02, accumulator03));
          __m512i accumulator1 = _mm512_add_epi32(
              _mm512_add_epi32(accumulator10, accumulator11),
              _mm512_add_epi32(accumulator12, accumulator13));
          const __m512i correction = _mm512_slli_epi32(
              _mm512_loadu_si512(key_sums + block * 16), 7);
          accumulator0 = _mm512_sub_epi32(accumulator0, correction);
          accumulator1 = _mm512_sub_epi32(accumulator1, correction);
          const __m512 key_scale =
              _mm512_loadu_ps(key_scales + block * 16);
          const __m512 score0 = _mm512_mul_ps(
              _mm512_cvtepi32_ps(accumulator0),
              _mm512_mul_ps(key_scale, query_multiplier0));
          const __m512 score1 = _mm512_mul_ps(
              _mm512_cvtepi32_ps(accumulator1),
              _mm512_mul_ps(key_scale, query_multiplier1));
          if (block < full_key_blocks) {
            _mm512_storeu_ps(scores0 + block * 16, score0);
            _mm512_storeu_ps(scores1 + block * 16, score1);
            maximum0 = _mm512_max_ps(maximum0, score0);
            maximum1 = _mm512_max_ps(maximum1, score1);
          } else {
            tail_score0 = score0;
            tail_score1 = score1;
            _mm512_mask_storeu_ps(scores0 + block * 16, tail_mask, score0);
            _mm512_mask_storeu_ps(scores1 + block * 16, tail_mask, score1);
            maximum0 = _mm512_max_ps(
                maximum0,
                _mm512_mask_mov_ps(negative_infinity, tail_mask, score0));
            maximum1 = _mm512_max_ps(
                maximum1,
                _mm512_mask_mov_ps(negative_infinity, tail_mask, score1));
          }
        }

        const float inverse_sum0 =
            exponentiate_vnni_score_row(scores0, maximum0, tail_score0);
#if defined(KTCP_ATTENTION_PV_INT8)
        inverse_sums[token] = inverse_sum0;
#else
        normalize_vnni_score_row(scores0, inverse_sum0);
#endif
        if (has_second) {
          const float inverse_sum1 =
              exponentiate_vnni_score_row(scores1, maximum1, tail_score1);
#if defined(KTCP_ATTENTION_PV_INT8)
          inverse_sums[second_token] = inverse_sum1;
#else
          normalize_vnni_score_row(scores1, inverse_sum1);
#endif
        }
      }

      if (profile != nullptr) {
        profile->add(kProfileAttentionQk, qk_started, Clock::now());
      }

      auto pv_started = Clock::time_point{};
      if (profile != nullptr) pv_started = Clock::now();
#if defined(KTCP_ATTENTION_PV_INT8)
      for (size_t token = 0; token < kS; token += 2) {
        const size_t second_token = token + 1 < kS ? token + 1 : token;
        run_vnni_int8_pv_pair(
            head_probabilities + token * kS,
            inverse_sums[token],
            output_tokens_ + (token * kHeads + head) * kHeadDim,
            head_probabilities + second_token * kS,
            inverse_sums[second_token],
            output_tokens_ + (second_token * kHeads + head) * kHeadDim);
      }
#else
      MlasGemm(
          CBLAS_TRANSPOSE::CblasNoTrans,
          CBLAS_TRANSPOSE::CblasNoTrans,
          kS,
          kHeadDim,
          kS,
          1.0f,
          head_probabilities,
          kS,
          head_value,
          kHeadDim,
          0.0f,
          output_tokens_ + head * kHeadDim,
          kHeads * kHeadDim,
          nullptr);
#endif
      if (profile != nullptr) {
        profile->add(kProfileAttentionPv, pv_started, Clock::now());
      }
    }
  }
#endif

  const float* q_ = nullptr;
  const float* k_ = nullptr;
  const float* k_transposed_ = nullptr;
  const float* v_ = nullptr;
  float* workspace_ = nullptr;
  float* output_tokens_ = nullptr;
  MlasFlashAttentionThreadedArgs flash_{};
#if defined(KTCP_ATTENTION_VNNI)
  mutable std::vector<int8_t> vnni_packed_keys_ = std::vector<int8_t>(
      kHeads * ((kS + 15) / 16) * (kHeadDim / 4) * 64, 0);
  mutable std::vector<float> vnni_key_scales_ =
      std::vector<float>(kHeads * kAttentionPaddedS, 0.0f);
  mutable std::vector<int32_t> vnni_key_sums_ =
      std::vector<int32_t>(kHeads * kAttentionPaddedS, 0);
#if defined(KTCP_ATTENTION_PV_INT8)
  mutable std::vector<int8_t> vnni_packed_values_ = std::vector<int8_t>(
      ((kS + 3) / 4) * (kHeadDim / 16) * 64, 0);
  mutable std::vector<float> vnni_value_scales_ =
      std::vector<float>(kHeadDim, 0.0f);
#endif
#endif
};

void swiglu(const float* up_gate, float* output) {
  for (size_t token = 0; token < kS; ++token) {
    const float* up = up_gate + token * (2 * kFfn);
    const float* gate = up + kFfn;
    float* destination = output + token * kFfn;
    for (size_t channel = 0; channel < kFfn; channel += 8) {
      const __m256 value = _mm256_loadu_ps(up + channel);
      const __m256 sigmoid = _mm256_div_ps(
          _mm256_set1_ps(1.0f),
          _mm256_add_ps(_mm256_set1_ps(1.0f), exp256_ps(_mm256_sub_ps(
              _mm256_setzero_ps(), value))));
      const __m256 clip_min = _mm256_set1_ps(-kSwiGluClip);
      const __m256 clip_max = _mm256_set1_ps(kSwiGluClip);
      const __m256 activated = _mm256_min_ps(
          clip_max,
          _mm256_max_ps(clip_min, _mm256_mul_ps(value, sigmoid)));
      const __m256 clipped_gate = _mm256_min_ps(
          clip_max,
          _mm256_max_ps(clip_min, _mm256_loadu_ps(gate + channel)));
      const __m256 result = _mm256_mul_ps(
          activated, clipped_gate);
      _mm256_storeu_ps(destination + channel, result);
    }
  }
}

class TransformerBlock {
 public:
  TransformerBlock(
      int mode,
      size_t block,
      const TensorMap& tensors,
      float* trunk)
      : trunk_(trunk),
        mode_(mode),
        norm1_(padded_vector<float>(kS * kC)),
        qkv_(padded_vector<float>(kS * 3 * kC)),
        q_heads_(padded_vector<float>(kHeads * kS * kHeadDim)),
        k_heads_(padded_vector<float>(kHeads * kS * kHeadDim)),
        k_transposed_(padded_vector<float>(
            kHeads * kHeadDim * kAttentionPaddedS)),
        v_heads_(padded_vector<float>(kHeads * kS * kHeadDim)),
        attention_workspace_(padded_vector<float>(
            2 * kAttentionQBlock + kAttentionQBlock * kAttentionKvBlock +
            kAttentionQBlock * kHeadDim)),
        attention_(padded_vector<float>(kS * kC)),
        projected_(padded_vector<float>(kS * kC)),
        norm2_(padded_vector<float>(kS * kC)),
        up_gate_(padded_vector<float>(kS * 2 * kFfn)),
        swiglu_(padded_vector<float>(kS * kFfn)),
        down_(padded_vector<float>(kS * kC)) {
    const std::string prefix = "blocks." + std::to_string(block) + ".";
    gamma1_ = require_values(tensors, prefix + "norm1.weight", {kC});
    gamma2_ = require_values(tensors, prefix + "norm2.weight", {kC});
    q_gamma_ = require_values(tensors, prefix + "q_norm.weight", {kHeadDim});
    k_gamma_ = require_values(tensors, prefix + "k_norm.weight", {kHeadDim});
    const Tensor& q =
        requireQuantizedTensor(tensors, prefix + "q_proj.weight", {kC, kC});
    const Tensor& k =
        requireQuantizedTensor(tensors, prefix + "k_proj.weight", {kC, kC});
    const Tensor& v =
        requireQuantizedTensor(tensors, prefix + "v_proj.weight", {kC, kC});
    const Tensor& up = requireQuantizedTensor(
        tensors, prefix + "ffn_linear1.weight", {kFfn, kC});
    const Tensor& gate = requireQuantizedTensor(
        tensors, prefix + "ffn_linear_gate.weight", {kFfn, kC});

    const auto& frequencies =
        require_values(tensors, prefix + "rope_freqs", {kHeads, kPairs, 2});
#if defined(KTCP_POSTOPS_AVX512)
    rope_cos_.resize(kS * kHeads * kHeadDim);
    rope_sin_.resize(kS * kHeads * kHeadDim);
#else
    rope_cos_.resize(kS * kHeads * kPairs);
    rope_sin_.resize(kS * kHeads * kPairs);
#endif
    for (size_t token = 0; token < kS; ++token) {
      const float y = static_cast<float>(token / KTCP_BOARD_LEN);
      const float x = static_cast<float>(token % KTCP_BOARD_LEN);
      for (size_t head = 0; head < kHeads; ++head) {
        for (size_t pair = 0; pair < kPairs; ++pair) {
          const size_t frequency_index = (head * kPairs + pair) * 2;
          const float angle = x * frequencies[frequency_index] +
                              y * frequencies[frequency_index + 1];
          const float cosine = std::cos(angle);
          const float sine = std::sin(angle);
#if defined(KTCP_POSTOPS_AVX512)
          const size_t table_index =
              (token * kHeads + head) * kHeadDim + pair * 2;
          rope_cos_[table_index] = cosine;
          rope_cos_[table_index + 1] = cosine;
          rope_sin_[table_index] = -sine;
          rope_sin_[table_index + 1] = sine;
#else
          const size_t table_index = (token * kHeads + head) * kPairs + pair;
          rope_cos_[table_index] = cosine;
          rope_sin_[table_index] = sine;
#endif
        }
      }
    }

    int qkv_mode = mode;
#if defined(KTCP_QKV_IMPL_QD8)
    if (mode == KTCP_MODE_MLAS_U8S8) qkv_mode = KTCP_MODE_QD8;
#endif
    qkv_fused_ = qkv_mode == KTCP_MODE_MLAS_U8S8;
    QkvRopeOutput qkv_rope_output{};
    if (qkv_fused_) {
      float* transposed_key = nullptr;
#if defined(KTCP_ATTENTION_AVX512)
      transposed_key = k_transposed_.data();
#endif
      qkv_rope_output = {
          q_heads_.data(), k_heads_.data(), v_heads_.data(),
          transposed_key,
          rope_cos_.data(), rope_sin_.data(),
          q_gamma_.data(), k_gamma_.data()};
    }

    qkv_fc_ = std::make_unique<FullyConnected>(
        qkv_mode, kS, kC, 3 * kC, concatenate_s8_rows({&q, &k, &v}, kC),
        zero_bias(3 * kC), norm1_.data(), qkv_.data(), qkv_rope_output);
    out_fc_ = std::make_unique<FullyConnected>(
        mode, kS, kC, kC,
        copy_s8_weights(tensors, prefix + "out_proj.weight", {kC, kC}),
        zero_bias(kC), attention_.data(), projected_.data(),
        QkvRopeOutput{}, SwiGluOutput{},
        mode == KTCP_MODE_MLAS_U8S8
            ? ResidualOutput{trunk_, norm2_.data(), gamma2_.data()}
            : ResidualOutput{});
    up_gate_fc_ = std::make_unique<FullyConnected>(
        mode, kS, kC, 2 * kFfn, concatenate_s8_rows({&up, &gate}, kC),
        zero_bias(2 * kFfn), norm2_.data(), up_gate_.data(),
        QkvRopeOutput{},
        mode == KTCP_MODE_MLAS_U8S8 ? SwiGluOutput{swiglu_.data()} : SwiGluOutput{});
    down_fc_ = std::make_unique<FullyConnected>(
        mode, kS, kFfn, kC,
        copy_s8_weights(tensors, prefix + "ffn_linear2.weight", {kC, kFfn}),
        zero_bias(kC), swiglu_.data(), down_.data(),
        QkvRopeOutput{}, SwiGluOutput{},
        mode == KTCP_MODE_MLAS_U8S8
            ? ResidualOutput{trunk_, nullptr, nullptr}
            : ResidualOutput{});

    attention_kernel_ = std::make_unique<BatchedAttention>(
        q_heads_.data(), k_heads_.data(), k_transposed_.data(), v_heads_.data(),
        attention_workspace_.data(), attention_.data());

  }

  void run(Profile* profile = nullptr) {
    auto started = Clock::time_point{};
    if (profile != nullptr) started = Clock::now();
    rmsnorm(trunk_, norm1_.data(), gamma1_.data());
    if (profile != nullptr) profile->add(kProfileNorm1, started, Clock::now());

    if (profile != nullptr) started = Clock::now();
    qkv_fc_->run();
    if (profile != nullptr) profile->add(kProfileQkv, started, Clock::now());

    if (profile != nullptr) started = Clock::now();
    if (!qkv_fused_) {
      apply_rope();
    }
    if (profile != nullptr) profile->add(kProfileRope, started, Clock::now());

    attention_kernel_->run(profile);

    if (profile != nullptr) started = Clock::now();
    out_fc_->run();
    if (profile != nullptr) {
      profile->add(kProfileOutProjection, started, Clock::now());
    }

    if (profile != nullptr) started = Clock::now();
    if (mode_ != KTCP_MODE_MLAS_U8S8) {
      residual_rmsnorm(trunk_, projected_.data(), norm2_.data(), gamma2_.data());
    }
    if (profile != nullptr) {
      profile->add(kProfileResidualNorm2, started, Clock::now());
    }

    if (profile != nullptr) started = Clock::now();
    up_gate_fc_->run();
    if (profile != nullptr) profile->add(kProfileUpGate, started, Clock::now());

    if (profile != nullptr) started = Clock::now();
    if (mode_ != KTCP_MODE_MLAS_U8S8) {
      swiglu(up_gate_.data(), swiglu_.data());
    }
    if (profile != nullptr) profile->add(kProfileSwiGlu, started, Clock::now());

    if (profile != nullptr) started = Clock::now();
    down_fc_->run();
    if (profile != nullptr) profile->add(kProfileDown, started, Clock::now());

    if (profile != nullptr) started = Clock::now();
    if (mode_ != KTCP_MODE_MLAS_U8S8) {
      add_in_place(trunk_, down_.data(), kS * kC);
    }
    if (profile != nullptr) {
      profile->add(kProfileResidualAdd, started, Clock::now());
    }
  }

 private:
  void apply_rope() {
    for (size_t token = 0; token < kS; ++token) {
      const float* q = qkv_.data() + token * (3 * kC);
      const float* k = q + kC;
      const float* v = k + kC;
      for (size_t head = 0; head < kHeads; ++head) {
        const size_t channel_offset = head * kHeadDim;
        float* q_destination =
            q_heads_.data() + (head * kS + token) * kHeadDim;
        float* k_destination =
            k_heads_.data() + (head * kS + token) * kHeadDim;
        float* v_destination =
            v_heads_.data() + (head * kS + token) * kHeadDim;
        float q_sum_squares = 0.0f;
        float k_sum_squares = 0.0f;
        for (size_t channel = 0; channel < kHeadDim; ++channel) {
          const float q_value = q[channel_offset + channel];
          const float k_value = k[channel_offset + channel];
          q_sum_squares += q_value * q_value;
          k_sum_squares += k_value * k_value;
        }
        const float q_inv_rms = 1.0f / std::sqrt(
            q_sum_squares / static_cast<float>(kHeadDim) + kRmsEpsilon);
        const float k_inv_rms = 1.0f / std::sqrt(
            k_sum_squares / static_cast<float>(kHeadDim) + kRmsEpsilon);
        for (size_t pair = 0; pair < kPairs; ++pair) {
#if defined(KTCP_POSTOPS_AVX512)
          const size_t table_index =
              (token * kHeads + head) * kHeadDim + pair * 2;
          const float cosine = rope_cos_[table_index];
          const float sine = rope_sin_[table_index + 1];
#else
          const size_t table_index = (token * kHeads + head) * kPairs + pair;
          const float cosine = rope_cos_[table_index];
          const float sine = rope_sin_[table_index];
#endif
          const size_t channel = channel_offset + pair * 2;
          const float q0 =
              q[channel] * q_gamma_[pair * 2] * q_inv_rms;
          const float q1 =
              q[channel + 1] * q_gamma_[pair * 2 + 1] * q_inv_rms;
          q_destination[pair * 2] = q0 * cosine - q1 * sine;
          q_destination[pair * 2 + 1] = q0 * sine + q1 * cosine;
          const float k0 =
              k[channel] * k_gamma_[pair * 2] * k_inv_rms;
          const float k1 =
              k[channel + 1] * k_gamma_[pair * 2 + 1] * k_inv_rms;
          k_destination[pair * 2] = k0 * cosine - k1 * sine;
          k_destination[pair * 2 + 1] = k0 * sine + k1 * cosine;
#if defined(KTCP_ATTENTION_AVX512)
          k_transposed_[
              (head * kHeadDim + pair * 2) * kAttentionPaddedS + token] =
              k_destination[pair * 2];
          k_transposed_[
              (head * kHeadDim + pair * 2 + 1) * kAttentionPaddedS + token] =
              k_destination[pair * 2 + 1];
#endif
          v_destination[pair * 2] = v[channel];
          v_destination[pair * 2 + 1] = v[channel + 1];
        }
      }
    }
  }

  float* trunk_ = nullptr;
  int mode_ = KTCP_MODE_FP32;
  bool qkv_fused_ = false;
  std::vector<float> gamma1_;
  std::vector<float> gamma2_;
  std::vector<float> q_gamma_;
  std::vector<float> k_gamma_;
  std::vector<float> norm1_;
  std::vector<float> qkv_;
  std::vector<float> q_heads_;
  std::vector<float> k_heads_;
  std::vector<float> k_transposed_;
  std::vector<float> v_heads_;
  std::vector<float> attention_workspace_;
  std::vector<float> attention_;
  std::vector<float> projected_;
  std::vector<float> norm2_;
  std::vector<float> up_gate_;
  std::vector<float> swiglu_;
  std::vector<float> down_;
  std::vector<float> rope_cos_;
  std::vector<float> rope_sin_;
  std::unique_ptr<BatchedAttention> attention_kernel_;
  std::unique_ptr<FullyConnected> qkv_fc_;
  std::unique_ptr<FullyConnected> out_fc_;
  std::unique_ptr<FullyConnected> up_gate_fc_;
  std::unique_ptr<FullyConnected> down_fc_;
};

enum HeadActivation {
  kHeadRelu = 0,
  kHeadSilu = 1,
};

void affine_activate(
    float* values,
    const float* multiplier,
    const float* bias,
    size_t rows,
    size_t channels,
    int activation) {
  if (activation != kHeadRelu && activation != kHeadSilu) {
    throw std::runtime_error("Unsupported CPU PTQ head activation");
  }
  const __m256 zero = _mm256_setzero_ps();
  const __m256 one = _mm256_set1_ps(1.0f);
  for (size_t row = 0; row < rows; ++row) {
    float* destination = values + row * channels;
    size_t channel = 0;
    for (; channel + 8 <= channels; channel += 8) {
      const __m256 normalized = _mm256_fmadd_ps(
          _mm256_loadu_ps(destination + channel),
          _mm256_loadu_ps(multiplier + channel),
          _mm256_loadu_ps(bias + channel));
      const __m256 result = activation == kHeadRelu
          ? _mm256_max_ps(zero, normalized)
          : _mm256_div_ps(
                normalized,
                _mm256_add_ps(one, exp256_ps(_mm256_sub_ps(zero, normalized))));
      _mm256_storeu_ps(destination + channel, result);
    }
    for (; channel < channels; ++channel) {
      const float normalized =
          destination[channel] * multiplier[channel] + bias[channel];
      destination[channel] = activation == kHeadRelu
          ? std::max(0.0f, normalized)
          : normalized / (1.0f + std::exp(-normalized));
    }
  }
}

void pool_policy(const float* input, float* output) {
  constexpr size_t channels = 32;
  constexpr float scaled_mean = 0.1f;  // (sqrt(225)-14)/10
  for (size_t channel = 0; channel < channels; ++channel) {
    float sum = 0.0f;
    float maximum = -std::numeric_limits<float>::infinity();
    for (size_t token = 0; token < kS; ++token) {
      const float value = input[token * channels + channel];
      sum += value;
      maximum = std::max(maximum, value);
    }
    const float mean = sum / static_cast<float>(kS);
    output[channel] = mean;
    output[channels + channel] = mean * scaled_mean;
    output[2 * channels + channel] = maximum;
  }
}

void pool_value(const float* input, float* output) {
  constexpr size_t channels = 32;
  constexpr float scaled_mean_1 = 0.1f;
  constexpr float scaled_mean_2 = -0.09f;  // 1^2/100 - 0.1
  for (size_t channel = 0; channel < channels; ++channel) {
    float sum = 0.0f;
    for (size_t token = 0; token < kS; ++token) {
      sum += input[token * channels + channel];
    }
    const float mean = sum / static_cast<float>(kS);
    output[channel] = mean;
    output[channels + channel] = mean * scaled_mean_1;
    output[2 * channels + channel] = mean * scaled_mean_2;
  }
}

class Model final : public Kernel {
 public:
  explicit Model(const TensorMap& tensors)
      : mode_(KTCP_MODE_MLAS_U8S8),
        spatial_nhwc_(padded_vector<float>(kS * KTCP_SPATIAL_INPUTS)),
        global_input_(padded_vector<float>(KTCP_GLOBAL_INPUTS)),
        trunk_(padded_vector<float>(kS * kC)),
        global_projection_(padded_vector<float>(kC)),
        policy_p_(padded_vector<float>(kS * 32)),
        policy_g_(padded_vector<float>(kS * 32)),
        policy_pooled_(padded_vector<float>(96)),
        policy_global_(padded_vector<float>(32)),
        policy_pass_value_(padded_vector<float>(1)),
        policy_logits_nhwc_(padded_vector<float>(kS)),
        value_spatial_(padded_vector<float>(kS * 32)),
        value_pooled_(padded_vector<float>(96)),
        value_hidden_(padded_vector<float>(kValueHidden)),
        value_out_(padded_vector<float>(KTCP_VALUE_SIZE)),
        score_value_out_(padded_vector<float>(KTCP_SCORE_VALUE_SIZE)),
        ownership_(padded_vector<float>(KTCP_OWNERSHIP_SIZE)) {
    check_xnn(xnn_initialize(nullptr), "xnn_initialize");

    stem_ = std::make_unique<StemConvolution>(
        require_values(tensors, "conv_spatial.weight", {kC, KTCP_SPATIAL_INPUTS, 3, 3}),
        spatial_nhwc_.data(), trunk_.data());
    global_fc_ = std::make_unique<FullyConnected>(
        KTCP_MODE_FP32, 1, KTCP_GLOBAL_INPUTS, kC,
        require_values(tensors, "linear_global.weight", {kC, KTCP_GLOBAL_INPUTS}),
        zero_bias(kC), global_input_.data(), global_projection_.data());

    blocks_.reserve(kBlocks);
    for (size_t block = 0; block < kBlocks; ++block) {
      blocks_.push_back(
          std::make_unique<TransformerBlock>(mode_, block, tensors, trunk_.data()));
    }
    trunk_final_mul_ = require_values(tensors, "norm_trunkfinal.mul", {kC});
    trunk_final_bias_ = require_values(tensors, "norm_trunkfinal.bias", {kC});
    trunk_final_activation_ = static_cast<int>(
        require_values(tensors, "norm_trunkfinal.activation", {1})[0]);

    policy_g_mul_ = require_values(tensors, "policy_head.g1.mul", {32});
    policy_g_bias_ = require_values(tensors, "policy_head.g1.bias", {32});
    policy_g_activation_ = static_cast<int>(
        require_values(tensors, "policy_head.g1.activation", {1})[0]);
    policy_p_mul_ = require_values(tensors, "policy_head.p1.mul", {32});
    policy_p_bias_ = require_values(tensors, "policy_head.p1.bias", {32});
    policy_p_activation_ = static_cast<int>(
        require_values(tensors, "policy_head.p1.activation", {1})[0]);
    policy_conv_p_ = std::make_unique<FullyConnected>(
        KTCP_MODE_FP32, kS, kC, 32,
        require_values(tensors, "policy_head.conv1p.weight", {32, kC, 1, 1}),
        zero_bias(32), trunk_.data(), policy_p_.data());
    policy_conv_g_ = std::make_unique<FullyConnected>(
        KTCP_MODE_FP32, kS, kC, 32,
        require_values(tensors, "policy_head.conv1g.weight", {32, kC, 1, 1}),
        zero_bias(32), trunk_.data(), policy_g_.data());
    policy_linear_g_ = std::make_unique<FullyConnected>(
        KTCP_MODE_FP32, 1, 96, 32,
        require_values(tensors, "policy_head.linear_g.weight", {32, 96}),
        zero_bias(32), policy_pooled_.data(), policy_global_.data());
    policy_pass_ = std::make_unique<FullyConnected>(
        KTCP_MODE_FP32, 1, 96, 1,
        require_values(tensors, "policy_head.linear_pass.weight", {1, 96}),
        zero_bias(1), policy_pooled_.data(), policy_pass_value_.data());
    policy_conv_2_ = std::make_unique<FullyConnected>(
        KTCP_MODE_FP32, kS, 32, 1,
        require_values(
            tensors, "policy_head.conv2p.weight", {1, 32, 1, 1}),
        zero_bias(1), policy_p_.data(), policy_logits_nhwc_.data());

    value_spatial_mul_ = require_values(tensors, "value_head.v1.mul", {32});
    value_spatial_bias_ = require_values(tensors, "value_head.v1.bias", {32});
    value_spatial_activation_ = static_cast<int>(
        require_values(tensors, "value_head.v1.activation", {1})[0]);
    value_hidden_mul_ = require_values(
        tensors, "value_head.v2.mul", {kValueHidden});
    value_hidden_bias_ = require_values(
        tensors, "value_head.v2.activation_bias", {kValueHidden});
    value_hidden_activation_ = static_cast<int>(
        require_values(tensors, "value_head.v2.activation", {1})[0]);
    value_conv_1_ = std::make_unique<FullyConnected>(
        KTCP_MODE_FP32, kS, kC, 32,
        require_values(tensors, "value_head.conv1.weight", {32, kC, 1, 1}),
        zero_bias(32), trunk_.data(), value_spatial_.data());
    value_linear_2_ = std::make_unique<FullyConnected>(
        KTCP_MODE_FP32, 1, 96, kValueHidden,
        require_values(tensors, "value_head.linear2.weight", {kValueHidden, 96}),
        require_values(tensors, "value_head.linear2.bias", {kValueHidden}),
        value_pooled_.data(), value_hidden_.data());
    value_head_ = std::make_unique<FullyConnected>(
        KTCP_MODE_FP32, 1, kValueHidden, KTCP_VALUE_SIZE,
        require_values(tensors, "value_head.linear_valuehead.weight", {KTCP_VALUE_SIZE, kValueHidden}),
        require_values(tensors, "value_head.linear_valuehead.bias", {KTCP_VALUE_SIZE}),
        value_hidden_.data(), value_out_.data());
    score_value_head_ = std::make_unique<FullyConnected>(
        KTCP_MODE_FP32, 1, kValueHidden, KTCP_SCORE_VALUE_SIZE,
        require_values(
            tensors, "value_head.linear_scorevalue.weight", {KTCP_SCORE_VALUE_SIZE, kValueHidden}),
        require_values(
            tensors, "value_head.linear_scorevalue.bias", {KTCP_SCORE_VALUE_SIZE}),
        value_hidden_.data(), score_value_out_.data());
    ownership_head_ = std::make_unique<FullyConnected>(
        KTCP_MODE_FP32, kS, 32, 1,
        require_values(tensors, "value_head.conv_ownership.weight", {1, 32, 1, 1}),
        zero_bias(1), value_spatial_.data(), ownership_.data());
  }

  const ProfileSpec& profile() const override {
    return CPU_PTQ_PROFILE_FUNCTION();
  }

  void infer(
      const float* input_spatial,
      const float* input_global,
      float* out_policy,
      float* out_value,
      float* out_score_value,
      float* out_ownership) override {
    inferProfile(
        input_spatial, input_global, out_policy, out_value, out_score_value,
        out_ownership, nullptr);
  }

 private:
  void inferProfile(
      const float* input_spatial,
      const float* input_global,
      float* out_policy,
      float* out_value,
      float* out_score_value,
      float* out_ownership,
      Profile* profile = nullptr) {
    if (input_spatial == nullptr || input_global == nullptr || out_policy == nullptr ||
        out_value == nullptr || out_score_value == nullptr ||
        out_ownership == nullptr) {
      throw std::runtime_error("CPU PTQ infer received a null tensor pointer");
    }
    auto input_stem_started = Clock::time_point{};
    if (profile != nullptr) input_stem_started = Clock::now();
    // NCHW engine input -> fixed NHWC internal layout.
    for (size_t token = 0; token < kS; ++token) {
      for (size_t channel = 0; channel < KTCP_SPATIAL_INPUTS; ++channel) {
        spatial_nhwc_[token * KTCP_SPATIAL_INPUTS + channel] =
            input_spatial[channel * kS + token];
      }
    }
    std::copy_n(input_global, KTCP_GLOBAL_INPUTS, global_input_.data());

    stem_->run();
    global_fc_->run();
    for (size_t token = 0; token < kS; ++token) {
      add_in_place(trunk_.data() + token * kC, global_projection_.data(), kC);
    }
    if (profile != nullptr) {
      profile->add(kProfileInputStem, input_stem_started, Clock::now());
    }
    for (auto& block : blocks_) {
      block->run(profile);
    }
    auto heads_started = Clock::time_point{};
    if (profile != nullptr) heads_started = Clock::now();
    affine_activate(
        trunk_.data(), trunk_final_mul_.data(), trunk_final_bias_.data(),
        kS, kC, trunk_final_activation_);

    run_policy(out_policy);
    run_value(out_value, out_score_value, out_ownership);
    if (profile != nullptr) {
      profile->add(kProfileHeads, heads_started, Clock::now());
    }
  }

  void run_policy(float* output) {
    policy_conv_p_->run();
    policy_conv_g_->run();
    affine_activate(
        policy_g_.data(), policy_g_mul_.data(), policy_g_bias_.data(),
        kS, 32, policy_g_activation_);
    pool_policy(policy_g_.data(), policy_pooled_.data());
    policy_linear_g_->run();
    policy_pass_->run();
    for (size_t token = 0; token < kS; ++token) {
      add_in_place(policy_p_.data() + token * 32, policy_global_.data(), 32);
    }
    affine_activate(
        policy_p_.data(), policy_p_mul_.data(), policy_p_bias_.data(),
        kS, 32, policy_p_activation_);
    policy_conv_2_->run();
    for (size_t token = 0; token < kS; ++token) {
      output[token] = policy_logits_nhwc_[token];
    }
    output[kS] = policy_pass_value_[0];
  }

  void run_value(
      float* out_value,
      float* out_score_value,
      float* out_ownership) {
    value_conv_1_->run();
    affine_activate(
        value_spatial_.data(), value_spatial_mul_.data(),
        value_spatial_bias_.data(), kS, 32, value_spatial_activation_);
    pool_value(value_spatial_.data(), value_pooled_.data());
    value_linear_2_->run();
    affine_activate(
        value_hidden_.data(), value_hidden_mul_.data(), value_hidden_bias_.data(),
        1, kValueHidden, value_hidden_activation_);
    value_head_->run();
    score_value_head_->run();
    ownership_head_->run();
    std::copy_n(value_out_.data(), KTCP_VALUE_SIZE, out_value);
    std::copy_n(score_value_out_.data(), KTCP_SCORE_VALUE_SIZE, out_score_value);
    std::copy_n(ownership_.data(), KTCP_OWNERSHIP_SIZE, out_ownership);
  }

  int mode_ = KTCP_MODE_FP32;
  std::vector<float> spatial_nhwc_;
  std::vector<float> global_input_;
  std::vector<float> trunk_;
  std::vector<float> global_projection_;
  std::unique_ptr<StemConvolution> stem_;
  std::unique_ptr<FullyConnected> global_fc_;
  std::vector<std::unique_ptr<TransformerBlock>> blocks_;
  std::vector<float> trunk_final_mul_;
  std::vector<float> trunk_final_bias_;
  int trunk_final_activation_ = 0;

  std::vector<float> policy_p_;
  std::vector<float> policy_g_;
  std::vector<float> policy_pooled_;
  std::vector<float> policy_global_;
  std::vector<float> policy_pass_value_;
  std::vector<float> policy_logits_nhwc_;
  std::vector<float> policy_g_mul_;
  std::vector<float> policy_g_bias_;
  std::vector<float> policy_p_mul_;
  std::vector<float> policy_p_bias_;
  int policy_g_activation_ = 0;
  int policy_p_activation_ = 0;
  std::unique_ptr<FullyConnected> policy_conv_p_;
  std::unique_ptr<FullyConnected> policy_conv_g_;
  std::unique_ptr<FullyConnected> policy_linear_g_;
  std::unique_ptr<FullyConnected> policy_pass_;
  std::unique_ptr<FullyConnected> policy_conv_2_;

  std::vector<float> value_spatial_;
  std::vector<float> value_pooled_;
  std::vector<float> value_hidden_;
  std::vector<float> value_out_;
  std::vector<float> score_value_out_;
  std::vector<float> ownership_;
  std::vector<float> value_spatial_mul_;
  std::vector<float> value_spatial_bias_;
  int value_spatial_activation_ = 0;
  std::vector<float> value_hidden_mul_;
  std::vector<float> value_hidden_bias_;
  int value_hidden_activation_ = 0;
  std::unique_ptr<FullyConnected> value_conv_1_;
  std::unique_ptr<FullyConnected> value_linear_2_;
  std::unique_ptr<FullyConnected> value_head_;
  std::unique_ptr<FullyConnected> score_value_head_;
  std::unique_ptr<FullyConnected> ownership_head_;
};

std::unique_ptr<Kernel> create(const TensorMap& tensors) {
  return std::make_unique<Model>(tensors);
}

}  // namespace CPU_PTQ_PROFILE_NAMESPACE
}  // namespace CpuPtq

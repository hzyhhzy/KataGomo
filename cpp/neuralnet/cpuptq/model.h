#ifndef NEURALNET_CPUPTQ_MODEL_H_
#define NEURALNET_CPUPTQ_MODEL_H_

#include <cstdint>
#include <initializer_list>
#include <string>
#include <unordered_map>
#include <vector>

namespace CpuPtq {

constexpr int BASE_MODEL_VERSION = 205;
constexpr int MODEL_VERSION = 206;
constexpr int SOURCE_MODEL_VERSION = 11;
constexpr int BOARD_LEN = 7;
constexpr int BOARD_AREA = BOARD_LEN * BOARD_LEN;
constexpr int SPATIAL_INPUTS = 22;
constexpr int GLOBAL_INPUTS = 19;
constexpr int POLICY_SIZE = BOARD_AREA + 1;
constexpr int VALUE_SIZE = 3;
constexpr int MISC_VALUE_SIZE = 6;

enum class ProfileKind {
  B11C96H3F256,
  B16C128H4F384,
};

struct ProfileSpec {
  ProfileKind kind;
  const char* name;
  int blocks;
  int channels;
  int heads;
  int headDim;
  int ffnChannels;
  int valueHiddenChannels;
};

const ProfileSpec& b11Profile();
const ProfileSpec& b16Profile();

enum class TensorKind : uint8_t {
  FP32 = 1,
  S8PerOutput = 2,
};

struct Tensor {
  TensorKind kind;
  int quantizedMax;
  std::vector<uint32_t> shape;
  std::vector<float> values;
  std::vector<int8_t> codes;
  std::vector<float> scales;
};

struct Model {
  std::string name;
  std::string sha256;
  int version;
  const ProfileSpec* profile;
  std::unordered_map<std::string,Tensor> tensors;
};

Model loadModelFile(const std::string& fileName, const std::string& expectedSha256);

const Tensor& requireTensor(
  const Model& model,
  const std::string& name,
  TensorKind kind,
  std::initializer_list<uint32_t> shape
);

const std::vector<float>& requireFP32(
  const Model& model,
  const std::string& name,
  std::initializer_list<uint32_t> shape
);

const Tensor& requireS8(
  const Model& model,
  const std::string& name,
  std::initializer_list<uint32_t> shape
);

}  // namespace CpuPtq

#endif  // NEURALNET_CPUPTQ_MODEL_H_

#ifndef NEURALNET_CPUPTQ_KERNEL_H_
#define NEURALNET_CPUPTQ_KERNEL_H_

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct ModelDesc;

namespace CpuPtq {

constexpr int BOARD_LEN = 15;
constexpr int BOARD_AREA = BOARD_LEN * BOARD_LEN;
constexpr int SPATIAL_INPUTS = 22;
constexpr int GLOBAL_INPUTS = 39;
constexpr int POLICY_SIZE = BOARD_AREA + 1;
constexpr int VALUE_SIZE = 3;
constexpr int SCORE_VALUE_SIZE = 6;
constexpr int OWNERSHIP_SIZE = BOARD_AREA;

enum class ProfileKind {
  B16C128H4F384,
  B11C96H3F256,
};

struct ProfileSpec {
  ProfileKind kind;
  const char* name;
  int transformerBlocks;
  int trunkChannels;
  int heads;
  int headDim;
  int ffnChannels;
  int valueHiddenChannels;
};

const ProfileSpec& b16Profile();
const ProfileSpec& b11Profile();
const ProfileSpec& selectProfile(const ModelDesc& model);

struct Tensor {
  std::vector<uint64_t> shape;
  std::vector<float> values;
  // Present only for v106 transformer projections. Values are output-major,
  // with input channels contiguous, and scales has one entry per output.
  std::vector<int8_t> quantizedValues;
  std::vector<float> quantizedScales;
};
using TensorMap = std::unordered_map<std::string,Tensor>;

const std::vector<float>& requireTensor(
  const TensorMap& tensors,
  const std::string& name,
  std::initializer_list<uint64_t> shape
);

const Tensor& requireQuantizedTensor(
  const TensorMap& tensors,
  const std::string& name,
  std::initializer_list<uint64_t> shape
);

TensorMap makeKernelTensors(const ModelDesc& model, const ProfileSpec& profile);

class Kernel {
 public:
  virtual ~Kernel() = default;
  virtual const ProfileSpec& profile() const = 0;
  virtual void infer(
    const float* spatialNCHW,
    const float* global,
    float* policy,
    float* value,
    float* scoreValue,
    float* ownership
  ) = 0;
};

std::unique_ptr<Kernel> createB16Kernel(const TensorMap& tensors);
std::unique_ptr<Kernel> createB11Kernel(const TensorMap& tensors);

}  // namespace CpuPtq

#endif  // NEURALNET_CPUPTQ_KERNEL_H_

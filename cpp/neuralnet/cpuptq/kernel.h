#ifndef NEURALNET_CPUPTQ_KERNEL_H_
#define NEURALNET_CPUPTQ_KERNEL_H_

#include "model.h"

#include <memory>

namespace CpuPtq {

class Kernel {
 public:
  virtual ~Kernel() = default;
  virtual const ProfileSpec& profile() const = 0;
  virtual void infer(
    const float* spatialNCHW,
    const float* global,
    float* policy,
    float* value,
    float* miscValue
  ) = 0;
  virtual void inferWithTraces(
    const float* spatialNCHW,
    const float* global,
    float* policy,
    float* value,
    float* miscValue,
    std::vector<float>& traces
  ) = 0;
};

std::unique_ptr<Kernel> createKernel(const Model& model);

}  // namespace CpuPtq

#endif  // NEURALNET_CPUPTQ_KERNEL_H_

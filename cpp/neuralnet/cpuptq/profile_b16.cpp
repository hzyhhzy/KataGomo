#define CPU_PTQ_PROFILE_NAMESPACE b16_impl
#define CPU_PTQ_PROFILE_KIND CpuPtq::ProfileKind::B16C128H4F384
#define CPU_PTQ_PROFILE_FUNCTION CpuPtq::b16Profile
#define CPU_PTQ_TRUNK_CHANNELS 128
#define CPU_PTQ_HEADS 4
#define CPU_PTQ_FFN_CHANNELS 384
#define CPU_PTQ_BLOCKS 16
#define CPU_PTQ_VALUE_HIDDEN_CHANNELS 96

#include "kernel_impl.inc"

namespace CpuPtq {

std::unique_ptr<Kernel> createB16Kernel(const TensorMap& tensors) {
  return b16_impl::create(tensors);
}

}  // namespace CpuPtq

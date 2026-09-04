#define CPU_PTQ_PROFILE_NAMESPACE b24_impl
#define CPU_PTQ_PROFILE_KIND CpuPtq::ProfileKind::B24C192H6F512
#define CPU_PTQ_PROFILE_FUNCTION CpuPtq::b24Profile
#define CPU_PTQ_TRUNK_CHANNELS 192
#define CPU_PTQ_HEADS 6
#define CPU_PTQ_FFN_CHANNELS 512
#define CPU_PTQ_BLOCKS 24
#define CPU_PTQ_VALUE_HIDDEN_CHANNELS 96

#include "kernel_impl.inc"

namespace CpuPtq {

std::unique_ptr<Kernel> createB24Kernel(const TensorMap& tensors) {
  return b24_impl::create(tensors);
}

}  // namespace CpuPtq

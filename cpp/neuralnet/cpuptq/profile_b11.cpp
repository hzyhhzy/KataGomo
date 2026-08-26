#define CPU_PTQ_PROFILE_NAMESPACE b11_impl
#define CPU_PTQ_PROFILE_KIND CpuPtq::ProfileKind::B11C96H3F256
#define CPU_PTQ_PROFILE_FUNCTION CpuPtq::b11Profile
#define CPU_PTQ_TRUNK_CHANNELS 96
#define CPU_PTQ_HEADS 3
#define CPU_PTQ_FFN_CHANNELS 256
#define CPU_PTQ_BLOCKS 11
#define CPU_PTQ_VALUE_HIDDEN_CHANNELS 64
#define KTCP_ATTENTION_VNNI 1
#define KTCP_ATTENTION_PV_INT8 1
#define KTCP_POSTOPS_AVX512 1

#include "kernel_impl.inc"

namespace CpuPtq {

std::unique_ptr<Kernel> createB11Kernel(const TensorMap& tensors) {
  return b11_impl::create(tensors);
}

}  // namespace CpuPtq

#ifndef KATAGO_C384_RESIDUAL_AOT_ABI_H_
#define KATAGO_C384_RESIDUAL_AOT_ABI_H_

#include <cstdint>

// Engine-independent query ABI exported by generated residual-AOT bridges.
// The bridge owns no engine fallback. It describes one immutable exact-M
// implementation and is queried during ComputeHandle construction only.
struct C384ResidualRawDescriptorV1 {
  std::uint32_t abi_version;
  std::uint32_t family;
  std::uint32_t batch;
  std::uint32_t m;
  std::uint32_t k;
  std::uint32_t n;
  std::uint32_t tile_m;
  std::uint32_t tile_n;
  std::uint32_t tile_k;
  std::uint32_t stages;
  std::uint32_t atom_m;
  std::uint32_t atom_n;
  std::uint32_t atom_k;
  std::uint32_t epilogue_stages;
  std::uint32_t natural_ctas;
  std::uint32_t launch_ctas;
  std::uint32_t max_active_clusters;
  const char* candidate_id;
};

namespace C384ResidualAotAbi {

constexpr std::uint32_t kAbiVersion = 1;

enum class Family : std::uint32_t {
  OutProjResidual = 1,
  FfnDownResidual = 2,
};

}  // namespace C384ResidualAotAbi

#endif  // KATAGO_C384_RESIDUAL_AOT_ABI_H_

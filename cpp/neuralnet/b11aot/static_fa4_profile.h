#pragma once
// The engine must also gate the model, device and S361/H12/D32 geometry.
namespace b11aot {
constexpr bool supportsStaticFa4(int batch, bool packedQKV) {
  return packedQKV && (batch == 13 || batch == 16);
}
} // namespace b11aot

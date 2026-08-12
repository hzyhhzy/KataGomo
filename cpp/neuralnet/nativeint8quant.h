#ifndef NEURALNET_NATIVEINT8QUANT_H_
#define NEURALNET_NATIVEINT8QUANT_H_

#include <array>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

struct ModelDesc;

namespace NativeInt8Quant {

constexpr uint32_t MODEL_VERSION = 104;
constexpr uint32_t TRAILER_HEADER_SCHEMA_VERSION = 1;
constexpr uint32_t PAYLOAD_SCHEMA_VERSION = 1;
constexpr uint32_t REQUIRED_ENTRY_COUNT = 72;

enum class Role : uint32_t {
  QK = 1,
  FfnUp = 2,
  FfnGate = 3,
};

// Bytes are ready for the native GEMM kernels: for every output channel N,
// all K signed-int8 values are contiguous. QK uses Q outputs followed by K.
enum class PackedLayout : uint32_t {
  OutputMajorKContiguous = 1,
};

enum class QuantizationMode : uint32_t {
  SymmetricSignedInt8PerTensor = 1,
};

enum class RoundingMode : uint32_t {
  RoundTiesToEvenSaturate127 = 1,
};

struct Entry {
  uint32_t topologyIndex;
  Role role;
  PackedLayout layout;
  uint32_t inputChannels;
  uint32_t outputChannels;
  int32_t zeroPoint;
  uint32_t activationScaleBits;
  uint32_t weightScaleBits;
  std::vector<std::string> layerNames;
  std::array<uint8_t,32> masterSha256;
  std::array<uint8_t,32> packedSha256;
  std::vector<int8_t> packedWeights;
};

struct Metadata {
  uint32_t payloadSchemaVersion;
  QuantizationMode activationMode;
  QuantizationMode weightMode;
  RoundingMode roundingMode;
  int32_t zeroPoint;
  uint32_t activationClipBits;
  uint32_t activationScaleBits;
  std::vector<Entry> entries;

  Metadata();
  bool present() const;
};

// Deterministically derives the mandatory QK/up/gate metadata from the FP32
// masters in a native ModelDesc. This is intentionally weight-dependent while
// architecture/tactic matching remains weight-free.
Metadata build(const ModelDesc& model);

// Validates all topology/name/shape/scale/hash/byte bindings against masters.
// It rejects missing, duplicate, reordered, or extra records.
void validate(const ModelDesc& model, const Metadata& metadata);

// Canonical little-endian payload codec. Exposed so the training exporter and
// CPU tests can share golden bytes; production model parsing uses readTrailer.
std::vector<uint8_t> encodePayload(const Metadata& metadata);
Metadata decodePayload(const std::vector<uint8_t>& payload);

// Header wire:
// @KATAGO_QUANT_TRAILER@ 1 <payloadBytes> <payloadSha256> @BIN@<payload>
// No byte, including whitespace, is permitted after payload.
void writeTrailer(std::ostream& out, const Metadata& metadata);
Metadata readTrailer(std::istream& in, const ModelDesc& model);

}  // namespace NativeInt8Quant

#endif  // NEURALNET_NATIVEINT8QUANT_H_

#include "../neuralnet/nativeint8quant.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "../core/global.h"
#include "../core/sha2.h"
#include "../neuralnet/architecturedesc.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/renju15_int8_quantization.h"

using namespace std;

namespace NativeInt8Quant {
namespace {

constexpr char TRAILER_MARKER[] = "@KATAGO_QUANT_TRAILER@";
constexpr char BINARY_MARKER[] = "@BIN@";
constexpr uint8_t PAYLOAD_MAGIC[8] = {'K','Q','P','T','1','0','4',0};
constexpr uint32_t ENTRY_SCHEMA_VERSION = 1;
constexpr size_t MAX_PAYLOAD_BYTES = 64U * 1024U * 1024U;
constexpr size_t MAX_NAME_BYTES = 4096;
constexpr uint32_t MAX_NAMES_PER_ENTRY = 2;

uint32_t floatBits(float value) {
  static_assert(sizeof(float) == sizeof(uint32_t),"");
  uint32_t result;
  memcpy(&result,&value,sizeof(result));
  return result;
}

array<uint8_t,32> sha256(const uint8_t* data, size_t size) {
  array<uint8_t,32> result{};
  SHA2::get256(data,size,result.data());
  return result;
}

array<uint8_t,32> sha256(const vector<uint8_t>& bytes) {
  return sha256(bytes.data(),bytes.size());
}

array<uint8_t,32> sha256(const vector<int8_t>& bytes) {
  return sha256((const uint8_t*)bytes.data(),bytes.size());
}

class Writer {
 public:
  vector<uint8_t> bytes;

  void raw(const void* data, size_t size) {
    const uint8_t* p = (const uint8_t*)data;
    bytes.insert(bytes.end(),p,p+size);
  }
  void u32(uint32_t value) {
    for(int i = 0; i < 4; i++)
      bytes.push_back((uint8_t)(value >> (8*i)));
  }
  void i32(int32_t value) { u32((uint32_t)value); }
  void u64(uint64_t value) {
    for(int i = 0; i < 8; i++)
      bytes.push_back((uint8_t)(value >> (8*i)));
  }
  void stringValue(const string& value) {
    if(value.empty() || value.size() > MAX_NAME_BYTES ||
       value.find('\0') != string::npos)
      throw StringError("Native INT8 quantization has an invalid layer name");
    u32((uint32_t)value.size());
    raw(value.data(),value.size());
  }
};

class Reader {
 public:
  Reader(const uint8_t* data, size_t size, const string& context_)
    : ptr(data), end(data+size), context(context_) {}

  size_t remaining() const { return (size_t)(end-ptr); }
  bool empty() const { return ptr == end; }
  const uint8_t* position() const { return ptr; }

  void require(size_t size, const string& field) const {
    if(size > remaining())
      throw StringError(context + ": truncated " + field);
  }
  void raw(void* dest, size_t size, const string& field) {
    require(size,field);
    memcpy(dest,ptr,size);
    ptr += size;
  }
  void skip(size_t size, const string& field) {
    require(size,field);
    ptr += size;
  }
  uint32_t u32(const string& field) {
    require(4,field);
    uint32_t result = 0;
    for(int i = 0; i < 4; i++)
      result |= (uint32_t)ptr[i] << (8*i);
    ptr += 4;
    return result;
  }
  int32_t i32(const string& field) { return (int32_t)u32(field); }
  uint64_t u64(const string& field) {
    require(8,field);
    uint64_t result = 0;
    for(int i = 0; i < 8; i++)
      result |= (uint64_t)ptr[i] << (8*i);
    ptr += 8;
    return result;
  }
  string stringValue(const string& field) {
    uint32_t size = u32(field + " length");
    if(size == 0 || size > MAX_NAME_BYTES)
      throw StringError(context + ": invalid " + field + " length");
    require(size,field);
    string result((const char*)ptr,(size_t)size);
    ptr += size;
    if(result.find('\0') != string::npos)
      throw StringError(context + ": " + field + " contains NUL");
    return result;
  }

 private:
  const uint8_t* ptr;
  const uint8_t* end;
  string context;
};

struct Target {
  uint32_t topologyIndex;
  Role role;
  const MatMulLayerDesc* first;
  const MatMulLayerDesc* second;
};

struct TransformerBlockRef {
  int kind;
  const void* desc;
};

void collectTransformerBlocks(
  const vector<pair<int,unique_ptr_void>>& blocks,
  vector<TransformerBlockRef>& result
) {
  for(const auto& block: blocks) {
    if(block.second.get() == nullptr)
      throw StringError("Native INT8 quantization found a null trunk block");
    if(block.first == TRANSFORMER_ATTENTION_BLOCK_KIND ||
       block.first == TRANSFORMER_FFN_BLOCK_KIND) {
      result.push_back(TransformerBlockRef{block.first,block.second.get()});
    }
    else if(block.first == NESTED_BOTTLENECK_BLOCK_KIND) {
      const NestedBottleneckResidualBlockDesc* nested =
        (const NestedBottleneckResidualBlockDesc*)block.second.get();
      collectTransformerBlocks(nested->blocks,result);
    }
  }
}

vector<Target> collectTargets(const ModelDesc& model) {
  if(model.onnxHeader.isOnnx)
    throw StringError("Native INT8 quantization metadata is not supported for ONNX models");

  vector<TransformerBlockRef> blocks;
  collectTransformerBlocks(model.trunk.blocks,blocks);
  const NeuralNetArchitecture::ArchitectureDesc architecture =
    NeuralNetArchitecture::buildArchitectureDesc(model);
  vector<const NeuralNetArchitecture::ArchitectureOpDesc*> transformerOps;
  for(const auto& op: architecture.operators) {
    if(op.kind == NeuralNetArchitecture::ArchitectureOpKind::TransformerAttention ||
       op.kind == NeuralNetArchitecture::ArchitectureOpKind::TransformerFFN)
      transformerOps.push_back(&op);
  }
  if(transformerOps.size() != blocks.size())
    throw StringError("Native INT8 quantization topology mapping is inconsistent");

  vector<Target> targets;
  size_t numAttention = 0;
  size_t numFfn = 0;
  for(size_t i = 0; i < blocks.size(); i++) {
    const auto& op = *transformerOps[i];
    if(blocks[i].kind == TRANSFORMER_ATTENTION_BLOCK_KIND) {
      if(op.kind != NeuralNetArchitecture::ArchitectureOpKind::TransformerAttention)
        throw StringError("Native INT8 quantization attention topology kind mismatch");
      const TransformerAttentionDesc* desc =
        (const TransformerAttentionDesc*)blocks[i].desc;
      targets.push_back(Target{op.topologyIndex,Role::QK,&desc->qProj,&desc->kProj});
      numAttention++;
    }
    else {
      if(op.kind != NeuralNetArchitecture::ArchitectureOpKind::TransformerFFN)
        throw StringError("Native INT8 quantization FFN topology kind mismatch");
      const TransformerFFNDesc* desc = (const TransformerFFNDesc*)blocks[i].desc;
      if(!desc->useSwiGLU)
        throw StringError("Native INT8 quantization requires SwiGLU FFN blocks");
      targets.push_back(Target{op.topologyIndex,Role::FfnUp,&desc->linear1,nullptr});
      targets.push_back(Target{op.topologyIndex,Role::FfnGate,&desc->linearGate,nullptr});
      numFfn++;
    }
  }
  if(numAttention != 24 || numFfn != 24 || targets.size() != REQUIRED_ENTRY_COUNT)
    throw StringError(
      "Native INT8 v104 requires exactly 24 attention and 24 SwiGLU FFN blocks (72 entries)");
  return targets;
}

vector<float> canonicalMaster(const Target& target, uint32_t& k, uint32_t& n) {
  const MatMulLayerDesc& first = *target.first;
  if(first.inChannels <= 0 || first.outChannels <= 0 ||
     first.weights.size() != (size_t)first.inChannels * first.outChannels)
    throw StringError("Native INT8 quantization matrix has invalid dimensions");
  k = (uint32_t)first.inChannels;
  if(target.second == nullptr) {
    n = (uint32_t)first.outChannels;
    return first.weights;
  }

  const MatMulLayerDesc& second = *target.second;
  if(second.inChannels != first.inChannels || second.outChannels <= 0 ||
     second.weights.size() != (size_t)second.inChannels * second.outChannels)
    throw StringError("Native INT8 quantization Q/K dimensions are incompatible");
  n = (uint32_t)(first.outChannels + second.outChannels);
  vector<float> result((size_t)k*n);
  for(uint32_t inner = 0; inner < k; inner++) {
    const size_t destination = (size_t)inner*n;
    const size_t firstSource = (size_t)inner*first.outChannels;
    const size_t secondSource = (size_t)inner*second.outChannels;
    copy(
      first.weights.begin()+firstSource,
      first.weights.begin()+firstSource+first.outChannels,
      result.begin()+destination);
    copy(
      second.weights.begin()+secondSource,
      second.weights.begin()+secondSource+second.outChannels,
      result.begin()+destination+first.outChannels);
  }
  return result;
}

vector<uint8_t> canonicalFloatBytes(const vector<float>& weights) {
  static_assert(numeric_limits<float>::is_iec559,
    "Native INT8 quantization requires IEEE-754 float32");
  vector<uint8_t> result;
  result.reserve(weights.size()*4);
  for(float value: weights) {
    if(!isfinite(value))
      throw StringError("Native INT8 quantization master contains NaN or infinity");
    uint32_t bits = floatBits(value);
    for(int i = 0; i < 4; i++)
      result.push_back((uint8_t)(bits >> (8*i)));
  }
  return result;
}

Entry buildEntry(const Target& target) {
  uint32_t k;
  uint32_t n;
  vector<float> weights = canonicalMaster(target,k,n);
  const float scale = Renju15Int8Quantization::perMatrixScale(weights);
  vector<int8_t> packed = Renju15Int8Quantization::quantizeAndPackMatrix(
    weights,(int)k,(int)n,scale);
  vector<uint8_t> masterBytes = canonicalFloatBytes(weights);

  Entry entry{};
  entry.topologyIndex = target.topologyIndex;
  entry.role = target.role;
  entry.layout = PackedLayout::OutputMajorKContiguous;
  entry.inputChannels = k;
  entry.outputChannels = n;
  entry.zeroPoint = 0;
  entry.activationScaleBits = floatBits(4.0f/127.0f);
  entry.weightScaleBits = floatBits(scale);
  entry.layerNames.push_back(target.first->name);
  if(target.second != nullptr)
    entry.layerNames.push_back(target.second->name);
  entry.masterSha256 = sha256(masterBytes);
  entry.packedSha256 = sha256(packed);
  entry.packedWeights = std::move(packed);
  return entry;
}

bool equalEntry(const Entry& a, const Entry& b) {
  return a.topologyIndex == b.topologyIndex &&
    a.role == b.role && a.layout == b.layout &&
    a.inputChannels == b.inputChannels && a.outputChannels == b.outputChannels &&
    a.zeroPoint == b.zeroPoint &&
    a.activationScaleBits == b.activationScaleBits &&
    a.weightScaleBits == b.weightScaleBits &&
    a.layerNames == b.layerNames &&
    a.masterSha256 == b.masterSha256 &&
    a.packedSha256 == b.packedSha256 &&
    a.packedWeights == b.packedWeights;
}

string decimalToken(uint64_t value) {
  return Global::uint64ToString(value);
}

uint64_t parseCanonicalDecimal(const string& token, const string& field) {
  if(token.empty() || (token.size() > 1 && token[0] == '0'))
    throw StringError("Native INT8 quantization trailer has invalid " + field);
  uint64_t value = 0;
  for(char c: token) {
    if(c < '0' || c > '9')
      throw StringError("Native INT8 quantization trailer has invalid " + field);
    const uint32_t digit = (uint32_t)(c-'0');
    if(value > (numeric_limits<uint64_t>::max()-digit)/10)
      throw StringError("Native INT8 quantization trailer " + field + " overflows");
    value = value*10 + digit;
  }
  return value;
}

void requireCanonicalSha(const string& value) {
  if(value.size() != 64)
    throw StringError("Native INT8 quantization trailer has invalid payload SHA256");
  for(char c: value) {
    if(!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
      throw StringError("Native INT8 quantization trailer has noncanonical payload SHA256");
  }
}

}  // namespace

Metadata::Metadata()
  : payloadSchemaVersion(0),
    activationMode(QuantizationMode::SymmetricSignedInt8PerTensor),
    weightMode(QuantizationMode::SymmetricSignedInt8PerTensor),
    roundingMode(RoundingMode::RoundTiesToEvenSaturate127),
    zeroPoint(0),
    activationClipBits(0),
    activationScaleBits(0) {}

bool Metadata::present() const {
  return payloadSchemaVersion != 0;
}

Metadata build(const ModelDesc& model) {
  Metadata metadata;
  metadata.payloadSchemaVersion = PAYLOAD_SCHEMA_VERSION;
  metadata.activationMode = QuantizationMode::SymmetricSignedInt8PerTensor;
  metadata.weightMode = QuantizationMode::SymmetricSignedInt8PerTensor;
  metadata.roundingMode = RoundingMode::RoundTiesToEvenSaturate127;
  metadata.zeroPoint = 0;
  metadata.activationClipBits = floatBits(4.0f);
  metadata.activationScaleBits = floatBits(4.0f/127.0f);
  const vector<Target> targets = collectTargets(model);
  metadata.entries.reserve(targets.size());
  for(const Target& target: targets)
    metadata.entries.push_back(buildEntry(target));
  return metadata;
}

void validate(const ModelDesc& model, const Metadata& metadata) {
  if(metadata.payloadSchemaVersion != PAYLOAD_SCHEMA_VERSION ||
     metadata.activationMode != QuantizationMode::SymmetricSignedInt8PerTensor ||
     metadata.weightMode != QuantizationMode::SymmetricSignedInt8PerTensor ||
     metadata.roundingMode != RoundingMode::RoundTiesToEvenSaturate127 ||
     metadata.zeroPoint != 0 ||
     metadata.activationClipBits != floatBits(4.0f) ||
     metadata.activationScaleBits != floatBits(4.0f/127.0f))
    throw StringError("Native INT8 quantization global contract mismatch");
  if(metadata.entries.size() != REQUIRED_ENTRY_COUNT)
    throw StringError("Native INT8 quantization must contain exactly 72 entries");

  Metadata expected = build(model);
  for(size_t i = 0; i < expected.entries.size(); i++) {
    if(!equalEntry(metadata.entries[i],expected.entries[i]))
      throw StringError(
        "Native INT8 quantization entry " + Global::uint64ToString(i) +
        " does not match its canonical FP32 master/topology binding");
  }
}

vector<uint8_t> encodePayload(const Metadata& metadata) {
  if(metadata.entries.size() > numeric_limits<uint32_t>::max())
    throw StringError("Native INT8 quantization has too many entries");
  Writer out;
  out.raw(PAYLOAD_MAGIC,sizeof(PAYLOAD_MAGIC));
  out.u32(metadata.payloadSchemaVersion);
  out.u32((uint32_t)metadata.entries.size());
  out.u32((uint32_t)metadata.activationMode);
  out.u32((uint32_t)metadata.weightMode);
  out.u32((uint32_t)metadata.roundingMode);
  out.i32(metadata.zeroPoint);
  out.u32(metadata.activationClipBits);
  out.u32(metadata.activationScaleBits);
  out.u32(0);
  out.u32(0);

  for(const Entry& entry: metadata.entries) {
    if(entry.layerNames.empty() || entry.layerNames.size() > MAX_NAMES_PER_ENTRY)
      throw StringError("Native INT8 quantization entry has invalid name count");
    if(entry.packedWeights.size() !=
       (size_t)entry.inputChannels*entry.outputChannels)
      throw StringError("Native INT8 quantization entry has invalid packed byte count");
    Writer record;
    record.u32(ENTRY_SCHEMA_VERSION);
    record.u32(entry.topologyIndex);
    record.u32((uint32_t)entry.role);
    record.u32((uint32_t)entry.layout);
    record.u32(entry.inputChannels);
    record.u32(entry.outputChannels);
    record.i32(entry.zeroPoint);
    record.u32(entry.activationScaleBits);
    record.u32(entry.weightScaleBits);
    record.u32((uint32_t)entry.layerNames.size());
    record.u64((uint64_t)entry.packedWeights.size());
    for(const string& name: entry.layerNames)
      record.stringValue(name);
    record.raw(entry.masterSha256.data(),entry.masterSha256.size());
    record.raw(entry.packedSha256.data(),entry.packedSha256.size());
    record.raw(entry.packedWeights.data(),entry.packedWeights.size());
    out.u64((uint64_t)record.bytes.size());
    out.raw(record.bytes.data(),record.bytes.size());
  }
  if(out.bytes.size() > MAX_PAYLOAD_BYTES)
    throw StringError("Native INT8 quantization payload exceeds size limit");
  return out.bytes;
}

Metadata decodePayload(const vector<uint8_t>& payload) {
  if(payload.empty() || payload.size() > MAX_PAYLOAD_BYTES)
    throw StringError("Native INT8 quantization payload has invalid size");
  Reader in(payload.data(),payload.size(),"Native INT8 quantization payload");
  uint8_t magic[sizeof(PAYLOAD_MAGIC)];
  in.raw(magic,sizeof(magic),"magic");
  if(memcmp(magic,PAYLOAD_MAGIC,sizeof(magic)) != 0)
    throw StringError("Native INT8 quantization payload magic mismatch");

  Metadata metadata;
  metadata.payloadSchemaVersion = in.u32("schema version");
  if(metadata.payloadSchemaVersion != PAYLOAD_SCHEMA_VERSION)
    throw StringError("Unsupported native INT8 quantization payload schema");
  const uint32_t entryCount = in.u32("entry count");
  if(entryCount != REQUIRED_ENTRY_COUNT)
    throw StringError("Native INT8 quantization payload must contain exactly 72 entries");
  metadata.activationMode = (QuantizationMode)in.u32("activation mode");
  metadata.weightMode = (QuantizationMode)in.u32("weight mode");
  metadata.roundingMode = (RoundingMode)in.u32("rounding mode");
  metadata.zeroPoint = in.i32("zero point");
  metadata.activationClipBits = in.u32("activation clip bits");
  metadata.activationScaleBits = in.u32("activation scale bits");
  if(in.u32("reserved field 0") != 0 || in.u32("reserved field 1") != 0)
    throw StringError("Native INT8 quantization payload reserved field is nonzero");

  metadata.entries.reserve(entryCount);
  for(uint32_t i = 0; i < entryCount; i++) {
    const uint64_t recordBytes = in.u64("entry byte count");
    if(recordBytes > in.remaining())
      throw StringError("Native INT8 quantization payload has truncated entry");
    Reader record(
      in.position(),(size_t)recordBytes,
      "Native INT8 quantization entry " + Global::uint64ToString(i));
    in.skip((size_t)recordBytes,"entry");

    if(record.u32("schema version") != ENTRY_SCHEMA_VERSION)
      throw StringError("Unsupported native INT8 quantization entry schema");
    Entry entry{};
    entry.topologyIndex = record.u32("topology index");
    entry.role = (Role)record.u32("role");
    entry.layout = (PackedLayout)record.u32("layout");
    entry.inputChannels = record.u32("K");
    entry.outputChannels = record.u32("N");
    entry.zeroPoint = record.i32("zero point");
    entry.activationScaleBits = record.u32("activation scale bits");
    entry.weightScaleBits = record.u32("weight scale bits");
    const uint32_t nameCount = record.u32("name count");
    const uint64_t packedCount = record.u64("packed byte count");
    if(nameCount == 0 || nameCount > MAX_NAMES_PER_ENTRY)
      throw StringError("Native INT8 quantization entry has invalid name count");
    if(entry.inputChannels == 0 || entry.outputChannels == 0 ||
       (uint64_t)entry.inputChannels*entry.outputChannels != packedCount ||
       packedCount > MAX_PAYLOAD_BYTES)
      throw StringError("Native INT8 quantization entry has invalid dimensions/byte count");
    for(uint32_t name = 0; name < nameCount; name++)
      entry.layerNames.push_back(record.stringValue("layer name"));
    record.raw(entry.masterSha256.data(),entry.masterSha256.size(),"master SHA256");
    record.raw(entry.packedSha256.data(),entry.packedSha256.size(),"packed SHA256");
    entry.packedWeights.resize((size_t)packedCount);
    record.raw(entry.packedWeights.data(),entry.packedWeights.size(),"packed weights");
    if(!record.empty())
      throw StringError("Native INT8 quantization entry has trailing bytes");
    metadata.entries.push_back(std::move(entry));
  }
  if(!in.empty())
    throw StringError("Native INT8 quantization payload has trailing bytes");
  return metadata;
}

void writeTrailer(ostream& out, const Metadata& metadata) {
  const vector<uint8_t> payload = encodePayload(metadata);
  char sha[65];
  SHA2::get256(payload.data(),payload.size(),sha);
  out << TRAILER_MARKER << " " << TRAILER_HEADER_SCHEMA_VERSION << " "
      << decimalToken(payload.size()) << " " << sha << " " << BINARY_MARKER;
  out.write((const char*)payload.data(),payload.size());
  if(!out.good())
    throw StringError("Failed writing native INT8 quantization trailer");
}

Metadata readTrailer(istream& in, const ModelDesc& model) {
  string marker;
  string headerSchema;
  string payloadSizeToken;
  string expectedSha;
  in >> marker >> headerSchema >> payloadSizeToken >> expectedSha;
  if(in.fail() || marker != TRAILER_MARKER)
    throw StringError("Model v104 is missing mandatory native INT8 quantization trailer");
  if(headerSchema != decimalToken(TRAILER_HEADER_SCHEMA_VERSION))
    throw StringError("Unsupported native INT8 quantization trailer header schema");
  char separator = 0;
  char binaryMarker[sizeof(BINARY_MARKER)] = {};
  in.get(separator);
  in.read(binaryMarker,sizeof(BINARY_MARKER)-1);
  if(separator != ' ' || in.fail() || string(binaryMarker) != BINARY_MARKER)
    throw StringError("Native INT8 quantization trailer is missing @BIN@ marker");
  requireCanonicalSha(expectedSha);
  const uint64_t payloadSize = parseCanonicalDecimal(payloadSizeToken,"payload byte count");
  if(payloadSize == 0 || payloadSize > MAX_PAYLOAD_BYTES)
    throw StringError("Native INT8 quantization trailer payload size is out of range");
  vector<uint8_t> payload((size_t)payloadSize);
  in.read((char*)payload.data(),payload.size());
  if(in.gcount() != (streamsize)payload.size())
    throw StringError("Native INT8 quantization trailer payload is truncated");
  // A read ending exactly at EOF may set eofbit while still returning every
  // requested byte. Only badbit is an I/O failure here; strict EOF is checked
  // separately below.
  if(in.bad())
    throw StringError("Native INT8 quantization trailer payload read failed");
  if(in.peek() != char_traits<char>::eof())
    throw StringError("Model v104 has bytes after its native INT8 quantization payload");

  char actualSha[65];
  SHA2::get256(payload.data(),payload.size(),actualSha);
  if(expectedSha != actualSha)
    throw StringError("Native INT8 quantization trailer payload SHA256 mismatch");
  Metadata metadata = decodePayload(payload);
  validate(model,metadata);
  return metadata;
}

}  // namespace NativeInt8Quant

#include "model.h"

#include "../../core/fileutils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

using std::string;
using std::vector;

namespace CpuPtq {
namespace {

constexpr char MAGIC[16] = {
  'K','A','T','A','G','O','C','P','U','P','T','Q','2','0','6','\0'
};
constexpr uint32_t WIRE_REVISION = 1;
constexpr uint8_t DTYPE_FP32 = 1;
constexpr uint8_t DTYPE_S8_PER_OUTPUT = 2;

[[noreturn]] void fail(const string& message) {
  throw StringError("CPU-PTQ v206: " + message);
}

class Reader {
 public:
  explicit Reader(const string& bytes)
    : current(reinterpret_cast<const uint8_t*>(bytes.data())),
      end(current + bytes.size()) {}

  size_t remaining() const { return static_cast<size_t>(end - current); }

  const uint8_t* bytes(size_t count, const string& field) {
    if(count > remaining())
      fail("truncated " + field);
    const uint8_t* result = current;
    current += count;
    return result;
  }

  uint8_t u8(const string& field) {
    return *bytes(1,field);
  }

  uint16_t u16(const string& field) {
    const uint8_t* value = bytes(2,field);
    return static_cast<uint16_t>(value[0]) |
      static_cast<uint16_t>(value[1]) << 8;
  }

  uint32_t u32(const string& field) {
    const uint8_t* value = bytes(4,field);
    return static_cast<uint32_t>(value[0]) |
      static_cast<uint32_t>(value[1]) << 8 |
      static_cast<uint32_t>(value[2]) << 16 |
      static_cast<uint32_t>(value[3]) << 24;
  }

  string text(size_t count, const string& field) {
    const uint8_t* value = bytes(count,field);
    if(std::find(value,value + count,0) != value + count)
      fail(field + " contains a NUL byte");
    return string(reinterpret_cast<const char*>(value),count);
  }

  vector<float> fp32(size_t count, const string& field) {
    if(count > std::numeric_limits<size_t>::max() / sizeof(float))
      fail(field + " is too large");
    const uint8_t* raw = bytes(count * sizeof(float),field);
    vector<float> result(count);
    std::memcpy(result.data(),raw,count * sizeof(float));
    for(float value: result) {
      if(!std::isfinite(value))
        fail(field + " contains a non-finite value");
    }
    return result;
  }

 private:
  const uint8_t* current;
  const uint8_t* end;
};

size_t elementCount(const vector<uint32_t>& shape, const string& name) {
  size_t count = 1;
  for(uint32_t dimension: shape) {
    if(dimension == 0 || count > 100000000U / dimension)
      fail(name + " has an invalid shape");
    count *= dimension;
  }
  return count;
}

const ProfileSpec& selectProfile(
  uint32_t blocks,
  uint32_t channels,
  uint32_t heads,
  uint32_t ffn,
  uint32_t valueHidden
) {
  const ProfileSpec* profiles[] = {&b11Profile(),&b16Profile()};
  for(const ProfileSpec* profile: profiles) {
    if(
      blocks == static_cast<uint32_t>(profile->blocks) &&
      channels == static_cast<uint32_t>(profile->channels) &&
      heads == static_cast<uint32_t>(profile->heads) &&
      ffn == static_cast<uint32_t>(profile->ffnChannels) &&
      valueHidden == static_cast<uint32_t>(profile->valueHiddenChannels)
    )
      return *profile;
  }
  fail("model geometry does not match b11c96h3-f256 or b16c128h4-f384");
}

}  // namespace

const ProfileSpec& b11Profile() {
  static const ProfileSpec profile = {
    ProfileKind::B11C96H3F256,"b11c96h3-f256",11,96,3,32,256,64
  };
  return profile;
}

const ProfileSpec& b16Profile() {
  static const ProfileSpec profile = {
    ProfileKind::B16C128H4F384,"b16c128h4-f384",16,128,4,32,384,64
  };
  return profile;
}

Model loadModelFile(const string& fileName, const string& expectedSha256) {
  if(!Global::isSuffix(Global::toLower(fileName),".bin.gz"))
    fail("model file must end in .bin.gz");

  string payload;
  string actualSha256;
  FileUtils::uncompressAndLoadFileIntoString(
    fileName,expectedSha256,payload,&actualSha256);
  Reader reader(payload);
  const uint8_t* magic = reader.bytes(sizeof(MAGIC),"magic");
  if(std::memcmp(magic,MAGIC,sizeof(MAGIC)) != 0)
    fail("bad model magic");
  if(reader.u32("wire revision") != WIRE_REVISION)
    fail("unsupported wire revision");
  if(reader.u32("model version") != MODEL_VERSION)
    fail("model version is not 206");
  if(reader.u32("source model version") != SOURCE_MODEL_VERSION)
    fail("source model version is not 11");
  if(reader.u32("board length") != BOARD_LEN)
    fail("board length is not 7");
  const uint32_t blocks = reader.u32("blocks");
  const uint32_t channels = reader.u32("channels");
  const uint32_t heads = reader.u32("heads");
  const uint32_t ffn = reader.u32("ffn channels");
  const uint32_t valueHidden = reader.u32("value hidden channels");
  const ProfileSpec& profile = selectProfile(
    blocks,channels,heads,ffn,valueHidden);

  const uint32_t nameBytes = reader.u32("model name length");
  if(nameBytes == 0 || nameBytes > 4096)
    fail("invalid model name length");
  Model model;
  model.name = reader.text(nameBytes,"model name");
  model.sha256 = actualSha256;
  model.profile = &profile;

  const uint32_t tensorCount = reader.u32("tensor count");
  if(tensorCount == 0 || tensorCount > 4096)
    fail("invalid tensor count");
  model.tensors.reserve(tensorCount);
  for(uint32_t index = 0; index < tensorCount; index++) {
    const uint16_t nameLength = reader.u16("tensor name length");
    const uint8_t rawKind = reader.u8("tensor kind");
    const uint8_t rank = reader.u8("tensor rank");
    if(nameLength == 0 || rank == 0 || rank > 4)
      fail("invalid tensor header");
    const string name = reader.text(nameLength,"tensor name");
    Tensor tensor;
    if(rawKind == DTYPE_FP32)
      tensor.kind = TensorKind::FP32;
    else if(rawKind == DTYPE_S8_PER_OUTPUT)
      tensor.kind = TensorKind::S8PerOutput;
    else
      fail(name + " has an unknown tensor kind");
    tensor.shape.reserve(rank);
    for(uint8_t dimension = 0; dimension < rank; dimension++)
      tensor.shape.push_back(reader.u32(name + " shape"));
    const size_t elements = elementCount(tensor.shape,name);
    if(tensor.kind == TensorKind::FP32) {
      tensor.values = reader.fp32(elements,name);
    }
    else {
      if(rank != 2 || reader.u32(name + " qmax") != 127)
        fail(name + " is not canonical per-output S8");
      tensor.scales = reader.fp32(tensor.shape[0],name + " scales");
      for(float scale: tensor.scales) {
        if(!(scale > 0.0f))
          fail(name + " has a nonpositive scale");
      }
      const uint8_t* codes = reader.bytes(elements,name + " codes");
      tensor.codes.resize(elements);
      std::memcpy(tensor.codes.data(),codes,elements);
      if(std::find(tensor.codes.begin(),tensor.codes.end(),INT8_MIN) != tensor.codes.end())
        fail(name + " contains forbidden -128 code");
    }
    if(!model.tensors.emplace(name,std::move(tensor)).second)
      fail("duplicate tensor " + name);
  }
  if(reader.remaining() != 0)
    fail("trailing bytes after tensor table");
  return model;
}

const Tensor& requireTensor(
  const Model& model,
  const string& name,
  TensorKind kind,
  std::initializer_list<uint32_t> shape
) {
  const auto found = model.tensors.find(name);
  if(found == model.tensors.end())
    fail("missing tensor " + name);
  if(found->second.kind != kind)
    fail("tensor kind mismatch for " + name);
  if(found->second.shape != vector<uint32_t>(shape))
    fail("tensor shape mismatch for " + name);
  return found->second;
}

const vector<float>& requireFP32(
  const Model& model,
  const string& name,
  std::initializer_list<uint32_t> shape
) {
  return requireTensor(model,name,TensorKind::FP32,shape).values;
}

const Tensor& requireS8(
  const Model& model,
  const string& name,
  std::initializer_list<uint32_t> shape
) {
  return requireTensor(model,name,TensorKind::S8PerOutput,shape);
}

}  // namespace CpuPtq

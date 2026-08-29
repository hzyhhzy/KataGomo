#include "model.h"

#include "../../core/fileutils.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

using std::string;
using std::vector;

namespace CpuPtq {
namespace {

[[noreturn]] void fail(const string& message) {
  throw StringError("CPU-PTQ native v205/v206: " + message);
}

void require(bool condition, const string& message) {
  if(!condition)
    fail(message);
}

class Reader {
 public:
  explicit Reader(const string& bytes)
    : current(reinterpret_cast<const uint8_t*>(bytes.data())),
      end(current + bytes.size()) {}

  string token(const string& field) {
    skipWhitespace();
    if(current == end)
      fail("unexpected EOF while reading " + field);
    const uint8_t* start = current;
    while(current != end && !std::isspace(static_cast<unsigned char>(*current)))
      current++;
    if(start == current)
      fail("empty token for " + field);
    return string(reinterpret_cast<const char*>(start),
                  static_cast<size_t>(current - start));
  }

  void expect(const string& expected, const string& field) {
    const string actual = token(field);
    if(actual != expected)
      fail(field + ": expected " + expected + ", got " + actual);
  }

  int integer(const string& field) {
    const string text = token(field);
    size_t parsed = 0;
    int value;
    try {
      value = std::stoi(text,&parsed);
    }
    catch(const std::exception&) {
      fail(field + " is not an integer");
    }
    if(parsed != text.size())
      fail(field + " is not a canonical integer");
    return value;
  }

  float floating(const string& field) {
    const string text = token(field);
    size_t parsed = 0;
    float value;
    try {
      value = std::stof(text,&parsed);
    }
    catch(const std::exception&) {
      fail(field + " is not a float");
    }
    if(parsed != text.size() || !std::isfinite(value))
      fail(field + " is not a finite canonical float");
    return value;
  }

  bool flag(const string& field) {
    const int value = integer(field);
    if(value != 0 && value != 1)
      fail(field + " must be zero or one");
    return value != 0;
  }

  string marker(const string& field) {
    skipWhitespace();
    if(static_cast<size_t>(end - current) < 5)
      fail("truncated marker for " + field);
    const string result(reinterpret_cast<const char*>(current),5);
    current += 5;
    return result;
  }

  vector<float> rawFP32(size_t count, const string& field) {
    if(count > std::numeric_limits<size_t>::max() / sizeof(float) ||
       count * sizeof(float) > static_cast<size_t>(end - current))
      fail("truncated FP32 block for " + field);
    vector<float> result(count);
    std::memcpy(result.data(),current,count * sizeof(float));
    current += count * sizeof(float);
    for(float value: result)
      if(!std::isfinite(value))
        fail(field + " contains a non-finite value");
    return result;
  }

  vector<float> fp32(size_t count, const string& field) {
    if(marker(field) != "@BIN@")
      fail(field + " does not use canonical @BIN@ FP32 storage");
    return rawFP32(count,field);
  }

  vector<int8_t> rawS8(size_t count, const string& field) {
    if(count > static_cast<size_t>(end - current))
      fail("truncated S8 block for " + field);
    vector<int8_t> result(count);
    std::memcpy(result.data(),current,count);
    current += count;
    return result;
  }

  void requireEnd() {
    skipWhitespace();
    if(current != end)
      fail("trailing bytes after native model body");
  }

 private:
  void skipWhitespace() {
    while(current != end && std::isspace(static_cast<unsigned char>(*current)))
      current++;
  }

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

vector<float> inputMajorToOutputMajor(
  const vector<float>& values,
  int inputs,
  int outputs
) {
  require(values.size() == static_cast<size_t>(inputs) * outputs,
          "matmul element count mismatch");
  vector<float> result(values.size());
  for(int input = 0; input < inputs; input++)
    for(int output = 0; output < outputs; output++)
      result[static_cast<size_t>(output) * inputs + input] =
        values[static_cast<size_t>(input) * outputs + output];
  return result;
}

void addTensor(Model& model, const string& name, Tensor tensor) {
  const size_t count = elementCount(tensor.shape,name);
  if(tensor.kind == TensorKind::FP32) {
    require(tensor.quantizedMax == 0 && tensor.values.size() == count,
            name + " has malformed FP32 storage");
    require(tensor.codes.empty() && tensor.scales.empty(),
            name + " mixes FP32 and quantized storage");
  }
  else {
    require(tensor.shape.size() == 2 && tensor.values.empty(),
            name + " has malformed quantized geometry");
    require(tensor.quantizedMax == 63 || tensor.quantizedMax == 127,
            name + " has an invalid qmax");
    require(tensor.codes.size() == count &&
            tensor.scales.size() == tensor.shape[0],
            name + " has malformed quantized storage");
  }
  if(!model.tensors.emplace(name,std::move(tensor)).second)
    fail("duplicate tensor " + name);
}

void addFP32(
  Model& model,
  const string& name,
  std::initializer_list<uint32_t> shape,
  vector<float> values
) {
  Tensor tensor;
  tensor.kind = TensorKind::FP32;
  tensor.quantizedMax = 0;
  tensor.shape.assign(shape.begin(),shape.end());
  tensor.values = std::move(values);
  addTensor(model,name,std::move(tensor));
}

vector<float> readMatMul(
  Reader& reader,
  const string& expectedName,
  int expectedInputs,
  int expectedOutputs
) {
  reader.expect(expectedName,expectedName + " name");
  const int inputs = reader.integer(expectedName + " inputs");
  const int outputs = reader.integer(expectedName + " outputs");
  require(inputs == expectedInputs && outputs == expectedOutputs,
          expectedName + " shape mismatch");
  return inputMajorToOutputMajor(
    reader.fp32(static_cast<size_t>(inputs) * outputs,expectedName),
    inputs,outputs);
}

vector<float> readBias(
  Reader& reader,
  const string& expectedName,
  int expectedChannels
) {
  reader.expect(expectedName,expectedName + " name");
  const int channels = reader.integer(expectedName + " channels");
  require(channels == expectedChannels,expectedName + " shape mismatch");
  return reader.fp32(static_cast<size_t>(channels),expectedName);
}

vector<float> readConv(
  Reader& reader,
  const string& expectedName,
  int expectedY,
  int expectedX,
  int expectedInputs,
  int expectedOutputs
) {
  reader.expect(expectedName,expectedName + " name");
  const int ySize = reader.integer(expectedName + " y size");
  const int xSize = reader.integer(expectedName + " x size");
  const int inputs = reader.integer(expectedName + " inputs");
  const int outputs = reader.integer(expectedName + " outputs");
  const int dilationY = reader.integer(expectedName + " y dilation");
  const int dilationX = reader.integer(expectedName + " x dilation");
  require(ySize == expectedY && xSize == expectedX &&
          inputs == expectedInputs && outputs == expectedOutputs &&
          dilationY == 1 && dilationX == 1,
          expectedName + " convolution geometry mismatch");
  const vector<float> source = reader.fp32(
    static_cast<size_t>(ySize) * xSize * inputs * outputs,expectedName);
  vector<float> result(source.size());
  for(int y = 0; y < ySize; y++)
    for(int x = 0; x < xSize; x++)
      for(int input = 0; input < inputs; input++)
        for(int output = 0; output < outputs; output++)
          result[((static_cast<size_t>(output) * inputs + input) * ySize + y) *
                 xSize + x] =
            source[((static_cast<size_t>(y) * xSize + x) * inputs + input) *
                   outputs + output];
  return result;
}

struct FoldedNorm {
  vector<float> multiplier;
  vector<float> bias;
};

FoldedNorm readBatchNorm(
  Reader& reader,
  const string& expectedName,
  int expectedChannels
) {
  reader.expect(expectedName,expectedName + " name");
  const int channels = reader.integer(expectedName + " channels");
  const float epsilon = reader.floating(expectedName + " epsilon");
  const bool hasScale = reader.flag(expectedName + " hasScale");
  const bool hasBias = reader.flag(expectedName + " hasBias");
  require(channels == expectedChannels && epsilon > 0.0f,
          expectedName + " normalization header mismatch");
  const vector<float> mean = reader.fp32(channels,expectedName + " mean");
  const vector<float> variance = reader.fp32(channels,expectedName + " variance");
  const vector<float> scale = hasScale ?
    reader.fp32(channels,expectedName + " scale") : vector<float>(channels,1.0f);
  const vector<float> beta = hasBias ?
    reader.fp32(channels,expectedName + " bias") : vector<float>(channels,0.0f);
  FoldedNorm result;
  result.multiplier.resize(channels);
  result.bias.resize(channels);
  for(int channel = 0; channel < channels; channel++) {
    const float denominator = std::sqrt(variance[channel] + epsilon);
    require(std::isfinite(denominator) && denominator > 0.0f,
            expectedName + " has an invalid variance");
    result.multiplier[channel] = scale[channel] / denominator;
    result.bias[channel] = beta[channel] -
      mean[channel] * result.multiplier[channel];
  }
  return result;
}

vector<float> readRmsNorm(
  Reader& reader,
  const string& expectedName,
  int expectedChannels
) {
  reader.expect(expectedName,expectedName + " name");
  const int channels = reader.integer(expectedName + " channels");
  const float epsilon = reader.floating(expectedName + " epsilon");
  require(channels == expectedChannels,expectedName + " shape mismatch");
  require(std::abs(epsilon - 1.0e-6f) <= 1.0e-12f,
          expectedName + " requires epsilon 1e-6");
  return reader.fp32(channels,expectedName);
}

void requireSilu(Reader& reader, const string& expectedName) {
  reader.expect(expectedName,expectedName + " name");
  reader.expect("ACTIVATION_SILU",expectedName + " kind");
}

Tensor readProjection(
  Reader& reader,
  const string& expectedName,
  int expectedInputs,
  int expectedOutputs,
  int modelVersion,
  int& uniformQmax
) {
  reader.expect(expectedName,expectedName + " name");
  const int inputs = reader.integer(expectedName + " inputs");
  const int outputs = reader.integer(expectedName + " outputs");
  require(inputs == expectedInputs && outputs == expectedOutputs,
          expectedName + " shape mismatch");
  const string marker = reader.marker(expectedName);
  Tensor tensor;
  tensor.shape = {
    static_cast<uint32_t>(outputs),static_cast<uint32_t>(inputs)
  };
  if(modelVersion == BASE_MODEL_VERSION) {
    require(marker == "@BIN@",expectedName + " v205 projection is not FP32");
    tensor.kind = TensorKind::FP32;
    tensor.quantizedMax = 0;
    tensor.values = inputMajorToOutputMajor(
      reader.rawFP32(static_cast<size_t>(inputs) * outputs,expectedName),
      inputs,outputs);
  }
  else {
    if(marker == "@S7P@")
      tensor.quantizedMax = 63;
    else if(marker == "@S8P@")
      tensor.quantizedMax = 127;
    else
      fail(expectedName + " v206 projection lacks @S7P@/@S8P@ marker");
    if(uniformQmax == 0)
      uniformQmax = tensor.quantizedMax;
    require(uniformQmax == tensor.quantizedMax,
            "one v206 model may not mix S7 and S8 projections");
    tensor.kind = TensorKind::S8PerOutput;
    tensor.scales = reader.rawFP32(outputs,expectedName + " scales");
    for(float scale: tensor.scales)
      require(scale > 0.0f,expectedName + " has a nonpositive scale");
    tensor.codes = reader.rawS8(
      static_cast<size_t>(inputs) * outputs,expectedName + " codes");
    for(int8_t code: tensor.codes)
      require(code >= -tensor.quantizedMax && code <= tensor.quantizedMax,
              expectedName + " code exceeds its declared qmax");
  }
  return tensor;
}

void requireUnitMultiplier(const FoldedNorm& norm, const string& name) {
  for(float value: norm.multiplier)
    require(std::abs(value - 1.0f) <= 2.0e-5f,
            name + " requires a unit multiplier");
}

const ProfileSpec& selectProfile(
  int blocks,
  int channels,
  int heads,
  int ffn,
  int valueHidden
) {
  const ProfileSpec* profiles[] = {&b11Profile(),&b16Profile()};
  for(const ProfileSpec* profile: profiles) {
    if(blocks == profile->blocks && channels == profile->channels &&
       heads == profile->heads && ffn == profile->ffnChannels &&
       valueHidden == profile->valueHiddenChannels)
      return *profile;
  }
  fail("model geometry does not match b11c96h3-f256 or b16c128h4-f384");
}

Model parseNativeModel(const string& payload, const string& sha256) {
  Reader reader(payload);
  Model model;
  model.name = reader.token("model name");
  model.version = reader.integer("model version");
  require(model.version == BASE_MODEL_VERSION || model.version == MODEL_VERSION,
          "model version must be v205 or v206");
  require(reader.integer("spatial input count") == SPATIAL_INPUTS,
          "spatial input count must be 22");
  require(reader.integer("global input count") == GLOBAL_INPUTS,
          "global input count must be 19");
  model.sha256 = sha256;

  reader.expect("trunk","trunk name");
  const int logicalBlocks = reader.integer("logical block count");
  const int channels = reader.integer("trunk channels");
  const int midChannels = reader.integer("mid channels");
  const int regularChannels = reader.integer("regular channels");
  const int dilatedChannels = reader.integer("dilated channels");
  const int gpoolChannels = reader.integer("gpool channels");
  require(logicalBlocks > 0 && logicalBlocks % 2 == 0,
          "logical block count must contain attention/FFN pairs");
  require(channels > 0 && midChannels > 0 && regularChannels > 0 &&
          dilatedChannels > 0 && gpoolChannels > 0,
          "trunk channel header is invalid");
  const int blocks = logicalBlocks / 2;
  addFP32(
    model,"stem.weight",{static_cast<uint32_t>(channels),SPATIAL_INPUTS,3,3},
    readConv(reader,"model.conv_spatial",3,3,SPATIAL_INPUTS,channels));
  addFP32(
    model,"global.weight",{static_cast<uint32_t>(channels),GLOBAL_INPUTS},
    readMatMul(reader,"model.linear_global",GLOBAL_INPUTS,channels));

  int heads = 0;
  int ffnChannels = 0;
  int uniformQmax = 0;
  for(int block = 0; block < blocks; block++) {
    const string nativePrefix = "model.blocks." + std::to_string(block) + ".";
    const string tensorPrefix = "blocks." + std::to_string(block) + ".";
    reader.expect("transformer_attention_block","attention block kind");
    reader.expect(nativePrefix + "attention","attention block name");
    const int blockHeads = reader.integer("attention heads");
    const int kvHeads = reader.integer("attention KV heads");
    const int qDim = reader.integer("query head dimension");
    const int vDim = reader.integer("value head dimension");
    const bool useRope = reader.flag("use RoPE");
    const bool learnableRope = reader.flag("learnable RoPE");
    require(blockHeads > 0 && kvHeads == blockHeads && qDim == 32 && vDim == 32 &&
            channels == blockHeads * qDim,
            "attention geometry mismatch");
    if(heads == 0)
      heads = blockHeads;
    require(heads == blockHeads,"attention head count changes between blocks");
    require(useRope && !learnableRope,
            "Ataxx CPU-PTQ requires fixed 2D RoPE");
    addFP32(
      model,tensorPrefix + "norm1",{static_cast<uint32_t>(channels)},
      readRmsNorm(reader,nativePrefix + "attention.norm1",channels));
    addTensor(model,tensorPrefix + "q_proj",readProjection(
      reader,nativePrefix + "attention.q_proj",channels,channels,
      model.version,uniformQmax));
    addTensor(model,tensorPrefix + "k_proj",readProjection(
      reader,nativePrefix + "attention.k_proj",channels,channels,
      model.version,uniformQmax));
    addTensor(model,tensorPrefix + "v_proj",readProjection(
      reader,nativePrefix + "attention.v_proj",channels,channels,
      model.version,uniformQmax));
    addTensor(model,tensorPrefix + "out_proj",readProjection(
      reader,nativePrefix + "attention.out_proj",channels,channels,
      model.version,uniformQmax));
    reader.expect(nativePrefix + "attention.rope_theta","RoPE theta name");
    require(std::abs(reader.floating("RoPE theta") - 100.0f) <= 1.0e-5f,
            "Ataxx CPU-PTQ requires RoPE theta 100");

    reader.expect("transformer_ffn_block","FFN block kind");
    reader.expect(nativePrefix + "ffn","FFN block name");
    require(reader.integer("FFN input channels") == channels,
            "FFN input channel mismatch");
    const int blockFfn = reader.integer("FFN channels");
    require(reader.flag("use SwiGLU"),"Ataxx CPU-PTQ requires SwiGLU");
    if(ffnChannels == 0)
      ffnChannels = blockFfn;
    require(ffnChannels == blockFfn,"FFN width changes between blocks");
    addFP32(
      model,tensorPrefix + "norm2",{static_cast<uint32_t>(channels)},
      readRmsNorm(reader,nativePrefix + "ffn.norm",channels));
    addTensor(model,tensorPrefix + "ffn_linear1",readProjection(
      reader,nativePrefix + "ffn.ffn_linear1",channels,ffnChannels,
      model.version,uniformQmax));
    addTensor(model,tensorPrefix + "ffn_linear_gate",readProjection(
      reader,nativePrefix + "ffn.ffn_linear_gate",channels,ffnChannels,
      model.version,uniformQmax));
    addTensor(model,tensorPrefix + "ffn_linear2",readProjection(
      reader,nativePrefix + "ffn.ffn_linear2",ffnChannels,channels,
      model.version,uniformQmax));
  }

  const FoldedNorm trunk = readBatchNorm(
    reader,"model.norm_trunkfinal",channels);
  requireSilu(reader,"model.act_trunkfinal");
  addFP32(model,"trunk.mul",{static_cast<uint32_t>(channels)},trunk.multiplier);
  addFP32(model,"trunk.bias",{static_cast<uint32_t>(channels)},trunk.bias);

  reader.expect("model.policy_head","policy head name");
  addFP32(
    model,"policy.conv1p",{32,static_cast<uint32_t>(channels)},
    readConv(reader,"model.policy_head.conv1p",1,1,channels,32));
  addFP32(
    model,"policy.conv1g",{32,static_cast<uint32_t>(channels)},
    readConv(reader,"model.policy_head.conv1g",1,1,channels,32));
  const FoldedNorm policyG = readBatchNorm(reader,"model.policy_head.biasg",32);
  requireUnitMultiplier(policyG,"policy gpool bias");
  requireSilu(reader,"model.policy_head.actg");
  addFP32(
    model,"policy.linear_g",{32,96},
    readMatMul(reader,"model.policy_head.linear_g",96,32));
  const FoldedNorm policyP = readBatchNorm(reader,"model.policy_head.bias2",32);
  requireUnitMultiplier(policyP,"policy spatial bias");
  requireSilu(reader,"model.policy_head.act2");
  addFP32(
    model,"policy.conv2p",{32},
    readConv(reader,"model.policy_head.conv2p",1,1,32,1));
  addFP32(
    model,"policy.linear_pass",{96},
    readMatMul(reader,"model.policy_head.linear_pass",96,1));
  addFP32(model,"policy.biasg",{32},policyG.bias);
  addFP32(model,"policy.bias2",{32},policyP.bias);

  reader.expect("model.value_head","value head name");
  addFP32(
    model,"value.conv1",{32,static_cast<uint32_t>(channels)},
    readConv(reader,"model.value_head.conv1",1,1,channels,32));
  const FoldedNorm valueSpatial = readBatchNorm(reader,"model.value_head.bias1",32);
  requireUnitMultiplier(valueSpatial,"value spatial bias");
  requireSilu(reader,"model.value_head.act1");
  const int valueHidden = 64;
  addFP32(
    model,"value.linear2",{static_cast<uint32_t>(valueHidden),96},
    readMatMul(reader,"model.value_head.linear2",96,valueHidden));
  addFP32(
    model,"value.bias2",{static_cast<uint32_t>(valueHidden)},
    readBias(reader,"model.value_head.bias2",valueHidden));
  requireSilu(reader,"model.value_head.act2");
  addFP32(
    model,"value.linear_value",{3,static_cast<uint32_t>(valueHidden)},
    readMatMul(reader,"model.value_head.linear_valuehead",valueHidden,3));
  addFP32(
    model,"value.bias_value",{3},
    readBias(reader,"model.value_head.bias_valuehead",3));
  const vector<float> misc = readMatMul(
    reader,"model.value_head.linear_miscvaluehead",valueHidden,6);
  const vector<float> miscBias = readBias(
    reader,"model.value_head.bias_miscvaluehead",6);
  addFP32(
    model,"value.linear_misc",{4,static_cast<uint32_t>(valueHidden)},
    vector<float>(misc.begin(),misc.begin() + 4 * valueHidden));
  addFP32(
    model,"value.bias_misc",{4},
    vector<float>(miscBias.begin(),miscBias.begin() + 4));
  addFP32(
    model,"value.linear_more",{2,static_cast<uint32_t>(valueHidden)},
    vector<float>(misc.begin() + 4 * valueHidden,misc.end()));
  addFP32(
    model,"value.bias_more",{2},
    vector<float>(miscBias.begin() + 4,miscBias.end()));
  (void)readConv(reader,"model.value_head.conv_ownership",1,1,32,1);
  addFP32(model,"value.bias1",{32},valueSpatial.bias);

  reader.requireEnd();
  model.profile = &selectProfile(
    blocks,channels,heads,ffnChannels,valueHidden);
  if(model.version == BASE_MODEL_VERSION)
    require(uniformQmax == 0,"v205 unexpectedly contains quantized projections");
  else
    require(uniformQmax == 63 || uniformQmax == 127,
            "v206 has no declared projection qmax");
  return model;
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
  return parseNativeModel(payload,actualSha256);
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
  const Tensor& tensor = requireTensor(
    model,name,TensorKind::S8PerOutput,shape);
  if(tensor.quantizedMax != 63 && tensor.quantizedMax != 127)
    fail("tensor qmax mismatch for " + name);
  return tensor;
}

}  // namespace CpuPtq

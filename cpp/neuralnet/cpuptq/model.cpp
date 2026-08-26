#include "kernel.h"

#include "../activations.h"
#include "../desc.h"

#include <cassert>
#include <cmath>
#include <limits>
#include <utility>

using std::string;
using std::vector;

namespace CpuPtq {
namespace {

[[noreturn]] void failBoundary(const string& message) {
  assert(false && "CPU PTQ backend boundary mismatch");
  throw StringError("CPU PTQ backend boundary mismatch: " + message);
}

void requireBoundary(bool condition, const string& message) {
  if(!condition)
    failBoundary(message);
}

const TransformerAttentionDesc& attentionAt(const ModelDesc& model, int index) {
  const size_t wireIndex = (size_t)index * 2;
  requireBoundary(wireIndex < model.trunk.blocks.size(),"missing attention descriptor");
  const auto& entry = model.trunk.blocks[wireIndex];
  requireBoundary(entry.first == TRANSFORMER_ATTENTION_BLOCK_KIND,"expected attention descriptor");
  return *static_cast<const TransformerAttentionDesc*>(entry.second.get());
}

const TransformerFFNDesc& ffnAt(const ModelDesc& model, int index) {
  const size_t wireIndex = (size_t)index * 2 + 1;
  requireBoundary(wireIndex < model.trunk.blocks.size(),"missing FFN descriptor");
  const auto& entry = model.trunk.blocks[wireIndex];
  requireBoundary(entry.first == TRANSFORMER_FFN_BLOCK_KIND,"expected FFN descriptor");
  return *static_cast<const TransformerFFNDesc*>(entry.second.get());
}

bool geometryMatches(const ModelDesc& model, const ProfileSpec& profile) {
  if(model.trunk.trunkNumChannels != profile.trunkChannels ||
     model.trunk.blocks.size() != (size_t)profile.transformerBlocks * 2 ||
     model.valueHead.v2Mul.outChannels != profile.valueHiddenChannels)
    return false;
  for(int i = 0; i < profile.transformerBlocks; i++) {
    const size_t attentionIndex = (size_t)i * 2;
    const size_t ffnIndex = attentionIndex + 1;
    if(model.trunk.blocks[attentionIndex].first != TRANSFORMER_ATTENTION_BLOCK_KIND ||
       model.trunk.blocks[ffnIndex].first != TRANSFORMER_FFN_BLOCK_KIND)
      return false;
    const auto* attention = static_cast<const TransformerAttentionDesc*>(
      model.trunk.blocks[attentionIndex].second.get());
    const auto* ffn = static_cast<const TransformerFFNDesc*>(
      model.trunk.blocks[ffnIndex].second.get());
    if(attention->numHeads != profile.heads ||
       attention->numKVHeads != profile.heads ||
       attention->qHeadDim != profile.headDim ||
       attention->vHeadDim != profile.headDim ||
       ffn->ffnChannels != profile.ffnChannels)
      return false;
  }
  return true;
}

int mapActivation(int activation, const string& name) {
  if(activation == ACTIVATION_RELU)
    return 0;
  if(activation == ACTIVATION_SILU)
    return 1;
  failBoundary(name + " must use ReLU or SiLU");
}

void validateBatchNorm(const BatchNormLayerDesc& norm, int channels, const string& name) {
  requireBoundary(norm.numChannels == channels,name + " channel mismatch");
  requireBoundary(norm.mean.size() == (size_t)channels,name + " mean mismatch");
  requireBoundary(norm.variance.size() == (size_t)channels,name + " variance mismatch");
  requireBoundary(!norm.hasScale || norm.scale.size() == (size_t)channels,name + " scale mismatch");
  requireBoundary(!norm.hasBias || norm.bias.size() == (size_t)channels,name + " bias mismatch");
  requireBoundary(std::isfinite(norm.epsilon) && norm.epsilon > 0.0f,name + " epsilon invalid");
}

void validateProfile(const ModelDesc& model, const ProfileSpec& profile) {
  requireBoundary(model.version == 106,"model version must be v106");
  requireBoundary(model.numInputChannels == SPATIAL_INPUTS,"spatial input count must be 22");
  requireBoundary(model.numInputGlobalChannels == GLOBAL_INPUTS,"global input count must be 39");
  requireBoundary(geometryMatches(model,profile),"model geometry does not match selected profile");
  requireBoundary(model.trunk.numBlocks == profile.transformerBlocks * 2,"wire block count mismatch");
  requireBoundary(model.trunk.initialConv.convXSize == 3 && model.trunk.initialConv.convYSize == 3,
                  "stem must be 3x3");
  requireBoundary(model.trunk.initialConv.dilationX == 1 && model.trunk.initialConv.dilationY == 1,
                  "stem dilation must be one");
  requireBoundary(model.trunk.initialConv.inChannels == SPATIAL_INPUTS &&
                  model.trunk.initialConv.outChannels == profile.trunkChannels,
                  "stem channel mismatch");
  requireBoundary(model.trunk.initialMatMul.inChannels == GLOBAL_INPUTS &&
                  model.trunk.initialMatMul.outChannels == profile.trunkChannels,
                  "global projection mismatch");

  for(int i = 0; i < profile.transformerBlocks; i++) {
    const TransformerAttentionDesc& attention = attentionAt(model,i);
    const TransformerFFNDesc& ffn = ffnAt(model,i);
    requireBoundary(attention.useRope,
                    "current optimized profiles require RoPE");
    if(attention.learnableRope) {
      requireBoundary(
        attention.ropeNumKVHeads == profile.heads &&
        attention.ropeNumPairs == profile.headDim / 2 &&
        attention.ropeFreqs.size() ==
          (size_t)profile.heads * (profile.headDim / 2) * 2,
        "learned RoPE shape mismatch");
    }
    else {
      requireBoundary(profile.headDim % 4 == 0,
                      "fixed 2D RoPE requires headDim divisible by four");
      requireBoundary(std::isfinite(attention.ropeTheta) && attention.ropeTheta > 0.0f,
                      "fixed RoPE theta is invalid");
    }
    requireBoundary(attention.useQKNorm,"current optimized profiles require QK norm");
    requireBoundary(attention.qNorm.numChannels == profile.headDim &&
                    attention.kNorm.numChannels == profile.headDim,
                    "QK norm shape mismatch");
    requireBoundary(attention.preLN.numChannels == profile.trunkChannels &&
                    ffn.preLN.numChannels == profile.trunkChannels,
                    "RMSNorm channel mismatch");
    requireBoundary(attention.preLN.epsilon == 1.0e-6f &&
                    attention.qNorm.epsilon == 1.0e-6f &&
                    attention.kNorm.epsilon == 1.0e-6f &&
                    ffn.preLN.epsilon == 1.0e-6f,
                    "current kernels require RMSNorm epsilon 1e-6");
    requireBoundary(ffn.useSwiGLU && ffn.swigluClip == 4.0f,
                    "current optimized profiles require SwiGLU clip4");
  }

  validateBatchNorm(model.trunk.trunkTipBN,profile.trunkChannels,"trunk tip norm");
  requireBoundary(model.policyHead.p1Conv.inChannels == profile.trunkChannels &&
                  model.policyHead.p1Conv.outChannels == 32 &&
                  model.policyHead.g1Conv.inChannels == profile.trunkChannels &&
                  model.policyHead.g1Conv.outChannels == 32,
                  "policy projection mismatch");
  validateBatchNorm(model.policyHead.g1BN,32,"policy g1 norm");
  validateBatchNorm(model.policyHead.p1BN,32,"policy p1 norm");
  requireBoundary(model.policyHead.gpoolToBiasMul.inChannels == 96 &&
                  model.policyHead.gpoolToBiasMul.outChannels == 32 &&
                  model.policyHead.gpoolToPassMul.inChannels == 96 &&
                  model.policyHead.gpoolToPassMul.outChannels == 1 &&
                  model.policyHead.p2Conv.inChannels == 32 &&
                  model.policyHead.p2Conv.outChannels == 1,
                  "policy head shape mismatch");
  (void)mapActivation(model.trunk.trunkTipActivation.activation,"trunk tip activation");
  (void)mapActivation(model.policyHead.g1Activation.activation,"policy g1 activation");
  (void)mapActivation(model.policyHead.p1Activation.activation,"policy p1 activation");

  requireBoundary(model.valueHead.v1Conv.inChannels == profile.trunkChannels &&
                  model.valueHead.v1Conv.outChannels == 32,
                  "value spatial projection mismatch");
  validateBatchNorm(model.valueHead.v1BN,32,"value v1 norm");
  requireBoundary(model.valueHead.v2Mul.inChannels == 96 &&
                  model.valueHead.v2Mul.outChannels == profile.valueHiddenChannels &&
                  model.valueHead.v2Bias.numChannels == profile.valueHiddenChannels,
                  "value hidden projection mismatch");
  requireBoundary(model.valueHead.v3Mul.inChannels == profile.valueHiddenChannels &&
                  model.valueHead.v3Mul.outChannels == VALUE_SIZE &&
                  model.valueHead.sv3Mul.inChannels == profile.valueHiddenChannels &&
                  model.valueHead.sv3Mul.outChannels == SCORE_VALUE_SIZE &&
                  model.valueHead.vOwnershipConv.inChannels == 32 &&
                  model.valueHead.vOwnershipConv.outChannels == 1,
                  "value outputs mismatch");
  (void)mapActivation(model.valueHead.v1Activation.activation,"value v1 activation");
  (void)mapActivation(model.valueHead.v2Activation.activation,"value v2 activation");
}

vector<float> outputMajor(const MatMulLayerDesc& layer) {
  requireBoundary(!layer.isQuantized,layer.name + " unexpectedly contains quantized weights");
  vector<float> result((size_t)layer.inChannels * layer.outChannels);
  for(int output = 0; output < layer.outChannels; output++) {
    for(int input = 0; input < layer.inChannels; input++) {
      result[(size_t)output * layer.inChannels + input] =
        layer.weights[(size_t)input * layer.outChannels + output];
    }
  }
  return result;
}

void addTensor(
  TensorMap& tensors,
  const string& name,
  std::initializer_list<uint64_t> shape,
  vector<float> values
) {
  size_t expected = 1;
  for(uint64_t dimension : shape)
    expected *= (size_t)dimension;
  requireBoundary(values.size() == expected,name + " tensor element count mismatch");
  const bool inserted = tensors.emplace(
    name,Tensor{vector<uint64_t>(shape),std::move(values),{},{} }).second;
  requireBoundary(inserted,name + " tensor was inserted twice");
}

void addQuantizedMatMul(
  TensorMap& tensors,
  const string& name,
  const MatMulLayerDesc& layer
) {
  requireBoundary(layer.isQuantized,layer.name + " must use v106 S7/S8 storage");
  requireBoundary(
    layer.quantizedWeights.size() == (size_t)layer.inChannels * layer.outChannels,
    layer.name + " quantized element count mismatch");
  requireBoundary(
    layer.weightScales.size() == (size_t)layer.outChannels,
    layer.name + " quantized scale count mismatch");
  requireBoundary(
    layer.quantizedMax == 63 || layer.quantizedMax == 127,
    layer.name + " quantized range must be S7 or S8");
  requireBoundary(layer.weights.empty(),layer.name + " retained an FP32 projection copy");
  Tensor tensor;
  tensor.shape = {(uint64_t)layer.outChannels,(uint64_t)layer.inChannels};
  tensor.quantizedValues = layer.quantizedWeights;
  tensor.quantizedScales = layer.weightScales;
  tensor.quantizedMax = layer.quantizedMax;
  const bool inserted = tensors.emplace(name,std::move(tensor)).second;
  requireBoundary(inserted,name + " tensor was inserted twice");
}

void addMatMul(TensorMap& tensors, const string& name, const MatMulLayerDesc& layer) {
  addTensor(
    tensors,name,{(uint64_t)layer.outChannels,(uint64_t)layer.inChannels},
    outputMajor(layer));
}

void addConv(TensorMap& tensors, const string& name, const ConvLayerDesc& layer) {
  addTensor(
    tensors,name,
    {(uint64_t)layer.outChannels,(uint64_t)layer.inChannels,
     (uint64_t)layer.convYSize,(uint64_t)layer.convXSize},
    layer.weights);
}

vector<float> kernelRopeFrequencies(
  const TransformerAttentionDesc& attention,
  const ProfileSpec& profile
) {
  if(attention.learnableRope)
    return attention.ropeFreqs;

  const int numPairs = profile.headDim / 2;
  const int pairsPerDimension = numPairs / 2;
  vector<float> frequencies((size_t)profile.heads * numPairs * 2,0.0f);
  for(int head = 0; head < profile.heads; head++) {
    for(int pair = 0; pair < numPairs; pair++) {
      const int coordinatePair = pair < pairsPerDimension ? pair : pair - pairsPerDimension;
      const float frequency = 1.0f / std::pow(
        attention.ropeTheta,
        (float)(2 * coordinatePair) / (float)(profile.headDim / 2));
      const size_t index = ((size_t)head * numPairs + pair) * 2;
      if(pair < pairsPerDimension)
        frequencies[index + 1] = frequency;
      else
        frequencies[index] = frequency;
    }
  }
  return frequencies;
}

void addNormActivation(
  TensorMap& tensors,
  const string& prefix,
  const BatchNormLayerDesc& norm,
  const ActivationLayerDesc& activation
) {
  vector<float> multiplier((size_t)norm.numChannels);
  vector<float> bias((size_t)norm.numChannels);
  for(int channel = 0; channel < norm.numChannels; channel++) {
    const float scale = norm.hasScale ? norm.scale[(size_t)channel] : 1.0f;
    const float beta = norm.hasBias ? norm.bias[(size_t)channel] : 0.0f;
    const float denominator = std::sqrt(norm.variance[(size_t)channel] + norm.epsilon);
    requireBoundary(std::isfinite(denominator) && denominator > 0.0f,
                    prefix + " normalization denominator invalid");
    multiplier[(size_t)channel] = scale / denominator;
    bias[(size_t)channel] = beta - norm.mean[(size_t)channel] * multiplier[(size_t)channel];
  }
  addTensor(tensors,prefix + ".mul",{(uint64_t)norm.numChannels},std::move(multiplier));
  addTensor(tensors,prefix + ".bias",{(uint64_t)norm.numChannels},std::move(bias));
  addTensor(
    tensors,prefix + ".activation",{1},
    vector<float>{(float)mapActivation(activation.activation,prefix)});
}

}  // namespace

const ProfileSpec& b16Profile() {
  static const ProfileSpec profile = {
    ProfileKind::B16C128H4F384,"b16c128h4-f384",16,128,4,32,384,96
  };
  return profile;
}

const ProfileSpec& b24Profile() {
  static const ProfileSpec profile = {
    ProfileKind::B24C192H6F512,"b24c192h6-f512",24,192,6,32,512,96
  };
  return profile;
}

const ProfileSpec& b11Profile() {
  static const ProfileSpec profile = {
    ProfileKind::B11C96H3F256,"b11c96h3-f256",11,96,3,32,256,64
  };
  return profile;
}

const ProfileSpec& selectProfile(const ModelDesc& model) {
  requireBoundary(model.version == 106,"only v106 CPU-PTQ models are accepted");
  const ProfileSpec* profiles[] = {&b24Profile(),&b16Profile(),&b11Profile()};
  for(const ProfileSpec* profile: profiles) {
    if(geometryMatches(model,*profile)) {
      validateProfile(model,*profile);
      return *profile;
    }
  }
  failBoundary("model does not match a compiled b24c192, b16c128, or b11c96 profile");
}

const vector<float>& requireTensor(
  const TensorMap& tensors,
  const string& name,
  std::initializer_list<uint64_t> shape
) {
  const auto found = tensors.find(name);
  if(found == tensors.end())
    failBoundary("missing tensor " + name);
  if(found->second.shape != vector<uint64_t>(shape))
    failBoundary("shape mismatch for tensor " + name);
  if(!found->second.quantizedValues.empty() || !found->second.quantizedScales.empty())
    failBoundary("expected FP32 tensor but found quantized tensor " + name);
  return found->second.values;
}

const Tensor& requireQuantizedTensor(
  const TensorMap& tensors,
  const string& name,
  std::initializer_list<uint64_t> shape
) {
  const auto found = tensors.find(name);
  if(found == tensors.end())
    failBoundary("missing tensor " + name);
  if(found->second.shape != vector<uint64_t>(shape))
    failBoundary("shape mismatch for tensor " + name);
  if(!found->second.values.empty())
    failBoundary("expected quantized tensor but found FP32 tensor " + name);
  if(found->second.shape.size() != 2)
    failBoundary("quantized tensor must be a matrix " + name);
  const size_t outputs = (size_t)found->second.shape[0];
  const size_t inputs = (size_t)found->second.shape[1];
  if(found->second.quantizedValues.size() != outputs * inputs ||
     found->second.quantizedScales.size() != outputs)
    failBoundary("malformed quantized tensor payload " + name);
  return found->second;
}

TensorMap makeKernelTensors(const ModelDesc& model, const ProfileSpec& profile) {
  validateProfile(model,profile);
  TensorMap tensors;
  addConv(tensors,"conv_spatial.weight",model.trunk.initialConv);
  addMatMul(tensors,"linear_global.weight",model.trunk.initialMatMul);

  for(int block = 0; block < profile.transformerBlocks; block++) {
    const TransformerAttentionDesc& attention = attentionAt(model,block);
    const TransformerFFNDesc& ffn = ffnAt(model,block);
    const string prefix = "blocks." + std::to_string(block) + ".";
    addTensor(tensors,prefix + "norm1.weight",{(uint64_t)profile.trunkChannels},attention.preLN.weight);
    addTensor(tensors,prefix + "norm2.weight",{(uint64_t)profile.trunkChannels},ffn.preLN.weight);
    addTensor(tensors,prefix + "q_norm.weight",{(uint64_t)profile.headDim},attention.qNorm.weight);
    addTensor(tensors,prefix + "k_norm.weight",{(uint64_t)profile.headDim},attention.kNorm.weight);
    addQuantizedMatMul(tensors,prefix + "q_proj.weight",attention.qProj);
    addQuantizedMatMul(tensors,prefix + "k_proj.weight",attention.kProj);
    addQuantizedMatMul(tensors,prefix + "v_proj.weight",attention.vProj);
    addQuantizedMatMul(tensors,prefix + "out_proj.weight",attention.outProj);
    addQuantizedMatMul(tensors,prefix + "ffn_linear1.weight",ffn.linear1);
    addQuantizedMatMul(tensors,prefix + "ffn_linear_gate.weight",ffn.linearGate);
    addQuantizedMatMul(tensors,prefix + "ffn_linear2.weight",ffn.linear2);
    addTensor(
      tensors,prefix + "rope_freqs",
      {(uint64_t)profile.heads,(uint64_t)(profile.headDim / 2),2},
      kernelRopeFrequencies(attention,profile));
  }

  addNormActivation(
    tensors,"norm_trunkfinal",model.trunk.trunkTipBN,model.trunk.trunkTipActivation);
  addConv(tensors,"policy_head.conv1p.weight",model.policyHead.p1Conv);
  addConv(tensors,"policy_head.conv1g.weight",model.policyHead.g1Conv);
  addNormActivation(
    tensors,"policy_head.g1",model.policyHead.g1BN,model.policyHead.g1Activation);
  addNormActivation(
    tensors,"policy_head.p1",model.policyHead.p1BN,model.policyHead.p1Activation);
  addMatMul(tensors,"policy_head.linear_g.weight",model.policyHead.gpoolToBiasMul);
  addMatMul(tensors,"policy_head.linear_pass.weight",model.policyHead.gpoolToPassMul);
  addConv(tensors,"policy_head.conv2p.weight",model.policyHead.p2Conv);

  addConv(tensors,"value_head.conv1.weight",model.valueHead.v1Conv);
  addNormActivation(
    tensors,"value_head.v1",model.valueHead.v1BN,model.valueHead.v1Activation);
  addMatMul(tensors,"value_head.linear2.weight",model.valueHead.v2Mul);
  addTensor(
    tensors,"value_head.linear2.bias",{(uint64_t)profile.valueHiddenChannels},
    model.valueHead.v2Bias.weights);
  addTensor(
    tensors,"value_head.v2.mul",{(uint64_t)profile.valueHiddenChannels},
    vector<float>((size_t)profile.valueHiddenChannels,1.0f));
  addTensor(
    tensors,"value_head.v2.activation_bias",{(uint64_t)profile.valueHiddenChannels},
    vector<float>((size_t)profile.valueHiddenChannels,0.0f));
  addTensor(
    tensors,"value_head.v2.activation",{1},
    vector<float>{(float)mapActivation(model.valueHead.v2Activation.activation,"value v2")});
  addMatMul(tensors,"value_head.linear_valuehead.weight",model.valueHead.v3Mul);
  addTensor(tensors,"value_head.linear_valuehead.bias",{VALUE_SIZE},model.valueHead.v3Bias.weights);
  addMatMul(tensors,"value_head.linear_scorevalue.weight",model.valueHead.sv3Mul);
  addTensor(
    tensors,"value_head.linear_scorevalue.bias",{SCORE_VALUE_SIZE},
    model.valueHead.sv3Bias.weights);
  addConv(tensors,"value_head.conv_ownership.weight",model.valueHead.vOwnershipConv);
  return tensors;
}

}  // namespace CpuPtq

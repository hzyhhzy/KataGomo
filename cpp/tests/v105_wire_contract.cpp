#include "../main.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../neuralnet/desc.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/v105policy.h"

using namespace std;

namespace {

enum class RopeKind {
  None,
  Fixed,
  Learned,
};

[[noreturn]] void failContract(const string& message) {
  throw StringError("v105 wire contract: " + message);
}

void requireContract(bool condition, const string& message) {
  if(!condition)
    failContract(message);
}

template<typename Func>
void expectStringError(Func&& func, const string& context) {
  try {
    func();
  }
  catch(const StringError&) {
    return;
  }
  failContract(context + " was accepted");
}

template<typename Func>
void expectStringErrorContains(Func&& func, const string& expected, const string& context) {
  try {
    func();
  }
  catch(const StringError& error) {
    requireContract(string(error.what()).find(expected) != string::npos,
      context + " threw the wrong error: " + error.what());
    return;
  }
  failContract(context + " was accepted");
}

uint32_t floatBits(float value) {
  uint32_t bits;
  static_assert(sizeof(bits) == sizeof(value), "float must be 32 bits");
  memcpy(&bits,&value,sizeof(bits));
  return bits;
}

void appendFloats(ostringstream& out, size_t count, const string& value = "0.125") {
  for(size_t i = 0; i < count; i++)
    out << value << '\n';
}

void appendRMSNorm(
  ostringstream& out,
  const string& name,
  int channels,
  const string& epsilon = "0.000001"
) {
  out << name << '\n' << channels << '\n' << epsilon << '\n';
  appendFloats(out,(size_t)channels);
}

void appendMatMul(
  ostringstream& out,
  const string& name,
  int inChannels,
  int outChannels
) {
  out << name << '\n' << inChannels << '\n' << outChannels << '\n';
  appendFloats(out,(size_t)inChannels * (size_t)outChannels);
}

void appendConv1x1(
  ostringstream& out,
  const string& name,
  int inChannels,
  int outChannels
) {
  out << name << "\n1\n1\n" << inChannels << '\n' << outChannels << "\n1\n1\n";
  appendFloats(out,(size_t)inChannels * (size_t)outChannels);
}

void appendBatchNorm(ostringstream& out, const string& name, int channels) {
  out << name << '\n' << channels << "\n0.001\n0\n0\n";
  appendFloats(out,(size_t)channels,"0.0");
  appendFloats(out,(size_t)channels,"1.0");
}

void appendActivation(ostringstream& out, const string& name) {
  out << name << "\nACTIVATION_SILU\n";
}

struct AttentionWireOptions {
  int modelVersion = 105;
  bool useQKNorm = true;
  int qkNormFlag = -1;
  RopeKind rope = RopeKind::Learned;
  string inputRange = "3.25";
  string outputRange = "5.5";
  bool includeInputRange = true;
  bool includeOutputRange = true;
  int qNormChannels = 4;
  int kNormChannels = 4;
};

string attentionWire(const AttentionWireOptions& options) {
  ostringstream out;
  const bool useRope = options.rope != RopeKind::None;
  const bool learnableRope = options.rope == RopeKind::Learned;
  out << "attention\n2\n2\n4\n4\n"
      << (useRope ? 1 : 0) << '\n'
      << (learnableRope ? 1 : 0) << '\n';
  if(options.modelVersion >= 105) {
    const int qkNormFlag = options.qkNormFlag >= 0 ?
      options.qkNormFlag : (options.useQKNorm ? 1 : 0);
    out << qkNormFlag << '\n';
    if(options.includeInputRange)
      out << options.inputRange << '\n';
    if(options.includeOutputRange)
      out << options.outputRange << '\n';
  }
  appendRMSNorm(out,"attention.norm1",8);
  appendMatMul(out,"attention.q",8,8);
  appendMatMul(out,"attention.k",8,8);
  appendMatMul(out,"attention.v",8,8);
  appendMatMul(out,"attention.out",8,8);
  if(options.modelVersion >= 105 && options.useQKNorm && options.qkNormFlag != 0) {
    appendRMSNorm(out,"attention.qnorm",options.qNormChannels,"0.000001");
    appendRMSNorm(out,"attention.knorm",options.kNormChannels,"0.000002");
  }
  if(options.rope == RopeKind::Learned) {
    out << "attention.rope\n2\n2\n2\n";
    appendFloats(out,8,"0.0625");
  }
  else if(options.rope == RopeKind::Fixed) {
    out << "attention.theta\n10000\n";
  }
  return out.str();
}

struct FfnWireOptions {
  int modelVersion = 105;
  bool useSwiGLU = true;
  string clip = "7.0";
  string inputRange = "2.75";
  string productRange = "23.5";
  bool includeClip = true;
  bool includeInputRange = true;
  bool includeProductRange = true;
};

string ffnWire(const FfnWireOptions& options) {
  ostringstream out;
  out << "ffn\n8\n12\n" << (options.useSwiGLU ? 1 : 0) << '\n';
  if(options.modelVersion >= 105) {
    if(options.includeClip)
      out << options.clip << '\n';
    if(options.includeInputRange)
      out << options.inputRange << '\n';
    if(options.includeProductRange)
      out << options.productRange << '\n';
  }
  appendRMSNorm(out,"ffn.norm",8);
  appendMatMul(out,"ffn.up",8,12);
  if(options.useSwiGLU)
    appendMatMul(out,"ffn.gate",8,12);
  appendMatMul(out,"ffn.down",12,8);
  return out.str();
}

void requireFullyConsumed(istream& in, const string& context) {
  in >> ws;
  requireContract(in.peek() == EOF,context + " left unread wire fields");
}

vector<string> tokenize(const string& wire) {
  istringstream in(wire);
  vector<string> tokens;
  string token;
  while(in >> token)
    tokens.push_back(token);
  return tokens;
}

string prefixWire(const vector<string>& tokens, size_t count) {
  ostringstream out;
  for(size_t i = 0; i < count; i++)
    out << tokens[i] << '\n';
  return out.str();
}

void testVersions() {
  static_assert(NNModelVersion::latestModelVersionImplemented == 105,
    "native v105 must be the latest implemented wire version");
  static_assert(NNModelVersion::defaultModelVersion == 102,
    "v105 must not silently replace the ordinary v102 default");
  requireContract(NNModelVersion::getInputsVersion(102) == 101,"v102 input version changed");
  requireContract(NNModelVersion::getInputsVersion(105) == 101,"v105 must use V101 inputs");
  requireContract(NNModelVersion::getNumSpatialFeatures(102) == 22,"v102 spatial ABI changed");
  requireContract(NNModelVersion::getNumGlobalFeatures(102) == 39,"v102 global ABI changed");
  requireContract(NNModelVersion::getInputsVersion(103) == 102,"v103 input mapping changed");
  requireContract(NNModelVersion::getNumSpatialFeatures(103) == 32,"v103 spatial ABI changed");
  requireContract(NNModelVersion::getNumGlobalFeatures(103) == 64,"v103 global ABI changed");
  requireContract(NNModelVersion::getNumSpatialFeatures(105) == 22,"v105 spatial ABI is not V101");
  requireContract(NNModelVersion::getNumGlobalFeatures(105) == 39,"v105 global ABI is not V101");

  expectStringError([](){ (void)NNModelVersion::getInputsVersion(104); },"v104 input mapping");
  expectStringError([](){ (void)NNModelVersion::getNumSpatialFeatures(104); },"v104 spatial mapping");
  expectStringError([](){ (void)NNModelVersion::getNumGlobalFeatures(104); },"v104 global mapping");
  expectStringError([](){
    istringstream in("unsupported-v104\n104\n");
    (void)ModelDesc(in,"",false);
  },"native v104 model loader");
  expectStringErrorContains([](){
    istringstream in("v105-spatial-mismatch\n105\n23\n39\n");
    (void)ModelDesc(in,"",false);
  },"does not match model version 105 spatial features",
    "native v105 spatial input-channel mismatch");
  expectStringErrorContains([](){
    istringstream in("v105-global-mismatch\n105\n22\n40\n");
    (void)ModelDesc(in,"",false);
  },"does not match model version 105 global features",
    "native v105 global input-channel mismatch");
}

void testProjectedScratchLayout() {
  const V105CudaPolicy::ProjectedScratchLayout oddHalf =
    V105CudaPolicy::makeProjectedScratchLayout(3,1,1,1,2);
  requireContract(oddHalf.planeElements == 3,
    "odd FP16 projected plane changed its logical element count");
  requireContract(oddHalf.planeStrideElements == 4 && oddHalf.planeStrideBytes == 8,
    "odd FP16 projected plane was not padded to a 4-byte gate offset");
  requireContract(oddHalf.totalBytes == 16 && oddHalf.planeStrideBytes % 4 == 0,
    "odd FP16 projected allocation is not two aligned planes");

  const V105CudaPolicy::ProjectedScratchLayout oddProduct =
    V105CudaPolicy::makeProjectedScratchLayout(3,5,1,7,2);
  requireContract(oddProduct.planeElements == 105 && oddProduct.planeStrideElements == 106,
    "maxB*XY*F odd-product alignment contract changed");
  requireContract(
    oddProduct.planeStrideBytes / 2 == oddProduct.planeStrideElements,
    "FP16 GEMM stride lost its element-count semantics"
  );

  const V105CudaPolicy::ProjectedScratchLayout oddFloat =
    V105CudaPolicy::makeProjectedScratchLayout(3,1,1,1,4);
  requireContract(oddFloat.planeStrideElements == 3 && oddFloat.planeStrideBytes == 12,
    "FP32 projected plane was padded despite natural 4-byte alignment");

  static_assert(sizeof(size_t) >= 8,"v105 CUDA contracts require a 64-bit host size_t");
  const size_t intMax = (size_t)numeric_limits<int>::max();
  const V105CudaPolicy::ProjectedScratchLayout boundary =
    V105CudaPolicy::makeProjectedScratchLayout(intMax,1,1,1,2);
  requireContract(boundary.planeElements == intMax,
    "INT_MAX projected plane was not accepted exactly");
  requireContract(boundary.planeStrideElements == intMax + 1,
    "INT_MAX odd FP16 plane did not receive one alignment element");
  requireContract(boundary.planeStrideBytes == (intMax + 1) * 2 &&
    boundary.totalBytes == (intMax + 1) * 4,
    "INT_MAX projected allocation overflowed size_t");

  expectStringError([&](){
    (void)V105CudaPolicy::makeProjectedScratchLayout(intMax + 1,1,1,1,2);
  },"projected plane above INT_MAX");
  expectStringError([&](){
    (void)V105CudaPolicy::makeProjectedScratchLayout(
      numeric_limits<size_t>::max(),2,1,1,2
    );
  },"projected layout size_t multiplication overflow");
  expectStringError([&](){
    (void)V105CudaPolicy::makeProjectedScratchLayout(1,1,1,1,1);
  },"projected layout with unsupported element width");
}

void testLegacyV102() {
  AttentionWireOptions attentionOptions;
  attentionOptions.modelVersion = 102;
  attentionOptions.useQKNorm = false;
  attentionOptions.rope = RopeKind::Learned;
  const string attentionBytes = attentionWire(attentionOptions);
  istringstream attentionIn(attentionBytes);
  TransformerAttentionDesc attention(attentionIn,102,false);
  requireContract(!attention.useQKNorm,"v102 unexpectedly enabled QK norm");
  requireContract(attention.attentionInputQuantMaxAbs == 0.0f,"v102 attention input range is nonzero");
  requireContract(attention.attentionOutputQuantMaxAbs == 0.0f,"v102 attention output range is nonzero");
  requireContract(attention.qNorm.numChannels == 0 && attention.kNorm.numChannels == 0,
    "v102 consumed Q/K RMSNorm descriptors");
  requireContract(attention.learnableRope && attention.ropeFreqs.size() == 8,
    "v102 learned RoPE shifted on the wire");
  requireFullyConsumed(attentionIn,"v102 attention");

  FfnWireOptions ffnOptions;
  ffnOptions.modelVersion = 102;
  const string ffnBytes = ffnWire(ffnOptions);
  istringstream ffnIn(ffnBytes);
  TransformerFFNDesc ffn(ffnIn,102,false);
  requireContract(ffn.swigluClip == 0.0f,"v102 FFN clip is nonzero");
  requireContract(ffn.ffnInputQuantMaxAbs == 0.0f,"v102 FFN input range is nonzero");
  requireContract(ffn.productQuantMaxAbs == 0.0f,"v102 product range is nonzero");
  requireContract(ffn.linearGate.inChannels == 8,"v102 gate shifted on the wire");
  requireFullyConsumed(ffnIn,"v102 FFN");
}

void testV105AttentionVariantsAndMoves() {
  AttentionWireOptions learnedOptions;
  learnedOptions.useQKNorm = true;
  learnedOptions.rope = RopeKind::Learned;
  istringstream learnedIn(attentionWire(learnedOptions));
  TransformerAttentionDesc learned(learnedIn,105,false);
  requireContract(learned.useQKNorm,"v105 QK norm flag was lost");
  requireContract(floatBits(learned.attentionInputQuantMaxAbs) == floatBits(3.25f),
    "v105 attention input range was not bit-exact");
  requireContract(floatBits(learned.attentionOutputQuantMaxAbs) == floatBits(5.5f),
    "v105 attention output range was not bit-exact");
  requireContract(learned.qNorm.numChannels == 4 && learned.kNorm.numChannels == 4,
    "v105 Q/K norm dimensions were not preserved");
  requireContract(floatBits(learned.qNorm.epsilon) == floatBits(1e-6f),
    "v105 qNorm epsilon was not preserved");
  requireContract(floatBits(learned.kNorm.epsilon) == floatBits(2e-6f),
    "v105 kNorm epsilon was not preserved");
  requireContract(learned.learnableRope && learned.ropeFreqs.size() == 8,
    "v105 learned RoPE was not parsed after Q/K norms");
  requireFullyConsumed(learnedIn,"v105 learned-RoPE attention");

  TransformerAttentionDesc moved(std::move(learned));
  TransformerAttentionDesc assigned;
  assigned = std::move(moved);
  requireContract(assigned.useQKNorm && assigned.qNorm.numChannels == 4 && assigned.kNorm.numChannels == 4,
    "attention move lost Q/K norm state");
  requireContract(floatBits(assigned.attentionInputQuantMaxAbs) == floatBits(3.25f) &&
                  floatBits(assigned.attentionOutputQuantMaxAbs) == floatBits(5.5f),
    "attention move lost PTQ ranges");
  requireContract(assigned.ropeFreqs.size() == 8,"attention move lost learned RoPE");

  AttentionWireOptions fixedOptions;
  fixedOptions.useQKNorm = false;
  fixedOptions.rope = RopeKind::Fixed;
  fixedOptions.inputRange = "2.75";
  fixedOptions.outputRange = "3.25";
  istringstream fixedIn(attentionWire(fixedOptions));
  TransformerAttentionDesc fixed(fixedIn,105,false);
  requireContract(!fixed.useQKNorm && fixed.qNorm.numChannels == 0 && fixed.kNorm.numChannels == 0,
    "QKN-disabled v105 consumed norm fields");
  requireContract(fixed.useRope && !fixed.learnableRope && fixed.ropeTheta == 10000.0f,
    "fixed RoPE was not parsed in canonical position");
  requireFullyConsumed(fixedIn,"v105 fixed-RoPE attention");

  AttentionWireOptions noRopeOptions;
  noRopeOptions.useQKNorm = false;
  noRopeOptions.rope = RopeKind::None;
  istringstream noRopeIn(attentionWire(noRopeOptions));
  TransformerAttentionDesc noRope(noRopeIn,105,false);
  requireContract(!noRope.useRope && !noRope.learnableRope && noRope.ropeFreqs.empty(),
    "no-RoPE v105 consumed a RoPE descriptor");
  requireFullyConsumed(noRopeIn,"v105 no-RoPE attention");
}

void testV105FfnMixedLayersAndMoves() {
  const char* clips[] = {"0.0","4.0","7.0"};
  const char* inputRanges[] = {"2.75","3.25","5.5"};
  const char* productRanges[] = {"23.5","47.75","73.5"};
  string mixedWire;
  for(size_t i = 0; i < 3; i++) {
    FfnWireOptions options;
    options.clip = clips[i];
    options.inputRange = inputRanges[i];
    options.productRange = productRanges[i];
    mixedWire += ffnWire(options);
  }
  istringstream mixedIn(mixedWire);
  for(size_t i = 0; i < 3; i++) {
    TransformerFFNDesc ffn(mixedIn,105,false);
    requireContract(floatBits(ffn.swigluClip) == floatBits(strtof(clips[i],nullptr)),
      "mixed FFN clip shifted between layers");
    requireContract(floatBits(ffn.ffnInputQuantMaxAbs) == floatBits(strtof(inputRanges[i],nullptr)),
      "mixed FFN input range shifted between layers");
    requireContract(floatBits(ffn.productQuantMaxAbs) == floatBits(strtof(productRanges[i],nullptr)),
      "mixed FFN product range shifted between layers");
    if(i == 2) {
      TransformerFFNDesc moved(std::move(ffn));
      TransformerFFNDesc assigned;
      assigned = std::move(moved);
      requireContract(floatBits(assigned.swigluClip) == floatBits(7.0f) &&
                      floatBits(assigned.ffnInputQuantMaxAbs) == floatBits(5.5f) &&
                      floatBits(assigned.productQuantMaxAbs) == floatBits(73.5f),
        "FFN move lost v105 scalars");
      requireContract(assigned.linearGate.inChannels == 8,"FFN move lost gate weights");
    }
  }
  requireFullyConsumed(mixedIn,"mixed v105 FFNs");
}

void testVersionPropagationThroughTrunk() {
  ostringstream wire;
  wire << "trunk\n6\n8\n8\n4\n0\n4\n";
  appendConv1x1(wire,"trunk.initial",8,8);
  appendMatMul(wire,"trunk.global",8,8);
  for(int i = 0; i < 3; i++) {
    AttentionWireOptions attentionOptions;
    attentionOptions.useQKNorm = i == 0;
    attentionOptions.rope = i == 0 ? RopeKind::Learned :
      (i == 1 ? RopeKind::Fixed : RopeKind::None);
    attentionOptions.inputRange = i == 0 ? "3.25" : (i == 1 ? "5.5" : "2.75");
    attentionOptions.outputRange = i == 0 ? "2.75" : (i == 1 ? "3.25" : "5.5");
    FfnWireOptions ffnOptions;
    ffnOptions.clip = i == 0 ? "0.0" : (i == 1 ? "4.0" : "7.0");
    ffnOptions.inputRange = i == 0 ? "2.75" : (i == 1 ? "3.25" : "5.5");
    ffnOptions.productRange = i == 0 ? "23.5" : (i == 1 ? "47.75" : "73.5");
    wire << "transformer_attention_block\n" << attentionWire(attentionOptions);
    wire << "transformer_ffn_block\n" << ffnWire(ffnOptions);
  }
  appendBatchNorm(wire,"trunk.tip",8);
  wire << "trunk.tip.activation\nACTIVATION_SILU\n";

  istringstream in(wire.str());
  TrunkDesc trunk(in,105,false);
  requireContract(trunk.blocks.size() == 6,"trunk did not preserve dynamic block count");
  for(int i = 0; i < 3; i++) {
    requireContract(trunk.blocks[(size_t)i * 2].first == TRANSFORMER_ATTENTION_BLOCK_KIND,
      "trunk attention/FFN order changed");
    requireContract(trunk.blocks[(size_t)i * 2 + 1].first == TRANSFORMER_FFN_BLOCK_KIND,
      "trunk attention/FFN order changed");
    const TransformerAttentionDesc* attention = (const TransformerAttentionDesc*)
      trunk.blocks[(size_t)i * 2].second.get();
    const TransformerFFNDesc* ffn = (const TransformerFFNDesc*)
      trunk.blocks[(size_t)i * 2 + 1].second.get();
    requireContract(attention->attentionInputQuantMaxAbs > 0.0f &&
                    attention->attentionOutputQuantMaxAbs > 0.0f,
      "trunk parser did not forward v105 to attention");
    requireContract(ffn->ffnInputQuantMaxAbs > 0.0f && ffn->productQuantMaxAbs > 0.0f,
      "trunk parser did not forward v105 to FFN");
  }
  const V105CudaPolicy::Decision extended = V105CudaPolicy::classify(105,trunk);
  requireContract(extended.isV105 && extended.hasQKNorm && extended.hasPositiveSwiGLUClip,
    "CUDA v105 policy failed to detect QKN/clip semantics");
  requireContract(V105CudaPolicy::shouldUseCombinedQKV(false,true),
    "non-QKN MMA-eligible attention unexpectedly lost combined QKV");
  requireContract(V105CudaPolicy::shouldUseCombinedQKV(true,true) &&
                  !V105CudaPolicy::shouldUseCombinedQKV(true,false),
    "QKN attention did not preserve an eligible fused combined-QKV plan");
  requireContract(
    V105CudaPolicy::selectSwiGLUPlan(0.0f) ==
      V105CudaPolicy::SwiGLUPlan::LegacyUnclipped,
    "clip0 did not preserve the legacy SwiGLU helper");
  requireContract(
    V105CudaPolicy::selectSwiGLUPlan(4.0f) ==
      V105CudaPolicy::SwiGLUPlan::OrderedClippedFP32,
    "positive clip did not select ordered FP32 clamp semantics");
  V105CudaPolicy::requireCurrentExecution(105,trunk,true);
  expectStringError([&](){
    V105CudaPolicy::requireCurrentExecution(105,trunk,false);
  },"v105 QKN/clip non-FP16 CUDA execution");

  for(auto& entry: trunk.blocks) {
    if(entry.first == TRANSFORMER_ATTENTION_BLOCK_KIND) {
      TransformerAttentionDesc* attention = (TransformerAttentionDesc*)entry.second.get();
      attention->useQKNorm = false;
    }
    else if(entry.first == TRANSFORMER_FFN_BLOCK_KIND) {
      TransformerFFNDesc* ffn = (TransformerFFNDesc*)entry.second.get();
      ffn->swigluClip = 0.0f;
    }
  }
  const V105CudaPolicy::Decision legacySafe = V105CudaPolicy::classify(105,trunk);
  requireContract(legacySafe.isV105 && !legacySafe.needsQKNClipSemantics(),
    "CUDA v105 policy rejected no-QKN/clip0 fallback");
  V105CudaPolicy::requireCurrentExecution(105,trunk,true);
  expectStringError([&](){
    V105CudaPolicy::requireCurrentExecution(105,trunk,false);
  },"v105 no-QKN/clip0 non-FP16 CUDA fallback");
  V105CudaPolicy::requireCurrentExecution(102,trunk,false);
  requireFullyConsumed(in,"v105 trunk");
}

void testNestedBottleneckVersionPropagationAndPolicy() {
  AttentionWireOptions attentionOptions;
  attentionOptions.useQKNorm = true;
  attentionOptions.rope = RopeKind::Fixed;
  attentionOptions.inputRange = "3.25";
  attentionOptions.outputRange = "5.5";
  FfnWireOptions ffnOptions;
  ffnOptions.clip = "7.0";
  ffnOptions.inputRange = "2.75";
  ffnOptions.productRange = "23.5";

  ostringstream wire;
  wire << "nested-trunk\n1\n8\n8\n4\n0\n4\n";
  appendConv1x1(wire,"nested-trunk.initial",8,8);
  appendMatMul(wire,"nested-trunk.global",8,8);
  wire << "nested_bottleneck_block\nnested\n2\n";
  appendBatchNorm(wire,"nested.pre",8);
  appendActivation(wire,"nested.pre.activation");
  appendConv1x1(wire,"nested.pre.conv",8,8);
  wire << "transformer_attention_block\n" << attentionWire(attentionOptions);
  wire << "transformer_ffn_block\n" << ffnWire(ffnOptions);
  appendBatchNorm(wire,"nested.post",8);
  appendActivation(wire,"nested.post.activation");
  appendConv1x1(wire,"nested.post.conv",8,8);
  appendBatchNorm(wire,"nested-trunk.tip",8);
  appendActivation(wire,"nested-trunk.tip.activation");

  istringstream in(wire.str());
  TrunkDesc trunk(in,105,false);
  requireFullyConsumed(in,"nested v105 trunk");
  requireContract(trunk.hasAnyTransformerBlocks(),
    "recursive trunk traversal did not find nested transformer blocks");
  requireContract(trunk.blocks.size() == 1 &&
                  trunk.blocks[0].first == NESTED_BOTTLENECK_BLOCK_KIND,
    "nested bottleneck block was not preserved");
  const NestedBottleneckResidualBlockDesc* nested =
    (const NestedBottleneckResidualBlockDesc*)trunk.blocks[0].second.get();
  requireContract(nested->blocks.size() == 2 &&
                  nested->blocks[0].first == TRANSFORMER_ATTENTION_BLOCK_KIND &&
                  nested->blocks[1].first == TRANSFORMER_FFN_BLOCK_KIND,
    "nested transformer block order was not preserved");
  const TransformerAttentionDesc* attention =
    (const TransformerAttentionDesc*)nested->blocks[0].second.get();
  const TransformerFFNDesc* ffn =
    (const TransformerFFNDesc*)nested->blocks[1].second.get();
  requireContract(attention->useQKNorm &&
                  attention->qNorm.numChannels == attention->qHeadDim &&
                  attention->kNorm.numChannels == attention->qHeadDim &&
                  floatBits(attention->attentionInputQuantMaxAbs) == floatBits(3.25f) &&
                  floatBits(attention->attentionOutputQuantMaxAbs) == floatBits(5.5f),
    "nested v105 attention fields were not parsed recursively");
  requireContract(floatBits(ffn->swigluClip) == floatBits(7.0f) &&
                  floatBits(ffn->ffnInputQuantMaxAbs) == floatBits(2.75f) &&
                  floatBits(ffn->productQuantMaxAbs) == floatBits(23.5f),
    "nested v105 FFN fields were not parsed recursively");

  const V105CudaPolicy::Decision decision = V105CudaPolicy::classify(105,trunk);
  requireContract(decision.isV105 && decision.hasQKNorm && decision.hasPositiveSwiGLUClip,
    "CUDA v105 policy did not recursively classify nested QKN/clip semantics");
  V105CudaPolicy::requireCurrentExecution(105,trunk,true);
  expectStringError([&](){
    V105CudaPolicy::requireCurrentExecution(105,trunk,false);
  },"nested v105 QKN/clip non-FP16 CUDA execution");
}

void testMandatoryFieldsAndTruncation() {
  AttentionWireOptions reducedAttention;
  reducedAttention.includeOutputRange = false;
  expectStringError([&](){
    istringstream in(attentionWire(reducedAttention));
    (void)TransformerAttentionDesc(in,105,false);
  },"v105 attention without output range");
  reducedAttention.includeInputRange = false;
  expectStringError([&](){
    istringstream in(attentionWire(reducedAttention));
    (void)TransformerAttentionDesc(in,105,false);
  },"v105 attention without either range");

  FfnWireOptions reducedFfn;
  reducedFfn.includeProductRange = false;
  expectStringError([&](){
    istringstream in(ffnWire(reducedFfn));
    (void)TransformerFFNDesc(in,105,false);
  },"v105 FFN without product range");
  reducedFfn.includeInputRange = false;
  expectStringError([&](){
    istringstream in(ffnWire(reducedFfn));
    (void)TransformerFFNDesc(in,105,false);
  },"v105 FFN without either PTQ range");

  const vector<string> attentionTokens = tokenize(attentionWire(AttentionWireOptions()));
  for(size_t count = 0; count < attentionTokens.size(); count++) {
    expectStringError([&](){
      istringstream in(prefixWire(attentionTokens,count));
      (void)TransformerAttentionDesc(in,105,false);
    },"truncated v105 attention at token " + Global::uint64ToString((uint64_t)count));
  }
  const vector<string> ffnTokens = tokenize(ffnWire(FfnWireOptions()));
  for(size_t count = 0; count < ffnTokens.size(); count++) {
    expectStringError([&](){
      istringstream in(prefixWire(ffnTokens,count));
      (void)TransformerFFNDesc(in,105,false);
    },"truncated v105 FFN at token " + Global::uint64ToString((uint64_t)count));
  }
}

void testInvalidScalarsAndGeometry() {
  const char* invalidRanges[] = {"0","-1","nan","inf","-inf"};
  for(const char* invalid: invalidRanges) {
    AttentionWireOptions inputOptions;
    inputOptions.inputRange = invalid;
    expectStringError([&](){
      istringstream in(attentionWire(inputOptions));
      (void)TransformerAttentionDesc(in,105,false);
    },string("invalid attention input range ") + invalid);

    AttentionWireOptions outputOptions;
    outputOptions.outputRange = invalid;
    expectStringError([&](){
      istringstream in(attentionWire(outputOptions));
      (void)TransformerAttentionDesc(in,105,false);
    },string("invalid attention output range ") + invalid);

    FfnWireOptions inputFfnOptions;
    inputFfnOptions.inputRange = invalid;
    expectStringError([&](){
      istringstream in(ffnWire(inputFfnOptions));
      (void)TransformerFFNDesc(in,105,false);
    },string("invalid FFN input range ") + invalid);

    FfnWireOptions productOptions;
    productOptions.productRange = invalid;
    expectStringError([&](){
      istringstream in(ffnWire(productOptions));
      (void)TransformerFFNDesc(in,105,false);
    },string("invalid FFN product range ") + invalid);
  }

  AttentionWireOptions tinyAttention;
  tinyAttention.inputRange = "1e-45";
  expectStringError([&](){
    istringstream in(attentionWire(tinyAttention));
    (void)TransformerAttentionDesc(in,105,false);
  },"attention input range with overflowing quantization multiplier");
  tinyAttention.inputRange = "3.25";
  tinyAttention.outputRange = "1e-45";
  expectStringError([&](){
    istringstream in(attentionWire(tinyAttention));
    (void)TransformerAttentionDesc(in,105,false);
  },"attention output range with overflowing quantization multiplier");
  FfnWireOptions tinyFfn;
  tinyFfn.inputRange = "1e-45";
  expectStringError([&](){
    istringstream in(ffnWire(tinyFfn));
    (void)TransformerFFNDesc(in,105,false);
  },"FFN input range with overflowing quantization multiplier");
  tinyFfn.inputRange = "2.75";
  tinyFfn.productRange = "1e-45";
  expectStringError([&](){
    istringstream in(ffnWire(tinyFfn));
    (void)TransformerFFNDesc(in,105,false);
  },"FFN product range with overflowing quantization multiplier");

  const char* invalidClips[] = {"-1","nan","inf","-inf"};
  for(const char* invalid: invalidClips) {
    FfnWireOptions options;
    options.clip = invalid;
    expectStringError([&](){
      istringstream in(ffnWire(options));
      (void)TransformerFFNDesc(in,105,false);
    },string("invalid SwiGLU clip ") + invalid);
  }
  FfnWireOptions clipWithoutSwiGLU;
  clipWithoutSwiGLU.useSwiGLU = false;
  clipWithoutSwiGLU.clip = "7";
  expectStringError([&](){
    istringstream in(ffnWire(clipWithoutSwiGLU));
    (void)TransformerFFNDesc(in,105,false);
  },"positive clip without SwiGLU");

  AttentionWireOptions badFlag;
  badFlag.qkNormFlag = 2;
  expectStringError([&](){
    istringstream in(attentionWire(badFlag));
    (void)TransformerAttentionDesc(in,105,false);
  },"non-boolean QK norm flag");

  AttentionWireOptions badQNorm;
  badQNorm.qNormChannels = 5;
  expectStringError([&](){
    istringstream in(attentionWire(badQNorm));
    (void)TransformerAttentionDesc(in,105,false);
  },"qNorm dimension not equal to qHeadDim");
  AttentionWireOptions badKNorm;
  badKNorm.kNormChannels = 5;
  expectStringError([&](){
    istringstream in(attentionWire(badKNorm));
    (void)TransformerAttentionDesc(in,105,false);
  },"kNorm dimension not equal to qHeadDim");
}

}  // namespace

int MainCmds::testv105wire(const vector<string>& args) {
  if(args.size() == 4 && args[0] == "testv105wire" && args[1] == "--model") {
    ModelDesc model;
    ModelDesc::loadFromFileMaybeGZipped(args[2],model,args[3]);
    requireContract(model.version == 105,"loader fixture is not canonical v105");
    const V105CudaPolicy::Decision decision =
      V105CudaPolicy::classify(model.version,model.trunk);
    requireContract(decision.hasQKNorm && decision.hasPositiveSwiGLUClip,
      "loader fixture does not contain QKN and positive clip");
    cout << "V105_MODEL_LOADER_PASS" << endl;
    return 0;
  }
  if(args.size() != 1 || args[0] != "testv105wire")
    throw StringError("testv105wire takes no arguments or --model FILE SHA256");

  testVersions();
  testProjectedScratchLayout();
  testLegacyV102();
  testV105AttentionVariantsAndMoves();
  testV105FfnMixedLayersAndMoves();
  testVersionPropagationThroughTrunk();
  testNestedBottleneckVersionPropagationAndPolicy();
  testMandatoryFieldsAndTruncation();
  testInvalidScalarsAndGeometry();

  cout << "V105_WIRE_CONTRACT_PASS" << endl;
  return 0;
}

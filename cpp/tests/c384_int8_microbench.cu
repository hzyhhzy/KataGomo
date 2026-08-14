#include "neuralnet/cuda_specialized/sm120/c384/experimental_int8/kernels.h"
#include "neuralnet/cuda_specialized/sm120/c384/fixed_batch/qknorm_rope.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace C384Int8Experiment;

void checkCuda(cudaError_t status, const char* operation) {
  if(status != cudaSuccess)
    throw std::runtime_error(
      std::string(operation) + ": " + cudaGetErrorString(status));
}

struct Options {
  ProjectionMode mode = ProjectionMode::AggressiveQkv;
  ProjectionTactic projection = ProjectionTactic::M128N128K64S3Sw2;
  DualFfnTactic dual = DualFfnTactic::M128N64K64S3Sw4;
  DualFfnDivide127Tactic divide127 = DualFfnDivide127Tactic::Incumbent;
  DownTactic down = DownTactic::M128N128K64S3Sw2;
  AttentionOutTactic attentionOut =
    AttentionOutTactic::M128N128K64S3Sw2;
  int streams = 1;
  int warmup = 10;
  int iterations = 100;
  bool contractsOnly = false;
};

int parsePositive(const char* text, const char* name) {
  char* end = nullptr;
  const long value = std::strtol(text,&end,10);
  if(end == text || *end != '\0' || value <= 0 || value > 1000000)
    throw std::runtime_error(std::string("invalid ") + name + ": " + text);
  return int(value);
}

Options parseOptions(int argc, char** argv) {
  Options options;
  for(int i = 1; i < argc; i++) {
    const std::string arg = argv[i];
    const auto requireValue = [&]() -> const char* {
      if(i + 1 >= argc)
        throw std::runtime_error("missing value after " + arg);
      return argv[++i];
    };
    if(arg == "--variant") {
      const std::string value = requireValue();
      if(value == "conservative")
        options.mode = ProjectionMode::ConservativeQk;
      else if(value == "aggressive")
        options.mode = ProjectionMode::AggressiveQkv;
      else
        throw std::runtime_error("--variant must be conservative or aggressive");
    }
    else if(arg == "--streams") {
      options.streams = parsePositive(requireValue(),"stream count");
      if(options.streams != 1 && options.streams != 2)
        throw std::runtime_error("--streams must be 1 or 2");
    }
    else if(arg == "--warmup")
      options.warmup = parsePositive(requireValue(),"warmup");
    else if(arg == "--iterations")
      options.iterations = parsePositive(requireValue(),"iterations");
    else if(arg == "--rows") {
      const int rows = parsePositive(requireValue(),"rows");
      if(rows != kTokenRows)
        throw std::runtime_error("this exact prototype admits only M=6300");
    }
    else if(arg == "--projection-tactic") {
      const int value = parsePositive(requireValue(),"projection tactic");
      if(value < 1 || value > 3)
        throw std::runtime_error("projection tactic must be 1..3");
      options.projection = static_cast<ProjectionTactic>(value);
    }
    else if(arg == "--dual-tactic") {
      const int value = parsePositive(requireValue(),"dual tactic");
      if(value < 1 || value > 3)
        throw std::runtime_error("dual tactic must be 1..3");
      options.dual = static_cast<DualFfnTactic>(value);
    }
    else if(arg == "--divide127") {
      const std::string value = requireValue();
      if(value == "incumbent")
        options.divide127 = DualFfnDivide127Tactic::Incumbent;
      else if(value == "exact-branchless")
        options.divide127 = DualFfnDivide127Tactic::ExactBranchless;
      else
        throw std::runtime_error(
          "--divide127 must be incumbent or exact-branchless");
    }
    else if(arg == "--down-tactic") {
      const int value = parsePositive(requireValue(),"down tactic");
      if(value < 1 || value > 3)
        throw std::runtime_error("down tactic must be 1..3");
      options.down = static_cast<DownTactic>(value);
    }
    else if(arg == "--attention-out-tactic") {
      const int value = parsePositive(requireValue(),"attention-out tactic");
      if(value < 1 || value > 3)
        throw std::runtime_error("attention-out tactic must be 1..3");
      options.attentionOut = static_cast<AttentionOutTactic>(value);
    }
    else if(arg == "--contracts-only")
      options.contractsOnly = true;
    else if(arg == "--help") {
      std::cout
        << "c384_int8_microbench [--variant conservative|aggressive] "
        << "[--streams 1|2] [--rows 6300] [--projection-tactic 1..3] "
        << "[--dual-tactic 1..3] [--down-tactic 1..3] "
        << "[--divide127 incumbent|exact-branchless] "
        << "[--attention-out-tactic 1..3] "
        << "[--warmup N] [--iterations N] [--contracts-only]\n";
      std::exit(0);
    }
    else
      throw std::runtime_error("unknown argument: " + arg);
  }
  if(options.divide127 == DualFfnDivide127Tactic::ExactBranchless &&
     (options.mode != ProjectionMode::AggressiveQkv ||
      options.dual != DualFfnTactic::M128N64K64S3Sw4))
    throw std::runtime_error(
      "exact-branchless divide127 requires aggressive variant and D2");
  return options;
}

const char* divide127TacticName(DualFfnDivide127Tactic tactic) {
  switch(tactic) {
  case DualFfnDivide127Tactic::Incumbent: return "incumbent";
  case DualFfnDivide127Tactic::ExactBranchless: return "exact-branchless";
  }
  return "invalid";
}

class GuardedBuffer {
public:
  static constexpr std::size_t kGuardBytes = 256;
  static constexpr unsigned char kCanary = 0xa5;

  GuardedBuffer() = default;
  explicit GuardedBuffer(std::size_t payloadBytes) { allocate(payloadBytes); }
  GuardedBuffer(const GuardedBuffer&) = delete;
  GuardedBuffer& operator=(const GuardedBuffer&) = delete;
  ~GuardedBuffer() { release(); }

  void allocate(std::size_t payloadBytes) {
    release();
    bytes_ = payloadBytes;
    checkCuda(cudaMalloc(&raw_,bytes_ + 2 * kGuardBytes),"cudaMalloc guarded");
    checkCuda(cudaMemset(raw_,kCanary,bytes_ + 2 * kGuardBytes),
      "cudaMemset guarded");
    // Contract kernels run on cudaStreamNonBlocking streams. A legacy-default
    // stream memset has no ordering relationship with such a stream, so make
    // initialization complete before exposing the guarded payload.
    checkCuda(cudaDeviceSynchronize(),"sync guarded initialization");
    payload_ = static_cast<unsigned char*>(raw_) + kGuardBytes;
  }

  template<typename T>
  T* data() { return reinterpret_cast<T*>(payload_); }
  template<typename T>
  const T* data() const { return reinterpret_cast<const T*>(payload_); }

  void zeroPayload() {
    checkCuda(cudaMemset(payload_,0,bytes_),"cudaMemset payload");
    checkCuda(cudaDeviceSynchronize(),"sync guarded payload memset");
  }

  template<typename T>
  void upload(const std::vector<T>& source) {
    if(source.size() * sizeof(T) != bytes_)
      throw std::runtime_error("guarded upload size mismatch");
    checkCuda(cudaMemcpy(payload_,source.data(),bytes_,cudaMemcpyHostToDevice),
      "cudaMemcpy upload");
  }

  template<typename T>
  std::vector<T> download() const {
    if(bytes_ % sizeof(T) != 0)
      throw std::runtime_error("guarded download type mismatch");
    std::vector<T> result(bytes_ / sizeof(T));
    checkCuda(cudaMemcpy(result.data(),payload_,bytes_,cudaMemcpyDeviceToHost),
      "cudaMemcpy download");
    return result;
  }

  void requireCanary(const char* name) const {
    std::array<unsigned char,kGuardBytes> before{};
    std::array<unsigned char,kGuardBytes> after{};
    checkCuda(cudaMemcpy(before.data(),raw_,kGuardBytes,cudaMemcpyDeviceToHost),
      "cudaMemcpy canary before");
    checkCuda(cudaMemcpy(after.data(),
      static_cast<const unsigned char*>(payload_) + bytes_,kGuardBytes,
      cudaMemcpyDeviceToHost),"cudaMemcpy canary after");
    const auto valid = [](const auto& guard) {
      return std::all_of(guard.begin(),guard.end(),
        [](unsigned char value) { return value == kCanary; });
    };
    if(!valid(before) || !valid(after))
      throw std::runtime_error(std::string(name) + ": canary corruption");
  }

private:
  void* raw_ = nullptr;
  void* payload_ = nullptr;
  std::size_t bytes_ = 0;

  void release() noexcept {
    if(raw_ != nullptr)
      (void)cudaFree(raw_);
    raw_ = nullptr;
    payload_ = nullptr;
    bytes_ = 0;
  }
};

struct ProjectionDeleter {
  void operator()(void* pointer) const noexcept { destroyProjection(pointer); }
};
struct DualDeleter {
  void operator()(void* pointer) const noexcept { destroyDualFfn(pointer); }
};
struct DownDeleter {
  void operator()(void* pointer) const noexcept { destroyDown(pointer); }
};
struct AttentionOutDeleter {
  void operator()(void* pointer) const noexcept {
    destroyAttentionOut(pointer);
  }
};
using ProjectionHandle = std::unique_ptr<void,ProjectionDeleter>;
using DualHandle = std::unique_ptr<void,DualDeleter>;
using DownHandle = std::unique_ptr<void,DownDeleter>;
using AttentionOutHandle = std::unique_ptr<void,AttentionOutDeleter>;

struct HostData {
  std::vector<half> input;
  std::vector<half> gamma;
  std::vector<int8_t> projectionWeights;
  std::vector<int8_t> upWeights;
  std::vector<int8_t> gateWeights;
  std::vector<int8_t> downWeights;
  std::vector<int8_t> attentionOutWeights;
  std::vector<half> residual;
  float projectionWeightScale = 1.0f / 64.0f;
  float upWeightScale = 1.0f / 96.0f;
  float gateWeightScale = 1.0f / 80.0f;
  float downWeightScale = 1.0f / 112.0f;
  float attentionOutWeightScale = 1.0f / 104.0f;
};

float toFloat(half value) { return __half2float(value); }
half toHalf(float value) { return __float2half_rn(value); }

int8_t patternWeight(std::size_t index, int salt) {
  const int value = int((index * std::size_t(17 + salt) + 13 * salt) % 15) - 7;
  return static_cast<int8_t>(value);
}

HostData makeHostData(ProjectionMode mode) {
  HostData data;
  data.input.resize(std::size_t(kTokenRows) * kChannels);
  data.gamma.resize(kChannels);
  const int projectionChannels = mode == ProjectionMode::ConservativeQk ?
    kQkChannels : kQkvChannels;
  data.projectionWeights.resize(std::size_t(kChannels) * projectionChannels);
  data.upWeights.resize(std::size_t(kChannels) * kFfnChannels);
  data.gateWeights.resize(std::size_t(kChannels) * kFfnChannels);
  data.downWeights.resize(std::size_t(kFfnChannels) * kChannels);
  data.attentionOutWeights.resize(std::size_t(kChannels) * kChannels);
  data.residual.resize(std::size_t(kTokenRows) * kChannels);

  for(int channel = 0; channel < kChannels; channel++)
    data.gamma[channel] = toHalf(0.75f + float(channel % 23) * (0.5f / 22.0f));
  for(int row = 0; row < kTokenRows; row++) {
    for(int channel = 0; channel < kChannels; channel++) {
      const int code = (row * 7 + channel * 13 + 5) % 63 - 31;
      data.input[std::size_t(row) * kChannels + channel] =
        toHalf(float(code) / 32.0f);
      const int residualCode = (row * 11 + channel * 3 + 1) % 31 - 15;
      data.residual[std::size_t(row) * kChannels + channel] =
        toHalf(float(residualCode) / 64.0f);
    }
  }
  for(std::size_t i = 0; i < data.projectionWeights.size(); i++)
    data.projectionWeights[i] = patternWeight(i,1);
  for(std::size_t i = 0; i < data.upWeights.size(); i++)
    data.upWeights[i] = patternWeight(i,2);
  for(std::size_t i = 0; i < data.gateWeights.size(); i++)
    data.gateWeights[i] = patternWeight(i,3);
  for(std::size_t i = 0; i < data.downWeights.size(); i++)
    data.downWeights[i] = patternWeight(i,4);
  for(std::size_t i = 0; i < data.attentionOutWeights.size(); i++)
    data.attentionOutWeights[i] = patternWeight(i,5);
  return data;
}

struct DeviceWeights {
  GuardedBuffer projection;
  GuardedBuffer up;
  GuardedBuffer gate;
  GuardedBuffer down;
  GuardedBuffer attentionOut;
  GuardedBuffer gamma;
  GuardedBuffer input;
  GuardedBuffer residual;

  DeviceWeights(const HostData& host)
    : projection(host.projectionWeights.size()),
      up(host.upWeights.size()),
      gate(host.gateWeights.size()),
      down(host.downWeights.size()),
      attentionOut(host.attentionOutWeights.size()),
      gamma(host.gamma.size() * sizeof(half)),
      input(host.input.size() * sizeof(half)),
      residual(host.residual.size() * sizeof(half)) {
    projection.upload(host.projectionWeights);
    up.upload(host.upWeights);
    gate.upload(host.gateWeights);
    down.upload(host.downWeights);
    attentionOut.upload(host.attentionOutWeights);
    gamma.upload(host.gamma);
    input.upload(host.input);
    residual.upload(host.residual);
  }
};

struct Lane {
  cudaStream_t stream = nullptr;
  GuardedBuffer normHalf{std::size_t(kTokenRows) * kChannels * sizeof(half)};
  GuardedBuffer normInt8{std::size_t(kTokenRows) * kChannels};
  GuardedBuffer normInt8Only{std::size_t(kTokenRows) * kChannels};
  GuardedBuffer rawPackedQkv{
    std::size_t(kTokenRows) * kQkvChannels * sizeof(half)};
  GuardedBuffer productHalf{
    std::size_t(kTokenRows) * kFfnChannels * sizeof(half)};
  GuardedBuffer productInt8{std::size_t(kTokenRows) * kFfnChannels};
  GuardedBuffer downOutput{
    std::size_t(kTokenRows) * kChannels * sizeof(half)};
  GuardedBuffer attentionInt8{std::size_t(kTokenRows) * kChannels};
  GuardedBuffer attentionOutput{
    std::size_t(kTokenRows) * kChannels * sizeof(half)};
  ProjectionHandle projection{nullptr};
  DualHandle dual{nullptr};
  // Legacy half-output handle is retained only to time the superseded
  // half-product -> standalone-quant control. It is never used by the
  // aggressive connected subpath.
  DualHandle legacyHalfDual{nullptr};
  DownHandle down{nullptr};
  AttentionOutHandle attentionOut{nullptr};

  Lane() { checkCuda(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),
    "cudaStreamCreateWithFlags"); }
  Lane(const Lane&) = delete;
  Lane& operator=(const Lane&) = delete;
  ~Lane() {
    attentionOut.reset();
    down.reset();
    legacyHalfDual.reset();
    dual.reset();
    projection.reset();
    if(stream != nullptr)
      (void)cudaStreamDestroy(stream);
  }
};

void prepareLane(
  Lane& lane,
  const Options& options,
  const HostData& host,
  const DeviceWeights& weights
) {
  lane.normHalf.zeroPayload();
  lane.normInt8.zeroPayload();
  lane.normInt8Only.zeroPayload();
  lane.rawPackedQkv.zeroPayload();
  lane.productHalf.zeroPayload();
  lane.productInt8.zeroPayload();
  lane.downOutput.zeroPayload();
  lane.attentionInt8.zeroPayload();
  lane.attentionOutput.zeroPayload();
  ProjectionConfig projectionConfig;
  projectionConfig.mode = options.mode;
  projectionConfig.tactic = options.projection;
  projectionConfig.maxTokenRows = kTokenRows;
  projectionConfig.packedWeights = weights.projection.data<int8_t>();
  projectionConfig.weightScale = host.projectionWeightScale;
  lane.projection.reset(createProjection(projectionConfig));
  if(lane.projection == nullptr)
    throw std::runtime_error("projection handle preparation failed");

  DualFfnConfig dualConfig;
  dualConfig.tactic = options.dual;
  dualConfig.outputMode = options.mode == ProjectionMode::AggressiveQkv ?
    DualFfnOutputMode::Int8Product : DualFfnOutputMode::Fp16Product;
  dualConfig.divide127Tactic = options.divide127;
  dualConfig.maxTokenRows = kTokenRows;
  dualConfig.packedUpWeights = weights.up.data<int8_t>();
  dualConfig.packedGateWeights = weights.gate.data<int8_t>();
  dualConfig.upWeightScale = host.upWeightScale;
  dualConfig.gateWeightScale = host.gateWeightScale;
  lane.dual.reset(createDualFfn(dualConfig));
  if(lane.dual == nullptr)
    throw std::runtime_error("dual-FFN handle preparation failed");

  if(options.mode == ProjectionMode::AggressiveQkv) {
    DualFfnConfig legacyConfig = dualConfig;
    legacyConfig.outputMode = DualFfnOutputMode::Fp16Product;
    legacyConfig.divide127Tactic = DualFfnDivide127Tactic::Incumbent;
    lane.legacyHalfDual.reset(createDualFfn(legacyConfig));
    if(lane.legacyHalfDual == nullptr)
      throw std::runtime_error("legacy half-dual control preparation failed");
    DownConfig downConfig;
    downConfig.tactic = options.down;
    downConfig.maxTokenRows = kTokenRows;
    downConfig.packedWeights = weights.down.data<int8_t>();
    downConfig.weightScale = host.downWeightScale;
    lane.down.reset(createDown(downConfig));
    if(lane.down == nullptr)
      throw std::runtime_error("down handle preparation failed");

    AttentionOutConfig attentionOutConfig;
    attentionOutConfig.tactic = options.attentionOut;
    attentionOutConfig.maxTokenRows = kTokenRows;
    attentionOutConfig.packedWeights = weights.attentionOut.data<int8_t>();
    attentionOutConfig.weightScale = host.attentionOutWeightScale;
    lane.attentionOut.reset(createAttentionOut(attentionOutConfig));
    if(lane.attentionOut == nullptr)
      throw std::runtime_error("attention-out handle preparation failed");
  }
}

// This is only the connected INT8 prototype subpath. It is intentionally not
// a transformer layer and must never be interpreted as whole-network timing.
void enqueuePrototypeSubpath(
  Lane& lane,
  const Options& options,
  const DeviceWeights& weights
) {
  if(options.mode == ProjectionMode::AggressiveQkv)
    checkCuda(launchRmsNormInt8(
      weights.input.data<half>(),lane.normInt8.data<int8_t>(),
      weights.gamma.data<half>(),kTokenRows,kRmsEpsilon,lane.stream),
      "launch INT8-only RMS");
  else
    checkCuda(launchRmsNormFp16Int8(
      weights.input.data<half>(),lane.normHalf.data<half>(),
      lane.normInt8.data<int8_t>(),weights.gamma.data<half>(),kTokenRows,
      kRmsEpsilon,lane.stream),"launch RMS FP16+INT8");
  checkCuda(launchProjection(
    lane.projection.get(),kTokenRows,lane.normInt8.data<int8_t>(),
    lane.rawPackedQkv.data<half>(),kQkvChannels,lane.stream),
    "launch INT8 packed projection");
  if(options.mode == ProjectionMode::AggressiveQkv) {
    checkCuda(launchQuantizeAttentionOutput(
      weights.input.data<half>(),lane.attentionInt8.data<int8_t>(),
      kTokenRows,lane.stream),"launch attention-output quantization");
    checkCuda(launchAttentionOutResidual(
      lane.attentionOut.get(),kTokenRows,lane.attentionInt8.data<int8_t>(),
      weights.residual.data<half>(),lane.attentionOutput.data<half>(),
      lane.stream),"launch INT8 attention-out residual");
    checkCuda(launchDualFfnInt8(
      lane.dual.get(),kTokenRows,lane.normInt8.data<int8_t>(),
      lane.productInt8.data<int8_t>(),lane.stream),
      "launch fused-output INT8 dual FFN");
    checkCuda(launchDownResidual(
      lane.down.get(),kTokenRows,lane.productInt8.data<int8_t>(),
      weights.residual.data<half>(),lane.downOutput.data<half>(),lane.stream),
      "launch INT8 down residual");
  }
  else {
    checkCuda(launchDualFfnHalf(
      lane.dual.get(),kTokenRows,lane.normInt8.data<int8_t>(),
      lane.productHalf.data<half>(),lane.stream),"launch INT8 dual FFN");
  }
}

enum class TimingFamily {
  RmsInt8Only,
  RmsFp16Int8Control,
  Projection,
  DualFfn,
  ProductQuantization,
  DownResidual,
  AttentionOutputQuantization,
  AttentionOutResidual,
  PrototypeSubpath,
};

const char* timingFamilyName(TimingFamily family) {
  switch(family) {
  case TimingFamily::RmsInt8Only: return "rms_int8_only";
  case TimingFamily::RmsFp16Int8Control: return "rms_fp16_int8_control";
  case TimingFamily::Projection: return "packed_projection";
  case TimingFamily::DualFfn: return "dual_clip7_mode_output";
  case TimingFamily::ProductQuantization: return "legacy_product_quantization";
  case TimingFamily::DownResidual: return "down_beta1_residual";
  case TimingFamily::AttentionOutputQuantization:
    return "attention_output_clip4_quantization";
  case TimingFamily::AttentionOutResidual:
    return "attention_out_k384_beta1_residual";
  case TimingFamily::PrototypeSubpath: return "prototype_subpath";
  }
  return "invalid";
}

const char* omittedFp16AndAttention(ProjectionMode mode) {
  return mode == ProjectionMode::ConservativeQk ?
    "residual_add_after_attention,fp16_v_projection,qknorm_rope,fa4,"
    "attention_out_projection,second_rms,fp16_down_residual,"
    "policy_value_heads,36_layer_scheduling" :
    "qknorm_rope,fa4,second_rms,policy_value_heads,36_layer_scheduling";
}

void enqueueFamily(
  Lane& lane,
  TimingFamily family,
  const Options& options,
  const DeviceWeights& weights
) {
  switch(family) {
  case TimingFamily::RmsInt8Only:
    checkCuda(launchRmsNormInt8(
      weights.input.data<half>(),lane.normInt8.data<int8_t>(),
      weights.gamma.data<half>(),kTokenRows,kRmsEpsilon,lane.stream),
      "launch INT8-only RMS");
    return;
  case TimingFamily::RmsFp16Int8Control:
    checkCuda(launchRmsNormFp16Int8(
      weights.input.data<half>(),lane.normHalf.data<half>(),
      lane.normInt8.data<int8_t>(),weights.gamma.data<half>(),kTokenRows,
      kRmsEpsilon,lane.stream),"launch RMS FP16+INT8");
    return;
  case TimingFamily::Projection:
    checkCuda(launchProjection(
      lane.projection.get(),kTokenRows,lane.normInt8.data<int8_t>(),
      lane.rawPackedQkv.data<half>(),kQkvChannels,lane.stream),
      "launch INT8 packed projection");
    return;
  case TimingFamily::DualFfn:
    if(options.mode == ProjectionMode::AggressiveQkv)
      checkCuda(launchDualFfnInt8(
        lane.dual.get(),kTokenRows,lane.normInt8.data<int8_t>(),
        lane.productInt8.data<int8_t>(),lane.stream),
        "launch fused-output INT8 dual FFN");
    else
      checkCuda(launchDualFfnHalf(
        lane.dual.get(),kTokenRows,lane.normInt8.data<int8_t>(),
        lane.productHalf.data<half>(),lane.stream),"launch INT8 dual FFN");
    return;
  case TimingFamily::ProductQuantization:
    if(options.mode != ProjectionMode::AggressiveQkv)
      throw std::runtime_error("product quantization is aggressive-only");
    checkCuda(launchQuantizeClip7Product(
      lane.productHalf.data<half>(),lane.productInt8.data<int8_t>(),
      kTokenRows,lane.stream),"launch product quantization");
    return;
  case TimingFamily::DownResidual:
    if(options.mode != ProjectionMode::AggressiveQkv)
      throw std::runtime_error("INT8 down is aggressive-only");
    checkCuda(launchDownResidual(
      lane.down.get(),kTokenRows,lane.productInt8.data<int8_t>(),
      weights.residual.data<half>(),lane.downOutput.data<half>(),lane.stream),
      "launch INT8 down residual");
    return;
  case TimingFamily::AttentionOutputQuantization:
    if(options.mode != ProjectionMode::AggressiveQkv)
      throw std::runtime_error("attention-output quantization is aggressive-only");
    checkCuda(launchQuantizeAttentionOutput(
      weights.input.data<half>(),lane.attentionInt8.data<int8_t>(),
      kTokenRows,lane.stream),"launch attention-output quantization");
    return;
  case TimingFamily::AttentionOutResidual:
    if(options.mode != ProjectionMode::AggressiveQkv)
      throw std::runtime_error("INT8 attention-out is aggressive-only");
    checkCuda(launchAttentionOutResidual(
      lane.attentionOut.get(),kTokenRows,lane.attentionInt8.data<int8_t>(),
      weights.residual.data<half>(),lane.attentionOutput.data<half>(),
      lane.stream),"launch INT8 attention-out residual");
    return;
  case TimingFamily::PrototypeSubpath:
    enqueuePrototypeSubpath(lane,options,weights);
    return;
  }
  throw std::runtime_error("invalid timing family");
}

int quantize(float value, float clip) {
  value = std::max(-clip,std::min(clip,value));
  long result = std::lrint(value * (127.0f / clip));
  result = std::max(-127L,std::min(127L,result));
  return int(result);
}

int fusedClip7ProductOracle(float up, float gate) {
  const float silu = up / (1.0f + std::exp(-up));
  const int upFactor = quantize(silu,7.0f);
  const int gateFactor = quantize(gate,7.0f);
  long product = std::lrint(float(upFactor * gateFactor) / 127.0f);
  product = std::max(-127L,std::min(127L,product));
  return int(product);
}

int adjustableProductOracle(
  float up,
  float gate,
  float swigluClip,
  float productQuantMaxAbs
) {
  const float silu = up / (1.0f + std::exp(-up));
  const int upFactor = quantize(silu,swigluClip);
  const int gateFactor = quantize(gate,swigluClip);
  // Match createDualFfn's immutable FP32 epilogue parameter, including its
  // deliberate double-precision construction followed by one FP32 rounding.
  const float productMultiplier = float(
    double(swigluClip) * double(swigluClip) /
    (127.0 * double(productQuantMaxAbs)));
  long product = std::lrint(float(upFactor * gateFactor) * productMultiplier);
  product = std::max(-127L,std::min(127L,product));
  return int(product);
}

void requireFinite(const std::vector<half>& values, const char* name) {
  for(std::size_t i = 0; i < values.size(); i++) {
    if(!std::isfinite(toFloat(values[i])))
      throw std::runtime_error(
        std::string(name) + ": non-finite at " + std::to_string(i));
  }
}

int32_t dot(
  const std::vector<int8_t>& activation,
  int row,
  int innerChannels,
  const std::vector<int8_t>& packedWeights,
  int outputChannel
) {
  int32_t accum = 0;
  const std::size_t aBase = std::size_t(row) * innerChannels;
  const std::size_t wBase = std::size_t(outputChannel) * innerChannels;
  for(int k = 0; k < innerChannels; k++)
    accum += int32_t(activation[aBase + k]) * int32_t(packedWeights[wBase + k]);
  return accum;
}

void requireNear(
  float actual,
  float expected,
  float tolerance,
  const std::string& label
) {
  if(!std::isfinite(actual) || std::fabs(actual - expected) > tolerance)
    throw std::runtime_error(label + ": actual=" + std::to_string(actual) +
      " expected=" + std::to_string(expected) +
      " tolerance=" + std::to_string(tolerance));
}

void checkAttentionOutputQuantizerFixture(cudaStream_t stream) {
  std::vector<half> input(kChannels,toHalf(0.0f));
  input[0] = toHalf(100.0f);
  input[1] = toHalf(-100.0f);
  input[2] = toHalf(4.0f);
  input[3] = toHalf(-4.0f);
  // 2 * 127 / 4 = 63.5 exactly: RNE must choose even 64. The negative
  // counterpart must analogously choose -64.
  input[4] = toHalf(2.0f);
  input[5] = toHalf(-2.0f);
  GuardedBuffer inputDevice(input.size() * sizeof(half));
  GuardedBuffer outputDevice(input.size());
  inputDevice.upload(input);
  outputDevice.zeroPayload();
  checkCuda(launchQuantizeAttentionOutput(
    inputDevice.data<half>(),outputDevice.data<int8_t>(),1,stream),
    "launch attention-output quantizer fixture");
  checkCuda(cudaStreamSynchronize(stream),
    "attention-output quantizer fixture sync");
  const std::vector<int8_t> output = outputDevice.download<int8_t>();
  const std::array<int,7> expected{{127,-127,127,-127,64,-64,0}};
  for(std::size_t i = 0; i < expected.size(); i++)
    if(int(output[i]) != expected[i])
      throw std::runtime_error(
        "attention-output saturation/RNE fixture mismatch at " +
        std::to_string(i));
  if(std::find(output.begin(),output.end(),int8_t(-128)) != output.end())
    throw std::runtime_error(
      "attention-output saturation/RNE fixture emitted forbidden -128");
  inputDevice.requireCanary("attention-output quantizer fixture input");
  outputDevice.requireCanary("attention-output quantizer fixture output");
}

// Dedicated saturation fixture. It is never used by benchmarkFamily(): the
// timed path retains its representative weight scales and deterministic data.
// SiLU(x) cannot reach the negative clip boundary, so the reachable clip7
// contract is SiLU(up)>+7 together with gate>+7 and gate<-7.
void checkClip7SaturationContract(
  Lane& lane,
  const Options& options,
  const std::vector<int8_t>& normInt8
) {
  constexpr int positiveChannel = 0;
  constexpr int negativeChannel = 1;
  constexpr int clampWitnessChannel = 2;
  constexpr float saturationWeightScale = 1.0f / 8.0f;
  std::vector<int8_t> upWeights(std::size_t(kFfnChannels) * kChannels,0);
  std::vector<int8_t> gateWeights(std::size_t(kFfnChannels) * kChannels,0);
  int32_t magnitude = 0;
  for(int k = 0; k < kChannels; k++) {
    const int value = int(normInt8[k]);
    magnitude += std::abs(value);
    const int8_t alignedSign = value < 0 ? int8_t(-1) : int8_t(1);
    upWeights[std::size_t(positiveChannel) * kChannels + k] = alignedSign;
    upWeights[std::size_t(negativeChannel) * kChannels + k] = alignedSign;
    upWeights[std::size_t(clampWitnessChannel) * kChannels + k] = alignedSign;
    gateWeights[std::size_t(positiveChannel) * kChannels + k] = alignedSign;
    gateWeights[std::size_t(negativeChannel) * kChannels + k] =
      int8_t(-alignedSign);
  }
  // Channel 2 deliberately pairs a saturated up factor with a sub-clip gate.
  // Its fused result differs from a missing-up-clamp implementation even
  // though both endpoint channels still saturate to +/-49.
  const float alpha = kNormActivationScale * saturationWeightScale;
  const int32_t gateTarget = int32_t(std::lrint(3.0f / alpha));
  int32_t gateMagnitude = 0;
  for(int k = 0; k < kChannels && gateMagnitude < gateTarget; k++) {
    const int value = int(normInt8[k]);
    gateWeights[std::size_t(clampWitnessChannel) * kChannels + k] =
      value < 0 ? int8_t(-1) : int8_t(1);
    gateMagnitude += std::abs(value);
  }
  GuardedBuffer upDevice(upWeights.size());
  GuardedBuffer gateDevice(gateWeights.size());
  upDevice.upload(upWeights);
  gateDevice.upload(gateWeights);
  DualFfnConfig config;
  config.tactic = options.dual;
  config.outputMode = options.mode == ProjectionMode::AggressiveQkv ?
    DualFfnOutputMode::Int8Product : DualFfnOutputMode::Fp16Product;
  config.divide127Tactic = options.divide127;
  config.maxTokenRows = 1;
  config.packedUpWeights = upDevice.data<int8_t>();
  config.packedGateWeights = gateDevice.data<int8_t>();
  config.upWeightScale = saturationWeightScale;
  config.gateWeightScale = saturationWeightScale;
  DualHandle saturatedDual{createDualFfn(config)};
  if(saturatedDual == nullptr)
    throw std::runtime_error("clip7 saturation fixture handle preparation failed");
  if(options.mode == ProjectionMode::AggressiveQkv)
    checkCuda(launchDualFfnInt8(
      saturatedDual.get(),1,lane.normInt8.data<int8_t>(),
      lane.productInt8.data<int8_t>(),lane.stream),
      "launch fused clip7 saturation fixture");
  else
    checkCuda(launchDualFfnHalf(
      saturatedDual.get(),1,lane.normInt8.data<int8_t>(),
      lane.productHalf.data<half>(),lane.stream),
      "launch clip7 saturation fixture");
  checkCuda(cudaStreamSynchronize(lane.stream),
    "clip7 saturation fixture sync");
  const float up = float(magnitude) * alpha;
  const float positiveGate = float(magnitude) * alpha;
  const float negativeGate = -float(magnitude) * alpha;
  const float witnessGate = float(gateMagnitude) * alpha;
  const float silu = up / (1.0f + std::exp(-up));
  if(!(silu > 7.0f && positiveGate > 7.0f && negativeGate < -7.0f))
    throw std::runtime_error("clip7 saturation fixture did not cross all reachable boundaries");
  const float expectedPositive = 49.0f;
  const float expectedNegative = -49.0f;
  if(options.mode == ProjectionMode::AggressiveQkv) {
    std::array<int8_t,3> actual{};
    checkCuda(cudaMemcpy(actual.data(),lane.productInt8.data<int8_t>(),
      sizeof(actual),cudaMemcpyDeviceToHost),
      "copy fused clip7 saturation fixture");
    if(int(actual[positiveChannel]) != 127 ||
       int(actual[negativeChannel]) != -127)
      throw std::runtime_error("fused clip7 endpoints did not emit +/-127");
    const int expectedWitness = fusedClip7ProductOracle(up,witnessGate);
    const int missingClampWitness = quantize(silu * witnessGate,49.0f);
    if(expectedWitness == missingClampWitness)
      throw std::runtime_error("fused clip7 witness cannot detect missing clamp");
    if(int(actual[clampWitnessChannel]) != expectedWitness)
      throw std::runtime_error("fused clip7 clamp-witness oracle mismatch");
    lane.productInt8.requireCanary("fused clip7 fixture product");
  }
  else {
    std::array<half,3> actual{};
    checkCuda(cudaMemcpy(actual.data(),lane.productHalf.data<half>(),
      sizeof(actual),cudaMemcpyDeviceToHost),"copy clip7 saturation fixture");
    requireNear(toFloat(actual[positiveChannel]),expectedPositive,0.05f,
      "clip7 positive saturation oracle");
    requireNear(toFloat(actual[negativeChannel]),expectedNegative,0.05f,
      "clip7 negative-gate saturation oracle");
    lane.productHalf.requireCanary("clip7 fixture product");
  }
  if(std::fabs(silu * 7.0f - expectedPositive) < 1.0f ||
     std::fabs(7.0f * positiveGate - expectedPositive) < 1.0f ||
     std::fabs(silu * -7.0f - expectedNegative) < 1.0f ||
     std::fabs(7.0f * negativeGate - expectedNegative) < 1.0f)
    throw std::runtime_error("clip7 fixture cannot distinguish a missing clamp");
  upDevice.requireCanary("clip7 fixture up weights");
  gateDevice.requireCanary("clip7 fixture gate weights");
}

// Per-layer v105 calibration contract. This is intentionally a small M=29
// tail fixture and is called only from checkContracts(), never from any timed
// benchmark family. The ordinary lane above therefore remains the exact
// clip=7/productMax=49 production control.
void checkAdjustableProductScaleContracts(
  Lane& lane,
  const Options& options
) {
  if(options.mode != ProjectionMode::AggressiveQkv)
    return;

  constexpr int rows = 29;
  constexpr float upWeightScale = 0.25f;
  constexpr float gateWeightScale = 0.25f;
  struct ScaleCase {
    float swigluClip;
    float productQuantMaxAbs;
    const char* label;
    const char* expectedPath;
  };
  constexpr std::array<ScaleCase,3> cases{{
    {7.0f,73.4171f,"clip7-product73.4171",
     "clip7-fixed-factor-v105-product-float-rne-v1"},
    {7.0f,47.7371f,"clip7-product47.7371",
     "clip7-fixed-factor-v105-product-float-rne-v1"},
    {6.0f,48.0f,"clip6-product48","v105-per-layer-float-rne-v1"},
  }};

  std::vector<int8_t> activation(std::size_t(rows) * kChannels,0);
  std::vector<int8_t> upWeights(std::size_t(kFfnChannels) * kChannels,0);
  std::vector<int8_t> gateWeights(std::size_t(kFfnChannels) * kChannels,0);
  for(int row = 0; row < rows; row++) {
    // The final row is deliberately distinct so a missing M=29 tail cannot
    // accidentally agree with an earlier row.
    activation[std::size_t(row) * kChannels] =
      static_cast<int8_t>(31 + (row * 37) % 97);
  }
  for(int channel = 0; channel < kFfnChannels; channel++) {
    upWeights[std::size_t(channel) * kChannels] =
      static_cast<int8_t>((channel * 7 + 3) % 19 - 9);
    gateWeights[std::size_t(channel) * kChannels] =
      static_cast<int8_t>((channel * 11 + 5) % 23 - 11);
  }

  GuardedBuffer activationDevice(activation.size());
  GuardedBuffer upDevice(upWeights.size());
  GuardedBuffer gateDevice(gateWeights.size());
  activationDevice.upload(activation);
  upDevice.upload(upWeights);
  gateDevice.upload(gateWeights);

  for(std::size_t caseIndex = 0; caseIndex < cases.size(); caseIndex++) {
    const ScaleCase& scale = cases[caseIndex];
    GuardedBuffer productDevice(std::size_t(rows) * kFfnChannels);
    productDevice.zeroPayload();

    DualFfnConfig dualConfig;
    dualConfig.tactic = options.dual;
    dualConfig.outputMode = DualFfnOutputMode::Int8Product;
    dualConfig.maxTokenRows = rows;
    dualConfig.packedUpWeights = upDevice.data<int8_t>();
    dualConfig.packedGateWeights = gateDevice.data<int8_t>();
    dualConfig.upWeightScale = upWeightScale;
    dualConfig.gateWeightScale = gateWeightScale;
    dualConfig.swigluClip = scale.swigluClip;
    dualConfig.productQuantMaxAbs = scale.productQuantMaxAbs;
    DualHandle dual{createDualFfn(dualConfig)};
    if(dual == nullptr)
      throw std::runtime_error(
        std::string(scale.label) + ": dynamic dual preparation failed");
    if(std::strcmp(
         dualFfnProductQuantPath(dual.get()),
         scale.expectedPath) != 0)
      throw std::runtime_error(
        std::string(scale.label) + ": wrong dynamic product path marker");
    if(!dualFfnSupports(
         dual.get(),DualFfnOutputMode::Int8Product,rows) ||
       dualFfnSupports(
         dual.get(),DualFfnOutputMode::Int8Product,rows + 1))
      throw std::runtime_error(
        std::string(scale.label) + ": M=29 tail support gate mismatch");

    checkCuda(launchDualFfnInt8(
      dual.get(),rows,activationDevice.data<int8_t>(),
      productDevice.data<int8_t>(),lane.stream),
      "launch adjustable-scale M=29 dual fixture");
    checkCuda(cudaStreamSynchronize(lane.stream),
      "adjustable-scale M=29 dual fixture sync");
    const std::vector<int8_t> product = productDevice.download<int8_t>();
    if(std::find(product.begin(),product.end(),int8_t(-128)) != product.end())
      throw std::runtime_error(
        std::string(scale.label) + ": emitted forbidden -128");

    for(int row = 0; row < rows; row++) {
      for(int channel = 0; channel < kFfnChannels; channel++) {
        const int32_t upAccum = dot(
          activation,row,kChannels,upWeights,channel);
        const int32_t gateAccum = dot(
          activation,row,kChannels,gateWeights,channel);
        const float up = float(upAccum) * kNormActivationScale *
          upWeightScale;
        const float gate = float(gateAccum) * kNormActivationScale *
          gateWeightScale;
        const int expected = adjustableProductOracle(
          up,gate,scale.swigluClip,scale.productQuantMaxAbs);
        const std::size_t index =
          std::size_t(row) * kFfnChannels + channel;
        if(int(product[index]) != expected) {
          throw std::runtime_error(
            std::string(scale.label) +
            ": factorwise oracle mismatch at row=" +
            std::to_string(row) + " channel=" + std::to_string(channel) +
            " actual=" + std::to_string(int(product[index])) +
            " expected=" + std::to_string(expected));
        }
      }
    }

    // One calibrated producer is also connected to a paired beta=1 down
    // consumer. A second prepared consumer differs only in productMax and
    // must be rejected by the explicit producer/consumer contract.
    if(caseIndex == 0) {
      constexpr float downWeightScale = 1.0f / 64.0f;
      std::vector<int8_t> downWeights(
        std::size_t(kChannels) * kFfnChannels,0);
      std::vector<half> residual(std::size_t(rows) * kChannels);
      for(int channel = 0; channel < kChannels; channel++) {
        downWeights[std::size_t(channel) * kFfnChannels] =
          static_cast<int8_t>((channel * 3) % 9 - 4);
        downWeights[std::size_t(channel) * kFfnChannels + 1] =
          static_cast<int8_t>((channel * 5 + 1) % 9 - 4);
      }
      for(int row = 0; row < rows; row++) {
        for(int channel = 0; channel < kChannels; channel++) {
          const int code = (row * 13 + channel * 7 + 2) % 25 - 12;
          residual[std::size_t(row) * kChannels + channel] =
            toHalf(float(code) / 64.0f);
        }
      }

      GuardedBuffer downWeightsDevice(downWeights.size());
      GuardedBuffer residualDevice(residual.size() * sizeof(half));
      GuardedBuffer downOutputDevice(residual.size() * sizeof(half));
      downWeightsDevice.upload(downWeights);
      residualDevice.upload(residual);
      downOutputDevice.zeroPayload();

      DownConfig downConfig;
      downConfig.tactic = options.down;
      downConfig.maxTokenRows = rows;
      downConfig.packedWeights = downWeightsDevice.data<int8_t>();
      downConfig.weightScale = downWeightScale;
      downConfig.productQuantMaxAbs = scale.productQuantMaxAbs;
      DownHandle pairedDown{createDown(downConfig)};
      if(pairedDown == nullptr)
        throw std::runtime_error(
          "adjustable-scale paired down preparation failed");
      DownConfig mismatchedConfig = downConfig;
      mismatchedConfig.productQuantMaxAbs =
        scale.productQuantMaxAbs + 0.125f;
      DownHandle mismatchedDown{createDown(mismatchedConfig)};
      if(mismatchedDown == nullptr)
        throw std::runtime_error(
          "adjustable-scale mismatched down preparation failed");
      DownConfig overflowingConfig = downConfig;
      overflowingConfig.weightScale = std::numeric_limits<float>::max();
      overflowingConfig.productQuantMaxAbs =
        std::numeric_limits<float>::max();
      DownHandle overflowingDown{createDown(overflowingConfig)};
      if(overflowingDown != nullptr)
        throw std::runtime_error(
          "adjustable-scale down accepted non-finite derived alpha");
      if(!dualFfnDownProductQuantizationMatches(
           dual.get(),pairedDown.get()) ||
         dualFfnDownProductQuantizationMatches(
           dual.get(),mismatchedDown.get()))
        throw std::runtime_error(
          "adjustable-scale dual/down product domain match gate failed");

      checkCuda(launchDownResidual(
        pairedDown.get(),rows,productDevice.data<int8_t>(),
        residualDevice.data<half>(),downOutputDevice.data<half>(),lane.stream),
        "launch adjustable-scale paired down fixture");
      checkCuda(cudaStreamSynchronize(lane.stream),
        "adjustable-scale paired down fixture sync");
      const std::vector<half> downOutput =
        downOutputDevice.download<half>();
      requireFinite(downOutput,"adjustable-scale paired down output");
      const float alpha =
        (scale.productQuantMaxAbs / 127.0f) * downWeightScale;
      for(int row = 0; row < rows; row++) {
        for(int channel = 0; channel < kChannels; channel++) {
          const int32_t accum = dot(
            product,row,kFfnChannels,downWeights,channel);
          const std::size_t index =
            std::size_t(row) * kChannels + channel;
          const float expected =
            float(accum) * alpha + toFloat(residual[index]);
          requireNear(toFloat(downOutput[index]),expected,0.02f,
            "adjustable-scale down beta1 oracle");
        }
      }
      productDevice.requireCanary(
        "adjustable-scale paired down product input");
      downWeightsDevice.requireCanary(
        "adjustable-scale paired down weights");
      residualDevice.requireCanary(
        "adjustable-scale paired down residual");
      downOutputDevice.requireCanary(
        "adjustable-scale paired down output");
    }

    activationDevice.requireCanary("adjustable-scale dual input");
    upDevice.requireCanary("adjustable-scale dual up weights");
    gateDevice.requireCanary("adjustable-scale dual gate weights");
    productDevice.requireCanary("adjustable-scale dual output");
  }
}

// Real-kernel equivalence proof for the production clip7/per-layer-product
// hybrid. Auto uses fixed clip7 factor epilogues and a dynamic product
// multiplier; the test-only control forces the former fully-adjustable
// factor epilogues. They must produce identical bytes at all important D2
// boundaries, including both production M and its 28-row tile tail.
void checkClip7HybridGpuContract(
  Lane& lane,
  const Options& options,
  const HostData& host,
  const DeviceWeights& weights
) {
  if(options.mode != ProjectionMode::AggressiveQkv)
    return;

  constexpr std::array<float,2> productMaxCases{{73.4171f,47.7371f}};
  constexpr std::array<int,4> rowCases{{1,28,29,kTokenRows}};
  static_assert(kTokenRows == 6300,"production M contract changed");
  static_assert(kTokenRows % 128 == 28,"production D2 tail contract changed");

  std::size_t comparedElements = 0;
  for(const float productMax: productMaxCases) {
    DualFfnConfig hybridConfig;
    hybridConfig.tactic = DualFfnTactic::M128N64K64S3Sw4;
    hybridConfig.outputMode = DualFfnOutputMode::Int8Product;
    hybridConfig.divide127Tactic = DualFfnDivide127Tactic::Incumbent;
    hybridConfig.productPathTactic = DualFfnProductPathTactic::Auto;
    hybridConfig.maxTokenRows = kTokenRows;
    hybridConfig.packedUpWeights = weights.up.data<int8_t>();
    hybridConfig.packedGateWeights = weights.gate.data<int8_t>();
    hybridConfig.upWeightScale = host.upWeightScale;
    hybridConfig.gateWeightScale = host.gateWeightScale;
    hybridConfig.swigluClip = 7.0f;
    hybridConfig.productQuantMaxAbs = productMax;
    DualFfnConfig fullConfig = hybridConfig;
    fullConfig.productPathTactic =
      DualFfnProductPathTactic::FullyAdjustableFloat;

    DualHandle hybrid{createDualFfn(hybridConfig)};
    DualHandle full{createDualFfn(fullConfig)};
    if(hybrid == nullptr || full == nullptr)
      throw std::runtime_error(
        "clip7 hybrid/full GPU A/B handle preparation failed");
    if(std::strcmp(
         dualFfnProductQuantPath(hybrid.get()),
         "clip7-fixed-factor-v105-product-float-rne-v1") != 0 ||
       std::strcmp(
         dualFfnProductQuantPath(full.get()),
         "v105-per-layer-float-rne-v1") != 0)
      throw std::runtime_error("clip7 hybrid/full GPU A/B marker mismatch");

    DualFfnConfig forbidden = fullConfig;
    forbidden.divide127Tactic = DualFfnDivide127Tactic::ExactBranchless;
    DualHandle unexpected{createDualFfn(forbidden)};
    if(unexpected != nullptr)
      throw std::runtime_error(
        "test-only fully-adjustable path reused divide127 tactic");

    for(const int rows: rowCases) {
      const std::size_t activationElements =
        std::size_t(rows) * kChannels;
      const std::size_t outputElements =
        std::size_t(rows) * kFfnChannels;
      std::vector<int8_t> activation(activationElements);
      for(int row = 0; row < rows; row++) {
        for(int channel = 0; channel < kChannels; channel++) {
          const int code =
            (row * 47 + channel * 31 + row * channel * 3 + 19) % 255 - 127;
          activation[std::size_t(row) * kChannels + channel] =
            static_cast<int8_t>(code);
        }
      }
      GuardedBuffer activationDevice(activationElements);
      GuardedBuffer hybridOutput(outputElements);
      GuardedBuffer fullOutput(outputElements);
      activationDevice.upload(activation);
      hybridOutput.zeroPayload();
      fullOutput.zeroPayload();
      checkCuda(launchDualFfnInt8(
        hybrid.get(),rows,activationDevice.data<int8_t>(),
        hybridOutput.data<int8_t>(),lane.stream),
        "launch clip7 hybrid GPU contract");
      checkCuda(launchDualFfnInt8(
        full.get(),rows,activationDevice.data<int8_t>(),
        fullOutput.data<int8_t>(),lane.stream),
        "launch fully-adjustable GPU control");
      checkCuda(cudaStreamSynchronize(lane.stream),
        "clip7 hybrid/full GPU A/B sync");
      const std::vector<int8_t> hybridBytes =
        hybridOutput.download<int8_t>();
      const std::vector<int8_t> fullBytes = fullOutput.download<int8_t>();
      std::size_t mismatchCount = 0;
      std::size_t firstMismatch = outputElements;
      for(std::size_t i = 0; i < outputElements; i++) {
        if(hybridBytes[i] != fullBytes[i]) {
          if(firstMismatch == outputElements)
            firstMismatch = i;
          mismatchCount++;
        }
      }
      const std::size_t hybridNeg128 = std::count(
        hybridBytes.begin(),hybridBytes.end(),int8_t(-128));
      const std::size_t fullNeg128 = std::count(
        fullBytes.begin(),fullBytes.end(),int8_t(-128));
      activationDevice.requireCanary("clip7 hybrid GPU contract activation");
      hybridOutput.requireCanary("clip7 hybrid GPU contract output");
      fullOutput.requireCanary("fully-adjustable GPU control output");
      if(mismatchCount != 0) {
        const int firstRow = int(firstMismatch / kFfnChannels);
        const int firstChannel = int(firstMismatch % kFfnChannels);
        const int32_t upAccum = dot(
          activation,firstRow,kChannels,host.upWeights,firstChannel);
        const int32_t gateAccum = dot(
          activation,firstRow,kChannels,host.gateWeights,firstChannel);
        const float up = float(upAccum) * kNormActivationScale *
          host.upWeightScale;
        const float gate = float(gateAccum) * kNormActivationScale *
          host.gateWeightScale;
        const float silu = up / (1.0f + std::exp(-up));
        const int upFactor = quantize(silu,7.0f);
        const int gateFactor = quantize(gate,7.0f);
        const int oracle = adjustableProductOracle(
          up,gate,7.0f,productMax);
        std::cout
          << std::setprecision(9)
          << "KATAGO_C384_INT8_CLIP7_HYBRID_GPU_CONTRACT_DIAGNOSTIC"
          << " product_max=" << productMax
          << " rows=" << rows
          << " mismatches=" << mismatchCount
          << " first_index=" << firstMismatch
          << " first_row=" << firstRow
          << " first_channel=" << firstChannel
          << " hybrid=" << int(hybridBytes[firstMismatch])
          << " full=" << int(fullBytes[firstMismatch])
          << " oracle=" << oracle
          << " hybrid_matches_oracle="
          << (int(hybridBytes[firstMismatch]) == oracle ? 1 : 0)
          << " full_matches_oracle="
          << (int(fullBytes[firstMismatch]) == oracle ? 1 : 0)
          << " up_accum=" << upAccum
          << " gate_accum=" << gateAccum
          << " up=" << up
          << " silu_up=" << silu
          << " gate=" << gate
          << " up_factor_oracle=" << upFactor
          << " gate_factor_oracle=" << gateFactor
          << " factor_product=" << upFactor * gateFactor
          << " hybrid_neg128=" << hybridNeg128
          << " full_neg128=" << fullNeg128
          << " canary=1\n";
        throw std::runtime_error(
          "clip7 hybrid differs from fully-adjustable GPU control: " +
          std::to_string(mismatchCount) + " mismatches, first=" +
          std::to_string(firstMismatch));
      }
      if(std::find(hybridBytes.begin(),hybridBytes.end(),int8_t(-128)) !=
           hybridBytes.end())
        throw std::runtime_error("clip7 hybrid GPU contract emitted -128");
      comparedElements += outputElements;
    }
  }
  weights.up.requireCanary("clip7 hybrid GPU contract up weights");
  weights.gate.requireCanary("clip7 hybrid GPU contract gate weights");
  std::cout
    << "KATAGO_C384_INT8_CLIP7_HYBRID_GPU_CONTRACT_PASS"
    << " product_max=73.4171,47.7371"
    << " geometry=D2 M=1,28,29,6300 production_M6300_tail28=1"
    << " reference=fully-adjustable candidate=clip7-fixed-factor"
    << " elements=" << comparedElements
    << " mismatches=0 no_neg128=1 canary=1 divide127_orthogonal=1"
    << " timed_path=0\n";
}

void checkContracts(
  Lane& lane,
  const Options& options,
  const HostData& host,
  const DeviceWeights& weights
) {
  if(launchRmsNormInt8(
       nullptr,lane.normInt8.data<int8_t>(),weights.gamma.data<half>(),
       kTokenRows,kRmsEpsilon,lane.stream) != cudaErrorInvalidValue ||
     launchRmsNormInt8(
       weights.input.data<half>(),nullptr,weights.gamma.data<half>(),
       kTokenRows,kRmsEpsilon,lane.stream) != cudaErrorInvalidValue ||
     launchRmsNormInt8(
       weights.input.data<half>(),lane.normInt8.data<int8_t>(),
       weights.gamma.data<half>(),0,kRmsEpsilon,lane.stream) !=
       cudaErrorInvalidValue ||
     launchRmsNormInt8(
       weights.input.data<half>(),lane.normInt8.data<int8_t>(),
       weights.gamma.data<half>(),kTokenRows,0.0f,lane.stream) !=
       cudaErrorInvalidValue)
    throw std::runtime_error("INT8-only RMS invalid contract did not fail closed");

  if(options.mode == ProjectionMode::AggressiveQkv &&
     (launchQuantizeAttentionOutput(
        nullptr,lane.attentionInt8.data<int8_t>(),kTokenRows,lane.stream) !=
        cudaErrorInvalidValue ||
      launchQuantizeAttentionOutput(
        weights.input.data<half>(),nullptr,kTokenRows,lane.stream) !=
        cudaErrorInvalidValue ||
      launchAttentionOutResidual(
        nullptr,kTokenRows,lane.attentionInt8.data<int8_t>(),
        weights.residual.data<half>(),lane.attentionOutput.data<half>(),
        lane.stream) != cudaErrorInvalidValue))
    throw std::runtime_error(
      "attention-out invalid contract did not fail closed");
  if(options.mode == ProjectionMode::AggressiveQkv)
    checkAttentionOutputQuantizerFixture(lane.stream);

  // Use the legacy dual-output kernel as the bit-exact oracle. The
  // INT8-only path receives no FP16 destination at all.
  checkCuda(launchRmsNormFp16Int8(
    weights.input.data<half>(),lane.normHalf.data<half>(),
    lane.normInt8Only.data<int8_t>(),weights.gamma.data<half>(),kTokenRows,
    kRmsEpsilon,lane.stream),"launch RMS bit-exact control");
  checkCuda(cudaStreamSynchronize(lane.stream),"RMS control sync");
  const std::vector<int8_t> normControl =
    lane.normInt8Only.download<int8_t>();

  enqueuePrototypeSubpath(lane,options,weights);
  checkCuda(cudaStreamSynchronize(lane.stream),"contract stream sync");

  const std::vector<half> normHalf = lane.normHalf.download<half>();
  const std::vector<int8_t> normInt8 = lane.normInt8.download<int8_t>();
  const std::vector<half> qkv = lane.rawPackedQkv.download<half>();
  const std::vector<half> productHalf =
    options.mode == ProjectionMode::ConservativeQk ?
      lane.productHalf.download<half>() : std::vector<half>();
  const std::vector<int8_t> productInt8 =
    options.mode == ProjectionMode::AggressiveQkv ?
      lane.productInt8.download<int8_t>() : std::vector<int8_t>();
  const std::vector<int8_t> attentionInt8 =
    options.mode == ProjectionMode::AggressiveQkv ?
      lane.attentionInt8.download<int8_t>() : std::vector<int8_t>();
  requireFinite(normHalf,"RMS FP16");
  if(options.mode == ProjectionMode::AggressiveQkv && normInt8 != normControl)
    throw std::runtime_error(
      "INT8-only RMS differs from FP16+INT8 control bytes");
  if(std::find(normInt8.begin(),normInt8.end(),int8_t(-128)) != normInt8.end())
    throw std::runtime_error("RMS emitted forbidden -128");
  requireFinite(qkv,"packed QKV");
  if(options.mode == ProjectionMode::ConservativeQk)
    requireFinite(productHalf,"clip7 dual product");

  for(std::size_t i = 0; i < normHalf.size(); i++) {
    const int expected = quantize(toFloat(normHalf[i]),kNormActivationClip);
    if(int(normInt8[i]) != expected || normInt8[i] == int8_t(-128))
      throw std::runtime_error("RMS INT8 relation mismatch at " +
        std::to_string(i));
  }

  constexpr std::array<int,4> sampleRows{{0,1,kTokenRows / 2,kTokenRows - 1}};
  constexpr std::array<int,8> projectionColumns{{0,1,31,383,384,767,768,1151}};
  for(int row: sampleRows) {
    double sumSquares = 0.0;
    for(int channel = 0; channel < kChannels; channel++) {
      const float value = toFloat(host.input[std::size_t(row) * kChannels + channel]);
      sumSquares += double(value) * value;
    }
    const float scale = 1.0f / std::sqrt(float(sumSquares / kChannels) + kRmsEpsilon);
    for(int channel: std::array<int,4>{{0,31,191,383}}) {
      const float input = toFloat(host.input[std::size_t(row) * kChannels + channel]);
      const float gamma = toFloat(host.gamma[channel]);
      requireNear(toFloat(normHalf[std::size_t(row) * kChannels + channel]),
        input * scale * gamma,0.0015f,"RMS oracle");
    }
    for(int channel: projectionColumns) {
      const bool produced = options.mode == ProjectionMode::AggressiveQkv ||
        channel < kQkChannels;
      const float actual = toFloat(qkv[std::size_t(row) * kQkvChannels + channel]);
      if(!produced) {
        if(actual != 0.0f)
          throw std::runtime_error("conservative QK overwrote V plane");
        continue;
      }
      const int32_t accum = dot(
        normInt8,row,kChannels,host.projectionWeights,channel);
      const float expected = float(accum) * kNormActivationScale *
        host.projectionWeightScale;
      requireNear(actual,expected,0.005f,"projection int32-dequant oracle");
    }
  }

  constexpr std::array<int,6> ffnColumns{{0,1,63,511,767,1023}};
  for(int row: sampleRows) {
    for(int channel: ffnColumns) {
      const int32_t upAccum = dot(normInt8,row,kChannels,host.upWeights,channel);
      const int32_t gateAccum = dot(normInt8,row,kChannels,host.gateWeights,channel);
      const float up = float(upAccum) * kNormActivationScale *
        host.upWeightScale;
      const float gate = float(gateAccum) * kNormActivationScale *
        host.gateWeightScale;
      const float silu = up / (1.0f + std::exp(-up));
      const float expected = std::max(-7.0f,std::min(7.0f,silu)) *
        std::max(-7.0f,std::min(7.0f,gate));
      const std::size_t index = std::size_t(row) * kFfnChannels + channel;
      if(options.mode == ProjectionMode::AggressiveQkv) {
        const int expectedInt8 = fusedClip7ProductOracle(up,gate);
        if(int(productInt8[index]) != expectedInt8 ||
           productInt8[index] == int8_t(-128))
          throw std::runtime_error("fused dual INT8 oracle mismatch");
      }
      else {
        const float actual = toFloat(productHalf[index]);
        requireNear(actual,expected,0.08f,"dual clip7 oracle");
      }
    }
  }

  if(options.mode == ProjectionMode::AggressiveQkv) {
    const std::vector<half> attentionOutput =
      lane.attentionOutput.download<half>();
    requireFinite(attentionOutput,"INT8 attention-out residual");
    if(std::find(attentionInt8.begin(),attentionInt8.end(),int8_t(-128)) !=
       attentionInt8.end())
      throw std::runtime_error("attention-output quantizer emitted forbidden -128");
    for(int row: sampleRows) {
      for(int inputChannel: std::array<int,5>{{0,1,127,255,383}}) {
        const std::size_t index =
          std::size_t(row) * kChannels + inputChannel;
        const int expected = quantize(
          toFloat(host.input[index]),kNormActivationClip);
        if(int(attentionInt8[index]) != expected)
          throw std::runtime_error(
            "attention-output clip4 quantization relation mismatch");
      }
      for(int channel: std::array<int,5>{{0,1,127,255,383}}) {
        const int32_t accum = dot(
          attentionInt8,row,kChannels,host.attentionOutWeights,channel);
        const float residual = toFloat(
          host.residual[std::size_t(row) * kChannels + channel]);
        const float expected = float(accum) * kNormActivationScale *
          host.attentionOutWeightScale + residual;
        const float actual = toFloat(
          attentionOutput[std::size_t(row) * kChannels + channel]);
        requireNear(actual,expected,0.01f,
          "attention-out int32-dequant beta1 oracle");
      }
    }

    const std::vector<half> downOutput = lane.downOutput.download<half>();
    requireFinite(downOutput,"INT8 down residual");
    if(std::find(productInt8.begin(),productInt8.end(),int8_t(-128)) !=
       productInt8.end())
      throw std::runtime_error("fused product emitted forbidden -128");
    for(int row: sampleRows) {
      for(int channel: std::array<int,5>{{0,1,127,255,383}}) {
        const int32_t accum = dot(
          productInt8,row,kFfnChannels,host.downWeights,channel);
        const float residual = toFloat(
          host.residual[std::size_t(row) * kChannels + channel]);
        const float expected = float(accum) * kClip7ProductScale *
          host.downWeightScale + residual;
        const float actual = toFloat(
          downOutput[std::size_t(row) * kChannels + channel]);
        requireNear(actual,expected,0.01f,"down int32-dequant beta1 oracle");
      }
    }
  }

  lane.normHalf.requireCanary("normHalf");
  lane.normInt8.requireCanary("normInt8");
  lane.normInt8Only.requireCanary("normInt8Only");
  lane.rawPackedQkv.requireCanary("rawPackedQkv");
  lane.productHalf.requireCanary("productHalf");
  lane.productInt8.requireCanary("productInt8");
  lane.downOutput.requireCanary("downOutput");
  lane.attentionInt8.requireCanary("attentionInt8");
  lane.attentionOutput.requireCanary("attentionOutput");
  weights.projection.requireCanary("projection weights");
  weights.up.requireCanary("up weights");
  weights.gate.requireCanary("gate weights");
  weights.down.requireCanary("down weights");
  weights.attentionOut.requireCanary("attention-out weights");
  weights.gamma.requireCanary("gamma");
  weights.input.requireCanary("input");
  weights.residual.requireCanary("residual");
  checkClip7SaturationContract(lane,options,normInt8);
  if(options.mode == ProjectionMode::AggressiveQkv &&
     std::strcmp(
       dualFfnProductQuantPath(lane.dual.get()),
       "clip-squared-exact-int-v1") != 0)
    throw std::runtime_error("default timed lane left exact clip7/49 path");
  checkAdjustableProductScaleContracts(lane,options);
}

// Real-kernel A/B for the optional exact divide-by-127 epilogue. The
// candidate is deliberately unavailable to every per-layer adjustable-scale
// configuration: only Int8 + D2 + clip7/productMax49 reaches it.
void checkDivide127GpuContract(
  Lane& lane,
  const Options& options,
  const HostData& host,
  const DeviceWeights& weights
) {
  if(options.divide127 != DualFfnDivide127Tactic::ExactBranchless)
    return;
  static_assert(kTokenRows == 6300,"production M contract changed");
  static_assert(kTokenRows % 128 == 28,"production D2 tail contract changed");

  DualFfnConfig referenceConfig;
  referenceConfig.tactic = DualFfnTactic::M128N64K64S3Sw4;
  referenceConfig.outputMode = DualFfnOutputMode::Int8Product;
  referenceConfig.divide127Tactic = DualFfnDivide127Tactic::Incumbent;
  referenceConfig.maxTokenRows = kTokenRows;
  referenceConfig.packedUpWeights = weights.up.data<int8_t>();
  referenceConfig.packedGateWeights = weights.gate.data<int8_t>();
  referenceConfig.upWeightScale = host.upWeightScale;
  referenceConfig.gateWeightScale = host.gateWeightScale;
  referenceConfig.swigluClip = 7.0f;
  referenceConfig.productQuantMaxAbs = 49.0f;
  DualFfnConfig candidateConfig = referenceConfig;
  candidateConfig.divide127Tactic =
    DualFfnDivide127Tactic::ExactBranchless;
  DualHandle reference{createDualFfn(referenceConfig)};
  DualHandle candidate{createDualFfn(candidateConfig)};
  if(reference == nullptr || candidate == nullptr)
    throw std::runtime_error("divide127 GPU A/B handle preparation failed");
  if(std::strcmp(
       dualFfnProductQuantPath(candidate.get()),
       "clip-squared-exact-int-v1") != 0)
    throw std::runtime_error("divide127 candidate left exact clip7/49 path");

  const auto requireCandidateRejected = [&](const DualFfnConfig& config,
                                             const char* drift) {
    DualHandle unexpected{createDualFfn(config)};
    if(unexpected != nullptr)
      throw std::runtime_error(
        std::string("exact-branchless divide127 accepted ") + drift);
  };
  DualFfnConfig drift = candidateConfig;
  drift.outputMode = DualFfnOutputMode::Fp16Product;
  requireCandidateRejected(drift,"FP16 output");
  drift = candidateConfig;
  drift.tactic = DualFfnTactic::M128N64K64S3Sw1;
  requireCandidateRejected(drift,"non-D2 tactic");
  drift = candidateConfig;
  drift.swigluClip = 6.0f;
  requireCandidateRejected(drift,"dynamic clip");
  drift = candidateConfig;
  drift.productQuantMaxAbs = 73.4171f;
  requireCandidateRejected(drift,"dynamic productMax");

  // Prove that rejecting the candidate does not reject the dynamic-scale
  // implementation itself; fixed clip7 must use the orthogonal hybrid path.
  DualFfnConfig dynamicConfig = candidateConfig;
  dynamicConfig.divide127Tactic = DualFfnDivide127Tactic::Incumbent;
  dynamicConfig.productQuantMaxAbs = 73.4171f;
  DualHandle dynamic{createDualFfn(dynamicConfig)};
  if(dynamic == nullptr ||
     std::strcmp(
       dualFfnProductQuantPath(dynamic.get()),
       "clip7-fixed-factor-v105-product-float-rne-v1") != 0)
    throw std::runtime_error(
      "incumbent dynamic scale did not select clip7 hybrid path");

  constexpr std::array<int,3> rowCases{{1,kTokenRows % 128,kTokenRows}};
  std::size_t totalElements = 0;
  for(const int rows: rowCases) {
    const std::size_t activationElements =
      std::size_t(rows) * kChannels;
    const std::size_t outputElements =
      std::size_t(rows) * kFfnChannels;
    std::vector<int8_t> activation(activationElements);
    for(int row = 0; row < rows; row++) {
      for(int channel = 0; channel < kChannels; channel++) {
        const int code =
          (row * 47 + channel * 31 + row * channel * 3 + 19) % 255 - 127;
        activation[std::size_t(row) * kChannels + channel] =
          static_cast<int8_t>(code);
      }
    }
    GuardedBuffer activationDevice(activationElements);
    GuardedBuffer referenceOutput(outputElements);
    GuardedBuffer candidateOutput(outputElements);
    activationDevice.upload(activation);
    referenceOutput.zeroPayload();
    candidateOutput.zeroPayload();
    checkCuda(launchDualFfnInt8(
      reference.get(),rows,activationDevice.data<int8_t>(),
      referenceOutput.data<int8_t>(),lane.stream),
      "launch incumbent divide127 GPU contract");
    checkCuda(launchDualFfnInt8(
      candidate.get(),rows,activationDevice.data<int8_t>(),
      candidateOutput.data<int8_t>(),lane.stream),
      "launch exact-branchless divide127 GPU contract");
    checkCuda(cudaStreamSynchronize(lane.stream),
      "divide127 GPU A/B contract sync");
    const std::vector<int8_t> incumbent =
      referenceOutput.download<int8_t>();
    const std::vector<int8_t> exact = candidateOutput.download<int8_t>();
    if(exact != incumbent)
      throw std::runtime_error(
        "exact-branchless divide127 GPU output is not bit-exact");
    if(std::find(incumbent.begin(),incumbent.end(),int8_t(-128)) !=
         incumbent.end() ||
       std::find(exact.begin(),exact.end(),int8_t(-128)) != exact.end())
      throw std::runtime_error("divide127 GPU A/B emitted forbidden -128");
    activationDevice.requireCanary("divide127 GPU contract activation");
    referenceOutput.requireCanary("divide127 GPU contract incumbent output");
    candidateOutput.requireCanary("divide127 GPU contract candidate output");
    totalElements += outputElements;
  }
  weights.up.requireCanary("divide127 GPU contract up weights");
  weights.gate.requireCanary("divide127 GPU contract gate weights");
  std::cout
    << "KATAGO_C384_INT8_DIV127_GPU_CONTRACT_PASS"
    << " reference=incumbent candidate=exact-branchless"
    << " geometry=D2 M=1,28,6300 production_M6300_tail28=1"
    << " strict_gate=int8,d2,clip7,product49"
    << " dynamic_scale_candidate_rejected=1"
    << " dynamic_scale_incumbent_clip7_hybrid=1"
    << " elements=" << totalElements
    << " mismatches=0 no_neg128=1 canary=1 timed_path=0\n";
}

void checkFusedProjectionQknormRopeContract(
  Lane& lane,
  const Options& options
) {
  if(options.mode != ProjectionMode::AggressiveQkv ||
     options.projection != ProjectionTactic::M128N128K64S3Sw2)
    return;

  if(!projectionQknormRopeSupports(
       lane.projection.get(),kTokenRows,kChannels,kQkvChannels,
       kRmsEpsilon,kRmsEpsilon))
    throw std::runtime_error("fused QKV+QKNorm+RoPE production gate rejected P3");
  if(projectionQknormRopeSupports(
       lane.projection.get(),kTokenRows - 1,kChannels,kQkvChannels,
       kRmsEpsilon,kRmsEpsilon) ||
     projectionQknormRopeSupports(
       lane.projection.get(),kTokenRows,kChannels,kQkvChannels,
       1.0e-5f,kRmsEpsilon))
    throw std::runtime_error("fused QKV+QKNorm+RoPE exact gate accepted drift");

  const std::size_t qkvElements =
    std::size_t(kTokenRows) * kQkvChannels;
  GuardedBuffer reference(qkvElements * sizeof(half));
  GuardedBuffer candidate(qkvElements * sizeof(half));
  std::vector<half> qGamma(kHeadDim);
  std::vector<half> kGamma(kHeadDim);
  std::vector<half2> rope(
    std::size_t(kSequence) * kHeads * kRopePairsPerHead);
  for(int d = 0; d < kHeadDim; d++) {
    qGamma[d] = toHalf(0.61f + 0.017f * float(d));
    kGamma[d] = toHalf(1.39f - 0.011f * float(d));
  }
  for(int xy = 0; xy < kSequence; xy++) {
    for(int head = 0; head < kHeads; head++) {
      for(int pair = 0; pair < kRopePairsPerHead; pair++) {
        const float angle = 0.00071f * float(1 + xy + 3 * head + 5 * pair);
        const std::size_t index =
          (std::size_t(xy) * kHeads + head) * kRopePairsPerHead + pair;
        rope[index] = __floats2half2_rn(std::cos(angle),std::sin(angle));
      }
    }
  }
  GuardedBuffer qGammaDevice(qGamma.size() * sizeof(half));
  GuardedBuffer kGammaDevice(kGamma.size() * sizeof(half));
  GuardedBuffer ropeDevice(rope.size() * sizeof(half2));
  qGammaDevice.upload(qGamma);
  kGammaDevice.upload(kGamma);
  ropeDevice.upload(rope);
  reference.zeroPayload();
  candidate.zeroPayload();

  // Reference and candidate receive the same prepared P3 handle, input bytes,
  // weights/scales, output layout, gamma and RoPE table on one nonblocking
  // stream. The only arithmetic difference is standalone versus epilogue QKN.
  checkCuda(launchProjection(
    lane.projection.get(),kTokenRows,lane.normInt8.data<int8_t>(),
    reference.data<half>(),kQkvChannels,lane.stream),
    "launch reference P3 projection");
  C384QKNormRopeSm120::LaunchParams params;
  params.abiVersion = C384QKNormRopeSm120::kAbiVersion;
  params.batch = kBatch;
  params.sequence = kSequence;
  params.heads = kHeads;
  params.kvHeads = kHeads;
  params.headDim = kHeadDim;
  params.tokenRows = kTokenRows;
  params.deviceOrdinal = 0;
  params.computeCapability = 120;
  params.usingFp16 = true;
  params.usingNhwc = true;
  params.learnedRope = true;
  params.qkNorm = true;
  params.inputSemantic =
    C384QKNormRopeSm120::InputSemantic::RawPackedQkv;
  params.qEpsilon = kRmsEpsilon;
  params.kEpsilon = kRmsEpsilon;
  checkCuda(C384QKNormRopeSm120::launchInPlace(
    params,reference.data<half>(),qGammaDevice.data<half>(),
    kGammaDevice.data<half>(),ropeDevice.data<half2>(),lane.stream),
    "launch reference QKNorm+RoPE");
  checkCuda(launchProjectionQknormRope(
    lane.projection.get(),kTokenRows,lane.normInt8.data<int8_t>(),
    candidate.data<half>(),kQkvChannels,qGammaDevice.data<half>(),
    kGammaDevice.data<half>(),ropeDevice.data<half2>(),kRmsEpsilon,
    kRmsEpsilon,lane.stream),"launch fused P3 QKV+QKNorm+RoPE");
  checkCuda(cudaStreamSynchronize(lane.stream),
    "fused QKNorm comparison sync");

  const std::vector<half> referenceHost = reference.download<half>();
  const std::vector<half> candidateHost = candidate.download<half>();
  std::size_t qkBitMismatch = 0;
  std::size_t vBitMismatch = 0;
  std::size_t tailQkBitMismatch = 0;
  double maxAbs = 0.0;
  double tailMaxAbs = 0.0;
  constexpr int tailRows = kTokenRows % 128;
  static_assert(tailRows == 28,"P3 M-tail contract changed");
  for(int row = 0; row < kTokenRows; row++) {
    for(int channel = 0; channel < kQkvChannels; channel++) {
      const std::size_t index = std::size_t(row) * kQkvChannels + channel;
      uint16_t referenceBits = 0;
      uint16_t candidateBits = 0;
      std::memcpy(&referenceBits,&referenceHost[index],sizeof(referenceBits));
      std::memcpy(&candidateBits,&candidateHost[index],sizeof(candidateBits));
      if(channel >= kQkChannels) {
        if(referenceBits != candidateBits)
          vBitMismatch++;
        continue;
      }
      const float referenceValue = toFloat(referenceHost[index]);
      const float candidateValue = toFloat(candidateHost[index]);
      if(!std::isfinite(referenceValue) || !std::isfinite(candidateValue))
        throw std::runtime_error("fused QKNorm comparison produced non-finite Q/K");
      if(referenceBits != candidateBits) {
        qkBitMismatch++;
        if(row >= kTokenRows - tailRows)
          tailQkBitMismatch++;
      }
      const double absError = std::fabs(
        double(referenceValue) - double(candidateValue));
      maxAbs = std::max(maxAbs,absError);
      if(row >= kTokenRows - tailRows)
        tailMaxAbs = std::max(tailMaxAbs,absError);
    }
  }
  if(vBitMismatch != 0)
    throw std::runtime_error("fused projection changed V bytes");
  // The standalone half2 kernel and fused four-lane epilogue have different
  // FP32 sum-of-squares trees. Their allowed diagnostic delta is therefore
  // one small FP16 neighborhood, while all semantic rounding boundaries are
  // preserved. A larger delta is a lane-map, stride, or RoPE-index failure.
  if(maxAbs > 0.004 || tailMaxAbs > 0.004)
    throw std::runtime_error("fused Q/K differs from standalone beyond 0.004");
  reference.requireCanary("reference QKV+QKNorm+RoPE");
  candidate.requireCanary("fused QKV+QKNorm+RoPE");
  qGammaDevice.requireCanary("fused Q gamma");
  kGammaDevice.requireCanary("fused K gamma");
  ropeDevice.requireCanary("fused learned RoPE table");
  lane.normInt8.requireCanary("fused QKNorm input");
  std::cout << std::setprecision(9)
    << "KATAGO_C384_INT8_FUSED_QKNORM_ROPE_CONTRACT_PASS"
    << " stream=nonblocking"
    << " rows=" << kTokenRows
    << " tail_rows=" << tailRows
    << " v_bit_mismatch=" << vBitMismatch
    << " qk_bit_mismatch=" << qkBitMismatch
    << " qk_max_abs=" << maxAbs
    << " tail_qk_bit_mismatch=" << tailQkBitMismatch
    << " tail_qk_max_abs=" << tailMaxAbs
    << " canary=1 marker=" << projectionQknormRopeMarker() << '\n';
}

float benchmarkFamily(
  std::vector<std::unique_ptr<Lane>>& lanes,
  TimingFamily family,
  const Options& options,
  const DeviceWeights& weights
) {
  if(family == TimingFamily::AttentionOutResidual) {
    for(auto& lane: lanes) {
      checkCuda(launchQuantizeAttentionOutput(
        weights.input.data<half>(),lane->attentionInt8.data<int8_t>(),
        kTokenRows,lane->stream),"seed attention-output quantization");
      checkCuda(cudaStreamSynchronize(lane->stream),
        "seed attention-output quantization sync");
    }
  }
  if(family == TimingFamily::ProductQuantization) {
    // Seed the legacy FP16 product outside timing. This family measures only
    // the removed standalone conversion and is not part of the connected path.
    for(auto& lane: lanes) {
      checkCuda(launchDualFfnHalf(
        lane->legacyHalfDual.get(),kTokenRows,lane->normInt8.data<int8_t>(),
        lane->productHalf.data<half>(),lane->stream),
        "seed legacy product-quant control");
      checkCuda(cudaStreamSynchronize(lane->stream),
        "seed legacy product-quant control sync");
    }
  }
  for(int warmup = 0; warmup < options.warmup; warmup++)
    for(auto& lane: lanes)
      enqueueFamily(*lane,family,options,weights);
  for(auto& lane: lanes)
    checkCuda(cudaStreamSynchronize(lane->stream),"warmup sync");

  cudaStream_t control = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  checkCuda(cudaStreamCreateWithFlags(&control,cudaStreamNonBlocking),
    "create timing control stream");
  checkCuda(cudaEventCreate(&start),"create start event");
  checkCuda(cudaEventCreate(&stop),"create stop event");
  std::vector<cudaEvent_t> done(lanes.size(),nullptr);
  try {
    for(cudaEvent_t& event: done)
      checkCuda(cudaEventCreateWithFlags(&event,cudaEventDisableTiming),
        "create done event");
    checkCuda(cudaEventRecord(start,control),"record start");
    for(auto& lane: lanes)
      checkCuda(cudaStreamWaitEvent(lane->stream,start,0),"lane waits for start");
    for(int iteration = 0; iteration < options.iterations; iteration++)
      for(auto& lane: lanes)
        enqueueFamily(*lane,family,options,weights);
    for(std::size_t lane = 0; lane < lanes.size(); lane++)
      checkCuda(cudaEventRecord(done[lane],lanes[lane]->stream),"record lane done");
    for(cudaEvent_t event: done)
      checkCuda(cudaStreamWaitEvent(control,event,0),"control waits lane done");
    checkCuda(cudaEventRecord(stop,control),"record stop");
    checkCuda(cudaEventSynchronize(stop),"timing stop sync");
    float elapsedMs = 0.0f;
    checkCuda(cudaEventElapsedTime(&elapsedMs,start,stop),"elapsed time");
    for(cudaEvent_t event: done)
      (void)cudaEventDestroy(event);
    (void)cudaEventDestroy(stop);
    (void)cudaEventDestroy(start);
    (void)cudaStreamDestroy(control);
    return elapsedMs;
  }
  catch(...) {
    for(cudaEvent_t event: done)
      if(event != nullptr) (void)cudaEventDestroy(event);
    if(stop != nullptr) (void)cudaEventDestroy(stop);
    if(start != nullptr) (void)cudaEventDestroy(start);
    if(control != nullptr) (void)cudaStreamDestroy(control);
    throw;
  }
}

struct FamilyMeasurement {
  TimingFamily family;
  float elapsedMs;
};

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parseOptions(argc,argv);
    int device = -1;
    cudaDeviceProp prop{};
    checkCuda(cudaGetDevice(&device),"cudaGetDevice");
    checkCuda(cudaGetDeviceProperties(&prop,device),"cudaGetDeviceProperties");
    if(prop.major != 12 || prop.minor != 0)
      throw std::runtime_error("microbenchmark requires exact SM120");

    const HostData host = makeHostData(options.mode);
    const DeviceWeights weights(host);
    std::vector<std::unique_ptr<Lane>> lanes;
    for(int i = 0; i < options.streams; i++) {
      auto lane = std::make_unique<Lane>();
      prepareLane(*lane,options,host,weights);
      lanes.push_back(std::move(lane));
    }

    for(auto& lane: lanes)
      checkContracts(*lane,options,host,weights);
    checkClip7HybridGpuContract(*lanes.front(),options,host,weights);
    checkDivide127GpuContract(*lanes.front(),options,host,weights);
    for(auto& lane: lanes)
      checkFusedProjectionQknormRopeContract(*lane,options);
    std::cout
      << "KATAGO_C384_INT8_MICROBENCH_SCOPE"
      << " scope=component_and_prototype_subpath_latency_only"
      << " whole_network_throughput_claim=0"
      << " omitted=" << omittedFp16AndAttention(options.mode) << '\n'
      << "KATAGO_C384_INT8_MICROBENCH_CONTRACT_PASS variant="
      << (options.mode == ProjectionMode::ConservativeQk ?
        "conservative" : "aggressive")
      << " M=" << kTokenRows
      << " projection=" << projectionTacticName(options.projection)
      << " dual=" << dualFfnTacticName(options.dual)
      << " divide127=" << divide127TacticName(options.divide127)
      << " down=" << (options.mode == ProjectionMode::AggressiveQkv ?
        downTacticName(options.down) : "none")
      << " attention_out=" << (options.mode == ProjectionMode::AggressiveQkv ?
        attentionOutTacticName(options.attentionOut) : "none")
      << " finite=1 canary=1 int32_dequant=1"
      << " attention_out_clip4_quant=1 attention_out_beta1=1"
      << " attention_out_endpoints_plus4_minus4=1"
      << " attention_out_saturation=1 attention_out_rne_tie=1"
      << " attention_out_no_neg128=1 attention_out_tail=1"
      << " clip7_silu_positive=1 clip7_gate_positive=1"
      << " clip7_gate_negative=1 clip7_silu_negative_unreachable=1"
      << " product_endpoints_plus49_minus49=1 missing_clamp_witness=1"
      << " product_rne_saturate_no_neg128=1"
      << " rms_int8_only_bitexact=1 rms_fp16_unmaterialized=1\n";
    if(options.contractsOnly)
      return 0;

    const std::array<TimingFamily,9> requestedFamilies{{
      TimingFamily::RmsInt8Only,
      TimingFamily::RmsFp16Int8Control,
      TimingFamily::Projection,
      TimingFamily::DualFfn,
      TimingFamily::ProductQuantization,
      TimingFamily::DownResidual,
      TimingFamily::AttentionOutputQuantization,
      TimingFamily::AttentionOutResidual,
      TimingFamily::PrototypeSubpath,
    }};
    std::vector<FamilyMeasurement> measurements;
    for(const TimingFamily family: requestedFamilies) {
      const bool aggressiveOnly =
        family == TimingFamily::ProductQuantization ||
        family == TimingFamily::DownResidual ||
        family == TimingFamily::AttentionOutputQuantization ||
        family == TimingFamily::AttentionOutResidual;
      if(aggressiveOnly && options.mode != ProjectionMode::AggressiveQkv)
        continue;
      const float elapsedMs = benchmarkFamily(
        lanes,family,options,weights);
      measurements.push_back({family,elapsedMs});
      const double commonWallUs =
        double(elapsedMs) * 1000.0 / options.iterations;
      const double effectiveLaneUs = commonWallUs / options.streams;
      std::cout << std::fixed << std::setprecision(6)
        << "KATAGO_C384_INT8_MICROBENCH_TIMING"
        << " family=" << timingFamilyName(family)
        << " streams=" << options.streams
        << " common_wall_us_per_iteration=" << commonWallUs
        << " effective_us_per_lane=" << effectiveLaneUs << '\n';
    }
    const auto endMeasurement = std::find_if(
      measurements.begin(),measurements.end(),[](const FamilyMeasurement& item) {
        return item.family == TimingFamily::PrototypeSubpath;
      });
    if(endMeasurement == measurements.end())
      throw std::runtime_error("missing prototype-subpath measurement");
    const double sequences = double(options.streams) * options.iterations;
    const double prototypeSubpathEffectiveUs =
      double(endMeasurement->elapsedMs) * 1000.0 / sequences;
    std::cout << std::fixed << std::setprecision(6)
      << "{\"marker\":\"KATAGO_C384_INT8_MICROBENCH_RESULT\""
      << ",\"variant\":\""
      << (options.mode == ProjectionMode::ConservativeQk ?
        "conservative" : "aggressive") << "\""
      << ",\"M\":" << kTokenRows
      << ",\"streams\":" << options.streams
      << ",\"warmup\":" << options.warmup
      << ",\"iterations\":" << options.iterations
      << ",\"projection\":\"" << projectionTacticName(options.projection)
      << "\",\"dual\":\"" << dualFfnTacticName(options.dual)
      << "\",\"divide127\":\"" << divide127TacticName(options.divide127)
      << "\",\"down\":\""
      << (options.mode == ProjectionMode::AggressiveQkv ?
        downTacticName(options.down) : "none") << "\""
      << ",\"product_quantization\":\""
      << (options.mode == ProjectionMode::AggressiveQkv ?
        "fused-dual-epilogue-v2" : "none") << "\""
      << ",\"legacy_product_quantization_control\":\""
      << (options.mode == ProjectionMode::AggressiveQkv ?
        "timed-separately-not-connected" : "none") << "\""
      << ",\"scope\":\"component_and_prototype_subpath_latency_only\""
      << ",\"whole_network_throughput_claim\":false"
      << ",\"omitted_fp16_and_attention\":\""
      << omittedFp16AndAttention(options.mode) << "\""
      << ",\"timings_us\":{";
    for(std::size_t i = 0; i < measurements.size(); i++) {
      if(i > 0)
        std::cout << ',';
      const double commonWallUs =
        double(measurements[i].elapsedMs) * 1000.0 / options.iterations;
      const double effectiveLaneUs = commonWallUs / options.streams;
      std::cout << "\"" << timingFamilyName(measurements[i].family) << "\""
        << ":{\"common_wall_per_iteration\":" << commonWallUs
        << ",\"effective_per_lane\":" << effectiveLaneUs << '}';
    }
    std::cout << '}'
      << ",\"prototype_subpath_elapsed_ms\":" << endMeasurement->elapsedMs
      << ",\"prototype_subpath_effective_us_per_lane\":"
      << prototypeSubpathEffectiveUs
      << "}\n";
    return 0;
  }
  catch(const std::exception& error) {
    std::cerr << "KATAGO_C384_INT8_MICROBENCH_FAIL " << error.what() << "\n";
    return 1;
  }
}

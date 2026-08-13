#include "neuralnet/cuda_specialized/sm120/c384/fixed_batch/qknorm_rope.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

using C384QKNormRopeSm120::InputSemantic;
using C384QKNormRopeSm120::LaunchParams;

LaunchParams validParams() {
  LaunchParams params;
  params.abiVersion = C384QKNormRopeSm120::kAbiVersion;
  params.batch = C384QKNormRopeSm120::kBatch;
  params.sequence = C384QKNormRopeSm120::kSequence;
  params.heads = C384QKNormRopeSm120::kHeads;
  params.kvHeads = C384QKNormRopeSm120::kHeads;
  params.headDim = C384QKNormRopeSm120::kHeadDim;
  params.tokenRows = params.batch * params.sequence;
  params.deviceOrdinal = 0;
  params.computeCapability = 120;
  params.usingFp16 = true;
  params.usingNhwc = true;
  params.learnedRope = true;
  params.qkNorm = true;
  params.inputSemantic = InputSemantic::RawPackedQkv;
  params.qEpsilon = C384QKNormRopeSm120::kRmsEpsilon;
  params.kEpsilon = C384QKNormRopeSm120::kRmsEpsilon;
  return params;
}

bool check(bool condition, const char* message) {
  if(condition)
    return true;
  std::cerr << message << std::endl;
  return false;
}

uint16_t halfBits(half value) {
  uint16_t bits;
  static_assert(sizeof(bits) == sizeof(value),"half must be 16 bits");
  std::memcpy(&bits,&value,sizeof(bits));
  return bits;
}

half makeHalf(float value) {
  return __float2half_rn(value);
}

bool runGpuOracle(cudaStream_t stream, const char* streamName) {
  constexpr int guardElements = 256;
  constexpr int rows = C384QKNormRopeSm120::kBatch *
    C384QKNormRopeSm120::kSequence;
  constexpr int rowStride = C384QKNormRopeSm120::kPackedChannels;
  const size_t payloadElements = static_cast<size_t>(rows) * rowStride;
  const size_t allocationElements = payloadElements + 2 * guardElements;
  const half guardValue = makeHalf(-3.25f);

  std::vector<half> input(allocationElements,guardValue);
  std::vector<half> qGamma(C384QKNormRopeSm120::kHeadDim);
  std::vector<half> kGamma(C384QKNormRopeSm120::kHeadDim);
  std::vector<half2> rope(
    static_cast<size_t>(C384QKNormRopeSm120::kSequence) *
    C384QKNormRopeSm120::kHeads *
    C384QKNormRopeSm120::kRopePairsPerHead);

  for(int d = 0; d < C384QKNormRopeSm120::kHeadDim; d++) {
    qGamma[d] = makeHalf(0.55f + 0.019f * static_cast<float>(d));
    kGamma[d] = makeHalf(1.45f - 0.013f * static_cast<float>(d));
  }
  for(int xy = 0; xy < C384QKNormRopeSm120::kSequence; xy++) {
    for(int head = 0; head < C384QKNormRopeSm120::kHeads; head++) {
      for(int pair = 0; pair < C384QKNormRopeSm120::kRopePairsPerHead; pair++) {
        const float angle = 0.00073f * static_cast<float>(
          1 + xy + 3 * head + 5 * pair);
        const size_t index =
          (static_cast<size_t>(xy) * C384QKNormRopeSm120::kHeads + head) *
          C384QKNormRopeSm120::kRopePairsPerHead + pair;
        rope[index] = __floats2half2_rn(std::cos(angle),std::sin(angle));
      }
    }
  }

  half* const payload = input.data() + guardElements;
  for(int row = 0; row < rows; row++) {
    for(int channel = 0; channel < rowStride; channel++) {
      float value;
      if(channel >= 2 * C384QKNormRopeSm120::kChannels) {
        value = 0.003f * static_cast<float>(
          ((row * 19 + channel * 7) % 401) - 200);
      }
      else {
        const int planeChannel = channel % C384QKNormRopeSm120::kChannels;
        const int head = planeChannel / C384QKNormRopeSm120::kHeadDim;
        const int d = planeChannel % C384QKNormRopeSm120::kHeadDim;
        if(row == 0 && head == 0)
          value = 0.0f;
        else if(row == 0 && head == 1)
          value = (d & 1) == 0 ? 1.0e-4f : -1.0e-4f;
        else if(row == 0 && head == 2)
          value = (d & 1) == 0 ? 8.0f : -7.5f;
        else
          value = 0.021f * static_cast<float>(
            ((row * 13 + channel * 11) % 257) - 128);
      }
      payload[static_cast<size_t>(row) * rowStride + channel] = makeHalf(value);
    }
  }
  const std::vector<half> original = input;

  half* deviceAllocation = nullptr;
  half* deviceQGamma = nullptr;
  half* deviceKGamma = nullptr;
  half2* deviceRope = nullptr;
  bool ok = true;
  ok &= check(cudaMalloc(&deviceAllocation,allocationElements * sizeof(half)) ==
      cudaSuccess,"cudaMalloc packed QKV failed");
  ok &= check(cudaMalloc(&deviceQGamma,qGamma.size() * sizeof(half)) ==
      cudaSuccess,"cudaMalloc Q gamma failed");
  ok &= check(cudaMalloc(&deviceKGamma,kGamma.size() * sizeof(half)) ==
      cudaSuccess,"cudaMalloc K gamma failed");
  ok &= check(cudaMalloc(&deviceRope,rope.size() * sizeof(half2)) ==
      cudaSuccess,"cudaMalloc RoPE failed");
  if(!ok)
    goto cleanup;

  ok &= check(cudaMemcpyAsync(deviceAllocation,input.data(),
      allocationElements * sizeof(half),cudaMemcpyHostToDevice,stream) ==
      cudaSuccess,"copy packed QKV failed");
  ok &= check(cudaMemcpyAsync(deviceQGamma,qGamma.data(),
      qGamma.size() * sizeof(half),cudaMemcpyHostToDevice,stream) ==
      cudaSuccess,"copy Q gamma failed");
  ok &= check(cudaMemcpyAsync(deviceKGamma,kGamma.data(),
      kGamma.size() * sizeof(half),cudaMemcpyHostToDevice,stream) ==
      cudaSuccess,"copy K gamma failed");
  ok &= check(cudaMemcpyAsync(deviceRope,rope.data(),
      rope.size() * sizeof(half2),cudaMemcpyHostToDevice,stream) ==
      cudaSuccess,"copy RoPE failed");
  if(!ok)
    goto cleanup;

  {
    LaunchParams params = validParams();
    const cudaError_t launchStatus = C384QKNormRopeSm120::launchInPlace(
      params,deviceAllocation + guardElements,deviceQGamma,deviceKGamma,
      deviceRope,stream);
    ok &= check(launchStatus == cudaSuccess,"QKNorm+RoPE launch failed");
  }
  ok &= check(cudaMemcpyAsync(input.data(),deviceAllocation,
      allocationElements * sizeof(half),cudaMemcpyDeviceToHost,stream) ==
      cudaSuccess,"copy result failed");
  ok &= check(cudaStreamSynchronize(stream) == cudaSuccess,
    "QKNorm+RoPE synchronization failed");
  if(!ok)
    goto cleanup;

  for(int i = 0; i < guardElements; i++) {
    ok &= check(halfBits(input[i]) == halfBits(guardValue),
      "prefix redzone was modified");
    ok &= check(halfBits(input[guardElements + payloadElements + i]) ==
      halfBits(guardValue),"suffix redzone was modified");
  }

  {
    double maxAbs = 0.0;
    size_t checked = 0;
    for(int row = 0; row < rows; row++) {
      const int xy = row % C384QKNormRopeSm120::kSequence;
      for(int plane = 0; plane < 2; plane++) {
        const std::vector<half>& gamma = plane == 0 ? qGamma : kGamma;
        for(int head = 0; head < C384QKNormRopeSm120::kHeads; head++) {
          float sumSquares = 0.0f;
          const size_t headBase = static_cast<size_t>(guardElements) +
            static_cast<size_t>(row) * rowStride +
            plane * C384QKNormRopeSm120::kChannels +
            head * C384QKNormRopeSm120::kHeadDim;
          for(int d = 0; d < C384QKNormRopeSm120::kHeadDim; d++) {
            const float raw = __half2float(original[headBase + d]);
            sumSquares += raw * raw;
          }
          const float invRms = 1.0f / std::sqrt(
            sumSquares / static_cast<float>(C384QKNormRopeSm120::kHeadDim) +
            C384QKNormRopeSm120::kRmsEpsilon);
          for(int pair = 0; pair < C384QKNormRopeSm120::kRopePairsPerHead;
              pair++) {
            const int d0 = 2 * pair;
            const int d1 = d0 + 1;
            const float raw0 = __half2float(original[headBase + d0]);
            const float raw1 = __half2float(original[headBase + d1]);
            const float norm0 = __half2float(makeHalf(
              raw0 * invRms * __half2float(gamma[d0])));
            const float norm1 = __half2float(makeHalf(
              raw1 * invRms * __half2float(gamma[d1])));
            const size_t ropeIndex =
              (static_cast<size_t>(xy) * C384QKNormRopeSm120::kHeads + head) *
              C384QKNormRopeSm120::kRopePairsPerHead + pair;
            const float2 cs = __half22float2(rope[ropeIndex]);
            const half expected0 = makeHalf(norm0 * cs.x - norm1 * cs.y);
            const half expected1 = makeHalf(norm0 * cs.y + norm1 * cs.x);
            const float actual0 = __half2float(input[headBase + d0]);
            const float actual1 = __half2float(input[headBase + d1]);
            ok &= check(std::isfinite(actual0) && std::isfinite(actual1),
              "Q/K output is non-finite");
            maxAbs = std::max(maxAbs,std::fabs(
              static_cast<double>(actual0 - __half2float(expected0))));
            maxAbs = std::max(maxAbs,std::fabs(
              static_cast<double>(actual1 - __half2float(expected1))));
            checked += 2;
          }
        }
      }
      const size_t valueBase = static_cast<size_t>(guardElements) +
        static_cast<size_t>(row) * rowStride +
        2 * C384QKNormRopeSm120::kChannels;
      for(int d = 0; d < C384QKNormRopeSm120::kChannels; d++)
        ok &= check(halfBits(input[valueBase + d]) ==
          halfBits(original[valueBase + d]),"V plane was modified");
    }
    ok &= check(checked == static_cast<size_t>(rows) * 2 *
      C384QKNormRopeSm120::kChannels,"oracle element count drifted");
    ok &= check(maxAbs <= 0.004,"QKNorm+RoPE formula error exceeded tolerance");
    std::cout << "GPU_ORACLE stream=" << streamName << " count=" << checked
              << " max_abs=" << maxAbs << std::endl;
  }

cleanup:
  if(deviceRope != nullptr)
    cudaFree(deviceRope);
  if(deviceKGamma != nullptr)
    cudaFree(deviceKGamma);
  if(deviceQGamma != nullptr)
    cudaFree(deviceQGamma);
  if(deviceAllocation != nullptr)
    cudaFree(deviceAllocation);
  return ok;
}

}  // namespace

int main() {
  bool ok = true;
  const LaunchParams valid = validParams();
  ok &= check(C384QKNormRopeSm120::supports(valid),
    "valid exact QKNorm+RoPE shape was rejected");
  ok &= check(std::strcmp(C384QKNormRopeSm120::marker(),
      "c384-h12-d32-raw-packed-qknorm-rope-half2-g340-v1") == 0,
    "active marker drifted");

  LaunchParams changed = valid;
  changed.inputSemantic = static_cast<InputSemantic>(2);
  ok &= check(!C384QKNormRopeSm120::supports(changed),
    "non-raw QKV semantic was accepted");
  changed = valid;
  changed.batch = 24;
  changed.tokenRows = changed.batch * changed.sequence;
  ok &= check(!C384QKNormRopeSm120::supports(changed),
    "non-production batch was accepted");
  changed = valid;
  changed.qkNorm = false;
  ok &= check(!C384QKNormRopeSm120::supports(changed),
    "ordinary QKV was accepted by QKNorm kernel");
  changed = valid;
  changed.qEpsilon = 1.0e-5f;
  ok &= check(!C384QKNormRopeSm120::supports(changed),
    "wrong Q RMS epsilon was accepted");
  changed = valid;
  changed.kEpsilon = 1.0e-5f;
  ok &= check(!C384QKNormRopeSm120::supports(changed),
    "wrong K RMS epsilon was accepted");
  changed = valid;
  changed.tokenRows--;
  ok &= check(!C384QKNormRopeSm120::supports(changed),
    "partial packed QKV matrix was accepted");

  ok &= check(C384QKNormRopeSm120::launchInPlace(
      valid,nullptr,nullptr,nullptr,nullptr,nullptr) == cudaErrorInvalidValue,
    "valid shape with null pointers did not fail as invalid-value");
  changed = valid;
  changed.learnedRope = false;
  ok &= check(C384QKNormRopeSm120::launchInPlace(
      changed,nullptr,nullptr,nullptr,nullptr,nullptr) == cudaErrorNotSupported,
    "semantic mismatch did not fail before pointer validation");

  int deviceCount = 0;
  if(cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0) {
    cudaDeviceProp props;
    ok &= check(cudaGetDeviceProperties(&props,0) == cudaSuccess,
      "cudaGetDeviceProperties failed");
    if(ok && props.major * 10 + props.minor == 120) {
      ok &= check(cudaSetDevice(0) == cudaSuccess,"cudaSetDevice failed");
      ok &= runGpuOracle(nullptr,"default");
      cudaStream_t stream = nullptr;
      ok &= check(cudaStreamCreateWithFlags(
          &stream,cudaStreamNonBlocking) == cudaSuccess,
        "nonblocking stream creation failed");
      if(stream != nullptr) {
        ok &= runGpuOracle(stream,"nonblocking");
        ok &= check(cudaStreamDestroy(stream) == cudaSuccess,
          "nonblocking stream destruction failed");
      }
    }
    else
      std::cout << "GPU_ORACLE_SKIPPED reason=requires-sm120" << std::endl;
  }
  else {
    (void)cudaGetLastError();
    std::cout << "GPU_ORACLE_SKIPPED reason=no-cuda-device" << std::endl;
  }

  return ok ? 0 : 1;
}

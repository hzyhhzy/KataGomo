#include "../../shared/rms_norm.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

bool check(cudaError_t status, const char* expression) {
  if(status == cudaSuccess)
    return true;
  std::cerr << expression << ": " << cudaGetErrorString(status) << "\n";
  return false;
}

int explicitRoundTiesEven(float value) {
  const float lowerFloat = std::floor(value);
  const int lower = static_cast<int>(lowerFloat);
  const float fraction = value - lowerFloat;
  if(fraction < 0.5f)
    return lower;
  if(fraction > 0.5f)
    return lower + 1;
  return (lower % 2 == 0) ? lower : lower + 1;
}

uint16_t halfBits(half value) {
  uint16_t bits;
  std::memcpy(&bits,&value,sizeof(bits));
  return bits;
}

} // namespace

int main() {
  constexpr int Rows = 129;
  constexpr int Channels = 256;
  constexpr float Epsilon = 1.0e-6f;
  std::vector<half> input((size_t)Rows * Channels);
  std::vector<half> gamma(Channels);
  for(int row = 0; row < Rows; row++) {
    for(int channel = 0; channel < Channels; channel++) {
      const float value =
        3.75f * std::sin(0.013f * float(row * Channels + channel)) +
        0.125f * float((channel % 11) - 5);
      input[(size_t)row * Channels + channel] = __float2half_rn(value);
    }
  }
  for(int channel = 0; channel < Channels; channel++)
    gamma[channel] = __float2half_rn(
      0.7f + 0.6f * std::cos(0.071f * float(channel)));

  half* dInput = nullptr;
  half* dGamma = nullptr;
  half* dReference = nullptr;
  half* dDual = nullptr;
  int8_t* dQuantized = nullptr;
  cudaStream_t stream = nullptr;
  const size_t activationBytes = input.size() * sizeof(half);
  bool ok = check(cudaMalloc(&dInput,activationBytes),"cudaMalloc(dInput)") &&
    check(cudaMalloc(&dGamma,gamma.size() * sizeof(half)),"cudaMalloc(dGamma)") &&
    check(cudaMalloc(&dReference,activationBytes),"cudaMalloc(dReference)") &&
    check(cudaMalloc(&dDual,activationBytes),"cudaMalloc(dDual)") &&
    check(cudaMalloc(&dQuantized,input.size()),"cudaMalloc(dQuantized)") &&
    check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),
          "cudaStreamCreateWithFlags");
  if(ok)
    ok = check(cudaMemcpyAsync(dInput,input.data(),activationBytes,
          cudaMemcpyHostToDevice,stream),"copy input") &&
      check(cudaMemcpyAsync(dGamma,gamma.data(),gamma.size() * sizeof(half),
          cudaMemcpyHostToDevice,stream),"copy gamma");
  if(ok)
    ok = check(Renju15Sm120::launchRmsNorm256(
          dInput,dReference,dGamma,Rows,Epsilon,
          Renju15Sm120::RmsNorm256Tactic::Warp4Vec8,stream),
          "launchRmsNorm256") &&
      check(Renju15Sm120::launchRmsNorm256Fp16Int8(
          dInput,dDual,dQuantized,dGamma,Rows,Epsilon,
          Renju15Sm120::RmsNorm256Tactic::Warp4Vec8,stream),
          "launchRmsNorm256Fp16Int8");

  std::vector<half> reference(input.size());
  std::vector<half> dual(input.size());
  std::vector<int8_t> quantized(input.size());
  if(ok)
    ok = check(cudaMemcpyAsync(reference.data(),dReference,activationBytes,
          cudaMemcpyDeviceToHost,stream),"copy reference") &&
      check(cudaMemcpyAsync(dual.data(),dDual,activationBytes,
          cudaMemcpyDeviceToHost,stream),"copy dual") &&
      check(cudaMemcpyAsync(quantized.data(),dQuantized,quantized.size(),
          cudaMemcpyDeviceToHost,stream),"copy quantized") &&
      check(cudaStreamSynchronize(stream),"cudaStreamSynchronize");

  size_t fp16Mismatches = 0;
  size_t quantMismatches = 0;
  size_t minus128Count = 0;
  if(ok) {
    for(size_t i = 0; i < dual.size(); i++) {
      if(halfBits(reference[i]) != halfBits(dual[i]))
        fp16Mismatches++;
      const float roundedFp16 = __half2float(dual[i]);
      const float clipped = std::max(-4.0f,std::min(4.0f,roundedFp16));
      int expected = explicitRoundTiesEven(clipped * (127.0f / 4.0f));
      expected = std::max(-127,std::min(127,expected));
      if(static_cast<int>(quantized[i]) != expected)
        quantMismatches++;
      if(static_cast<int>(quantized[i]) == -128)
        minus128Count++;
    }
  }

  if(stream != nullptr)
    cudaStreamDestroy(stream);
  cudaFree(dQuantized);
  cudaFree(dDual);
  cudaFree(dReference);
  cudaFree(dGamma);
  cudaFree(dInput);

  std::cout << "{\"rows\":" << Rows
            << ",\"elements\":" << dual.size()
            << ",\"fp16_bit_mismatches\":" << fp16Mismatches
            << ",\"cpu_quant_mismatches\":" << quantMismatches
            << ",\"minus128_count\":" << minus128Count
            << ",\"pass\":"
            << (ok && fp16Mismatches == 0 && quantMismatches == 0 &&
                minus128Count == 0 ? "true" : "false") << "}\n";
  return ok && fp16Mismatches == 0 && quantMismatches == 0 &&
    minus128Count == 0 ? 0 : 1;
}

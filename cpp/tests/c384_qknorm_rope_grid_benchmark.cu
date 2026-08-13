#include "neuralnet/cuda_specialized/sm120/c384/fixed_batch/qknorm_rope.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

using C384QKNormRopeSm120::LaunchParams;

void require(cudaError_t status, const char* operation) {
  if(status == cudaSuccess)
    return;
  std::cerr << operation << ": " << cudaGetErrorString(status) << std::endl;
  std::exit(1);
}

LaunchParams params() {
  LaunchParams value;
  value.abiVersion = C384QKNormRopeSm120::kAbiVersion;
  value.batch = C384QKNormRopeSm120::kBatch;
  value.sequence = C384QKNormRopeSm120::kSequence;
  value.heads = C384QKNormRopeSm120::kHeads;
  value.kvHeads = C384QKNormRopeSm120::kHeads;
  value.headDim = C384QKNormRopeSm120::kHeadDim;
  value.tokenRows = value.batch * value.sequence;
  value.deviceOrdinal = 0;
  value.computeCapability = 120;
  value.usingFp16 = true;
  value.usingNhwc = true;
  value.learnedRope = true;
  value.qkNorm = true;
  value.inputSemantic = C384QKNormRopeSm120::InputSemantic::RawPackedQkv;
  value.qEpsilon = C384QKNormRopeSm120::kRmsEpsilon;
  value.kEpsilon = C384QKNormRopeSm120::kRmsEpsilon;
  return value;
}

struct Buffers {
  half* packed[2] = {nullptr,nullptr};
  half* qGamma = nullptr;
  half* kGamma = nullptr;
  half2* rope = nullptr;
  cudaStream_t streams[2] = {nullptr,nullptr};

  ~Buffers() {
    if(streams[1] != nullptr) cudaStreamDestroy(streams[1]);
    if(streams[0] != nullptr) cudaStreamDestroy(streams[0]);
    if(rope != nullptr) cudaFree(rope);
    if(kGamma != nullptr) cudaFree(kGamma);
    if(qGamma != nullptr) cudaFree(qGamma);
    if(packed[1] != nullptr) cudaFree(packed[1]);
    if(packed[0] != nullptr) cudaFree(packed[0]);
  }
};

}  // namespace

int main(int argc, char** argv) {
  const int iterations = argc > 1 ? std::atoi(argv[1]) : 2000;
  if(iterations <= 0)
    return 2;
  cudaDeviceProp properties;
  require(cudaGetDeviceProperties(&properties,0),"cudaGetDeviceProperties");
  if(properties.major * 10 + properties.minor != 120) {
    std::cerr << "SM120 device required" << std::endl;
    return 2;
  }
  require(cudaSetDevice(0),"cudaSetDevice");

  constexpr size_t packedElements =
    static_cast<size_t>(C384QKNormRopeSm120::kBatch) *
    C384QKNormRopeSm120::kSequence * C384QKNormRopeSm120::kPackedChannels;
  constexpr size_t ropeElements =
    static_cast<size_t>(C384QKNormRopeSm120::kSequence) *
    C384QKNormRopeSm120::kHeads * C384QKNormRopeSm120::kRopePairsPerHead;
  std::vector<half> hostPacked(packedElements,__float2half_rn(0.125f));
  std::vector<half> hostGamma(C384QKNormRopeSm120::kHeadDim,__float2half_rn(1.0f));
  std::vector<half2> hostRope(ropeElements,__floats2half2_rn(1.0f,0.0f));

  Buffers buffers;
  for(int lane = 0; lane < 2; lane++) {
    require(cudaMalloc(&buffers.packed[lane],packedElements * sizeof(half)),
      "cudaMalloc packed");
    require(cudaMemcpy(buffers.packed[lane],hostPacked.data(),
      packedElements * sizeof(half),cudaMemcpyHostToDevice),"copy packed");
    require(cudaStreamCreateWithFlags(
      &buffers.streams[lane],cudaStreamNonBlocking),"create stream");
  }
  require(cudaMalloc(&buffers.qGamma,hostGamma.size() * sizeof(half)),
    "cudaMalloc qGamma");
  require(cudaMalloc(&buffers.kGamma,hostGamma.size() * sizeof(half)),
    "cudaMalloc kGamma");
  require(cudaMalloc(&buffers.rope,hostRope.size() * sizeof(half2)),
    "cudaMalloc rope");
  require(cudaMemcpy(buffers.qGamma,hostGamma.data(),hostGamma.size() * sizeof(half),
    cudaMemcpyHostToDevice),"copy qGamma");
  require(cudaMemcpy(buffers.kGamma,hostGamma.data(),hostGamma.size() * sizeof(half),
    cudaMemcpyHostToDevice),"copy kGamma");
  require(cudaMemcpy(buffers.rope,hostRope.data(),hostRope.size() * sizeof(half2),
    cudaMemcpyHostToDevice),"copy rope");

  const LaunchParams launchParams = params();
  struct Geometry { int threads; int grid; };
  constexpr Geometry geometries[] = {
    {64,1360},{128,680},{256,340},{512,170},
    {64,2720},{128,1360},{256,680},{512,340},
    {64,5440},{128,2720},{256,1360},{512,680},
  };
  std::cout << std::fixed << std::setprecision(4);
  for(const Geometry geometry: geometries) {
    for(int warmup = 0; warmup < 50; warmup++) {
      require(C384QKNormRopeSm120::launchInPlaceForGeometryQualification(
        launchParams,buffers.packed[0],buffers.qGamma,buffers.kGamma,
        buffers.rope,geometry.grid,geometry.threads,buffers.streams[0]),
        "S1 warmup");
    }
    require(cudaStreamSynchronize(buffers.streams[0]),"S1 warmup sync");
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    require(cudaEventCreate(&start),"event start create");
    require(cudaEventCreate(&stop),"event stop create");
    require(cudaEventRecord(start,buffers.streams[0]),"event start record");
    for(int i = 0; i < iterations; i++) {
      require(C384QKNormRopeSm120::launchInPlaceForGeometryQualification(
        launchParams,buffers.packed[0],buffers.qGamma,buffers.kGamma,
        buffers.rope,geometry.grid,geometry.threads,buffers.streams[0]),
        "S1 launch");
    }
    require(cudaEventRecord(stop,buffers.streams[0]),"event stop record");
    require(cudaEventSynchronize(stop),"event stop sync");
    float s1Ms = 0.0f;
    require(cudaEventElapsedTime(&s1Ms,start,stop),"event elapsed");
    require(cudaEventDestroy(stop),"event stop destroy");
    require(cudaEventDestroy(start),"event start destroy");

    for(int warmup = 0; warmup < 50; warmup++) {
      for(int lane = 0; lane < 2; lane++) {
        require(C384QKNormRopeSm120::launchInPlaceForGeometryQualification(
          launchParams,buffers.packed[lane],buffers.qGamma,buffers.kGamma,
          buffers.rope,geometry.grid,geometry.threads,buffers.streams[lane]),
          "S2 warmup");
      }
    }
    require(cudaDeviceSynchronize(),"S2 warmup sync");
    const auto s2Start = std::chrono::steady_clock::now();
    for(int i = 0; i < iterations; i++) {
      for(int lane = 0; lane < 2; lane++) {
        require(C384QKNormRopeSm120::launchInPlaceForGeometryQualification(
          launchParams,buffers.packed[lane],buffers.qGamma,buffers.kGamma,
          buffers.rope,geometry.grid,geometry.threads,buffers.streams[lane]),
          "S2 launch");
      }
    }
    require(cudaDeviceSynchronize(),"S2 sync");
    const auto s2Stop = std::chrono::steady_clock::now();
    const double s2Us = std::chrono::duration<double,std::micro>(
      s2Stop - s2Start).count();
    std::cout << "GEOMETRY threads=" << geometry.threads
              << " grid=" << geometry.grid
              << " s1_us=" << (1000.0 * s1Ms / iterations)
              << " s2_pair_us=" << (s2Us / iterations)
              << " s2_effective_us_per_launch=" << (s2Us / (2.0 * iterations))
              << std::endl;
  }
  return 0;
}

#include "quantization.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

int main() {
  constexpr int K = 4;
  constexpr int N = 5;
  const std::vector<float> weights = {
    -4.25f,-2.5f,-1.5f,-0.5f,0.5f,
    1.5f,2.5f,3.75f,4.25f,-3.125f,
    0.0f,0.03125f,-0.03125f,2.0f,-2.0f,
    1.0f,-1.0f,0.25f,-0.25f,3.0f,
  };
  try {
    Renju15Int8Quantization::validateContract();
    const float scale = Renju15Int8Quantization::perMatrixScale(weights);
    const std::vector<int8_t> packed =
      Renju15Int8Quantization::quantizeAndPackMatrix(weights,K,N,scale);
    uint32_t scaleBits = 0;
    std::memcpy(&scaleBits,&scale,sizeof(scaleBits));
    std::cout << "{\"scale_bits\":" << scaleBits << ",\"packed\":[";
    for(size_t i = 0; i < packed.size(); i++) {
      if(i > 0)
        std::cout << ',';
      std::cout << static_cast<int>(packed[i]);
    }
    std::cout << "]}\n";
  }
  catch(const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
  return 0;
}

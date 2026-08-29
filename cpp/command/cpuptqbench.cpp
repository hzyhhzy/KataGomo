#ifdef USE_CPU_PTQ_BACKEND

#include "../main.h"

#include "../neuralnet/cpuptq/kernel.h"
#include "../neuralnet/cpuptq/model.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace std;

namespace {

constexpr size_t INPUT_FLOATS =
  CpuPtq::SPATIAL_INPUTS * CpuPtq::BOARD_AREA + CpuPtq::GLOBAL_INPUTS;
constexpr size_t OUTPUT_FLOATS =
  CpuPtq::POLICY_SIZE + CpuPtq::VALUE_SIZE + CpuPtq::MISC_VALUE_SIZE;

[[noreturn]] void fail(const string& message) {
  throw StringError("cpuptqbench: " + message);
}

int parseNonNegative(const string& text, const string& option) {
  char* end = nullptr;
  const long value = std::strtol(text.c_str(),&end,10);
  if(end == text.c_str() || *end != '\0' || value < 0 || value > 1000000000L)
    fail("invalid value for " + option + ": " + text);
  return static_cast<int>(value);
}

vector<float> readFloats(const string& path) {
  ifstream input(path,ios::binary | ios::ate);
  if(!input)
    fail("could not open input " + path);
  const streamoff bytes = input.tellg();
  if(bytes <= 0 || bytes % static_cast<streamoff>(sizeof(float)) != 0)
    fail("input must contain a nonempty whole number of FP32 values");
  vector<float> values(static_cast<size_t>(bytes) / sizeof(float));
  input.seekg(0,ios::beg);
  input.read(reinterpret_cast<char*>(values.data()),bytes);
  if(!input)
    fail("failed reading input " + path);
  return values;
}

void writeFloats(const string& path, const vector<float>& values) {
  ofstream output(path,ios::binary | ios::trunc);
  if(!output)
    fail("could not open output " + path);
  output.write(
    reinterpret_cast<const char*>(values.data()),
    static_cast<streamsize>(values.size() * sizeof(float)));
  if(!output)
    fail("failed writing output " + path);
}

vector<float> makeDeterministicInput() {
  vector<float> values(INPUT_FLOATS);
  uint32_t state = 0x9E3779B9U;
  for(float& value: values) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    value = static_cast<float>(static_cast<int32_t>(state)) *
      (1.0f / 2147483648.0f);
  }
  return values;
}

void inferRow(CpuPtq::Kernel& kernel, const float* input, float* output) {
  kernel.infer(
    input,
    input + CpuPtq::SPATIAL_INPUTS * CpuPtq::BOARD_AREA,
    output,
    output + CpuPtq::POLICY_SIZE,
    output + CpuPtq::POLICY_SIZE + CpuPtq::VALUE_SIZE);
}

}  // namespace

int MainCmds::cpuptqbench(const vector<string>& args) {
  string modelPath;
  string inputPath;
  string outputPath;
  string tracePath;
  int warmup = 20;
  int iterations = 1000;

  for(size_t index = 0; index < args.size(); index++) {
    const string& option = args[index];
    if(index == 0 && option == "cpuptqbench")
      continue;
    if(option == "-model" || option == "-input" || option == "-output" || option == "-trace" ||
       option == "-warmup" || option == "-iters") {
      if(index + 1 >= args.size())
        fail("missing value after " + option);
      const string value = args[++index];
      if(option == "-model") modelPath = value;
      else if(option == "-input") inputPath = value;
      else if(option == "-output") outputPath = value;
      else if(option == "-trace") tracePath = value;
      else if(option == "-warmup") warmup = parseNonNegative(value,option);
      else iterations = parseNonNegative(value,option);
    }
    else if(option == "-help" || option == "--help" || option == "-h") {
      cout << "Usage: katago cpuptqbench -model MODEL.bin.gz "
           << "[-input INPUT.f32 -output OUTPUT.f32] "
           << "[-trace TRACES.f32] [-warmup N] [-iters N]" << endl;
      return 0;
    }
    else
      fail("unknown option " + option);
  }
  if(modelPath.empty())
    fail("-model is required");
  if(!outputPath.empty() && inputPath.empty())
    fail("-output requires -input");
  if(!tracePath.empty() && inputPath.empty())
    fail("-trace requires -input");

  vector<float> input = inputPath.empty() ? makeDeterministicInput() : readFloats(inputPath);
  if(input.size() % INPUT_FLOATS != 0)
    fail("input row width must be exactly " + Global::uint64ToString(INPUT_FLOATS));
  const size_t rows = input.size() / INPUT_FLOATS;

  CpuPtq::Model model = CpuPtq::loadModelFile(modelPath,"");
  unique_ptr<CpuPtq::Kernel> kernel = CpuPtq::createKernel(model);
  vector<float> output(OUTPUT_FLOATS);

  if(!tracePath.empty()) {
    vector<float> allTraces;
    vector<float> rowTraces;
    for(size_t row = 0; row < rows; row++) {
      kernel->inferWithTraces(
        input.data() + row * INPUT_FLOATS,
        input.data() + row * INPUT_FLOATS +
          CpuPtq::SPATIAL_INPUTS * CpuPtq::BOARD_AREA,
        output.data(),output.data() + CpuPtq::POLICY_SIZE,
        output.data() + CpuPtq::POLICY_SIZE + CpuPtq::VALUE_SIZE,
        rowTraces);
      allTraces.insert(allTraces.end(),rowTraces.begin(),rowTraces.end());
    }
    writeFloats(tracePath,allTraces);
  }

  if(!outputPath.empty()) {
    vector<float> allOutputs(rows * OUTPUT_FLOATS);
    for(size_t row = 0; row < rows; row++)
      inferRow(*kernel,input.data() + row * INPUT_FLOATS,
               allOutputs.data() + row * OUTPUT_FLOATS);
    writeFloats(outputPath,allOutputs);
  }

  for(int iteration = 0; iteration < warmup; iteration++)
    inferRow(*kernel,input.data() + (static_cast<size_t>(iteration) % rows) * INPUT_FLOATS,
             output.data());

  double checksum = 0.0;
  const auto start = chrono::steady_clock::now();
  for(int iteration = 0; iteration < iterations; iteration++) {
    inferRow(*kernel,input.data() + (static_cast<size_t>(iteration) % rows) * INPUT_FLOATS,
             output.data());
    checksum += output[static_cast<size_t>(iteration) % OUTPUT_FLOATS];
  }
  const auto end = chrono::steady_clock::now();
  const double seconds = chrono::duration<double>(end - start).count();
  const double perSecond = iterations > 0 && seconds > 0.0 ? iterations / seconds : 0.0;
  const double microseconds = iterations > 0 ? seconds * 1.0e6 / iterations : 0.0;

  cout << fixed << setprecision(6)
       << "profile=" << kernel->profile().name << '\n'
       << "input_rows=" << rows << '\n'
       << "warmup=" << warmup << '\n'
       << "iterations=" << iterations << '\n'
       << "seconds=" << seconds << '\n'
       << "inferences_per_second=" << perSecond << '\n'
       << "microseconds_per_inference=" << microseconds << '\n'
       << "checksum=" << setprecision(9) << checksum << endl;
  return 0;
}

#endif  // USE_CPU_PTQ_BACKEND

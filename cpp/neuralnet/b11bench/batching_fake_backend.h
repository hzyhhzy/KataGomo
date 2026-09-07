#ifndef B11_BATCHING_FAKE_BACKEND_H_
#define B11_BATCHING_FAKE_BACKEND_H_

// Test-only observations: the fake backend never loads weights or initializes a GPU.
#include <cstdint>
#include <string>
#include <vector>

namespace BatchingFake {
struct Row {
  uintptr_t input;
  uintptr_t output;
  uintptr_t owner;
  bool requestedOwner;
  bool hasMeta;
  int boardX;
  int boardY;
  int symmetry;
  double optimism;
  std::vector<float> spatial;
  std::vector<float> global;
  std::vector<float> meta;
};
struct Call {
  int batch;
  int capacity;
  int serverThread;
  std::vector<Row> rows;
};
void reset();
void blockNextCall();
bool waitUntilBlocked(int milliseconds);
void releaseCall();
std::vector<Call> calls();
std::vector<std::string> errors();
int liveHandles();
int liveBuffers();
std::vector<int> expectedThreadMapping();
}
#endif

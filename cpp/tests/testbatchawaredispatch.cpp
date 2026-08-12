#include "../tests/tests.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "../core/test.h"
#include "../neuralnet/nneval.h"

using namespace std;

namespace {

struct TrackedDummyOutput {
  static atomic<int> liveCount;
  TrackedDummyOutput() { liveCount.fetch_add(1); }
  ~TrackedDummyOutput() { liveCount.fetch_sub(1); }
};

atomic<int> TrackedDummyOutput::liveCount(0);

static void assertDispatchPlan(
  bool enabled,
  int logicalRows,
  int expectedPhysicalRows,
  int expectedPaddedRows
) {
  const NNBatchDispatchPlan plan = getNNBatchDispatchPlan(enabled,logicalRows,36);
  testAssert(plan.logicalRows == logicalRows);
  testAssert(plan.physicalRows == expectedPhysicalRows);
  testAssert(plan.paddedRows == expectedPaddedRows);
  for(int row = 0; row < plan.physicalRows; row++)
    testAssert(plan.sourceRowForPhysicalRow(row) == min(row,logicalRows-1));
}

}  // namespace

void Tests::runBatchAwareDispatchTests() {
  // Evaluator concurrency is local to one normalized physical device. The
  // default CUDA ordinal -1 aliases GPU 0; lanes on other GPUs are not counted.
  testAssert(getSameGpuEvaluatorConcurrency({0},0) == 1);
  testAssert(getSameGpuEvaluatorConcurrency({0,0},0) == 2);
  testAssert(getSameGpuEvaluatorConcurrency({0,0},1) == 2);
  testAssert(getSameGpuEvaluatorConcurrency({0,1},0) == 1);
  testAssert(getSameGpuEvaluatorConcurrency({0,1},1) == 1);
  testAssert(getSameGpuEvaluatorConcurrency({-1,0,1},0) == 2);
  testAssert(getSameGpuEvaluatorConcurrency({-1,0,1},1) == 2);
  testAssert(getSameGpuEvaluatorConcurrency({-1,0,1},2) == 1);
  testAssert(getSameGpuEvaluatorConcurrency({2,2,2,3},1) == 3);

  // Fixed-B planning for the qualification boundary cases. A 37-row arrival is
  // represented below as one full B36 launch followed by one logical row.
  assertDispatchPlan(true,1,36,35);
  assertDispatchPlan(true,35,36,1);
  assertDispatchPlan(true,36,36,0);
  assertDispatchPlan(false,1,1,0);
  assertDispatchPlan(false,35,35,0);
  assertDispatchPlan(false,36,36,0);

  // Padded lanes repeat the final real row, and shrinking ownership back to the
  // logical size destroys every dummy immediately.
  {
    vector<unique_ptr<TrackedDummyOutput>> owned;
    for(int row = 0; row < 36; row++)
      owned.push_back(make_unique<TrackedDummyOutput>());
    testAssert(TrackedDummyOutput::liveCount.load() == 36);
    owned.resize(1);
    testAssert(TrackedDummyOutput::liveCount.load() == 1);
  }
  testAssert(TrackedDummyOutput::liveCount.load() == 0);

  // An idle GPU flushes either a 1-row or 35-row partial immediately.
  NNBatchAwareDispatchState idle(true,{0,0});
  testAssert(idle.canStartBatch(0,false,1));
  testAssert(idle.canStartBatch(1,false,35));

  // Once one same-GPU lane is active, partial work gathers. Full B36 work is
  // still dispatchable on the second lane, and a partial waits until both full
  // launches complete.
  NNBatchAwareDispatchState sameGpu(true,{0,0});
  sameGpu.markBatchStarted(0);
  testAssert(!sameGpu.canStartBatch(1,false,35));
  testAssert(sameGpu.canStartBatch(1,true,0));
  sameGpu.markBatchStarted(1);
  testAssert(!sameGpu.canStartBatch(0,false,1));
  sameGpu.markBatchCompleted(0);
  testAssert(!sameGpu.canStartBatch(0,false,1));
  sameGpu.markBatchCompleted(1);
  testAssert(sameGpu.canStartBatch(0,false,1));

  // Explicit 37-row sequence: full launch first, remainder aggregates while
  // that device is busy, then idle-flushes after completion.
  NNBatchAwareDispatchState rows37(true,{0,0});
  testAssert(rows37.canStartBatch(0,true,0));
  rows37.markBatchStarted(0);
  testAssert(!rows37.canStartBatch(1,false,1));
  rows37.markBatchCompleted(0);
  testAssert(rows37.canStartBatch(1,false,1));

  // Device activity is local. CUDA's default ordinal (-1) normalizes to GPU 0,
  // while work on GPU 1 remains independently eligible.
  NNBatchAwareDispatchState multiGpu(true,{-1,0,1});
  multiGpu.markBatchStarted(0);
  testAssert(!multiGpu.canStartBatch(1,false,1));
  testAssert(multiGpu.canStartBatch(2,false,1));
  multiGpu.markBatchCompleted(0);

  // Default-off mode retains dynamic dispatch even when another same-device
  // server lane is active.
  NNBatchAwareDispatchState disabled(false,{0,0});
  disabled.markBatchStarted(0);
  testAssert(disabled.canStartBatch(1,false,1));
  testAssert(!disabled.hasActiveBatch(0));

  // Two real CPU threads contending for one idle GPU may start only one partial.
  NNBatchAwareDispatchState concurrent(true,{0,0});
  mutex stateMutex;
  atomic<int> ready(0);
  atomic<bool> go(false);
  int winner = -1;
  int starts = 0;
  vector<thread> lanes;
  for(int lane = 0; lane < 2; lane++) {
    lanes.emplace_back([&,lane]() {
      ready.fetch_add(1);
      while(!go.load())
        this_thread::yield();
      lock_guard<mutex> lock(stateMutex);
      if(concurrent.canStartBatch(lane,false,1)) {
        concurrent.markBatchStarted(lane);
        starts += 1;
        winner = lane;
      }
    });
  }
  while(ready.load() < 2)
    this_thread::yield();
  go.store(true);
  for(thread& lane : lanes)
    lane.join();
  testAssert(starts == 1);
  testAssert(winner == 0 || winner == 1);
  concurrent.markBatchCompleted(winner);
  testAssert(concurrent.canStartBatch(1-winner,false,1));
}

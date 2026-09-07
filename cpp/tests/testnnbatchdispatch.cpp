#include "../neuralnet/nnbatchdispatch.h"
#include "../core/threadsafequeue.h"

#include <chrono>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

using namespace std::chrono;

void require(bool condition, const char* message) {
  if(!condition)
    throw std::runtime_error(std::string("NN batch dispatcher test: ") + message);
}

struct Fixture {
  ThreadSafeQueue<int> queue;
  NNBatchingDispatcher dispatcher;
  std::atomic<int> cap;
  const int capacity;

  Fixture(bool enabled, const std::vector<int>& gpuIds, int batch = 13)
    : dispatcher(enabled, gpuIds), cap(batch), capacity(batch) {}

  void push(int value) {
    require(queue.forcePush(value), "enqueue into closed/read-only fixture");
    dispatcher.notify();
  }
  void setCap(int value) {
    cap.store(value, std::memory_order_release);
    dispatcher.notify();
  }
  void finishInput() {
    queue.setReadOnly();
    dispatcher.notify();
  }
};

struct PopResult {
  bool gotAnything;
  std::vector<int> rows;
};

// Unlike std::async, this future's destructor never blocks indefinitely. The
// worker owns the fixture and result promise, so a failed wakeup test can
// detach safely after a bounded cleanup attempt and report its failure.
class PopTask {
 public:
  PopTask(const std::shared_ptr<Fixture>& fixture_, int lane, bool preferFull,
          bool completeImmediately = true)
    : fixture(fixture_) {
    auto result = std::make_shared<std::promise<PopResult>>();
    ready = result->get_future().share();
    worker = std::thread([fixture_, result, lane, preferFull, completeImmediately] {
      try {
        PopResult output;
        output.gotAnything = fixture_->dispatcher.waitForBatch(
          fixture_->queue, output.rows, fixture_->capacity, fixture_->cap, lane, preferFull
        );
        if(output.gotAnything && completeImmediately)
          fixture_->dispatcher.completeBatch(lane);
        result->set_value(std::move(output));
      }
      catch(...) {
        result->set_exception(std::current_exception());
      }
    });
  }

  ~PopTask() {
    if(!worker.joinable())
      return;
    if(ready.wait_for(milliseconds(0)) != std::future_status::ready) {
      fixture->finishInput();
      if(ready.wait_for(seconds(2)) != std::future_status::ready) {
        worker.detach();
        return;
      }
    }
    worker.join();
  }

  PopResult get() {
    require(ready.wait_for(seconds(3)) == std::future_status::ready,
            "consumer failed to wake within 3 seconds");
    PopResult output = ready.get();
    worker.join();
    return output;
  }

  void requirePending() {
    require(ready.wait_for(milliseconds(30)) == std::future_status::timeout,
            "busy-device partial batch should remain queued");
  }

 private:
  std::shared_ptr<Fixture> fixture;
  std::shared_future<PopResult> ready;
  std::thread worker;
};

class ActiveLane {
 public:
  ActiveLane(const std::shared_ptr<Fixture>& fixture_, int lane_, int request,
             bool preferFull = true)
    : fixture(fixture_), lane(lane_), active(false) {
    fixture->push(request);
    PopTask task(fixture, lane, preferFull, false);
    PopResult result = task.get();
    require(result.gotAnything && result.rows == std::vector<int>{request},
            "failed to establish an active lane");
    active = true;
  }
  ~ActiveLane() {
    if(active)
      fixture->dispatcher.completeBatch(lane);
  }
  void complete() {
    require(active, "test attempted to complete a lane twice");
    fixture->dispatcher.completeBatch(lane);
    active = false;
  }
 private:
  std::shared_ptr<Fixture> fixture;
  int lane;
  bool active;
};

void testIdleAndBusy() {
  auto fixture = std::make_shared<Fixture>(true, std::vector<int>{0,0});
  ActiveLane first(fixture, 0, 100);
  fixture->push(101);
  PopTask second(fixture, 1, true);
  second.requirePending();
  first.complete();
  PopResult result = second.get();
  require(result.gotAnything && result.rows == std::vector<int>{101},
          "idle partial should return only real rows, with no dispatcher padding");
}

void testFullWhileBusy() {
  auto fixture = std::make_shared<Fixture>(true, std::vector<int>{0,0});
  ActiveLane first(fixture, 0, -1);
  fixture->push(0);
  PopTask second(fixture, 1, true);
  second.requirePending();
  for(int i = 1; i < 13; i++)
    fixture->push(i);
  PopResult result = second.get();
  require(result.gotAnything && result.rows.size() == 13, "full batch must run on busy GPU");
  for(int i = 0; i < 13; i++)
    require(result.rows[i] == i, "full batch lost FIFO order");
  first.complete();
}

void testDevicesAndDefaultAlias() {
  auto fixture = std::make_shared<Fixture>(true, std::vector<int>{-1,0,1});
  ActiveLane first(fixture, 0, 1);
  fixture->push(2);
  PopTask sameDevice(fixture, 1, true);
  sameDevice.requirePending();
  PopTask otherDevice(fixture, 2, true);
  PopResult result = otherDevice.get();
  require(result.gotAnything && result.rows == std::vector<int>{2},
          "idle GPU1 should not be held by busy GPU0");
  fixture->push(3);
  sameDevice.requirePending();
  first.complete();
  result = sameDevice.get();
  require(result.gotAnything && result.rows == std::vector<int>{3},
          "default GPU -1 must share activity with GPU0");
}

void testCapNotification() {
  auto fixture = std::make_shared<Fixture>(true, std::vector<int>{0,0});
  ActiveLane first(fixture, 0, 1);
  fixture->push(2);
  fixture->push(3);
  PopTask second(fixture, 1, true);
  second.requirePending();
  fixture->setCap(2);
  PopResult result = second.get();
  require(result.gotAnything && result.rows == std::vector<int>({2,3}),
          "lowering dispatch cap must wake a now-full batch");
  first.complete();
}

void testMixedWorkers() {
  auto fixture = std::make_shared<Fixture>(true, std::vector<int>{0,0,0});
  ActiveLane first(fixture, 0, 1);
  fixture->push(2);
  PopTask eligible(fixture, 1, true);
  eligible.requirePending();
  // A non-specialized lane skips busy holding, but participates in the same
  // serialized pop and active ownership. It may safely consume the pending row.
  PopTask generic(fixture, 2, false, false);
  PopResult result = generic.get();
  require(result.gotAnything && result.rows == std::vector<int>{2},
          "generic lane must remain able to consume a partial batch");
  first.complete();
  fixture->push(3);
  eligible.requirePending();
  fixture->dispatcher.completeBatch(2);
  result = eligible.get();
  require(result.gotAnything && result.rows == std::vector<int>{3},
          "generic active lane must count as GPU busy for eligible lanes");
}

void testReadOnlyDrainAndRespawn() {
  auto fixture = std::make_shared<Fixture>(true, std::vector<int>{0,0});
  {
    PopTask empty(fixture, 0, true);
    empty.requirePending();
    fixture->finishInput();
    require(!empty.get().gotAnything, "read-only empty queue must stop its waiter");
  }
  for(int cycle = 0; cycle < 8; cycle++) {
    fixture->dispatcher.resetGpuIdxByServerThread({0,0});
    fixture->queue.unsetReadOnly();
    fixture->dispatcher.notify();
    ActiveLane first(fixture, 0, 10);
    fixture->push(11);
    PopTask tail(fixture, 1, true);
    tail.requirePending();
    fixture->finishInput();
    first.complete();
    PopResult result = tail.get();
    require(result.gotAnything && result.rows == std::vector<int>{11},
            "shutdown must drain a pending partial batch");
    PopTask drained(fixture, 0, true);
    require(!drained.get().gotAnything, "drained read-only queue must return false");
  }
  fixture->dispatcher.resetGpuIdxByServerThread({0,1});
  fixture->queue.unsetReadOnly();
  fixture->dispatcher.notify();
  ActiveLane first(fixture, 0, 20);
  fixture->push(21);
  PopTask remapped(fixture, 1, true);
  require(remapped.get().rows == std::vector<int>{21}, "respawn retained stale GPU mapping");
  first.complete();
  fixture->finishInput();
}

void testCloseAfterConsumption() {
  auto fixture = std::make_shared<Fixture>(true, std::vector<int>{0});
  fixture->push(1);
  PopTask consumed(fixture, 0, true);
  require(consumed.get().rows == std::vector<int>{1}, "close setup did not consume its row");
  PopTask waiting(fixture, 0, true);
  waiting.requirePending();
  fixture->queue.close();
  fixture->dispatcher.notify();
  require(!waiting.get().gotAnything, "close must release the waiting consumer");
  require(fixture->queue.size() == 0, "closed queue must have a valid empty-size invariant");
  PopTask closed(fixture, 0, true);
  require(!closed.get().gotAnything, "already-closed queue should not block");
}

void testDisabledAndOwnershipChecks() {
  auto fixture = std::make_shared<Fixture>(false, std::vector<int>{0,0});
  ActiveLane first(fixture, 0, 1);
  fixture->setCap(2);
  fixture->push(2);
  fixture->push(3);
  fixture->push(4);
  PopTask normal(fixture, 1, true);
  require(normal.get().rows == std::vector<int>({2,3}), "disabled mode changed original batch cap");
  PopTask partial(fixture, 1, true);
  require(partial.get().rows == std::vector<int>{4}, "disabled mode must not hold partial rows");
  first.complete();
  fixture->finishInput();

  auto checked = std::make_shared<Fixture>(true, std::vector<int>{0});
  ActiveLane active(checked, 0, 1);
  bool rejected = false;
  try { checked->dispatcher.resetGpuIdxByServerThread({1}); }
  catch(const std::logic_error&) { rejected = true; }
  require(rejected, "reconfiguration must reject unfinished ownership");
  active.complete();
  rejected = false;
  try { checked->dispatcher.completeBatch(0); }
  catch(const std::logic_error&) { rejected = true; }
  require(rejected, "double completion must be rejected");
  checked->finishInput();
}

void testModeChangesAtStartup() {
  auto fixture = std::make_shared<Fixture>(false, std::vector<int>{0,0});
  // Queueing before startup must remain safe even when its notification sees
  // the initially disabled coordinator. The first worker observes the queue.
  fixture->push(1);
  fixture->dispatcher.setEnabled(true);
  PopTask initial(fixture, 0, true);
  require(initial.get().rows == std::vector<int>{1}, "startup mode change lost an earlier request");

  for(int cycle = 0; cycle < 4; cycle++) {
    fixture->dispatcher.setEnabled(true);
    fixture->dispatcher.resetGpuIdxByServerThread({0,0});
    fixture->queue.unsetReadOnly();
    fixture->dispatcher.notify();
    ActiveLane active(fixture, 0, 10);
    bool rejected = false;
    try { fixture->dispatcher.setEnabled(false); }
    catch(const std::logic_error&) { rejected = true; }
    require(rejected, "disabling must reject an unfinished active batch");
    fixture->push(11);
    PopTask coordinated(fixture, 1, true);
    coordinated.requirePending();
    active.complete();
    require(coordinated.get().rows == std::vector<int>{11}, "enabled mode did not resume its waiter");
    fixture->finishInput();

    // All preceding workers have completed and joined before changing mode.
    fixture->dispatcher.setEnabled(false);
    fixture->dispatcher.resetGpuIdxByServerThread({-1,0});
    fixture->queue.unsetReadOnly();
    fixture->dispatcher.notify();
    ActiveLane generic(fixture, 0, 20);
    fixture->push(21);
    PopTask immediate(fixture, 1, true);
    require(immediate.get().rows == std::vector<int>{21}, "disabled mode retained busy holding");
    generic.complete();
    fixture->finishInput();
  }

  // Producer notifications can overlap the startup setting without any
  // dispatcher workers. Exercise the atomic fast path without introducing
  // an unsupported concurrent mode change during an active waitForBatch.
  std::thread notifier([fixture] {
    for(int i = 0; i < 1000; i++)
      fixture->dispatcher.notify();
  });
  for(int i = 0; i < 1000; i++)
    fixture->dispatcher.setEnabled((i & 1) != 0);
  notifier.join();
  fixture->dispatcher.setEnabled(false);
}

void testConcurrentStress() {
  struct StressState {
    std::shared_ptr<Fixture> fixture = std::make_shared<Fixture>(true, std::vector<int>{0,0,1});
    std::mutex resultMutex;
    std::vector<int> seen = std::vector<int>(4000, 0);
  };
  auto state = std::make_shared<StressState>();
  std::vector<std::thread> consumers;
  std::vector<std::future<void>> finished;
  for(int lane = 0; lane < 3; lane++) {
    auto done = std::make_shared<std::promise<void>>();
    finished.push_back(done->get_future());
    consumers.emplace_back([state, done, lane] {
      try {
        std::vector<int> rows;
        while(state->fixture->dispatcher.waitForBatch(
                state->fixture->queue, rows, state->fixture->capacity,
                state->fixture->cap, lane, lane != 1)) {
          {
            std::lock_guard<std::mutex> lock(state->resultMutex);
            for(int row : rows) {
              require(row >= 0 && row < 4000, "stress delivered an invalid request ID");
              state->seen[row]++;
            }
          }
          state->fixture->dispatcher.completeBatch(lane);
          rows.clear();
          std::this_thread::yield();
        }
        done->set_value();
      }
      catch(...) { done->set_exception(std::current_exception()); }
    });
  }
  std::vector<std::thread> producers;
  for(int producer = 0; producer < 4; producer++) {
    producers.emplace_back([state, producer] {
      for(int i = 0; i < 1000; i++) {
        state->fixture->push(producer * 1000 + i);
        if(i % 17 == 0)
          std::this_thread::yield();
      }
    });
  }
  for(std::thread& producer : producers)
    producer.join();
  state->fixture->finishInput();
  bool timedOut = false;
  std::exception_ptr error;
  for(size_t i = 0; i < consumers.size(); i++) {
    if(finished[i].wait_for(seconds(5)) != std::future_status::ready) {
      consumers[i].detach();
      timedOut = true;
    }
    else {
      consumers[i].join();
      try { finished[i].get(); }
      catch(...) { error = std::current_exception(); }
    }
  }
  require(!timedOut, "concurrent stress did not drain within timeout");
  if(error)
    std::rethrow_exception(error);
  for(int count : state->seen)
    require(count == 1, "concurrent stress duplicated or dropped a request");
}

} // namespace

namespace Tests {
void runNNBatchDispatchTests() {
  testIdleAndBusy();
  testFullWhileBusy();
  testDevicesAndDefaultAlias();
  testCapNotification();
  testMixedWorkers();
  testReadOnlyDrainAndRespawn();
  testCloseAfterConsumption();
  testDisabledAndOwnershipChecks();
  testModeChangesAtStartup();
  testConcurrentStress();
  std::cout << "NN batch dispatcher tests passed" << std::endl;
}
} // namespace Tests

#ifdef KATAGO_NNBATCHDISPATCH_TEST_MAIN
int main() {
  try {
    Tests::runNNBatchDispatchTests();
    return 0;
  }
  catch(const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
#endif

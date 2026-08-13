# SM120 fixed-batch FA4 search method

This note records the reusable method used for the 15x15, no-mask,
FP16 C384/H12 attention search. It deliberately separates *batch selection*,
*kernel selection*, and *production packaging*. A winner from another head
count (for example the older H8/TN128 result) is evidence, not a default.

## 1. Freeze the semantic shape first

Write the physical attention problem as `(B,S,H,Dq,Dv,layout,dtype,mask)`.
For this search it is:

```
S=225, H=12, Dq=Dv=32, dtype=fp16, mask=none
B in {24,28} after native prescreen
layout in {planar-qkv, packed-token-qkv}
```

`B` is the engine's physical batch, including padding. It is not the logical
number of live positions. Shape, model recipe, layout, device `sm_120`, and
artifact identity are hard dispatch inputs. A neighbor batch must fall back
before any candidate kernel is enqueued.

For tile `(TM,TN)`, useful first-order quantities are:

```
q_tiles        = ceil(S / TM)
k_tiles        = ceil(S / TN)
CTAs per call  = B * H * q_tiles
QK tile work   = B * H * q_tiles * k_tiles
head waves     = (B * H * q_tiles) / SM_count
tail fraction  = ceil(CTAs / SM_count) - CTAs / SM_count
```

Compute these for the actual GPU SM count. They explain occupancy tails and
why changing H from 8 to 12 can reverse a prior tile winner. They do not
replace measurement: register pressure, shared memory, scheduler behavior,
and surrounding QKV/FFN work can dominate.

## 2. Prescreen batch independently, then rerank with exact FA4

Use the production engine and the intended stream count for a cheap native
prescreen across reasonable batches. Reject undersized batches and batches
whose latency or memory is operationally unreasonable. The present balanced
native result retained B24 and B28 (S1: 5552 versus 5764 eval/s; S2: 6134.50
versus 6180.58 eval/s). B16/B20 were clearly behind. B40 remains historical
evidence only and is not admitted by the final registry.

This prescreen cannot name the final batch. Exact FA4 changes the CTA count and
tail, so every surviving batch is regenerated at the *same* kernel coordinate
and rerun in one fat search binary. Comparing B24 at one tile with B28 at a
different tile confounds batch and kernel effects and is rejected by the
package and registry validators.

## 3. Bounded kernel search, not an inherited single tile

Stage 1 is intentionally bounded:

```
TM=128, stages=1, warps=4
TN in {64,96,128}
accumulation in {fp32,qk16,pv16,both16}
layout fixed to the layout actually produced by the selected QKV path
```

That is 12 coordinates, generated separately at both B24 and B28 only after a
coordinate survives static compilation/resource checks. `fp32` is the numeric
anchor. Reduced accumulation modes are candidates only after raw-output
correctness passes; compilation or speed alone never qualifies them.

Prune Stage 1 before constructing a fat binary:

1. Generator and metadata contract pass.
2. The object contains an embedded `sm_120` cubin and its exported symbols use
   the native unique search prefix.
3. Stack and local memory are zero. Registers are at most 255/thread and
   dynamic/static shared-memory use stays below the device/launch limit.
4. Raw outputs versus the same-model cuDNN baseline pass all tolerances.
5. Short timing is stable enough to rank; keep only a small Pareto set across
   latency, throughput, registers, and shared memory.

Only then may Stage 2 expand around finalists, one dimension at a time:
`TM in {64,128}`, `stages in {1,2}`, and `warps in {4,8}`. Do not take their
Cartesian product up front. If a change cannot plausibly improve waves,
latency hiding, or resource pressure, do not add it. Rerun resource and
correctness gates after every expansion.

There are two distinct search packages:

- **Tile refinement package:** a small set of unique native ABIs at one exact
  batch. It answers which tile/accumulation coordinate survives.
- **Batch search package:** exactly two artifacts, B24 and B28, at one identical
  coordinate. It answers which physical batch wins.

`BatchSearch count=2` therefore does not mean two tiles. The checked provider
intentionally rejects a mixed-coordinate B24/B28 registry.

## 4. Unique search ABI and immutable layout proof

Every search artifact is generated with its own C stem and prefix containing
batch, tile, stages, warps, accumulation, and layout. Multiple objects can then
coexist in one binary without `objcopy` symbol rewriting. The object SHA is
recorded before packaging and verified by CMake.

Input layout is an enum, never a boolean or pointer heuristic:

- `PlanarQkv`: independent logical Q, K, and V tensors.
- `PackedTokenQkv`: each token row has contiguous `[Q,H,D][K,H,D][V,H,D]`
  segments and row stride `3*H*D` elements.

The QKV+RoPE path may select exact packed FA4 only if
`prepareProofForExactBatch(B,device,PackedTokenQkv,proof)` succeeds for the same
immutable artifact. The proof binds ABI version, B/S/H/D, device, layout,
candidate ID, and launch-function cookie. A planar proof cannot authorize a
packed launch. CUDA stream value 0 is the valid default stream and is not
rejected.

## 5. Correctness gate

Build two binaries from the same source/model inputs: package-off cuDNN
baseline and package-on candidate. Use deterministic symmetry and exact
physical padding. Capture unrounded raw policy, value, score, and ownership
head tensors. Require finite values, identical shapes, and:

| Tensor | max absolute | RMSE |
|---|---:|---:|
| policy | 0.03 | 0.005 |
| value | 0.03 | 0.005 |
| score | 0.05 | 0.01 |
| ownership | 0.03 | 0.005 |

The logs must show exactly one batch-aware dispatch and the expected candidate
marker, with no cuDNN marker in the candidate run. The baseline must show
cuDNN and no FA4 marker. Archive binary/model/config/input/log/dump SHA256s.

## 6. Balanced performance decision

Use the same fat binary, model, config, stream count, warmup, and iteration
count for both batches. The long S2 protocol is W30/I150 with two server
threads. Run two balanced quartets:

```
A B B A   B A A B        (A=B24, B=B28)
```

This gives four samples per candidate and balances position and thermal drift.
Audit exact marker count per server thread, batch, iteration count, GPU, mask,
and throughput arithmetic. Require population CV <= 1% and first-to-last drift
<= 2% for each case. A failed stability gate invalidates the ranking; it is not
fixed by selecting the faster noisy sample. Always audit that the GPU compute
process list is empty before the first leg and in a `finally` block, and retain
logs plus an exception report on failure.

Select on mean eval/s after the stability gate, report median as a robustness
check, and report the percent margin over runner-up. If the margin is close to
noise, extend balanced quartets instead of guessing.

## 7. Production collapse

After batch and kernel coordinates win:

1. Regenerate exactly one artifact in `production` mode with stable stem
   `c384_fa4_winner` and prefix `c384fa4win`.
2. Package one batch only. Search ABIs and losing objects must not be linked.
3. Verify all asset, registry, and runtime SHA256s during configure.
4. Re-run raw correctness and an ABBA comparison against the search winner to
   prove that ABI collapse did not change code generation or dispatch.
5. Keep strict automatic model/shape/device/layout gates. Unsupported models,
   masks, layouts, neighbor batches, and devices fall back before enqueue.

The generated object is never edited in place. Production identity comes from
fresh generation, not symbol rewriting.

## 8. Rejected tactics and common failure modes

- Carrying the H8/TN128 winner into H12 without a TN64/96/128 search: different
  CTA/wave geometry, so the premise is invalid.
- Choosing B28 from cuDNN/native prescreen alone: exact FA4 can reorder batches.
- Comparing different tiles while choosing a batch: confounded experiment.
- Building every batch/tile/accumulation combination into one binary: slow
  builds, large link surface, and weak auditability. Prune in stages.
- Inferring packed versus planar from pointer adjacency: row strides differ and
  a false inference can silently read the wrong tensor.
- Treating `cudaStream_t{0}` as invalid: it is CUDA's default stream.
- Accepting a kernel with spill/local memory because it benchmarks once: it is
  fragile under concurrency and must be rejected or explicitly requalified.
- Timing startup or teardown/padding transients: measure steady state only.
- Keeping search ABIs in production or mutating symbols with `objcopy`: both
  weaken identity and supply-chain evidence.

## 9. Adapting to another shape

Before reusing the process, answer every item:

1. Record B/S/H/Dq/Dv, dtype, mask, layout, architecture, SM count, and streams.
2. Recompute CTA, QK-tile, waves, and tail metrics for every proposed TM/TN.
3. Re-run native batch prescreen; do not reuse B24/B28 by habit.
4. Start from at least three TN values appropriate for S, with fp32 anchor and
   separately correctness-qualified reduced accumulation modes.
5. Give every search artifact a native unique ABI and immutable metadata/SHA.
6. Audit cubin architecture, registers, shared, stack, local, and symbols.
7. Prove the producer/consumer layout with a same-batch prepared proof.
8. Pass raw-output correctness against the same-model baseline.
9. Rerank exact candidates with balanced ABBA/CV/drift gates.
10. Fresh-generate and retest a single stable-ABI production artifact.

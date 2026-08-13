# SM120 fixed-batch transformer optimization playbook

This document records the reusable method used to specialize KataGo-style
transformers for NVIDIA SM120. It is shape-driven: trained weight values,
checkpoint names, artifact names, and model-file hashes are not dispatch keys.
A generated object is evidence only after its source, object, registry,
correctness, and performance records have all passed.

## 1. Evidence labels

| Label | Meaning |
|---|---|
| **Measured** | A saved result with binary/model/config identity and a reproducible runner. |
| **Contract** | A source invariant or fail-closed test. It proves safety, not speed. |
| **Hypothesis** | A candidate worth measuring, not a production winner. |
| **Rejected** | A route excluded for a recorded technical or measured reason. |

Current relevant evidence:

| Item | Label | Evidence / limitation |
|---|---|---|
| Dynamic C384 production is faster than its generic baseline over B1..B128, S1/S2 | Measured | `runtime_data/renju15_sm120/c384_fp16_stage_20260813/production_formal_abba_4067fe85_v1.summary.json`; it does not measure these exact AOT kernels. |
| Historical C256 outer search selected B36 among B32/B36/B40 | Measured | `runtime_data/renju15_sm120/batch_outer_aot/DENSE_BATCH_OUTER_REPORT.md`; process evidence, not a C384 winner. |
| B28 was provisionally strong in the first C384 native outer scan | Measured, preliminary | Coordinator status only; balanced B16/B20/B24/B28 v2 must supersede it. |
| B24/B28 generator and registry contracts exist | Contract | `python/c384_exact_fixed_aot/`; no CuTe object or speed result is checked in. |
| A candidate in `search_space.json` is faster | Hypothesis | Unproven until target generation, correctness, and balanced timing. |

## 2. Derive operator shapes from the model

Let board width/height be `X,Y`, sequence `S=X*Y`, physical batch `B`,
GEMM rows `M=B*S`, trunk channels `C`, query/KV heads `Hq,Hkv`, head
dimensions `Dq,Dv`, and FFN width `F`.

The current true target is:

| Field | Value |
|---|---:|
| Model depth | 36 attention + 36 FFN blocks |
| `X,Y,S` | `15,15,225` |
| `C` | 384 |
| `Hq,Hkv` | 12,12 |
| `Dq,Dv` | 32,32 |
| `F` | 1024 |
| Numeric/layout/mask | FP16, NHWC, no mask |
| Device family | SM120 |

For this non-GQA shape:

```text
Q width = Hq * Dq   = 12 * 32 = 384
K width = Hkv * Dq  = 12 * 32 = 384
V width = Hkv * Dv  = 12 * 32 = 384
QKV N    = 384 + 384 + 384 = 1152
RoPE half2 pairs = Hkv * (Dq / 2) = 12 * 16 = 192
```

| Physical batch | `M=B*225` | QKV GEMM | Dual FFN projection |
|---:|---:|---|---|
| 28 | 6300 | `[6300,384] x [384,1152]` | `[6300,384] x [384,2048] -> [6300,1024]` |
| 24 | 5400 | `[5400,384] x [384,1152]` | `[5400,384] x [384,2048] -> [5400,1024]` |

Depth is not a GEMM dimension. A local kernel may be reused at another depth
with identical local shapes after whole-topology validation. Conversely,
`b36c384...` means 36 model layers, not runtime B36.

## 3. Select physical batch before inner-kernel tuning

For each physical candidate `Bp` and logical batch `Bl`, record:

```text
padding rows       = (Bp - Bl) * S
useful fraction    = Bl / Bp
physical NN rows/s = Bp * S * iterations / elapsed
logical NN rows/s  = Bl * S * iterations / elapsed
```

Method:

1. Hold model, board, mask, server lanes, inputs, warmup, and iterations fixed.
2. Separate S1 and S2; two evaluator lanes sharing a GPU change pressure.
3. Use a short scan to eliminate poor batches, then fresh-process balanced
   ABBA (or Latin-balanced equivalent) for finalists.
4. Do not stop a long workload at the report endpoint. Startup and natural
   shutdown distort padding and utilization.
5. Choose on end-to-end logical throughput, not one GEMM microbenchmark.

This generator emits B28 first and B24 second. B40 remains a fail-closed
runtime input from the earlier scan, but is not a selector coordinate and no B40 object is required by
the current bounded set. If v2 selects B20 or B16, add that exact batch through
the same search-space and ABI process; never relabel a B24/B28 object.

## 4. QKV + learned-RoPE

### 4.1 Layout contract

Source Q/K/V each use row-major `[input,output]=[384,384]`. Packing is per
input row:

```text
packed[k,0:384]     = Q[k,:]
packed[k,384:768]   = K[k,:]
packed[k,768:1152]  = V[k,:]
```

| Tensor | Type | Shape / layout |
|---|---|---|
| RMS-normalized input | half | row-major `[M,384]` |
| packed QKV weights | half | row-major `[384,1152]` |
| learned RoPE table | half2 | row-major `[225,192]`, `(cos,sin)` |
| packed output | half | row-major `[M,1152]=[Q384,K384,V384]` per token |

Q and K occupy N128 tiles 0..5; V occupies 6..8. RoPE applies only to Q/K
register fragments. `xy=global_m%225` selects board location. B24 and B28 both
have M128 tails, so the final valid row and predicate-masked rows are mandatory
correctness cases.

Packed output cannot go to generic planar SDPA. Selection is atomic with a
typed same-batch, same-device packed-input FA4 proof. FA4 generation belongs
to the separate FA4 workstream; this branch owns only the generic proof edge.

### 4.2 Bounded QKV search

| Coordinate | Values | Reason |
|---|---|---|
| GEMM tile K | 32,64 | Both divide C384; K depth changes pipeline occupancy. |
| MMA atom layout | `2x2x1`,`4x2x1` | Old CuTe workflow supports both; different register/thread pressure. |
| M/N tile | `128/128` | Audited RoPE fragment mapping assumes N128 and pinned M128 partition. |
| stage policy | pinned CUTLASS auto | Old generator exposes no safe explicit stage override; metadata records `auto`. |
| persistent limit | 170 | One physical 5090 SM wave, matching old QKV scheduler derivation. |

QKV grid340 is initially excluded: retained evidence does not establish it as
a valid two-wave QKV scheduler win on 170 SMs. Reopen after inspecting emitted
scheduler semantics and measuring grid170. M64 is excluded because the exact
RoPE epilogue patch was audited for M128; changing M partition in the first
factorial would obscure failures.

### 4.3 Generation contract

`generate_qkv_rope.py` validates `(batch,id)`, requires pinned clean CUTLASS
and dense-source SHA, applies exact-once transformations, uses inert DLPack
descriptors only for compilation, and emits `.h/.o`, typed bridge, and schema-2
metadata. Metadata states `verified_on_target=false` and makes no speed claim.

The bridge has separate `prepare_v1(device)` and `launch_v1(...)`. Module load
occurs only in prepare. Launch receives exact `tokenRows` and `deviceOrdinal`,
rejects mismatch, and contains no module load, allocation, copy, mutex, or
registry lookup.

## 5. Dual FFN + fused SwiGLU

### 5.1 Math and storage

```text
up   = input @ Wup
gate = input @ Wgate
out  = SiLU(up) * gate
```

`Wup,Wgate` are row-major `[384,1024]`. The paired buffer is `[384,2048]`;
each 64-column block stores `up64,gate64`. For `j`, with `block=j/64` and
`lane=j%64`:

```text
up index   = input*2048 + block*128 + lane
gate index = input*2048 + block*128 + 64 + lane
```

The launch retains an unused separate gate pointer for old bridge-shape
compatibility. A generated kernel ignores it and writes fused `[M,1024]`.

### 5.2 Bounded dual search

| Coordinate | Values | Reason |
|---|---|---|
| M/N tile | `128/128`, effective N64 | Retains audited paired accumulator/epilogue partition. |
| K tile | 32,64 | Both divide C384 and are accepted by the pinned dense constructor. |
| atom layout | `4x2x1` | Retained Stage47 barrier/thread contract. |
| AB / epilogue stages | 2 / 4 | Retained paired-projection contract. |
| persistent limit | 170,340 | Explicit one-wave/two-wave hypotheses. |

M64 is initially excluded because the transformation halves N while retaining
the pinned M128 partition. No retained M64 fused-SwiGLU object proves its
accumulator/predicate map. Reopen only if M128 variants are correct but
under-parallelized.

## 6. Registry, ABI, and preparation

The portable key is `(family, exact batch, exact token rows, candidate id,
registry ABI)`. It excludes weights/names. IDs bind C/S/H/D/F, tile, scheduler,
batch, and native ABI so S361/F1152 artifacts cannot masquerade as S225/F1024.

| Layer | ABI |
|---|---:|
| Registry record | 2 |
| QKV+RoPE launch | 1 |
| Dual FFN launch | 1 |
| Packed FA4 proof | 1 |

The generated-set emitter requires every checked-in coordinate and validates
SHA-256 for each header, object, and bridge before rendering a provider. CMake
recomputes these hashes at configure. Missing, stale, duplicate, or tampered
files fail before build.

Default builds link an empty provider, so absent generated artifacts do not
change production recipes. Later backend integration stores prepared
descriptor pointers and packed weights per block. Runtime mismatch falls back
before enqueue to the already-prepared generic/current-C384 route; it must not
generate, initialize, allocate, or repack on first inference.

CuTe's generated `Kernel_Module_Load` helper is not a production prepare ABI:
it returns `void`, prints instead of propagating failures, and walks every
visible device. The bridge calls the raw init/load-to-device entry points,
checks their explicit `cudaError_t` records, prepares only the requested
ordinal, and publishes the module only after success. Generated ELF `.o` files
also belong directly on the final executable link line. Attaching an
`EXTERNAL_OBJECT` to an object library is insufficient: CMake's
`$<TARGET_OBJECTS:...>` does not propagate that external object. Preserve the
order `generated objects -> CUDA-dialect runtime archive`, and retain a link
fixture that proves the object-island-only form is unresolved while the direct
final-link form builds and executes.

## 7. Correctness gates

### 7.1 CPU/source

- exact `M=B*225`, unique IDs/symbols, schema and ABI;
- no S361/F1152 dual constants or 19x19 metadata copied from reference;
- QKV bridge uses `half2*`, exact rows, and device ordinal;
- launch body has no `Module_Load`, `call_once`, allocation, or copy;
- QKV packing checks plane boundaries;
- dual packing checks columns 63/64 and 1023;
- stale ABI, wrong batch/M/grid/id, duplicate registry, missing artifact, and
  tampered hash all fail closed.

### 7.2 Per-family GPU correctness (future coordinated work)

QKV+RoPE versus planar Q/K/V plus learned-RoPE:

- B24/B28; first row, `xy=224`, next-batch `xy=0`, and last M row;
- Q/K/V planes, all 12 heads, pair 0 and pair 15;
- finite values, max-absolute error, RMSE, and ULP histogram saved;
- packed address normalization verified;
- same-batch FA4 proof required; no generic planar consumer.

Dual versus two GEMMs plus SwiGLU:

- B24/B28 and every K/grid candidate;
- columns 0,63,64,1023 and final M row;
- finite values, max-absolute error, RMSE, and saturation counts;
- unused gate argument tested as null/non-aliasing sentinel;
- packed weights immutable across repeated launches.

Use predeclared tolerances from the existing high-precision raw runner and
record them in the run manifest. Do not choose tolerances after seeing output.
A launch error is fatal; never retry generic work after a candidate may have
partially enqueued output.

### 7.3 Whole model

- same true 36-layer model and deterministic inputs;
- current combined production binary as baseline;
- B24/B28 plus logical partial batches exercising padding;
- policy, value, score, and ownership raw heads;
- markers prove QKV/FA4 atomic activation and exact dual activation;
- INT8 markers absent for C384;
- fallback produces only prepared generic/current-C384 markers.

## 8. Performance ladder

1. **Microbenchmark:** one family/candidate, exact M, launch latency and rows/s;
   include empty-launch/driver overhead.
2. **Family composition:** QKV+RoPE with same-batch FA4; dual independently.
3. **One-block replay:** capture synchronization and layout costs.
4. **Whole-model short screen:** eliminate negative candidates under S1/S2.
5. **Balanced confirmation:** fresh process, fixed warmup/iterations, ABBA,
   endpoint drift and CV.
6. **Padding-aware workload:** report physical and useful throughput.
7. **Long run:** verify clocks, power wall, CPU saturation, padding fraction,
   and no startup/shutdown bias.

Compilation is not a speed result. Correctness, identity, and marker integrity
are hard gates regardless of performance.

### 8.1 Current C384 evidence boundary

The historical C256/H8 exact FA4 path was generated and measured on the target
GPU. The C384/H12 provider and packed-token contract in this branch do **not**
yet include a real B24/B28 object and have not run a GPU correctness or speed
test. Fixed shapes make AOT specialization possible; they do not establish a
winner. Treat QKV layout, learned RoPE, and FA4 as one atomic chain. Its first
qualification must include a raw-output oracle, default-stream (`stream=0`)
launch, eager module-load audit, and balanced ABBA timing. Never route its
packed output to planar SDPA after QKV has enqueued.

The retained B28/S2 Nsight profile reports 25.818% under one shared 128x128
stage-5 kernel symbol. That sample aggregates out-projection and FFN-down calls;
the operations are neither fused nor adjacent, and the percentage must not be
split in half. Attribute them separately only after piece-specific NVTX ranges
or independent ablations. This P0 wiring therefore leaves both kernels intact
and records them as the next fixed-exact profiling target.

The formal S2 milestone is 8000 N/s versus the current B28 6180.58 N/s, but it
is not a stopping condition. After crossing it, continue profiling and bounded
experiments until repeated candidate rounds show no stable positive gain.

## 9. Rejection log

| Route | Status | Reason / reopen condition |
|---|---|---|
| Dynamic-M as C384 mainline | Rejected this phase | User selected exact fixed-batch methodology; dynamic stays prepared fallback. |
| Reusing S361/F1152 objects | Rejected | Wrong M/F and possible OOB; regenerate S225/F1024. |
| TileLang `wide_qkv` as fused QKV+RoPE | Rejected | Planar QKV, no learned-RoPE fusion. |
| QKV grid340 first pass | Deferred | Scheduler meaning lacks retained evidence; revisit after grid170 profile. |
| Dual M64 first pass | Deferred | M128 epilogue mapping audited; M64 is not. |
| First-launch `call_once(Module_Load)` | Rejected | Cold hot-path work and not per-device safe. |
| CuTe `.o` only on an object-library source list | Rejected | External objects do not propagate through `$<TARGET_OBJECTS>`; direct final link is required. |
| Weight/checkpoint SHA dispatch | Rejected | Same structure with retrained weights must reuse kernels. |
| Packed QKV to generic SDPA | Rejected | Layout mismatch; only typed packed FA4 may consume it. |
| Claim speed from generation | Rejected | Compilation is not benchmarking. |

## 10. Generalization checklist

| Change | Reusable | Must regenerate/requalify |
|---|---|---|
| New weights, same structure | Kernel code/tactic IDs | Repack/upload and correctness smoke; no structural search. |
| Depth only | Local kernels | Topology audit/end-to-end timing. |
| Batch only | Generic/dynamic fallback | Exact object/ID/M-tail and outer batch study. |
| Board/sequence | Method | Exact M, RoPE index/table, FA shape, tails, IDs. |
| C | Method | QKV K/N, FFN K, packers, divisibility, family. |
| H or D | FFN if C/F same | Q/K/V widths, RoPE pairs, packed layout, FA proof. |
| F | Attention if C/H/D same | Dual `2F`, output F, pair divisibility, down projection. |
| GQA `Hq!=Hkv` | Some method | Unequal Q/K/V widths invalidate current `3C`/tile assumptions. |
| Mask enabled | Prepared generic | No-mask AOT stays ineligible unless separately keyed. |
| GPU/SM count | Model structural match | Architecture guard, grids, objects, correctness/performance. |

Before merging a winner, confirm structural weight-independent matching, depth
versus batch separation, exact M/layout-bound metadata, construction-time
packing/preparation per device, local pre-enqueue fallback, atomic packed-QKV
consumer coupling, archived hashes/results, and representative padding,
server-lane, CPU, power-wall, startup, and shutdown behavior.

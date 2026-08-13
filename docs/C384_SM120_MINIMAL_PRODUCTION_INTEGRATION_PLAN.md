# C384/SM120 minimal production integration plan

Status: CPU-only design skeleton. No engine source, production package, SSH
host, or GPU was modified while preparing this document.

Base source: `07d9dbb07e77a8cc8c29cd7583eea91fe6bba21a`.

The production default is **B24**. B28 may replace it only after the strict
formal comparison in
`C384_SM120_B24_B28_PRODUCTION_BATCH_POLICY.md` proves a throughput ratio
strictly greater than 1.005 with the required confidence and stability. A
ratio at or below 1.005, an unstable result, or incomplete/non-equivalent
evidence keeps B24. Neither batch may be chosen from an isolated-kernel mean,
a projected N/s value, or a comparison across unrelated binaries.

## 1. Scope and fixed decisions

The production patch is a small extension of the existing exact-AOT path. It
must not replace the generic CUDA implementation. The package is selected only
for the exact Linux x86-64, SM120, FP16/NHWC, no-mask shape:

| Field | Required value |
| --- | --- |
| board / sequence | 15x15 / 225 |
| transformer structure | 36 attention + 36 FFN, alternating |
| channels | 384 |
| attention | H12, KVH12, QD32, VD32, learned RoPE |
| FFN | SwiGLU, F1024 |
| physical batch | B24 by default; B28 only after the formal `> 1.005` promotion gate |
| token rows | `selected_batch * 225` exactly |
| device | compute capability 12.0 exactly |

The four promoted AOT families are:

1. packed QKV projection plus learned RoPE;
2. packed-token FA4 consumer for the same physical batch;
3. dual up/gate projection plus SwiGLU;
4. FFN-down `1024 -> 384` with beta-one in-place residual.

The attention output projection remains cuBLAS beta-one. It is intentionally
not a vendored AOT family: the external screen found all tested out-projection
objects slower than cuBLAS. QKV and FA4 remain an atomic producer/consumer
pair. FFN-down is independent of dual FFN and gets a separate registry/ABI.

INT8, RMSNorm experiments, QKNorm, clip7, TensorRT, NVRTC cleanup, and Windows
artifact generation are outside this patch. They must not be mixed into the
first production integration or its performance attribution.

## 2. Candidate inputs before final batch selection

The following are inputs to the combined sequence, not final engine choices.
The combined gate may retain an incumbent or reject an entire batch.

| Batch | QKV/RoPE | packed FA4 | dual FFN | FFN-down | out projection |
| --- | --- | --- | --- | --- | --- | --- |
| B24 | `...g340-b24-abi1` | `tm128-tn64-s2-w4-both16-packed-token-b24-s225-h12-d32` | `...ab2e3-g340-b24-abi1` | `...m128n128k32s3-natural-b24-abi1` | cuBLAS beta-one |
| B28 | incumbent `...g170-b28-abi1` | incumbent `tm128-tn64-s1-w4-both16-packed-token-b28-s225-h12-d32` | `...ab2e3-g340-b28-abi1` | `...m128n128k32s3-natural-b28-abi1` | cuBLAS beta-one |

The exact full IDs are already bound by the formal Stage2 and residual
reports. The production materializer must read the winning combined report
and copy them; it must not infer IDs from the abbreviated table above.

Relevant immutable evidence at plan time:

- Stage2 formal report SHA256
  `498819a02b8782f30083d276aacf696d6562f70d4fe2d19d37960a7ff7eeb5b6`;
- Stage2 generation manifest SHA256
  `2f288e0e085fc1cb97d6fd2a31e90ad2570c598928c7bc9158c9090a330ef3cb`;
- residual stable-rerank report SHA256
  `ffe32fcb3dd35a012d81a8d5c4483fe783a42132598ba7b393cc7f5e9373d82f`;
- residual registry SHA256
  `dfeeb925ada471429cf1bcabb838d4c523f581d7acad880c697fbf3b5f375348`;
- CuTe CUDA-dialect static runtime SHA256
  `5ffeb7ae24bf02755a6ad751d196b2f1efd18a834b747972ffdd8bbc6550b899`.

Do not make a production package until the combined/full-engine report is
archived and has passed correctness, absolute stability, pairwise stability,
telemetry, and the batch-policy evidence gate. The early same-setting screen
(`B28=6105.38 N/s`, `B24=6086.55 N/s`, about `+0.31%`) therefore leaves B24
as the current default. The standalone best B28 full-engine result
(`6957.434661 N/s`) has no same-version final B24 arm and cannot overturn that
decision.

## 3. Repository package and exact vendor inventory

After the combined result selects one batch, create exactly one directory:

```text
cpp/external/renju15-sm120-c384/
  PRODUCTION_MANIFEST.json
  production_package.cmake
  LICENSES/
  evidence/
    combined_36_report.json
    full_engine_activation.json
    full_engine_correctness.json
    full_engine_long_gate.json
    full_engine_s2_abba.json
  include/
    c384_residual_aot_abi.h
  registry/
    c384_exact_generated_registry.cu
    c384_h12_fa4_registry.cpp
    c384_exact_ffn_down_registry.cu
  artifacts/
    qkv_rope/
      <selected>.h
      <selected>.o
      <selected>_bridge.cu
      <selected>.json
    fa4/
      <selected>.h
      <selected>.o
      <selected>_bridge.cpp
      <selected>.json
    dual_ffn/
      <selected>.h
      <selected>.o
      <selected>_bridge.cu
      <selected>.json
    ffn_down/
      <selected>.h
      <selected>.o
      <selected>_bridge.cu
      <selected>_dense_patch.py
      <selected>.json
    libcuda_dialect_runtime_static.a
```

The FFN-down patched dense source is not compiled by KataGo. It is retained
because the original residual metadata names and hashes it as part of the
generated artifact closure. It is provenance evidence, not a Python runtime
dependency.

Every payload, registry, evidence, header, object, bridge, licence and runtime
asset is listed exactly once in `PRODUCTION_MANIFEST.json` with SHA256 and a
role. The manifest does not hash itself or the CMake driver. The generated
CMake driver embeds the expected manifest hash and the same payload hash list;
the source commit protects the driver itself without creating a cyclic hash
dependency. The manifest also records:

- selected batch and token rows;
- the immutable batch policy (default B24, challenger B28, exclusive 1.005
  ratio threshold), comparison kind, ABBA/BAAB schedule, raw-arm throughput,
  paired ratio, confidence interval, stability metrics, and all comparison
  provenance hashes;
- the four full candidate IDs and exported prepare/query/launch symbols;
- source commit, generator commits/hashes, CUDA/nvcc/ptxas identity, CuTe DSL
  package provenance, SM120 target and Linux x86-64 object format;
- combined-selection evidence and all engine promotion evidence hashes;
- explicit decisions `out_projection=cublas-beta1` and `int8=false`;
- registry ABI versions and the static runtime SHA.

Do not vendor the 40-object search package, the 14-object Stage2 expansion,
rejected candidates, Python/CuTe environments, build directories, absolute
server paths, or raw timing scratch files. Full raw evidence stays under the
reproducibility archive; only the minimal promotion documents needed to audit
the checked-in choice belong in the normal source package.

Licences for NVIDIA CUTLASS DSL and every generator-derived component must be
preserved. The existing B36 package licence set is a useful checklist, but the
new manifest must enumerate the exact licences used by these selected files.

## 4. ABI boundaries

### 4.1 Existing QKV/RoPE and dual-FFN ABI

Keep `C384ExactFixedAot::QkvRopeLaunchFn` and
`C384ExactFixedAot::DualFfnLaunchFn` unchanged. A production registry contains
one descriptor per family and one batch. Preparation occurs once per device
during block construction. The hot path contains no registry scan, module
load, allocation, mutex, weight packing, or pointer discovery.

QKV emits `[rows, Q384|K384|V384]`. It may enqueue only after a typed,
same-batch, same-device packed-token FA4 proof is prepared. Once QKV enqueues,
FA4 failure is fatal; planar fallback is no longer legal for those bytes.

### 4.2 Existing packed FA4 ABI

Keep `C384H12Fa4Sm120::Candidate`, `PreparedProof`, and `LaunchFn` unchanged.
The production registry has `ArtifactMode::Production`, count one, and layout
`PackedTokenQkv`. Compiled accessors, metadata, registry coordinates, proof
coordinates, and selected QKV batch must all agree before the first enqueue.

### 4.3 New FFN-down ABI

Do not add FFN-down to the dual-FFN descriptor. Use a separate header and
registry based on `docs/c384_integration_skeleton/c384_exact_ffn_down_aot.h.in`.
The native generated symbols retain the already-tested signatures:

```cpp
cudaError_t prepare(int deviceOrdinal);
cudaError_t query(C384ResidualRawDescriptorV1* output);
cudaError_t launch(
  const half* activation,             // [M,1024], row major
  const half* rowMajorWeights,        // [1024,384], row major
  half* residualInOut,                // [M,384], C and D alias
  int tokenRows,
  int deviceOrdinal,
  cudaStream_t stream);
```

Required descriptor/query values are ABI 1, family
`FfnDownResidual`, batch equal to the selected physical batch,
`M=batch*225`, K1024, N384, tile M128/N128/K32, three stages, and the exact
candidate ID. The bridge requires 16-byte-aligned non-null pointers, an exact
M, the prepared device as current device, and preparation before launch.

The engine registry wraps those native symbols in a typed immutable tactic.
It calls `query` and `prepare` at construction, compares every query field to
the hash-bound registry, and stores only a prepared pointer. The hot path calls
the launch pointer directly.

### 4.4 Eligibility and fallback semantics

All four AOT families share the existing full-model eligibility proof. The
selected production batch is exact. A tail request, smaller actual batch,
different board/model/layout/dtype/mask/device, missing artifact, registry
mismatch, query mismatch, or preparation failure is rejected before enqueue.

Fallback is complete and coherent:

- QKV/FA4 fall back together to the already-prepared planar QKV and cuDNN
  path;
- dual FFN falls back to the existing implementation;
- FFN-down falls back to cuBLAS Hgemm beta-one;
- out projection always remains cuBLAS Hgemm beta-one.

No AOT launch failure may retry another implementation after enqueue. A
selected launch returning non-success is fatal through `CUDA_ERR`.

## 5. Minimal engine patch map

The final patch should touch only these integration surfaces:

1. `cpp/CMakeLists.txt`
   - enable a repository-relative production package when
     `KATAGO_ENABLE_SM120_TRANSFORMER_WINNER=1`;
   - remove the need for external `KATAGO_C384_*=/root/...` inputs in the
     production configuration;
   - compile the three registries and four bridge sources in the SM120 object
     island;
   - link QKV, dual, FA4, then FFN-down AOT objects directly into `katago`,
     followed by one `libcuda_dialect_runtime_static.a`;
   - verify the complete manifest and SHA set at configure time;
   - keep the generic CUDA/CUTLASS fallback sources compiled.
2. `cpp/neuralnet/c384_exact_ffn_down_aot.h/.cu`
   - typed registry, query validation, construction-time prepare, exact-shape
     support predicate, and empty-provider fallback.
3. `cpp/neuralnet/cudabackend.cpp`
   - add per-FFN-block prepared down selection;
   - call the exact launch with `linear2`'s existing device weight buffer;
   - gate on selected batch, exact token rows, no mask, FP16/NHWC and the
     already-proven 36-layer model identity;
   - preserve cuBLAS beta-one fallback and never route out projection through
     the down object;
   - discard all down selections unless preparation reaches 36/36.
4. `cpp/neuralnet/c384_exact_fixed_aot_*` and
   `cpp/neuralnet/c384_h12_fa4_*`
   - change only package loading/relative manifests and active-count
     telemetry; do not change the launch ABIs or packed layouts.
5. CPU contract tests and the SM120 acceptance runner described below.

Do not merge an external harness registry directly into the engine. The
external registry contains both batches and 24 residual candidates and is a
search artifact. Production registries must collapse to one batch and one
candidate per family.

## 6. CMake direct-link rules

`docs/c384_integration_skeleton/production_package.cmake.in` shows the intended
target boundary. The generated final file must use only paths relative to
`CMAKE_CURRENT_LIST_DIR` and fail closed on:

- wrong package schema/mode/platform/SM/batch;
- missing, extra, duplicated, or SHA-mismatched assets;
- inconsistent IDs, token rows, symbol names, ABI versions, or evidence;
- CUDA older than 13;
- a non-CUDA backend or disabled SM120 winner;
- a package that contains B24 and B28 together.

Generated ELF objects attached as `EXTERNAL_OBJECT` to an object library do
not automatically flow through `$<TARGET_OBJECTS:...>`. Link all four `.o`
files explicitly into the final executable. Keep the support archive after
every AOT object so its underscore-prefixed CUDA adapters resolve correctly.

The package is build-time self-contained with respect to CuTe: Python, CuTe,
and CUTLASS DSL are not required to compile or run KataGo. CUDA 13, cuDNN 9,
and the pinned CUTLASS source checkout remain ordinary build dependencies for
the current fallback implementation. The deployed executable still needs the
audited CUDA/cuDNN/NVRTC loader closure and the host NVIDIA driver.

## 7. 36/36 telemetry contract

Preparation and execution are separate claims. Require both markers:

```text
KATAGO_C384_EXACT_FIXED_PREPARED batch=<B> qkv_fa4=36/36 dual_ffn=36/36 ffn_down=36/36 out_proj=cublas-beta1
KATAGO_C384_EXACT_FIXED_ACTIVE batch=<B> qkv_fa4=36/36 dual_ffn=36/36 ffn_down=36/36 out_proj=cublas-beta1
```

Each block owns a once-only counted flag for its exact family. The ACTIVE
marker is emitted only after every one of the 36 attention blocks has run the
exact QKV/FA4 pair and every one of the 36 FFN blocks has run exact dual and
down. A single first-use family log is useful diagnostics but is not the
36/36 proof.

Fallback runs must never print the ACTIVE marker. Their PREPARED marker must
state the observed count and `fallback`. The marker contains the four full
candidate IDs elsewhere in adjacent, once-only per-family diagnostics.

## 8. Test and promotion checklist

### CPU-only / configure-time

- compile and run selector tests for exact batch, neighboring batch, tail
  rows, wrong model depth, wrong block order/count, C/F/H/D/S mismatch,
  FP32/NCHW/masked paths, SM mismatch and device mismatch;
- test malformed/empty/duplicate registries, null function pointers, ABI
  drift, query-field drift, selected-ID mismatch and prepare failure;
- test that down accepts exactly K1024/N384/M=B*225 and cannot be selected for
  out projection K384/N384;
- test 36/36 commit/discard and ACTIVE count behavior;
- test production package schema, relative paths, exact file-set coverage,
  SHA tampering, extra files, wrong runtime archive, B24+B28 rejection and
  server-absolute-path rejection;
- build with no package and prove the stub/generic path still links;
- build non-CUDA backends and prove the package is neither parsed nor linked;
- inspect the final link command: four AOT objects precede the single static
  runtime archive.

### Fresh Linux build / binary audit

- fresh clone, empty build directory, network disabled, no search tree or
  CuTe Python environment visible;
- CUDA 13 exact compiler identity and pinned CUTLASS commit checks pass;
- `readelf -d`, `nm -D --undefined-only`, `ldd`, `file`, and `cuobjdump` audit
  are archived;
- final binary and CMake cache contain no benchmark-server paths;
- all selected bridge/object symbols are present once and rejected-candidate
  symbols are absent;
- move/delete the generator environment and run the binary again.

### GPU correctness before timing

- stream 0 and nonblocking-stream oracles for each family using the same
  thresholds as the external gates;
- exact QKV-to-packed-FA4 chain oracle, not isolated QKV only;
- FFN-down FP32-output plus original-residual oracle with C/D aliasing and
  canaries;
- full 36-layer activation smoke with the exact production model and 36/36
  PREPARED/ACTIVE markers;
- generic-vs-exact deterministic model replay, including policy/value/score
  outputs, finite checks, configured tolerances, and a preserved corpus;
- long correctness/stability gate under the intended two-server-lane S2
  configuration.

### Explicit fallback matrix

- actual tail batch below the fixed physical batch;
- the other candidate batch (B24 package under B28 and vice versa);
- B1, wrong board, wrong C/F/H/D, wrong depth, mask present, FP32, NCHW;
- SM not equal to 120 (where a compatible generic CUDA binary is available);
- injected registry/query/prepare rejection before enqueue;
- missing production package at build time.

Every case must complete through the generic path with no packed-layout
reinterpretation, no partial exact family, and no exact ACTIVE marker.

### Performance

- real full-engine S2 ABBA against the same fresh generic/native baseline;
- select B28 only by
  `C384_SM120_B24_B28_PRODUCTION_BATCH_POLICY.md`; specifically, a microbench,
  the old 36x3 QKV/FA4/dual upper-bound projection, a fastest single leg, or
  an old-B24/new-B28 cross-binary comparison is ineligible;
- fixed production batch only, equal warmup and iteration counts, GPU
  exclusivity/telemetry, raw child logs, binary SHA and configuration SHA;
- report per-server median, batch wall time, N/s, CV, endpoint drift and paired
  ratios;
- retain Nsight Systems kernel attribution for QKV, FA4, dual, down, out,
  RMSNorm and other residual time;
- do not promote a projection or external hotpath N/s as full-engine N/s.

## 9. Production materialization order

1. Validate and archive a qualified full-engine or proven-equivalent complete
   combined comparison.
2. Apply the batch policy: keep B24 unless every B28 promotion predicate is
   true. Choose exactly one batch from that result; fill no other source
   manually.
3. Copy the four selected artifact closures and one runtime archive to a fresh
   integration-candidate staging directory; re-hash every source and
   destination. This staging package is not yet allowed to claim production
   promotion.
4. Emit one exact registry, one FA4 registry, one down registry, a relative
   CMake package and a complete production manifest.
5. Run all CPU negative fixtures against the staged package.
6. Commit the candidate package and minimal engine patch on an integration
   branch.
7. Fresh-build on the benchmark host with no external AOT paths.
8. Pass activation, fallback, correctness, long-gate and S2 ABBA tests.
9. Materialize a new immutable `PRODUCTION` package revision containing the
   engine evidence hashes; never relabel the earlier candidate manifest or
   edit a signed manifest in place.
10. Repeat the fresh build and final binary audit from that immutable commit.

## 10. Reusing the method for another model size

Treat a model size as a new exact contract, not a numeric substitution in a
candidate ID:

1. derive operator shapes and data layouts from the native exporter/parser;
2. establish a correct generic native baseline and a real full-engine metric;
3. move each hotspot to an engine-external harness with exact stream/layout,
   FP32 oracles, canaries, preconditioning and position-balanced rounds;
4. search bounded coordinates independently for QKV/attention, dual FFN,
   residual down and RMSNorm; do not combine distant operations merely to
   reduce launch count;
5. keep producer/consumer dependencies together, especially packed QKV/FA4;
6. freeze only stable winners, then run the true layer-count ordered sequence
   with cross-layer state dependency;
7. default to the smaller fixed batch and promote a larger batch only through
   a predeclared full-engine/proven-equivalent paired threshold; never use
   isolated kernel throughput or projected N/s;
8. vendor one selected artifact per family with typed ABI, complete hashes,
   construction-time preparation and pre-enqueue fallback;
9. require full block-count PREPARED and ACTIVE markers;
10. validate correctness, tails, long stability and real engine throughput
    before calling the package production-ready.

Changing C, F, S, H, D, layout, accumulation, mask semantics, weight packing,
RoPE/QKNorm order, or clip behavior invalidates the corresponding artifact and
its oracle. Changing only model depth may reuse a per-operator object when all
operator shapes are identical, but it still requires a new model-structure
gate, block-count marker and true-depth sequence/full-engine validation.

# C384 B28 production integration scaffold

Status: source/compile-contract integration. This change intentionally vendors
no winner object and performs no GPU or SSH qualification.

## Authoritative base

The implementation branch starts at
`a47b099df71c8d98face9414b134df13e2c1a7a0`. Git ancestry was checked rather
than inferred:

- `205ebbff7c6159146148eceb96bbd7bea41f44fb` is an ancestor of
  `07d9dbb07e77a8cc8c29cd7583eea91fe6bba21a`;
- `07d9dbb0` is an ancestor of `a47b099d`.

The base therefore contains the combined production engine, the final exact
QKV/FA4/dual hot path, and the B28-only integration decision. The existing
exact code is extended; it is not reimplemented.

## Fixed production route

Production exact AOT is eligible only for the true 36-attention/36-FFN,
C384/H12/D32/F1024/S225 model at physical B28/M6300, SM120, FP16, NHWC,
learned RoPE, SwiGLU, and no mask. `productionBatchEligible` is the shared
QKV/dual production gate; FFN-down independently encodes the same B28/M6300
contract. B24 remains search/diagnostic evidence but is generic CUDA in the
engine.

The production sequence is:

1. exact packed QKV+RoPE followed by the already-typed packed-token FA4 proof;
2. cuBLAS Hgemm beta-one attention output projection;
3. exact dual up/gate plus SwiGLU;
4. independent typed exact FFN-down beta-one residual.

FFN-down is not added to `C384ExactFixedAot::RegistryView`. It owns
`C384ResidualRawDescriptorV1`, `C384ExactFfnDownAot::Tactic`, and
`generatedTactics` in separate files.

## Preparation and launch transaction

Every FFN block publishes its dual and down selections together only after:

- the dual descriptor and generated module prepare successfully;
- the down registry contains exactly one B28 descriptor;
- down `query` reports ABI 1, family `FfnDownResidual`, B28/M6300,
  K1024/N384, M128/N128/K32, three mainloop stages, atom 2/2/1, four
  epilogue stages, natural/launch grid 150/150, no cluster cap, and the exact
  registry ID;
- down `prepare(deviceOrdinal)` succeeds;
- packed dual weights upload succeeds.

The model transaction becomes reachable only when QKV+FA4, dual, and down are
all prepared `36/36`. Any smaller count discards all exact block selections.
The active marker is emitted only after all three families have executed
`36/36` distinct blocks. Attention output projection is reported explicitly as
`cublas-hgemm-beta1`.

The FFN hot path performs one common preflight for dual and down before dual
enqueues. It checks the complete B28 shape, device ownership, dtype/layout/mask,
exact rows, all three pointers, and 16-byte alignment. A miss uses the already
prepared generic dual path and cuBLAS down path. Once exact dual enqueues, exact
down launch failure is fatal through `CUDA_ERR`; it never retries cuBLAS.
QKV/FA4 keeps the same rule: a consumer failure after packed QKV enqueue is a
fatal contract violation, never planar fallback.

Non-B28 requests, tails, other model shapes, devices, layouts, dtypes, masks,
missing/empty providers, bad registry/query metadata, or preparation failures
all resolve before any exact-family enqueue to coherent generic CUDA. The
normal build's provider is deliberately empty.

## Generated FFN-down package interface

Set `KATAGO_C384_EXACT_FFN_DOWN_PACKAGE_CMAKE` only after a B28 winner is
materialized. The included file must define:

```cmake
set(KATAGO_C384_FFN_DOWN_PACKAGE_SCHEMA 1)
set(KATAGO_C384_FFN_DOWN_PACKAGE_MODE PRODUCTION)
set(KATAGO_C384_FFN_DOWN_SELECTED_BATCH 28)
set(KATAGO_C384_FFN_DOWN_SELECTED_ID
  c384-s225-residual-ffn-down-m6300-k1024-n384-m128n128k32s3-natural-b28-abi1)
set(KATAGO_C384_FFN_DOWN_TACTIC_IDS
  ${KATAGO_C384_FFN_DOWN_SELECTED_ID})
set(KATAGO_C384_FFN_DOWN_REGISTRY_PROVIDER <generated-registry.cu>)
set(KATAGO_C384_FFN_DOWN_GENERATED_INCLUDE_DIR <include-directory>)
set(KATAGO_C384_FFN_DOWN_GENERATED_HEADERS <one-header>)
set(KATAGO_C384_FFN_DOWN_GENERATED_METADATA <one-metadata-file>)
set(KATAGO_C384_FFN_DOWN_GENERATED_BRIDGES <one-bridge.cu>)
set(KATAGO_C384_FFN_DOWN_GENERATED_OBJECTS <one-object.o>)
set(KATAGO_C384_FFN_DOWN_GENERATED_FILE_SHA256
  <registry>;<sha>;<header>;<sha>;<metadata>;<sha>;<bridge>;<sha>;<object>;<sha>)
```

The single generated header is the selected candidate header. The package must
not copy or redefine `c384_residual_aot_abi.h`: generated bridges resolve that
basename from the engine's canonical `cpp/neuralnet` include root. Both the
candidate header and bridge remain package-hash-bound, while the ABI is bound
to the engine source commit.

The generated provider implements only
`C384ExactFfnDownAot::generatedTactics`. It does not implement fallback and
does not modify the QKV/dual registry. CMake verifies complete asset hash
coverage and links the external object directly into the final executable.
Supplying a down production package requires the existing exact QKV/dual and
packed FA4 packages at production B28 in the same configure. A production
QKV/FA4 package without down is rejected as incomplete. Search-pair packages
remain usable without down and cannot become production-active.

## Local contracts

The no-GPU selector/provider contracts are:

```text
python -m unittest python.tests.test_c384_b28_production_policy
cmake -S cpp/tests/c384_exact_ffn_down_compile_fixture -B <contract-build>
cmake --build <contract-build> --config Release
<contract-build>/Release/c384_exact_ffn_down_contract
<contract-build>/Release/c384_exact_ffn_down_generated_provider_contract
```

When local cuDNN headers are available, configure the same fixture with
`-DKATAGO_CUDNN_INCLUDE_DIR=<cudnn-include>` and explicitly build
`c384_cudabackend_compile_contract`. This compile-only target covers the real
production integration translation unit with all exact-family macros enabled;
it does not link or vendor a generated winner object.

The C++ contracts cover both the normal empty provider and a synthetic
generated-provider replacement, plus B28 acceptance;
B24/tail/layout/mask/device/alignment
pre-enqueue rejection; family, exact-M and candidate-ID query mismatch; query
grid mismatch; query and prepare failure; one-entry production collapse; and
the empty-provider disabled state. Full completion still requires CUDA
13/SM120 compilation with real generated assets, correctness, stability,
`36/36` activation, and full-engine S2 ABBA qualification.

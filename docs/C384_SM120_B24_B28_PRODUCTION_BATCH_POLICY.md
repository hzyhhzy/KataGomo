# C384/SM120 fixed B28 production batch policy

Status: CPU-only integration contract. This document does not modify the
KataGo engine, a server, or a GPU.

The user has fixed the production batch to B28:

```text
production_batch = 28
selection_reason = user-directed-fixed-batch
```

This is an explicit configuration decision, not a benchmark promotion. There
is no benchmark-based selector or automatic batch negotiation in this policy.

## 1. Runtime routing

The exact-AOT route is eligible only when every production shape and runtime
predicate matches, including physical batch 28 and token rows 6300. The B28
route contains the selected QKV/learned-RoPE, packed FA4, dual FFN, and
FFN-down artifacts; attention output projection remains cuBLAS beta-one.

Every non-B28 request uses the coherent generic CUDA path selected before any
AOT enqueue. This includes B24, tails, smaller/larger batches, and otherwise
eligible C384 requests whose physical batch is not 28. No partial exact route,
packed-layout reinterpretation, or post-enqueue retry is permitted.

```text
if full_model_eligible && physical_batch == 28 && token_rows == 6300:
    use prepared B28 exact-AOT route
else:
    use generic CUDA route
```

Correctness, preparation, stability, telemetry, and 36/36 activation remain
hard production gates for B28. If B28 fails one of those gates, production
promotion fails closed; the materializer must not silently switch the fixed
production declaration to B24.

## 2. B24 status

B24 is retained only as diagnostic and compatibility evidence:

- its historical reports and exact artifacts remain in the reproducibility
  archive;
- it may be run in external diagnostics to detect regressions or understand
  batch geometry;
- engine requests at B24 must run the generic CUDA fallback;
- no B24 AOT artifact is included or active in the final B28 production
  package;
- B24 performance, correctness, stability, or availability does not block B28
  production acceptance and cannot promote or demote a batch.

The existing B28 `6957.434661 N/s` record remains useful performance evidence,
but it is not the reason for the selection. The selection reason is the user's
fixed-batch directive. Any future batch change requires a new explicit policy
and immutable production manifest revision.

## 3. Manifest contract

The production manifest must state:

```json
{
  "production_batch": 28,
  "token_rows": 6300,
  "selection_reason": "user-directed-fixed-batch",
  "aot_batches": [28],
  "b24_status": "diagnostic-and-generic-fallback-only",
  "non_b28_policy": "pre-enqueue-generic-cuda-fallback",
  "dynamic_batch_selection": false
}
```

It binds one B28 candidate ID and artifact closure for each exact family. The
manifest and generated registries must not contain a B24 exact descriptor,
object, bridge, symbol, or candidate ID. B24 evidence may be referenced as an
external diagnostic archive but is not part of the production payload closure.

## 4. Acceptance tests

Required positive B28 tests:

- exact batch 28 / M6300 / S225 / C384 / H12 / D32 / F1024 eligibility;
- construction-time query and preparation for all selected B28 artifacts;
- stream-0 and nonblocking-stream correctness oracles, canaries, residual
  aliasing, packed QKV-to-FA4 compatibility, and deterministic engine replay;
- full 36-layer PREPARED and ACTIVE markers showing QKV/FA4, dual FFN, and
  FFN-down at 36/36, with out projection reported as cuBLAS beta-one;
- long stability/telemetry gate and real integrated full-engine performance
  report using the final binary and configuration hashes.

Required fallback and negative tests:

- B24 selects generic CUDA before enqueue and emits no exact ACTIVE marker;
- B1, B27, B29, tails and every other non-B28 batch select generic CUDA;
- wrong board, sequence, C/F/H/D, model depth/order, dtype, layout, mask, SM,
  device, registry/query fields, preparation state, or candidate ID selects
  generic CUDA before enqueue where fallback is safe;
- a B24 object, descriptor, symbol, candidate ID, or active registry entry in
  the production package is rejected;
- a production manifest with `production_batch` other than 28, token rows
  other than 6300, or a different/missing selection reason is rejected;
- dynamic benchmark-based selection or environment/CLI batch overrides are
  rejected by the production package contract;
- a B28 launch failure after enqueue is fatal rather than retried through the
  generic route.

CPU package tests additionally verify exact file-set hashes, relative paths,
one B28 descriptor per family, no B24 payload, and the generic no-package build.
The final Linux audit verifies only the B28 AOT symbols are linked and that B24
and all other batches remain available through the generic CUDA implementation.

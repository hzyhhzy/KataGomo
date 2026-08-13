# C384 exact fixed-batch AOT search overlay

This branch is a construction-time integration skeleton for the true
36-layer C384/H12/KV12/D32/F1024 model. It intentionally does **not** replace
the combined production C384 dynamic kernels or change their plan recipes.
The checked-in registry provider is empty, so ordinary builds retain the
existing prepared generic/dynamic paths and their markers and fingerprints.

## Fixed search coordinates

Runtime batch and model depth are separate quantities. The model has 36
attention and 36 FFN descriptors. The only AOT runtime batches in this search
are, in priority order:

| priority | physical batch | exact token rows (`B * 225`) |
| ---: | ---: | ---: |
| 1 | 28 | 6300 |
| 2 | 24 | 5400 |
| 3 | 40 | 9000 |

The selector validates a coherent alternating transformer structure and marks
36 attention + 36 FFN blocks as the primary target. Model depth is not a kernel
dimension, however: after that structural audit a winning local operator may
still be reused by a coherent 32- or 48-layer model with the same local shape.
Whole-model policy remains responsible for deciding whether to enable padding
to one of the fixed physical batches.

## ABI and fail-closed rules

The portable selector lives in `cpp/neuralnet/c384_exact_fixed_aot_plan.*`.
CUDA descriptors and the launch ABI live in
`cpp/neuralnet/c384_exact_fixed_aot_kernels.h`.

- QKV+RoPE is exact `[B*225,384] x [384,1152]`; weights are row-major with
  stride `(1152,1)` rather than three planar matrices. It uses a
  `[225,192] half2(cos,sin)` learned-RoPE table and packed per-token
  `[Q384,K384,V384]` output.
- Packed QKV is selected only with a typed immutable proof for an exact
  packed-input FA4 of the **same** batch. The proof locks FA4 ABI, geometry,
  packed-input layout, device ordinal, implementation cookie, and candidate identity. A
  missing, stale, or different-batch proof rejects the whole QKV+FA4 pair
  before any enqueue; packed QKV must never fall through to planar cuDNN
  attention.
- Dual FFN is independent. Its paired weights hold 64-channel up/gate chunks
  for C384/F1024, and its search coordinates include grid 170 and grid 340.
- Every launch is exact-M. A partial/tail row count, a non-candidate batch,
  mask, FP32, NCHW, another board, or a non-SM120 device misses the overlay.
- Registry lookup and selection happen while each block is constructed. Each
  generated descriptor must expose a non-null, idempotent per-device eager
  prepare hook; construction calls it before publishing the descriptor. The
  eventual hot path stores immutable descriptor pointers and performs no
  lookup, allocation, module load, weight packing, or kernel initialization.
- Model names, model bytes, checkpoint hashes, and weight hashes are absent
  from all keys.

FA4 artifacts and the atomic QKV+FA4 launch wrapper are intentionally outside
this change and are owned by the separate FA4 workstream.

## Generated registry injection

With the SM120 winner enabled, CMake always builds the selector bridge and one
registry provider. By default it uses the checked-in empty provider:

```text
cpp/neuralnet/c384_exact_fixed_aot_registry_stub.cu
```

A search build replaces only that provider:

```text
-DKATAGO_C384_EXACT_AOT_REGISTRY_SOURCE=/absolute/path/generated_registry.cu
```

The generated source must define both provider functions declared by
`c384_exact_fixed_aot_kernels.h`. It may return an empty family while searching
the other family. Records are keyed by exact batch and explicit candidate ID;
duplicates and artifact manifests are validated by the generator workflow,
not selected heuristically at runtime.

Current candidate geometry:

- QKV+RoPE: `M128 N128 K64`, atom `4x2x1`, packed output.
- dual FFN: `M128 N64x2 K32`, AB stages 2, epilogue stages 4,
  paired weights, grids 170 and 340.

The reference workflow is the clean `doomoooo/KataGomo_fork` mirror at
`38a99ee43252f4f6e8979d2f3944bfed55ce7f7a`. No CUDA generation or GPU run was
performed while creating this skeleton.

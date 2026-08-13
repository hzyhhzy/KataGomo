# C384 exact fixed-batch AOT search overlay

This branch is a construction-time integration skeleton for the true
36-layer C384/H12/KV12/D32/F1024 model. It intentionally does **not** replace
the combined production C384 dynamic kernels or change their plan recipes.
The checked-in registry provider is empty, so ordinary builds retain the
existing prepared generic/dynamic paths and their markers and fingerprints.

## Fixed search coordinates

Runtime batch and model depth are separate quantities. The model has 36
attention and 36 FFN descriptors. The portable selector retains the earlier
runtime coordinates below. The current bounded generator emits objects only
for B28 and B24. B40 is not a selector coordinate and stays fail-closed.

| priority | physical batch | exact token rows (`B * 225`) |
| ---: | ---: | ---: |
| 1 | 28 | 6300 |
| 2 | 24 | 5400 |

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
- Every launch receives and rechecks exact token rows. A partial/tail row
  count, a non-candidate batch, mask, FP32, NCHW, another board, or a
  non-SM120 device misses the overlay.
- Registry lookup and selection happen while each block is constructed. Each
  generated descriptor must expose a non-null, idempotent per-device eager
  prepare hook; construction calls it before publishing the descriptor. The
  eventual hot path stores immutable descriptor pointers and performs no
  lookup, allocation, module load, weight packing, or kernel initialization.
- Eager prepare bypasses CuTe's generated `void Kernel_Module_Load` convenience
  helper, because that helper only prints failures and iterates over every
  visible device. The bridge calls the raw init/load-to-device ABI, propagates
  its `cudaError_t`, loads only the requested device ordinal, and never
  publishes a failed module. A failed ordinal remains sticky and fail-closed.
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

A generated build is either `search-pair` or `production`. In both modes it
contains exactly one QKV+RoPE artifact and one dual-FFN artifact for the same
fixed batch. A bounded search therefore builds each QKV/dual coordinate pair
separately; it does not link a fat registry and choose heuristically at
runtime. Emit a search pair with:

```text
python python/c384_exact_fixed_aot/emit_registry.py \
  --mode search-pair \
  --metadata /path/qkv.json --qkv-id <qkv-id> \
  --metadata /path/dual.json --dual-id <dual-id> \
  --output-dir /path/generated-pair

-DKATAGO_C384_EXACT_AOT_GENERATED_MANIFEST=/absolute/path/c384_exact_generated_manifest.cmake
```

The manifest is the source of truth for selected IDs. The cache variables
`KATAGO_C384_EXACT_QKV_TACTIC_ID` and
`KATAGO_C384_EXACT_DUAL_FFN_TACTIC_ID` are optional expectations: when omitted,
CMake derives the effective compile definitions from the manifest; when set,
they must match it exactly. Family-specific ID lists, selected batch, two
headers, two objects, two bridges, two metadata files, and every SHA-256 are
validated before the target is created.

Artifact verification follows the actual CuTe export boundary: the generated
header must contain the unique inline wrapper and raw ABI declarations, the
ELF object must contain the unique raw runtime and launch symbols, and the
bridge must reference both. Metadata binds the top-level generator, bridge
emitter, imported CUTLASS DSL source/version, and loaded CUDA bindings binary;
visible distribution metadata must describe those exact imported versions.

`production` mode additionally requires a hash-bound promotion JSON with
`activation`, `long_gate`, and `accuracy` gates all marked `PASSED`, and must be
paired with a same-batch FA4 `PRODUCTION` package. Search pairs require a FA4
`BATCH_SEARCH` package. Any exact manifest, selected-ID expectation, or FA4
package supplied to a non-CUDA backend, or while the SM120 winner is disabled,
is a configure-time error rather than a silent fallback.

Current bounded geometry (at B28 and B24):

- QKV+RoPE: `M128 N128 K{32,64}` x atom `{2x2x1,4x2x1}`,
  pinned-CUTLASS automatic stages, grid 170, packed output.
- dual FFN: `M128 N64x2 K{32,64}`, atom `4x2x1`, AB stages 2,
  epilogue stages 4, paired weights, grids 170 and 340.

The reusable derivation, measurement ladder, candidate rejection log, and
generalization checklist are recorded in
`docs/SM120_FIXED_BATCH_TRANSFORMER_OPTIMIZATION_PLAYBOOK.md`.

The reference workflow is the clean `doomoooo/KataGomo_fork` mirror at
`38a99ee43252f4f6e8979d2f3944bfed55ce7f7a`. No CUDA generation or GPU run was
performed while creating this skeleton.

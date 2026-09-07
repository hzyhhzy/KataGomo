# B11 FA4 AOT build tools

Standalone, optional, **CPU-only generation** for the official Go Transformer's
inner attention: FP16 Q/K/V/O, H12, D32, S361, noncausal full attention, exact
or key-mask mode. QK and PV accumulation are FP16, with the validated FP16 cast
after online-softmax output rescaling. The default schema-1 format keeps batch
dynamic. The optional schema-2 bundle fixes B13/B16 and packed-QKV metadata;
it contains four independently compiled exact/mask objects. Neither format
changes the attention math. Dynamic tile M/N and stage count remain explicit
generation parameters; the static production bundle fixes M128/N96/S1.

This directory has no runtime dependency on an experiment folder, the reference
Doom repository, or a remote host path. The scripts never allocate CUDA tensors,
query a GPU, or execute a kernel. They hide all GPU devices, describe types with
inert DLPack metadata, and compile with a fake stream and explicit SM120 target.
All generated artifacts/caches use an explicitly supplied output directory.

## Dependencies and isolated preparation

The tested toolchain is Linux x86-64, Python 3.12.13, CUDA toolkit 13.0.48 and
the package versions in `requirements-build.txt`. `dependencies.json` records
exact package versions plus both the original validated byte hashes and
canonical-LF hashes of the four numerical implementation sources. A newline
normalization does not change the algorithm, but its actual byte hash is still
recorded in every artifact. Development/nightly wheels may require the original
wheel archive/index; the script does not install or silently replace packages.

Use an existing matching build environment, or install into a **new build-only
environment**, not the engine runtime. The CUDA toolkit is a separate prerequisite:

```sh
# Set these to explicit paths of your choice, outside the repository root.
TOOLS=/absolute/path/to/KataGo/cpp/neuralnet/b11aot
DEPS=/absolute/path/to/external-build-data/fa4-dependencies
OUT=/absolute/path/to/external-build-data/fa4-exact-m128-n96-s1
PYTHON=/absolute/path/to/build-environment/bin/python
CUDA=/usr/local/cuda-13.0

"$PYTHON" "$TOOLS/prepare_dependencies.py" --output-dir "$DEPS"
```

Preparation reads the installed `flash-attn-4` distribution metadata; optionally
specify `--source-package /path/to/flash_attn`. It verifies the pinned source,
copies the package into the new/empty output directory, applies the small FP16
accumulator patch, and supplies the pinned SM120 adapter. It never edits
site-packages, an existing nonempty output, or the input package. Existing
validated both16 input is accepted idempotently. `preparation.json` records
source identities.

The complete both16 patch changes QK/PV MMA accumulator types and matching
register tensors, gives mask negative-infinity literals the accumulator dtype,
and casts softmax exponentials before storing them into the accumulator tensor.
Constructor defaults remain FP32; generation selects FP16 explicitly. The
separate softmax-rescale cast is applied only in the compiler process. All patch
sites and post-patch source hashes are checked. The original readable diff is
preserved in `patches/flash-attention-sm120-both16.patch`; preparation applies
those changes without an external patch utility. This is not a transformation
of another generator.

## Generate exact or masked AOT

```sh
"$PYTHON" "$TOOLS/generate_fa4.py" --mode exact --dependency-root "$DEPS" \
  --output-dir "$OUT" --cuda-root "$CUDA" --tile-m 128 --tile-n 96 --num-stages 1

# Use a DIFFERENT new/empty output directory for the mask variant.
"$PYTHON" "$TOOLS/generate_fa4.py" --mode mask --dependency-root "$DEPS" \
  --output-dir /absolute/path/to/external-build-data/fa4-mask-m128-n96-s1 \
  --cuda-root "$CUDA" --tile-m 128 --tile-n 96 --num-stages 1
```

Other CPU-compiled candidate shapes include M128/N64/S1, M128/N128/S1,
M64/N128/S1, M128/N128/S2, M128/N192/S1. Compilation is not a claim of accuracy
or speed; validate chosen artifacts against the engine's unchanged generic
path. N192 previously incurred stack usage and is only a candidate, not a
recommended default. M128/N96/S1 is the retained production tile; batch size
was selected jointly with whole-network throughput.

The default probe batch is 16 and the runtime guard is B1..96. Change
`--probe-batch` or `--max-batch` explicitly if needed. `--symbol-prefix` allows
multiple objects in one binary. The generator refuses unexpected dependency
source hashes. `--allow-toolchain-mismatch` is only for deliberate compiler
migration experiments, records version differences, and does not bypass source
identity checks. Generation writes `.o`, `.h`, `_api.h`, `_bridge.cpp`, `.json`,
PTX/cubin diagnostic artifacts, and licenses only below `--output-dir`.

## Link into the C++ engine

The emitted object is a **Linux x86-64 ELF host object containing SM120 CUDA
code**. Do not feed this binary to MSVC. The engine's optional integration uses
the retained M128/N96/S1 exact and mask pair; other generator shapes are for
separate experiments, not silently interchangeable production artifacts.

### Static metadata bundle (schema 2)

Generate four packed token-major profiles together. This removes runtime shape
and stride fields from the tensor ABI; data pointers, softmax scale and CUDA
stream remain runtime arguments. It uses the same pinned dependency overlay,
FP16 arithmetic, mask, tile and safe current-device loader as schema 1; it does
not import or transform an experimental generator.

```sh
ROOT=/absolute/path/to/KataGo
TOOLS="$ROOT/cpp/neuralnet/b11aot"
PYTHON=/absolute/path/to/matching-build-environment/bin/python
DEPS=/absolute/path/to/external-build-data/fa4-dependencies
AOT=/absolute/path/to/external-build-data/fa4-static-b13-b16
CUDA=/usr/local/cuda-13.0
BUILD=/absolute/path/to/external-build-data/engine-build

# Prepare DEPS once using the earlier command, or reuse an intact pinned overlay.
"$PYTHON" "$TOOLS/generate_static_bundle.py" --dependency-root "$DEPS" \
  --output-dir "$AOT" --cuda-root "$CUDA"
cmake -S "$ROOT/cpp" -B "$BUILD" -DUSE_BACKEND=CUDA \
  -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA" \
  -DKATAGO_B11_FA4_AOT_DIR="$AOT"
cmake --build "$BUILD" --parallel 4
```

The explicit output root must be new or empty. Its layout is:

```text
fa4-static-b13-b16/
  bundle.json
  b13/exact/b11_fa4_static_b13_exact_m128_n96_s1{.o,.h,.json,_api.h,_bridge.cpp}
  b13/mask/b11_fa4_static_b13_mask_m128_n96_s1{.o,.h,.json,_api.h,_bridge.cpp}
  b16/exact/b11_fa4_static_b16_exact_m128_n96_s1{.o,.h,.json,_api.h,_bridge.cpp}
  b16/mask/b11_fa4_static_b16_mask_m128_n96_s1{.o,.h,.json,_api.h,_bridge.cpp}
```

Each directory also retains PTX/cubin, provenance and licenses. The bundle lists
four manifest hashes; each manifest hashes its object/header/API/safe bridge.
CMake checks this chain, fixed B, S361/H12/D32, M128/N96/S1, FP16 QK/PV/I/O,
packed stride 1152/output stride 384, pointer-only tensor ABI and ELF64 x86-64
identity. Missing or mismatched entries are errors, not silent partial bundles.
These checks establish consistency, not authenticity of untrusted downloads.

`static_fa4_dispatch.cpp` preserves the existing two engine-level exact/mask
entry points and routes B13/B16 to their objects. Both the bridges and
`static_fa4_profile.h` reject other B or unpacked inputs. The engine must gate
model/GPU/geometry/batch/packedness **before** launch and fall back to the generic
backend when unsupported; an error return is not itself a fallback. Schema 2
defines `USE_B11_STATIC_FA4`. Setup prepares both supported batches on the
handle's current device; hot dispatch does no setup/device query, locks or
allocation. Do not link both formats under the same engine symbols at once.

The fixed-metadata method was first GPU-tested in a research binary at B13/B16,
exact and masked modes. Those earlier measurements were research validation;
the separate production acceptance is recorded below. Revalidate newly
generated objects/builds with real-corpus numerics and serialized GPU timing.
Masked mode is still dense S361 attention, not token pruning.

### Production integration status and numerical scope

With a schema-2 bundle selected at build time, the matching B11/RTX5090/19x19
FP16-NHWC B13/B16 profile now selects static FA4 and the local tanh-based SwiGLU
functor by default. Omitting the optional bundle still builds the generic
backend; this is not an unconditional change to every model or GPU. Unsupported
actual batches passed directly to the backend use the original FFN. Normal
queued evaluation now defaults to Doom-style coalescing and physical batch
padding for eligible handles (`nnBatchAwareDispatch=auto`); see
[`B11_RTX5090.md`](../B11_RTX5090.md#queued-evaluation-doom-style-batching).
That scheduling policy does not change the AOT ABI or require regenerating
the bundle. Planar QKV bypasses static attention, while
the independently eligible FFN may still use the B11 functor. The original
generic CUTLASS implementation, probe and packed weight layout are unchanged.
`cudaB11TanhFFN=false` disables the arithmetic variant without disabling FA4.

The local tanh functor preserves each FP16 rounding boundary but approximates
sigmoid differently from the original exponential-based functor. It is **not**
bitwise equivalent to the original model computation. Corpus results are
numerical measurements, not proof of identical playing strength or loss.

Production validation completed on 2026-09-07 for the official
`kata1-tf3-b11c768-s11500M-d6163M` model and an RTX5090, using CUDA 13.0.48 and
cuDNN 9.14. The Linux build retained the normal broad CUDA architecture list;
all four generated cubins **and PTX** were byte-identical to the validated
research artifacts. The tested pure-forward benchmark SHA256 is
`316b883a625923acc2fa74ae75419f46f4720fcb7ca7f8f8f14854c1eb152b7c`.

Separate integration checks covered B13/B16 exact, fullmask and real partial
mask inputs, numerical replay against the original FP16 backend, changing mask
and actual batch on an existing handle, planar-QKV fallback, unsupported
model/board/precision and master-off cases. The isolated no-AOT backend/link
diagnostic matched the generic reference; it was not a second complete CMake
build. B13/S3 and B16/S3 GTP smoke tests each passed 17 commands, including
19-to-13-to-19 board changes. Only RTX5090 received GPU validation; compiling
other CUDA targets does not replace runtime testing on those devices.

Final timing used 24 serial processes: eight explicitly selected batch/stream
profiles, three reverse-alternating repeats of 1000 iterations each, preuploaded
whole-network eager forward with all output heads. Setup, transfers, search and
output postprocessing were outside timing. The independently audited medians
were:

| Mode | Production B/S | Production nnEval/s | Doom B/S | Doom nnEval/s | Same-run gain |
|---|---|---:|---|---:|---:|
| exact | 13/3 | 4316.631144 | 13/3 | 4016.978059 | 7.4597% |
| fullmask | 13/3 | 4317.039038 | 16/2 | 2515.789284 | 71.5978% |
| partial | 13/3 | 4419.244752 | 13/3 | 2639.680413 | 67.4159% |

Doom exact used its retained optimized N96/both16 FA4 path. Its fullmask and
partial controls used the **generic masked backend**, not optimized masked
FA4; their B/S selections came from an earlier version and were freshly timed,
not proved globally optimal. Production fullmask means the `exact=false` API
with all 361 points valid, which automatically dispatches the exact FA4 kernel.
Partial input mixes 19x19, 13x13 and 9x9 valid regions in dense S361 storage.

Production exact B16/S3 measured 4294.903600 nnEval/s and Doom B24/S2 measured
3998.027870. B13/S3 led this run, but its approximately 0.51% advantage over
production B16/S3 is small. Clocks were not locked: production B13 exact fell
3.49% from the first to the last repeat, while Doom B13 fell 2.63%. These are
same-run medians, not portable speed guarantees or a universal optimum. The
earlier research screen's **8.06%** exact improvement remains a historical
result, not the final production measurement. There is no fixed percentage
acceptance threshold. Unintegrated fusion microbenchmarks are not included in
these production results, and neither timing nor numerical replay proves
unchanged dataset loss or playing strength.

### Dynamic pair (schema 1, retained)

Use the same CMake option `KATAGO_B11_FA4_AOT_DIR` to select an external
artifact root with this layout:

```text
artifacts/
  exact/b11_fa4_exact_m128_n96_s1{.o,.h,.json,_api.h,_bridge.cpp}
  mask/b11_fa4_mask_m128_n96_s1{.o,.h,.json,_api.h,_bridge.cpp}
```

The complete generation/build recipe (all output directories are external):

```sh
ROOT=/absolute/path/to/KataGo
TOOLS="$ROOT/cpp/neuralnet/b11aot"
DEPS=/absolute/path/to/external-build-data/fa4-dependencies
AOT=/absolute/path/to/external-build-data/artifacts
BUILD=/absolute/path/to/external-build-data/engine-build
PYTHON=/absolute/path/to/matching-build-environment/bin/python
CUDA=/usr/local/cuda-13.0

# Run once in a new/empty overlay, if not already prepared above.
"$PYTHON" "$TOOLS/prepare_dependencies.py" --output-dir "$DEPS"
for mode in exact mask; do
  "$PYTHON" "$TOOLS/generate_fa4.py" --mode "$mode" \
    --dependency-root "$DEPS" --output-dir "$AOT/$mode" --cuda-root "$CUDA" \
    --tile-m 128 --tile-n 96 --num-stages 1 --max-batch 96
done

cmake -S "$ROOT/cpp" -B "$BUILD" -DUSE_BACKEND=CUDA \
  -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA" \
  -DKATAGO_B11_FA4_AOT_DIR="$AOT"
cmake --build "$BUILD" --parallel 4
```

The normal engine build dependencies, including cuDNN, still apply. Add their
normal CMake search paths if they are not installed in discoverable locations.
This optional integration requires Linux x86-64, CMake 3.19+, and CUDA toolkit
13.0+. Generation remains pinned to the tested versions listed above. CMake
validates each manifest's shape, precision, runtime batch range, symbol prefix,
and hashes of the object, generated headers and safe bridge. An intact bundle
may be relocated; recorded absolute provenance paths are not runtime inputs.

Omit `KATAGO_B11_FA4_AOT_DIR` (the empty default) to build without FA4 artifacts;
generic CUDA attention remains available. This option does not narrow the
engine's normal CUDA architecture list. An intentionally RTX5090-only local
build may separately specify `-DKATAGO_CUDA_ARCHITECTURES=120`; omit that override
when distributing a build that must retain support for other GPUs. Unsupported
runtime models/devices/shapes use the engine's unchanged generic path.

The final engine requires neither Python, PyTorch, FA4, nor `libcute_dsl_runtime`.
CMake compiles the selected generated safe bridges and this directory's
`cuda_runtime_abi.cpp` **once**, and links the selected AOT objects (two for
schema 1, four for schema 2) with CUDA runtime/driver libraries. The earlier
duplicate `cudab11aot.cpp` bridge is no longer used.
Neither configuration nor engine compilation invokes the Python generator,
downloads dependencies, or runs kernels.

Every generated `_api.h` exposes a unique `prefix_prepare()` and uniform
`prefix_launch(q,k,v,out,mask,batch,scale,packedQKV,stream)`. Exact requires null
mask; masked requires a nonnull aligned FP16 `[B,361]` key mask. The engine's
existing masked-query/output handling must remain intact. Schema 1 inputs may
be planar or token-packed QKV; schema 2 requires token-packed QKV. Output is
contiguous `[B,361,12,32]` in both formats.

Call prepare for the selected variant on each consuming handle's current
device, after structural/GPU/shape gating and before warmup, timing or graph
capture. The supplied loader checks only the current RTX5090/CC12.0 device,
does not enumerate/switch devices, and propagates setup failures. Hot launch
does no setup/locks/queries; launch only after successful preparation.
Unsupported conditions and fallback remain the engine's responsibility.

## CPU checks

```sh
"$PYTHON" "$TOOLS/test_cpu.py"
"$PYTHON" "$TOOLS/test_artifacts.py" --output-dir /absolute/external/cpu-tests \
  --cmake cmake --cxx g++
g++ -std=c++14 -pthread "$TOOLS/aot_loader_test.cpp" -o /absolute/output/loader-test
/absolute/output/loader-test
```

The Python test uses only the standard library and can optionally validate an
actual base patch with `--source-package`. The C++ loader test injects a fake
runtime and tests rejection, concurrency, error caching and explicit cleanup.
Artifact tests invoke CMake without a CUDA toolchain to exercise missing files,
tampered object/manifest hashes, wrong batch/packedness/runtime bounds/ABI/ELF,
relocation and schema-1 compatibility. `--cxx` optionally compiles the real static
dispatcher and rendered bridges against a fake runtime, executing B=-1..97,
null/misaligned inputs, packedness and prepare-error propagation. This uses no
CUDA headers, library or context; on Windows it may use `--cxx cl` from a VS
developer prompt. Test fixtures remain below the explicit output for diagnostics.
Real-driver preparation, numerical outputs,
mask/exact equivalence where appropriate, and steady-state speed still require
separate serialized GPU validation.

See `NOTICE.md` and `licenses/` for attribution and redistribution requirements.

## Recorded schema-1 offline reproduction (historical)

The standalone package prepared a fresh overlay from the pinned base package
and CPU-compiled both N96/M128/S1 modes. Their GPU cubins are byte-identical to
the previously numerically validated controls, despite distinct host prefixes:

- Exact cubin SHA256:
  `118b87adfe5dc6a8736e62775423610d0636ed160fd9ab994910c6c0c67e07aa`.
- Mask cubin SHA256:
  `83a9a4501a20d4a905edef41c652cf0b88911cb535248fb97a0f525c5be7bb30`.

For this historical schema-1 pair, the generated Q/K/V/O ABI has a pointer,
four runtime int32 shape values and three runtime int64 strides; mask adds a
pointer, two shapes and one stride. This matches its validated controls and
retains runtime B. It is not the schema-2 pointer-only fixed-metadata tensor
ABI described above. CPU tests, the fake loader test, C++ bridge compilation
and strict undefined-symbol linking passed. No GPU execution was part of this
offline packaging verification.

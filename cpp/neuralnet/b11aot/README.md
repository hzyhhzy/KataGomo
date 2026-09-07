# B11 FA4 AOT build tools

Standalone, optional, **CPU-only generation** for the official Go Transformer's
inner attention: FP16 Q/K/V/O, H12, D32, S361, noncausal full attention, exact
or key-mask mode. QK and PV accumulation are FP16, with the validated FP16 cast
after online-softmax output rescaling. Batch is a runtime argument, not compiled
to the probe batch. Tile M/N and stage count are explicit generation parameters.

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
recommended default. N96 is a convenient default while the final winner is
selected jointly with batch size.

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

Use the single CMake option `KATAGO_B11_FA4_AOT_DIR` to select an external
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
CMake compiles the two generated safe bridges and this directory's
`cuda_runtime_abi.cpp` **once**, and links both AOT objects with CUDA runtime/driver
libraries. The earlier duplicate `cudab11aot.cpp` bridge is no longer used.
Neither configuration nor engine compilation invokes the Python generator,
downloads dependencies, or runs kernels.

Every generated `_api.h` exposes a unique `prefix_prepare()` and uniform
`prefix_launch(q,k,v,out,mask,batch,scale,packedQKV,stream)`. Exact requires null
mask; masked requires a nonnull aligned FP16 `[B,361]` key mask. The engine's
existing masked-query/output handling must remain intact. Inputs may be planar
or token-packed QKV; output is contiguous `[B,361,12,32]`.

Call prepare for the selected variant on each consuming handle's current
device, after structural/GPU/shape gating and before warmup, timing or graph
capture. The supplied loader checks only the current RTX5090/CC12.0 device,
does not enumerate/switch devices, and propagates setup failures. Hot launch
does no setup/locks/queries; launch only after successful preparation.
Unsupported conditions and fallback remain the engine's responsibility.

## CPU checks

```sh
"$PYTHON" "$TOOLS/test_cpu.py"
g++ -std=c++14 -pthread "$TOOLS/aot_loader_test.cpp" -o /absolute/output/loader-test
/absolute/output/loader-test
```

The Python test uses only the standard library and can optionally validate an
actual base patch with `--source-package`. The C++ loader test injects a fake
runtime and tests rejection, concurrency, error caching and explicit cleanup.
Neither test needs a CUDA context. Real-driver preparation, numerical outputs,
mask/exact equivalence where appropriate, and steady-state speed still require
separate serialized GPU validation.

See `NOTICE.md` and `licenses/` for attribution and redistribution requirements.

## Recorded offline reproduction

The standalone package prepared a fresh overlay from the pinned base package
and CPU-compiled both N96/M128/S1 modes. Their GPU cubins are byte-identical to
the previously numerically validated controls, despite distinct host prefixes:

- Exact cubin SHA256:
  `118b87adfe5dc6a8736e62775423610d0636ed160fd9ab994910c6c0c67e07aa`.
- Mask cubin SHA256:
  `83a9a4501a20d4a905edef41c652cf0b88911cb535248fb97a0f525c5be7bb30`.

The generated Q/K/V/O ABI has a pointer, four runtime int32 shape values and
three runtime int64 strides; mask adds a pointer, two shapes and one stride.
This matches the validated controls and retains runtime B. CPU tests, the fake
loader test, C++ bridge compilation and strict undefined-symbol linking passed.
No GPU execution was part of this packaging verification.

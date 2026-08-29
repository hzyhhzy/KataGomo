# Ataxx 7x7 CPU-PTQ backend

`CPU-PTQ` is a deliberately specialized batch-1 backend for Intel Ice Lake
server CPUs. It requires AVX-512F/DQ/BW/VL and AVX-512 VNNI, consumes native
model version 206, and implements the Ataxx v11 input contract on an unmasked
7x7 board. Native v205 is the matching FP32 staging format and is parsed for
validation, but is deliberately rejected for inference.

Supported profiles are kept in `model.cpp` and selected from model metadata:

| profile | blocks | channels | heads | head dim | FFN |
|---|---:|---:|---:|---:|---:|
| `b11c96h3-f256` | 11 | 96 | 3 | 32 | 256 |
| `b16c128h4-f384` | 16 | 128 | 4 | 32 | 384 |

All transformer projection matrices are stored canonically as signed INT8
codes plus one FP32 scale per output channel. `@S7P@` declares qmax 63 and
`@S8P@` declares qmax 127; one model may not mix them. The loader repacks both
forms into the same VNNI microkernel layout, so v206 files do not depend on the
current tile size.
The convolutional stem, normalization parameters, and required policy/value
head weights remain FP32. Activations are quantized per token at runtime and
executed as U8 x S8 `VPDPBUSD` dot products with zero-point correction.

Build from the repository root with Clang 8 or newer. GCC also compiles the
backend, but is not recommended for release binaries: GCC 12 generated a
substantially slower VNNI kernel in the Ice Lake measurements for this backend.

```sh
cmake -S cpp -B runtime_data/build-cpuptq \
  -DUSE_BACKEND=CPU-PTQ -DNO_GIT_REVISION=1 \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Release
cmake --build runtime_data/build-cpuptq -j
```

The source and resulting binary have been built on Ubuntu 16.04/glibc 2.23.
The complete `CPU-PTQ` target is compiled with `-march=icelake-server`, so the
binary is intentionally not portable to a non-Ice-Lake-compatible runtime.
Build-only validation may be done on another x86-64 host, but every execution,
including `cpuptqbench -check-model-only`, requires the declared ISA and OS ZMM
state.

Export a model with the unified script in the `KataGo_Transformer` repository
after producing a matching GPTQ manifest:

```sh
python train/export_cpu_ptq.py \
  --checkpoint CHECKPOINT.ckpt --manifest MANIFEST.npz \
  --output MODEL-v206.bin.gz --base-output MODEL-v205.bin.gz \
  --model-name MODEL_NAME --pos-len 7
```

The same script also accepts `--source MODEL-v205.bin.gz`; its v105/v106 path
serves source-v102 models. The wire body is the normal KataGo native schema,
with only the seven Transformer projections per block changed from `@BIN@` to
`@S7P@`/`@S8P@`. The file itself does not store a board size.

For direct validation and timing without search:

```sh
runtime_data/build-cpuptq/katago cpuptqbench \
  -model MODEL-v206.bin.gz -input INPUTS.f32 -warmup 100 -iters 3000
```

To validate only the native schema and profile without executing AVX-512:

```sh
runtime_data/build-cpuptq/katago cpuptqbench \
  -model MODEL-v206.bin.gz -check-model-only
```

Adding a profile requires adding its geometry to `ProfileSpec`, teaching the
exporter to accept the same geometry, and validating both v206 output loss and
Ice Lake latency. Geometry, board, batch, mask, and ISA mismatches are outside
this backend's contract and fail immediately.

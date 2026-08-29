# Ataxx 7x7 CPU-PTQ backend

`CPU-PTQ` is a deliberately specialized batch-1 backend for Intel Ice Lake
server CPUs. It requires AVX-512F/DQ/BW/VL and AVX-512 VNNI, consumes model
version 206, and implements the Ataxx v11 input contract on an unmasked 7x7
board.

Supported profiles are kept in `model.cpp` and selected from model metadata:

| profile | blocks | channels | heads | head dim | FFN |
|---|---:|---:|---:|---:|---:|
| `b11c96h3-f256` | 11 | 96 | 3 | 32 | 256 |
| `b16c128h4-f384` | 16 | 128 | 4 | 32 | 384 |

All transformer projection matrices are stored canonically as symmetric S8
codes plus one FP32 scale per output channel. The loader repacks them into the
VNNI microkernel layout, so v206 files do not depend on the current tile size.
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
A host without the required ISA may compile the binary but must not run it.

Export a model after producing a matching GPTQ manifest:

```sh
python python/export_cpu_ptq_v206.py \
  --checkpoint CHECKPOINT.ckpt --manifest MANIFEST.npz \
  --output MODEL-v206.bin.gz --model-name MODEL_NAME
```

For direct validation and timing without search:

```sh
runtime_data/build-cpuptq/katago cpuptqbench \
  -model MODEL-v206.bin.gz -input INPUTS.f32 -warmup 100 -iters 3000
```

Adding a profile requires adding its geometry to `ProfileSpec`, teaching the
exporter to accept the same geometry, and validating both v206 output loss and
Ice Lake latency. Geometry, board, batch, mask, and ISA mismatches are outside
this backend's contract and fail immediately.

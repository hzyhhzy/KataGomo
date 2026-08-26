# CPU-PTQ single-thread backend

`CPU-PTQ` is a deliberately narrow inference backend for 15x15, batch-one
transformer models on 64-bit Windows CPUs with AVX-512 VNNI. It is called
through KataGo's ordinary C++ `NeuralNet` interface and loads native
`.bin.gz` models; ONNX Runtime is used only as a benchmark baseline.

## Compiled profiles and hard boundaries

| profile | transformer blocks | trunk | heads | head dim | FFN | value hidden |
|---|---:|---:|---:|---:|---:|---:|
| `b11c96h3-f256` | 11 | 96 | 3 | 32 | 256 | 64 |
| `b16c128h4-f384` | 16 | 128 | 4 | 32 | 384 | 96 |

Both profiles require all of the following:

- native model version 106 with Inputs101 (`22` spatial and `39` global
  features);
- exactly 15x15, physical batch size one, one backend thread, and exact NN
  dimensions;
- Q/K RMSNorm with epsilon `1e-6`, RoPE, and clipped SwiGLU with clip `4`;
- the fixed KataGo policy/value head geometry used by these two models;
- AVX-512F, DQ, BW, VL, VNNI, and matching OS XSAVE state.

Boundary failures call `assert(false)` in assertion builds and also throw a
descriptive exception for release builds. There is intentionally no fallback
to another model size, board size, batch size, or instruction set.

The v106 wire grammar itself retains v105's optional QKN, arbitrary
nonnegative clip, and fixed/learned/no-RoPE encodings. That keeps the format
extensible for future profiles. The two kernels compiled today intentionally
accept only the qkn+clip4 cases above. Both fixed RoPE (`tfrs`) and learned RoPE
(`tflrs`) are implemented.

## v106 model format

Version 106 keeps v105's topology and PTQ metadata semantics, but changes the
payload of every Transformer Q/K/V/O and FFN up/gate/down matrix. Each of
those matrices is stored as `@S8P@`, followed by one little-endian positive
FP32 scale per output channel and symmetric S8 weights in output-major order
(input channels contiguous). `-128` is forbidden. Quantization uses round to
nearest, ties to even, and saturation to `[-127,127]`.

This canonical representation is not an MLAS packed-B buffer, so it is stable
across MLAS versions and target CPUs. The loader only transposes and packs the
stored S8 bytes; it no longer retains or quantizes an FP32 projection master.
Non-Transformer weights remain ordinary native FP32 blocks. The format is a
fail-closed CPU deployment format, not an ONNX container.

Convert an exported native v105 model without overwriting the source:

```text
python python/convert_cpu_ptq_v106.py model-v105.bin.gz model-cpuptq-v106.bin.gz
```

The converter rejects ONNX, non-v105 inputs, and geometries outside the two
compiled profiles. It parses the complete source, quantizes all seven matrices
per Transformer layer, writes deterministic gzip (`mtime=0`) through an atomic
temporary file, then reparses and verifies the complete v106 body.

For the current fixtures, the storage change is:

| profile | v105 FP32 `.bin.gz` | v106 mostly-S8 `.bin.gz` | reduction |
|---|---:|---:|---:|
| b11c96 | 4,697,655 bytes | 1,370,480 bytes | 70.8% |
| b16c128 | 12,718,213 bytes | 3,721,126 bytes | 70.7% |

In the uncompressed v106 bodies, canonical S8 projection bytes plus their
scales account for about 86.8% (b11) and 92.9% (b16) of all bytes.

## What is INT8 and what remains FP32

For every transformer block, the seven conceptual dense projections are INT8
both on disk and during inference:
Q/K/V are concatenated into one GEMM, FFN up/gate are concatenated into one
GEMM, and attention output plus FFN down are separate GEMMs. Each call uses:

- one dynamic asymmetric U8 scale/zero point for the complete activation
  matrix;
- symmetric S8 weights with one scale per output channel;
- S32 accumulation with per-channel FP32 dequantization;
- fused QKN/RoPE, SwiGLU, residual add, and residual RMSNorm output processors.

Attention has two additional VNNI paths. Q/K uses dynamically quantized
32-element rows and `VPDPBUSD` with the signed-key correction. P/V quantizes
softmax probabilities to U8 and values to per-channel S8, accumulates in S32,
then applies the probability-sum and value scales.

The following stay FP32: stem convolution, global projection, all policy and
value heads, RMSNorm arithmetic, RoPE arithmetic, softmax/exp, residual state,
clipped SwiGLU arithmetic, and final outputs. Stem and heads use XNNPACK;
vector post-ops and attention use hand-written AVX-512 code.

## Building

The qualified build uses MSVC x64, XNNPACK static libraries from a matching
PyTorch wheel, and MLAS source from ONNX Runtime v1.23.0. The ORT checkout is
not modified: CMake copies the two exact MLAS files into the build tree,
applies the small panel/scratch patch there, and fails configuration if an ORT
upgrade no longer matches the audited anchors.

Example PowerShell configuration:

```powershell
cmake -S cpp -B cpp/build/cpuptq -G Ninja `
  -DUSE_BACKEND=CPU-PTQ `
  -DCPU_PTQ_TORCH_ROOT=C:/path/to/site-packages/torch `
  -DCPU_PTQ_ONNXRUNTIME_SOURCE_ROOT=C:/src/onnxruntime-1.23.0 `
  -DCPU_PTQ_MLAS_PACKED_STRIDE_N=768 `
  -DCPU_PTQ_MLAS_PACKED_STRIDE_M=48
cmake --build cpp/build/cpuptq --config Release
```

`768x48` is the common panel qualified on AMD EPYC 9K65. Scans of N
`512/640/768/1024` and M `32/48/64` on both compiled profiles found 768x48 to
be the smallest panel covering the maximum projection width and the best
non-noise compromise. Changing the values regenerates only the patched MLAS
translation unit and relinks the engine.

`KATAGO_BUILD_NNRAWGATE=ON` adds the test-only `nnrawgate` command. Leave it
off for an ordinary engine build.

## GTP use

Use the backend like an ordinary KataGo build, but keep its strict limits in
the config:

```text
katago.exe gtp -model model-cpuptq-v106.bin.gz -config gtp_cpuptq.cfg
```

At minimum, the config must resolve to one NN server thread, batch one, exact
15x15 NN buffers, and NCHW engine inputs. The backend logs the selected profile
at model load.

## EPYC 9K65 validation snapshot

Tests used one pinned logical processor on an AMD EPYC 9K65 Windows SA9 VM.
ORT used 1.23.0, sequential execution, one intra-op/inter-op thread,
preallocated OrtValues, 100 warmups, and 1000 rotating timed calls. The native
backend timing replays 4096 batch-one rows after one warmup and excludes model
load and output-file I/O.

| model / runner | latency per row | relative to ORT FP32 |
|---|---:|---:|
| real b11 ORT FP32 | 6.7675 ms | 1.00x |
| real b11 ORT dynamic INT8 | 5.4539 ms | 1.24x |
| real b11 CPU-PTQ, mostly-S8 v106 | 2.7084 ms | 2.50x |
| qkn+clip4 b16 ORT FP32 | 17.5121 ms | 1.00x |
| qkn+clip4 b16 ORT dynamic INT8 per-channel | 13.1914 ms | 1.33x |
| b16 CPU-PTQ | about 6.09 ms | about 2.88x |

The current mostly-S8 b11 entry is the average of two processor-affinity-pinned
4096-row passes (`2.70949` and `2.70740 ms`); an independent 18,432-row labeled
corpus pass measured `2.70846 ms`. Against ORT dynamic INT8 this is `2.01x`.
The b16 entry is the earlier average of two equivalent-quantization 4096-row
passes (`6.0887 ms`) and remains `2.17x` faster than its ORT dynamic baseline.

The mostly-S8 real b11 model also completed an actual GTP startup and one
`genmove` through this backend, returning `L8`. A generated v102-ABI b16 fixture
using the earlier v106 development container did the same.

For the real b11 checkpoint, all 77 serialized scale bit patterns and S8 code
arrays were checked bit-for-bit against the SWA checkpoint recipe. Two fresh,
independent 18,432-row validation simulations then produced identical
per-sample hashes: FP32 policy/value loss was `1.69447912/0.57124859`, and the
complete quantization recipe was `1.70306604/0.57714096`.

The actual C++ backend then replayed all 18,432 labeled rows at physical batch
one. Its policy/value loss was `1.70340036/0.57708396`, a delta from FP32 of
`+0.00892124/+0.00583537`. The difference between actual C++ and the
quantization-only simulator was only `+0.00033433/-0.00005700`. The older ORT
dynamic trunk delta was `+0.01345666/+0.01458691`.

An independent 32-row C++ raw-output gate for the mostly-S8 container produced
policy/value cosine similarity of `0.999014/0.999893` against the quantization
simulator, covering the native parser, RoPE, packed kernels, fused post-ops,
and head mapping. The new loader no longer derives those codes from an FP32
master.

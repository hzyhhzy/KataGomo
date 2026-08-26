# CPU-PTQ fixed-shape single-thread backend

`CPU-PTQ` is a deliberately narrow KataGo inference backend for 15x15,
batch-one Transformer models. It is called through the ordinary C++
`NeuralNet` interface, loads native `.bin.gz` files, and has two separately
compiled implementations:

- strict AVX2/FMA3, using saturation-safe S7 projection weights;
- AVX-512 VNNI, using full-range S8 projection weights.

ONNX Runtime is not part of model loading or inference. It was used for the
FP32 and dynamic-INT8 baselines. The backend builds the standalone MLAS
kernels from an ONNX Runtime source checkout and uses XNNPACK libraries from a
matching PyTorch wheel where described below.

## Compiled profiles and hard boundaries

| profile | Transformer layers | wire blocks | trunk | heads | head dim | FFN | value hidden |
|---|---:|---:|---:|---:|---:|---:|---:|
| `b11c96h3-f256` | 11 | 22 | 96 | 3 | 32 | 256 | 64 |
| `b16c128h4-f384` | 16 | 32 | 128 | 4 | 32 | 384 | 96 |
| `b24c192h6-f512` | 24 | 48 | 192 | 6 | 32 | 512 | 96 |

All profiles require:

- native model version 106 and Inputs101 (`22` spatial plus `39` global
  features);
- exactly 15x15, all 225 locations valid (no padding mask), physical batch
  size one, one NN server thread, and NCHW engine input;
- exact dimensions from the table and the fixed KataGo policy/value head
  geometry;
- Q/K RMSNorm with epsilon `1e-6`, RoPE, and clipped SwiGLU with clip `4`;
- either fixed RoPE (`tfrs`) or learned RoPE (`tflrs`).

The AVX2 binary additionally requires SSE4.1, SSE4.2, POPCNT, AVX, AVX2,
FMA3, and AVX XSAVE state. The VNNI binary requires AVX-512F, DQ, BW, VL,
VNNI, and matching OS XSAVE state.

Unsupported models, board sizes, batch sizes, thread counts, layouts, and
instruction sets are intentional hard failures. Assertion builds call
`assert(false)`; release builds also throw a descriptive exception. There is
no generic fallback in this backend.

The v106 grammar remains compatible with v105's optional QKN, arbitrary
nonnegative SwiGLU clip, and fixed/learned/no-RoPE encodings so future profiles
can be added without another container version. The three compiled kernels
accept only the qkn+clip4 boundary above.

## v106 model format

Version 106 keeps v105 topology and PTQ metadata, but replaces every
Transformer Q/K/V/O and FFN up/gate/down FP32 matrix payload with one of two
canonical records:

| marker | declared code range | intended backend |
|---|---:|---|
| `@S7P@` | `[-63,63]` | AVX2 A8W7 |
| `@S8P@` | `[-127,127]` | AVX-512 VNNI U8S8 |

The marker is followed by one positive little-endian FP32 scale per output
channel, then output-major signed-byte codes with input channels contiguous.
Quantization is symmetric per output channel, round-to-nearest ties-to-even,
and saturating. A code outside the marker's declared range is rejected.

This is not an MLAS packed-B buffer. The stable canonical bytes can be loaded
across MLAS versions and target CPUs; the loader validates, transposes, and
packs them for the selected kernel. It does not retain or reconstruct an FP32
master for these projections. Non-Transformer weights remain native FP32.

Convert a native v105 model without overwriting the source:

```text
python python/convert_cpu_ptq_v106.py model-v105.bin.gz model-s8-v106.bin.gz --projection-bits 8
python python/convert_cpu_ptq_v106.py model-v105.bin.gz model-s7-v106.bin.gz --projection-bits 7
```

The converter rejects ONNX, non-v105 inputs, and unsupported geometries. It
parses the complete source, quantizes all seven matrices per layer, writes
deterministic gzip (`mtime=0`) through an atomic temporary file, then reparses
and validates the complete v106 body.

Qualified fixture sizes are:

| profile | v105 FP32 | v106 S8 | v106 S7 |
|---|---:|---:|---:|
| b11c96 | 4,697,655 B | 1,370,480 B | 1,235,644 B |
| b16c128 | 12,718,213 B | 3,721,126 B | 3,320,432 B |
| b24c192 | 39,239,296 B | 11,115,787 B | 9,837,276 B |

## Numerical paths

The seven dense projections per Transformer layer are quantized both on disk
and in execution. Q/K/V and FFN up/gate are fused into two wider GEMMs;
attention output and FFN down remain separate. S32 results are dequantized
with per-output scales while QKN/RoPE, clipped SwiGLU, residual addition, and
residual RMSNorm are fused into output processors.

The VNNI projection path dynamically quantizes each projection input matrix to
U8 with one min/max range and uses S8 weights with `VPDPBUSD`. Its attention
path dynamically quantizes each Q/K token/head row, computes QK with
`VPDPBUSD` plus signed-key correction, keeps softmax in FP32, then uses U8
probabilities and per-channel S8 values for PV. Four-query PV fusion reuses
each packed value load.

The AVX2 projection path uses A8W7: U8 activations and S7 weights avoid
`VPMADDUBSW`'s signed-16 intermediate saturation before widening to S32.
Attention uses the independently saturation-safe Q7K8 and P8V7 recipe.
QKV/RoPE, SwiGLU, residual/RMSNorm, packing, and attention are hand-vectorized
with AVX2/FMA3.

RMSNorm arithmetic, RoPE arithmetic, softmax/exp, residual state, clipped
SwiGLU arithmetic, stem/global projection, policy/value heads, and final
outputs remain FP32. In the VNNI build the FP32 stem and heads use XNNPACK. In
the strict AVX2 build they use packed MLAS FP32 SGEMM (including an im2col
stem), and MLAS runtime feature selection is capped below AVX-VNNI/AVX-512.

## Building

The qualified build is MSVC x64 with:

- XNNPACK static libraries and headers from a matching PyTorch wheel;
- MLAS sources from ONNX Runtime v1.23.0;
- Release optimization and one selected `CPU_PTQ_ISA` per executable.

CMake copies the two patched MLAS files into the build tree, checks audited
source anchors, and leaves the ORT checkout unchanged.

Example from an x64 Visual Studio developer shell:

```powershell
$common = @(
  '-DUSE_BACKEND=CPU-PTQ',
  '-DCPU_PTQ_TORCH_ROOT=C:/path/to/site-packages/torch',
  '-DCPU_PTQ_ONNXRUNTIME_SOURCE_ROOT=C:/src/onnxruntime-1.23.0'
)

cmake -S cpp -B runtime_data/build-cpuptq-avx2 -G Ninja @common `
  -DCPU_PTQ_ISA=AVX2
cmake --build runtime_data/build-cpuptq-avx2 --config Release

cmake -S cpp -B runtime_data/build-cpuptq-vnni -G Ninja @common `
  -DCPU_PTQ_ISA=AVX512VNNI
cmake --build runtime_data/build-cpuptq-vnni --config Release
```

Qualified panel defaults are selected by ISA:

| setting | AVX2 | VNNI |
|---|---:|---:|
| packed K | 512 | 384 |
| ordinary packed N | 768 | 768 |
| large packed N | 1024 | 1024 |
| large-N threshold | 1024 | 1024 |
| packed M | 48 | 48 |

They can be changed with
`CPU_PTQ_MLAS_PACKED_STRIDE_K/N/N_LARGE/N_LARGE_THRESHOLD/M`. The accepted
AVX2 defaults use `CPU_PTQ_AVX2_PROJECTION_RECIPE=A8W7` and
`CPU_PTQ_AVX2_ATTENTION_RECIPE=Q7K8_P8V7`. VNNI four-query PV fusion is
controlled by `CPU_PTQ_VNNI_PV_QUAD` and is enabled by default.

`KATAGO_BUILD_NNRAWGATE=ON` adds the test-only `nnrawgate` command. Leave it
off for a tournament engine. `KATAGO_BUILD_V105_WIRE_CONTRACT=ON` adds
`testv105wire` and its custom target.

## GTP use

```text
katago.exe gtp -model model-v106.bin.gz -config gtp_cpuptq.cfg
```

The config must resolve to one NN server thread, batch one, exact 15x15 NN
buffers, and NCHW input. The selected profile is logged at load. All three
profiles have passed GTP startup with both final ISA binaries.

## EPYC 9K65 validation

Formal timing used one process, one backend thread, one pinned logical CPU
(CPU 3, affinity mask 8) on an AMD EPYC 9K65 Windows SA9 VM. Model load,
warmup, and output-file I/O are excluded. Each result below is the mean of two
paired passes with profiling disabled.

| profile | AVX2/FMA3 | AVX-512 VNNI | VNNI speedup over AVX2 |
|---|---:|---:|---:|
| b11c96 | 5.8593 ms | 2.6428 ms | 2.22x |
| b16c128 | 13.6211 ms | 6.0329 ms | 2.26x |
| b24c192 | 36.0854 ms | 15.1436 ms | 2.38x |

The VM exposes two physical cores and four logical processors. A final
contention test used logical CPU 3 for one process, one hardware thread from
each physical core for two processes (masks `0x1,0x4`), and all SMT threads
for four processes:

| profile / ISA | 1-process aggregate | 2-process aggregate (scale) | 4-process aggregate (scale) |
|---|---:|---:|---:|
| b11 AVX2 | 170.81 rows/s | 331.93 (1.94x) | 376.04 (2.20x) |
| b16 AVX2 | 73.29 rows/s | 133.99 (1.83x) | 161.44 (2.20x) |
| b24 AVX2 | 27.77 rows/s | 55.06 (1.98x) | 61.37 (2.21x) |
| b11 VNNI | 389.40 rows/s | 689.10 (1.77x) | 834.56 (2.14x) |
| b16 VNNI | 166.76 rows/s | 317.18 (1.90x) | 387.07 (2.32x) |
| b24 VNNI | 66.58 rows/s | 129.65 (1.95x) | 159.81 (2.40x) |

Two independent physical cores therefore scale well. Enabling both SMT
siblings raises total throughput a further 10-23%, but also raises individual
request latency; a many-engine tournament host should optimize for aggregate
throughput rather than extrapolate isolated latency linearly. Four processes
are the maximum meaningful concurrency sample on this VM, so this is not a
direct simulation of a host with dozens of physical cores.

The final ORT 1.23.0 baselines were rerun on logical CPU 3 with the same
15x15, batch-1, no-mask contract. Each model ran in a fresh process with
sequential execution, one intra-op/inter-op thread, preallocated inputs and
outputs, 100 warmups, and 1000 alternating FP32/INT8 timed calls. For a fair
three-size comparison, these use the same onnxsim-simplified FP32 export and
the same dynamic trunk quantizer: U8 activations, symmetric per-tensor S8
weights, `MatMulInteger`, and FP32 stem/global/policy/value heads. In
particular, the b11-only fixed-RoPE attention rewrite used by an older baseline
is not mixed into this table because it does not support learned-RoPE+QKN.

| model | ORT FP32 | ORT dynamic INT8 | ORT INT8 speedup |
|---|---:|---:|---:|
| real b11c96 | 8.5955 ms | 7.2497 ms | 1.19x |
| fixture b16c128 | 16.8193 ms | 12.5732 ms | 1.34x |
| fixture b24c192 | 43.4409 ms | 29.0129 ms | 1.50x |

Only b11 has a real checkpoint and labeled validation corpus. b16 and b24 are
deterministic structural/performance fixtures; their numerical outputs were
checked byte-for-byte between each accepted tuning baseline and final binary.

The real b11 corpus contains 18,432 rows. Against the same FP32 reference
(`p0loss=1.6944791187`, `vloss=0.5712485925`), actual C++ replay measured:

| deployment recipe | p0loss | p0loss delta | vloss delta | gate |
|---|---:|---:|---:|---|
| VNNI S8/U8S8 | 1.70340036 | +0.00892124 | +0.00583537 | pass |
| AVX2 S7/A8W7 + Q7K8/P8V7 | 1.70623627 | +0.01175715 | +0.00682329 | pass |

Both are below the required `+0.03` p0loss limit. The strict AVX2 FP32-edge
change preserved the AVX2 result to approximately `1e-9`; converter reruns
also reproduce the accepted b11 S8 and b24 S7 model hashes exactly.

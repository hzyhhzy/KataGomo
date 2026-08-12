# Native model v104 explicit INT8 metadata

Model v104 preserves the complete native-v102 body and all FP32 master
weights. The three-byte model-version token changes from `102` to `104`, and a
mandatory quantization trailer follows the body's existing trailing
whitespace. Native v102 remains unchanged and has no trailer.

The canonical trailer header is ASCII:

```
@KATAGO_QUANT_TRAILER@ 1 <payloadBytes> <payloadSha256> @BIN@
```

There is no newline after `@BIN@`. Exactly `payloadBytes` binary bytes follow,
and EOF must follow the payload immediately. `payloadSha256` is lowercase
SHA-256 of the binary payload. Payload and record integers are little-endian.

Payload schema 1 starts with `KQPT104\0`, its schema and entry count, the
activation/weight/rounding enums, zero point, bit-exact float32 clip and scale,
and two zero reserved fields. It then contains 72 length-delimited records in
topology order:

- 24 QK records, with Q followed by K in one logical `[K,N]` matrix;
- 24 FFN-up records;
- 24 FFN-gate records.

Each record binds a stable architecture topology index and role, exact native
layer names, K/N, packed layout, zero point, bit-exact scales, packed byte
count, SHA-256 of the canonical FP32 master, SHA-256 of the packed bytes, and
the packed signed-int8 weights. Packing is output-major, K-contiguous. The
contract is symmetric per-tensor quantization, zero point 0, `[-127,127]`, and
round-to-nearest-ties-to-even. RMS-normalized activations use clip 4 and scale
`float32(4/127)`; each matrix has one weight scale.

Loading is fail-closed. The parser checks the outer hash, schemas, bounds,
reserved fields, strict EOF, record order/names/shapes, both hashes, scales,
and every packed byte by deterministically rebuilding all 72 entries from the
retained FP32 masters. The v104 production architecture signature is
`bbfa5957d8d87f1225fe4e679be055ff4de4c40c4f8437a19c7be3052c23f49b`;
the corresponding legacy v102 signature remains `ad026614455c0475b31997f1c5452af99d1eb347713f77950671fc5d1a522f24`.

After that model-load validation, CUDA resolves each block by topology index,
role, exact layer names, and K/N, and uploads the embedded packed bytes and
stored scale directly. It does not quantize the FP32 matrices again during
backend preparation. Native v102 retains the historical load-time-quantized
compatibility path and reports `legacy-v102-load-time-quant` in its markers;
v104 reports `embedded-v104`.

The authoritative training exporter and the independent
`python/build_v104_int8_trailer.py` bridge both require the explicit
`-int8-pt-clip4` flag. Default training export remains byte-identical v102.

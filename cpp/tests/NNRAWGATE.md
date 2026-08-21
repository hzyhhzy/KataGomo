# Stage1 raw numerical gate

`nnrawgate` is a test-only, untimed CUDA command. It is excluded unless
`KATAGO_BUILD_NNRAWGATE=ON`; the candidate's route assertion additionally
requires `KATAGO_BUILD_BENCHMARKNN=ON`. Performance runs must continue to use
the unmodified timed candidate build.

The command verifies a full model SHA-256, requires model v102 with V101 22/39
inputs, replays all corpus rows using actual (not padded) tail batches, and then
uses the same compute handle for `B,1,B-1,2,7,B`. It dumps raw policy, value,
score-value, and ownership floats in the identity-bound `NNRAWG1` format.
With `-expected-official-stage1`, every call must publish an incremented route
serial, the exact actual batch, active official attention and FFN for every
block, FP16/NHWC/exact/no-mask, and no scalar/planar/cuDNN/fallback route.

Build the numeric sidecar separately:

```text
cmake -S cpp -B build-nnrawgate \
  -DUSE_BACKEND=CUDA \
  -DKATAGO_BUILD_NNRAWGATE=ON \
  -DKATAGO_BUILD_BENCHMARKNN=ON
cmake --build build-nnrawgate --config Release --target katago
```

Reference builds from `7a46d0b0` need only
`KATAGO_BUILD_NNRAWGATE=ON`; they omit the candidate-only route assertion.

## Locked Stage1 invocations

P1 and C384 use the committed 8192-row full-board `R15CORP1` corpus. P2 has no
local authoritative 19x corpus, so both source trees generate the same dense
synthetic-v1 inputs and bind their canonical float bytes into the output input
identity.

```text
# P1, replace ARM and OUT for each of the three builds
katago nnrawgate -model reviewed.bin.gz -config gate.cfg \
  -expected-model-sha256 40cfa5ab15e23b12d065a2b4611e6b9aad0e02ade724c91657851acc53ffd4c6 \
  -corpus renju15_val_data0_full8192.r15c -board 15 -batch-size 36 \
  -output OUT <ARM_ARGS>

# P2
katago nnrawgate -model b24c256h8tflrs-connect6-swa-v102.bin.gz -config gate.cfg \
  -expected-model-sha256 7684bab8d6e3a3351f7ebcde9add34ab66594423de82df76099b5f597b869e1d \
  -board 19 -batch-size 28 -synthetic-rows 512 -output OUT <ARM_ARGS>

# C384
katago nnrawgate -model b36c384h12tflrs-bng-silu-v102-noqkn-clip0-seed20260821.bin.gz -config gate.cfg \
  -expected-model-sha256 6db378f28c3169fab9eb23b74ec645cfa2b67eb56e394af0888c5ae425ccc785 \
  -corpus renju15_val_data0_full8192.r15c -board 15 -batch-size 24 \
  -output OUT <ARM_ARGS>
```

For `<ARM_ARGS>`, use an override that forces NHWC and respectively:

- exact 7a FP32: `cudaUseFP16=false`;
- exact 7a FP16 control: `cudaUseFP16=true`;
- candidate FP16: `cudaUseFP16=true` plus `-expected-official-stage1`.

Keep randomization off and symmetry zero. The reference FP16 route should be
the locked predecessor route chosen for the qualification; its FP16-vs-FP32
error calibrates the candidate tolerance.

Compare the three dumps:

```text
python -B cpp/tests/nnrawgate_compare.py \
  --fp32 p1-7a-fp32.nnraw \
  --reference-fp16 p1-7a-fp16.nnraw \
  --candidate-fp16 p1-candidate-fp16.nnraw \
  --corpus renju15_val_data0_full8192.r15c \
  --output p1-gate.json
```

Omit `--corpus` for P2. The comparator rejects identity, precision, layout,
shape, full-batch trace, or schedule mismatch before numerical comparison. It
also requires the two reference arms to embed the same nonempty predecessor
revision and the candidate to embed a different revision, preventing a
candidate-vs-itself gate. For each semantic metric the candidate must satisfy
both
`error_vs_7a_FP32 <= max(floor, 2 * 7a_FP16_error_vs_7a_FP32)` and an absolute
cap. It gates raw RMSE/scaled score error, policy softmax TV/max/regret and
decisive top-1, value softmax error/top-1, ownership `sigmoid(2x)` error,
same-handle cross-batch drift, exact repeated-B determinism, and optional R15
p0/v loss.

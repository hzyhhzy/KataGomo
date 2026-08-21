# Canonical v105 QKN+clip4 cross-implementation gate

This is a test-only, compile-time-enabled extension of `nnrawgate`. It does
not change inference arithmetic. The model contract is deliberately narrow:

- model v105 with Inputs101 (`22` spatial and `39` global features);
- 15x15 full-board `R15CORP1`, identical model SHA and corpus identity in all arms;
- 36 transformer layers, kept distinct from the exact physical batch size 28;
- actual-batch sequence exactly `28,1,27,2,7,28` on one handle;
- four pre-postprocessing raw heads: policy 226, value 3, score-value 6,
  and ownership 225;
- NHWC inputs in all arms.

The candidate must be built with both `KATAGO_BUILD_NNRAWGATE=ON` and
`KATAGO_BUILD_BENCHMARKNN=ON`. Run its FP16 arm with:

```text
katago nnrawgate -config gate.cfg -model MODEL \
  --expected-model-sha256 MODEL_SHA --corpus R15CORP1 --output candidate.raw \
  --model-contract v105 --board 15 --batch-size 28 \
  --batch-schedule 28,1,27,2,7,28 --expected-official-v105-qkn-clip4
```

After every successful inference call the command requires one new structured
route snapshot. All 36 attention/FFN layers must report planar QKV, QKN,
learned-RoPE-FP32, MMA, and ordered clipped-SwiGLU. Combined-QKV, fixed RoPE,
scalar attention, cuDNN attention, and fallback counts must all be zero.

The two reference arms are built from exact revision
`c3b882e7aa01c2c250c76c01464a8850ed622329` plus the reviewed test-only raw
gate overlay. Configure that worktree with every non-generic path disabled:

```text
-DKATAGO_ENABLE_CUDNN_FRONTEND_SDPA=OFF
-DKATAGO_ENABLE_SM120_TRANSFORMER_WINNER=OFF
-DKATAGO_ENABLE_C384_EXACT_FIXED_AOT=OFF
-DKATAGO_ENABLE_RENJU15_INT8_EXPERIMENT=OFF
-DKATAGO_ENABLE_C384_INT8_EXPERIMENT=OFF
-DKATAGO_C384_EXACT_AOT_GENERATED_MANIFEST=
-DKATAGO_C384_H12_FA4_PACKAGE_CMAKE=
-DKATAGO_C384_EXACT_FFN_DOWN_PACKAGE_CMAKE=
```

Run reference FP32 and FP16 with the same command except omit the expected
route flag. Select precision only in their immutable config files. The dump
stores precision, NHWC, model/input versions, shape, physical batch, exact
schedule, model SHA, corpus identity, source revision, and route-contract id.

Compare the three arms with `nnrawgate_compare_v105.py`. It requires explicit
model, corpus, candidate-revision, and comparator-revision locks. Full-corpus
raw/probability, loss, top-1, and mean-regret checks use FP32 plus the reference
FP16 control. Dynamic calls are gated only as same-arm drift from the first
B28 prefix; B1 and B2 are never used as FP32 population gates. The report
includes candidate-worse, candidate-better, and mean-regret aggregates and has
no isolated maximum-regret decision gate.

```text
python cpp/tests/nnrawgate_compare_v105.py \
  --fp32 reference-fp32.raw --reference-fp16 reference-fp16.raw \
  --candidate-fp16 candidate.raw --corpus R15CORP1 \
  --expected-model-sha256 MODEL_SHA --expected-corpus-sha256 CORPUS_SHA \
  --candidate-revision CANDIDATE_COMMIT --comparator-revision COMPARATOR_COMMIT \
  --output v105-gate-report.json
```

`python cpp/tests/nnrawgate_compare_v105.py --self-test` includes negative
tests for physical batch 36 and mixed v102/v105 arms. Neither the build check
nor this self-test runs GPU inference.

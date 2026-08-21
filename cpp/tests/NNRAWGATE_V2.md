# NNRAWG1 comparator v2 contract

`nnrawgate_compare_v2.py` is a test-only, offline comparator for the existing
`NNRAWG1` raw dumps. It does not change inference, dump generation, or the v1
report. A v2 run must use a new output path and must pass `--v1-report` and the
exact v2 source commit as `--comparator-revision`. The comparator verifies that
the v1 report names the exact same three raw SHA-256 values and corpus SHA-256,
then hashes and records the immutable v1 result it follows.

The three arms and all v1 identity contracts remain unchanged:

- authoritative predecessor FP32;
- the same predecessor revision in FP16;
- a different candidate revision in FP16 with the in-process official-route
  assertion;
- exact model, input identity, board, row count, full actual-batch trace, and
  dynamic schedule identity;
- FP32/FP16/FP16 precision identity and NHWC for all three arms.

## Full-corpus checks

All v1 full-corpus adaptive floors and absolute caps remain unchanged for raw
policy RMSE, policy probability total variation and maximum error, value,
ownership, score-value, and optional R15 policy/value losses. Policy
probability, TV, both KL directions, raw/probability top-1, and decisive top-1
remain in the report. KL remains diagnostic because v1 did not assign it a
threshold.

The only full-corpus semantic change is the policy-decision check. For every
row, regret is measured using the FP32 probability distribution. V2 compares
the candidate regret to the predecessor-FP16 control regret:

```
new_regret = max(candidate_regret - control_regret, 0)
```

If candidate and control select the same policy index, `new_regret` is exactly
zero and that row passes regardless of their shared regret versus FP32. If they
select different indices, only positive candidate regression is accumulated;
the maximum new regret retains the v1 `0.01` absolute cap. The existing
candidate-relative-to-control aggregate value and decisive-policy top-1 checks
are retained.

## Dynamic-batch checks

Per-call FP16-versus-FP32 metrics for `B,1,B-1,2,7,B` remain in the JSON for
diagnosis, but are not gated. A one- or two-row slice is not a trustworthy
estimate of an implementation's FP16 error distribution.

Dynamic correctness is gated only by comparisons within each implementation:

- each actual-`B` call is compared with the same implementation's prefix from
  the first max-batch call;
- candidate drift must be no larger than the existing v1 adaptive floor or
  twice the worse of the FP32 and predecessor-FP16 drift, and must remain below
  the unchanged v1 absolute cap;
- the first and last max-batch calls must be bit-exact in every raw head;
- the first max-batch call must be bit-exact with the full replay prefix in
  every raw head.

The policy absolute-regret metric is not used for cross-batch drift. Raw policy
and probability drift, TV, value, ownership, and score-value still gate the
same-arm comparison.

## Evidence separation

Never overwrite a v1 report or any raw dump. Store v2 output in a separate
evidence directory together with:

- the v2 report and stdout/stderr;
- SHA-256 of the v2 comparator and imported v1 comparator;
- SHA-256 and embedded revisions of all three raw dumps;
- SHA-256 of the optional R15 corpus and immutable v1 report.

The report embeds those paths and hashes under `comparator` and `artifacts`.

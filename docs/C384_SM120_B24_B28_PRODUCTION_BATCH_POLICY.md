# C384/SM120 B24 versus B28 production batch policy

Status: CPU-only acceptance and manifest design. This policy does not select,
build, or install an AOT object and does not modify the KataGo engine.

This document makes B24 the fail-closed production default. B28 is promoted
only when one eligible, position-balanced comparison proves that its normalized
full-path throughput is strictly more than 0.5% above B24. The threshold is
exclusive and is applied before percentage rounding.

## 1. Decision rule

Let `A` be B24, `B` be B28, and let `R` be the paired geometric throughput
ratio defined in section 4. The selector is exactly:

```text
promote_b28 =
  evidence_eligible
  && correctness_passed
  && telemetry_passed
  && stability_passed
  && R > 1.005
  && lower_one_sided_95pct_confidence_bound(R) > 1.005

selected_batch = promote_b28 ? 28 : 24
```

Consequences are intentional:

- `R == 1.005` selects B24.
- `R < 1.005` selects B24.
- A point estimate above 1.005 whose confidence bound touches or crosses
  1.005 selects B24.
- Missing, malformed, unstable, non-equivalent, or mixed-provenance evidence
  selects B24.
- The decision uses N/s, not batch wall time. Comparing latency without
  normalizing for 24 versus 28 positions is invalid.
- No rounded display such as `+0.5%` may drive the decision. The manifest
  stores and compares the unrounded ratio.

This is a promotion rule, not a claim that B24 is intrinsically faster. Its
purpose is to require a useful, repeatable B28 gain before accepting the
larger fixed batch and its integration complexity.

## 2. Eligible comparison scope

Only either of the following comparison kinds is eligible.

### 2.1 Same formal full-engine comparison

The preferred evidence is the true KataGo timed inference path with both arms
run under the same formal configuration. It must bind:

- the same source commit, model SHA256, CUDA/cuDNN/driver identities, GPU,
  power/clock policy, server-lane count, search/evaluation configuration,
  warmup count, timed iteration count, and throughput definition;
- the same executable SHA256, preferably one binary containing both exact
  batch routes and selecting the arm before timing;
- if one binary cannot contain both fixed-M routes, an audited binary-pair
  manifest proving that the only code/data differences are the B24/B28 exact
  batch artifacts, IDs, fixed-M constants, and selected-batch manifest fields;
- identical correctness inputs and required 36/36 PREPARED and ACTIVE
  telemetry for each arm;
- raw child logs and telemetry for every leg, without post-hoc leg removal.

An old B24 number and a new B28 number are not a pair even if both were called
"full engine". A different executable, candidate set, model, S1/S2 lane count,
warmup, timing method, or configuration makes the evidence ineligible unless
the audited binary-pair exception above covers exactly the fixed-batch delta.

### 2.2 Proven-equivalent complete combined comparison

An engine-external combined harness is eligible only when a hash-bound
equivalence manifest proves all of the following:

- it executes the true 36-layer order and carries dependent state from layer
  `i` to layer `i+1`;
- each layer times the complete selected trunk sequence in order:
  QKV/learned-RoPE, packed FA4, attention output projection plus residual,
  dual up/gate plus SwiGLU, and FFN-down plus residual;
- both arms execute the same logical operations and launch counts; for the
  current five-family sequence this is 36 times 5, or 180 ordered launches;
- tensor shapes, layouts, data types, stream semantics, residual aliasing,
  candidate IDs, preparation state, warmup, and timed repetitions match the
  intended engine route for that batch;
- correctness, canaries, cross-layer dependency, full sequence wall timing,
  and clean telemetry pass for both arms;
- every engine operation omitted from the harness is enumerated, and the
  equivalence record explains why its omission cannot reverse the B24/B28
  normalized-throughput decision; any unaccounted batch-dependent work makes
  the comparison ineligible;
- the report labels its result as a paired combined-path ratio, not as real
  full-engine N/s.

A reviewer/validator must set `equivalence_status=passed` from those facts and
bind the equivalence manifest SHA256 into the final report. Merely naming a
program "36 layer" or summing isolated kernel times is not equivalence.

The following are always ineligible for batch selection:

- any single-family microbenchmark;
- the old 36x3 `QKV -> FA4 -> dual` subchain or its upper-bound N/s
  projection;
- a component-time sum reconstructed from separate processes;
- TensorRT versus native, generic versus exact, or pre-optimization versus
  post-optimization cross-binary comparisons;
- a fastest leg, unpaired mean, or result with missing raw legs;
- a combined harness whose coverage/equivalence record is absent or failed.

## 3. Formal ABBA/BAAB schedule

Each leg is a fresh, equally warmed child measurement. Process startup and
artifact preparation are outside the timed interval, but their success and
hashes remain in the report. Use at least three complete superrounds. One
superround has this fixed eight-leg order:

```text
A1 B1 B2 A2 | B3 A3 A4 B4

A = B24
B = B28
left half  = ABBA
right half = BAAB
```

This gives four adjacent, direction-balanced ratios per superround:

```text
q[s,1] = throughput(B1) / throughput(A1)
q[s,2] = throughput(B2) / throughput(A2)
q[s,3] = throughput(B3) / throughput(A3)
q[s,4] = throughput(B4) / throughput(A4)
```

No leg may be reused in two ratios. All legs use the same warmup and timed
iteration counts. The runner must archive the declared schedule before the
first timed leg and reject missing, duplicated, reordered, retried, or
manually censored legs. A failed leg invalidates the superround; it is not
silently replaced after inspecting throughput.

Before each superround, require GPU exclusivity and record temperature,
power, clocks, P-state, memory use, driver identity, and competing processes.
Any foreign GPU use, throttling, OOM/retry, fallback, non-finite output, or
missing 36/36 activation marker fails telemetry rather than becoming an
outlier to remove.

## 4. Ratio, confidence, and stability

For superround `s`, compute in log space:

```text
L[s] = mean(log(q[s,1]), ..., log(q[s,4]))
L     = mean(L[1], ..., L[n])
R     = exp(L)
gain_percent = 100 * (R - 1)
```

The independent statistical units are complete superrounds, not individual
iterations inside a child process. With `n >= 3`, the required one-sided 95%
lower confidence bound is:

```text
LCB95 = exp(L - t(0.95, n-1) * sample_stddev(L[s]) / sqrt(n))
```

If a different pre-registered confidence estimator is used, its name,
version, parameters, and raw inputs must be in the manifest, and it must not
be chosen after looking at the result. The default t-bound above is required
when that declaration is absent.

The formal stability gates are:

- at least 3 complete superrounds and 12 adjacent paired ratios;
- log-ratio CV, reported as `sqrt(exp(var(log(q))) - 1)`, at most 0.005;
- absolute endpoint drift for both arms at most 0.005, where endpoint drift
  is `mean(last chronological half) / mean(first chronological half) - 1`;
- every superround ratio `exp(L[s])` is above 1.0;
- no correctness, fallback, telemetry, thermal/throttle, exclusivity, or
  provenance failure;
- no result-dependent outlier removal or early stopping.

These gates do not replace the threshold. B28 is selected only when both the
point ratio and `LCB95` are strictly greater than 1.005. Increasing repetitions
to resolve an inconclusive result is allowed only by appending complete,
predeclared superrounds and preserving all earlier legs.

## 5. Current decision on 2026-08-13

The available evidence does not promote B28:

| Evidence | B24 | B28 | Ratio / interpretation | Policy result |
| --- | ---: | ---: | --- | --- |
| early same-setting screen | 6086.55 N/s | 6105.38 N/s | `1.003093706615`, about `+0.30937%` | B24; below exclusive threshold |
| current best real full-engine result | no same-version final arm | 6957.434661 N/s | denominator/provenance pair missing | B24; incomplete evidence |
| old 36x3 upper-bound projection | 9413.33 N/s | 10059.22 N/s | about `+6.86%`, but subchain projection only | ignored; ineligible scope |

Therefore `current_default_batch=24`. The 6957.434661 B28 result remains a
valid B28 performance record, but it cannot be divided by an older B24 result
to manufacture a batch-selection ratio. A future eligible ABBA/BAAB report may
change the selected production package to B28 only by passing every predicate
in section 1.

## 6. Acceptance-runner configuration

The benchmark gate should expose typed, immutable configuration fields with
these defaults:

| Field | Required/default value |
| --- | --- |
| `batch_policy_default` | `24` |
| `batch_policy_challenger` | `28` |
| `promotion_ratio_exclusive` | `1.005` |
| `confidence_kind` | `one-sided-student-t-on-superround-log-ratios` |
| `confidence_level` | `0.95` |
| `confidence_lcb_ratio_exclusive` | `1.005` |
| `min_superrounds` | `3` |
| `schedule` | `ABBA_BAAB` |
| `max_log_ratio_cv` | `0.005` |
| `max_abs_endpoint_drift` | `0.005` |
| `incomplete_or_unstable_policy` | `select-b24` |
| `allowed_comparison_kind` | `full-engine` or `engine-external-complete-equivalent` |

The runner must print the parsed policy before timing, include its canonical
configuration SHA256 in every child record, and fail if a CLI/environment
override changes a policy field without also producing a new declared config
hash. The engine itself does not dynamically benchmark these batches; it
consumes one already-selected, hash-bound fixed-batch production package.

## 7. Production manifest fields

The final `PRODUCTION_MANIFEST.json` should contain a required
`batch_selection` object. Suggested fields are:

```json
{
  "batch_selection": {
    "policy_version": 1,
    "default_batch": 24,
    "challenger_batch": 28,
    "selected_batch": 24,
    "decision": "default_b24_or_promoted_b28",
    "comparison_kind": "full-engine_or_engine-external-complete-equivalent",
    "schedule": "ABBA_BAAB",
    "superrounds": 3,
    "promotion_ratio_exclusive": 1.005,
    "paired_geomean_ratio": 0.0,
    "gain_percent_unrounded": 0.0,
    "confidence_kind": "one-sided-student-t-on-superround-log-ratios",
    "confidence_level": 0.95,
    "confidence_lcb_ratio": 0.0,
    "max_log_ratio_cv_allowed": 0.005,
    "observed_log_ratio_cv": 0.0,
    "max_abs_endpoint_drift_allowed": 0.005,
    "observed_b24_endpoint_drift": 0.0,
    "observed_b28_endpoint_drift": 0.0,
    "b24_nps": 0.0,
    "b28_nps": 0.0,
    "correctness_status": "passed",
    "telemetry_status": "passed",
    "stability_status": "passed",
    "equivalence_status": "not-required_or_passed",
    "source_commit": "sha1",
    "executable_b24_sha256": "sha256",
    "executable_b28_sha256": "sha256",
    "binary_pair_diff_audit_sha256": "same-binary_or_sha256",
    "model_sha256": "sha256",
    "benchmark_config_sha256": "sha256",
    "runtime_environment_sha256": "sha256",
    "raw_report_sha256": "sha256",
    "raw_logs_archive_sha256": "sha256",
    "equivalence_manifest_sha256": "not-required_or_sha256"
  }
}
```

The real materializer must replace all illustrative values, use JSON numbers
for numeric fields, and validate internal consistency. In particular,
`selected_batch=28` is legal only when both stored ratios are strictly above
1.005 and every status is `passed`. Otherwise it must emit B24 or fail closed.
The manifest also binds lane count, GPU UUID, CUDA/cuDNN/driver versions,
candidate IDs for both arms, raw per-leg records, 36/36 markers, and the exact
throughput numerator/denominator definition, either directly or through the
hash-bound raw report.

## 8. Required negative tests

The policy validator and production materializer must reject or default to B24
for each fixture below.

Threshold and statistics:

- ratios `1.004999999`, exactly `1.005`, NaN, infinity, zero, or a negative
  throughput;
- point ratio above 1.005 with `LCB95 <= 1.005`;
- ratio/endpoint drift above its maximum, a non-positive superround, fewer
  than three complete superrounds, or fewer than twelve paired ratios;
- a displayed rounded `+0.5%` disagreeing with the raw ratio;
- arithmetic-mean ratio substituted for the declared paired geometric ratio;
- post-hoc outlier deletion, early stop, or replacement of a slow/failed leg.

Order and raw-record integrity:

- ABAB, AABB, ABBA without the mirrored BAAB half, missing/reordered/duplicate
  legs, unequal warmup/iteration counts, or reused legs;
- a fastest-leg-only record, summaries with no raw child logs, or report/hash
  disagreement;
- foreign GPU use, fallback, a missing 36/36 marker, throttling, retry, OOM,
  non-finite output, correctness failure, or incomplete telemetry.

Scope and provenance:

- single-op microbenchmarks and the 36x3 QKV/FA4/dual upper-bound projection;
- B28 `6957.434661` compared with an older/different-binary B24 number;
- mismatched source/model/config/lane/driver/runtime/candidate IDs, or two
  binaries without the restricted binary-pair diff audit;
- a combined report with fewer than 36 dependent layers, fewer than the
  declared 180 operations, summed component times, absent equivalence
  manifest, unaccounted batch-dependent omitted work, or failed equivalence;
- a comparison that mixes real engine N/s and projected combined-path N/s.

Manifest consistency:

- `selected_batch=28` with a ratio or confidence bound at/below 1.005;
- `selected_batch=28` with any non-passed status or missing evidence hash;
- `selected_batch=24` mislabeled as `promoted_b28`;
- policy thresholds changed from this document without a new policy version
  and config hash;
- B24 and B28 production artifacts both active in one supposedly fixed-batch
  package.

Every fail-closed test asserts the resulting selection is B24 and that no B28
production manifest is emitted. Parser corruption, contradictory hashes, or
an impossible numeric state should additionally return a non-zero validation
status rather than silently repairing the evidence.

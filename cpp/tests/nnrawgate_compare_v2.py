#!/usr/bin/env python3
"""Stage1 NNRAWG1 comparator v2.

V2 preserves the v1 full-corpus numerical checks, but makes the two narrowly
reviewed corrections documented in NNRAWGATE_V2.md:

* actual-batch calls are gated by same-arm cross-batch drift, not by a noisy
  one- or two-row estimate of each FP16 arm's error versus FP32;
* policy decisions are gated on candidate regression relative to the FP16
  control. A row where candidate and control select the same policy index has
  exactly zero incremental regret, even if both differ from FP32.

The v1 comparator is imported rather than copied so raw parsing, identity,
provenance, layout validation, metric definitions, and numeric floors/caps
remain shared and reviewable.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import nnrawgate_compare as v1


REPORT_SCHEMA = 2
INCREMENTAL_POLICY_REGRET_CAP = 0.01
ADAPTIVE_SPECS = {
    name: limits
    for name, limits in v1.ADAPTIVE_SPECS.items()
    if name != "policyDecision.decisionRegret.max"
}


def add_adaptive_checks(
    label: str,
    candidate_metrics: dict[str, Any],
    control_metrics: list[dict[str, Any]],
    checks: list[dict[str, Any]],
) -> None:
    """Apply every v1 adaptive/cap check except absolute policy regret."""
    for metric, (floor, cap) in ADAPTIVE_SPECS.items():
        candidate_value = v1.get_path(candidate_metrics, metric)
        control_value = max(v1.get_path(control, metric) for control in control_metrics)
        adaptive_limit = max(floor, 2.0 * control_value)
        passed = (
            math.isfinite(candidate_value)
            and candidate_value <= adaptive_limit
            and candidate_value <= cap
        )
        checks.append({
            "label": f"{label}.{metric}",
            "candidate": candidate_value,
            "control": control_value,
            "floor": floor,
            "adaptiveLimit": adaptive_limit,
            "absoluteCap": cap,
            "pass": passed,
        })


def relative_policy_decision_metrics(
    candidate: dict[str, np.ndarray],
    control: dict[str, np.ndarray],
    fp32: dict[str, np.ndarray],
) -> dict[str, Any]:
    candidate_probability = v1.softmax(candidate["policy"])
    control_probability = v1.softmax(control["policy"])
    fp32_probability = v1.softmax(fp32["policy"])
    rows = np.arange(fp32_probability.shape[0])
    fp32_top1 = fp32_probability.argmax(axis=1)
    control_top1 = control_probability.argmax(axis=1)
    candidate_top1 = candidate_probability.argmax(axis=1)
    control_regret = (
        fp32_probability[rows, fp32_top1] - fp32_probability[rows, control_top1]
    )
    candidate_regret = (
        fp32_probability[rows, fp32_top1] - fp32_probability[rows, candidate_top1]
    )
    different = candidate_top1 != control_top1
    signed_increment = candidate_regret - control_regret
    new_regret = np.where(different, np.maximum(signed_increment, 0.0), 0.0)
    worse = different & (signed_increment > 0.0)
    better = different & (signed_increment < 0.0)
    equal_regret = different & (signed_increment == 0.0)
    row_count = int(rows.size)
    different_count = int(np.count_nonzero(different))
    return {
        "rows": row_count,
        "sameSelectionRows": row_count - different_count,
        "selectionDisagreementRows": different_count,
        "selectionDisagreementRate": float(different_count / row_count),
        "candidateWorseSelectionRows": int(np.count_nonzero(worse)),
        "candidateBetterSelectionRows": int(np.count_nonzero(better)),
        "differentSelectionEqualRegretRows": int(np.count_nonzero(equal_regret)),
        "sameSelectionFastPath": bool(different_count == 0),
        "incrementalRegret": {
            "meanAllRows": float(np.mean(new_regret)),
            "meanWorseRows": (
                float(np.mean(new_regret[worse])) if np.any(worse) else 0.0
            ),
            "max": float(np.max(new_regret)),
        },
        "signedRegretDelta": {
            "min": float(np.min(signed_increment[different])) if different_count else 0.0,
            "max": float(np.max(signed_increment[different])) if different_count else 0.0,
        },
        "controlRegret": {
            "mean": float(np.mean(control_regret)),
            "max": float(np.max(control_regret)),
        },
        "candidateRegret": {
            "mean": float(np.mean(candidate_regret)),
            "max": float(np.max(candidate_regret)),
        },
    }


def add_full_relative_decision_checks(
    candidate_metrics: dict[str, Any],
    control_metrics: dict[str, Any],
    relative_decision: dict[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    """Keep v1 aggregate top-1 checks and gate only newly added regret."""
    candidate_value_top1 = candidate_metrics["valueProbability"]["top1Agreement"]
    control_value_top1 = control_metrics["valueProbability"]["top1Agreement"]
    value_minimum = max(0.99, control_value_top1 - 0.002)
    checks.append({
        "label": "full.valueProbability.top1Agreement",
        "candidate": candidate_value_top1,
        "control": control_value_top1,
        "minimum": value_minimum,
        "pass": candidate_value_top1 >= value_minimum,
    })

    candidate_decisive = candidate_metrics["policyDecision"]["decisiveSubset"]
    control_decisive = control_metrics["policyDecision"]["decisiveSubset"]
    if (
        candidate_decisive["rows"] >= 16
        and control_decisive["rows"] == candidate_decisive["rows"]
    ):
        candidate_top1 = candidate_decisive["top1Agreement"]
        control_top1 = control_decisive["top1Agreement"]
        minimum = max(0.99, control_top1 - 0.002)
        checks.append({
            "label": "full.policyDecision.decisiveTop1Agreement",
            "rows": candidate_decisive["rows"],
            "candidate": candidate_top1,
            "control": control_top1,
            "minimum": minimum,
            "pass": candidate_top1 is not None and candidate_top1 >= minimum,
        })

    incremental_max = relative_decision["incrementalRegret"]["max"]
    checks.append({
        "label": "full.policyDecision.incrementalRegret.max",
        "candidate": incremental_max,
        "control": 0.0,
        "absoluteCap": INCREMENTAL_POLICY_REGRET_CAP,
        "selectionDisagreementRows": relative_decision["selectionDisagreementRows"],
        "candidateWorseSelectionRows": relative_decision["candidateWorseSelectionRows"],
        "sameSelectionFastPath": relative_decision["sameSelectionFastPath"],
        "pass": math.isfinite(incremental_max)
        and incremental_max <= INCREMENTAL_POLICY_REGRET_CAP,
    })


def compare(args: argparse.Namespace) -> dict[str, Any]:
    fp32 = v1.read_dump(args.fp32)
    reference_fp16 = v1.read_dump(args.reference_fp16)
    candidate_fp16 = v1.read_dump(args.candidate_fp16)
    v1.validate_arm_provenance(fp32, reference_fp16, candidate_fp16)
    v1.validate_arm_layout(fp32, reference_fp16, candidate_fp16)
    identity = v1.identity_view(fp32)
    if (
        v1.identity_view(reference_fp16) != identity
        or v1.identity_view(candidate_fp16) != identity
    ):
        raise ValueError(
            "model/input/shape/full-batch/schedule identities differ across implementations"
        )
    if (
        fp32["meta"]["usingFP16"] != 0
        or reference_fp16["meta"]["usingFP16"] != 1
        or candidate_fp16["meta"]["usingFP16"] != 1
    ):
        raise ValueError("precision-arm identities must be FP32/FP16/FP16")
    if (
        fp32["meta"]["expectedOfficialStage1"] != 0
        or reference_fp16["meta"]["expectedOfficialStage1"] != 0
    ):
        raise ValueError("reference arms must not claim the candidate route assertion")
    if candidate_fp16["meta"]["expectedOfficialStage1"] != 1:
        raise ValueError("candidate arm lacks the in-process official-route assertion")

    checks: list[dict[str, Any]] = []
    full_control = v1.pair_metrics(reference_fp16["full"], fp32["full"])
    full_candidate = v1.pair_metrics(candidate_fp16["full"], fp32["full"])
    full_direct = v1.pair_metrics(candidate_fp16["full"], reference_fp16["full"])
    relative_decision = relative_policy_decision_metrics(
        candidate_fp16["full"], reference_fp16["full"], fp32["full"]
    )
    add_adaptive_checks("full", full_candidate, [full_control], checks)
    add_full_relative_decision_checks(
        full_candidate, full_control, relative_decision, checks
    )

    # Dynamic FP16-vs-FP32 metrics remain in the report for diagnosis, but are
    # deliberately not gated: B=1/B=2 are not population error estimates.
    dynamic: list[dict[str, Any]] = []
    for index, batch in enumerate(fp32["schedule"]):
        fp32_call = v1.call_arrays(fp32["calls"][index])
        reference_call = v1.call_arrays(reference_fp16["calls"][index])
        candidate_call = v1.call_arrays(candidate_fp16["calls"][index])
        dynamic.append({
            "index": index,
            "batch": batch,
            "gated": False,
            "referenceFP16VsFP32": v1.pair_metrics(reference_call, fp32_call),
            "candidateFP16VsFP32": v1.pair_metrics(candidate_call, fp32_call),
            "candidateVsReferenceFP16": v1.pair_metrics(candidate_call, reference_call),
        })

    anchor_fp32 = v1.call_arrays(fp32["calls"][0])
    anchor_reference = v1.call_arrays(reference_fp16["calls"][0])
    anchor_candidate = v1.call_arrays(candidate_fp16["calls"][0])
    cross_batch: list[dict[str, Any]] = []
    for index in range(1, len(fp32["schedule"])):
        batch = fp32["schedule"][index]
        fp32_call = v1.call_arrays(fp32["calls"][index])
        reference_call = v1.call_arrays(reference_fp16["calls"][index])
        candidate_call = v1.call_arrays(candidate_fp16["calls"][index])
        fp32_prefix = {name: values[:batch] for name, values in anchor_fp32.items()}
        reference_prefix = {
            name: values[:batch] for name, values in anchor_reference.items()
        }
        candidate_prefix = {
            name: values[:batch] for name, values in anchor_candidate.items()
        }
        fp32_drift = v1.pair_metrics(fp32_call, fp32_prefix)
        reference_drift = v1.pair_metrics(reference_call, reference_prefix)
        candidate_drift = v1.pair_metrics(candidate_call, candidate_prefix)
        add_adaptive_checks(
            f"crossBatch[{index}].B{batch}",
            candidate_drift,
            [fp32_drift, reference_drift],
            checks,
        )
        cross_batch.append({
            "index": index,
            "batch": batch,
            "fp32DriftFromFirstMaxBatch": fp32_drift,
            "referenceFP16DriftFromFirstMaxBatch": reference_drift,
            "candidateFP16DriftFromFirstMaxBatch": candidate_drift,
        })

    repeated_exact: dict[str, bool] = {}
    full_prefix_exact: dict[str, bool] = {}
    for label, dump in (
        ("fp32", fp32),
        ("referenceFP16", reference_fp16),
        ("candidateFP16", candidate_fp16),
    ):
        first = v1.call_arrays(dump["calls"][0])
        last = v1.call_arrays(dump["calls"][-1])
        exact = all(np.array_equal(first[name], last[name]) for name in first)
        repeated_exact[label] = exact
        checks.append({"label": f"repeatMaxBatchExact.{label}", "pass": exact})
        prefix_exact = all(
            np.array_equal(dump["full"][name][: dump["meta"]["maxBatch"]], first[name])
            for name in first
        )
        full_prefix_exact[label] = prefix_exact
        checks.append({
            "label": f"fullPrefixMatchesDynamicMaxExact.{label}",
            "pass": prefix_exact,
        })

    loss: dict[str, Any] | None = None
    if args.corpus is not None:
        corpus_identity, policy_target, global_target = v1.read_r15_targets(args.corpus)
        if (
            corpus_identity != fp32["inputIdentity"]
            or policy_target.shape[0] != fp32["meta"]["rows"]
        ):
            raise ValueError("R15 target corpus identity/rows differ from raw dumps")
        loss_fp32 = v1.weighted_losses(
            fp32["full"]["policy"], fp32["full"]["value"], policy_target, global_target
        )
        loss_reference = v1.weighted_losses(
            reference_fp16["full"]["policy"],
            reference_fp16["full"]["value"],
            policy_target,
            global_target,
        )
        loss_candidate = v1.weighted_losses(
            candidate_fp16["full"]["policy"],
            candidate_fp16["full"]["value"],
            policy_target,
            global_target,
        )
        loss = {
            "fp32": loss_fp32,
            "referenceFP16": loss_reference,
            "candidateFP16": loss_candidate,
        }
        for name in ("p0loss", "vloss"):
            control_error = abs(loss_reference[name] - loss_fp32[name])
            candidate_error = abs(loss_candidate[name] - loss_fp32[name])
            adaptive_limit = max(0.001, 2.0 * control_error)
            checks.append({
                "label": f"loss.{name}",
                "candidateAbsErrorVsFP32": candidate_error,
                "controlAbsErrorVsFP32": control_error,
                "floor": 0.001,
                "adaptiveLimit": adaptive_limit,
                "absoluteCap": 0.01,
                "pass": candidate_error <= adaptive_limit and candidate_error <= 0.01,
            })
    elif fp32["meta"]["sourceKind"] == 1:
        raise ValueError("R15CORP1 dumps require --corpus so p0/v loss is gated")

    comparator_path = Path(__file__).resolve()
    base_comparator_path = Path(v1.__file__).resolve()
    v1_report_json = json.loads(args.v1_report.read_text(encoding="utf-8"))
    if v1_report_json.get("schema") != 1:
        raise ValueError("v2 provenance requires a schema-1 v1 report")
    for name, dump in (
        ("fp32", fp32),
        ("referenceFP16", reference_fp16),
        ("candidateFP16", candidate_fp16),
    ):
        if v1_report_json.get("artifacts", {}).get(name, {}).get("sha256") != dump["sha256"]:
            raise ValueError(f"v1 report {name} raw SHA-256 differs from the v2 input")
    if args.corpus is not None:
        v1_corpus_sha = v1_report_json.get("artifacts", {}).get("corpus", {}).get("sha256")
        if v1_corpus_sha != v1.file_sha256(args.corpus):
            raise ValueError("v1 report corpus SHA-256 differs from the v2 input")
    v1_report = {
        "path": str(args.v1_report.resolve()),
        "sha256": v1.file_sha256(args.v1_report),
        "schema": v1_report_json.get("schema"),
        "pass": v1_report_json.get("pass"),
    }
    failures = [check for check in checks if not check["pass"]]
    return {
        "schema": REPORT_SCHEMA,
        "createdUtc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pass": not failures,
        "comparator": {
            "name": "nnrawgate_compare_v2",
            "sourceRevision": args.comparator_revision,
            "path": str(comparator_path),
            "sha256": v1.file_sha256(comparator_path),
            "baseV1Path": str(base_comparator_path),
            "baseV1Sha256": v1.file_sha256(base_comparator_path),
            "v1Report": v1_report,
        },
        "identity": identity,
        "artifacts": {
            "fp32": {key: fp32[key] for key in ("path", "sha256", "revision")},
            "referenceFP16": {
                key: reference_fp16[key] for key in ("path", "sha256", "revision")
            },
            "candidateFP16": {
                key: candidate_fp16[key] for key in ("path", "sha256", "revision")
            },
            "corpus": None
            if args.corpus is None
            else {
                "path": str(args.corpus.resolve()),
                "sha256": v1.file_sha256(args.corpus),
            },
        },
        "gateRules": {
            "fullCorpus": "v1 adaptive floors and absolute caps unchanged except policy decision uses incremental candidate-vs-control regret",
            "dynamic": "diagnostic only; numerical gating uses same-arm actual-B vs first-max-batch prefix drift",
            "crossBatch": "candidate drift <= max(floor,2*max(fp32 drift,reference FP16 drift)) and <= v1 absolute cap",
            "policyDecision": "same candidate/control selection adds zero regret; only positive candidate regret delta is capped",
        },
        "full": {
            "referenceFP16VsFP32": full_control,
            "candidateFP16VsFP32": full_candidate,
            "candidateVsReferenceFP16": full_direct,
            "candidateRelativeToControlDecision": relative_decision,
        },
        "dynamic": dynamic,
        "crossBatch": cross_batch,
        "repeatMaxBatchExact": repeated_exact,
        "fullPrefixMatchesDynamicMaxExact": full_prefix_exact,
        "loss": loss,
        "checks": checks,
        "failures": failures,
    }


def self_test() -> None:
    def arm(policy: list[float]) -> dict[str, np.ndarray]:
        return {"policy": np.asarray([policy], dtype=np.float64)}

    fp32 = arm([2.0, 1.0, 0.0])
    same_wrong_control = arm([0.0, 2.0, 0.0])
    same_wrong_candidate = arm([0.0, 3.0, 0.0])
    same = relative_policy_decision_metrics(
        same_wrong_candidate, same_wrong_control, fp32
    )
    assert same["selectionDisagreementRows"] == 0
    assert same["incrementalRegret"]["max"] == 0.0
    assert same["controlRegret"]["max"] == same["candidateRegret"]["max"]

    better = relative_policy_decision_metrics(
        arm([3.0, 0.0, 0.0]), arm([0.0, 0.0, 3.0]), fp32
    )
    assert better["candidateBetterSelectionRows"] == 1
    assert better["incrementalRegret"]["max"] == 0.0

    worse = relative_policy_decision_metrics(
        arm([0.0, 0.0, 3.0]), arm([3.0, 0.0, 0.0]), fp32
    )
    assert worse["candidateWorseSelectionRows"] == 1
    assert worse["incrementalRegret"]["max"] > INCREMENTAL_POLICY_REGRET_CAP
    checks: list[dict[str, Any]] = []
    dummy_pair = {
        "valueProbability": {"top1Agreement": 1.0},
        "policyDecision": {
            "decisiveSubset": {"rows": 16, "top1Agreement": 1.0}
        },
    }
    add_full_relative_decision_checks(dummy_pair, dummy_pair, same, checks)
    assert all(check["pass"] for check in checks)
    assert "policyDecision.decisionRegret.max" not in ADAPTIVE_SPECS
    print(json.dumps({
        "schema": REPORT_SCHEMA,
        "selfTest": "pass",
        "checks": len(checks) + 7,
    }))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32", type=Path)
    parser.add_argument("--reference-fp16", type=Path)
    parser.add_argument("--candidate-fp16", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--v1-report", type=Path)
    parser.add_argument("--comparator-revision")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    required = (
        args.fp32,
        args.reference_fp16,
        args.candidate_fp16,
        args.v1_report,
        args.comparator_revision,
        args.output,
    )
    if any(value is None for value in required):
        parser.error(
            "--fp32, --reference-fp16, --candidate-fp16, --v1-report, "
            "--comparator-revision, and --output are required"
        )
    if (
        len(args.comparator_revision) != 40
        or any(character not in "0123456789abcdef" for character in args.comparator_revision)
    ):
        parser.error("--comparator-revision must be a normalized 40-hex commit")
    result = compare(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "schema": REPORT_SCHEMA,
        "pass": result["pass"],
        "failures": len(result["failures"]),
        "output": str(args.output),
    }))
    if not result["pass"]:
        raise SystemExit("nnrawgate v2 numerical gate failed")


if __name__ == "__main__":
    main()

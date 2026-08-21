#!/usr/bin/env python3
"""Three-arm canonical-v105 QKN+clip4 numerical gate.

The full R15CORP1 population is compared as FP32 reference, FP16 reference,
and FP16 candidate. Dynamic batches are evaluated only as same-arm drift from
the first B=28 call, so B=1/B=2 are never treated as population estimates.
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
import nnrawgate_compare_v2 as v2


REPORT_SCHEMA = 1
PROFILE = "v105-qkn-clip4"
REFERENCE_BASE_REVISION = "c3b882e7aa01c2c250c76c01464a8850ed622329"
REFERENCE_TEST_OVERLAY_REVISION = "62a30377872fb6ceab2677484baabb1e96d48acd"
CANDIDATE_SEMANTICS_BASE_REVISION = "16999f76f0fd58681966d44324b8154f89323168"
ROUTE_NONE = 0
ROUTE_V105_QKN_CLIP4 = 2
LAYER_COUNT = 36
PHYSICAL_BATCH_SIZE = 28
EXACT_SCHEDULE = [28, 1, 27, 2, 7, 28]
POLICY_MEAN_REGRET_FLOOR = 1.0e-5
POLICY_MEAN_REGRET_CAP = 1.0e-3


def normalized_sha(value: str, label: str, length: int) -> str:
    if len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a normalized {length}-hex value")
    return value


def validate_revision_lock(
    fp32: dict[str, Any],
    reference_fp16: dict[str, Any],
    candidate_fp16: dict[str, Any],
    candidate_revision: str,
) -> dict[str, Any]:
    reference_frozen_overlay = f"{REFERENCE_TEST_OVERLAY_REVISION}-cuda"
    if fp32["revision"] != reference_fp16["revision"]:
        raise ValueError("FP32 and FP16 reference dumps have different source revisions")
    if fp32["revision"] != reference_frozen_overlay:
        raise ValueError(
            "reference dumps do not match the frozen test-only overlay whose sole parent is exact c3b882e7a"
        )
    expected_candidate = f"{candidate_revision}-cuda"
    if candidate_fp16["revision"] != expected_candidate:
        raise ValueError("candidate dump does not match the exact clean candidate revision lock")
    return {
        "referenceBaseRevision": REFERENCE_BASE_REVISION,
        "referenceFrozenTestOverlayRevision": REFERENCE_TEST_OVERLAY_REVISION,
        "referenceEmbeddedRevision": fp32["revision"],
        "referenceOverlayParentIsExactC3": True,
        "candidateSemanticsBaseRevision": CANDIDATE_SEMANTICS_BASE_REVISION,
        "candidateEmbeddedRevision": candidate_fp16["revision"],
    }


def validate_v105_contract(
    fp32: dict[str, Any],
    reference_fp16: dict[str, Any],
    candidate_fp16: dict[str, Any],
    expected_model_sha256: str,
) -> dict[str, Any]:
    dumps = (fp32, reference_fp16, candidate_fp16)
    identity = v1.identity_view(fp32)
    if any(v1.identity_view(dump) != identity for dump in dumps[1:]):
        raise ValueError("model/input/shape/full-batch/schedule identities differ across arms")
    for label, dump in zip(("fp32", "referenceFP16", "candidateFP16"), dumps):
        meta = dump["meta"]
        exact = (
            meta["modelVersion"], meta["inputsVersion"], meta["board"],
            meta["maxBatch"], meta["spatialFeatures"], meta["globalFeatures"],
            meta["policyDim"], meta["valueDim"], meta["scoreValueDim"],
            meta["ownershipDim"], meta["nhwc"], meta["sourceKind"],
        )
        if exact != (105, 101, 15, PHYSICAL_BATCH_SIZE, 22, 39, 226, 3, 6, 225, 1, 1):
            raise ValueError(f"{label} does not satisfy the exact canonical-v105/R15/NHWC contract")
        if dump["schedule"] != EXACT_SCHEDULE:
            raise ValueError(f"{label} dynamic schedule is not exactly B,1,B-1,2,7,B")
        if dump["modelSha256"] != expected_model_sha256:
            raise ValueError(f"{label} model SHA-256 differs from the explicit model lock")
    if tuple(dump["meta"]["usingFP16"] for dump in dumps) != (0, 1, 1):
        raise ValueError("precision-arm identities must be FP32/FP16/FP16")
    route_contracts = tuple(dump["meta"]["expectedOfficialStage1"] for dump in dumps)
    if route_contracts != (ROUTE_NONE, ROUTE_NONE, ROUTE_V105_QKN_CLIP4):
        raise ValueError("route contracts must be reference=none/none and candidate=v105-QKN-clip4")
    result = dict(identity)
    result["routeContract"] = {
        "fp32": route_contracts[0],
        "referenceFP16": route_contracts[1],
        "candidateFP16": route_contracts[2],
        "candidateMeaning": "36 layers: planar/QKN/learnedRopeFp32/MMA/clippedSwiGLU; combined/scalar/cudnn/fallback=0; physical batch 28",
    }
    return result


def add_full_decision_checks(
    candidate_metrics: dict[str, Any],
    control_metrics: dict[str, Any],
    relative_decision: dict[str, Any],
    checks: list[dict[str, Any]],
) -> None:
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
    if candidate_decisive["rows"] >= 16 and candidate_decisive["rows"] == control_decisive["rows"]:
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

    candidate_mean = candidate_metrics["policyDecision"]["decisionRegret"]["mean"]
    control_mean = control_metrics["policyDecision"]["decisionRegret"]["mean"]
    adaptive_limit = max(POLICY_MEAN_REGRET_FLOOR, 2.0 * control_mean)
    checks.append({
        "label": "full.policyDecision.decisionRegret.mean",
        "candidate": candidate_mean,
        "control": control_mean,
        "floor": POLICY_MEAN_REGRET_FLOOR,
        "adaptiveLimit": adaptive_limit,
        "absoluteCap": POLICY_MEAN_REGRET_CAP,
        "candidateWorseSelectionRows": relative_decision["candidateWorseSelectionRows"],
        "candidateBetterSelectionRows": relative_decision["candidateBetterSelectionRows"],
        "incrementalRegretMeanAllRows": relative_decision["incrementalRegret"]["meanAllRows"],
        "pass": math.isfinite(candidate_mean)
        and candidate_mean <= adaptive_limit
        and candidate_mean <= POLICY_MEAN_REGRET_CAP,
    })


def compare(args: argparse.Namespace) -> dict[str, Any]:
    fp32 = v1.read_dump(args.fp32, allowed_model_versions=(105,))
    reference_fp16 = v1.read_dump(args.reference_fp16, allowed_model_versions=(105,))
    candidate_fp16 = v1.read_dump(args.candidate_fp16, allowed_model_versions=(105,))
    candidate_revision = normalized_sha(args.candidate_revision, "candidate-revision", 40)
    model_sha = normalized_sha(args.expected_model_sha256, "expected-model-sha256", 64)
    corpus_sha = normalized_sha(args.expected_corpus_sha256, "expected-corpus-sha256", 64)
    revision_lock = validate_revision_lock(fp32, reference_fp16, candidate_fp16, candidate_revision)
    identity = validate_v105_contract(fp32, reference_fp16, candidate_fp16, model_sha)

    actual_corpus_sha = v1.file_sha256(args.corpus)
    if actual_corpus_sha != corpus_sha:
        raise ValueError("R15CORP1 file SHA-256 differs from the explicit corpus lock")
    corpus_identity, policy_target, global_target = v1.read_r15_targets(args.corpus)
    if corpus_identity != fp32["inputIdentity"] or policy_target.shape[0] != fp32["meta"]["rows"]:
        raise ValueError("R15 target corpus identity/rows differ from raw dumps")

    checks: list[dict[str, Any]] = []
    full_control = v1.pair_metrics(reference_fp16["full"], fp32["full"])
    full_candidate = v1.pair_metrics(candidate_fp16["full"], fp32["full"])
    full_direct = v1.pair_metrics(candidate_fp16["full"], reference_fp16["full"])
    relative_decision = v2.relative_policy_decision_metrics(
        candidate_fp16["full"], reference_fp16["full"], fp32["full"]
    )
    v2.add_adaptive_checks("full", full_candidate, [full_control], checks)
    add_full_decision_checks(full_candidate, full_control, relative_decision, checks)

    anchor = {
        "fp32": v1.call_arrays(fp32["calls"][0]),
        "referenceFP16": v1.call_arrays(reference_fp16["calls"][0]),
        "candidateFP16": v1.call_arrays(candidate_fp16["calls"][0]),
    }
    cross_batch: list[dict[str, Any]] = []
    for index in range(1, len(EXACT_SCHEDULE)):
        batch = EXACT_SCHEDULE[index]
        current = {
            "fp32": v1.call_arrays(fp32["calls"][index]),
            "referenceFP16": v1.call_arrays(reference_fp16["calls"][index]),
            "candidateFP16": v1.call_arrays(candidate_fp16["calls"][index]),
        }
        drifts = {
            label: v1.pair_metrics(
                current[label], {name: values[:batch] for name, values in anchor[label].items()}
            )
            for label in current
        }
        v2.add_adaptive_checks(
            f"crossBatch[{index}].B{batch}",
            drifts["candidateFP16"],
            [drifts["fp32"], drifts["referenceFP16"]],
            checks,
        )
        cross_batch.append({
            "index": index,
            "batch": batch,
            "comparison": "same-arm actual-B vs first-B28 prefix",
            "fp32": drifts["fp32"],
            "referenceFP16": drifts["referenceFP16"],
            "candidateFP16": drifts["candidateFP16"],
        })

    repeated_exact: dict[str, bool] = {}
    full_prefix_exact: dict[str, bool] = {}
    for label, dump in (
        ("fp32", fp32), ("referenceFP16", reference_fp16), ("candidateFP16", candidate_fp16)
    ):
        first = v1.call_arrays(dump["calls"][0])
        last = v1.call_arrays(dump["calls"][-1])
        repeated_exact[label] = all(np.array_equal(first[name], last[name]) for name in first)
        checks.append({"label": f"repeatB28Exact.{label}", "pass": repeated_exact[label]})
        full_prefix_exact[label] = all(
            np.array_equal(dump["full"][name][:PHYSICAL_BATCH_SIZE], first[name]) for name in first
        )
        checks.append({
            "label": f"fullPrefixMatchesDynamicB28Exact.{label}",
            "pass": full_prefix_exact[label],
        })

    loss_values = {
        "fp32": v1.weighted_losses(fp32["full"]["policy"], fp32["full"]["value"], policy_target, global_target),
        "referenceFP16": v1.weighted_losses(
            reference_fp16["full"]["policy"], reference_fp16["full"]["value"], policy_target, global_target
        ),
        "candidateFP16": v1.weighted_losses(
            candidate_fp16["full"]["policy"], candidate_fp16["full"]["value"], policy_target, global_target
        ),
    }
    for name in ("p0loss", "vloss"):
        control_error = abs(loss_values["referenceFP16"][name] - loss_values["fp32"][name])
        candidate_error = abs(loss_values["candidateFP16"][name] - loss_values["fp32"][name])
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

    comparator_path = Path(__file__).resolve()
    base_v1_path = Path(v1.__file__).resolve()
    base_v2_path = Path(v2.__file__).resolve()
    failures = [check for check in checks if not check["pass"]]
    return {
        "schema": REPORT_SCHEMA,
        "profile": PROFILE,
        "createdUtc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pass": not failures,
        "comparator": {
            "sourceRevision": args.comparator_revision,
            "path": str(comparator_path),
            "sha256": v1.file_sha256(comparator_path),
            "baseV1Path": str(base_v1_path),
            "baseV1Sha256": v1.file_sha256(base_v1_path),
            "baseV2Path": str(base_v2_path),
            "baseV2Sha256": v1.file_sha256(base_v2_path),
        },
        "revisionLock": revision_lock,
        "identity": identity,
        "artifacts": {
            "fp32": {key: fp32[key] for key in ("path", "sha256", "revision")},
            "referenceFP16": {key: reference_fp16[key] for key in ("path", "sha256", "revision")},
            "candidateFP16": {key: candidate_fp16[key] for key in ("path", "sha256", "revision")},
            "modelSha256": model_sha,
            "corpus": {"path": str(args.corpus.resolve()), "sha256": actual_corpus_sha},
        },
        "gateRules": {
            "fullCorpus": "three-arm v1 raw/prob adaptive floors and caps plus loss, top1, and mean policy regret",
            "dynamic": "same-arm actual-B drift only; no B1/B2 FP16-vs-FP32 population gate",
            "policyDecision": "reports candidate worse/better selections and aggregate mean regret; no isolated maximum-regret gate",
        },
        "full": {
            "referenceFP16VsFP32": full_control,
            "candidateFP16VsFP32": full_candidate,
            "candidateVsReferenceFP16": full_direct,
            "candidateRelativeToControlDecision": relative_decision,
        },
        "crossBatch": cross_batch,
        "repeatB28Exact": repeated_exact,
        "fullPrefixMatchesDynamicB28Exact": full_prefix_exact,
        "loss": loss_values,
        "checks": checks,
        "failures": failures,
    }


def self_test() -> None:
    def arm(policy: list[float]) -> dict[str, np.ndarray]:
        return {"policy": np.asarray([policy], dtype=np.float64)}

    fp32 = arm([2.0, 1.0, 0.0])
    same_control = arm([0.0, 2.0, 0.0])
    same_candidate = arm([0.0, 3.0, 0.0])
    decision = v2.relative_policy_decision_metrics(same_candidate, same_control, fp32)
    assert decision["selectionDisagreementRows"] == 0
    assert decision["candidateWorseSelectionRows"] == 0
    assert decision["candidateBetterSelectionRows"] == 0
    assert decision["incrementalRegret"]["meanAllRows"] == 0.0
    assert normalized_sha("a" * 40, "test", 40) == "a" * 40
    assert LAYER_COUNT == 36
    assert PHYSICAL_BATCH_SIZE == 28
    assert EXACT_SCHEDULE == [28, 1, 27, 2, 7, 28]

    model_sha = "b" * 64

    def fake_dump(model_version: int, batch: int, using_fp16: int, route: int) -> dict[str, Any]:
        board = 15
        rows = batch
        schedule = [batch, 1, batch - 1, 2, 7, batch]
        meta = {
            "schema": 1,
            "board": board,
            "rows": rows,
            "maxBatch": batch,
            "spatialFeatures": 22,
            "globalFeatures": 39,
            "policyDim": 226,
            "valueDim": 3,
            "scoreValueDim": 6,
            "ownershipDim": 225,
            "usingFP16": using_fp16,
            "nhwc": 1,
            "sourceKind": 1,
            "modelVersion": model_version,
            "inputsVersion": 101,
            "expectedOfficialStage1": route,
            "reserved": 0,
        }
        return {
            "meta": meta,
            "inputIdentity": "c" * 64,
            "modelSha256": model_sha,
            "fullBatches": [batch],
            "schedule": schedule,
        }

    valid_arms = (
        fake_dump(105, 28, 0, ROUTE_NONE),
        fake_dump(105, 28, 1, ROUTE_NONE),
        fake_dump(105, 28, 1, ROUTE_V105_QKN_CLIP4),
    )
    validate_v105_contract(*valid_arms, model_sha)
    revision_result = validate_revision_lock(
        {"revision": f"{REFERENCE_TEST_OVERLAY_REVISION}-cuda"},
        {"revision": f"{REFERENCE_TEST_OVERLAY_REVISION}-cuda"},
        {"revision": f"{'d' * 40}-cuda"},
        "d" * 40,
    )
    assert revision_result["referenceBaseRevision"] == REFERENCE_BASE_REVISION
    assert revision_result["referenceFrozenTestOverlayRevision"] == REFERENCE_TEST_OVERLAY_REVISION
    try:
        validate_revision_lock(
            {"revision": f"{REFERENCE_BASE_REVISION}-dirty-cuda"},
            {"revision": f"{REFERENCE_BASE_REVISION}-dirty-cuda"},
            {"revision": f"{'d' * 40}-cuda"},
            "d" * 40,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("revision lock accepted an unauditable generic c3-dirty overlay")
    v1.validate_structure(valid_arms[0]["meta"], [28], EXACT_SCHEDULE, (105,))
    try:
        v1.validate_structure(valid_arms[0]["meta"], [28], EXACT_SCHEDULE)
    except ValueError:
        pass
    else:
        raise AssertionError("legacy Stage1 parser implicitly accepted model v105")

    def assert_contract_rejected(arms: tuple[dict[str, Any], ...], reason: str) -> None:
        try:
            validate_v105_contract(*arms, model_sha)
        except ValueError:
            return
        raise AssertionError(f"v105 self-test accepted {reason}")

    assert_contract_rejected(
        (
            fake_dump(105, 36, 0, ROUTE_NONE),
            fake_dump(105, 36, 1, ROUTE_NONE),
            fake_dump(105, 36, 1, ROUTE_V105_QKN_CLIP4),
        ),
        "physical batch 36 confused with 36 transformer layers",
    )
    wrong_b36 = fake_dump(105, 36, 0, ROUTE_NONE)
    try:
        v1.validate_structure(wrong_b36["meta"], [36], wrong_b36["schedule"], (105,))
    except ValueError:
        pass
    else:
        raise AssertionError("explicit v105 parser accepted physical batch 36")
    assert_contract_rejected(
        (
            fake_dump(105, 28, 0, ROUTE_NONE),
            fake_dump(105, 28, 1, ROUTE_NONE),
            fake_dump(102, 28, 1, ROUTE_V105_QKN_CLIP4),
        ),
        "mixed v102/v105 arms",
    )
    print(json.dumps({
        "schema": REPORT_SCHEMA,
        "profile": PROFILE,
        "selfTest": "pass",
        "negativeTests": [
            "physical-batch-36", "mixed-v102-v105", "unfrozen-c3-dirty-overlay"
        ],
    }))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32", type=Path)
    parser.add_argument("--reference-fp16", type=Path)
    parser.add_argument("--candidate-fp16", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--expected-model-sha256")
    parser.add_argument("--expected-corpus-sha256")
    parser.add_argument("--candidate-revision")
    parser.add_argument("--comparator-revision")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    required = (
        args.fp32, args.reference_fp16, args.candidate_fp16, args.corpus,
        args.expected_model_sha256, args.expected_corpus_sha256,
        args.candidate_revision, args.comparator_revision, args.output,
    )
    if any(value is None for value in required):
        parser.error(
            "all three dumps, corpus, both SHA-256 locks, candidate/comparator revisions, and output are required"
        )
    try:
        normalized_sha(args.comparator_revision, "comparator-revision", 40)
        result = compare(args)
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "schema": REPORT_SCHEMA,
        "profile": PROFILE,
        "pass": result["pass"],
        "failures": len(result["failures"]),
        "output": str(args.output),
    }))
    if not result["pass"]:
        raise SystemExit("canonical v105 QKN+clip4 numerical gate failed")


if __name__ == "__main__":
    main()

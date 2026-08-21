#!/usr/bin/env python3
"""Compare independent NNRAWG1 dumps and enforce the Stage1 numerical gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np


MAGIC = b"NNRAWG1\0"
SCHEMA = 1
HEADER_FIELDS = (
    "schema", "board", "rows", "maxBatch", "spatialFeatures", "globalFeatures",
    "policyDim", "valueDim", "scoreValueDim", "ownershipDim", "usingFP16", "nhwc",
    "sourceKind", "scheduleCount", "fullCallCount", "expectedOfficialStage1",
    "modelVersion", "inputsVersion", "reserved",
)

ADAPTIVE_SPECS = {
    "policyRaw.rmse": (0.005, 0.05),
    "policyProbabilityTotalVariationMean": (0.001, 0.01),
    "policyProbability.maxAbs": (0.0001, 0.05),
    "policyDecision.decisionRegret.max": (0.0001, 0.01),
    "valueProbability.rmse": (0.001, 0.005),
    "valueProbability.maxAbs": (0.01, 0.10),
    "ownershipProbability.rmse": (0.001, 0.005),
    "ownershipProbability.maxAbs": (0.01, 0.03),
    "scoreValueScaled.p99Abs": (0.005, 0.02),
    "scoreValueScaled.maxAbs": (0.01, 0.05),
}


def read_exact(source: BinaryIO, count: int, what: str) -> bytes:
    data = source.read(count)
    if len(data) != count:
        raise ValueError(f"truncated {what}")
    return data


def read_u32(source: BinaryIO, what: str) -> int:
    return struct.unpack("<I", read_exact(source, 4, what))[0]


def read_u32s(source: BinaryIO, count: int, what: str) -> tuple[int, ...]:
    return struct.unpack(f"<{count}I", read_exact(source, count * 4, what))


def read_floats(source: BinaryIO, rows: int, dim: int, what: str) -> np.ndarray:
    data = read_exact(source, rows * dim * 4, what)
    return np.frombuffer(data, dtype="<f4").reshape(rows, dim).astype(np.float64)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_structure(
    meta: dict[str, Any],
    full_batches: list[int],
    schedule: list[int],
    allowed_model_versions: tuple[int, ...] = (102,),
) -> None:
    if meta["schema"] != SCHEMA or meta["reserved"] != 0:
        raise ValueError("unsupported NNRAWG1 schema or nonzero reserved field")
    board = meta["board"]
    if board not in (15, 19):
        raise ValueError("gate board must be 15 or 19")
    expected_dims = (board * board + 1, 3, 6, board * board)
    actual_dims = (
        meta["policyDim"], meta["valueDim"], meta["scoreValueDim"], meta["ownershipDim"]
    )
    if actual_dims != expected_dims:
        raise ValueError(f"raw head dimensions {actual_dims} do not match v102 {expected_dims}")
    model_contract = (
        meta["modelVersion"], meta["inputsVersion"],
        meta["spatialFeatures"], meta["globalFeatures"],
    )
    supported_contracts = {
        102: (102, 101, 22, 39),
        105: (105, 101, 22, 39),
    }
    if (
        meta["modelVersion"] not in allowed_model_versions
        or supported_contracts.get(meta["modelVersion"]) != model_contract
    ):
        raise ValueError(
            f"gate model contract is not in the explicit allowlist {allowed_model_versions}"
        )
    allowed_routes = (0, 1) if meta["modelVersion"] == 102 else (0, 2)
    if meta["expectedOfficialStage1"] not in allowed_routes:
        raise ValueError("route contract is incompatible with the explicit model contract")
    if meta["modelVersion"] == 105 and (board != 15 or meta["maxBatch"] != 28):
        raise ValueError("v105 gate contract requires board 15 and physical batch 28")
    if meta["rows"] <= 0 or meta["maxBatch"] < 8:
        raise ValueError("invalid row or max-batch count")
    expected_full = [meta["maxBatch"]] * (meta["rows"] // meta["maxBatch"])
    if meta["rows"] % meta["maxBatch"]:
        expected_full.append(meta["rows"] % meta["maxBatch"])
    if full_batches != expected_full:
        raise ValueError(f"full replay actual-batch trace {full_batches} != {expected_full}")
    required = {1, 2, 7, meta["maxBatch"] - 1, meta["maxBatch"]}
    if len(schedule) < 6 or schedule[0] != meta["maxBatch"] or schedule[-1] != meta["maxBatch"]:
        raise ValueError("dynamic schedule must begin and end with max batch")
    if not required.issubset(schedule) or any(batch <= 0 or batch > meta["maxBatch"] for batch in schedule):
        raise ValueError("dynamic schedule does not cover the required actual batches")


def read_dump(path: Path, allowed_model_versions: tuple[int, ...] = (102,)) -> dict[str, Any]:
    with path.open("rb") as source:
        if read_exact(source, 8, "magic") != MAGIC:
            raise ValueError(f"{path}: bad NNRAWG1 magic")
        values = read_u32s(source, len(HEADER_FIELDS), "header")
        meta = dict(zip(HEADER_FIELDS, values))
        input_identity = read_exact(source, 32, "input identity").hex()
        model_sha = read_exact(source, 64, "model sha256").decode("ascii")
        if len(model_sha) != 64 or any(c not in "0123456789abcdef" for c in model_sha):
            raise ValueError(f"{path}: invalid normalized model sha256")
        revision_len = read_u32(source, "revision length")
        if revision_len > 4096:
            raise ValueError(f"{path}: unreasonable revision length")
        revision = read_exact(source, revision_len, "revision").decode("utf-8")
        full_batches = list(read_u32s(source, meta["fullCallCount"], "full batch trace"))
        schedule = list(read_u32s(source, meta["scheduleCount"], "dynamic schedule"))
        validate_structure(meta, full_batches, schedule, allowed_model_versions)
        dims = (
            meta["policyDim"], meta["valueDim"], meta["scoreValueDim"], meta["ownershipDim"]
        )
        names = ("policy", "value", "scoreValue", "ownership")
        full = {
            name: read_floats(source, meta["rows"], dim, f"full {name}")
            for name, dim in zip(names, dims)
        }
        calls: list[dict[str, Any]] = []
        for expected_batch in schedule:
            actual_batch = read_u32(source, "dynamic call batch")
            if actual_batch != expected_batch:
                raise ValueError(f"{path}: dynamic payload batch {actual_batch} != schedule {expected_batch}")
            call = {"batch": actual_batch}
            for name, dim in zip(names, dims):
                call[name] = read_floats(source, actual_batch, dim, f"dynamic {name}")
            calls.append(call)
        if source.read(1):
            raise ValueError(f"{path}: unexpected trailing bytes")
    arrays = list(full.values()) + [call[name] for call in calls for name in names]
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError(f"{path}: non-finite raw output")
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "meta": meta,
        "inputIdentity": input_identity,
        "modelSha256": model_sha,
        "revision": revision,
        "fullBatches": full_batches,
        "schedule": schedule,
        "full": full,
        "calls": calls,
    }


def log_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def softmax(values: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax(values))


def ownership_probability(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(2.0 * values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def error_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    if candidate.shape != reference.shape or candidate.size == 0:
        raise ValueError("metric arrays have different or empty shapes")
    absolute = np.abs(candidate - reference)
    return {
        "rmse": float(np.sqrt(np.mean(absolute * absolute))),
        "meanAbs": float(np.mean(absolute)),
        "p99Abs": float(np.quantile(absolute, 0.99)),
        "maxAbs": float(np.max(absolute)),
    }


def with_top1(metrics: dict[str, float], candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    result = dict(metrics)
    result["top1Agreement"] = float(np.mean(candidate.argmax(axis=1) == reference.argmax(axis=1)))
    return result


def decision_metrics(candidate: np.ndarray, reference: np.ndarray, margin: float = 1e-4) -> dict[str, Any]:
    rows = np.arange(reference.shape[0])
    ref_top1 = reference.argmax(axis=1)
    cand_top1 = candidate.argmax(axis=1)
    top2 = np.partition(reference, -2, axis=1)[:, -2:]
    ref_gap = top2[:, 1] - top2[:, 0]
    regret = reference[rows, ref_top1] - reference[rows, cand_top1]
    decisive = ref_gap > margin
    decisive_rows = int(np.count_nonzero(decisive))
    decisive_top1 = (
        float(np.mean(ref_top1[decisive] == cand_top1[decisive])) if decisive_rows else None
    )
    return {
        "decisionRegret": {"mean": float(np.mean(regret)), "max": float(np.max(regret))},
        "decisiveSubset": {
            "referenceProbabilityMarginThreshold": margin,
            "rows": decisive_rows,
            "top1Agreement": decisive_top1,
        },
    }


def pair_metrics(candidate: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> dict[str, Any]:
    policy_candidate = softmax(candidate["policy"])
    policy_reference = softmax(reference["policy"])
    value_candidate = softmax(candidate["value"])
    value_reference = softmax(reference["value"])
    own_candidate = ownership_probability(candidate["ownership"])
    own_reference = ownership_probability(reference["ownership"])
    score_scaled_candidate = (candidate["scoreValue"] - reference["scoreValue"]) / (
        1.0 + np.abs(reference["scoreValue"])
    )
    score_scaled_reference = np.zeros_like(score_scaled_candidate)
    log_policy_candidate = log_softmax(candidate["policy"])
    log_policy_reference = log_softmax(reference["policy"])
    return {
        "rows": int(candidate["policy"].shape[0]),
        "policyRaw": with_top1(error_metrics(candidate["policy"], reference["policy"]), candidate["policy"], reference["policy"]),
        "policyProbability": with_top1(error_metrics(policy_candidate, policy_reference), policy_candidate, policy_reference),
        "policyProbabilityTotalVariationMean": float(np.mean(0.5 * np.sum(np.abs(policy_candidate - policy_reference), axis=1))),
        "policyProbabilityKLDivergence": {
            "referenceToCandidateMean": float(np.mean(np.sum(policy_reference * (log_policy_reference - log_policy_candidate), axis=1))),
            "candidateToReferenceMean": float(np.mean(np.sum(policy_candidate * (log_policy_candidate - log_policy_reference), axis=1))),
        },
        "policyDecision": decision_metrics(policy_candidate, policy_reference),
        "valueRaw": with_top1(error_metrics(candidate["value"], reference["value"]), candidate["value"], reference["value"]),
        "valueProbability": with_top1(error_metrics(value_candidate, value_reference), value_candidate, value_reference),
        "scoreValueRaw": error_metrics(candidate["scoreValue"], reference["scoreValue"]),
        "scoreValueScaled": error_metrics(score_scaled_candidate, score_scaled_reference),
        "ownershipRaw": error_metrics(candidate["ownership"], reference["ownership"]),
        "ownershipProbability": error_metrics(own_candidate, own_reference),
    }


def get_path(value: dict[str, Any], dotted: str) -> float:
    current: Any = value
    for key in dotted.split("."):
        current = current[key]
    return float(current)


def add_adaptive_checks(
    label: str,
    candidate_metrics: dict[str, Any],
    control_metrics: list[dict[str, Any]],
    checks: list[dict[str, Any]],
) -> None:
    for metric, (floor, cap) in ADAPTIVE_SPECS.items():
        candidate_value = get_path(candidate_metrics, metric)
        control_value = max(get_path(control, metric) for control in control_metrics)
        adaptive_limit = max(floor, 2.0 * control_value)
        passed = (
            math.isfinite(candidate_value) and candidate_value <= adaptive_limit and candidate_value <= cap
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


def add_full_decision_checks(
    label: str,
    candidate_metrics: dict[str, Any],
    control_metrics: dict[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    value_top1 = candidate_metrics["valueProbability"]["top1Agreement"]
    control_value_top1 = control_metrics["valueProbability"]["top1Agreement"]
    checks.append({
        "label": f"{label}.valueProbability.top1Agreement",
        "candidate": value_top1,
        "control": control_value_top1,
        "minimum": max(0.99, control_value_top1 - 0.002),
        "pass": value_top1 >= max(0.99, control_value_top1 - 0.002),
    })
    candidate_decisive = candidate_metrics["policyDecision"]["decisiveSubset"]
    control_decisive = control_metrics["policyDecision"]["decisiveSubset"]
    if candidate_decisive["rows"] >= 16 and control_decisive["rows"] == candidate_decisive["rows"]:
        candidate_top1 = candidate_decisive["top1Agreement"]
        control_top1 = control_decisive["top1Agreement"]
        minimum = max(0.99, control_top1 - 0.002)
        checks.append({
            "label": f"{label}.policyDecision.decisiveTop1Agreement",
            "rows": candidate_decisive["rows"],
            "candidate": candidate_top1,
            "control": control_top1,
            "minimum": minimum,
            "pass": candidate_top1 is not None and candidate_top1 >= minimum,
        })


def read_r15_targets(path: Path) -> tuple[str, np.ndarray, np.ndarray]:
    with path.open("rb") as source:
        if read_exact(source, 8, "R15 magic") != b"R15CORP1":
            raise ValueError("bad R15 corpus magic")
        n, pos, spatial, global_features, packed_width, policy_dim, global_target_dim = read_u32s(source, 7, "R15 header")
        identity = read_exact(source, 32, "R15 identity").hex()
        expected = (15, 22, 39, 29, 226, 64)
        if (pos, spatial, global_features, packed_width, policy_dim, global_target_dim) != expected:
            raise ValueError("unexpected R15 corpus dimensions")
        read_exact(source, n * spatial * packed_width, "R15 packed input")
        read_exact(source, n * global_features * 4, "R15 global input")
        policy = np.frombuffer(read_exact(source, n * 2 * policy_dim * 2, "R15 policy targets"), dtype="<i2").reshape(n, 2, policy_dim).astype(np.float64)
        global_target = np.frombuffer(read_exact(source, n * global_target_dim * 4, "R15 global targets"), dtype="<f4").reshape(n, global_target_dim).astype(np.float64)
        if source.read(1):
            raise ValueError("unexpected trailing R15 corpus bytes")
    return identity, policy, global_target


def weighted_losses(
    policy_logits: np.ndarray,
    value_logits: np.ndarray,
    policy_target: np.ndarray,
    global_target: np.ndarray,
) -> dict[str, float]:
    target_policy = policy_target[:, 0, :].copy()
    target_policy /= target_policy.sum(axis=1, keepdims=True)
    global_weight = global_target[:, 25]
    policy_weight = global_target[:, 26]
    value_weight = 1.0 - global_target[:, 35]
    target_value = global_target[:, :3]
    p0_rows = -np.sum(target_policy * log_softmax(policy_logits), axis=1)
    v_rows = 1.20 * -np.sum(target_value * log_softmax(value_logits), axis=1)
    weight_sum = float(global_weight.sum())
    return {
        "p0loss": float(np.sum(global_weight * policy_weight * p0_rows) / weight_sum),
        "vloss": float(np.sum(global_weight * value_weight * v_rows) / weight_sum),
        "weightSum": weight_sum,
    }


def identity_view(dump: dict[str, Any]) -> dict[str, Any]:
    meta = dump["meta"]
    keys = (
        "schema", "board", "rows", "maxBatch", "spatialFeatures", "globalFeatures",
        "policyDim", "valueDim", "scoreValueDim", "ownershipDim", "nhwc", "sourceKind",
        "modelVersion", "inputsVersion",
    )
    return {
        "meta": {key: meta[key] for key in keys},
        "inputIdentity": dump["inputIdentity"],
        "modelSha256": dump["modelSha256"],
        "fullBatches": dump["fullBatches"],
        "schedule": dump["schedule"],
    }


def call_arrays(call: dict[str, Any]) -> dict[str, np.ndarray]:
    return {name: call[name] for name in ("policy", "value", "scoreValue", "ownership")}


def validate_arm_provenance(
    fp32: dict[str, Any], reference_fp16: dict[str, Any], candidate_fp16: dict[str, Any]
) -> None:
    reference_revision = fp32["revision"]
    if not reference_revision or not reference_fp16["revision"] or not candidate_fp16["revision"]:
        raise ValueError("all three arms must embed a nonempty source revision")
    if reference_fp16["revision"] != reference_revision:
        raise ValueError("FP32 and FP16 reference arms must come from the same predecessor revision")
    if candidate_fp16["revision"] == reference_revision:
        raise ValueError("candidate arm must not come from the predecessor revision (candidate-vs-itself gate)")


def validate_arm_layout(
    fp32: dict[str, Any], reference_fp16: dict[str, Any], candidate_fp16: dict[str, Any]
) -> None:
    if any(dump["meta"]["nhwc"] != 1 for dump in (fp32, reference_fp16, candidate_fp16)):
        raise ValueError("all three Stage1 arms must use NHWC inputs")


def compare(args: argparse.Namespace) -> dict[str, Any]:
    fp32 = read_dump(args.fp32)
    reference_fp16 = read_dump(args.reference_fp16)
    candidate_fp16 = read_dump(args.candidate_fp16)
    if any(
        dump["meta"]["modelVersion"] != 102
        for dump in (fp32, reference_fp16, candidate_fp16)
    ):
        raise ValueError("schema-1 comparator profile is restricted to Stage1 model v102")
    validate_arm_provenance(fp32, reference_fp16, candidate_fp16)
    validate_arm_layout(fp32, reference_fp16, candidate_fp16)
    identity = identity_view(fp32)
    if identity_view(reference_fp16) != identity or identity_view(candidate_fp16) != identity:
        raise ValueError("model/input/shape/full-batch/schedule identities differ across implementations")
    if fp32["meta"]["usingFP16"] != 0 or reference_fp16["meta"]["usingFP16"] != 1 or candidate_fp16["meta"]["usingFP16"] != 1:
        raise ValueError("precision-arm identities must be FP32/FP16/FP16")
    if fp32["meta"]["expectedOfficialStage1"] != 0 or reference_fp16["meta"]["expectedOfficialStage1"] != 0:
        raise ValueError("reference arms must not claim the candidate official-route assertion")
    if candidate_fp16["meta"]["expectedOfficialStage1"] != 1:
        raise ValueError("candidate arm lacks the in-process official-route assertion")

    checks: list[dict[str, Any]] = []
    full_control = pair_metrics(reference_fp16["full"], fp32["full"])
    full_candidate = pair_metrics(candidate_fp16["full"], fp32["full"])
    full_direct = pair_metrics(candidate_fp16["full"], reference_fp16["full"])
    add_adaptive_checks("full", full_candidate, [full_control], checks)
    add_full_decision_checks("full", full_candidate, full_control, checks)

    dynamic: list[dict[str, Any]] = []
    for index, batch in enumerate(fp32["schedule"]):
        fp32_call = call_arrays(fp32["calls"][index])
        reference_call = call_arrays(reference_fp16["calls"][index])
        candidate_call = call_arrays(candidate_fp16["calls"][index])
        control = pair_metrics(reference_call, fp32_call)
        candidate = pair_metrics(candidate_call, fp32_call)
        direct = pair_metrics(candidate_call, reference_call)
        add_adaptive_checks(f"dynamic[{index}].B{batch}", candidate, [control], checks)
        dynamic.append({
            "index": index,
            "batch": batch,
            "referenceFP16VsFP32": control,
            "candidateFP16VsFP32": candidate,
            "candidateVsReferenceFP16": direct,
        })

    anchor_fp32 = call_arrays(fp32["calls"][0])
    anchor_reference = call_arrays(reference_fp16["calls"][0])
    anchor_candidate = call_arrays(candidate_fp16["calls"][0])
    cross_batch: list[dict[str, Any]] = []
    for index in range(1, len(fp32["schedule"])):
        batch = fp32["schedule"][index]
        fp32_call = call_arrays(fp32["calls"][index])
        reference_call = call_arrays(reference_fp16["calls"][index])
        candidate_call = call_arrays(candidate_fp16["calls"][index])
        fp32_anchor_prefix = {name: values[:batch] for name, values in anchor_fp32.items()}
        reference_anchor_prefix = {name: values[:batch] for name, values in anchor_reference.items()}
        candidate_anchor_prefix = {name: values[:batch] for name, values in anchor_candidate.items()}
        fp32_drift = pair_metrics(fp32_call, fp32_anchor_prefix)
        reference_drift = pair_metrics(reference_call, reference_anchor_prefix)
        candidate_drift = pair_metrics(candidate_call, candidate_anchor_prefix)
        add_adaptive_checks(
            f"crossBatch[{index}].B{batch}", candidate_drift, [fp32_drift, reference_drift], checks
        )
        cross_batch.append({
            "index": index,
            "batch": batch,
            "fp32DriftFromFirstMaxBatch": fp32_drift,
            "referenceFP16DriftFromFirstMaxBatch": reference_drift,
            "candidateFP16DriftFromFirstMaxBatch": candidate_drift,
        })

    repeated_exact: dict[str, bool] = {}
    for label, dump in (("fp32", fp32), ("referenceFP16", reference_fp16), ("candidateFP16", candidate_fp16)):
        first = call_arrays(dump["calls"][0])
        last = call_arrays(dump["calls"][-1])
        exact = all(np.array_equal(first[name], last[name]) for name in first)
        repeated_exact[label] = exact
        checks.append({"label": f"repeatMaxBatchExact.{label}", "pass": exact})
        full_prefix_exact = all(
            np.array_equal(dump["full"][name][: dump["meta"]["maxBatch"]], first[name])
            for name in first
        )
        checks.append({"label": f"fullPrefixMatchesDynamicMaxExact.{label}", "pass": full_prefix_exact})

    loss: dict[str, Any] | None = None
    if args.corpus is not None:
        corpus_identity, policy_target, global_target = read_r15_targets(args.corpus)
        if corpus_identity != fp32["inputIdentity"] or policy_target.shape[0] != fp32["meta"]["rows"]:
            raise ValueError("R15 target corpus identity/rows differ from raw dumps")
        loss_fp32 = weighted_losses(fp32["full"]["policy"], fp32["full"]["value"], policy_target, global_target)
        loss_reference = weighted_losses(reference_fp16["full"]["policy"], reference_fp16["full"]["value"], policy_target, global_target)
        loss_candidate = weighted_losses(candidate_fp16["full"]["policy"], candidate_fp16["full"]["value"], policy_target, global_target)
        loss = {"fp32": loss_fp32, "referenceFP16": loss_reference, "candidateFP16": loss_candidate}
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

    failures = [check for check in checks if not check["pass"]]
    return {
        "schema": 1,
        "createdUtc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pass": not failures,
        "identity": identity,
        "artifacts": {
            "fp32": {key: fp32[key] for key in ("path", "sha256", "revision")},
            "referenceFP16": {key: reference_fp16[key] for key in ("path", "sha256", "revision")},
            "candidateFP16": {key: candidate_fp16[key] for key in ("path", "sha256", "revision")},
            "corpus": None if args.corpus is None else {"path": str(args.corpus.resolve()), "sha256": file_sha256(args.corpus)},
        },
        "thresholdRule": "candidate_error_vs_fp32 <= max(floor,2*reference_fp16_error_vs_fp32) and <= absolute_cap",
        "full": {
            "referenceFP16VsFP32": full_control,
            "candidateFP16VsFP32": full_candidate,
            "candidateVsReferenceFP16": full_direct,
        },
        "dynamic": dynamic,
        "crossBatch": cross_batch,
        "repeatMaxBatchExact": repeated_exact,
        "loss": loss,
        "checks": checks,
        "failures": failures,
    }


def self_test() -> None:
    reference = {
        "policy": np.array([[0.0, 1.0, -1.0], [1.0, 0.5, -0.5]]),
        "value": np.array([[1.0, 0.0, -1.0], [0.2, 0.3, 0.1]]),
        "scoreValue": np.zeros((2, 6)),
        "ownership": np.zeros((2, 2)),
    }
    candidate = {name: values.copy() for name, values in reference.items()}
    candidate["policy"][0, 0] += 1e-4
    result = pair_metrics(candidate, reference)
    assert result["policyRaw"]["rmse"] > 0
    assert result["policyProbabilityTotalVariationMean"] > 0
    checks: list[dict[str, Any]] = []
    zero = pair_metrics(reference, reference)
    add_adaptive_checks("self", result, [zero], checks)
    assert all(check["pass"] for check in checks)
    assert softmax(reference["policy"]).shape == reference["policy"].shape
    validate_arm_provenance(
        {"revision": "reference-cuda"},
        {"revision": "reference-cuda"},
        {"revision": "candidate-cuda"},
    )
    try:
        validate_arm_provenance(
            {"revision": "same-cuda"}, {"revision": "same-cuda"}, {"revision": "same-cuda"}
        )
    except ValueError:
        pass
    else:
        raise AssertionError("candidate-vs-itself provenance must fail")
    nhwc_arm = {"meta": {"nhwc": 1}}
    validate_arm_layout(nhwc_arm, nhwc_arm, nhwc_arm)
    try:
        validate_arm_layout({"meta": {"nhwc": 0}}, nhwc_arm, nhwc_arm)
    except ValueError:
        pass
    else:
        raise AssertionError("FP32 NCHW mixed-layout gate must fail")
    print(json.dumps({"schema": 1, "selfTest": "pass", "checks": len(checks) + 4}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32", type=Path)
    parser.add_argument("--reference-fp16", type=Path)
    parser.add_argument("--candidate-fp16", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    required = (args.fp32, args.reference_fp16, args.candidate_fp16, args.output)
    if any(value is None for value in required):
        parser.error("--fp32, --reference-fp16, --candidate-fp16, and --output are required")
    result = compare(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"pass": result["pass"], "failures": len(result["failures"]), "output": str(args.output)}))
    if not result["pass"]:
        raise SystemExit("nnrawgate numerical gate failed")


if __name__ == "__main__":
    main()

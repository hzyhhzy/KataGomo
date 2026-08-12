#!/usr/bin/env python3
"""Manifest-driven Renju15 SM120 production acceptance runner.

The runner intentionally executes argv arrays without a shell.  A manifest can
therefore bind the final production binary and marker spellings without
changing this evidence-producing code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import statistics
import subprocess
import sys
from typing import Any


class AcceptanceError(RuntimeError):
    pass


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expand(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, str):
        try:
            return value.format_map(variables)
        except KeyError as exc:
            raise AcceptanceError(f"unknown manifest variable {exc.args[0]!r}") from exc
    if isinstance(value, list):
        return [expand(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: expand(item, variables) for key, item in value.items()}
    return value


def resolve_variables(raw: dict[str, Any], manifest_dir: pathlib.Path) -> dict[str, str]:
    variables: dict[str, str] = {"manifestDir": str(manifest_dir)}
    pending = dict(raw)
    while pending:
        progressed = False
        for key, value in list(pending.items()):
            if not isinstance(value, (str, int, float)):
                raise AcceptanceError(f"variable {key!r} must be scalar")
            try:
                variables[key] = str(value).format_map(variables)
            except KeyError:
                continue
            del pending[key]
            progressed = True
        if not progressed:
            raise AcceptanceError(
                "unresolved or cyclic manifest variables: " + ", ".join(sorted(pending))
            )
    return variables


def check_markers(text: str, spec: dict[str, Any], label: str) -> None:
    for marker in spec.get("requiredMarkers", []):
        minimum = 1
        needle = marker
        if isinstance(marker, dict):
            needle = marker["text"]
            minimum = int(marker.get("minCount", 1))
        actual = text.count(str(needle))
        if actual < minimum:
            raise AcceptanceError(
                f"{label}: required marker count {actual} < {minimum}: {needle}"
            )
    for marker in spec.get("forbiddenMarkers", []):
        if str(marker).lower() in text.lower():
            raise AcceptanceError(f"{label}: forbidden marker present: {marker}")


def parse_last_json_line(text: str, label: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise AcceptanceError(f"{label}: no one-line JSON object found in output")


def run_command(
    label: str,
    raw_spec: dict[str, Any],
    variables: dict[str, str],
    result_dir: pathlib.Path,
) -> dict[str, Any]:
    spec = expand(raw_spec, variables)
    argv = spec.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(arg, str) for arg in argv):
        raise AcceptanceError(f"{label}: argv must be a nonempty string array")
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in spec.get("env", {}).items()})
    cwd = pathlib.Path(spec.get("cwd", variables["manifestDir"]))
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = result_dir / f"{label}.log"
    print(f"[{label}] {' '.join(argv)}", flush=True)
    completed = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise AcceptanceError(
            f"{label}: command exited {completed.returncode}; see {log_path}"
        )
    check_markers(completed.stdout, spec, label)
    result: dict[str, Any] = {
        "argv": argv,
        "cwd": str(cwd),
        "returnCode": completed.returncode,
        "log": str(log_path),
        "logSha256": sha256_file(log_path),
    }
    if spec.get("parseJson", False):
        result["json"] = parse_last_json_line(completed.stdout, label)
    return result


def dotted_get(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for component in path.split("."):
        if not isinstance(current, dict) or component not in current:
            raise AcceptanceError(f"report has no field {path!r}")
        current = current[component]
    return current


def apply_gates(value: dict[str, Any], gates: list[dict[str, Any]], label: str) -> None:
    for gate in gates:
        path = gate["path"]
        actual = dotted_get(value, path)
        if "equals" in gate and actual != gate["equals"]:
            raise AcceptanceError(f"{label}: {path}={actual!r}, expected {gate['equals']!r}")
        if "min" in gate and float(actual) < float(gate["min"]):
            raise AcceptanceError(f"{label}: {path}={actual} < {gate['min']}")
        if "max" in gate and float(actual) > float(gate["max"]):
            raise AcceptanceError(f"{label}: {path}={actual} > {gate['max']}")
        if "absMax" in gate and abs(float(actual)) > float(gate["absMax"]):
            raise AcceptanceError(f"{label}: abs({path})={abs(float(actual))} > {gate['absMax']}")


def verify_artifacts(
    artifacts: list[dict[str, Any]], variables: dict[str, str]
) -> list[dict[str, Any]]:
    evidence = []
    for raw in artifacts:
        artifact = expand(raw, variables)
        path = pathlib.Path(artifact["path"])
        if not path.is_file():
            raise AcceptanceError(f"missing artifact {artifact['name']}: {path}")
        actual = sha256_file(path)
        expected = str(artifact["sha256"]).lower()
        if actual != expected:
            raise AcceptanceError(
                f"artifact SHA mismatch {artifact['name']}: {actual} != {expected}"
            )
        evidence.append({"name": artifact["name"], "path": str(path), "sha256": actual})
    return evidence


def run_accuracy(
    manifest: dict[str, Any], variables: dict[str, str], result_dir: pathlib.Path
) -> dict[str, Any]:
    commands = manifest["commands"]
    result = {
        "referenceReplay": run_command(
            "accuracy_reference", commands["referenceReplay"], variables, result_dir
        ),
        "candidateReplay": run_command(
            "accuracy_candidate", commands["candidateReplay"], variables, result_dir
        ),
    }
    result["compare"] = run_command(
        "accuracy_compare", commands["compareAccuracy"], variables, result_dir
    )
    report_path = pathlib.Path(variables["accuracyReport"])
    if not report_path.is_file():
        raise AcceptanceError(f"accuracy comparator did not write {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    apply_gates(report, manifest.get("accuracyGates", []), "accuracy")
    result["metrics"] = report
    result["report"] = str(report_path)
    result["reportSha256"] = sha256_file(report_path)
    return result


def run_perf(
    manifest: dict[str, Any], variables: dict[str, str], result_dir: pathlib.Path
) -> dict[str, Any]:
    perf = manifest["perf"]
    sequence = perf.get("sequence", ["production", "experimental", "experimental", "production"])
    rounds = int(perf.get("rounds", 1))
    if rounds <= 0 or not sequence:
        raise AcceptanceError("perf rounds and sequence must be nonempty")
    samples: dict[str, list[float]] = {"production": [], "experimental": []}
    legs = []
    for round_index in range(rounds):
        for leg_index, candidate in enumerate(sequence):
            if candidate not in samples:
                raise AcceptanceError(f"unknown perf candidate {candidate!r}")
            command_name = f"{candidate}Benchmark"
            label = f"perf_r{round_index:02d}_l{leg_index:02d}_{candidate}"
            leg = run_command(label, manifest["commands"][command_name], variables, result_dir)
            bench = leg.get("json")
            if not isinstance(bench, dict):
                raise AcceptanceError(f"{label}: benchmark command must set parseJson=true")
            apply_gates(bench, perf.get("benchmarkGates", []), label)
            throughput = float(bench["combinedNNEvalsPerSec"])
            if throughput <= 0.0:
                raise AcceptanceError(f"{label}: nonpositive throughput {throughput}")
            samples[candidate].append(throughput)
            legs.append({"candidate": candidate, "throughput": throughput, **leg})
    if not samples["production"] or not samples["experimental"]:
        raise AcceptanceError("perf sequence must contain both production and experimental")
    production = statistics.mean(samples["production"])
    experimental = statistics.mean(samples["experimental"])
    ratio = production / experimental
    minimum = float(perf.get("minProductionOverExperimental", 0.99))
    if ratio < minimum:
        raise AcceptanceError(
            f"production/experimental throughput {ratio:.6f} < required {minimum:.6f}"
        )
    return {
        "sequence": sequence,
        "rounds": rounds,
        "legs": legs,
        "samples": samples,
        "productionMean": production,
        "experimentalMean": experimental,
        "productionOverExperimental": ratio,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("contract", "accuracy", "perf"),
        default=("contract", "accuracy", "perf"),
    )
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    variables = resolve_variables(manifest.get("variables", {}), manifest_path.parent)
    variables = {key: str(expand(value, variables)) for key, value in variables.items()}
    result_dir = pathlib.Path(variables["resultDir"])
    result_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema": "renju15-sm120-production-acceptance-v1",
        "manifest": str(manifest_path),
        "manifestSha256": sha256_file(manifest_path),
        "artifacts": verify_artifacts(manifest.get("artifacts", []), variables),
        "stages": {},
    }
    if "contract" in args.stages:
        report["stages"]["contract"] = run_command(
            "cpu_contract", manifest["commands"]["contract"], variables, result_dir
        )
    if "accuracy" in args.stages:
        report["stages"]["accuracy"] = run_accuracy(manifest, variables, result_dir)
    if "perf" in args.stages:
        report["stages"]["perf"] = run_perf(manifest, variables, result_dir)

    report_path = result_dir / "acceptance_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report_sha = sha256_file(report_path)
    (result_dir / "acceptance_report.sha256").write_text(
        f"{report_sha}  {report_path.name}\n", encoding="utf-8"
    )
    print(json.dumps({"report": str(report_path), "sha256": report_sha}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AcceptanceError as exc:
        print(f"ACCEPTANCE FAILED: {exc}", file=sys.stderr)
        raise SystemExit(2)

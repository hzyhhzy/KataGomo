#!/usr/bin/env python3
"""Run the six-case Renju15 SM120 benchmarknn matrix reproducibly.

This is an evidence runner, not a tuner. It invokes one final KataGo binary,
lets KataGo select the CUDA path from the model/runtime shape, captures the
complete log for every process, and emits a machine-readable aggregate report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence


CASE_IDS = ("A", "B", "C", "D", "E", "F")
BALANCED_ROUNDS = (
    ("A", "B", "F", "C", "E", "D"),
    ("B", "C", "A", "D", "F", "E"),
    ("C", "D", "B", "E", "A", "F"),
    ("D", "E", "C", "F", "B", "A"),
    ("E", "F", "D", "A", "C", "B"),
    ("F", "A", "E", "B", "D", "C"),
)


class MatrixError(RuntimeError):
    pass


@dataclass(frozen=True)
class Case:
    case_id: str
    model_shape: str
    model_key: str
    batch_size: int
    board_size: int
    force_mask_all_ones: bool = False
    reference_nnevals_per_sec: float | None = None

    @property
    def label(self) -> str:
        mask = "mask1" if self.force_mask_all_ones else "nomask"
        return f"{self.model_shape}_bs{self.batch_size}_{self.board_size}x{self.board_size}_{mask}"


CASES = {
    "A": Case("A", "b36c384", "b36c384", 28, 15, reference_nnevals_per_sec=6950.0),
    "B": Case("B", "b24c256", "b24c256", 36, 15, reference_nnevals_per_sec=20000.0),
    "C": Case("C", "b24c384", "b24c384", 28, 15),
    "D": Case("D", "b36c384", "b36c384", 24, 15),
    "E": Case("E", "b36c384", "b36c384", 28, 19),
    "F": Case("F", "b36c384", "b36c384", 28, 15, force_mask_all_ones=True),
}


def validate_schedule(rounds: Sequence[Sequence[str]] = BALANCED_ROUNDS) -> None:
    expected = set(CASE_IDS)
    if len(rounds) != len(CASE_IDS):
        raise MatrixError("a complete position-balanced schedule must have six rounds")
    for round_cases in rounds:
        if len(round_cases) != len(CASE_IDS) or set(round_cases) != expected:
            raise MatrixError(f"invalid balanced round: {round_cases!r}")
    for position in range(len(CASE_IDS)):
        if {round_cases[position] for round_cases in rounds} != expected:
            raise MatrixError(f"position {position + 1} is not balanced")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_last_json_line(text: str, label: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise MatrixError(f"{label}: no one-line JSON object in output")


def benchmark_override(case: Case, gpu_index: int) -> str:
    values = (
        ("nnMaxBatchSize", str(case.batch_size)),
        ("numNNServerThreadsPerModel", "2"),
        ("gpuToUseThread0", str(gpu_index)),
        ("gpuToUseThread1", str(gpu_index)),
        ("requireMaxBoardSize", "true"),
        ("maxBoardSizeForNNBuffer", str(case.board_size)),
        ("useFP16", "true"),
        ("useNHWC", "true"),
        ("nnRandomize", "false"),
        ("nnCacheSizePowerOfTwo", "-1"),
        ("cudaUseINT8", "false"),
    )
    return ",".join(f"{key}={value}" for key, value in values)


def build_argv(
    binary: pathlib.Path,
    config: pathlib.Path,
    model: pathlib.Path,
    case: Case,
    warmup: int,
    iterations: int,
    gpu_index: int,
) -> list[str]:
    # KataGo intentionally compiles TCLAP with TCLAP_NAMESTARTSTRING="-".
    # These are long option names with one dash; GNU-style two-dash spellings
    # are not aliases in this command-line parser.
    argv = [
        str(binary),
        "benchmarknn",
        "-model",
        str(model),
        "-config",
        str(config),
        "-boardsize",
        str(case.board_size),
        "-batch-size",
        str(case.batch_size),
        "-warmup",
        str(warmup),
        "-iterations",
        str(iterations),
        "-override-config",
        benchmark_override(case, gpu_index),
    ]
    if case.force_mask_all_ones:
        argv.append("-force-mask-all-ones")
    argv.append("-json")
    return argv


def required_markers(case: Case) -> tuple[tuple[str, int], ...]:
    if case.case_id in ("A", "C"):
        depth = 36 if case.case_id == "A" else 24
        transaction = (
            f"KATAGO_C384_EXACT_FIXED_ACTIVE batch=28 depth={depth} "
            f"qkv_fa4={depth}/{depth} dual_ffn={depth}/{depth} "
            f"ffn_down={depth}/{depth}"
        )
        return (
            (transaction, 2),
            ("KATAGO_C384_EXACT_QKV_FA4_ACTIVE batch=28", 2),
            ("KATAGO_C384_EXACT_DUAL_FFN_ACTIVE batch=28", 2),
            ("KATAGO_C384_EXACT_FFN_DOWN_ACTIVE batch=28", 2),
        )
    if case.case_id == "B":
        return (
            ("CUDA_TRANSFORMER_WINNER_ACTIVE qualification=certified-fast plan=", 2),
            ("RENJU15_SM120_RMS_ACTIVE marker=warp4-vec8", 2),
            ("RENJU15_SM120_QKV_ROPE_ACTIVE marker=qkv-rope-m128-n128-k32-s3-sw1", 2),
            ("RENJU15_SM120_FA4_ACTIVE marker=tm128-tn128-s1-both16-b36-s225-h8-d32", 2),
            ("RENJU15_SM120_DUAL_FFN_ACTIVE marker=dual-ffn-c256-f768-m128-n64-k32-s3-sw4", 2),
            ("RENJU15_SM120_RESIDUAL_GEMM_ACTIVE family=out-proj marker=m128-n128-k32-s3-sw1", 2),
            ("RENJU15_SM120_RESIDUAL_GEMM_ACTIVE family=ffn-down marker=m128-n128-k32-s3-sw1", 2),
        )
    return ()


def forbidden_markers(case: Case) -> tuple[str, ...]:
    if case.case_id in ("D", "E", "F"):
        # PREPARED/fallback is diagnostic and is allowed. ACTIVE would mean the
        # non-control variable accidentally entered the bs28/15/no-mask path.
        return (
            "KATAGO_C384_EXACT_FIXED_ACTIVE",
            "KATAGO_C384_EXACT_QKV_FA4_ACTIVE",
            "KATAGO_C384_EXACT_DUAL_FFN_ACTIVE",
            "KATAGO_C384_EXACT_FFN_DOWN_ACTIVE",
        )
    return ()


def validate_benchmark(
    case: Case,
    result: dict[str, Any],
    output: str,
    iterations: int,
    gpu_index: int,
    label: str,
    expected_model: pathlib.Path | None = None,
) -> None:
    expected_scalars = {
        "batchSize": case.batch_size,
        "numServerThreads": 2,
        "numIterations": iterations,
        "forceMaskAllOnes": case.force_mask_all_ones,
    }
    for key, expected in expected_scalars.items():
        if result.get(key) != expected:
            raise MatrixError(f"{label}: {key}={result.get(key)!r}, expected {expected!r}")
    if result.get("gpuIdxs") != [gpu_index]:
        raise MatrixError(f"{label}: gpuIdxs={result.get('gpuIdxs')!r}, expected [{gpu_index}]")
    if "cuda" not in str(result.get("revision", "")).lower():
        raise MatrixError(f"{label}: revision does not identify the CUDA backend")
    if expected_model is not None:
        actual_model = pathlib.Path(str(result.get("modelFile", ""))).resolve()
        if actual_model != expected_model.resolve():
            raise MatrixError(
                f"{label}: modelFile={actual_model}, expected {expected_model.resolve()}"
            )
    for key in ("combinedWallSeconds", "combinedNNEvalsPerSec"):
        try:
            value = float(result[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise MatrixError(f"{label}: invalid {key}") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise MatrixError(f"{label}: nonpositive/nonfinite {key}={value}")
    for key in ("perServerMedianMs", "perServerNNEvalsPerSec"):
        values = result.get(key)
        if not isinstance(values, list) or len(values) != 2:
            raise MatrixError(f"{label}: {key} must contain two S2 lanes")
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in values):
            raise MatrixError(f"{label}: invalid {key} values")
    board_marker = f"benchmarknn board size {case.board_size}"
    if board_marker not in output:
        raise MatrixError(f"{label}: missing board-size log marker {board_marker!r}")
    mode_marker = " useFP16 true useNHWC true useINT8 false"
    if mode_marker not in output:
        raise MatrixError(f"{label}: missing FP16/NHWC/INT8 mode marker")
    buffer_marker = (
        f"Initializing neural net buffer to be size {case.board_size} * "
        f"{case.board_size} exactly"
    )
    if buffer_marker not in output:
        raise MatrixError(f"{label}: missing exact NN-buffer marker {buffer_marker!r}")
    for marker, minimum in required_markers(case):
        count = output.count(marker)
        if count < minimum:
            raise MatrixError(f"{label}: marker count {count} < {minimum}: {marker}")
    lowered = output.lower()
    for marker in forbidden_markers(case):
        if marker.lower() in lowered:
            raise MatrixError(f"{label}: forbidden active marker present: {marker}")


def summarize_samples(values: Sequence[float]) -> dict[str, float | int | bool]:
    if not values:
        raise MatrixError("cannot summarize no samples")
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) >= 2 else 0.0
    cv_percent = 100.0 * stdev / mean if mean else math.inf
    endpoint = 100.0 * (values[-1] - values[0]) / mean if mean else math.inf
    return {
        "count": len(values),
        "mean": mean,
        "median": statistics.median(values),
        "sampleStdev": stdev,
        "cvPercent": cv_percent,
        "minimum": min(values),
        "maximum": max(values),
        "endpointDriftPercent": endpoint,
        "absEndpointDriftPercent": abs(endpoint),
    }


def geometric_mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values or any(value <= 0.0 for value in values):
        raise MatrixError("geometric mean requires positive samples")
    return math.exp(statistics.mean(math.log(value) for value in values))


def summarize_report(report: dict[str, Any], max_cv: float, max_drift: float) -> None:
    legs = report["formalLegs"]
    samples: dict[str, list[float]] = {case_id: [] for case_id in CASE_IDS}
    by_round: dict[str, dict[str, float]] = {}
    for leg in legs:
        case_id = leg["caseId"]
        throughput = float(leg["json"]["combinedNNEvalsPerSec"])
        samples[case_id].append(throughput)
        by_round.setdefault(leg["roundKey"], {})[case_id] = throughput

    summaries: dict[str, Any] = {}
    for case_id, values in samples.items():
        summary = summarize_samples(values)
        case = CASES[case_id]
        summary["case"] = asdict(case)
        summary["label"] = case.label
        if case.reference_nnevals_per_sec is not None:
            summary["referenceNNEvalsPerSec"] = case.reference_nnevals_per_sec
            summary["meanOverReference"] = float(summary["mean"]) / case.reference_nnevals_per_sec
        summary["stability"] = {
            "maxCvPercent": max_cv,
            "maxAbsEndpointDriftPercent": max_drift,
            "cvPass": float(summary["cvPercent"]) <= max_cv,
            "endpointPass": float(summary["absEndpointDriftPercent"]) <= max_drift,
        }
        summaries[case_id] = summary

    paired: dict[str, Any] = {}
    for case_id in CASE_IDS:
        ratios = []
        for round_key in sorted(by_round):
            values = by_round[round_key]
            if set(values) != set(CASE_IDS):
                raise MatrixError(f"incomplete formal round {round_key}: {sorted(values)}")
            ratios.append(values[case_id] / values["A"])
        paired[case_id] = {
            "caseOverAByRound": ratios,
            "caseOverAGeometricMean": geometric_mean(ratios),
            "caseOverAMinimum": min(ratios),
            "caseOverAMaximum": max(ratios),
        }
    report["caseSummaries"] = summaries
    report["pairedRatiosVsA"] = paired
    report["stabilityPass"] = all(
        item["stability"]["cvPass"] and item["stability"]["endpointPass"]
        for item in summaries.values()
    )


def query_gpu_processes(nvidia_smi: str, gpu_index: int) -> list[dict[str, Any]]:
    command = [
        nvidia_smi,
        f"--id={gpu_index}",
        "--query-compute-apps=pid,process_name",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise MatrixError(
            f"nvidia-smi process query failed ({completed.returncode}): {completed.stderr.strip()}"
        )
    processes = []
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line or "no running processes" in line.lower():
            continue
        fields = [field.strip() for field in line.split(",", 1)]
        if len(fields) != 2 or not fields[0].isdigit():
            raise MatrixError(f"cannot parse nvidia-smi process row: {raw_line!r}")
        processes.append({"pid": int(fields[0]), "processName": fields[1]})
    return processes


def assert_gpu_idle(nvidia_smi: str, gpu_index: int, label: str) -> None:
    processes = query_gpu_processes(nvidia_smi, gpu_index)
    if processes:
        raise MatrixError(
            f"{label}: GPU {gpu_index} is not idle; refusing to kill unrelated processes: {processes}"
        )


def terminate_process_group(process: subprocess.Popen[str], grace_seconds: float = 5.0) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGKILL)
    else:
        process.kill()
    process.wait(timeout=grace_seconds)


def clean_environment() -> tuple[dict[str, str], list[str]]:
    env = os.environ.copy()
    prefixes = ("KATAGO_EXPERIMENTAL_", "KATAGO_RENJU15_SM120_", "KATAGO_C384_")
    removed = sorted(key for key in env if key.startswith(prefixes))
    for key in removed:
        del env[key]
    return env, removed


def run_leg(
    *,
    label: str,
    argv: Sequence[str],
    cwd: pathlib.Path,
    log_path: pathlib.Path,
    case: Case,
    iterations: int,
    gpu_index: int,
    timeout_seconds: float,
    nvidia_smi: str,
    skip_gpu_idle_check: bool,
    env: dict[str, str],
) -> dict[str, Any]:
    if not skip_gpu_idle_check:
        assert_gpu_idle(nvidia_smi, gpu_index, f"before {label}")
    print(f"[{label}] {case.label}", flush=True)
    started = time.time()
    process = subprocess.Popen(
        list(argv),
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=(os.name == "posix"),
    )
    try:
        output, _ = process.communicate(timeout=timeout_seconds)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        terminate_process_group(process)
        output, _ = process.communicate()
        log_path.write_text(output, encoding="utf-8")
        raise
    log_path.write_text(output, encoding="utf-8")
    elapsed = time.time() - started
    if process.returncode != 0:
        raise MatrixError(f"{label}: exit {process.returncode}; see {log_path}")
    parsed = parse_last_json_line(output, label)
    model_arg = pathlib.Path(argv[list(argv).index("-model") + 1])
    validate_benchmark(
        case, parsed, output, iterations, gpu_index, label, expected_model=model_arg
    )
    if not skip_gpu_idle_check:
        assert_gpu_idle(nvidia_smi, gpu_index, f"after {label}")
    throughput = float(parsed["combinedNNEvalsPerSec"])
    print(f"[{label}] {throughput:.3f} nnEval/s ({elapsed:.1f}s)", flush=True)
    return {
        "label": label,
        "caseId": case.case_id,
        "caseLabel": case.label,
        "argv": list(argv),
        "log": str(log_path),
        "logSha256": sha256_file(log_path),
        "wallSeconds": elapsed,
        "json": parsed,
    }


def atomic_write_json(path: pathlib.Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def require_file(path: pathlib.Path, label: str) -> pathlib.Path:
    path = path.resolve()
    if not path.is_file():
        raise MatrixError(f"missing {label}: {path}")
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--model-b36c384", required=True, type=pathlib.Path)
    parser.add_argument("--model-b24c256", required=True, type=pathlib.Path)
    parser.add_argument("--model-b24c384", required=True, type=pathlib.Path)
    parser.add_argument("--result-dir", required=True, type=pathlib.Path)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--preflight-warmup", type=int, default=3)
    parser.add_argument("--preflight-iterations", type=int, default=5)
    parser.add_argument("--formal-warmup", type=int, default=30)
    parser.add_argument("--formal-iterations", type=int, default=150)
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="number of complete six-round position-balanced cycles",
    )
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--max-cv-percent", type=float, default=0.5)
    parser.add_argument("--max-endpoint-drift-percent", type=float, default=1.0)
    parser.add_argument("--strict-stability", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="run the six short path/marker checks, write their report, and stop",
    )
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--skip-gpu-idle-check", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_schedule()
    if args.cycles <= 0:
        raise MatrixError("cycles must be positive")
    for name in (
        "preflight_warmup",
        "preflight_iterations",
        "formal_warmup",
        "formal_iterations",
    ):
        value = getattr(args, name)
        if value < 0 or ("iterations" in name and value == 0):
            raise MatrixError(f"invalid --{name.replace('_', '-')}: {value}")

    binary = require_file(args.binary, "binary")
    config = require_file(args.config, "config")
    models = {
        "b36c384": require_file(args.model_b36c384, "b36c384 model"),
        "b24c256": require_file(args.model_b24c256, "b24c256 model"),
        "b24c384": require_file(args.model_b24c384, "b24c384 model"),
    }
    result_dir = args.result_dir.resolve()
    report_path = result_dir / "benchmark_matrix_report.json"
    if report_path.exists() and not args.overwrite:
        raise MatrixError(f"report already exists (use --overwrite): {report_path}")
    result_dir.mkdir(parents=True, exist_ok=True)
    env, removed_env = clean_environment()
    report: dict[str, Any] = {
        "schema": "renju15-sm120-benchmarknn-matrix-v1",
        "binary": {"path": str(binary), "sha256": sha256_file(binary)},
        "config": {"path": str(config), "sha256": sha256_file(config)},
        "models": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in models.items()
        },
        "gpuIndex": args.gpu_index,
        "s2": {
            "numNNServerThreadsPerModel": 2,
            "gpuToUseThread0": args.gpu_index,
            "gpuToUseThread1": args.gpu_index,
            "meaning": "two evaluator handles/streams on one physical GPU; each lane uses one bsXX batch",
        },
        "removedTacticEnvironmentKeys": removed_env,
        "preflightLegs": [],
        "formalLegs": [],
        "balancedRounds": [list(round_cases) for round_cases in BALANCED_ROUNDS],
        "parameters": {
            "preflightWarmup": args.preflight_warmup,
            "preflightIterations": args.preflight_iterations,
            "formalWarmup": args.formal_warmup,
            "formalIterations": args.formal_iterations,
            "cycles": args.cycles,
            "preflightOnly": args.preflight_only,
        },
    }
    atomic_write_json(report_path, report)
    try:
        if not args.skip_preflight:
            for case_id in CASE_IDS:
                case = CASES[case_id]
                label = f"preflight_{case_id}_{case.label}"
                leg = run_leg(
                    label=label,
                    argv=build_argv(
                        binary,
                        config,
                        models[case.model_key],
                        case,
                        args.preflight_warmup,
                        args.preflight_iterations,
                        args.gpu_index,
                    ),
                    cwd=binary.parent,
                    log_path=result_dir / f"{label}.log",
                    case=case,
                    iterations=args.preflight_iterations,
                    gpu_index=args.gpu_index,
                    timeout_seconds=args.timeout_seconds,
                    nvidia_smi=args.nvidia_smi,
                    skip_gpu_idle_check=args.skip_gpu_idle_check,
                    env=env,
                )
                report["preflightLegs"].append(leg)
                atomic_write_json(report_path, report)

        if args.preflight_only:
            if args.skip_preflight:
                raise MatrixError("--preflight-only cannot be combined with --skip-preflight")
            report["status"] = "preflight-complete"
            atomic_write_json(report_path, report)
            report_sha = sha256_file(report_path)
            (result_dir / "benchmark_matrix_report.sha256").write_text(
                f"{report_sha}  {report_path.name}\n", encoding="utf-8"
            )
            print(json.dumps({"report": str(report_path), "sha256": report_sha}, indent=2))
            return 0

        round_number = 0
        for cycle_index in range(args.cycles):
            for schedule_index, round_cases in enumerate(BALANCED_ROUNDS):
                round_number += 1
                round_key = f"c{cycle_index + 1:02d}_r{schedule_index + 1:02d}"
                for position, case_id in enumerate(round_cases, start=1):
                    case = CASES[case_id]
                    label = f"formal_{round_key}_p{position:02d}_{case_id}_{case.label}"
                    leg = run_leg(
                        label=label,
                        argv=build_argv(
                            binary,
                            config,
                            models[case.model_key],
                            case,
                            args.formal_warmup,
                            args.formal_iterations,
                            args.gpu_index,
                        ),
                        cwd=binary.parent,
                        log_path=result_dir / f"{label}.log",
                        case=case,
                        iterations=args.formal_iterations,
                        gpu_index=args.gpu_index,
                        timeout_seconds=args.timeout_seconds,
                        nvidia_smi=args.nvidia_smi,
                        skip_gpu_idle_check=args.skip_gpu_idle_check,
                        env=env,
                    )
                    leg.update(
                        {
                            "cycle": cycle_index + 1,
                            "round": round_number,
                            "roundKey": round_key,
                            "position": position,
                        }
                    )
                    report["formalLegs"].append(leg)
                    atomic_write_json(report_path, report)
    except BaseException as exc:
        report["status"] = "failed"
        report["failure"] = f"{type(exc).__name__}: {exc}"
        atomic_write_json(report_path, report)
        raise

    summarize_report(report, args.max_cv_percent, args.max_endpoint_drift_percent)
    report["status"] = "complete"
    atomic_write_json(report_path, report)
    report_sha = sha256_file(report_path)
    (result_dir / "benchmark_matrix_report.sha256").write_text(
        f"{report_sha}  {report_path.name}\n", encoding="utf-8"
    )
    print(json.dumps({"report": str(report_path), "sha256": report_sha}, indent=2))
    if args.strict_stability and not report["stabilityPass"]:
        return 3
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MatrixError as exc:
        print(f"MATRIX FAILED: {exc}", file=sys.stderr)
        raise SystemExit(2)

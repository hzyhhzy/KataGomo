#!/usr/bin/env python3
"""Pure-CPU contract and registry rendering for C384 exact SM120 AOT.

This module imports no CUDA package. It is the fail-closed boundary between
the reviewed search coordinates, generated CuTe artifacts, and CMake. A
generated object is never made selectable merely because a file exists: its
metadata, hashes, exact shape, native ABI, and unique exported symbols must all
match the checked-in search space.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable


HERE = Path(__file__).resolve().parent
DEFAULT_SPACE = HERE / "search_space.json"
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
CANDIDATE_ID = re.compile(r"[a-z0-9][a-z0-9-]*\Z")
FAMILIES = ("qkv_rope", "dual_ffn")
PACKAGE_MODES = ("SEARCH_PAIR", "PRODUCTION")
PINNED_CUTLASS_COMMIT = "dcf215af68a2d08d305076c152a06f201728cd53"
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _canonical_version(value: object) -> str:
    # Keep this identical to generator_common.py. The imported module is the
    # authority; visible dist-info is only accepted when it describes that
    # exact imported version.
    return str(value).strip().lower().replace("_", "-")


def _verify_imported_module_provenance(
    provenance: dict, key: str, expected_distribution: str,
) -> None:
    record = provenance.get(key)
    require(isinstance(record, dict),
            f"artifact {key} provenance must be a record")
    require(record.get("distribution") == expected_distribution,
            f"artifact {key} distribution mismatch")
    for field in ("module", "version", "version_module", "module_file"):
        value = record.get(field)
        require(isinstance(value, str) and value.strip() != "",
                f"artifact {key} {field} is missing")
    require(
        isinstance(record.get("module_file_sha256"), str) and
        HEX_SHA256.fullmatch(record["module_file_sha256"]) is not None,
        f"artifact {key} module file hash is invalid",
    )
    status = record.get("metadata_status")
    require(status in ("absent", "present-matching"),
            f"artifact {key} metadata status is invalid")
    distribution_version = record.get("distribution_version")
    if status == "absent":
        require(distribution_version is None,
                f"artifact {key} absent dist-info must have null version")
    else:
        require(isinstance(distribution_version, str) and
                distribution_version.strip() != "",
                f"artifact {key} dist-info version is missing")
        require(_canonical_version(distribution_version) ==
                _canonical_version(record["version"]),
                f"artifact {key} dist-info version mismatch")


@dataclass(frozen=True)
class Task:
    family: str
    batch: int
    token_rows: int
    priority: int
    candidate_id: str
    max_active_clusters: int
    tile: tuple[int, int, int]
    atom_layout: tuple[int, int, int]
    effective_output_tile: tuple[int, int] | None
    ab_stages: int | None
    epilogue_stages: int | None
    stage_policy: str | None
    symbol_token: str
    artifact_stem: str
    prepare_symbol: str
    launch_symbol: str
    native_abi: int


def _symbol_token(family: str, batch: int, candidate_id: str) -> str:
    coordinate = f"{family}\0{batch}\0{candidate_id}".encode("ascii")
    return f"{family}_b{batch}_{hashlib.sha256(coordinate).hexdigest()[:16]}"


def load_space(path: Path = DEFAULT_SPACE) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    validate_space(value)
    return value


def validate_space(space: dict) -> None:
    require(space.get("schema") == 2, "unsupported search-space schema")
    require(
        space.get("kind") == "katago-c384-sm120-exact-fixed-aot-search",
        "unexpected search-space kind",
    )
    target = space.get("target", {})
    require(target.get("model_depth") == 36, "model depth must be 36")
    require(target.get("board") == [15, 15], "board must be 15x15")
    require(target.get("sequence") == 225, "sequence must be 225")
    require(
        tuple(target.get(name) for name in (
            "channels", "heads", "kv_heads", "head_dim", "ffn_channels",
        )) == (384, 12, 12, 32, 1024),
        "target transformer dimensions drifted",
    )
    require(
        (target.get("numeric"), target.get("layout"), target.get("mask"),
         target.get("compute_capability")) == ("fp16", "nhwc", "none", 120),
        "runtime target contract drifted",
    )
    require(
        space.get("native_abi") == {
            "registry": 2, "qkv_rope": 1, "dual_ffn": 1,
            "packed_fa4_proof": 1,
        },
        "native ABI contract drifted",
    )
    batches = space.get("batches", [])
    require([item.get("batch") for item in batches] == [28, 24],
            "generator batch priority must be B28 then B24")
    seen_ids: set[str] = set()
    for priority, item in enumerate(batches, start=1):
        batch = item.get("batch")
        require(item.get("priority") == priority, "batch priority drifted")
        require(item.get("token_rows") == batch * 225,
                f"B{batch} token-row mismatch")
        for family in FAMILIES:
            candidates = item.get(family, [])
            require(candidates, f"B{batch} has no {family} candidates")
            for candidate in candidates:
                candidate_id = candidate.get("id", "")
                require(CANDIDATE_ID.fullmatch(candidate_id) is not None,
                        f"unsafe candidate id: {candidate_id!r}")
                require(candidate_id not in seen_ids,
                        f"duplicate candidate id: {candidate_id}")
                seen_ids.add(candidate_id)
                require(f"b{batch}-abi1" in candidate_id,
                        f"candidate id does not bind B{batch}/ABI1")
                grid = candidate.get("max_active_clusters")
                require(grid in (170, 340), "unsupported scheduler grid")
                if family == "qkv_rope":
                    require(candidate.get("tile") in
                            ([128, 128, 32], [128, 128, 64]),
                            "QKV tile must vary only K32/K64")
                    require(candidate.get("atom_layout") in
                            ([2, 2, 1], [4, 2, 1]),
                            "QKV atom layout must be 2x2 or 4x2")
                    require(candidate.get("stage_policy") ==
                            "cutlass-pinned-auto", "QKV stage policy drifted")
                    require(grid == 170,
                            "QKV persistent scheduler is bounded to one SM wave")
                else:
                    require(candidate.get("tile") in
                            ([128, 128, 32], [128, 128, 64]),
                            "dual-FFN tile must vary only K32/K64")
                    require(candidate.get("effective_output_tile") == [128, 64],
                            "dual-FFN output tile drifted")
                    require(candidate.get("atom_layout") == [4, 2, 1],
                            "dual-FFN atom layout drifted")
                    require((candidate.get("ab_stages"),
                             candidate.get("epilogue_stages")) == (2, 4),
                            "dual-FFN stage policy drifted")
    for item in batches:
        qkv_coordinates = {
            (tuple(c["tile"]), tuple(c["atom_layout"]))
            for c in item["qkv_rope"]
        }
        require(qkv_coordinates == {
            ((128, 128, 32), (2, 2, 1)),
            ((128, 128, 32), (4, 2, 1)),
            ((128, 128, 64), (2, 2, 1)),
            ((128, 128, 64), (4, 2, 1)),
        }, "QKV bounded factorial coordinates drifted")
        dual_coordinates = {
            (tuple(c["tile"]), c["max_active_clusters"])
            for c in item["dual_ffn"]
        }
        require(dual_coordinates == {
            ((128, 128, 32), 170), ((128, 128, 32), 340),
            ((128, 128, 64), 170), ((128, 128, 64), 340),
        }, "dual-FFN bounded coordinates drifted")
    layouts = space.get("layouts", {})
    require(layouts.get("qkv_weights") ==
            "row-major-Kx3C-[384,1152]", "QKV weight layout drifted")
    require(layouts.get("rope_table") ==
            "row-major-[225,192]-half2-cos-sin", "RoPE layout drifted")
    require(layouts.get("dual_weights") ==
            "row-major-Kx2F-[384,2048]-paired-linear64-gate64",
            "dual weight layout drifted")


def materialize_tasks(space: dict) -> list[Task]:
    validate_space(space)
    native = space["native_abi"]
    tasks: list[Task] = []
    for item in space["batches"]:
        batch = int(item["batch"])
        for family in FAMILIES:
            for candidate in item[family]:
                candidate_id = candidate["id"]
                token = _symbol_token(family, batch, candidate_id)
                stem = "katago_c384_exact_" + token
                tasks.append(Task(
                    family=family,
                    batch=batch,
                    token_rows=int(item["token_rows"]),
                    priority=int(item["priority"]),
                    candidate_id=candidate_id,
                    max_active_clusters=int(candidate["max_active_clusters"]),
                    tile=tuple(int(x) for x in candidate["tile"]),
                    atom_layout=tuple(int(x) for x in
                                      candidate.get("atom_layout", [4, 2, 1])),
                    effective_output_tile=(
                        tuple(int(x) for x in candidate["effective_output_tile"])
                        if "effective_output_tile" in candidate else None
                    ),
                    ab_stages=(int(candidate["ab_stages"])
                               if "ab_stages" in candidate else None),
                    epilogue_stages=(int(candidate["epilogue_stages"])
                                     if "epilogue_stages" in candidate else None),
                    stage_policy=candidate.get("stage_policy"),
                    symbol_token=token,
                    artifact_stem=stem,
                    prepare_symbol=stem + "_prepare_v1",
                    launch_symbol=stem + "_launch_v1",
                    native_abi=int(native[family]),
                ))
    require(len(tasks) == 16,
            f"expected sixteen bounded generated tasks, got {len(tasks)}")
    require(len({task.symbol_token for task in tasks}) == len(tasks),
            "symbol-token collision")
    return tasks


def find_task(space: dict, family: str, batch: int, candidate_id: str) -> Task:
    matches = [task for task in materialize_tasks(space)
               if (task.family, task.batch, task.candidate_id) ==
                  (family, batch, candidate_id)]
    require(len(matches) == 1, "candidate is absent or ambiguous")
    return matches[0]


def _resolve_artifact_file(metadata_path: Path, entry: dict, label: str) -> Path:
    require(isinstance(entry, dict), f"missing {label} file record")
    relative = entry.get("path", "")
    require(isinstance(relative, str) and relative and
            Path(relative).name == relative,
            f"{label} path must be a local basename")
    path = metadata_path.parent / relative
    require(path.is_file(), f"missing generated {label}: {path}")
    actual = sha256_file(path)
    require(entry.get("sha256") == actual,
            f"generated {label} hash mismatch: {path}")
    return path.resolve()


@dataclass(frozen=True)
class VerifiedArtifact:
    task: Task
    metadata_path: Path
    header: Path
    object_file: Path
    bridge: Path


def verify_artifact(space: dict, metadata_path: Path) -> VerifiedArtifact:
    metadata_path = metadata_path.resolve()
    require(metadata_path.is_file(), f"missing metadata: {metadata_path}")
    value = json.loads(metadata_path.read_text(encoding="utf-8"))
    require(value.get("schema") == 2, "unsupported artifact metadata schema")
    require(value.get("kind") == "katago-c384-exact-aot-artifact",
            "unexpected artifact kind")
    require(value.get("generation_complete") is True,
            "artifact generation is not complete")
    require(value.get("verified_on_target") is False,
            "search generator must not claim target verification")
    require(value.get("compute_capability") == "sm_120",
            "artifact must be generated for sm_120")
    require(value.get("dtype") == "fp16", "artifact dtype must be fp16")
    require(value.get("search_space_sha256") == canonical_json_sha256(space),
            "artifact search-space hash mismatch")
    task = find_task(
        space, value.get("family"), int(value.get("batch", -1)),
        value.get("candidate_id", ""),
    )
    require(value.get("token_rows") == task.token_rows,
            "artifact token rows mismatch")
    require(value.get("native_abi") == task.native_abi,
            "artifact native ABI mismatch")
    require(value.get("artifact_stem") == task.artifact_stem,
            "artifact stem mismatch")
    require(value.get("max_active_clusters") == task.max_active_clusters,
            "artifact scheduler grid mismatch")
    require(value.get("shape") == {
        "sequence": 225, "channels": 384, "heads": 12,
        "kv_heads": 12, "head_dim": 32, "ffn_channels": 1024,
    }, "artifact fixed shape mismatch")
    provenance = value.get("provenance", {})
    generator = HERE / (
        "generate_qkv_rope.py" if task.family == "qkv_rope"
        else "generate_dual_ffn.py"
    )
    require(provenance.get("generator_sha256") == sha256_file(generator),
            "artifact generator provenance mismatch")
    require(provenance.get("cutlass_commit") == PINNED_CUTLASS_COMMIT,
            "artifact CUTLASS provenance mismatch")
    _verify_imported_module_provenance(
        provenance, "nvidia_cutlass_dsl", "nvidia-cutlass-dsl",
    )
    _verify_imported_module_provenance(
        provenance, "cuda_bindings", "cuda-bindings",
    )
    for label in ("dense_gemm_sha256", "patched_dense_gemm_sha256"):
        require(HEX_SHA256.fullmatch(str(provenance.get(label, ""))) is not None,
                f"artifact {label} is invalid")
    require(provenance.get("gpu_kernel_executed") is False,
            "generation provenance must remain CPU-only")
    require(re.search(r"release 13(?:\.|,)", str(provenance.get("nvcc", "")),
                      re.IGNORECASE) is not None,
            "artifact must be generated with CUDA 13.x")
    expected_coordinate = {
        "tile": list(task.tile),
        "atom_layout": list(task.atom_layout),
        "input": [task.token_rows, 384],
        "output": [task.token_rows, 1024],
    }
    coordinate = value.get("coordinate", {})
    for label, expected in expected_coordinate.items():
        if label == "output" and task.family == "qkv_rope":
            continue
        require(coordinate.get(label) == expected,
                f"artifact coordinate {label} mismatch")
    if task.family == "qkv_rope":
        require(coordinate.get("packed_weights") == [384, 1152] and
                coordinate.get("packed_output") == [task.token_rows, 1152] and
                coordinate.get("rope_table_half2") == [225, 192],
                "QKV artifact coordinate/layout mismatch")
    else:
        require(coordinate.get("paired_weights") == [384, 2048] and
                coordinate.get("effective_output_tile") ==
                  list(task.effective_output_tile or ()) and
                coordinate.get("epilogue") == "silu-linear-times-gate",
                "dual-FFN artifact coordinate/layout mismatch")
    require(value.get("symbols") == {
        "prepare": task.prepare_symbol, "launch": task.launch_symbol,
    }, "artifact exported symbols mismatch")
    files = value.get("files", {})
    verified = VerifiedArtifact(
        task=task,
        metadata_path=metadata_path,
        header=_resolve_artifact_file(metadata_path, files.get("header"), "header"),
        object_file=_resolve_artifact_file(metadata_path, files.get("object"), "object"),
        bridge=_resolve_artifact_file(metadata_path, files.get("bridge"), "bridge"),
    )
    object_bytes = verified.object_file.read_bytes()
    require(object_bytes.startswith(b"\x7fELF"),
            "generated object is not an ELF object")
    wrapper = f"cute_dsl_{task.artifact_stem}_wrapper".encode("ascii")
    require(wrapper in object_bytes,
            "generated object lacks the expected unique wrapper symbol")
    bridge_text = verified.bridge.read_text(encoding="utf-8")
    require(task.prepare_symbol in bridge_text and task.launch_symbol in bridge_text and
            wrapper.decode("ascii") in bridge_text,
            "generated bridge lacks the selected ABI symbols")
    require("MaxPreparedDevices" in bridge_text and
            "cudaPeekAtLastError" in bridge_text,
            "generated bridge lacks per-device eager preparation")
    return verified


def verify_complete_artifact_set(
    space: dict, metadata_paths: Iterable[Path],
) -> list[VerifiedArtifact]:
    artifacts = [verify_artifact(space, path) for path in metadata_paths]
    wanted = materialize_tasks(space)
    keys = [(a.task.family, a.task.batch, a.task.candidate_id) for a in artifacts]
    require(len(keys) == len(set(keys)), "duplicate artifact metadata")
    wanted_keys = {(t.family, t.batch, t.candidate_id) for t in wanted}
    require(set(keys) == wanted_keys,
            "artifact set must contain every B28/B24 QKV and dual candidate")
    order = {(t.family, t.batch, t.candidate_id): i for i, t in enumerate(wanted)}
    return sorted(artifacts, key=lambda a: order[
        (a.task.family, a.task.batch, a.task.candidate_id)])


def normalize_package_mode(mode: str) -> str:
    normalized = mode.upper().replace("-", "_")
    require(normalized in PACKAGE_MODES,
            "exact-AOT package mode must be SEARCH_PAIR or PRODUCTION")
    return normalized


def verify_production_promotion(
    path: Path, qkv: VerifiedArtifact, dual: VerifiedArtifact,
) -> dict:
    path = path.resolve()
    require(path.is_file(), f"missing production promotion evidence: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(value.get("schema") == 1,
            "unsupported production promotion schema")
    require(value.get("kind") ==
            "katago-c384-exact-aot-production-promotion",
            "unexpected production promotion kind")
    require(value.get("status") == "PASSED",
            "production promotion did not pass")
    require(value.get("selected") == {
        "batch": qkv.task.batch,
        "qkv_rope": qkv.task.candidate_id,
        "dual_ffn": dual.task.candidate_id,
    }, "production promotion selected pair mismatch")
    gates = value.get("gates", {})
    require(set(gates) == {"activation", "long_gate", "accuracy"},
            "production promotion must contain all required gates")
    for name in ("activation", "long_gate", "accuracy"):
        gate = gates.get(name, {})
        require(gate.get("status") == "PASSED",
                f"production promotion {name} gate did not pass")
        require(HEX_SHA256.fullmatch(
            str(gate.get("evidence_sha256", ""))) is not None,
            f"production promotion {name} evidence hash is invalid")
    return value


def render_registry(artifacts: list[VerifiedArtifact]) -> str:
    qkv = [a for a in artifacts if a.task.family == "qkv_rope"]
    dual = [a for a in artifacts if a.task.family == "dual_ffn"]
    lines = [
        '#include "neuralnet/c384_exact_fixed_aot_kernels.h"', "",
    ]
    for artifact in artifacts:
        task = artifact.task
        require(IDENTIFIER.fullmatch(task.prepare_symbol) is not None,
                "unsafe prepare symbol")
        require(IDENTIFIER.fullmatch(task.launch_symbol) is not None,
                "unsafe launch symbol")
        lines.append(f'extern "C" cudaError_t {task.prepare_symbol}(int);')
        if task.family == "qkv_rope":
            lines.extend([
                f'extern "C" cudaError_t {task.launch_symbol}(',
                "  const half*, const half*, const half2*, half*, int, int, cudaStream_t);",
            ])
        else:
            lines.extend([
                f'extern "C" cudaError_t {task.launch_symbol}(',
                "  const half*, const half*, const half*, half*, int, int, cudaStream_t);",
            ])
    lines.extend(["", "namespace C384ExactFixedAot {", "", "namespace {"])
    lines.append("constexpr QkvRopeTactic QkvTactics[] = {")
    for artifact in qkv:
        t = artifact.task
        lines.append(
            "  {{Family::QkvRope,%d,%d,0,\"%s\",true,false,"
            "kRegistryAbiVersion},kQkvRopeNativeAbiVersion,%s,%s}," % (
                t.batch, t.token_rows, t.candidate_id,
                t.prepare_symbol, t.launch_symbol,
            )
        )
    lines.extend(["};", "", "constexpr DualFfnTactic DualTactics[] = {"])
    for artifact in dual:
        t = artifact.task
        lines.append(
            "  {{Family::DualFfn,%d,%d,%d,\"%s\",false,true,"
            "kRegistryAbiVersion},kDualFfnNativeAbiVersion,%s,%s}," % (
                t.batch, t.token_rows, t.max_active_clusters, t.candidate_id,
                t.prepare_symbol, t.launch_symbol,
            )
        )
    lines.extend([
        "};", "", "}  // namespace", "",
        "const QkvRopeTactic* generatedQkvRopeTactics(std::size_t& count) {",
        "  count = sizeof(QkvTactics) / sizeof(QkvTactics[0]);",
        "  return QkvTactics;", "}", "",
        "const DualFfnTactic* generatedDualFfnTactics(std::size_t& count) {",
        "  count = sizeof(DualTactics) / sizeof(DualTactics[0]);",
        "  return DualTactics;", "}", "",
        "}  // namespace C384ExactFixedAot", "",
    ])
    return "\n".join(lines)


def _cmake_path(path: Path) -> str:
    value = path.resolve().as_posix()
    require('"' not in value and ";" not in value, "unsafe CMake path")
    return value


def render_cmake_manifest(
    registry_path: Path,
    artifacts: list[VerifiedArtifact],
    *,
    mode: str = "SEARCH_PAIR",
    promotion_evidence: Path | None = None,
) -> str:
    mode = normalize_package_mode(mode)
    qkv = [a for a in artifacts if a.task.family == "qkv_rope"]
    dual = [a for a in artifacts if a.task.family == "dual_ffn"]
    require(len(artifacts) == 2 and len(qkv) == 1 and len(dual) == 1,
            "one build manifest must select exactly one QKV and one dual artifact")
    require(qkv[0].task.batch == dual[0].task.batch,
            "selected QKV and dual artifacts must bind the same fixed batch")
    if mode == "SEARCH_PAIR":
        require(promotion_evidence is None,
                "SEARCH_PAIR must not claim production promotion evidence")
        promotion_path = None
    else:
        require(promotion_evidence is not None,
                "PRODUCTION requires promotion evidence")
        promotion_path = promotion_evidence.resolve()
        verify_production_promotion(promotion_path, qkv[0], dual[0])
    dirs = {a.metadata_path.parent.resolve() for a in artifacts}
    require(len(dirs) == 1, "all generated artifacts must share one directory")
    include_dir = next(iter(dirs))
    registry_path = registry_path.resolve()
    require(registry_path.is_file(), "generated registry does not exist")
    headers = [a.header for a in artifacts]
    bridges = [a.bridge for a in artifacts]
    objects = [a.object_file for a in artifacts]
    metadata = [a.metadata_path for a in artifacts]
    hashed_files = [registry_path]
    for artifact in artifacts:
        hashed_files.extend((
            artifact.header, artifact.object_file, artifact.bridge,
            artifact.metadata_path,
        ))
    if promotion_path is not None:
        hashed_files.append(promotion_path)
    require(len(hashed_files) == len(set(hashed_files)),
            "generated manifest contains duplicate files")
    lines = [
        "# Generated by emit_registry.py; all paths were hash-verified.",
        'set(KATAGO_C384_EXACT_AOT_PACKAGE_SCHEMA "1")',
        f'set(KATAGO_C384_EXACT_AOT_PACKAGE_MODE "{mode}")',
        f'set(KATAGO_C384_EXACT_AOT_SELECTED_BATCH "{qkv[0].task.batch}")',
        f'set(KATAGO_C384_EXACT_AOT_SELECTED_QKV_ID "{qkv[0].task.candidate_id}")',
        f'set(KATAGO_C384_EXACT_AOT_SELECTED_DUAL_FFN_ID "{dual[0].task.candidate_id}")',
        'set(KATAGO_C384_EXACT_AOT_SELECTED_FAMILIES "QKV_ROPE;DUAL_FFN")',
        f'set(KATAGO_C384_EXACT_AOT_QKV_ROPE_IDS "{qkv[0].task.candidate_id}")',
        f'set(KATAGO_C384_EXACT_AOT_DUAL_FFN_IDS "{dual[0].task.candidate_id}")',
        'set(KATAGO_C384_EXACT_AOT_PROMOTION_EVIDENCE "{}")'.format(
            _cmake_path(promotion_path) if promotion_path is not None else ""
        ),
        'set(KATAGO_C384_EXACT_AOT_PROMOTION_EVIDENCE_SHA256 "{}")'.format(
            sha256_file(promotion_path) if promotion_path is not None else ""
        ),
        f'set(KATAGO_C384_EXACT_AOT_REGISTRY_PROVIDER "{_cmake_path(registry_path)}")',
        f'set(KATAGO_C384_EXACT_AOT_GENERATED_INCLUDE_DIR "{_cmake_path(include_dir)}")',
        "set(KATAGO_C384_EXACT_AOT_GENERATED_HEADERS",
    ]
    lines.extend(f'  "{_cmake_path(path)}"' for path in headers)
    lines.extend([
        ")", "set(KATAGO_C384_EXACT_AOT_GENERATED_METADATA",
    ])
    lines.extend(f'  "{_cmake_path(path)}"' for path in metadata)
    lines.extend([
        ")",
        "set(KATAGO_C384_EXACT_AOT_GENERATED_BRIDGES",
    ])
    lines.extend(f'  "{_cmake_path(path)}"' for path in bridges)
    lines.extend([
        ")", "set(KATAGO_C384_EXACT_AOT_GENERATED_OBJECTS",
    ])
    lines.extend(f'  "{_cmake_path(path)}"' for path in objects)
    lines.extend([
        ")", "set(KATAGO_C384_EXACT_AOT_GENERATED_FILE_SHA256",
    ])
    for path in hashed_files:
        lines.append(f'  "{_cmake_path(path)}" "{sha256_file(path)}"')
    lines.extend([")", ""])
    return "\n".join(lines)

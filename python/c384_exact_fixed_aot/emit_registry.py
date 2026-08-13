#!/usr/bin/env python3
"""Verify a complete generated set and emit registry/CMake integration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contract import (
    DEFAULT_SPACE,
    canonical_json_sha256,
    load_space,
    normalize_package_mode,
    render_cmake_manifest,
    render_registry,
    require,
    sha256_file,
    verify_artifact,
    verify_production_promotion,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", type=Path, default=DEFAULT_SPACE)
    parser.add_argument("--metadata", type=Path, action="append", required=True)
    parser.add_argument(
        "--mode", choices=("search-pair", "production"), required=True,
    )
    parser.add_argument("--qkv-id", required=True)
    parser.add_argument("--dual-id", required=True)
    parser.add_argument("--promotion-evidence", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    space = load_space(args.space.resolve())
    artifacts = [verify_artifact(space,path) for path in args.metadata]
    require(len(artifacts) == 2,
            "one binary must materialize exactly one QKV and one dual artifact")
    qkv = [a for a in artifacts if a.task.family == "qkv_rope"]
    dual = [a for a in artifacts if a.task.family == "dual_ffn"]
    require(len(qkv) == 1 and len(dual) == 1,
            "selected artifact pair must contain QKV and dual families")
    require(qkv[0].task.candidate_id == args.qkv_id,
            "--qkv-id does not match QKV metadata")
    require(dual[0].task.candidate_id == args.dual_id,
            "--dual-id does not match dual metadata")
    require(qkv[0].task.batch == dual[0].task.batch,
            "selected QKV and dual artifacts must bind the same fixed batch")
    artifacts = [qkv[0],dual[0]]
    mode = normalize_package_mode(args.mode)
    promotion_path = (
        args.promotion_evidence.resolve()
        if args.promotion_evidence is not None else None
    )
    if mode == "SEARCH_PAIR":
        require(promotion_path is None,
                "search-pair must not claim production promotion evidence")
        promotion = None
    else:
        require(promotion_path is not None,
                "production requires --promotion-evidence")
        promotion = verify_production_promotion(
            promotion_path, qkv[0], dual[0],
        )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = output_dir / "c384_exact_generated_registry.cu"
    manifest_path = output_dir / "c384_exact_generated_manifest.cmake"
    set_path = output_dir / "c384_exact_generated_set.json"
    for path in (registry_path, manifest_path, set_path):
        require(not path.exists(), f"refusing to overwrite generated file: {path}")

    registry_path.write_text(render_registry(artifacts), encoding="utf-8")
    manifest_path.write_text(
        render_cmake_manifest(
            registry_path, artifacts, mode=mode,
            promotion_evidence=promotion_path,
        ),
        encoding="utf-8",
    )
    set_manifest = {
        "schema": 1,
        "kind": "katago-c384-exact-aot-generated-set",
        "mode": mode,
        "verified_on_target": mode == "PRODUCTION",
        "search_space_sha256": canonical_json_sha256(space),
        "artifact_count": len(artifacts),
        "selected_batch": qkv[0].task.batch,
        "selected_qkv_id": qkv[0].task.candidate_id,
        "selected_dual_ffn_id": dual[0].task.candidate_id,
        "artifacts": [
            {
                "family": artifact.task.family,
                "batch": artifact.task.batch,
                "candidate_id": artifact.task.candidate_id,
                "metadata": str(artifact.metadata_path),
                "metadata_sha256": sha256_file(artifact.metadata_path),
            }
            for artifact in artifacts
        ],
        "registry": {
            "path": str(registry_path),
            "sha256": sha256_file(registry_path),
        },
        "cmake_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "production_promotion": ({
            "path": str(promotion_path),
            "sha256": sha256_file(promotion_path),
            "status": promotion["status"],
            "gates": promotion["gates"],
        } if promotion_path is not None and promotion is not None else None),
        "performance_claim": None,
    }
    set_path.write_text(
        json.dumps(set_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(set_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
    render_cmake_manifest,
    render_registry,
    require,
    sha256_file,
    verify_complete_artifact_set,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", type=Path, default=DEFAULT_SPACE)
    parser.add_argument("--metadata", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    space = load_space(args.space.resolve())
    artifacts = verify_complete_artifact_set(space, args.metadata)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = output_dir / "c384_exact_generated_registry.cu"
    manifest_path = output_dir / "c384_exact_generated_manifest.cmake"
    set_path = output_dir / "c384_exact_generated_set.json"
    for path in (registry_path, manifest_path, set_path):
        require(not path.exists(), f"refusing to overwrite generated file: {path}")

    registry_path.write_text(render_registry(artifacts), encoding="utf-8")
    manifest_path.write_text(
        render_cmake_manifest(registry_path, artifacts), encoding="utf-8",
    )
    set_manifest = {
        "schema": 1,
        "kind": "katago-c384-exact-aot-generated-set",
        "verified_on_target": False,
        "search_space_sha256": canonical_json_sha256(space),
        "artifact_count": len(artifacts),
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

#!/usr/bin/env python3
"""Validate FA4 artifacts and emit a SHA-locked registry/CMake package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


EXPECTED_SHAPE = {"sequence": 225, "heads": 12, "head_dim": 32}
VALID_BATCHES = {24, 28}
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quote(value: str) -> str:
    return json.dumps(value)


def cmake_quote(value: str) -> str:
    return '"' + value.replace('\\', '/') .replace('"', '\\"') + '"'


def load_metadata(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    stem = data.get("artifact_stem")
    prefix = data.get("symbol_prefix")
    batch = data.get("batch")
    if data.get("schema") != 1 or batch not in VALID_BATCHES:
        raise RuntimeError(f"{path}: schema/batch mismatch")
    if data.get("fixed_shape") != EXPECTED_SHAPE or data.get("mask") is not False or \
       data.get("compute_capability") != "sm_120" or \
       data.get("gpu_used_for_generation") is not False:
        raise RuntimeError(f"{path}: fixed shape/capability contract failed")
    if not isinstance(stem, str) or not IDENTIFIER.fullmatch(stem) or \
       not isinstance(prefix, str) or not IDENTIFIER.fullmatch(prefix):
        raise RuntimeError(f"{path}: bad C identifier")
    layout = data.get("layout")
    accumulation = data.get("accumulation")
    if layout not in ("planar", "packed-token") or \
       accumulation not in ("fp32", "qk16", "pv16", "both16"):
        raise RuntimeError(f"{path}: layout/accumulation mismatch")
    tile = data.get("tile", {})
    if tile.get("m") not in (64, 128) or tile.get("n") not in (64, 96, 128) or \
       tile.get("num_stages") not in (1, 2) or tile.get("num_warps") not in (4, 8):
        raise RuntimeError(f"{path}: tile contract failed")
    assets = {
        "header": path.parent / f"{stem}.h",
        "object": path.parent / f"{stem}.o",
        "bridge": path.parent / f"{stem}_bridge.cpp",
        "metadata": path,
    }
    for label, asset in assets.items():
        if not asset.is_file():
            raise RuntimeError(f"{path}: missing {label}: {asset}")
    actual = {label: sha256(asset) for label, asset in assets.items()}
    recorded = data.get("sha256", {})
    for label in ("header", "object", "bridge"):
        if recorded.get(label) != actual[label]:
            raise RuntimeError(f"{path}: {label} SHA mismatch")
    data["source_metadata"] = str(path.resolve())
    data["source_assets"] = assets
    data["actual_sha256"] = actual
    return data


def same_search_coordinate(candidates: list[dict]) -> bool:
    keys = (
        "layout", "accumulation",
    )
    first = candidates[0]
    return all(
        all(item[key] == first[key] for key in keys) and item["tile"] == first["tile"]
        for item in candidates[1:]
    )


def registry_source(candidates: list[dict], mode: str) -> str:
    declarations = []
    rows = []
    layout = {"planar": "InputLayout::PlanarQkv", "packed-token": "InputLayout::PackedTokenQkv"}
    accumulation = {
        "fp32": "Accumulation::Fp32", "qk16": "Accumulation::QkFp16",
        "pv16": "Accumulation::PvFp16", "both16": "Accumulation::BothFp16",
    }
    for item in candidates:
        prefix = item["symbol_prefix"]
        declarations.append(f'''extern "C" int {prefix}_batch();
extern "C" int {prefix}_sequence();
extern "C" int {prefix}_heads();
extern "C" int {prefix}_head_dim();
extern "C" const char* {prefix}_id();
extern "C" cudaError_t {prefix}_prepare(int);
extern "C" cudaError_t {prefix}_launch(
  void*, void*, void*, void*, int, int, int, int, float, uint32_t, cudaStream_t);
''')
        tile = item["tile"]
        rows.append(f'''  {{kRegistryAbiVersion,{item["batch"]},225,12,32,
   {tile["m"]},{tile["n"]},{tile["num_stages"]},{tile["num_warps"]},
   {layout[item["layout"]]},{accumulation[item["accumulation"]]},
   {quote(item["candidate_id"])},
   {prefix}_batch,{prefix}_sequence,{prefix}_heads,{prefix}_head_dim,
   {prefix}_id,{prefix}_prepare,{prefix}_launch}},''')
    artifact_mode = "BatchSearch" if mode == "batch-search" else "Production"
    return f'''#include "neuralnet/c384_h12_fa4_sm120.h"

{''.join(declarations)}
namespace C384H12Fa4Sm120 {{
namespace {{
const Candidate candidates[] = {{
{chr(10).join(rows)}
}};
}}

RegistryView generatedRegistry() {{
  return {{ArtifactMode::{artifact_mode},candidates,sizeof(candidates) / sizeof(candidates[0])}};
}}

}}  // namespace C384H12Fa4Sm120
'''


def emit_cmake(
    output: Path, mode: str, candidates: list[dict], registry: Path,
    runtime: Path, package_assets: list[Path]
) -> Path:
    batches = [str(item["batch"]) for item in candidates]
    headers = [str(item["packaged_assets"]["header"]) for item in candidates]
    objects = [str(item["packaged_assets"]["object"]) for item in candidates]
    bridges = [str(item["packaged_assets"]["bridge"]) for item in candidates]
    metadata = [str(item["packaged_assets"]["metadata"]) for item in candidates]
    values = {
        "BATCHES": batches, "HEADERS": headers, "OBJECTS": objects,
        "BRIDGES": bridges, "METADATA": metadata,
        "ASSETS": [str(path) for path in package_assets],
        "ASSET_SHAS": [sha256(path) for path in package_assets],
    }
    lines = [
        'set(KATAGO_C384_H12_FA4_PACKAGE_SCHEMA "1")',
        f'set(KATAGO_C384_H12_FA4_PACKAGE_MODE "{mode.upper().replace("-", "_")}")',
    ]
    for name, items in values.items():
        lines.append(f"set(KATAGO_C384_H12_FA4_{name}")
        lines.extend(f"  {cmake_quote(item)}" for item in items)
        lines.append(")")
    lines.extend((
        f"set(KATAGO_C384_H12_FA4_REGISTRY {cmake_quote(str(registry))})",
        f"set(KATAGO_C384_H12_FA4_REGISTRY_SHA {cmake_quote(sha256(registry))})",
        f"set(KATAGO_C384_H12_FA4_RUNTIME {cmake_quote(str(runtime))})",
        f"set(KATAGO_C384_H12_FA4_RUNTIME_SHA {cmake_quote(sha256(runtime))})",
    ))
    path = output / "c384_h12_fa4_package.cmake"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("batch-search", "production"), required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    expected_count = 2 if args.mode == "batch-search" else 1
    if len(args.candidate) != expected_count:
        raise RuntimeError(f"{args.mode} requires exactly {expected_count} candidate(s)")
    candidates = sorted((load_metadata(path.resolve()) for path in args.candidate),
                        key=lambda item: item["batch"])
    if args.mode == "batch-search":
        if [item["batch"] for item in candidates] != [24, 28]:
            raise RuntimeError("final batch search is exactly B24/B28")
        if any(item["abi_mode"] != "search" for item in candidates):
            raise RuntimeError("batch search requires native unique search ABI")
        if not same_search_coordinate(candidates):
            raise RuntimeError("B24/B28 batch comparison must use one identical tile/accumulation/layout")
    elif candidates[0]["abi_mode"] != "production" or \
         candidates[0]["symbol_prefix"] != "c384fa4win":
        raise RuntimeError("production must be regenerated with stable c384fa4win ABI")
    runtime = args.runtime.resolve()
    if not runtime.is_file():
        raise RuntimeError(f"missing CuTe runtime: {runtime}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    asset_dir = output / "artifacts"
    asset_dir.mkdir()
    package_assets: list[Path] = []
    for item in candidates:
        packaged = {}
        for label, source in item["source_assets"].items():
            destination = asset_dir / source.name
            shutil.copy2(source, destination)
            if sha256(destination) != item["actual_sha256"][label]:
                raise RuntimeError(f"copy verification failed: {destination}")
            packaged[label] = destination.resolve()
            package_assets.append(destination.resolve())
        item["packaged_assets"] = packaged
    runtime_copy = asset_dir / runtime.name
    shutil.copy2(runtime, runtime_copy)
    if sha256(runtime_copy) != sha256(runtime):
        raise RuntimeError("CuTe runtime copy verification failed")
    registry = output / "c384_h12_fa4_registry.cpp"
    registry.write_text(registry_source(candidates,args.mode), encoding="utf-8")
    cmake = emit_cmake(output,args.mode,candidates,registry.resolve(),
                       runtime_copy.resolve(),package_assets)
    report = {
        "schema": 1, "status": "PACKAGE_PASS", "mode": args.mode,
        "batches": [item["batch"] for item in candidates],
        "registry": {"path": str(registry.resolve()), "sha256": sha256(registry)},
        "cmake": {"path": str(cmake.resolve()), "sha256": sha256(cmake)},
        "runtime": {"path": str(runtime_copy.resolve()), "sha256": sha256(runtime_copy)},
        "candidates": [{
            key: item[key] for key in (
                "batch", "candidate_id", "artifact_stem", "symbol_prefix",
                "abi_mode", "tile", "layout", "accumulation",
            )
        } for item in candidates],
    }
    manifest = output / "package_manifest.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

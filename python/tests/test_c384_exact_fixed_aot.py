#!/usr/bin/env python3
"""CPU-only contracts for the C384 exact fixed-batch AOT generators."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


TOOLS = Path(__file__).resolve().parents[1] / "c384_exact_fixed_aot"
sys.path.insert(0, str(TOOLS))

from bridge_codegen import render_dual_ffn_bridge, render_qkv_rope_bridge
from contract import (
    canonical_json_sha256,
    load_space,
    materialize_tasks,
    render_cmake_manifest,
    render_registry,
    sha256_file,
    verify_artifact,
    verify_complete_artifact_set,
)


class C384ExactFixedAotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.space_path = TOOLS / "search_space.json"
        self.space = load_space(self.space_path)
        self.tasks = materialize_tasks(self.space)

    def test_bounded_coordinate_space(self) -> None:
        self.assertEqual(len(self.tasks), 16)
        self.assertEqual({task.batch for task in self.tasks}, {24, 28})
        self.assertEqual(
            {task.token_rows for task in self.tasks}, {5400, 6300},
        )
        self.assertEqual(len({task.candidate_id for task in self.tasks}), 16)
        self.assertEqual(len({task.symbol_token for task in self.tasks}), 16)
        for batch in (24, 28):
            qkv = [t for t in self.tasks
                   if t.batch == batch and t.family == "qkv_rope"]
            dual = [t for t in self.tasks
                    if t.batch == batch and t.family == "dual_ffn"]
            self.assertEqual(
                {(t.tile[2], t.atom_layout[0]) for t in qkv},
                {(32, 2), (32, 4), (64, 2), (64, 4)},
            )
            self.assertTrue(all(t.max_active_clusters == 170 for t in qkv))
            self.assertEqual(
                {(t.tile[2], t.max_active_clusters) for t in dual},
                {(32, 170), (32, 340), (64, 170), (64, 340)},
            )

    def test_bridge_native_abi_and_no_hot_initialization(self) -> None:
        for task in self.tasks:
            source = (render_qkv_rope_bridge(task)
                      if task.family == "qkv_rope"
                      else render_dual_ffn_bridge(task))
            self.assertIn(task.prepare_symbol, source)
            self.assertIn(task.launch_symbol, source)
            self.assertIn(f"tokenRows != {task.token_rows}", source)
            self.assertIn("int deviceOrdinal", source)
            launch = source.split(
                f'extern "C" cudaError_t {task.launch_symbol}', 1,
            )[1]
            self.assertNotIn("Kernel_Module_Load", launch)
            self.assertNotIn("call_once", launch)
            self.assertNotIn("cudaMalloc", launch)
            self.assertNotIn("cudaMemcpy", launch)
            self.assertNotIn("cudaGetDevice(", launch)
            if task.family == "qkv_rope":
                self.assertIn("const half2* cosSin", launch)
                self.assertIn("Tensor_table_arg_t", launch)
            else:
                self.assertIn("(void)unusedGateWeights", launch)

    def _write_artifact(self, directory: Path, task) -> Path:
        header = directory / f"{task.artifact_stem}.h"
        object_file = directory / f"{task.artifact_stem}.o"
        bridge = directory / f"{task.artifact_stem}_bridge.cu"
        header.write_text("// synthetic header\n", encoding="utf-8")
        object_file.write_bytes(b"synthetic-sm120-object\0" +
                                task.symbol_token.encode("ascii"))
        bridge.write_text("// synthetic audited bridge\n", encoding="utf-8")
        metadata = {
            "schema": 2,
            "kind": "katago-c384-exact-aot-artifact",
            "generation_complete": True,
            "verified_on_target": False,
            "search_space_sha256": canonical_json_sha256(self.space),
            "family": task.family,
            "candidate_id": task.candidate_id,
            "batch": task.batch,
            "token_rows": task.token_rows,
            "native_abi": task.native_abi,
            "artifact_stem": task.artifact_stem,
            "max_active_clusters": task.max_active_clusters,
            "symbols": {
                "prepare": task.prepare_symbol, "launch": task.launch_symbol,
            },
            "files": {
                "header": {"path": header.name, "sha256": sha256_file(header)},
                "object": {
                    "path": object_file.name, "sha256": sha256_file(object_file),
                },
                "bridge": {"path": bridge.name, "sha256": sha256_file(bridge)},
            },
        }
        path = directory / f"{task.artifact_stem}.json"
        path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return path

    def test_complete_set_registry_and_cmake_are_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory(
            dir=TOOLS, prefix="contract-fixture-",
        ) as temporary:
            root = Path(temporary)
            metadata = [self._write_artifact(root, task) for task in self.tasks]
            artifacts = verify_complete_artifact_set(self.space, metadata)
            registry_path = root / "registry.cu"
            registry = render_registry(artifacts)
            registry_path.write_text(registry, encoding="utf-8")
            self.assertEqual(registry.count("kQkvRopeNativeAbiVersion"), 8)
            self.assertEqual(registry.count("kDualFfnNativeAbiVersion"), 8)
            self.assertIn("const half2*", registry)
            self.assertIn("half*, int, int, cudaStream_t", registry)
            for task in self.tasks:
                self.assertIn(task.candidate_id, registry)
                self.assertIn(task.prepare_symbol, registry)
                self.assertIn(task.launch_symbol, registry)
            cmake = render_cmake_manifest(registry_path, artifacts)
            self.assertIn("KATAGO_C384_EXACT_AOT_GENERATED_FILE_SHA256", cmake)
            self.assertIn(sha256_file(registry_path), cmake)
            for artifact in artifacts:
                self.assertIn(sha256_file(artifact.object_file), cmake)

            # Hash drift must be rejected before registry emission.
            artifacts[0].object_file.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_artifact(self.space, artifacts[0].metadata_path)

    def test_incomplete_set_and_stale_abi_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(
            dir=TOOLS, prefix="contract-fixture-",
        ) as temporary:
            root = Path(temporary)
            metadata = [self._write_artifact(root, task) for task in self.tasks]
            with self.assertRaisesRegex(ValueError, "every B28/B24"):
                verify_complete_artifact_set(self.space, metadata[:-1])
            value = json.loads(metadata[0].read_text(encoding="utf-8"))
            value["native_abi"] = 0
            metadata[0].write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "native ABI mismatch"):
                verify_artifact(self.space, metadata[0])

    def test_old_shape_literals_are_not_silently_reused(self) -> None:
        qkv = (TOOLS / "generate_qkv_rope.py").read_text(encoding="utf-8")
        dual = (TOOLS / "generate_dual_ffn.py").read_text(encoding="utf-8")
        self.assertNotIn("SEQUENCE = 361", qkv)
        self.assertNotIn("% 361", qkv)
        self.assertNotIn("fixed_board", qkv)
        self.assertNotIn("SEQUENCE = 361", dual)
        self.assertNotIn("OUTPUT_CHANNELS = 1152", dual)
        self.assertNotIn("WIDE_CHANNELS = 2304", dual)
        self.assertNotIn("fixed_board", dual)
        self.assertIn("OUTPUT_CHANNELS = 1024", dual)
        self.assertIn("WIDE_CHANNELS = 2048", dual)


if __name__ == "__main__":
    unittest.main()

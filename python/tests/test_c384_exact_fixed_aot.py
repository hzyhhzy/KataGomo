#!/usr/bin/env python3
"""CPU-only contracts for the C384 exact fixed-batch AOT generators."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
import uuid
from contextlib import contextmanager


TOOLS = Path(__file__).resolve().parents[1] / "c384_exact_fixed_aot"
CMAKE_POLICY = (
    Path(__file__).resolve().parents[2] / "cpp" / "cmake" /
    "C384ExactAotPolicy.cmake"
)
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


@contextmanager
def writable_fixture():
    # Windows sandbox ACLs may make tempfile-created directories unreadable to
    # the creating subprocess. A unique child of this writable test directory
    # keeps the fixture scoped and still guarantees cleanup.
    root = Path(__file__).resolve().parent / ("c384-exact-fixture-" + uuid.uuid4().hex)
    root.mkdir()
    try:
        yield root
    finally:
        shutil.rmtree(root)


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
            self.assertNotIn("_cuda_init", launch)
            self.assertNotIn("_cuda_load_to_device", launch)
            self.assertNotIn(
                f"{task.artifact_stem}_Kernel_Module_Load(&", source,
            )
            self.assertIn(
                f"_mlir_{task.artifact_stem}_cuda_init", source,
            )
            self.assertIn(
                f"_mlir_{task.artifact_stem}_cuda_load_to_device", source,
            )
            self.assertIn("int32_t deviceId = deviceOrdinal", source)
            self.assertNotIn("for (int32_t i", source)
            self.assertIn("preparationFailures[deviceOrdinal] = status", source)
            if task.family == "qkv_rope":
                self.assertIn("const half2* cosSin", launch)
                self.assertIn("Tensor_table_arg_t", launch)
            else:
                self.assertIn("(void)unusedGateWeights", launch)

    def test_bridge_raw_loader_compiles_and_propagates_failures(self) -> None:
        task = next(task for task in self.tasks
                    if task.family == "qkv_rope" and task.batch == 28)
        with writable_fixture() as root:
            (root / "cuda_runtime.h").write_text(r'''
#pragma once
#include <cstdint>
struct CUlib_st;
using cudaLibrary_t = CUlib_st*;
using cudaStream_t = void*;
enum cudaError_t {
  cudaSuccess = 0,
  cudaErrorInvalidValue = 1,
  cudaErrorInvalidDevice = 10,
  cudaErrorNotReady = 34,
  cudaErrorUnknown = 999,
};
extern int fakeDeviceCount;
extern int fakeUnloadCalls;
inline cudaError_t cudaGetDeviceCount(int* count) {
  *count = fakeDeviceCount;
  return cudaSuccess;
}
inline cudaError_t cudaLibraryUnload(cudaLibrary_t) {
  ++fakeUnloadCalls;
  return cudaSuccess;
}
inline cudaError_t cudaPeekAtLastError() { return cudaSuccess; }
''', encoding="utf-8")
            (root / "cuda_fp16.h").write_text(r'''
#pragma once
struct half { unsigned short bits; };
struct half2 { half x; half y; };
''', encoding="utf-8")
            header = f'''#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
typedef struct {{ cudaLibrary_t module; }} {task.artifact_stem}_Kernel_Module_t;
typedef struct {{ void* data; }} {task.artifact_stem}_Tensor_a_arg_t;
typedef struct {{ void* data; }} {task.artifact_stem}_Tensor_b_arg_t;
typedef struct {{ void* data; }} {task.artifact_stem}_Tensor_c_arg_t;
typedef struct {{ void* data; }} {task.artifact_stem}_Tensor_table_arg_t;
extern int fakeMode;
extern int fakeInitCalls;
extern int fakeLoadCalls;
extern int fakeLastDevice;
extern int fakeWrapperCalls;
extern "C" inline void _mlir_{task.artifact_stem}_cuda_init(void** opaque) {{
  struct Args {{ cudaLibrary_t** library; cudaError_t* status; }};
  auto* args = reinterpret_cast<Args*>(opaque);
  ++fakeInitCalls;
  **args->library = reinterpret_cast<cudaLibrary_t>(uintptr_t(0x1234));
  *args->status = fakeMode == 1 ? cudaErrorUnknown : cudaSuccess;
}}
extern "C" inline void _mlir_{task.artifact_stem}_cuda_load_to_device(
    void** opaque) {{
  struct Args {{
    cudaLibrary_t** library;
    int32_t* deviceId;
    cudaError_t* status;
  }};
  auto* args = reinterpret_cast<Args*>(opaque);
  (void)args->library;
  ++fakeLoadCalls;
  fakeLastDevice = *args->deviceId;
  *args->status = fakeMode == 2 ? cudaErrorUnknown : cudaSuccess;
}}
extern "C" inline int32_t cute_dsl_{task.artifact_stem}_wrapper(
    {task.artifact_stem}_Kernel_Module_t*,
    {task.artifact_stem}_Tensor_a_arg_t*,
    {task.artifact_stem}_Tensor_b_arg_t*,
    {task.artifact_stem}_Tensor_c_arg_t*,
    {task.artifact_stem}_Tensor_table_arg_t*, cudaStream_t) {{
  ++fakeWrapperCalls;
  return 0;
}}
'''
            (root / f"{task.artifact_stem}.h").write_text(
                header, encoding="utf-8",
            )
            source = render_qkv_rope_bridge(task) + f'''
int fakeDeviceCount = 3;
int fakeUnloadCalls = 0;
int fakeMode = 0;
int fakeInitCalls = 0;
int fakeLoadCalls = 0;
int fakeLastDevice = -1;
int fakeWrapperCalls = 0;

int main() {{
  half* pointer = nullptr;
  if({task.prepare_symbol}(3) != cudaErrorInvalidDevice || fakeInitCalls != 0)
    return 1;

  fakeMode = 1;
  if({task.prepare_symbol}(0) != cudaErrorUnknown || fakeInitCalls != 1 ||
     fakeLoadCalls != 0 || fakeUnloadCalls != 1)
    return 2;
  if({task.launch_symbol}(pointer,pointer,
       reinterpret_cast<half2*>(pointer),pointer,{task.token_rows},0,nullptr) !=
       cudaErrorNotReady)
    return 3;
  if({task.prepare_symbol}(0) != cudaErrorUnknown || fakeInitCalls != 1)
    return 4;

  fakeMode = 2;
  if({task.prepare_symbol}(1) != cudaErrorUnknown || fakeInitCalls != 2 ||
     fakeLoadCalls != 1 || fakeLastDevice != 1 || fakeUnloadCalls != 2)
    return 5;
  if({task.launch_symbol}(pointer,pointer,
       reinterpret_cast<half2*>(pointer),pointer,{task.token_rows},1,nullptr) !=
       cudaErrorNotReady)
    return 6;
  if({task.prepare_symbol}(1) != cudaErrorUnknown || fakeLoadCalls != 1)
    return 7;

  fakeMode = 0;
  if({task.prepare_symbol}(2) != cudaSuccess || fakeInitCalls != 3 ||
     fakeLoadCalls != 2 || fakeLastDevice != 2 || fakeUnloadCalls != 2)
    return 8;
  if({task.prepare_symbol}(2) != cudaSuccess || fakeInitCalls != 3)
    return 9;
  if({task.launch_symbol}(pointer,pointer,
       reinterpret_cast<half2*>(pointer),pointer,{task.token_rows},2,nullptr) !=
       cudaSuccess || fakeWrapperCalls != 1)
    return 10;
  return 0;
}}
'''
            source_path = root / "bridge_contract.cpp"
            source_path.write_text(source, encoding="utf-8")
            executable = root / ("bridge_contract.exe" if os.name == "nt"
                                 else "bridge_contract")

            if os.name == "nt":
                roots = [Path(r"C:\Program Files\Microsoft Visual Studio")]
                vcvars = next((path for base in roots for path in
                               base.glob("*/Community/VC/Auxiliary/Build/"
                                         "vcvars64.bat")), None)
                if vcvars is None:
                    self.skipTest("MSVC vcvars64.bat is required")
                compile_args = [
                    "cl", "/nologo", "/EHsc", "/std:c++17",
                    f"/I{root}", str(source_path), f"/Fe:{executable}",
                    f"/Fo:{root / 'bridge_contract.obj'}",
                ]
                compile_batch = root / "compile-contract.bat"
                compile_batch.write_text(
                    f'@call "{vcvars}" >nul\n@' +
                    subprocess.list2cmdline(compile_args) + "\n",
                    encoding="utf-8",
                )
                compiled = subprocess.run(
                    ["cmd", "/d", "/c", str(compile_batch)],
                    text=True, capture_output=True, check=False,
                )
            else:
                compiler = next((shutil.which(name)
                                 for name in ("c++", "clang++", "g++")
                                 if shutil.which(name)), None)
                if compiler is None:
                    self.skipTest("a C++17 compiler is required")
                compiled = subprocess.run(
                    [compiler, "-std=c++17", "-pthread", "-I", str(root),
                     str(source_path), "-o", str(executable)],
                    text=True, capture_output=True, check=False,
                )
            self.assertEqual(
                compiled.returncode, 0, compiled.stdout + compiled.stderr,
            )
            executed = subprocess.run(
                [str(executable)], text=True, capture_output=True, check=False,
            )
            self.assertEqual(
                executed.returncode, 0, executed.stdout + executed.stderr,
            )

    def _write_artifact(self, directory: Path, task) -> Path:
        header = directory / f"{task.artifact_stem}.h"
        object_file = directory / f"{task.artifact_stem}.o"
        bridge = directory / f"{task.artifact_stem}_bridge.cu"
        wrapper = f"cute_dsl_{task.artifact_stem}_wrapper"
        raw_ciface = f"_mlir_{task.artifact_stem}__mlir_ciface_cutlass_launch"
        header.write_text(
            f'extern "C" void _mlir_{task.artifact_stem}_cuda_init(void**);\n'
            f'extern "C" void '
            f'_mlir_{task.artifact_stem}_cuda_load_to_device(void**);\n'
            f'extern "C" void {raw_ciface}(void**);\n'
            f'static inline int32_t {wrapper}() {{ return 0; }}\n',
            encoding="utf-8",
        )
        raw_symbols = (
            f"_mlir_{task.artifact_stem}_cuda_init",
            f"_mlir_{task.artifact_stem}_cuda_load",
            f"_mlir_{task.artifact_stem}_cuda_load_to_device",
            f"_mlir_{task.artifact_stem}_cuda_num_binaries",
            raw_ciface,
        )
        object_file.write_bytes(
            b"\x7fELFsynthetic-sm120-object\0" +
            b"\0".join(symbol.encode("ascii") for symbol in raw_symbols)
        )
        bridge.write_text(
            render_qkv_rope_bridge(task) if task.family == "qkv_rope"
            else render_dual_ffn_bridge(task), encoding="utf-8",
        )
        metadata = {
            "schema": 2,
            "kind": "katago-c384-exact-aot-artifact",
            "generation_complete": True,
            "verified_on_target": False,
            "compute_capability": "sm_120",
            "dtype": "fp16",
            "search_space_sha256": canonical_json_sha256(self.space),
            "family": task.family,
            "candidate_id": task.candidate_id,
            "batch": task.batch,
            "token_rows": task.token_rows,
            "native_abi": task.native_abi,
            "artifact_stem": task.artifact_stem,
            "max_active_clusters": task.max_active_clusters,
            "shape": {
                "sequence": 225, "channels": 384, "heads": 12,
                "kv_heads": 12, "head_dim": 32, "ffn_channels": 1024,
            },
            "provenance": {
                "generator_sha256": sha256_file(
                    TOOLS / ("generate_qkv_rope.py" if task.family == "qkv_rope"
                             else "generate_dual_ffn.py")
                ),
                "bridge_codegen_sha256": sha256_file(
                    TOOLS / "bridge_codegen.py"
                ),
                "cutlass_commit": "dcf215af68a2d08d305076c152a06f201728cd53",
                "dense_gemm_sha256": "1" * 64,
                "patched_dense_gemm_sha256": "2" * 64,
                "nvidia_cutlass_dsl": {
                    "module": "cutlass",
                    "version": "4.6.0.dev0",
                    "version_module": "cutlass",
                    "module_file": "/staged/cutlass/__init__.py",
                    "module_file_sha256": "6" * 64,
                    "distribution": "nvidia-cutlass-dsl",
                    "metadata_status": "absent",
                    "distribution_version": None,
                },
                "cuda_bindings": {
                    "module": "cuda.bindings.driver",
                    "version": "13.3.1",
                    "version_module": "cuda.bindings._version",
                    "module_file": "/staged/cuda/bindings/driver.pyd",
                    "module_file_sha256": "7" * 64,
                    "distribution": "cuda-bindings",
                    "metadata_status": "present-matching",
                    "distribution_version": "13.3.1",
                },
                "nvcc": "Cuda compilation tools, release 13.0, V13.0.0",
                "gpu_kernel_executed": False,
            },
            "coordinate": ({
                "tile": list(task.tile),
                "atom_layout": list(task.atom_layout),
                "input": [task.token_rows, 384],
                "packed_weights": [384, 1152],
                "packed_output": [task.token_rows, 1152],
                "rope_table_half2": [225, 192],
            } if task.family == "qkv_rope" else {
                "tile": list(task.tile),
                "atom_layout": list(task.atom_layout),
                "input": [task.token_rows, 384],
                "paired_weights": [384, 2048],
                "output": [task.token_rows, 1024],
                "effective_output_tile": list(task.effective_output_tile or ()),
                "epilogue": "silu-linear-times-gate",
            }),
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

    def _write_promotion(self, directory: Path, qkv, dual) -> Path:
        path = directory / "production-promotion.json"
        value = {
            "schema": 1,
            "kind": "katago-c384-exact-aot-production-promotion",
            "status": "PASSED",
            "selected": {
                "batch": qkv.task.batch,
                "qkv_rope": qkv.task.candidate_id,
                "dual_ffn": dual.task.candidate_id,
            },
            "gates": {
                name: {"status": "PASSED", "evidence_sha256": digit * 64}
                for name, digit in (
                    ("activation", "3"), ("long_gate", "4"),
                    ("accuracy", "5"),
                )
            },
        }
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        return path

    def _run_cmake_policy(self, directory: Path, source: str):
        script = directory / ("policy-" + uuid.uuid4().hex + ".cmake")
        policy = CMAKE_POLICY.resolve().as_posix()
        script.write_text(
            f'include("{policy}")\n{source}\n', encoding="utf-8",
        )
        return subprocess.run(
            [shutil.which("cmake") or "cmake", "-P", str(script)],
            text=True, capture_output=True, check=False,
        )

    @staticmethod
    def _cmake_pair_policy_source(qkv_id: str, dual_id: str) -> str:
        return f'''
set(KATAGO_C384_EXACT_AOT_PACKAGE_SCHEMA "1")
set(KATAGO_C384_EXACT_AOT_PACKAGE_MODE "SEARCH_PAIR")
set(KATAGO_C384_EXACT_AOT_SELECTED_BATCH "28")
set(KATAGO_C384_EXACT_AOT_SELECTED_QKV_ID "{qkv_id}")
set(KATAGO_C384_EXACT_AOT_SELECTED_DUAL_FFN_ID "{dual_id}")
set(KATAGO_C384_EXACT_AOT_SELECTED_FAMILIES "QKV_ROPE;DUAL_FFN")
set(KATAGO_C384_EXACT_AOT_QKV_ROPE_IDS "{qkv_id}")
set(KATAGO_C384_EXACT_AOT_DUAL_FFN_IDS "{dual_id}")
set(KATAGO_C384_EXACT_AOT_PROMOTION_EVIDENCE "")
set(KATAGO_C384_EXACT_AOT_PROMOTION_EVIDENCE_SHA256 "")
set(KATAGO_C384_EXACT_AOT_GENERATED_HEADERS h1 h2)
set(KATAGO_C384_EXACT_AOT_GENERATED_OBJECTS o1 o2)
set(KATAGO_C384_EXACT_AOT_GENERATED_BRIDGES b1 b2)
set(KATAGO_C384_EXACT_AOT_GENERATED_METADATA m1 m2)
set(KATAGO_C384_EXACT_REQUESTED_QKV_TACTIC_ID "")
set(KATAGO_C384_EXACT_REQUESTED_DUAL_FFN_TACTIC_ID "")
set(KATAGO_C384_H12_FA4_PACKAGE_MODE "BATCH_SEARCH")
set(KATAGO_C384_H12_FA4_BATCHES 24 28)
'''

    def test_complete_set_registry_and_cmake_are_hash_bound(self) -> None:
        with writable_fixture() as root:
            metadata = [self._write_artifact(root, task) for task in self.tasks]
            artifacts = verify_complete_artifact_set(self.space, metadata)
            registry = render_registry(artifacts)
            self.assertEqual(registry.count("kQkvRopeNativeAbiVersion"), 8)
            self.assertEqual(registry.count("kDualFfnNativeAbiVersion"), 8)
            self.assertIn("const half2*", registry)
            self.assertIn("half*, int, int, cudaStream_t", registry)
            for task in self.tasks:
                self.assertIn(task.candidate_id, registry)
                self.assertIn(task.prepare_symbol, registry)
                self.assertIn(task.launch_symbol, registry)
            selected = [
                next(a for a in artifacts
                     if a.task.family == "qkv_rope" and a.task.batch == 28),
                next(a for a in artifacts
                     if a.task.family == "dual_ffn" and a.task.batch == 28),
            ]
            registry_path = root / "registry.cu"
            registry_path.write_text(render_registry(selected), encoding="utf-8")
            cmake = render_cmake_manifest(registry_path, selected)
            self.assertIn("KATAGO_C384_EXACT_AOT_GENERATED_FILE_SHA256", cmake)
            self.assertIn(sha256_file(registry_path), cmake)
            for artifact in selected:
                self.assertIn(sha256_file(artifact.object_file), cmake)

            # Hash drift must be rejected before registry emission.
            artifacts[0].object_file.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_artifact(self.space, artifacts[0].metadata_path)

    def test_pair_and_production_manifests_are_family_bound(self) -> None:
        with writable_fixture() as root:
            metadata = [self._write_artifact(root, task) for task in self.tasks]
            artifacts = verify_complete_artifact_set(self.space, metadata)
            qkv = next(a for a in artifacts
                       if a.task.family == "qkv_rope" and a.task.batch == 28)
            dual = next(a for a in artifacts
                        if a.task.family == "dual_ffn" and a.task.batch == 28)
            losing = next(a for a in artifacts
                          if a.task.batch == 28 and a not in (qkv, dual))
            registry_path = root / "selected-registry.cu"
            registry_path.write_text(
                render_registry([qkv, dual]), encoding="utf-8",
            )

            search = render_cmake_manifest(
                registry_path, [qkv, dual], mode="SEARCH_PAIR",
            )
            self.assertIn('KATAGO_C384_EXACT_AOT_PACKAGE_SCHEMA "1"', search)
            self.assertIn('KATAGO_C384_EXACT_AOT_PACKAGE_MODE "SEARCH_PAIR"', search)
            self.assertIn(
                f'KATAGO_C384_EXACT_AOT_QKV_ROPE_IDS "{qkv.task.candidate_id}"',
                search,
            )
            self.assertIn(
                f'KATAGO_C384_EXACT_AOT_DUAL_FFN_IDS "{dual.task.candidate_id}"',
                search,
            )
            self.assertNotIn(losing.task.candidate_id, search)
            self.assertEqual(search.count('.o" "'), 2)

            with self.assertRaisesRegex(ValueError, "one QKV and one dual"):
                render_cmake_manifest(registry_path, [qkv, qkv])
            dual24 = next(a for a in artifacts
                          if a.task.family == "dual_ffn" and a.task.batch == 24)
            with self.assertRaisesRegex(ValueError, "same fixed batch"):
                render_cmake_manifest(registry_path, [qkv, dual24])
            with self.assertRaisesRegex(ValueError, "requires promotion evidence"):
                render_cmake_manifest(
                    registry_path, [qkv, dual], mode="PRODUCTION",
                )

            promotion = self._write_promotion(root, qkv, dual)
            production = render_cmake_manifest(
                registry_path, [qkv, dual], mode="PRODUCTION",
                promotion_evidence=promotion,
            )
            self.assertIn('KATAGO_C384_EXACT_AOT_PACKAGE_MODE "PRODUCTION"',
                          production)
            self.assertIn(sha256_file(promotion), production)
            value = json.loads(promotion.read_text(encoding="utf-8"))
            value["selected"]["dual_ffn"] = qkv.task.candidate_id
            promotion.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selected pair mismatch"):
                render_cmake_manifest(
                    registry_path, [qkv, dual], mode="PRODUCTION",
                    promotion_evidence=promotion,
                )

    @unittest.skipUnless(shutil.which("cmake"), "cmake is required")
    def test_cmake_policy_derives_ids_and_rejects_wrong_family(self) -> None:
        qkv = next(t for t in self.tasks
                   if t.family == "qkv_rope" and t.batch == 28)
        dual = next(t for t in self.tasks
                    if t.family == "dual_ffn" and t.batch == 28)
        with writable_fixture() as root:
            metadata = [self._write_artifact(root, task) for task in (qkv, dual)]
            pair = [verify_artifact(self.space, path) for path in metadata]
            registry = root / "registry.cu"
            registry.write_text(render_registry(pair), encoding="utf-8")
            manifest = root / "manifest.cmake"
            manifest.write_text(
                render_cmake_manifest(registry, pair, mode="SEARCH_PAIR"),
                encoding="utf-8",
            )
            emitted = f'''
set(KATAGO_C384_EXACT_EFFECTIVE_QKV_TACTIC_ID "")
set(KATAGO_C384_EXACT_EFFECTIVE_DUAL_FFN_TACTIC_ID "")
set(KATAGO_C384_EXACT_REQUESTED_QKV_TACTIC_ID "")
set(KATAGO_C384_EXACT_REQUESTED_DUAL_FFN_TACTIC_ID "")
set(KATAGO_C384_H12_FA4_PACKAGE_MODE "BATCH_SEARCH")
set(KATAGO_C384_H12_FA4_BATCHES 24 28)
include("{manifest.resolve().as_posix()}")
katago_c384_exact_resolve_manifest_policy()
if(NOT KATAGO_C384_EXACT_EFFECTIVE_QKV_TACTIC_ID STREQUAL "{qkv.candidate_id}" OR
   NOT KATAGO_C384_EXACT_EFFECTIVE_DUAL_FFN_TACTIC_ID STREQUAL "{dual.candidate_id}")
  message(FATAL_ERROR "emitted manifest did not derive effective IDs")
endif()
'''
            result = self._run_cmake_policy(root, emitted)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            valid = self._cmake_pair_policy_source(
                qkv.candidate_id, dual.candidate_id,
            ) + f'''
katago_c384_exact_resolve_manifest_policy()
if(NOT KATAGO_C384_EXACT_EFFECTIVE_QKV_TACTIC_ID STREQUAL "{qkv.candidate_id}" OR
   NOT KATAGO_C384_EXACT_EFFECTIVE_DUAL_FFN_TACTIC_ID STREQUAL "{dual.candidate_id}")
  message(FATAL_ERROR "effective IDs were not derived from the manifest")
endif()
'''
            result = self._run_cmake_policy(root, valid)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            wrong_family = self._cmake_pair_policy_source(
                dual.candidate_id, qkv.candidate_id,
            ) + "katago_c384_exact_resolve_manifest_policy()"
            result = self._run_cmake_policy(root, wrong_family)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("wrong family", result.stdout + result.stderr)

            override_mismatch = self._cmake_pair_policy_source(
                qkv.candidate_id, dual.candidate_id,
            ).replace(
                'set(KATAGO_C384_EXACT_REQUESTED_QKV_TACTIC_ID "")',
                'set(KATAGO_C384_EXACT_REQUESTED_QKV_TACTIC_ID "wrong-qkv")',
            ) + "katago_c384_exact_resolve_manifest_policy()"
            result = self._run_cmake_policy(root, override_mismatch)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("disagrees", result.stdout + result.stderr)

            cross_batch_dual = next(
                t for t in self.tasks
                if t.family == "dual_ffn" and t.batch == 24
            )
            cross_batch = self._cmake_pair_policy_source(
                qkv.candidate_id, cross_batch_dual.candidate_id,
            ) + "katago_c384_exact_resolve_manifest_policy()"
            result = self._run_cmake_policy(root, cross_batch)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("wrong family or fixed batch",
                          result.stdout + result.stderr)

            wrong_fa4_mode = self._cmake_pair_policy_source(
                qkv.candidate_id, dual.candidate_id,
            ).replace(
                'set(KATAGO_C384_H12_FA4_PACKAGE_MODE "BATCH_SEARCH")',
                'set(KATAGO_C384_H12_FA4_PACKAGE_MODE "PRODUCTION")',
            ) + "katago_c384_exact_resolve_manifest_policy()"
            result = self._run_cmake_policy(root, wrong_fa4_mode)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("requires a BATCH_SEARCH FA4 package",
                          result.stdout + result.stderr)

    @unittest.skipUnless(shutil.which("cmake"), "cmake is required")
    def test_cmake_request_policy_fails_closed_without_gpu(self) -> None:
        qkv = next(t for t in self.tasks
                   if t.family == "qkv_rope" and t.batch == 28)
        with writable_fixture() as root:
            non_cuda = '''
set(USE_BACKEND TENSORRT)
set(KATAGO_ENABLE_SM120_TRANSFORMER_WINNER 1)
set(KATAGO_C384_EXACT_AOT_GENERATED_MANIFEST generated.cmake)
set(KATAGO_C384_EXACT_QKV_TACTIC_ID "")
set(KATAGO_C384_EXACT_DUAL_FFN_TACTIC_ID "")
set(KATAGO_C384_H12_FA4_PACKAGE_CMAKE "")
katago_c384_exact_validate_request_policy()
'''
            result = self._run_cmake_policy(root, non_cuda)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("only with USE_BACKEND=CUDA",
                          result.stdout + result.stderr)

            no_manifest = f'''
set(USE_BACKEND CUDA)
set(KATAGO_ENABLE_SM120_TRANSFORMER_WINNER 1)
set(KATAGO_C384_EXACT_AOT_GENERATED_MANIFEST "")
set(KATAGO_C384_EXACT_QKV_TACTIC_ID "{qkv.candidate_id}")
set(KATAGO_C384_EXACT_DUAL_FFN_TACTIC_ID "")
set(KATAGO_C384_H12_FA4_PACKAGE_CMAKE fa4.cmake)
katago_c384_exact_validate_request_policy()
'''
            result = self._run_cmake_policy(root, no_manifest)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("require a generated exact-AOT manifest",
                          result.stdout + result.stderr)

            old_cuda = '''
set(CMAKE_CUDA_COMPILER_VERSION 12.9)
set(KATAGO_C384_EXACT_AOT_GENERATED_MANIFEST generated.cmake)
set(KATAGO_C384_EXACT_QKV_TACTIC_ID "")
set(KATAGO_C384_EXACT_DUAL_FFN_TACTIC_ID "")
set(KATAGO_C384_H12_FA4_PACKAGE_CMAKE "")
katago_c384_exact_validate_cuda_version()
'''
            result = self._run_cmake_policy(root, old_cuda)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("require CUDA 13.0 or newer",
                          result.stdout + result.stderr)

    def test_incomplete_set_and_stale_abi_fail_closed(self) -> None:
        with writable_fixture() as root:
            metadata = [self._write_artifact(root, task) for task in self.tasks]
            with self.assertRaisesRegex(ValueError, "every B28/B24"):
                verify_complete_artifact_set(self.space, metadata[:-1])
            value = json.loads(metadata[0].read_text(encoding="utf-8"))
            value["native_abi"] = 0
            metadata[0].write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "native ABI mismatch"):
                verify_artifact(self.space, metadata[0])

    def test_imported_tool_provenance_fails_closed(self) -> None:
        task = self.tasks[0]
        with writable_fixture() as root:
            metadata_path = self._write_artifact(root, task)
            baseline = json.loads(metadata_path.read_text(encoding="utf-8"))

            def missing_cutlass(value):
                del value["provenance"]["nvidia_cutlass_dsl"]

            def wrong_distribution(value):
                value["provenance"]["cuda_bindings"]["distribution"] = (
                    "cuda-python"
                )

            def bad_module_hash(value):
                value["provenance"]["nvidia_cutlass_dsl"][
                    "module_file_sha256"
                ] = "not-a-sha256"

            def mismatched_dist_version(value):
                value["provenance"]["cuda_bindings"][
                    "distribution_version"
                ] = "13.4.0"

            cases = (
                ("missing record", missing_cutlass, "must be a record"),
                ("wrong distribution", wrong_distribution,
                 "distribution mismatch"),
                ("bad module hash", bad_module_hash,
                 "module file hash is invalid"),
                ("mismatched version", mismatched_dist_version,
                 "dist-info version mismatch"),
            )
            for label, mutate, expected in cases:
                with self.subTest(label=label):
                    value = json.loads(json.dumps(baseline))
                    mutate(value)
                    metadata_path.write_text(
                        json.dumps(value), encoding="utf-8",
                    )
                    with self.assertRaisesRegex(ValueError, expected):
                        verify_artifact(self.space, metadata_path)

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

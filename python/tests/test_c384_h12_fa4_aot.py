#!/usr/bin/env python3
"""CPU/compiler contracts for the C384/H12 exact FA4 bridge package."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
import uuid


TOOLS = Path(__file__).resolve().parents[1] / "c384_h12_fa4"
sys.path.insert(0, str(TOOLS))

from build_exact_fa4_aot import candidate_id, render_bridge
from package_artifacts import load_metadata, sha256


@contextmanager
def writable_fixture():
    root = Path(__file__).resolve().parent / ("c384-fa4-fixture-" + uuid.uuid4().hex)
    root.mkdir()
    try:
        yield root
    finally:
        shutil.rmtree(root)


class C384H12Fa4AotTest(unittest.TestCase):
    @staticmethod
    def _args() -> argparse.Namespace:
        return argparse.Namespace(
            batch=28, tile_m=128, tile_n=128, num_stages=1,
            num_threads=128, accumulation="both16", layout="packed-token",
            abi_mode="search",
        )

    def test_raw_loader_compiles_and_propagates_failures(self) -> None:
        args = self._args()
        stem = "c384_fa4_b28m128n128s1w4both16t"
        prefix = "c384fa4b28m128n128s1w4both16t"
        identity = candidate_id(args)
        with writable_fixture() as root:
            (root / "cuda_runtime.h").write_text(r'''
#pragma once
#include <cstdint>
struct CUlib_st;
using cudaLibrary_t = CUlib_st*;
using cudaStream_t = void*;
enum cudaError_t {
  cudaSuccess = 0, cudaErrorInvalidValue = 1, cudaErrorInvalidDevice = 10,
  cudaErrorNotReady = 34, cudaErrorUnknown = 999,
};
extern int fakeDeviceCount;
extern int fakeUnloadCalls;
inline cudaError_t cudaGetDeviceCount(int* count) {
  *count = fakeDeviceCount; return cudaSuccess;
}
inline cudaError_t cudaLibraryUnload(cudaLibrary_t) {
  ++fakeUnloadCalls; return cudaSuccess;
}
inline cudaError_t cudaPeekAtLastError() { return cudaSuccess; }
''', encoding="utf-8")
            (root / f"{stem}.h").write_text(f'''#pragma once
#include <cuda_runtime.h>
#include <cstdint>
typedef struct {{ cudaLibrary_t module; }} {prefix}_Kernel_Module_t;
typedef struct {{ void* data; }} {prefix}_Tensor_mQ_t;
typedef struct {{ void* data; }} {prefix}_Tensor_mK_t;
typedef struct {{ void* data; }} {prefix}_Tensor_mV_t;
typedef struct {{ void* data; }} {prefix}_Tensor_mO_t;
extern int fakeMode;
extern int fakeInitCalls;
extern int fakeLoadCalls;
extern int fakeLastDevice;
extern int fakeWrapperCalls;
extern "C" inline void _mlir_{prefix}_cuda_init(void** opaque) {{
  struct Args {{ cudaLibrary_t** library; cudaError_t* status; }};
  auto* args = reinterpret_cast<Args*>(opaque);
  ++fakeInitCalls;
  **args->library = reinterpret_cast<cudaLibrary_t>(uintptr_t(0x1234));
  *args->status = fakeMode == 1 ? cudaErrorUnknown : cudaSuccess;
}}
extern "C" inline void _mlir_{prefix}_cuda_load_to_device(void** opaque) {{
  struct Args {{
    cudaLibrary_t** library; int32_t* deviceId; cudaError_t* status;
  }};
  auto* args = reinterpret_cast<Args*>(opaque);
  (void)args->library;
  ++fakeLoadCalls;
  fakeLastDevice = *args->deviceId;
  *args->status = fakeMode == 2 ? cudaErrorUnknown : cudaSuccess;
}}
extern "C" inline int32_t cute_dsl_{prefix}_wrapper(
    {prefix}_Kernel_Module_t*, {prefix}_Tensor_mQ_t*,
    {prefix}_Tensor_mK_t*, {prefix}_Tensor_mV_t*,
    {prefix}_Tensor_mO_t*, float, cudaStream_t) {{
  ++fakeWrapperCalls; return 0;
}}
''', encoding="utf-8")
            source = render_bridge(stem, prefix, identity, args) + f'''
int fakeDeviceCount = 3;
int fakeUnloadCalls = 0;
int fakeMode = 0;
int fakeInitCalls = 0;
int fakeLoadCalls = 0;
int fakeLastDevice = -1;
int fakeWrapperCalls = 0;

int main() {{
  if({prefix}_prepare(3) != cudaErrorInvalidDevice || fakeInitCalls != 0)
    return 1;
  fakeMode = 1;
  if({prefix}_prepare(0) != cudaErrorUnknown || fakeInitCalls != 1 ||
     fakeLoadCalls != 0 || fakeUnloadCalls != 1)
    return 2;
  if({prefix}_launch(nullptr,nullptr,nullptr,nullptr,28,225,12,32,1.0f,2,0,
                     nullptr) != cudaErrorInvalidValue)
    return 3;
  if({prefix}_prepare(0) != cudaErrorUnknown || fakeInitCalls != 1)
    return 4;
  fakeMode = 2;
  if({prefix}_prepare(1) != cudaErrorUnknown || fakeInitCalls != 2 ||
     fakeLoadCalls != 1 || fakeLastDevice != 1 || fakeUnloadCalls != 2)
    return 5;
  if({prefix}_prepare(1) != cudaErrorUnknown || fakeLoadCalls != 1)
    return 6;
  fakeMode = 0;
  void* pointer = reinterpret_cast<void*>(uintptr_t(0x1000));
  if({prefix}_prepare(2) != cudaSuccess || fakeInitCalls != 3 ||
     fakeLoadCalls != 2 || fakeLastDevice != 2 || fakeUnloadCalls != 2)
    return 7;
  if({prefix}_prepare(2) != cudaSuccess || fakeInitCalls != 3)
    return 8;
  if({prefix}_launch(pointer,pointer,pointer,pointer,28,225,12,32,1.0f,2,2,
                     nullptr) != cudaSuccess || fakeWrapperCalls != 1)
    return 9;
  return 0;
}}
'''
            source_path = root / "fa4_bridge_contract.cpp"
            source_path.write_text(source, encoding="utf-8")
            executable = root / ("fa4_bridge_contract.exe" if os.name == "nt"
                                 else "fa4_bridge_contract")
            if os.name == "nt":
                visual_studio = Path(r"C:\Program Files\Microsoft Visual Studio")
                vcvars = next(iter(visual_studio.glob(
                    "*/Community/VC/Auxiliary/Build/vcvars64.bat"
                )), None)
                if vcvars is None:
                    self.skipTest("MSVC vcvars64.bat is required")
                compile_args = [
                    "cl", "/nologo", "/EHsc", "/std:c++17", f"/I{root}",
                    str(source_path), f"/Fe:{executable}",
                    f"/Fo:{root / 'fa4_bridge_contract.obj'}",
                ]
                batch = root / "compile-contract.bat"
                batch.write_text(
                    f'@call "{vcvars}" >nul\n@' +
                    subprocess.list2cmdline(compile_args) + "\n",
                    encoding="utf-8",
                )
                compiled = subprocess.run(
                    ["cmd", "/d", "/c", str(batch)], text=True,
                    capture_output=True, check=False,
                )
            else:
                compiler = next((shutil.which(name)
                                 for name in ("c++", "clang++", "g++")
                                 if shutil.which(name)), None)
                if compiler is None:
                    self.skipTest("a C++17 compiler is required")
                compiled = subprocess.run(
                    [compiler, "-std=c++17", "-pthread", "-I", str(root),
                     str(source_path), "-o", str(executable)], text=True,
                    capture_output=True, check=False,
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

    def test_package_audits_real_cute_symbol_placement(self) -> None:
        args = self._args()
        stem = "c384_fa4_b28m128n128s1w4both16t"
        prefix = "c384fa4b28m128n128s1w4both16t"
        identity = candidate_id(args)
        with writable_fixture() as root:
            header = root / f"{stem}.h"
            object_file = root / f"{stem}.o"
            bridge = root / f"{stem}_bridge.cpp"
            metadata = root / f"{stem}.json"
            raw_ciface = f"_mlir_{prefix}__mlir_ciface_cutlass_launch"
            header.write_text(
                f"void _mlir_{prefix}_cuda_init(void**);\n"
                f"void _mlir_{prefix}_cuda_load_to_device(void**);\n"
                f"void {raw_ciface}(void**);\n"
                f"static inline int32_t cute_dsl_{prefix}_wrapper() {{return 0;}}\n",
                encoding="utf-8",
            )
            symbols = (
                f"_mlir_{prefix}_cuda_init", f"_mlir_{prefix}_cuda_load",
                f"_mlir_{prefix}_cuda_load_to_device",
                f"_mlir_{prefix}_cuda_num_binaries", raw_ciface,
            )
            object_file.write_bytes(
                b"\x7fELFsynthetic\0" +
                b"\0".join(symbol.encode("ascii") for symbol in symbols)
            )
            bridge.write_text(
                render_bridge(stem, prefix, identity, args), encoding="utf-8",
            )
            value = {
                "schema": 1, "abi_mode": "search", "artifact_stem": stem,
                "symbol_prefix": prefix, "candidate_id": identity, "batch": 28,
                "fixed_shape": {"sequence": 225, "heads": 12, "head_dim": 32},
                "tile": {"m": 128, "n": 128, "num_stages": 1,
                         "num_threads": 128, "num_warps": 4},
                "layout": "packed-token", "layout_id": 2,
                "accumulation": "both16", "accumulation_id": 4,
                "dtype": "fp16", "mask": False,
                "compute_capability": "sm_120", "gpu_used_for_generation": False,
                "generator": {
                    "python": sys.version.split()[0], "cutlass_cuda": "13.0",
                    "generator_sha256": sha256(
                        TOOLS / "build_exact_fa4_aot.py"
                    ),
                    "bridge_generator_sha256": sha256(
                        TOOLS / "build_exact_fa4_aot.py"
                    ),
                },
                "sha256": {
                    "header": sha256(header), "object": sha256(object_file),
                    "bridge": sha256(bridge),
                },
            }
            metadata.write_text(json.dumps(value), encoding="utf-8")
            self.assertEqual(load_metadata(metadata)["candidate_id"], identity)

            # The inline wrapper is a header symbol, never an ELF requirement.
            self.assertNotIn(b"cute_dsl_", object_file.read_bytes())
            value["sha256"]["object"] = sha256(object_file)
            object_file.write_bytes(b"\x7fELFmissing-raw-symbols")
            value["sha256"]["object"] = sha256(object_file)
            metadata.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "raw-symbol gate"):
                load_metadata(metadata)


if __name__ == "__main__":
    unittest.main()

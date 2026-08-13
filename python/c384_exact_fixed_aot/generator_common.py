#!/usr/bin/env python3
"""Shared audited CuTe AOT generator utilities.

The fake DLPack tensors describe CUDA addresses but are never dereferenced by
the Python process. CuTe uses them only to specialize types and emit an SM120
object. Therefore generation requires a CUDA toolchain and the pinned CUTLASS
tree, but it does not execute a kernel or benchmark a GPU.
"""

from __future__ import annotations

import ctypes
import hashlib
import inspect
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from contract import Task, canonical_json_sha256, require, sha256_file


CUTLASS_COMMIT = "dcf215af68a2d08d305076c152a06f201728cd53"
DENSE_GEMM_SHA256 = "613052799aff35d5564d49c8bbb4bbac2e22bc58cb3e27499c4c9c3ee95c6e03"
DENSE_GEMM_RELATIVE = Path(
    "examples/python/CuTeDSL/cute/blackwell_geforce/kernel/"
    "dense_gemm/dense_gemm.py"
)


def git_output(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout.strip()


def validate_cutlass(cutlass_root: Path) -> tuple[Path, str]:
    cutlass_root = cutlass_root.resolve()
    actual = git_output(cutlass_root, "rev-parse", "HEAD")
    require(actual == CUTLASS_COMMIT,
            f"CUTLASS commit mismatch: {actual} != {CUTLASS_COMMIT}")
    require(not git_output(cutlass_root, "status", "--short"),
            "CUTLASS worktree must be clean")
    dense = cutlass_root / DENSE_GEMM_RELATIVE
    require(dense.is_file(), f"missing pinned dense source: {dense}")
    require(sha256_file(dense) == DENSE_GEMM_SHA256,
            "pinned dense_gemm.py hash mismatch")
    return dense, actual


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8), ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p), ("device", _DLDevice),
        ("ndim", ctypes.c_int), ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


_DL_DELETER = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", _DLTensor), ("manager_ctx", ctypes.c_void_p),
        ("deleter", _DL_DELETER),
    ]


_PY_CAPSULE_NEW = ctypes.pythonapi.PyCapsule_New
_PY_CAPSULE_NEW.restype = ctypes.py_object
_PY_CAPSULE_NEW.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
_KEEPALIVE: list["StaticCudaTensorSpec"] = []


class StaticCudaTensorSpec:
    def __init__(self, shape: tuple[int, ...], strides: tuple[int, ...],
                 address: int, device_id: int = 0) -> None:
        require(len(shape) == len(strides), "shape/stride rank mismatch")
        self._shape = (ctypes.c_int64 * len(shape))(*shape)
        self._strides = (ctypes.c_int64 * len(strides))(*strides)
        tensor = _DLTensor(
            ctypes.c_void_p(address), _DLDevice(2, device_id), len(shape),
            _DLDataType(2, 16, 1), self._shape, self._strides, 0,
        )
        self._managed = _DLManagedTensor(tensor, None, _DL_DELETER())

    def __dlpack__(self, stream: int | None = None):
        del stream
        return _PY_CAPSULE_NEW(
            ctypes.addressof(self._managed), b"dltensor", None,
        )

    def __dlpack_device__(self) -> tuple[int, int]:
        return (2, self._managed.dl_tensor.device.device_id)


def make_tensor(from_dlpack, shape: tuple[int, ...], strides: tuple[int, ...],
                address: int, device_id: int = 0):
    spec = StaticCudaTensorSpec(shape, strides, address, device_id)
    _KEEPALIVE.append(spec)
    return from_dlpack(spec, assumed_align=16, enable_tvm_ffi=False)


def load_module_from_source(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import generated source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def replace_exactly_once(source: str, before: str, after: str) -> str:
    require(source.count(before) == 1,
            f"pinned dense source drift near: {before[:100]}")
    return source.replace(before, after)


def bind_local_stream_annotation(function, stream_type):
    """Replace a deferred local ``cuda.CUstream`` annotation with its type.

    CuTe's Python 3.12 path resolves annotations with ``inspect`` against the
    defining module globals. The CUDA binding is deliberately imported only
    inside each generator's main function, so leaving the deferred string
    would make compilation fail with ``NameError: cuda``. Binding the concrete
    type preserves lazy imports and gives the compiler an inspectable callable.
    """
    current = function
    depth = 0
    seen: set[int] = set()
    while current is not None:
        require(id(current) not in seen,
                "generated launch __wrapped__ chain contains a cycle")
        seen.add(id(current))
        annotations = dict(getattr(current, "__annotations__", {}))
        require("stream" in annotations,
                f"generated launch wrapper depth {depth} lacks stream annotation")
        annotations["stream"] = stream_type
        current.__annotations__ = annotations
        require(inspect.get_annotations(current, eval_str=True).get("stream") is stream_type,
                f"generated launch stream annotation did not bind at depth {depth}")
        current = getattr(current, "__wrapped__", None)
        depth += 1
    require(inspect.signature(function, eval_str=True).parameters["stream"].annotation
            is stream_type,
            "generated launch signature did not expose the bound stream type")
    return function


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def artifact_metadata(
    task: Task,
    space: dict,
    generator_path: Path,
    dense_path: Path,
    patched_dense_path: Path,
    output_dir: Path,
    bridge_path: Path,
    cutlass_commit: str,
    extra: dict,
) -> dict:
    base = output_dir / task.artifact_stem
    header = base.with_suffix(".h")
    object_file = base.with_suffix(".o")
    require(header.is_file() and object_file.is_file() and bridge_path.is_file(),
            "CuTe export did not produce header/object/bridge")
    files = {}
    for label, path in (
        ("header", header), ("object", object_file), ("bridge", bridge_path),
    ):
        files[label] = {"path": path.name, "sha256": sha256_file(path)}
    return {
        "schema": 2,
        "kind": "katago-c384-exact-aot-artifact",
        "generation_complete": True,
        "verified_on_target": False,
        "compute_capability": "sm_120",
        "dtype": "fp16",
        "search_space_sha256": canonical_json_sha256(space),
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
        "files": files,
        "shape": {
            "sequence": 225, "channels": 384, "heads": 12,
            "kv_heads": 12, "head_dim": 32, "ffn_channels": 1024,
        },
        "provenance": {
            "generator_sha256": sha256_file(generator_path.resolve()),
            "cutlass_commit": cutlass_commit,
            "dense_gemm_sha256": sha256_file(dense_path),
            "patched_dense_gemm_sha256": sha256_file(patched_dense_path),
            "python": sys.version.split()[0],
            "nvidia_cutlass_dsl": package_version("nvidia-cutlass-dsl"),
            "cuda_python": package_version("cuda-python"),
            "nvcc": subprocess.run(
                ["nvcc", "--version"], check=True, text=True,
                capture_output=True,
            ).stdout.strip().splitlines()[-1],
            "gpu_kernel_executed": False,
        },
        "coordinate": extra,
    }


def write_metadata(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")

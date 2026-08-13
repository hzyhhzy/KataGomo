#!/usr/bin/env python3
"""CPU-only exact C384/H12/S225/D32 SM120 FlashAttention-4 generator.

The generator specializes from static DLPack descriptors and a fake stream;
it never allocates a CUDA tensor or selects a GPU. Search artifacts receive a
native unique C ABI. Production is regenerated with the stable
``c384fa4win`` ABI instead of rewriting an object file.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import sys


SEQUENCE = 225
HEADS = 12
HEAD_DIM = 32
LAYOUT_IDS = {"planar": 1, "packed-token": 2}
ACCUMULATION_IDS = {"fp32": 1, "qk16": 2, "pv16": 3, "both16": 4}
_KEEPALIVE: list[object] = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--tile-m", type=int, choices=(64, 128), default=128)
    parser.add_argument("--tile-n", type=int, choices=(64, 96, 128), required=True)
    parser.add_argument("--num-stages", type=int, choices=(1, 2), default=1)
    parser.add_argument("--num-threads", type=int, choices=(128, 256), default=128)
    parser.add_argument(
        "--accumulation", choices=tuple(ACCUMULATION_IDS), required=True,
        help="fp32, qk16, pv16, or both16; every non-fp32 result needs correctness qualification",
    )
    parser.add_argument("--layout", choices=tuple(LAYOUT_IDS), default="planar")
    parser.add_argument("--abi-mode", choices=("search", "production"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.batch <= 128:
        parser.error("--batch must be in 1..128")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_id(args: argparse.Namespace) -> str:
    warps = args.num_threads // 32
    return (
        f"tm{args.tile_m}-tn{args.tile_n}-s{args.num_stages}-w{warps}-"
        f"{args.accumulation}-{args.layout}-b{args.batch}-"
        f"s{SEQUENCE}-h{HEADS}-d{HEAD_DIM}"
    )


def abi(args: argparse.Namespace) -> tuple[str, str]:
    if args.abi_mode == "production":
        return "c384_fa4_winner", "c384fa4win"
    warps = args.num_threads // 32
    layout = "p" if args.layout == "planar" else "t"
    suffix = (
        f"b{args.batch}m{args.tile_m}n{args.tile_n}s{args.num_stages}"
        f"w{warps}{args.accumulation}{layout}"
    )
    return f"c384_fa4_{suffix}", f"c384fa4{suffix}"


class DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8), ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p), ("device", DLDevice),
        ("ndim", ctypes.c_int), ("dtype", DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


DLDeleter = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


class DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", DLTensor), ("manager_ctx", ctypes.c_void_p),
        ("deleter", DLDeleter),
    ]


CAPSULE_NEW = ctypes.pythonapi.PyCapsule_New
CAPSULE_NEW.restype = ctypes.py_object
CAPSULE_NEW.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]


class StaticCudaTensor:
    def __init__(self, shape: tuple[int, ...], strides: tuple[int, ...], address: int):
        self.shape = (ctypes.c_int64 * len(shape))(*shape)
        self.strides = (ctypes.c_int64 * len(strides))(*strides)
        self.managed = DLManagedTensor(
            DLTensor(
                ctypes.c_void_p(address), DLDevice(2, 0), len(shape),
                DLDataType(2, 16, 1), self.shape, self.strides, 0,
            ),
            None, DLDeleter(),
        )

    def __dlpack__(self, stream=None):
        del stream
        return CAPSULE_NEW(ctypes.addressof(self.managed), b"dltensor", None)

    def __dlpack_device__(self):
        return (2, 0)


def patch_aux_header(CuteCHeaderGenerator, AuxData) -> None:
    original = CuteCHeaderGenerator._generate_arguments

    def generate(instance, symbol_prefix, args_spec, args, kwargs):
        rectified = args_spec.get_rectified_args(args, kwargs)

        class Spec:
            pass

        spec = Spec()
        spec.signature = args_spec.signature
        spec.get_rectified_args = lambda _args, _kwargs: [
            None if isinstance(value, AuxData) else value for value in rectified
        ]
        return original(instance, symbol_prefix, spec, args, kwargs)

    CuteCHeaderGenerator._generate_arguments = generate


def render_bridge(
    stem: str, prefix: str, identity: str, args: argparse.Namespace
) -> str:
    layout_id = LAYOUT_IDS[args.layout]
    accumulation_id = ACCUMULATION_IDS[args.accumulation]
    warps = args.num_threads // 32
    return f'''#include "{stem}.h"

#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <mutex>

namespace {{
constexpr int MaxPreparedDevices = 32;
std::array<{prefix}_Kernel_Module_t,MaxPreparedDevices> modules{{}};
std::array<std::atomic<bool>,MaxPreparedDevices> prepared{{}};
std::array<cudaError_t,MaxPreparedDevices> preparationFailures{{}};
std::array<std::mutex,MaxPreparedDevices> prepareMutexes;

cudaError_t validateDeviceOrdinal(int deviceOrdinal) {{
  int deviceCount = 0;
  cudaError_t status = cudaGetDeviceCount(&deviceCount);
  if(status != cudaSuccess)
    return status;
  return deviceOrdinal >= 0 && deviceOrdinal < deviceCount &&
      deviceOrdinal < MaxPreparedDevices ? cudaSuccess : cudaErrorInvalidDevice;
}}

cudaError_t loadModuleForDevice(int deviceOrdinal) {{
  // The generated void helper logs failures and loads every visible device.
  // Use the raw CuTe ABI so prepare owns one ordinal and propagates its status.
  auto& module = modules[deviceOrdinal];
  module = {{}};
  cudaLibrary_t* library = &module.module;
  cudaError_t rawStatus = cudaSuccess;
  struct InitArgs {{
    cudaLibrary_t** library;
    cudaError_t* status;
  }} initArgs{{&library,&rawStatus}};
  _mlir_{prefix}_cuda_init(reinterpret_cast<void**>(&initArgs));
  if(rawStatus != cudaSuccess) {{
    if(module.module != nullptr)
      (void)cudaLibraryUnload(module.module);
    module = {{}};
    return rawStatus;
  }}

  int32_t deviceId = deviceOrdinal;
  struct LoadArgs {{
    cudaLibrary_t** library;
    int32_t* deviceId;
    cudaError_t* status;
  }} loadArgs{{&library,&deviceId,&rawStatus}};
  _mlir_{prefix}_cuda_load_to_device(reinterpret_cast<void**>(&loadArgs));
  if(rawStatus != cudaSuccess) {{
    (void)cudaLibraryUnload(module.module);
    module = {{}};
  }}
  return rawStatus;
}}
}}

extern "C" int {prefix}_batch() {{ return {args.batch}; }}
extern "C" int {prefix}_sequence() {{ return {SEQUENCE}; }}
extern "C" int {prefix}_heads() {{ return {HEADS}; }}
extern "C" int {prefix}_head_dim() {{ return {HEAD_DIM}; }}
extern "C" int {prefix}_tile_m() {{ return {args.tile_m}; }}
extern "C" int {prefix}_tile_n() {{ return {args.tile_n}; }}
extern "C" int {prefix}_num_stages() {{ return {args.num_stages}; }}
extern "C" int {prefix}_num_warps() {{ return {warps}; }}
extern "C" int {prefix}_input_layout() {{ return {layout_id}; }}
extern "C" int {prefix}_accumulation() {{ return {accumulation_id}; }}
extern "C" const char* {prefix}_id() {{ return {json.dumps(identity)}; }}

extern "C" cudaError_t {prefix}_prepare(int deviceOrdinal) {{
  cudaError_t status = validateDeviceOrdinal(deviceOrdinal);
  if(status != cudaSuccess)
    return status;
  if(prepared[deviceOrdinal].load(std::memory_order_acquire))
    return cudaSuccess;
  std::lock_guard<std::mutex> lock(prepareMutexes[deviceOrdinal]);
  if(prepared[deviceOrdinal].load(std::memory_order_relaxed))
    return cudaSuccess;
  if(preparationFailures[deviceOrdinal] != cudaSuccess)
    return preparationFailures[deviceOrdinal];
  status = loadModuleForDevice(deviceOrdinal);
  if(status == cudaSuccess)
    prepared[deviceOrdinal].store(true,std::memory_order_release);
  else
    preparationFailures[deviceOrdinal] = status;
  return status;
}}

extern "C" cudaError_t {prefix}_launch(
  void* q, void* k, void* v, void* output,
  int batch, int sequence, int heads, int headDim, float softmaxScale,
  uint32_t inputLayout, int deviceOrdinal, cudaStream_t stream
) {{
  if(batch != {args.batch} || sequence != {SEQUENCE} || heads != {HEADS} ||
     headDim != {HEAD_DIM} || inputLayout != {layout_id} ||
     q == nullptr || k == nullptr || v == nullptr || output == nullptr)
    return cudaErrorInvalidValue;
  if(deviceOrdinal < 0 || deviceOrdinal >= MaxPreparedDevices ||
     !prepared[deviceOrdinal].load(std::memory_order_acquire))
    return cudaErrorNotReady;
  {prefix}_Tensor_mQ_t tensorQ = {{q}};
  {prefix}_Tensor_mK_t tensorK = {{k}};
  {prefix}_Tensor_mV_t tensorV = {{v}};
  {prefix}_Tensor_mO_t tensorOutput = {{output}};
  const int32_t result = cute_dsl_{prefix}_wrapper(
    &modules[deviceOrdinal],&tensorQ,&tensorK,&tensorV,&tensorOutput,
    softmaxScale,stream);
  return result == 0 ? cudaPeekAtLastError() : cudaErrorUnknown;
}}
'''


def generate(args: argparse.Namespace) -> Path:
    # Heavy imports occur only after argument validation, keeping offline
    # contract tests and --help independent of the pinned FA4 environment.
    os.environ["FLASH_ATTENTION_ARCH"] = "sm_120"
    os.environ["CUTE_DSL_ARCH"] = "sm_120"
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    os.environ.setdefault("CUDA_TOOLKIT_PATH", cuda_home)
    os.environ.setdefault("CUTE_DSL_PTXAS_PATH", os.path.join(cuda_home, "bin", "ptxas"))
    os.environ.setdefault("CUTE_DSL_KEEP_PTX", "1")
    os.environ.setdefault("CUTE_DSL_KEEP_CUBIN", "1")
    os.environ.setdefault("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "0")

    import cutlass
    import cutlass.cute as cute
    from cutlass import Float16, Float32
    from cutlass.cute.export.c_header_generator import CuteCHeaderGenerator
    from cutlass.cute.runtime import from_dlpack
    from flash_attn.cute.flash_fwd_sm120 import FlashAttentionForwardSm120
    from flash_attn.cute.softmax import Softmax
    from flash_attn.cute.utils import AuxData
    from quack import layout_utils

    patch_aux_header(CuteCHeaderGenerator, AuxData)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("CUTE_DSL_DUMP_DIR", str(output))
    os.environ.setdefault("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR", str(output / ".aot-cache"))

    shape = (args.batch, SEQUENCE, HEADS, HEAD_DIM)
    row_stride = HEADS * HEAD_DIM
    if args.layout == "packed-token":
        row_stride *= 3
    strides = (SEQUENCE * row_stride, row_stride, HEAD_DIM, 1)
    base = 0x10000
    addresses = (
        (base, base + HEADS * HEAD_DIM * 2, base + 2 * HEADS * HEAD_DIM * 2)
        if args.layout == "packed-token" else (0x10000, 0x20000, 0x30000)
    )

    def tensor(address: int, tensor_strides: tuple[int, ...]):
        spec = StaticCudaTensor(shape, tensor_strides, address)
        _KEEPALIVE.append(spec)
        return from_dlpack(spec, assumed_align=16, enable_tvm_ffi=False)

    q = tensor(addresses[0], strides)
    k = tensor(addresses[1], strides)
    v = tensor(addresses[2], strides)
    out_strides = (SEQUENCE * HEADS * HEAD_DIM, HEADS * HEAD_DIM, HEAD_DIM, 1)
    out = tensor(0x40000, out_strides)

    qk_dtype = Float16 if args.accumulation in ("qk16", "both16") else Float32
    pv_dtype = Float16 if args.accumulation in ("pv16", "both16") else Float32
    if pv_dtype is Float16:
        @cute.jit
        def rescale_with_accumulator_cast(
            self, acc_o: cute.Tensor, row_scale: cute.Tensor
        ) -> None:
            acc_o_mn = layout_utils.reshape_acc_to_mn(acc_o)
            assert cute.size(row_scale) == cute.size(acc_o_mn, mode=[0])
            for row in cutlass.range(cute.size(row_scale), unroll_full=True):
                scaled = acc_o_mn[row, None].load() * row_scale[row]
                acc_o_mn[row, None].store(scaled.to(acc_o_mn.element_type))

        Softmax.rescale_O = rescale_with_accumulator_cast

    fa = FlashAttentionForwardSm120(
        Float16, HEAD_DIM, HEAD_DIM, 1,
        is_causal=False, is_local=False, pack_gqa=False,
        tile_m=args.tile_m, tile_n=args.tile_n,
        num_stages=args.num_stages, num_threads=args.num_threads,
        Q_in_regs=False, score_mod=None, mask_mod=None,
        has_aux_tensors=False, qk_acc_dtype=qk_dtype, pv_acc_dtype=pv_dtype,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False)
    scale = 1.0 / (HEAD_DIM ** 0.5)
    compiled = cute.compile(
        fa,q,k,v,out,None,scale,None,None,None,None,None,None,None,None,None,
        AuxData(),None,None,stream,
    )
    stem, prefix = abi(args)
    identity = candidate_id(args)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", prefix):
        raise RuntimeError(f"invalid generated C prefix: {prefix}")
    compiled.export_to_c(str(output), stem, prefix)
    bridge = output / f"{stem}_bridge.cpp"
    bridge.write_text(render_bridge(stem, prefix, identity, args), encoding="utf-8")
    metadata = {
        "schema": 1,
        "abi_mode": args.abi_mode,
        "artifact_stem": stem,
        "symbol_prefix": prefix,
        "candidate_id": identity,
        "batch": args.batch,
        "fixed_shape": {"sequence": SEQUENCE, "heads": HEADS, "head_dim": HEAD_DIM},
        "tile": {
            "m": args.tile_m, "n": args.tile_n,
            "num_stages": args.num_stages, "num_threads": args.num_threads,
            "num_warps": args.num_threads // 32,
        },
        "layout": args.layout,
        "layout_id": LAYOUT_IDS[args.layout],
        "accumulation": args.accumulation,
        "accumulation_id": ACCUMULATION_IDS[args.accumulation],
        "dtype": "fp16",
        "mask": False,
        "compute_capability": "sm_120",
        "gpu_used_for_generation": False,
        "generator": {
            "python": sys.version.split()[0],
            "cutlass_cuda": str(cutlass.CUDA_VERSION),
            "generator_sha256": sha256(Path(__file__).resolve()),
            # render_bridge intentionally lives in this audited source file.
            "bridge_generator_sha256": sha256(Path(__file__).resolve()),
        },
        "sha256": {},
    }
    for label, path in (
        ("header", output / f"{stem}.h"),
        ("object", output / f"{stem}.o"),
        ("bridge", bridge),
    ):
        metadata["sha256"][label] = sha256(path)
    metadata_path = output / f"{stem}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def main() -> int:
    args = parse_args()
    print(generate(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

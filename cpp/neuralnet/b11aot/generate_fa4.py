#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# AOT export/softmax setup follows doomoooo/KataGomo_fork; see NOTICE.md.
"""CPU-only SM120 FA4: S361/H12/D32, FP16 QK/PV, dynamic or fixed packed layout."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def file_identity(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
            "canonical_lf_sha256": hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("exact", "mask"), required=True)
    parser.add_argument("--dependency-root", required=True, type=Path,
                        help="Isolated overlay containing the pinned patched flash_attn package")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="New/empty explicit artifact directory outside b11aot")
    parser.add_argument("--cuda-root", type=Path,
                        default=Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")))
    parser.add_argument("--tile-m", type=int, default=128)
    parser.add_argument("--tile-n", type=int, default=96)
    parser.add_argument("--num-stages", type=int, default=1)
    parser.add_argument("--probe-batch", type=int, default=16,
                        help="Metadata specialization sample; batch remains runtime")
    parser.add_argument("--max-batch", type=int, default=96)
    parser.add_argument("--fixed-batch", type=int, choices=(13, 16),
                        help="Schema 2: fixed metadata and packed-QKV strides, not runtime B")
    parser.add_argument("--symbol-prefix", help="Unique C identifier; defaults to the mode/tile name")
    parser.add_argument("--allow-toolchain-mismatch", action="store_true",
                        help="Explicit migration experiment only; source hashes still must match")
    args = parser.parse_args()
    if min(args.tile_m, args.tile_n, args.num_stages, args.probe_batch, args.max_batch) < 1:
        parser.error("tile, stage and batch parameters must be positive")
    if args.probe_batch > args.max_batch:
        parser.error("probe batch exceeds the runtime guard")
    if args.fixed_batch is not None:
        if (args.tile_m, args.tile_n, args.num_stages) != (128, 96, 1):
            parser.error("static production profiles require M128/N96/S1")
        args.probe_batch = args.max_batch = args.fixed_batch
    fixed_prefix = f"static_b{args.fixed_batch}_" if args.fixed_batch is not None else ""
    args.symbol_prefix = args.symbol_prefix or (
        f"b11_fa4_{fixed_prefix}{args.mode}_m{args.tile_m}_n{args.tile_n}_s{args.num_stages}")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.symbol_prefix):
        parser.error("symbol prefix must be a C identifier")
    args.output_dir = args.output_dir.resolve()
    args.dependency_root = args.dependency_root.resolve()
    args.cuda_root = args.cuda_root.resolve()
    if args.output_dir == HERE or HERE in args.output_dir.parents or args.output_dir == HERE.parents[2]:
        parser.error("output must not be the repository root or inside b11aot source")
    if args.output_dir == args.dependency_root or args.dependency_root in args.output_dir.parents:
        parser.error("output must be separate from the dependency overlay")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory must be new or empty; refusing to overwrite artifacts")
    for tool in ("nvcc", "ptxas"):
        if not (args.cuda_root / "bin" / tool).is_file():
            parser.error("CUDA toolkit executable not found: " + tool)
    return args


def configure_environment(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = args.output_dir / ".tmp"
    temporary.mkdir()
    # Set these before importing torch, CUDA bindings, CuTe or FA4. No GPU query
    # is needed: SM120 is an explicit compilation target, not a detected device.
    os.environ.update({
        "CUDA_VISIBLE_DEVICES": "",
        "CUDA_HOME": str(args.cuda_root),
        "CUDA_TOOLKIT_PATH": str(args.cuda_root),
        "FLASH_ATTENTION_ARCH": "sm_120",
        "CUTE_DSL_ARCH": "sm_120",
        "CUTE_DSL_PTXAS_PATH": str(args.cuda_root / "bin/ptxas"),
        "CUTE_DSL_KEEP_PTX": "1",
        "CUTE_DSL_KEEP_CUBIN": "1",
        "CUTE_DSL_DUMP_DIR": str(args.output_dir),
        "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "0",
        "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": str(args.output_dir / ".aot-cache"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(temporary),
    })
    os.environ["PATH"] = str(args.cuda_root / "bin") + os.pathsep + os.environ["PATH"]
    sys.path.insert(0, str(args.dependency_root))
    os.chdir(args.output_dir)


def verify_dependencies(args):
    lock = json.loads((HERE / "dependencies.json").read_text())
    versions = {}
    mismatches = []
    python_version = sys.version.split()[0]
    if python_version != lock["python"]:
        mismatches.append(f"Python: expected {lock['python']}, found {python_version}")
    nvcc_output = subprocess.run([str(args.cuda_root / "bin/nvcc"),"--version"],
                                check=True,text=True,capture_output=True).stdout
    nvcc_match = re.search(r"V(\d+\.\d+\.\d+)",nvcc_output)
    nvcc_version = nvcc_match.group(1) if nvcc_match else "unknown"
    if nvcc_version != lock["cuda_toolkit"]:
        mismatches.append(f"CUDA toolkit: expected {lock['cuda_toolkit']}, found {nvcc_version}")
    for package, expected in lock["packages"].items():
        actual = importlib.metadata.version(package)
        versions[package] = actual
        if actual != expected:
            mismatches.append(f"{package}: expected {expected}, found {actual}")
    if mismatches and not args.allow_toolchain_mismatch:
        raise RuntimeError("toolchain mismatch: " + "; ".join(mismatches))
    sources = {}
    for relative, expected in lock["sources"].items():
        identity = file_identity(args.dependency_root / relative)
        if identity["canonical_lf_sha256"] != expected["canonical_lf_sha256"]:
            raise RuntimeError("dependency source mismatch: " + relative)
        sources[relative] = identity
    return {"lock": file_identity(HERE / "dependencies.json"),
            "python": python_version, "cuda_toolkit": nvcc_version,
            "packages": versions, "sources": sources, "toolchain_mismatches": mismatches}


def render_bridge(args):
    p = args.symbol_prefix
    masked = args.mode == "mask"
    mask_guard = "!mask" if masked else "mask != nullptr"
    mask_tensor = f"  {p}_Tensor_mMask_t tm = {{mask,{{batch,seq}},{{seq}}}};\n" if masked else ""
    mask_arg = "&tm," if masked else ""
    fixed_batch = getattr(args, "fixed_batch", None)
    batch_guard = f"batch!={fixed_batch} || !packedQKV" if fixed_batch is not None else f"batch<1 || batch>{args.max_batch}"
    layout_comment = (f"Fixed B{fixed_batch}, packed-QKV stride 1152; output stride 384."
                      if fixed_batch is not None else f"Batch is runtime, 1..{args.max_batch}; planar or packed QKV.")
    tensors = f'''  constexpr int seq=361,heads=12,dim=32;
  const int64_t stride=(packedQKV?3:1)*heads*dim;
  {p}_Tensor_mQ_t tq={{q,{{batch,seq,heads,dim}},{{seq*stride,stride,dim}}}};
  {p}_Tensor_mK_t tk={{k,{{batch,seq,heads,dim}},{{seq*stride,stride,dim}}}};
  {p}_Tensor_mV_t tv={{v,{{batch,seq,heads,dim}},{{seq*stride,stride,dim}}}};
  {p}_Tensor_mO_t to={{output,{{batch,seq,heads,dim}},{{seq*heads*dim,heads*dim,dim}}}};
{mask_tensor}'''
    if fixed_batch is not None:
        tensors = "".join(f"  {p}_Tensor_m{name}_t t{name.lower()}={{{value}}};\n"
                          for name,value in (("Q","q"),("K","k"),("V","v"),("O","output")))
        if masked:
            tensors += f"  {p}_Tensor_mMask_t tm={{mask}};\n"
    api = f'''#pragma once
#include <cuda_runtime_api.h>
// prepare on the current eligible device before warmup/timing/graph capture.
extern "C" cudaError_t {p}_prepare();
// {args.mode}: mask must be {"nonnull FP16 [B,361]" if masked else "nullptr"}.
// Every tensor must be 16-byte aligned. {layout_comment}
extern "C" cudaError_t {p}_launch(void* q,void* k,void* v,void* output,
  void* mask,int batch,float scale,bool packedQKV,cudaStream_t stream);
'''
    bridge = f'''#include "{p}.h"
#include "{p}_api.h"
#include "aot_loader_cuda.h"
#include <cstdint>
namespace {{
b11aot::CurrentDeviceLoader<{p}_Kernel_Module_t,b11aot::Rtx5090Runtime>
  loader(_mlir_{p}_cuda_init,_mlir_{p}_cuda_load_to_device);
}}
extern "C" cudaError_t {p}_prepare() {{ return loader.prepareCurrentDevice().error; }}
extern "C" cudaError_t {p}_launch(void* q,void* k,void* v,void* output,
  void* mask,int batch,float scale,bool packedQKV,cudaStream_t stream) {{
  if({batch_guard} || !q || !k || !v || !output || {mask_guard})
    return cudaErrorInvalidValue;
  auto alignment=reinterpret_cast<std::uintptr_t>(q)|reinterpret_cast<std::uintptr_t>(k)|
    reinterpret_cast<std::uintptr_t>(v)|reinterpret_cast<std::uintptr_t>(output)|
    reinterpret_cast<std::uintptr_t>(mask);
  if(alignment & 15) return cudaErrorInvalidValue;
{tensors}  int32_t status=cute_dsl_{p}_wrapper(loader.module(),&tq,&tk,&tv,&to,{mask_arg}scale,stream);
  return status==0?cudaPeekAtLastError():cudaErrorUnknown;
}}
'''
    return api, bridge


def compile_artifact(args, dependencies):
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float16, Float32
    from cuda.bindings import driver as cuda
    from flash_attn.cute.flash_fwd_sm120 import FlashAttentionForwardSm120
    from flash_attn.cute import utils
    from flash_attn.cute.utils import AuxData
    from flash_attn.cute.softmax import Softmax
    from cutlass.cute.export.c_header_generator import CuteCHeaderGenerator
    from quack import layout_utils
    from fake_tensors import dynamic_fp16_tensor, static_fp16_tensor

    # Confirm Python imported the overlay we checked, not an unrelated cache.
    for relative, identity in dependencies["sources"].items():
        module_name = relative[:-3].replace("/", ".")
        module = sys.modules.get(module_name)
        if module is None or Path(module.__file__).resolve() != Path(identity["path"]):
            raise RuntimeError("unexpected imported implementation: " + module_name)

    # AuxData is compile-time-only. The C exporter must not invent an ABI field
    # for it; masked mode exposes its real tensor explicitly through a wrapper.
    original_arguments = CuteCHeaderGenerator._generate_arguments
    def export_arguments(self, symbol_prefix, args_spec, positional, keywords):
        rectified = args_spec.get_rectified_args(positional, keywords)
        class Spec:
            signature = args_spec.signature
            def get_rectified_args(self, unused_a, unused_kw):
                return [None if isinstance(v, AuxData) else v for v in rectified]
        return original_arguments(self, symbol_prefix, Spec(), positional, keywords)
    CuteCHeaderGenerator._generate_arguments = export_arguments

    # Same FP16 online-softmax rescale rounding as the validated both16 control.
    @cute.jit
    def rescale_output(self, acc_o: cute.Tensor, row_scale: cute.Tensor):
        acc_o_mn = layout_utils.reshape_acc_to_mn(acc_o)
        assert cute.size(row_scale) == cute.size(acc_o_mn, mode=[0])
        for row in cutlass.range(cute.size(row_scale), unroll_full=True):
            scaled = acc_o_mn[row, None].load() * row_scale[row]
            acc_o_mn[row, None].store(scaled.to(acc_o_mn.element_type))
    Softmax.rescale_O = rescale_output

    @cute.jit
    def board_mask(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        b = cutlass.Int32(utils.ssa_to_scalar(batch_idx))
        k = cutlass.Int32(utils.ssa_to_scalar(kv_idx))
        live = aux_tensors[0][b, k] != Float16(0.0)
        return utils.scalar_to_ssa(live, cutlass.Boolean)

    if not FlashAttentionForwardSm120.can_implement(
        Float16,32,32,args.tile_m,args.tile_n,args.num_stages,128,False,False):
        raise ValueError("FA4 rejects this tile/stage configuration on SM120")
    masked = args.mode == "mask"
    forward = FlashAttentionForwardSm120(
        Float16,32,32,1,is_causal=False,is_local=False,pack_gqa=False,
        tile_m=args.tile_m,tile_n=args.tile_n,num_stages=args.num_stages,
        num_threads=128,Q_in_regs=False,score_mod=None,
        mask_mod=board_mask if masked else None,has_aux_tensors=masked,
        qk_acc_dtype=Float16,pv_acc_dtype=Float16)
    batch, seq, heads, dim = args.probe_batch, 361, 12, 32
    fixed = getattr(args, "fixed_batch", None) is not None
    make_tensor = static_fp16_tensor if fixed else dynamic_fp16_tensor
    tensors = []
    for address in (0x10000,0x20000,0x30000,0x40000):
        stride = heads*dim*(3 if fixed and address != 0x40000 else 1)
        tensors.append(make_tensor((batch,seq,heads,dim),(seq*stride,stride,dim,1),address,3))
    q,k,v,o = tensors
    scale = 1.0/(dim**0.5)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False)
    if masked:
        mask = make_tensor((batch,seq),(seq,1),0x50000,1)
        @cute.jit
        def masked_forward(mQ,mK,mV,mO,mMask,softmax_scale:Float32,stream:cuda.CUstream):
            forward(mQ,mK,mV,mO,None,softmax_scale,
                None,None,None,None,None,None,None,None,
                None,AuxData(tensors=(mMask,)),None,None,stream)
        compiled = cute.compile(masked_forward,q,k,v,o,mask,scale,stream)
    else:
        compiled = cute.compile(forward,q,k,v,o,None,scale,
            None,None,None,None,None,None,None,None,
            None,AuxData(),None,None,stream)
    compiled.export_to_c(str(args.output_dir),args.symbol_prefix,args.symbol_prefix)


def verify_static_abi(header, prefix, masked):
    """Reject accidentally dynamic metadata before producing a schema-2 artifact."""
    structs = {name: body for body,name in re.findall(
        r"typedef\s+struct\s*\{([^{}]*)\}\s*(\w+)\s*;", header, re.S)}
    for name in (("mQ","mK","mV","mO","mMask") if masked else ("mQ","mK","mV","mO")):
        body = structs.get(prefix + "_Tensor_" + name + "_t", "")
        if not re.fullmatch(r"\s*void\s*\*\s*data\s*;\s*", body):
            raise ValueError("static tensor ABI must be data-only: " + name)
    if "float softmax_scale" not in header or "cudaStream_t stream" not in header:
        raise ValueError("static scale/stream ABI mismatch")


def main():
    args = parse_args()
    configure_environment(args)
    dependencies = verify_dependencies(args)
    compile_artifact(args, dependencies)
    api, bridge = render_bridge(args)
    p = args.symbol_prefix
    if args.fixed_batch is not None:
        verify_static_abi((args.output_dir / (p + ".h")).read_text(),p,args.mode=="mask")
    (args.output_dir / f"{p}_api.h").write_text(api)
    (args.output_dir / f"{p}_bridge.cpp").write_text(bridge)
    nvcc = subprocess.run([str(args.cuda_root / "bin/nvcc"),"--version"],
                          check=True,text=True,capture_output=True).stdout
    metadata = {
        "schema":2 if args.fixed_batch is not None else 1,"mode":args.mode,"symbol_prefix":p,
        "prepare_symbol":p+"_prepare","launch_symbol":p+"_launch",
        "fixed_shape":{"sequence":361,"heads":12,"head_dim":32},
        "runtime_batch":{"minimum":args.fixed_batch or 1,"maximum":args.max_batch,"probe":args.probe_batch},
        "tile":{"m":args.tile_m,"n":args.tile_n,"stages":args.num_stages,"threads":128},
        "io":"fp16","accumulation":{"qk":"fp16","pv":"fp16"},
        "mask_semantics":"key mask via auxiliary tensor" if args.mode=="mask" else "exact, no mask",
        "generation":{"GPU_used":False,"CUDA_VISIBLE_DEVICES":os.environ["CUDA_VISIBLE_DEVICES"],
                      "tensor_storage":"inert DLPack","stream":"fake stream"},
        "python":sys.version,"nvcc_version":nvcc.strip(),"dependencies":dependencies,
        "generator":file_identity(__file__),"fake_tensors":file_identity(HERE/"fake_tensors.py"),
        "loader_core":file_identity(HERE/"aot_loader_core.h"),
        "loader_cuda":file_identity(HERE/"aot_loader_cuda.h"),
        "artifacts":{suffix:file_identity(args.output_dir/(p+suffix))
                     for suffix in (".o",".h","_api.h","_bridge.cpp")},
        "device_artifacts":[file_identity(path) for suffix in ("*.ptx","*.cubin")
                            for path in sorted(args.output_dir.glob(suffix))],
    }
    if args.fixed_batch is not None:
        metadata.update(fixed_batch=args.fixed_batch,target="sm120",
            tensor_abi="pointer_only",qkv_layout="packed",qkv_stride=1152,output_stride=384)
    (args.output_dir / f"{p}.json").write_text(json.dumps(metadata,indent=2)+"\n")
    shutil.copytree(HERE/"licenses",args.output_dir/"licenses")
    shutil.copyfile(HERE/"NOTICE.md",args.output_dir/"NOTICE.md")
    print(args.output_dir/f"{p}.json")


if __name__ == "__main__":
    main()

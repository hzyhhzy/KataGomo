#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CPU-only bundle failure-injection tests and optional fake-runtime C++ bridge test.

All fixtures/build products live below the explicitly supplied --output-dir.
No real CUDA headers, library, context, allocations or kernels are used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import unittest
import uuid
sys.dont_write_bytecode = True
from generate_fa4 import render_bridge, verify_static_abi
from generate_static_bundle import write_bundle

HERE = Path(__file__).resolve().parent
OPTIONS = None


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_header(prefix, mode, static=True):
    names = ("mQ","mK","mV","mO","mMask") if mode == "mask" else ("mQ","mK","mV","mO")
    structs = "".join(f"typedef struct {{ void *data;{' int shape[4];' if not static else ''} }} {prefix}_Tensor_{name}_t;\n" for name in names)
    args = ",".join(f"{prefix}_Tensor_{name}_t*" for name in names)
    return f'''#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>
typedef struct {{ void* unused; }} {prefix}_Kernel_Module_t;
{structs}
extern "C" void _mlir_{prefix}_cuda_init(void**);
extern "C" void _mlir_{prefix}_cuda_load_to_device(void**);
extern "C" int32_t cute_dsl_{prefix}_wrapper({prefix}_Kernel_Module_t*,{args},float softmax_scale,cudaStream_t stream);
'''


def make_fixture(root, static=True):
    for batch in ((13,16) if static else (0,)):
        for mode in ("exact","mask"):
            prefix = f"b11_fa4_{'static_b'+str(batch)+'_' if static else ''}{mode}_m128_n96_s1"
            directory = root / (f"b{batch}" if static else "") / mode
            directory.mkdir(parents=True)
            args = SimpleNamespace(symbol_prefix=prefix,mode=mode,max_batch=batch or 96,fixed_batch=batch or None)
            api,bridge = render_bridge(args)
            # A minimal ELF64 identity prefix is enough for the non-linking fixture.
            elf = bytearray(64)
            elf[:6] = b"\x7fELF\x02\x01"
            elf[18:20] = b"\x3e\x00"
            (directory/(prefix+".o")).write_bytes(elf)
            (directory/(prefix+".h")).write_text(fixture_header(prefix,mode))
            (directory/(prefix+"_api.h")).write_text(api)
            (directory/(prefix+"_bridge.cpp")).write_text(bridge)
            item = dict(schema=2 if static else 1,mode=mode,symbol_prefix=prefix,
                prepare_symbol=prefix+"_prepare",launch_symbol=prefix+"_launch",
                fixed_shape=dict(sequence=361,heads=12,head_dim=32),
                runtime_batch=dict(minimum=batch or 1,maximum=batch or 96,probe=batch or 16),
                tile=dict(m=128,n=96,stages=1,threads=128),io="fp16",accumulation=dict(qk="fp16",pv="fp16"))
            if static:
                item.update(fixed_batch=batch,target="sm120",tensor_abi="pointer_only",qkv_layout="packed",qkv_stride=1152,output_stride=384)
            item["artifacts"] = {suffix:{"sha256":digest(directory/(prefix+suffix))} for suffix in (".o",".h","_api.h","_bridge.cpp")}
            (directory/(prefix+".json")).write_text(json.dumps(item))
    if static:
        write_bundle(root)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        # Retain isolated fixtures for diagnostics. Do not use mode-0700 temp dirs:
        # their Windows ACL can exclude a sandbox's delegated process identity.
        self.directory = OPTIONS.output_dir/("b11-aot-"+uuid.uuid4().hex[:12])
        self.directory.mkdir()
        self.root = self.directory/"bundle"
        make_fixture(self.root)

    def run_cmake(self, root=None, ok=True, static=True):
        root = root or self.root
        script = self.directory/"validate.cmake"
        script.write_text(f'''cmake_minimum_required(VERSION 3.19)
include("{(HERE/'artifacts.cmake').as_posix()}")
b11_fa4_collect_artifacts("{root.as_posix()}" sources dirs is_static)
list(LENGTH sources count)
if(NOT count EQUAL {8 if static else 4} OR NOT "${{is_static}}" STREQUAL "{'TRUE' if static else 'FALSE'}")
  message(FATAL_ERROR "unexpected collection result")
endif()
''')
        result = subprocess.run([OPTIONS.cmake,"-P",str(script)],capture_output=True,text=True)
        self.assertEqual(result.returncode == 0,ok,result.stdout+result.stderr)

    def modify_item(self, edit, refresh_artifacts=False):
        prefix = "b11_fa4_static_b13_exact_m128_n96_s1"
        path = self.root/"b13/exact"/(prefix+".json")
        item = json.loads(path.read_text())
        edit(item,path.parent,prefix)
        if refresh_artifacts:
            for suffix in item["artifacts"]:
                item["artifacts"][suffix]["sha256"] = digest(path.parent/(prefix+suffix))
        path.write_text(json.dumps(item))
        bundle_path = self.root/"bundle.json"
        bundle = json.loads(bundle_path.read_text())
        bundle["entries"][0]["manifest_sha256"] = digest(path)
        bundle_path.write_text(json.dumps(bundle))

    def test_valid_static_and_relocation(self):
        self.run_cmake()
        relocated = self.directory/"relocated"
        shutil.copytree(self.root,relocated)
        self.run_cmake(relocated)

    def test_valid_dynamic_compatibility(self):
        dynamic = self.directory/"dynamic"
        make_fixture(dynamic,False)
        self.run_cmake(dynamic,static=False)

    def test_tampered_object(self):
        path = next((self.root/"b13/exact").glob("*.o"))
        path.write_bytes(path.read_bytes()+b"tamper")
        self.run_cmake(ok=False)

    def test_missing_object(self):
        next((self.root/"b16/mask").glob("*.o")).unlink()
        self.run_cmake(ok=False)

    def test_missing_entry(self):
        path = self.root/"bundle.json"
        item = json.loads(path.read_text())
        item["entries"].pop()
        path.write_text(json.dumps(item))
        self.run_cmake(ok=False)

    def test_wrong_batch(self):
        self.modify_item(lambda item,unused,prefix:item.update(fixed_batch=12))
        self.run_cmake(ok=False)

    def test_wrong_packedness(self):
        self.modify_item(lambda item,unused,prefix:item.update(qkv_layout="planar",qkv_stride=384))
        self.run_cmake(ok=False)

    def test_wrong_runtime_bounds(self):
        self.modify_item(lambda item,unused,prefix:item.update(runtime_batch=dict(minimum=1,maximum=96,probe=13)))
        self.run_cmake(ok=False)

    def test_wrong_abi_even_with_new_hashes(self):
        def change(item,directory,prefix):
            (directory/(prefix+".h")).write_text(fixture_header(prefix,"exact",False))
        self.modify_item(change,True)
        self.run_cmake(ok=False)

    def test_wrong_elf_even_with_new_hashes(self):
        self.modify_item(lambda item,directory,prefix:(directory/(prefix+".o")).write_bytes(b"X"*64),True)
        self.run_cmake(ok=False)

    def test_wrong_manifest_hash(self):
        next((self.root/"b13/exact").glob("*.json")).write_text("{}")
        self.run_cmake(ok=False)

    def test_static_abi_and_hot_guard(self):
        for batch in (13,16):
            for mode in ("exact","mask"):
                args = SimpleNamespace(symbol_prefix="test",mode=mode,max_batch=batch,fixed_batch=batch)
                api,bridge = render_bridge(args)
                verify_static_abi(fixture_header("test",mode),"test",mode=="mask")
                with self.assertRaises(ValueError):
                    verify_static_abi(fixture_header("test",mode,False),"test",mode=="mask")
                hot = bridge.split('extern "C" cudaError_t test_launch',1)[1]
                self.assertIn(f"batch!={batch} || !packedQKV",hot)
                self.assertIn("!mask" if mode=="mask" else "mask != nullptr",hot)
                self.assertIn("alignment & 15",hot)
                self.assertNotIn("cudaGetDevice",hot)
                self.assertNotIn("prepareCurrentDevice",hot)
                self.assertNotIn("{{batch",hot)


def test_fake_cpp_runtime(output,cxx):
    """Compile production dispatcher + rendered static bridges against a fake runtime."""
    directory = output/"fake-runtime"
    directory.mkdir()
    (directory/"cuda_runtime_api.h").write_text('''#pragma once
using cudaStream_t=void*;
enum cudaError_t {cudaSuccess=0,cudaErrorInvalidValue=1,cudaErrorUnknown=999};
inline cudaError_t cudaPeekAtLastError(){return cudaSuccess;}
''')
    (directory/"aot_loader_cuda.h").write_text('''#pragma once
#include "cuda_runtime_api.h"
extern int test_prepare_calls; extern cudaError_t test_prepare_error;
namespace b11aot {
struct Rtx5090Runtime {};
template<class M,class R> struct CurrentDeviceLoader {
  M value{}; struct Result{cudaError_t error;};
  CurrentDeviceLoader(void(*)(void**),void(*)(void**)){}
  Result prepareCurrentDevice(){++test_prepare_calls;return {test_prepare_error};}
  M* module(){return &value;}
};}
''')
    sources = []
    stubs = ['#include "cuda_runtime_api.h"\nint test_prepare_calls=0; cudaError_t test_prepare_error=cudaSuccess; int test_calls=0;\n']
    checks = []
    for batch in (13,16):
        for mode in ("exact","mask"):
            prefix = f"b11_fa4_static_b{batch}_{mode}_m128_n96_s1"
            api,bridge = render_bridge(SimpleNamespace(symbol_prefix=prefix,mode=mode,max_batch=batch,fixed_batch=batch))
            (directory/(prefix+".h")).write_text(fixture_header(prefix,mode))
            (directory/(prefix+"_api.h")).write_text(api)
            path = directory/(prefix+"_bridge.cpp")
            path.write_text(bridge)
            sources.append(path)
            names = ("mQ","mK","mV","mO","mMask") if mode=="mask" else ("mQ","mK","mV","mO")
            args = ",".join(f"{prefix}_Tensor_{name}_t*" for name in names)
            stubs.append(f'''#include "{prefix}.h"
#include "{prefix}_api.h"
extern "C" void _mlir_{prefix}_cuda_init(void**){{}}
extern "C" void _mlir_{prefix}_cuda_load_to_device(void**){{}}
extern "C" int32_t cute_dsl_{prefix}_wrapper({prefix}_Kernel_Module_t*,{args},float,cudaStream_t){{++test_calls;return 0;}}
''')
            mask = "p" if mode=="mask" else "nullptr"
            checks.append(f'''  check({prefix}_launch(p,p,p,p,{mask},{batch},1,true,nullptr)==cudaSuccess);
  for(int b=-1;b<=97;++b) if(b!={batch}) check({prefix}_launch(p,p,p,p,{mask},b,1,true,nullptr)==cudaErrorInvalidValue);
  check({prefix}_launch(p,p,p,p,{mask},{batch},1,false,nullptr)==cudaErrorInvalidValue);
  check({prefix}_launch(p,p,p,p,{'nullptr' if mode=='mask' else 'p'},{batch},1,true,nullptr)==cudaErrorInvalidValue);
  check({prefix}_launch(nullptr,p,p,p,{mask},{batch},1,true,nullptr)==cudaErrorInvalidValue);
  check({prefix}_launch(p,nullptr,p,p,{mask},{batch},1,true,nullptr)==cudaErrorInvalidValue);
  check({prefix}_launch(p,p,nullptr,p,{mask},{batch},1,true,nullptr)==cudaErrorInvalidValue);
  check({prefix}_launch(p,p,p,nullptr,{mask},{batch},1,true,nullptr)==cudaErrorInvalidValue);
  check({prefix}_launch(bad,p,p,p,{mask},{batch},1,true,nullptr)==cudaErrorInvalidValue);
''')
    test = directory/"test.cpp"
    test.write_text("".join(stubs)+'''#include "b11_fa4_exact_m128_n96_s1_api.h"
#include "b11_fa4_mask_m128_n96_s1_api.h"
#include "static_fa4_profile.h"
#include <cstdlib>
void check(bool x){if(!x)std::exit(2);}
int main(){void* p=reinterpret_cast<void*>(0x10000);void* bad=reinterpret_cast<void*>(0x10001);
'''+"".join(checks)+'''
  check(test_calls==4);
  for(int b=-1;b<=97;++b){
    bool supported=b==13||b==16;
    check(b11aot::supportsStaticFa4(b,true)==supported);
    check(!b11aot::supportsStaticFa4(b,false));
    check(b11_fa4_exact_m128_n96_s1_launch(p,p,p,p,nullptr,b,1,true,nullptr)==(supported?cudaSuccess:cudaErrorInvalidValue));
    check(b11_fa4_mask_m128_n96_s1_launch(p,p,p,p,p,b,1,true,nullptr)==(supported?cudaSuccess:cudaErrorInvalidValue));
    check(b11_fa4_exact_m128_n96_s1_launch(p,p,p,p,nullptr,b,1,false,nullptr)==cudaErrorInvalidValue);
  }
  check(test_calls==8);
  check(b11_fa4_exact_m128_n96_s1_prepare()==cudaSuccess);check(test_prepare_calls==2);
  check(b11_fa4_mask_m128_n96_s1_prepare()==cudaSuccess);check(test_prepare_calls==4);
  test_prepare_error=cudaErrorUnknown;
  check(b11_fa4_exact_m128_n96_s1_prepare()==cudaErrorUnknown);check(test_prepare_calls==5);
  return 0;
}
''')
    sources += [HERE/"static_fa4_dispatch.cpp",test]
    exe = directory/("test.exe" if sys.platform=="win32" else "test")
    if Path(cxx).stem.lower()=="cl":
        command = [cxx,"/nologo","/std:c++17","/EHsc",f"/I{directory}",f"/I{HERE}",f"/Fe{exe}"]+[str(p) for p in sources]
    else:
        command = [cxx,"-std=c++17","-I",str(directory),"-I",str(HERE),"-o",str(exe)]+[str(p) for p in sources]
    result = subprocess.run(command,cwd=directory,capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError("fake runtime compile failed\n"+result.stdout+result.stderr)
    subprocess.run([str(exe)],cwd=directory,check=True)
    print("PASS: actual C++ static bridges/dispatcher, B=-1..97, packedness, mask/null/alignment and prepare-error propagation")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--cmake",default="cmake")
    parser.add_argument("--cxx",help="Optional host compiler (g++/clang++/cl) for fake-runtime execution")
    OPTIONS = parser.parse_args()
    OPTIONS.output_dir = OPTIONS.output_dir.resolve()
    if OPTIONS.output_dir==HERE or HERE in OPTIONS.output_dir.parents or OPTIONS.output_dir==HERE.parents[2]:
        parser.error("test outputs must be external to source and repository root")
    OPTIONS.output_dir.mkdir(parents=True,exist_ok=True)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ArtifactTests))
    if not result.wasSuccessful():
        sys.exit(1)
    if OPTIONS.cxx:
        test_fake_cpp_runtime(OPTIONS.output_dir,OPTIONS.cxx)

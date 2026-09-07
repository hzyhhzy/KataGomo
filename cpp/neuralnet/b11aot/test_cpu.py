#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Standard-library-only tests; no CUDA runtime or CuTe import is needed."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
sys.dont_write_bytecode = True
from fake_tensors import StaticCudaTensorSpec
from generate_fa4 import render_bridge
from prepare_dependencies import patch_both16, patch_auxiliary

HERE=Path(__file__).resolve().parent


class CpuTests(unittest.TestCase):
    def test_inert_tensor(self):
        spec=StaticCudaTensorSpec((16,361,12,32),(138624,384,32,1),0x10000)
        self.assertEqual(spec.__dlpack_device__(),(2,0))
        t=spec._managed.dl_tensor
        self.assertEqual(t.data,0x10000)
        self.assertEqual((t.dtype.code,t.dtype.bits,t.dtype.lanes),(2,16,1))
        self.assertEqual(list(t.shape[:4]),[16,361,12,32])
        self.assertEqual(list(t.strides[:4]),[138624,384,32,1])
        self.assertEqual(type(spec.__dlpack__()).__name__,"PyCapsule")

    def test_invalid_metadata(self):
        for shape,strides,address in (((1,2),(1,),16),((0,),(1,),16),((1,),(-1,),16),((1,),(1,),17)):
            with self.assertRaises(ValueError):
                StaticCudaTensorSpec(shape,strides,address)

    def test_safe_bridge(self):
        for mode in ("exact","mask"):
            api,bridge=render_bridge(SimpleNamespace(symbol_prefix="test",mode=mode,max_batch=96))
            self.assertIn("test_prepare",api)
            self.assertIn("CurrentDeviceLoader",bridge)
            self.assertNotIn("Kernel_Module_Load(",bridge)
            self.assertNotIn("std::call_once",bridge)
            hot=bridge.split('extern "C" cudaError_t test_launch',1)[1]
            self.assertNotIn("prepareCurrentDevice",hot)
            self.assertNotIn("cudaGetDevice",hot)
            self.assertIn("batch>96",hot)
            self.assertIn("!mask" if mode=="mask" else "mask != nullptr",hot)

    def test_adapter_identity(self):
        lock=json.loads((HERE/"dependencies.json").read_text())
        data=(HERE/"patches/flash_fwd_sm120.py").read_bytes().replace(b"\r\n",b"\n")
        self.assertEqual(hashlib.sha256(data).hexdigest(),
            lock["sources"]["flash_attn/cute/flash_fwd_sm120.py"]["canonical_lf_sha256"])

    def test_unexpected_patch_source(self):
        lock=json.loads((HERE/"dependencies.json").read_text())
        with self.assertRaises(ValueError):
            patch_both16(b"unexpected source",lock)
        for name in ("mask.py","softmax.py"):
            with self.assertRaises(ValueError):
                patch_auxiliary(b"unexpected source",lock,name)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package",type=Path)
    args=parser.parse_args()
    if args.source_package:
        lock=json.loads((HERE/"dependencies.json").read_text())
        output=patch_both16((args.source_package/"cute/flash_fwd.py").read_bytes(),lock)
        assert patch_both16(output,lock)==output
        for name in ("mask.py","softmax.py"):
            output=patch_auxiliary((args.source_package/"cute"/name).read_bytes(),lock,name)
            assert patch_auxiliary(output,lock,name)==output
        print("PASS: actual dependency patch and idempotence")
    unittest.main(argv=[sys.argv[0]])

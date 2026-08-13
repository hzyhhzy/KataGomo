#!/usr/bin/env python3
"""Regression for Python 3.12 inspection of lazily imported CUDA types."""

from __future__ import annotations

import inspect
from pathlib import Path
import sys
import unittest


TOOLS = Path(__file__).resolve().parents[1] / "c384_exact_fixed_aot"
sys.path.insert(0, str(TOOLS))

from generator_common import bind_local_stream_annotation


class C384ExactGeneratorAnnotationTest(unittest.TestCase):
    def test_local_cuda_annotation_is_bound_before_inspection(self) -> None:
        class FakeCuda:
            class CUstream:
                pass

        def local_launch(stream: "cuda.CUstream"):
            del stream

        # Use the exact deferred spelling produced by the real launch nested
        # in main(). No module-global ``cuda`` exists, so Python 3.12 fails.
        local_launch.__annotations__["stream"] = "cuda.CUstream"
        with self.assertRaises(NameError):
            inspect.get_annotations(local_launch, eval_str=True)
        bind_local_stream_annotation(local_launch,FakeCuda.CUstream)
        self.assertIs(
            inspect.get_annotations(local_launch, eval_str=True)["stream"],
            FakeCuda.CUstream,
        )


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Regression for Python 3.12 inspection of lazily imported CUDA types."""

from __future__ import annotations

import inspect
import functools
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

        def implementation(stream: "cuda.CUstream"):
            del stream

        @functools.wraps(implementation)
        def local_launch(*args, **kwargs):
            return implementation(*args, **kwargs)

        # CuTe's decorator returns a function whose public wrapper and
        # __wrapped__ implementation both retain the deferred local spelling.
        # No module-global ``cuda`` exists, so Python 3.12 fails at either one.
        local_launch.__annotations__["stream"] = "cuda.CUstream"
        implementation.__annotations__["stream"] = "cuda.CUstream"
        with self.assertRaises(NameError):
            inspect.signature(local_launch, eval_str=True)
        bind_local_stream_annotation(local_launch,FakeCuda.CUstream)
        for function in (local_launch,local_launch.__wrapped__):
            self.assertIs(
                inspect.get_annotations(function, eval_str=True)["stream"],
                FakeCuda.CUstream,
            )
        self.assertIs(
            inspect.signature(local_launch, eval_str=True)
              .parameters["stream"].annotation,
            FakeCuda.CUstream,
        )

    def test_wrapped_annotation_cycle_fails_closed(self) -> None:
        def launch(stream: "cuda.CUstream"):
            del stream

        launch.__annotations__["stream"] = "cuda.CUstream"
        launch.__wrapped__ = launch
        with self.assertRaisesRegex(ValueError, "cycle"):
            bind_local_stream_annotation(launch,object)


if __name__ == "__main__":
    unittest.main()

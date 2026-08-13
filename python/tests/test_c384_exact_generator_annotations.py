#!/usr/bin/env python3
"""Regression for Python 3.12 inspection of lazily imported CUDA types."""

from __future__ import annotations

import inspect
import functools
import importlib.metadata
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid


TOOLS = Path(__file__).resolve().parents[1] / "c384_exact_fixed_aot"
sys.path.insert(0, str(TOOLS))

from generator_common import bind_local_stream_annotation, imported_module_provenance


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

    def setUp(self) -> None:
        self.fixture = Path(__file__).resolve().parent / (
            "c384-provenance-" + uuid.uuid4().hex
        )
        self.fixture.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.fixture)

    def _fake_module(self, version: str = "4.6.0.dev0"):
        root = self.fixture
        source = root / "module.py"
        source.write_text("# authoritative imported source\n", encoding="utf-8")
        return SimpleNamespace(
            __name__="fake_module", __version__=version, __file__=str(source),
        )

    def test_imported_module_provenance_without_dist_info(self) -> None:
        module = self._fake_module()
        with patch("importlib.metadata.version",
                   side_effect=importlib.metadata.PackageNotFoundError):
            result = imported_module_provenance(module,"fake-dist")
        self.assertEqual(result["version"],"4.6.0.dev0")
        self.assertEqual(result["metadata_status"],"absent")
        self.assertIsNone(result["distribution_version"])
        self.assertEqual(len(result["module_file_sha256"]),64)

    def test_imported_module_provenance_matching_dist_info(self) -> None:
        module = self._fake_module()
        with patch("importlib.metadata.version",return_value="4.6.0.dev0"):
            result = imported_module_provenance(module,"fake-dist")
        self.assertEqual(result["metadata_status"],"present-matching")
        self.assertEqual(result["distribution_version"],"4.6.0.dev0")

    def test_extension_uses_package_version_but_hashes_loaded_binary(self) -> None:
        extension = self._fake_module()
        del extension.__version__
        package = SimpleNamespace(
            __name__="fake_package", __version__="13.3.1",
        )
        with patch("importlib.metadata.version",return_value="13.3.1"):
            result = imported_module_provenance(
                extension,"fake-dist",version_module=package,
            )
        self.assertEqual(result["module"],"fake_module")
        self.assertEqual(result["version_module"],"fake_package")
        self.assertEqual(result["version"],"13.3.1")

    def test_imported_module_provenance_rejects_other_dist_code(self) -> None:
        module = self._fake_module()
        with patch("importlib.metadata.version",return_value="4.6.2"):
            with self.assertRaisesRegex(ValueError,"does not match imported"):
                imported_module_provenance(module,"fake-dist")

if __name__ == "__main__":
    unittest.main()

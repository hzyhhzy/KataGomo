from __future__ import annotations

import hashlib
import sys
import unittest
import uuid
from pathlib import Path

import numpy as np
import torch


PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))
TEST_RUNTIME_DATA = PYTHON_DIR.parent / "runtime_data"
TEST_RUNTIME_DATA.mkdir(exist_ok=True)

import calibrate_cpu_ptq_v106 as calibration  # noqa: E402
import convert_cpu_ptq_v106 as converter  # noqa: E402


class CalibrationTests(unittest.TestCase):
    def test_symmetric_codes_are_per_row_and_ties_to_even(self) -> None:
        values = torch.tensor(
            [[0.0, 0.0], [1.0, -0.5], [2.0, 1.0]],
            dtype=torch.float32,
        )
        codes, scales = calibration.symmetric_codes(values, 127.0, 1)
        torch.testing.assert_close(
            scales,
            torch.tensor([[1.0], [1.0 / 127.0], [2.0 / 127.0]]),
            rtol=0.0,
            atol=0.0,
        )
        self.assertTrue(
            torch.equal(
                codes,
                torch.tensor([[0.0, 0.0], [127.0, -64.0], [127.0, 64.0]]),
            )
        )

    def test_gptq_codes_preserve_geometry_and_declared_range(self) -> None:
        moments = calibration.Moments(4, torch.device("cpu"))
        moments.observe(
            torch.tensor(
                [
                    [1.0, -2.0, 0.5, 0.25],
                    [-0.5, 0.75, 1.5, -1.0],
                    [2.0, 1.0, -0.25, 0.5],
                ],
                dtype=torch.float32,
            )
        )
        weight = torch.tensor(
            [[0.25, -0.5, 0.75, 1.0], [-1.0, 0.5, 0.125, -0.25]],
            dtype=torch.float32,
        )
        codes, scales = calibration.gptq_matrix(
            weight, moments, 63, 0.001, False, 0.0, 1.0
        )
        self.assertEqual(codes.shape, weight.shape)
        self.assertEqual(scales.shape, (2, 1))
        self.assertLessEqual(int(codes.abs().max()), 63)
        torch.testing.assert_close(
            scales[:, 0], weight.abs().amax(dim=1) / 63.0
        )


class ConverterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = converter.Matrix(
            "model.blocks.0.attention.q_proj",
            inputs=2,
            outputs=3,
            values=np.asarray(
                [[0.25, -0.5, 0.75], [1.0, 0.5, -0.25]],
                dtype=np.float32,
            ),
            payload_span=(0, 24),
        )
        self.codes = np.asarray(
            [[1, -2], [3, 4], [-5, 6]], dtype=np.int8
        )
        self.scales = np.asarray([0.25, 0.5, 0.75], dtype=np.float32)

    def write_manifest(self, path: Path, source_hash: str) -> None:
        np.savez(
            path,
            format=np.asarray("cpuptq-gptq-v1"),
            qmax=np.asarray(63, dtype=np.int32),
            activation_quantizer=np.asarray("row-sym"),
            activation_scale_factor=np.asarray(1.0, dtype=np.float32),
            recipe=np.asarray("test"),
            names=np.asarray(["blocks.0.q_proj"]),
            codes_0=self.codes,
            scales_0=self.scales,
            source_sha256_0=np.asarray(source_hash),
        )

    def test_manifest_contract_and_canonical_payload(self) -> None:
        source_hash = converter._source_weight_sha256(self.source)
        manifest = TEST_RUNTIME_DATA / f"cpuptq-v106-{uuid.uuid4().hex}.npz"
        try:
            self.write_manifest(manifest, source_hash)
            loaded = converter._load_gptq_overrides(
                manifest, [self.source], 63
            )
        finally:
            manifest.unlink(missing_ok=True)
        override = loaded.matrices["blocks.0.q_proj"]
        payload, code_bytes, scale_bytes = converter._quantize(
            self.source, 63, override
        )
        self.assertTrue(payload.startswith(converter.S7_MARKER))
        self.assertEqual(code_bytes, self.codes.size)
        self.assertEqual(scale_bytes, self.scales.nbytes)
        offset = len(converter.S7_MARKER) + self.scales.nbytes
        self.assertEqual(payload[offset:], self.codes.tobytes(order="C"))

    def test_manifest_rejects_a_different_checkpoint(self) -> None:
        wrong_hash = hashlib.sha256(b"different checkpoint").hexdigest()
        manifest = TEST_RUNTIME_DATA / f"cpuptq-v106-{uuid.uuid4().hex}.npz"
        try:
            self.write_manifest(manifest, wrong_hash)
            with self.assertRaisesRegex(ValueError, "checkpoint does not match"):
                converter._load_gptq_overrides(manifest, [self.source], 63)
        finally:
            manifest.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

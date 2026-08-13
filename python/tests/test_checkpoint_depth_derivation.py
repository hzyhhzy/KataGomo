from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import sys
import textwrap
import unittest
import uuid

try:
    import torch
    from torch.optim.swa_utils import AveragedModel
except Exception:  # pragma: no cover - the class is explicitly skipped below
    torch = None
    AveragedModel = None


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "python" / "tools" / "derive_checkpoint_depth.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@unittest.skipIf(torch is None, "PyTorch is unavailable; checkpoint CPU tests skipped")
class CheckpointDepthDerivationTests(unittest.TestCase):
    SOURCE_DEPTH = 4
    TARGET_DEPTH = 2

    def setUp(self) -> None:
        scratch_root = ROOT / "tmp" / "checkpoint-depth-tests"
        scratch_root.mkdir(parents=True, exist_ok=True)
        self.temp_dir = scratch_root / f"case-{uuid.uuid4().hex}"
        self.temp_dir.mkdir()
        self.train_dir = self.temp_dir / "train"
        self.train_dir.mkdir()
        self._write_train_api(self.train_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    @staticmethod
    def _write_train_api(train_dir: Path) -> None:
        (train_dir / "model_pytorch.py").write_text(
            textwrap.dedent(
                """
                import torch

                class Model(torch.nn.Module):
                    def __init__(self, config, pos_len):
                        super().__init__()
                        self.pos_len = pos_len
                        self.blocks = torch.nn.ModuleList([
                            torch.nn.Linear(3, 3, bias=True)
                            for _ in config["block_kind"]
                        ])
                        self.head = torch.nn.Linear(3, 2, bias=False)

                    def initialize(self):
                        return None
                """
            ).lstrip(),
            encoding="utf-8",
        )
        (train_dir / "load_model.py").write_text(
            textwrap.dedent(
                """
                from torch.optim.swa_utils import AveragedModel
                from model_pytorch import Model

                def load_model(checkpoint_file, use_swa, device, pos_len=7, verbose=False):
                    raise RuntimeError("derivation validator must use strict direct loads")
                """
            ).lstrip(),
            encoding="utf-8",
        )

    def _checkpoint(self, swa_key: str = "swa_model_0") -> dict:
        sys.path.insert(0, str(self.train_dir))
        try:
            from model_pytorch import Model

            torch.manual_seed(12345)
            config = {
                "block_kind": [
                    [f"block{index + 1}", "synthetic"]
                    for index in range(self.SOURCE_DEPTH)
                ],
                "marker": "must-remain-identical",
            }
            model = Model(config, 7)
            averaged = AveragedModel(model, device="cpu")
            return {
                "config": config,
                "model": model.state_dict(),
                swa_key: averaged.state_dict(),
                "extra_tensor": torch.arange(5, dtype=torch.int32),
                "train_state": {
                    "step": 17,
                    "nested_tensor": torch.tensor([1.5], dtype=torch.float64),
                },
            }
        finally:
            sys.path.pop(0)
            sys.modules.pop("model_pytorch", None)

    def _save_source(self, checkpoint: dict, name: str = "source.ckpt") -> Path:
        source = self.temp_dir / name
        torch.save(checkpoint, source)
        return source

    def _run(
        self,
        source: Path,
        *,
        output: Path | None = None,
        source_sha: str | None = None,
        train_dir: Path | None = None,
        target_depth: int | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], Path]:
        if output is None:
            output = self.temp_dir / "derived.ckpt"
        command = [
            sys.executable,
            str(TOOL),
            "--source",
            str(source),
            "--output",
            str(output),
            "--source-sha",
            source_sha if source_sha is not None else sha256_file(source),
            "--target-depth",
            str(self.TARGET_DEPTH if target_depth is None else target_depth),
            "--train-dir",
            str(self.train_dir if train_dir is None else train_dir),
        ]
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return result, output

    def _assert_failure_left_no_artifacts(
        self, result: subprocess.CompletedProcess[str], output: Path
    ) -> None:
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("ERROR:", result.stderr)
        self.assertFalse(output.exists())
        self.assertFalse(output.with_name(output.name + ".manifest.json").exists())

    def test_derives_exact_prefix_with_swa_model_0_and_manifest(self) -> None:
        source_checkpoint = self._checkpoint("swa_model_0")
        source = self._save_source(source_checkpoint)
        source_nonblock = source_checkpoint["model"]["head.weight"].clone()
        source_extra = source_checkpoint["extra_tensor"].clone()

        result, output = self._run(source)
        self.assertEqual(result.returncode, 0, result.stderr)
        derived = torch.load(output, map_location="cpu", weights_only=False)
        self.assertEqual(
            derived["config"]["block_kind"],
            source_checkpoint["config"]["block_kind"][: self.TARGET_DEPTH],
        )
        self.assertEqual(derived["config"]["marker"], "must-remain-identical")
        for state_name in ("model", "swa_model_0"):
            block_indices = {
                int(part)
                for key in derived[state_name]
                for part in key.split(".")
                if part.isdigit()
            }
            self.assertEqual(block_indices, {0, 1})
        self.assertTrue(torch.equal(derived["model"]["head.weight"], source_nonblock))
        self.assertTrue(torch.equal(derived["extra_tensor"], source_extra))

        manifest_path = output.with_name(output.name + ".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["input"]["sha256"], sha256_file(source))
        self.assertEqual(manifest["output"]["sha256"], sha256_file(output))
        self.assertEqual(manifest["input"]["source_depth"], self.SOURCE_DEPTH)
        self.assertEqual(manifest["output"]["target_depth"], self.TARGET_DEPTH)
        self.assertEqual(manifest["states"]["swa"], "swa_model_0")
        self.assertTrue(manifest["validation"]["regular_strict_load"])
        self.assertTrue(manifest["validation"]["swa_strict_load"])
        self.assertEqual(
            manifest["validation"]["constructor_arguments"]["pos_len"],
            {
                "source": "train-dir load_model.load_model pos_len default",
                "value": 7,
            },
        )
        self.assertGreater(manifest["deletion"]["total"]["count"], 0)
        self.assertEqual(len(manifest["retained_tensor_digest"]["sha256"]), 64)
        self.assertEqual(
            len(manifest["retained_non_block_tensor_digest"]["sha256"]), 64
        )
        self.assertEqual(manifest["retained_auxiliary_tensor_digest"]["count"], 2)

    def test_accepts_legacy_swa_model_name(self) -> None:
        checkpoint = self._checkpoint("swa_model")
        checkpoint["model"] = {
            "module." + key: value for key, value in checkpoint["model"].items()
        }
        source = self._save_source(checkpoint)
        result, output = self._run(source)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads(
            output.with_name(output.name + ".manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["states"]["swa"], "swa_model")

    def test_rejects_wrong_source_sha_and_existing_output(self) -> None:
        source = self._save_source(self._checkpoint())
        result, output = self._run(source, source_sha="0" * 64)
        self._assert_failure_left_no_artifacts(result, output)
        self.assertIn("source SHA-256 mismatch", result.stderr)

        output.write_bytes(b"do-not-overwrite")
        result, _ = self._run(source, output=output)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(output.read_bytes(), b"do-not-overwrite")
        self.assertIn("refusing to overwrite", result.stderr)

    def test_rejects_missing_noncontiguous_and_extra_block_indices(self) -> None:
        cases = []
        missing = self._checkpoint()
        for key in list(missing["model"]):
            if key.startswith("blocks.1."):
                del missing["model"][key]
        cases.append(("missing", missing, "not the complete contiguous range"))

        extra = self._checkpoint()
        extra["model"]["blocks.4.weight"] = torch.zeros((3, 3))
        cases.append(("extra", extra, "outside config depth"))

        malformed = self._checkpoint()
        malformed["model"]["trunk.blocks.0.shadow"] = torch.zeros(1)
        cases.append(("malformed", malformed, "unsupported block-key pattern"))

        depth_coupled = self._checkpoint()
        depth_coupled["config"]["has_intermediate_head"] = True
        depth_coupled["config"]["intermediate_head_blocks"] = self.SOURCE_DEPTH
        cases.append(
            (
                "depth-coupled-config",
                depth_coupled,
                "intermediate_head_blocks lies in the discarded suffix",
            )
        )

        for name, checkpoint, expected_error in cases:
            with self.subTest(name=name):
                source = self._save_source(checkpoint, f"{name}.ckpt")
                output = self.temp_dir / f"{name}-derived.ckpt"
                result, output = self._run(source, output=output)
                self._assert_failure_left_no_artifacts(result, output)
                self.assertIn(expected_error, result.stderr)

    def test_rejects_unhandled_state_and_ambiguous_swa(self) -> None:
        unhandled = self._checkpoint()
        unhandled["ema_model"] = {"blocks.0.weight": torch.zeros(1)}
        source = self._save_source(unhandled, "unhandled.ckpt")
        result, output = self._run(source, output=self.temp_dir / "unhandled-out.ckpt")
        self._assert_failure_left_no_artifacts(result, output)
        self.assertIn("outside model/SWA", result.stderr)

        ambiguous = self._checkpoint("swa_model_0")
        ambiguous["swa_model"] = ambiguous["swa_model_0"].copy()
        source = self._save_source(ambiguous, "ambiguous.ckpt")
        result, output = self._run(source, output=self.temp_dir / "ambiguous-out.ckpt")
        self._assert_failure_left_no_artifacts(result, output)
        self.assertIn("exactly one", result.stderr)

    def test_rejects_strict_model_shape_failure(self) -> None:
        checkpoint = self._checkpoint()
        checkpoint["model"]["head.weight"] = torch.zeros((4, 3))
        checkpoint["swa_model_0"]["module.head.weight"] = torch.zeros((4, 3))
        source = self._save_source(checkpoint)
        result, output = self._run(source)
        self._assert_failure_left_no_artifacts(result, output)
        self.assertIn("strict regular Model load failed", result.stderr)

    def test_rejects_missing_training_dependencies_and_invalid_target(self) -> None:
        source = self._save_source(self._checkpoint())
        empty_train_dir = self.temp_dir / "empty-train"
        empty_train_dir.mkdir()
        result, output = self._run(source, train_dir=empty_train_dir)
        self._assert_failure_left_no_artifacts(result, output)
        self.assertIn("must provide importable", result.stderr)

        result, output = self._run(
            source,
            output=self.temp_dir / "invalid-target.ckpt",
            target_depth=self.SOURCE_DEPTH,
        )
        self._assert_failure_left_no_artifacts(result, output)
        self.assertIn("must be smaller", result.stderr)


if __name__ == "__main__":
    unittest.main()

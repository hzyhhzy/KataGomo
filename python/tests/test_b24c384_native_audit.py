from __future__ import annotations

import copy
from contextlib import contextmanager
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock
import uuid

try:
    import torch
except Exception:  # pragma: no cover - the end-to-end test is skipped below
    torch = None


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "python" / "tools" / "audit_b24c384_native.py"
TEST_TMP = ROOT / "tmp" / "b24c384-native-audit-tests"
TEST_TMP.mkdir(parents=True, exist_ok=True)
SPEC = importlib.util.spec_from_file_location("audit_b24c384_native", TOOL)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


@contextmanager
def scratch_dir():
    path = TEST_TMP / f"case-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def valid_parsed() -> dict:
    blocks = []
    for layer in range(24):
        blocks.extend(
            [
                {
                    "kind": "transformer_attention_block",
                    "name": f"rconv{layer + 1}.attention",
                    "heads": 12,
                    "kv_heads": 12,
                    "q_head_dim": 32,
                    "v_head_dim": 32,
                    "use_rope": True,
                    "learnable_rope": True,
                    "q": {"in": 384, "out": 384},
                    "rope": {"shape": [12, 16, 2]},
                },
                {
                    "kind": "transformer_ffn_block",
                    "name": f"rconv{layer + 1}.ffn",
                    "channels": 384,
                    "ffn_channels": 1024,
                    "swiglu": True,
                },
            ]
        )
    return {
        "name": AUDIT.EXPECTED_MODEL_NAME,
        "version": 102,
        "spatial_features": 22,
        "global_features": 39,
        "trunk": {"descriptor_count": 48, "channels": 384, "blocks": blocks},
        "policy": {"p1": {"out": 48}, "g1": {"out": 48}},
        "value": {"v1": {"out": 96}},
        "serialized_float_count": 42_707_177,
        "weight_block_count": 268,
    }


def valid_manifest(checkpoint: Path) -> dict:
    per_block = {str(index): 10 for index in range(24, 36)}
    return {
        "schema": 1,
        "operation": "strict-checkpoint-depth-prefix-derivation",
        "status": "cpu-derived-and-strict-load-validated",
        "input": {
            "sha256": AUDIT.TRUSTED_SOURCE_CHECKPOINT_SHA256,
            "source_depth": 36,
        },
        "output": {
            "sha256": sha256_file(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "target_depth": 24,
        },
        "states": {"regular": "model", "swa": "swa_model"},
        "config": {"only_field_changed": "config.block_kind"},
        "deletion": {
            "index_range": [24, 35],
            "total": {"count": 240, "bytes": 169_979_904},
            "by_state": {
                state: {
                    "keys": 120,
                    "tensor_bytes": 84_989_952,
                    "per_block_key_count": per_block,
                }
                for state in ("model", "swa_model")
            },
        },
        "validation": {
            "device": "cpu",
            "regular_strict_load": True,
            "swa_strict_load": True,
        },
        "atomic_no_clobber_publish": True,
        "gpu_work_performed": False,
        "native_export_performed": False,
    }


def valid_checkpoint() -> dict:
    config = {
        "version": 102,
        "block_kind": [
            [f"rconv{index}", "transformerropesg"] for index in range(1, 25)
        ],
        "trunk_num_channels": 384,
        "mid_num_channels": 384,
        "transformer_ffn_channels": 1024,
        "transformer_heads": 12,
        "transformer_kv_heads": 12,
        "learnable_rope": True,
        "activation": "silu",
        "norm_kind": "bnorm",
        "bnorm_use_gamma": True,
        "initial_conv_1x1": False,
        "p1_num_channels": 48,
        "g1_num_channels": 48,
        "v1_num_channels": 96,
    }
    n_averaged = torch.tensor(1) if torch is not None else 1
    return {
        "config": config,
        "model": {"fixture": n_averaged},
        "swa_model": {"n_averaged": n_averaged},
    }


class B24C384NativeAuditTests(unittest.TestCase):
    def test_accepts_exact_native_contract(self) -> None:
        report = AUDIT.audit_native_structure(valid_parsed())
        self.assertEqual(report["name"], "b24c384h12tflrs-renju15-swa")
        self.assertEqual(report["descriptor_count"], 48)
        self.assertEqual(report["descriptor_kind_counts"], {
            "transformer_attention_block": 24,
            "transformer_ffn_block": 24,
        })
        self.assertEqual(report["serialized_float_count"], 42_707_177)
        self.assertEqual(report["weight_block_count"], 268)
        self.assertTrue(report["eof_clean"])

    def test_rejects_name_and_each_specialization_dimension(self) -> None:
        mutations = {
            "model name": lambda value: value.update(name="wrong"),
            "version": lambda value: value.update(version=104),
            "spatial features": lambda value: value.update(spatial_features=21),
            "global features": lambda value: value.update(global_features=40),
            "descriptor count": lambda value: value["trunk"].update(descriptor_count=46),
            "trunk channels": lambda value: value["trunk"].update(channels=256),
            "attention heads": lambda value: value["trunk"]["blocks"][0].update(heads=8),
            "KV heads": lambda value: value["trunk"]["blocks"][0].update(kv_heads=8),
            "head dimension": lambda value: value["trunk"]["blocks"][0].update(q_head_dim=64),
            "RoPE": lambda value: value["trunk"]["blocks"][0].update(use_rope=False),
            "FFN channels": lambda value: value["trunk"]["blocks"][1].update(ffn_channels=768),
            "SwiGLU": lambda value: value["trunk"]["blocks"][1].update(swiglu=False),
            "weight blocks": lambda value: value.update(weight_block_count=267),
            "serialized floats": lambda value: value.update(serialized_float_count=42_707_176),
            "policy head": lambda value: value["policy"]["p1"].update(out=32),
            "value head": lambda value: value["value"]["v1"].update(out=64),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                parsed = valid_parsed()
                mutate(parsed)
                with self.assertRaises(AUDIT.AuditError):
                    AUDIT.audit_native_structure(parsed)

    def test_rejects_nonalternating_or_nonexact_depth(self) -> None:
        wrong_order = valid_parsed()
        wrong_order["trunk"]["blocks"][0], wrong_order["trunk"]["blocks"][1] = (
            wrong_order["trunk"]["blocks"][1],
            wrong_order["trunk"]["blocks"][0],
        )
        with self.assertRaisesRegex(AUDIT.AuditError, "descriptor order"):
            AUDIT.audit_native_structure(wrong_order)

        short = valid_parsed()
        short["trunk"]["blocks"] = short["trunk"]["blocks"][:-2]
        with self.assertRaisesRegex(AUDIT.AuditError, "parsed block count"):
            AUDIT.audit_native_structure(short)

    def test_parse_model_supports_raw_and_gzip(self) -> None:
        with scratch_dir() as scratch:
            raw = scratch / "model.bin"
            compressed = scratch / "model.bin.gz"
            payload = b"strict-wire-payload\n"
            raw.write_bytes(payload)
            with compressed.open("wb") as raw_handle:
                with gzip.GzipFile(
                    filename="", mode="wb", fileobj=raw_handle, mtime=0
                ) as handle:
                    handle.write(payload)

            def parser(path: Path) -> dict:
                self.assertEqual(path.read_bytes(), payload)
                return valid_parsed()

            raw_parsed, raw_sha = AUDIT.parse_model(parser, raw)
            gzip_parsed, gzip_sha = AUDIT.parse_model(parser, compressed)
            self.assertEqual(raw_parsed, gzip_parsed)
            self.assertEqual(raw_sha, hashlib.sha256(payload).hexdigest())
            self.assertEqual(gzip_sha, raw_sha)

    def test_derivation_manifest_is_bound_fail_closed(self) -> None:
        with scratch_dir() as scratch:
            checkpoint = scratch / "derived.ckpt"
            checkpoint.write_bytes(b"derived checkpoint")
            manifest = valid_manifest(checkpoint)
            report = AUDIT.audit_derivation_manifest(
                manifest, checkpoint, sha256_file(checkpoint)
            )
            self.assertEqual(report["source_depth"], 36)
            self.assertEqual(report["target_depth"], 24)
            self.assertEqual(report["deleted_tensor_count"], 240)

            mutations = {
                "source SHA": lambda value: value["input"].update(sha256="0" * 64),
                "output SHA": lambda value: value["output"].update(sha256="0" * 64),
                "SWA name": lambda value: value["states"].update(swa="swa_model_0"),
                "index range": lambda value: value["deletion"].update(index_range=[25, 35]),
                "deleted count": lambda value: value["deletion"]["total"].update(count=239),
                "strict load": lambda value: value["validation"].update(swa_strict_load=False),
            }
            for label, mutate in mutations.items():
                with self.subTest(label=label):
                    candidate = copy.deepcopy(manifest)
                    mutate(candidate)
                    with self.assertRaises(AUDIT.AuditError):
                        AUDIT.audit_derivation_manifest(
                            candidate, checkpoint, sha256_file(checkpoint)
                        )

    def test_checkpoint_locks_unclipped_non_qkn_c384_prefix(self) -> None:
        checkpoint = valid_checkpoint()
        report = AUDIT.audit_checkpoint(checkpoint, "f" * 64)
        self.assertEqual(report["combined_layers"], 24)
        self.assertEqual(report["swa_state"], "swa_model")

        for key, value in (
            ("use_qk_norm", True),
            ("full_int8_clip", 7.0),
            ("swiglu_clip", 7.0),
        ):
            with self.subTest(key=key):
                candidate = copy.deepcopy(checkpoint)
                candidate["config"][key] = value
                with self.assertRaises(AUDIT.AuditError):
                    AUDIT.audit_checkpoint(candidate, "f" * 64)

    def test_untrusted_parser_hash_is_rejected_before_import(self) -> None:
        with scratch_dir() as scratch:
            parser = scratch / "parser.py"
            parser.write_text("raise RuntimeError('must not import')\n", encoding="utf-8")
            with self.assertRaisesRegex(AUDIT.AuditError, "SHA-256 mismatch"):
                AUDIT.load_trusted_parser(parser)

    @unittest.skipIf(torch is None, "PyTorch is unavailable")
    def test_end_to_end_gzip_report_and_blank_provenance(self) -> None:
        with scratch_dir() as scratch:
            checkpoint = scratch / "derived.ckpt"
            torch.save(valid_checkpoint(), checkpoint)
            manifest = scratch / "derived.ckpt.manifest.json"
            manifest.write_text(
                json.dumps(valid_manifest(checkpoint)), encoding="utf-8"
            )
            exporter = scratch / "export_model_pytorch.py"
            exporter.write_bytes(b"pinned exporter fixture")
            parser = scratch / "native_model_audit.py"
            parser.write_bytes(b"pinned parser fixture")
            model = scratch / "b24c384h12tflrs-renju15-swa.bin.gz"
            payload = b"native v102 wire fixture"
            with model.open("wb") as raw_handle:
                with gzip.GzipFile(
                    filename="", mode="wb", fileobj=raw_handle, mtime=0
                ) as handle:
                    handle.write(payload)

            def fake_load_parser(path: Path):
                self.assertEqual(path, parser.resolve())

                def parse(path_to_parse: Path) -> dict:
                    self.assertEqual(path_to_parse.read_bytes(), payload)
                    return valid_parsed()

                return parse

            with (
                mock.patch.object(
                    AUDIT, "TRUSTED_EXPORTER_SHA256", sha256_file(exporter)
                ),
                mock.patch.object(AUDIT, "load_trusted_parser", fake_load_parser),
            ):
                report = AUDIT.audit(
                    model=model,
                    checkpoint=checkpoint,
                    derivation_manifest=manifest,
                    exporter=exporter,
                    native_parser=parser,
                )
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["native_model"]["compression"], "gzip")
            self.assertEqual(
                report["native_model"]["raw_payload_sha256"],
                hashlib.sha256(payload).hexdigest(),
            )
            self.assertIn("blank/untrained", report["provenance"]["source_checkpoint_kind"])
            self.assertIn("not a trained", report["provenance"]["claim_boundary"])


if __name__ == "__main__":
    unittest.main()

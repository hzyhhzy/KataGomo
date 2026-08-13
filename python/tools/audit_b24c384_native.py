#!/usr/bin/env python3
"""Fail-closed audit for the derived b24 C384 native-v102 model.

Wire-format parsing is deliberately delegated to the independently reviewed
``native_model_audit.py`` parser.  This wrapper only accepts that exact parser
revision, adds gzip handling, binds the native model to the audited checkpoint
derivation manifest, and enforces the C384 specialization contract.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence


TRUSTED_NATIVE_PARSER_SHA256 = (
    "cd133a560d61c5f2878bd4cc8ea1a0cbc3bd7119bd97d5d1c9fb0028b5b1bae5"
)
TRUSTED_EXPORTER_SHA256 = (
    "4395d41b863ce1bdeafb46ba678ddb929364ba5c0769aef916fb577d7cc0b3f9"
)
TRUSTED_SOURCE_CHECKPOINT_SHA256 = (
    "98483d4265cba07984e7da136817a40b487da73a52bddd38a5bbd076e17e37f8"
)
EXPECTED_MODEL_NAME = "b24c384h12tflrs-renju15-swa"

EXPECTED = {
    "version": 102,
    "source_depth": 36,
    "target_depth": 24,
    "descriptors": 48,
    "channels": 384,
    "heads": 12,
    "kv_heads": 12,
    "head_dim": 32,
    "ffn_channels": 1024,
    "weight_blocks": 268,
    "serialized_floats": 42_707_177,
    "spatial_features": 22,
    "global_features": 39,
}


class AuditError(RuntimeError):
    """A contract mismatch that must prevent artifact acceptance."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_payload(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise AuditError(f"{label} is not a regular file: {resolved}")
    return resolved


def require_sha256(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise AuditError(f"{label} SHA-256 mismatch: expected {expected}, actual {actual}")
    return actual


def load_trusted_parser(path: Path) -> Any:
    require_sha256(path, TRUSTED_NATIVE_PARSER_SHA256, "native parser")
    spec = importlib.util.spec_from_file_location("trusted_native_v102_parser", path)
    if spec is None or spec.loader is None:
        raise AuditError(f"cannot import native parser: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise AuditError(
            f"cannot import native parser: {type(exc).__name__}: {exc}"
        ) from exc
    parse_native = getattr(module, "parse_native", None)
    if not callable(parse_native):
        raise AuditError("trusted native parser does not expose callable parse_native")
    return parse_native


def parse_model(parse_native: Any, model: Path) -> tuple[dict[str, Any], str]:
    try:
        if model.suffix.lower() != ".gz":
            return parse_native(model), sha256_file(model)
        handle = tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{model.name}.",
            suffix=".bin.tmp",
            dir=model.parent,
            delete=False,
        )
        raw = Path(handle.name)
        handle.close()
        try:
            with gzip.open(model, "rb") as source, raw.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=8 << 20)
            raw_sha = sha256_file(raw)
            parsed = parse_native(raw)
            return parsed, raw_sha
        finally:
            raw.unlink(missing_ok=True)
    except Exception as exc:
        raise AuditError(
            f"native parse failed: {type(exc).__name__}: {exc}"
        ) from exc


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AuditError(f"{label} mismatch: expected {expected!r}, actual {actual!r}")


def audit_derivation_manifest(
    manifest: Mapping[str, Any], checkpoint: Path, checkpoint_sha: str
) -> dict[str, Any]:
    require_equal(manifest.get("schema"), 1, "derivation manifest schema")
    require_equal(
        manifest.get("operation"),
        "strict-checkpoint-depth-prefix-derivation",
        "derivation operation",
    )
    require_equal(
        manifest.get("status"),
        "cpu-derived-and-strict-load-validated",
        "derivation status",
    )
    input_record = manifest.get("input")
    output_record = manifest.get("output")
    states = manifest.get("states")
    config = manifest.get("config")
    deletion = manifest.get("deletion")
    validation = manifest.get("validation")
    if not all(
        isinstance(value, Mapping)
        for value in (input_record, output_record, states, config, deletion, validation)
    ):
        raise AuditError("derivation manifest is missing required object records")

    require_equal(
        input_record.get("sha256"),
        TRUSTED_SOURCE_CHECKPOINT_SHA256,
        "source checkpoint SHA-256",
    )
    require_equal(
        input_record.get("source_depth"), EXPECTED["source_depth"], "source depth"
    )
    require_equal(
        output_record.get("sha256"), checkpoint_sha, "derived checkpoint SHA-256"
    )
    require_equal(
        output_record.get("bytes"), checkpoint.stat().st_size, "derived checkpoint bytes"
    )
    require_equal(
        output_record.get("target_depth"), EXPECTED["target_depth"], "target depth"
    )
    require_equal(states.get("regular"), "model", "regular checkpoint state")
    require_equal(states.get("swa"), "swa_model", "SWA checkpoint state")
    require_equal(
        config.get("only_field_changed"), "config.block_kind", "derived config mutation"
    )
    require_equal(deletion.get("index_range"), [24, 35], "deleted block index range")
    total = deletion.get("total")
    if not isinstance(total, Mapping):
        raise AuditError("derivation manifest is missing deletion.total")
    require_equal(total.get("count"), 240, "deleted tensor count")
    require_equal(total.get("bytes"), 169_979_904, "deleted tensor bytes")
    by_state = deletion.get("by_state")
    if not isinstance(by_state, Mapping):
        raise AuditError("derivation manifest is missing deletion.by_state")
    expected_per_block = {str(index): 10 for index in range(24, 36)}
    for state_name in ("model", "swa_model"):
        record = by_state.get(state_name)
        if not isinstance(record, Mapping):
            raise AuditError(f"derivation manifest is missing deletion state {state_name}")
        require_equal(record.get("keys"), 120, f"{state_name} deleted tensor count")
        require_equal(
            record.get("tensor_bytes"),
            84_989_952,
            f"{state_name} deleted tensor bytes",
        )
        require_equal(
            record.get("per_block_key_count"),
            expected_per_block,
            f"{state_name} per-block deletion count",
        )
    require_equal(validation.get("device"), "cpu", "strict-load validation device")
    require_equal(
        validation.get("regular_strict_load"), True, "regular strict-load validation"
    )
    require_equal(
        validation.get("swa_strict_load"), True, "SWA strict-load validation"
    )
    require_equal(
        manifest.get("atomic_no_clobber_publish"), True, "atomic publication gate"
    )
    require_equal(manifest.get("gpu_work_performed"), False, "derivation GPU flag")
    require_equal(
        manifest.get("native_export_performed"), False, "derivation native-export flag"
    )
    return {
        "source_checkpoint_sha256": input_record["sha256"],
        "source_depth": input_record["source_depth"],
        "target_depth": output_record["target_depth"],
        "states": dict(states),
        "deleted_index_range": deletion["index_range"],
        "deleted_tensor_count": total["count"],
        "deleted_tensor_bytes": total["bytes"],
        "only_config_field_changed": config["only_field_changed"],
        "strict_load_validated": True,
    }


def audit_checkpoint(checkpoint: Any, checkpoint_sha: str) -> dict[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise AuditError("checkpoint root must be a mapping")
    config = checkpoint.get("config")
    regular = checkpoint.get("model")
    swa = checkpoint.get("swa_model")
    if not isinstance(config, Mapping):
        raise AuditError("checkpoint config must be a mapping")
    if not isinstance(regular, Mapping) or not isinstance(swa, Mapping):
        raise AuditError("checkpoint must contain model and swa_model mappings")
    if "swa_model_0" in checkpoint:
        raise AuditError("checkpoint unexpectedly contains ambiguous swa_model_0")

    require_equal(config.get("version"), EXPECTED["version"], "checkpoint version")
    block_kind = config.get("block_kind")
    if not isinstance(block_kind, list):
        raise AuditError("checkpoint config.block_kind must be a list")
    require_equal(len(block_kind), EXPECTED["target_depth"], "checkpoint depth")
    require_equal(
        block_kind,
        [[f"rconv{index}", "transformerropesg"] for index in range(1, 25)],
        "checkpoint block prefix",
    )
    expected_config = {
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
    for key, expected in expected_config.items():
        require_equal(config.get(key), expected, f"checkpoint config.{key}")
    require_equal(config.get("use_qk_norm", False), False, "checkpoint QK norm")
    require_equal(config.get("full_int8_clip"), None, "checkpoint full INT8 clip")
    require_equal(config.get("swiglu_clip"), None, "checkpoint SwiGLU clip")

    return {
        "sha256": checkpoint_sha,
        "bytes": None,
        "version": config["version"],
        "combined_layers": len(block_kind),
        "swa_state": "swa_model",
        "swa_n_averaged": int(swa["n_averaged"]),
    }


def audit_native_structure(parsed: Mapping[str, Any]) -> dict[str, Any]:
    require_equal(parsed.get("name"), EXPECTED_MODEL_NAME, "native model name")
    require_equal(parsed.get("version"), EXPECTED["version"], "native version")
    require_equal(
        parsed.get("spatial_features"),
        EXPECTED["spatial_features"],
        "native spatial features",
    )
    require_equal(
        parsed.get("global_features"),
        EXPECTED["global_features"],
        "native global features",
    )
    trunk = parsed.get("trunk")
    policy = parsed.get("policy")
    value = parsed.get("value")
    if not all(isinstance(section, Mapping) for section in (trunk, policy, value)):
        raise AuditError("native parser result is missing trunk/policy/value mappings")
    require_equal(trunk.get("descriptor_count"), EXPECTED["descriptors"], "descriptor count")
    require_equal(trunk.get("channels"), EXPECTED["channels"], "trunk channels")
    blocks = trunk.get("blocks")
    if not isinstance(blocks, list):
        raise AuditError("native trunk.blocks must be a list")
    require_equal(len(blocks), EXPECTED["descriptors"], "parsed block count")
    expected_kinds = [
        kind
        for _ in range(EXPECTED["target_depth"])
        for kind in ("transformer_attention_block", "transformer_ffn_block")
    ]
    require_equal([block.get("kind") for block in blocks], expected_kinds, "descriptor order")

    for layer in range(EXPECTED["target_depth"]):
        attention, ffn = blocks[2 * layer : 2 * layer + 2]
        require_equal(
            (
                attention.get("q", {}).get("in"),
                attention.get("heads"),
                attention.get("kv_heads"),
                attention.get("q_head_dim"),
                attention.get("v_head_dim"),
            ),
            (384, 12, 12, 32, 32),
            f"attention layer {layer} geometry",
        )
        require_equal(attention.get("use_rope"), True, f"attention layer {layer} RoPE")
        require_equal(
            attention.get("learnable_rope"),
            True,
            f"attention layer {layer} learnable RoPE",
        )
        rope = attention.get("rope")
        if not isinstance(rope, Mapping):
            raise AuditError(f"attention layer {layer} is missing learned RoPE weights")
        require_equal(rope.get("shape"), [12, 16, 2], f"attention layer {layer} RoPE shape")
        require_equal(
            (ffn.get("channels"), ffn.get("ffn_channels"), ffn.get("swiglu")),
            (384, 1024, True),
            f"FFN layer {layer} geometry",
        )

    require_equal(
        parsed.get("weight_block_count"), EXPECTED["weight_blocks"], "weight block count"
    )
    require_equal(
        parsed.get("serialized_float_count"),
        EXPECTED["serialized_floats"],
        "serialized float count",
    )
    head_widths = {
        "p1": policy.get("p1", {}).get("out"),
        "g1": policy.get("g1", {}).get("out"),
        "v1": value.get("v1", {}).get("out"),
    }
    require_equal(head_widths, {"p1": 48, "g1": 48, "v1": 96}, "native head widths")
    kind_counts = Counter(block["kind"] for block in blocks)
    return {
        "name": parsed["name"],
        "version": parsed["version"],
        "input_features": {"spatial": 22, "global": 39},
        "descriptor_count": trunk["descriptor_count"],
        "descriptor_kind_counts": dict(sorted(kind_counts.items())),
        "strict_alternation": True,
        "combined_layers": EXPECTED["target_depth"],
        "channels": EXPECTED["channels"],
        "heads": EXPECTED["heads"],
        "kv_heads": EXPECTED["kv_heads"],
        "head_dim": EXPECTED["head_dim"],
        "ffn_channels": EXPECTED["ffn_channels"],
        "weight_block_count": parsed["weight_block_count"],
        "serialized_float_count": parsed["serialized_float_count"],
        "head_widths": head_widths,
        "eof_clean": True,
    }


def audit(
    model: Path,
    checkpoint: Path,
    derivation_manifest: Path,
    exporter: Path,
    native_parser: Path,
) -> dict[str, Any]:
    model = require_file(model, "native model")
    checkpoint = require_file(checkpoint, "derived checkpoint")
    derivation_manifest = require_file(derivation_manifest, "derivation manifest")
    exporter = require_file(exporter, "native exporter")
    native_parser = require_file(native_parser, "native parser")
    exporter_sha = require_sha256(exporter, TRUSTED_EXPORTER_SHA256, "native exporter")
    checkpoint_sha = sha256_file(checkpoint)

    try:
        manifest = json.loads(derivation_manifest.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AuditError(
            f"cannot read derivation manifest: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(manifest, Mapping):
        raise AuditError("derivation manifest root must be an object")
    derivation_report = audit_derivation_manifest(manifest, checkpoint, checkpoint_sha)

    try:
        import torch

        checkpoint_object = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise AuditError(f"cannot load checkpoint on CPU: {type(exc).__name__}: {exc}") from exc
    checkpoint_report = audit_checkpoint(checkpoint_object, checkpoint_sha)
    checkpoint_report["bytes"] = checkpoint.stat().st_size

    parse_native = load_trusted_parser(native_parser)
    parsed, raw_payload_sha = parse_model(parse_native, model)
    native_structure = audit_native_structure(parsed)
    native_record = {
        "path": str(model),
        "bytes": model.stat().st_size,
        "sha256": sha256_file(model),
        "raw_payload_sha256": raw_payload_sha,
        "compression": "gzip" if model.suffix.lower() == ".gz" else "none",
        "structure": native_structure,
    }
    if model.suffix.lower() == ".gz":
        require_equal(
            sha256_gzip_payload(model), raw_payload_sha, "gzip payload SHA-256"
        )

    return {
        "schema": 1,
        "status": "PASS",
        "operation": "b24c384-native-v102-audit",
        "derivation": derivation_report,
        "provenance": {
            "source_checkpoint_sha256": TRUSTED_SOURCE_CHECKPOINT_SHA256,
            "source_checkpoint_kind": "deterministic blank/untrained architecture fixture",
            "claim_boundary": (
                "This artifact validates model geometry and engine dispatch/performance only; "
                "it is not a trained playing-strength model."
            ),
        },
        "checkpoint": checkpoint_report,
        "native_model": native_record,
        "exporter": {"path": str(exporter), "sha256": exporter_sha},
        "native_parser": {
            "path": str(native_parser),
            "sha256": TRUSTED_NATIVE_PARSER_SHA256,
            "role": "independent strict v102 wire parser",
        },
        "gpu_work_performed": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the derived b24 C384 native-v102 model fail-closed."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--derivation-manifest", type=Path, required=True)
    parser.add_argument("--exporter", type=Path, required=True)
    parser.add_argument("--native-parser", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report_path = args.report.resolve()
    if report_path.exists():
        print(f"ERROR: AuditError: refusing to overwrite report: {report_path}")
        return 2
    if not report_path.parent.is_dir():
        print(f"ERROR: AuditError: report parent does not exist: {report_path.parent}")
        return 2
    try:
        report = audit(
            model=args.model,
            checkpoint=args.checkpoint,
            derivation_manifest=args.derivation_manifest,
            exporter=args.exporter,
            native_parser=args.native_parser,
        )
        payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{report_path.name}.",
            suffix=".tmp",
            dir=report_path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        try:
            temporary.replace(report_path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2
    print(json.dumps({
        "status": report["status"],
        "checkpoint_sha256": report["checkpoint"]["sha256"],
        "native_model_sha256": report["native_model"]["sha256"],
        "serialized_float_count": report["native_model"]["structure"]["serialized_float_count"],
        "descriptors": report["native_model"]["structure"]["descriptor_count"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

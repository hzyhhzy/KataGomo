#!/usr/bin/env python3
"""Derive a shallower KataGo checkpoint by retaining a strict block prefix.

This tool is intentionally fail-closed.  It accepts only checkpoints whose
regular and SWA state dictionaries have one contiguous ``blocks.<index>``
namespace matching ``config.block_kind``.  It then removes only state entries
belonging to the discarded suffix and truncates ``config.block_kind``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, MutableMapping, Sequence


SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
BLOCK_KEY_RE = re.compile(
    r"^(?P<prefix>(?:module\.)*)blocks\.(?P<index>0|[1-9][0-9]*)\.(?P<suffix>.+)$"
)
SWA_KEY_RE = re.compile(r"^swa_model(?:_[0-9]+)?$")


class DerivationError(RuntimeError):
    """A contract violation that must prevent checkpoint generation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_torch() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised only without torch
        raise DerivationError(
            "PyTorch is required for CPU checkpoint derivation and validation: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return torch


def _tensor_fingerprint(torch: Any, value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, torch.Tensor):
        raise DerivationError(
            f"state entry {location!r} is {type(value).__name__}, expected torch.Tensor"
        )
    if value.layout != torch.strided:
        raise DerivationError(
            f"state entry {location!r} uses unsupported tensor layout {value.layout}"
        )
    if value.is_quantized:
        raise DerivationError(
            f"state entry {location!r} is quantized; raw-byte identity is ambiguous"
        )
    cpu_value = value.detach().to(device="cpu").contiguous()
    try:
        # Flatten first because PyTorch does not permit a zero-dimensional
        # scalar to be reinterpreted as a smaller element type.
        raw = cpu_value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    except Exception as exc:
        raise DerivationError(
            f"cannot obtain canonical tensor bytes for {location!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return {
        "dtype": str(cpu_value.dtype),
        "shape": [int(dim) for dim in cpu_value.shape],
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _has_blocks_component(key: str) -> bool:
    return "blocks" in key.split(".")


def _normalized_model_key(key: str) -> str:
    while key.startswith("module."):
        key = key[7:]
    return key


def _validate_block_config(config: Any, target_depth: int) -> tuple[list[Any], int]:
    if not isinstance(config, dict):
        raise DerivationError("checkpoint config must be a dict")
    block_kind = config.get("block_kind")
    if not isinstance(block_kind, list) or not block_kind:
        raise DerivationError("checkpoint config.block_kind must be a non-empty list")
    seen_names: set[str] = set()
    for index, entry in enumerate(block_kind):
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            raise DerivationError(
                f"config.block_kind[{index}] must contain a block name and kind"
            )
        name, kind = entry[0], entry[1]
        if not isinstance(name, str) or not name:
            raise DerivationError(f"config.block_kind[{index}] has an invalid name")
        if not isinstance(kind, str) or not kind:
            raise DerivationError(f"config.block_kind[{index}] has an invalid kind")
        if name in seen_names:
            raise DerivationError(f"duplicate config.block_kind name: {name!r}")
        seen_names.add(name)
    source_depth = len(block_kind)
    if target_depth <= 0:
        raise DerivationError("--target-depth must be positive")
    if target_depth >= source_depth:
        raise DerivationError(
            f"--target-depth must be smaller than source depth {source_depth}, "
            f"got {target_depth}"
        )
    if config.get("has_intermediate_head", False):
        intermediate_depth = config.get("intermediate_head_blocks")
        if (
            not isinstance(intermediate_depth, int)
            or isinstance(intermediate_depth, bool)
            or intermediate_depth <= 0
        ):
            raise DerivationError(
                "config.has_intermediate_head requires a positive integer "
                "config.intermediate_head_blocks"
            )
        if intermediate_depth > target_depth:
            raise DerivationError(
                "config.intermediate_head_blocks lies in the discarded suffix; "
                "derivation is forbidden because only config.block_kind may change"
            )
    return block_kind, source_depth


def _select_states(checkpoint: Any) -> tuple[MutableMapping[str, Any], str, MutableMapping[str, Any]]:
    if not isinstance(checkpoint, dict):
        raise DerivationError("checkpoint root must be a dict")
    model_state = checkpoint.get("model")
    if not isinstance(model_state, MutableMapping):
        raise DerivationError("checkpoint model state must be a mutable mapping")

    swa_candidates = [key for key in ("swa_model_0", "swa_model") if key in checkpoint]
    extra_swa = [
        key
        for key in checkpoint
        if isinstance(key, str)
        and SWA_KEY_RE.fullmatch(key)
        and key not in ("swa_model_0", "swa_model")
    ]
    if extra_swa:
        raise DerivationError(
            "unsupported additional SWA state(s): " + ", ".join(sorted(extra_swa))
        )
    if len(swa_candidates) != 1:
        raise DerivationError(
            "checkpoint must contain exactly one of swa_model_0 or swa_model"
        )
    swa_key = swa_candidates[0]
    swa_state = checkpoint[swa_key]
    if not isinstance(swa_state, MutableMapping):
        raise DerivationError(f"checkpoint {swa_key} state must be a mutable mapping")
    return model_state, swa_key, swa_state


def _inspect_state(
    torch: Any,
    state_name: str,
    state: Mapping[str, Any],
    source_depth: int,
) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
    block_indices: dict[str, int] = {}
    fingerprints: dict[str, dict[str, Any]] = {}
    prefixes: set[str] = set()
    normalized: set[str] = set()
    for key, value in state.items():
        if not isinstance(key, str):
            raise DerivationError(
                f"{state_name} contains non-string state key {key!r}"
            )
        match = BLOCK_KEY_RE.fullmatch(key)
        if match is None and _has_blocks_component(key):
            raise DerivationError(
                f"{state_name} contains unsupported block-key pattern {key!r}; "
                "expected optional module. prefixes followed by blocks.<index>.<name>"
            )
        if match is not None:
            index = int(match.group("index"))
            if index >= source_depth:
                raise DerivationError(
                    f"{state_name} contains block index {index}, outside config depth "
                    f"{source_depth}"
                )
            block_indices[key] = index
            prefixes.add(match.group("prefix"))
        normalized_key = _normalized_model_key(key)
        if normalized_key in normalized:
            raise DerivationError(
                f"{state_name} has colliding keys after module-prefix normalization: "
                f"{key!r}"
            )
        normalized.add(normalized_key)
        fingerprints[key] = _tensor_fingerprint(
            torch, value, f"{state_name}.{key}"
        )

    if len(prefixes) != 1:
        raise DerivationError(
            f"{state_name} must use exactly one consistent block prefix, found "
            f"{sorted(prefixes)!r}"
        )
    present = {index for index in block_indices.values()}
    expected = set(range(source_depth))
    if present != expected:
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        raise DerivationError(
            f"{state_name} block indices are not the complete contiguous range "
            f"0..{source_depth - 1}; missing={missing}, extra={extra}"
        )
    return block_indices, fingerprints


def _validate_state_schema_pair(
    model_fingerprints: Mapping[str, Mapping[str, Any]],
    swa_fingerprints: Mapping[str, Mapping[str, Any]],
    swa_key: str,
) -> None:
    normalized_model = {
        _normalized_model_key(key): value for key, value in model_fingerprints.items()
    }
    normalized_swa = {
        _normalized_model_key(key): value
        for key, value in swa_fingerprints.items()
        if key != "n_averaged"
    }
    if "n_averaged" not in swa_fingerprints:
        raise DerivationError(f"{swa_key} is missing AveragedModel n_averaged state")
    model_keys = set(normalized_model)
    swa_keys = set(normalized_swa)
    if model_keys != swa_keys:
        raise DerivationError(
            "regular and SWA model schemas differ after module-prefix normalization; "
            f"regular_only={sorted(model_keys - swa_keys)!r}, "
            f"swa_only={sorted(swa_keys - model_keys)!r}"
        )
    incompatible = []
    for key in sorted(model_keys):
        regular = normalized_model[key]
        swa = normalized_swa[key]
        if regular["dtype"] != swa["dtype"] or regular["shape"] != swa["shape"]:
            incompatible.append(key)
    if incompatible:
        raise DerivationError(
            "regular and SWA tensor dtype/shape schemas differ for keys: "
            + ", ".join(incompatible)
        )


def _scan_for_unhandled_block_states(
    value: Any,
    ignored_mapping_ids: set[int],
    path: str = "checkpoint",
    seen: set[int] | None = None,
) -> None:
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen or value_id in ignored_mapping_ids:
        return
    seen.add(value_id)
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}[{key!r}]"
            if isinstance(key, str) and _has_blocks_component(key):
                raise DerivationError(
                    f"unhandled block-indexed state outside model/SWA at {child_path}"
                )
            _scan_for_unhandled_block_states(
                child, ignored_mapping_ids, child_path, seen
            )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _scan_for_unhandled_block_states(
                child, ignored_mapping_ids, f"{path}[{index}]", seen
            )


def _stable_mapping_key(key: Any) -> str:
    if key is None or isinstance(key, (bool, int, float, str)):
        try:
            encoded = json.dumps(
                key,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise DerivationError(
                f"checkpoint mapping key is not canonically representable: {key!r}"
            ) from exc
        return f"{type(key).__name__}:{encoded}"
    if isinstance(key, tuple):
        return "tuple:[" + ",".join(_stable_mapping_key(item) for item in key) + "]"
    raise DerivationError(
        "checkpoint mapping keys used outside model/SWA must be JSON scalars or "
        f"tuples, got {type(key).__name__}: {key!r}"
    )


def _collect_auxiliary_tensor_fingerprints(
    torch: Any,
    value: Any,
    ignored_mapping_ids: set[int],
    path: str = "checkpoint",
    seen: set[int] | None = None,
    output: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Fingerprint tensors outside the two explicitly transformed state maps."""
    if seen is None:
        seen = set()
    if output is None:
        output = {}
    if isinstance(value, torch.Tensor):
        if path in output:
            raise DerivationError(f"duplicate canonical checkpoint tensor path: {path}")
        output[path] = _tensor_fingerprint(torch, value, path)
        return output
    value_id = id(value)
    if value_id in seen or value_id in ignored_mapping_ids:
        return output
    if isinstance(value, Mapping):
        seen.add(value_id)
        for key, child in value.items():
            key_token = _stable_mapping_key(key)
            _collect_auxiliary_tensor_fingerprints(
                torch,
                child,
                ignored_mapping_ids,
                f"{path}[{key_token}]",
                seen,
                output,
            )
    elif isinstance(value, (list, tuple)):
        seen.add(value_id)
        for index, child in enumerate(value):
            _collect_auxiliary_tensor_fingerprints(
                torch,
                child,
                ignored_mapping_ids,
                f"{path}[{type(value).__name__}:{index}]",
                seen,
                output,
            )
    return output


def _auxiliary_records(
    fingerprints: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {"state": "checkpoint", "key": path, **fingerprint}
        for path, fingerprint in fingerprints.items()
    ]


def _digest_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(records, key=lambda record: (record["state"], record["key"]))
    return {
        "algorithm": "sha256(canonical-json(key,dtype,shape,bytes,tensor-sha256))",
        "count": len(ordered),
        "bytes": sum(int(record["bytes"]) for record in ordered),
        "sha256": canonical_json_sha256(ordered),
    }


def _records_for(
    fingerprints_by_state: Mapping[str, Mapping[str, Mapping[str, Any]]],
    block_indices_by_state: Mapping[str, Mapping[str, int]],
    predicate: Any,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for state_name, fingerprints in fingerprints_by_state.items():
        indices = block_indices_by_state[state_name]
        for key, fingerprint in fingerprints.items():
            block_index = indices.get(key)
            if predicate(block_index):
                records.append(
                    {
                        "state": state_name,
                        "key": key,
                        **fingerprint,
                    }
                )
    return records


def _resolve_model_arguments(
    config: Mapping[str, Any], model_class: Any, load_model_module: Any
) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    try:
        signature = inspect.signature(model_class)
    except (TypeError, ValueError) as exc:
        raise DerivationError(f"cannot inspect Model constructor: {exc}") from exc
    parameters = list(signature.parameters.values())
    if not parameters:
        raise DerivationError("Model constructor does not accept config")

    args: list[Any] = [config]
    kwargs: dict[str, Any] = {}
    resolution: dict[str, Any] = {"config": "checkpoint.config"}
    remaining = parameters[1:]
    for parameter in remaining:
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if parameter.name != "pos_len":
            if parameter.default is inspect.Parameter.empty:
                raise DerivationError(
                    "unsupported required Model constructor parameter "
                    f"{parameter.name!r}; --train-dir API is not safely inferable"
                )
            continue

        pos_len = None
        source = None
        for config_key in ("pos_len", "board_size", "board_len", "max_board_size"):
            candidate = config.get(config_key)
            if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
                pos_len = candidate
                source = f"checkpoint.config.{config_key}"
                break
        if pos_len is None:
            load_function = getattr(load_model_module, "load_model", None)
            if load_function is not None:
                try:
                    load_signature = inspect.signature(load_function)
                    load_parameter = load_signature.parameters.get("pos_len")
                except (TypeError, ValueError):
                    load_parameter = None
                if (
                    load_parameter is not None
                    and load_parameter.default is not inspect.Parameter.empty
                    and isinstance(load_parameter.default, int)
                    and not isinstance(load_parameter.default, bool)
                    and load_parameter.default > 0
                ):
                    pos_len = load_parameter.default
                    source = "train-dir load_model.load_model pos_len default"
        if pos_len is None and parameter.default is not inspect.Parameter.empty:
            if (
                isinstance(parameter.default, int)
                and not isinstance(parameter.default, bool)
                and parameter.default > 0
            ):
                pos_len = parameter.default
                source = "train-dir Model pos_len default"
        if pos_len is None:
            raise DerivationError(
                "Model requires pos_len, but it is absent from checkpoint config and "
                "the --train-dir API provides no positive default"
            )
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            args.append(pos_len)
        else:
            kwargs[parameter.name] = pos_len
        resolution["pos_len"] = {"value": pos_len, "source": source}
    return args, kwargs, resolution


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _strict_load_with_train_dir(
    torch: Any,
    checkpoint: Mapping[str, Any],
    train_dir: Path,
    swa_key: str,
) -> dict[str, Any]:
    train_dir = train_dir.resolve()
    if not train_dir.is_dir():
        raise DerivationError(f"--train-dir is not a directory: {train_dir}")
    model_file = train_dir / "model_pytorch.py"
    load_file = train_dir / "load_model.py"
    if not model_file.is_file() or not load_file.is_file():
        raise DerivationError(
            "--train-dir must provide importable model_pytorch.py and load_model.py "
            "with Model and AveragedModel"
        )

    tracked_names = ("model_pytorch", "modelconfigs", "load_model")
    previous_modules = {name: sys.modules.get(name) for name in tracked_names}
    for name in tracked_names:
        sys.modules.pop(name, None)
    old_path = list(sys.path)
    sys.path.insert(0, str(train_dir))
    imported_before = set(sys.modules)
    try:
        try:
            model_module = importlib.import_module("model_pytorch")
            load_module = importlib.import_module("load_model")
        except Exception as exc:
            raise DerivationError(
                "cannot import Model/AveragedModel from --train-dir; install its CPU "
                f"training dependencies first: {type(exc).__name__}: {exc}"
            ) from exc
        model_origin = Path(getattr(model_module, "__file__", "")).resolve()
        load_origin = Path(getattr(load_module, "__file__", "")).resolve()
        if not _is_within(model_origin, train_dir) or not _is_within(load_origin, train_dir):
            raise DerivationError(
                "Model/AveragedModel imports did not resolve inside --train-dir"
            )
        model_class = getattr(model_module, "Model", None)
        averaged_class = getattr(load_module, "AveragedModel", None)
        if model_class is None or averaged_class is None:
            raise DerivationError(
                "--train-dir load API must expose model_pytorch.Model and "
                "load_model.AveragedModel"
            )

        config = checkpoint["config"]
        args, kwargs, argument_resolution = _resolve_model_arguments(
            config, model_class, load_module
        )
        try:
            model = model_class(*args, **kwargs)
            initialize = getattr(model, "initialize", None)
            if callable(initialize):
                initialize()
        except Exception as exc:
            raise DerivationError(
                f"failed to instantiate train-dir Model: {type(exc).__name__}: {exc}"
            ) from exc

        regular_state: dict[str, Any] = {}
        for key, value in checkpoint["model"].items():
            normalized = _normalized_model_key(key)
            if normalized in regular_state:
                raise DerivationError(
                    f"regular state normalization collision for {normalized!r}"
                )
            regular_state[normalized] = value
        try:
            regular_result = model.load_state_dict(regular_state, strict=True)
        except Exception as exc:
            raise DerivationError(
                f"strict regular Model load failed: {type(exc).__name__}: {exc}"
            ) from exc
        if regular_result.missing_keys or regular_result.unexpected_keys:
            raise DerivationError(
                "strict regular Model load returned incompatibilities: "
                f"missing={regular_result.missing_keys}, "
                f"unexpected={regular_result.unexpected_keys}"
            )

        try:
            averaged_model = averaged_class(model, device="cpu")
            swa_result = averaged_model.load_state_dict(
                checkpoint[swa_key], strict=True
            )
        except Exception as exc:
            raise DerivationError(
                f"strict AveragedModel load failed: {type(exc).__name__}: {exc}"
            ) from exc
        if swa_result.missing_keys or swa_result.unexpected_keys:
            raise DerivationError(
                "strict AveragedModel load returned incompatibilities: "
                f"missing={swa_result.missing_keys}, unexpected={swa_result.unexpected_keys}"
            )
        return {
            "device": "cpu",
            "regular_strict_load": True,
            "swa_strict_load": True,
            "model_class": f"{model_class.__module__}.{model_class.__qualname__}",
            "averaged_model_class": (
                f"{averaged_class.__module__}.{averaged_class.__qualname__}"
            ),
            "model_module": str(model_origin),
            "load_module": str(load_origin),
            "constructor_arguments": argument_resolution,
        }
    finally:
        sys.path[:] = old_path
        for name, module in list(sys.modules.items()):
            if name in imported_before:
                continue
            origin = getattr(module, "__file__", None)
            if origin is not None:
                try:
                    if _is_within(Path(origin).resolve(), train_dir):
                        sys.modules.pop(name, None)
                except (OSError, ValueError):
                    pass
        for name in tracked_names:
            sys.modules.pop(name, None)
            previous = previous_modules[name]
            if previous is not None:
                sys.modules[name] = previous


def _assert_retained_identity(
    torch: Any,
    checkpoint: Mapping[str, Any],
    expected_fingerprints: Mapping[str, Mapping[str, Mapping[str, Any]]],
    expected_keys: Mapping[str, set[str]],
) -> None:
    for state_name, keys in expected_keys.items():
        state = checkpoint.get(state_name)
        if not isinstance(state, Mapping):
            raise DerivationError(f"serialized checkpoint lost {state_name} mapping")
        if set(state) != keys:
            raise DerivationError(
                f"serialized {state_name} keys differ from the exact expected set"
            )
        for key in sorted(keys):
            actual = _tensor_fingerprint(
                torch, state[key], f"serialized {state_name}.{key}"
            )
            expected = expected_fingerprints[state_name][key]
            if actual != expected:
                raise DerivationError(
                    f"retained tensor identity changed for {state_name}.{key}: "
                    f"expected={expected}, actual={actual}"
                )


def _write_temp_checkpoint(torch: Any, checkpoint: Any, output: Path) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
        delete=False,
    )
    temp_path = Path(handle.name)
    handle.close()
    try:
        torch.save(checkpoint, temp_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return temp_path


def _write_temp_manifest(manifest: Mapping[str, Any], manifest_path: Path) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{manifest_path.name}.",
        suffix=".tmp",
        dir=manifest_path.parent,
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return temp_path


def _publish_new_file(temp_path: Path, destination: Path) -> None:
    """Atomically publish without ever overwriting an existing destination."""
    try:
        os.link(temp_path, destination)
    except FileExistsError as exc:
        raise DerivationError(f"refusing to overwrite existing file: {destination}") from exc
    except OSError as exc:
        raise DerivationError(
            f"atomic no-clobber publish failed for {destination}: {exc}"
        ) from exc
    else:
        try:
            temp_path.unlink()
        except OSError as exc:
            destination.unlink(missing_ok=True)
            raise DerivationError(
                f"atomic publish cleanup failed for {destination}: {exc}"
            ) from exc


def derive_checkpoint(
    source: Path,
    output: Path,
    source_sha: str,
    target_depth: int,
    train_dir: Path,
) -> dict[str, Any]:
    torch = _require_torch()
    source = source.resolve()
    output = output.resolve()
    train_dir = train_dir.resolve()
    manifest_path = output.with_name(output.name + ".manifest.json")

    if not SHA256_RE.fullmatch(source_sha):
        raise DerivationError("--source-sha must be exactly 64 hexadecimal characters")
    expected_sha = source_sha.lower()
    if not source.is_file():
        raise DerivationError(f"--source is not a regular file: {source}")
    if not output.parent.is_dir():
        raise DerivationError(f"output parent directory does not exist: {output.parent}")
    if source == output or source == manifest_path:
        raise DerivationError("source, output, and manifest paths must be distinct")
    for path in (output, manifest_path):
        if path.exists():
            raise DerivationError(f"refusing to overwrite existing file: {path}")

    actual_sha = sha256_file(source)
    if actual_sha != expected_sha:
        raise DerivationError(
            f"source SHA-256 mismatch: expected {expected_sha}, actual {actual_sha}"
        )
    try:
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise DerivationError(
            f"CPU torch.load failed for source checkpoint: {type(exc).__name__}: {exc}"
        ) from exc

    block_kind, source_depth = _validate_block_config(
        checkpoint.get("config") if isinstance(checkpoint, dict) else None,
        target_depth,
    )
    try:
        source_config_sha = canonical_json_sha256(checkpoint["config"])
        source_config_without_blocks = copy.deepcopy(checkpoint["config"])
        del source_config_without_blocks["block_kind"]
        non_block_config_sha = canonical_json_sha256(source_config_without_blocks)
    except (TypeError, ValueError) as exc:
        raise DerivationError(
            f"checkpoint config is not canonical JSON data: {exc}"
        ) from exc
    model_state, swa_key, swa_state = _select_states(checkpoint)
    _scan_for_unhandled_block_states(
        checkpoint, {id(model_state), id(swa_state)}
    )
    auxiliary_fingerprints = _collect_auxiliary_tensor_fingerprints(
        torch, checkpoint, {id(model_state), id(swa_state)}
    )

    state_items = (("model", model_state), (swa_key, swa_state))
    block_indices_by_state: dict[str, dict[str, int]] = {}
    fingerprints_by_state: dict[str, dict[str, dict[str, Any]]] = {}
    for state_name, state in state_items:
        indices, fingerprints = _inspect_state(
            torch, state_name, state, source_depth
        )
        block_indices_by_state[state_name] = indices
        fingerprints_by_state[state_name] = fingerprints
    _validate_state_schema_pair(
        fingerprints_by_state["model"],
        fingerprints_by_state[swa_key],
        swa_key,
    )

    deleted_keys_by_state: dict[str, list[str]] = {}
    expected_retained_keys: dict[str, set[str]] = {}
    for state_name, state in state_items:
        deleted = sorted(
            key
            for key, index in block_indices_by_state[state_name].items()
            if target_depth <= index < source_depth
        )
        deleted_keys_by_state[state_name] = deleted
        expected_retained_keys[state_name] = set(state) - set(deleted)
        for key in deleted:
            del state[key]

    derived_config = copy.deepcopy(checkpoint["config"])
    derived_config["block_kind"] = copy.deepcopy(block_kind[:target_depth])
    checkpoint["config"] = derived_config
    try:
        target_config_sha = canonical_json_sha256(derived_config)
        target_config_without_blocks = copy.deepcopy(derived_config)
        del target_config_without_blocks["block_kind"]
        if canonical_json_sha256(target_config_without_blocks) != non_block_config_sha:
            raise DerivationError(
                "a config field other than config.block_kind changed unexpectedly"
            )
    except (TypeError, ValueError) as exc:
        raise DerivationError(f"derived config is not canonical JSON data: {exc}") from exc

    _assert_retained_identity(
        torch, checkpoint, fingerprints_by_state, expected_retained_keys
    )
    retained_records = _records_for(
        fingerprints_by_state,
        block_indices_by_state,
        lambda index: index is None or index < target_depth,
    )
    non_block_records = _records_for(
        fingerprints_by_state,
        block_indices_by_state,
        lambda index: index is None,
    )
    prefix_block_records = _records_for(
        fingerprints_by_state,
        block_indices_by_state,
        lambda index: index is not None and index < target_depth,
    )
    deleted_records = _records_for(
        fingerprints_by_state,
        block_indices_by_state,
        lambda index: index is not None and index >= target_depth,
    )
    auxiliary_records = _auxiliary_records(auxiliary_fingerprints)
    all_retained_records = retained_records + auxiliary_records

    temp_checkpoint: Path | None = None
    temp_manifest: Path | None = None
    published_output = False
    try:
        temp_checkpoint = _write_temp_checkpoint(torch, checkpoint, output)
        try:
            reloaded = torch.load(
                temp_checkpoint, map_location="cpu", weights_only=False
            )
        except Exception as exc:
            raise DerivationError(
                "CPU reload of staged checkpoint failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if reloaded.get("config", {}).get("block_kind") != block_kind[:target_depth]:
            raise DerivationError("serialized config.block_kind is not the exact target prefix")
        if canonical_json_sha256(reloaded["config"]) != target_config_sha:
            raise DerivationError(
                "serialized config differs beyond the exact requested block prefix"
            )
        _assert_retained_identity(
            torch, reloaded, fingerprints_by_state, expected_retained_keys
        )
        reloaded_model_state = reloaded.get("model")
        reloaded_swa_state = reloaded.get(swa_key)
        if not isinstance(reloaded_model_state, Mapping) or not isinstance(
            reloaded_swa_state, Mapping
        ):
            raise DerivationError("serialized checkpoint lost model/SWA state mappings")
        reloaded_auxiliary = _collect_auxiliary_tensor_fingerprints(
            torch,
            reloaded,
            {id(reloaded_model_state), id(reloaded_swa_state)},
        )
        if reloaded_auxiliary != auxiliary_fingerprints:
            source_paths = set(auxiliary_fingerprints)
            reloaded_paths = set(reloaded_auxiliary)
            changed = sorted(
                path
                for path in source_paths & reloaded_paths
                if auxiliary_fingerprints[path] != reloaded_auxiliary[path]
            )
            raise DerivationError(
                "auxiliary checkpoint tensor identity changed after serialization; "
                f"missing={sorted(source_paths - reloaded_paths)!r}, "
                f"extra={sorted(reloaded_paths - source_paths)!r}, changed={changed!r}"
            )
        strict_validation = _strict_load_with_train_dir(
            torch, reloaded, train_dir, swa_key
        )
        output_sha = sha256_file(temp_checkpoint)
        output_bytes = temp_checkpoint.stat().st_size

        deletion_by_state = {}
        for state_name, keys in deleted_keys_by_state.items():
            per_block: dict[str, int] = {}
            state_bytes = 0
            for key in keys:
                index = block_indices_by_state[state_name][key]
                per_block[str(index)] = per_block.get(str(index), 0) + 1
                state_bytes += int(fingerprints_by_state[state_name][key]["bytes"])
            deletion_by_state[state_name] = {
                "keys": len(keys),
                "tensor_bytes": state_bytes,
                "per_block_key_count": per_block,
            }

        manifest = {
            "schema": 1,
            "operation": "strict-checkpoint-depth-prefix-derivation",
            "status": "cpu-derived-and-strict-load-validated",
            "input": {
                "path": str(source),
                "bytes": source.stat().st_size,
                "sha256": actual_sha,
                "source_depth": source_depth,
            },
            "output": {
                "path": str(output),
                "bytes": output_bytes,
                "sha256": output_sha,
                "target_depth": target_depth,
            },
            "states": {"regular": "model", "swa": swa_key},
            "config": {
                "source_config_sha256": source_config_sha,
                "target_config_sha256": target_config_sha,
                "unchanged_non_block_kind_config_sha256": non_block_config_sha,
                "source_block_kind_sha256": canonical_json_sha256(block_kind),
                "target_block_kind_sha256": canonical_json_sha256(
                    block_kind[:target_depth]
                ),
                "only_field_changed": "config.block_kind",
            },
            "deletion": {
                "index_range": [target_depth, source_depth - 1],
                "total": _digest_records(deleted_records),
                "by_state": deletion_by_state,
            },
            "retained_tensor_digest": _digest_records(all_retained_records),
            "retained_model_state_tensor_digest": _digest_records(retained_records),
            "retained_auxiliary_tensor_digest": _digest_records(auxiliary_records),
            "retained_non_block_tensor_digest": _digest_records(non_block_records),
            "retained_prefix_block_tensor_digest": _digest_records(
                prefix_block_records
            ),
            "identity_contract": (
                "Every retained model/SWA and mapping/list-reachable auxiliary "
                "checkpoint tensor was reloaded on CPU and matched its source key, "
                "dtype, shape, byte count, and byte SHA-256."
            ),
            "validation": strict_validation,
            "atomic_no_clobber_publish": True,
            "gpu_work_performed": False,
            "native_export_performed": False,
        }
        temp_manifest = _write_temp_manifest(manifest, manifest_path)
        _publish_new_file(temp_checkpoint, output)
        temp_checkpoint = None
        published_output = True
        try:
            _publish_new_file(temp_manifest, manifest_path)
            temp_manifest = None
        except Exception:
            output.unlink(missing_ok=True)
            published_output = False
            raise
        if sha256_file(output) != output_sha:
            manifest_path.unlink(missing_ok=True)
            output.unlink(missing_ok=True)
            published_output = False
            raise DerivationError("published output SHA-256 changed unexpectedly")
        return {**manifest, "manifest_path": str(manifest_path)}
    finally:
        if temp_checkpoint is not None:
            temp_checkpoint.unlink(missing_ok=True)
        if temp_manifest is not None:
            temp_manifest.unlink(missing_ok=True)
        if published_output and not manifest_path.exists():
            output.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly derive a shallower KataGo checkpoint prefix on CPU."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--target-depth", type=int, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = derive_checkpoint(
            source=args.source,
            output=args.output,
            source_sha=args.source_sha,
            target_depth=args.target_depth,
            train_dir=args.train_dir,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

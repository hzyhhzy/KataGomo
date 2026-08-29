#!/usr/bin/env python3
"""Export a v11 Ataxx Transformer checkpoint as a native CPU-PTQ v206 model.

The v206 wire format is deliberately small and backend-owned. Transformer
projection matrices use canonical output-major symmetric S8 storage with one
FP32 scale per output channel. The spatial stem, global projection,
normalization parameters, and the policy/value heads remain FP32. The C++
backend repacks the canonical S8 matrices for Ice Lake AVX-512 VNNI at load
time, so the file itself is independent of a particular microkernel layout.

Only the two production profiles compiled by the Ataxx 7x7 CPU backend are
accepted. The input checkpoint is read-only; output and its provenance report
are written atomically beside the requested destination.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import struct
import tempfile
from typing import Dict, Iterable, Tuple

import numpy as np
import torch


MAGIC = b"KATAGOCPUPTQ206\0"
WIRE_REVISION = 1
MODEL_VERSION = 206
SOURCE_MODEL_VERSION = 11
BOARD_LEN = 7
DTYPE_FP32 = 1
DTYPE_S8_PER_OUTPUT = 2


PROJECTION_ROLES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "ffn_linear1",
    "ffn_linear_gate",
    "ffn_linear2",
)


PROFILES = {
    (11, 96, 3, 256): "b11c96h3-f256",
    (16, 128, 4, 384): "b16c128h4-f384",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fp32(value: torch.Tensor | np.ndarray | Iterable[float]) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value, dtype="<f4"))


def source_weight_sha256(value: torch.Tensor) -> str:
    return sha256_bytes(fp32(value).tobytes(order="C"))


def load_swa_state(checkpoint: Path) -> Tuple[dict, Dict[str, torch.Tensor]]:
    document = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if "config" not in document:
        raise ValueError("v206 export requires a checkpoint-embedded model config")
    if "swa_model_0" not in document:
        raise ValueError("v206 export requires swa_model_0")
    state: Dict[str, torch.Tensor] = {}
    for name, value in document["swa_model_0"].items():
        if name == "n_averaged":
            continue
        while name.startswith("module."):
            name = name[7:]
        if name in state:
            raise ValueError(f"duplicate SWA tensor {name!r}")
        state[name] = value.detach().cpu()
    return dict(document["config"]), state


def require_tensor(
    state: Dict[str, torch.Tensor], name: str, shape: Tuple[int, ...]
) -> torch.Tensor:
    if name not in state:
        raise ValueError(f"checkpoint is missing {name}")
    value = state[name]
    if tuple(value.shape) != shape:
        raise ValueError(
            f"{name} has shape {tuple(value.shape)}, expected {shape}"
        )
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains a non-finite value")
    return value


def validate_profile(config: dict) -> Tuple[str, int, int, int, int, int]:
    version = int(config.get("version", -1))
    blocks = config.get("block_kind")
    channels = int(config.get("trunk_num_channels", -1))
    heads = int(config.get("transformer_heads", -1))
    kv_heads = int(config.get("transformer_kv_heads", -1))
    ffn = int(config.get("transformer_ffn_channels", -1))
    value_hidden = int(config.get("v2_size", -1))
    if version != SOURCE_MODEL_VERSION:
        raise ValueError(f"v206 requires source model v11, got v{version}")
    if not isinstance(blocks, list) or any(
        not isinstance(item, list)
        or len(item) != 2
        or item[1] != "transformerropesg"
        for item in blocks
    ):
        raise ValueError("v206 requires a pure transformerropesg trunk")
    profile_key = (len(blocks), channels, heads, ffn)
    if profile_key not in PROFILES:
        raise ValueError(f"unsupported v206 profile {profile_key}")
    if kv_heads != heads or channels // heads != 32:
        raise ValueError("v206 requires ordinary MHA with head_dim=32")
    if value_hidden != 64:
        raise ValueError("v206 requires value hidden size 64")
    if config.get("use_qk_norm", False):
        raise ValueError("Ataxx v206 does not accept QK norm")
    if config.get("swiglu_clip", None) not in (None, 0, 0.0):
        raise ValueError("Ataxx v206 does not accept SwiGLU clipping")
    if config.get("learnable_rope", False):
        raise ValueError("Ataxx v206 requires fixed 2D RoPE")
    if config.get("norm_kind") != "bnorm" or config.get("activation") != "silu":
        raise ValueError("v206 requires bnorm trunk output and SiLU heads")
    if int(config.get("p1_num_channels", -1)) != 32:
        raise ValueError("v206 requires 32 policy channels")
    if int(config.get("g1_num_channels", -1)) != 32:
        raise ValueError("v206 requires 32 policy gpool channels")
    if int(config.get("v1_num_channels", -1)) != 32:
        raise ValueError("v206 requires 32 value channels")
    return PROFILES[profile_key], len(blocks), channels, heads, ffn, value_hidden


def load_manifest(
    path: Path,
    state: Dict[str, torch.Tensor],
    blocks: int,
) -> Tuple[dict, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    expected = {
        f"blocks.{block}.{role}"
        for block in range(blocks)
        for role in PROJECTION_ROLES
    }
    matrices: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["format"].item()) != "cpuptq-gptq-v1":
            raise ValueError("unsupported CPU-PTQ manifest format")
        if int(archive["qmax"].item()) != 127:
            raise ValueError("Ice Lake v206 requires full-range S8 projection weights")
        if str(archive["activation_quantizer"].item()) != "row-sym":
            raise ValueError("v206 requires row-symmetric activation quantization")
        if float(archive["activation_scale_factor"].item()) != 1.0:
            raise ValueError("v206 requires activation_scale_factor=1")
        names = [str(item) for item in archive["names"].tolist()]
        if len(names) != len(set(names)) or set(names) != expected:
            raise ValueError(
                "projection set mismatch: "
                f"missing={sorted(expected - set(names))}, "
                f"extra={sorted(set(names) - expected)}"
            )
        for index, name in enumerate(names):
            block_text, role = name[len("blocks."):].split(".", 1)
            checkpoint_name = f"blocks.{int(block_text)}.{role}.weight"
            weight = state.get(checkpoint_name)
            if weight is None or weight.ndim != 2:
                raise ValueError(f"missing projection {checkpoint_name}")
            codes = np.asarray(archive[f"codes_{index}"])
            scales = np.asarray(archive[f"scales_{index}"], dtype="<f4")
            recorded = str(archive[f"source_sha256_{index}"].item())
            if codes.dtype != np.int8 or codes.shape != tuple(weight.shape):
                raise ValueError(f"{name}: invalid S8 code shape/dtype")
            if scales.shape != (weight.shape[0],):
                raise ValueError(f"{name}: invalid scale shape")
            if not np.isfinite(scales).all() or not np.all(scales > 0.0):
                raise ValueError(f"{name}: invalid scales")
            if int(np.abs(codes.astype(np.int16)).max()) > 127:
                raise ValueError(f"{name}: S8 code out of range")
            if np.any(codes == np.int8(-128)):
                raise ValueError(f"{name}: symmetric S8 contains -128")
            if recorded != source_weight_sha256(weight):
                raise ValueError(f"{name}: manifest was calibrated from another model")
            matrices[name] = (
                np.ascontiguousarray(codes),
                np.ascontiguousarray(scales),
            )
        metadata = {
            "recipe": str(archive["recipe"].item()),
            "qmax": 127,
            "activationQuantizer": "row-sym",
        }
    return metadata, matrices


def folded_batch_norm(
    state: Dict[str, torch.Tensor], prefix: str, channels: int
) -> Tuple[np.ndarray, np.ndarray]:
    gamma = require_tensor(state, prefix + ".gamma", (1, channels, 1, 1))
    beta = require_tensor(state, prefix + ".beta", (1, channels, 1, 1))
    mean = require_tensor(state, prefix + ".running_mean", (channels,))
    std = require_tensor(state, prefix + ".running_std", (channels,))
    if not torch.all(std > 0.0):
        raise ValueError(f"{prefix}.running_std must be positive")
    multiplier = (gamma.reshape(-1) + 1.0) / std
    bias = beta.reshape(-1) - mean * multiplier
    return fp32(multiplier), fp32(bias)


def collect_tensors(
    state: Dict[str, torch.Tensor],
    quantized: Dict[str, Tuple[np.ndarray, np.ndarray]],
    blocks: int,
    channels: int,
    ffn: int,
) -> list[Tuple[str, int, np.ndarray, np.ndarray | None]]:
    tensors: list[Tuple[str, int, np.ndarray, np.ndarray | None]] = []

    def add_fp(name: str, value: torch.Tensor | np.ndarray | Iterable[float]) -> None:
        array = fp32(value)
        tensors.append((name, DTYPE_FP32, array, None))

    def add_s8(name: str, outputs: int, inputs: int) -> None:
        codes, scales = quantized[name]
        if codes.shape != (outputs, inputs):
            raise ValueError(
                f"{name} has shape {codes.shape}, expected {(outputs, inputs)}"
            )
        tensors.append((name, DTYPE_S8_PER_OUTPUT, codes, scales))

    add_fp(
        "stem.weight",
        require_tensor(state, "conv_spatial.weight", (channels, 22, 3, 3)),
    )
    add_fp(
        "global.weight",
        require_tensor(state, "linear_global.weight", (channels, 19)),
    )
    add_fp("rope.theta", [100.0])
    for block in range(blocks):
        prefix = f"blocks.{block}."
        add_fp(
            prefix + "norm1",
            require_tensor(state, prefix + "norm1.weight", (channels,)),
        )
        add_fp(
            prefix + "norm2",
            require_tensor(state, prefix + "norm2.weight", (channels,)),
        )
        add_s8(prefix + "q_proj", channels, channels)
        add_s8(prefix + "k_proj", channels, channels)
        add_s8(prefix + "v_proj", channels, channels)
        add_s8(prefix + "out_proj", channels, channels)
        add_s8(prefix + "ffn_linear1", ffn, channels)
        add_s8(prefix + "ffn_linear_gate", ffn, channels)
        add_s8(prefix + "ffn_linear2", channels, ffn)

    trunk_mul, trunk_bias = folded_batch_norm(
        state, "norm_trunkfinal", channels
    )
    add_fp("trunk.mul", trunk_mul)
    add_fp("trunk.bias", trunk_bias)

    add_fp(
        "policy.conv1p",
        require_tensor(
            state, "policy_head.conv1p.weight", (32, channels, 1, 1)
        ).reshape(32, channels),
    )
    add_fp(
        "policy.conv1g",
        require_tensor(
            state, "policy_head.conv1g.weight", (32, channels, 1, 1)
        ).reshape(32, channels),
    )
    add_fp(
        "policy.biasg",
        require_tensor(state, "policy_head.biasg.beta", (1, 32, 1, 1)).reshape(32),
    )
    add_fp(
        "policy.linear_g",
        require_tensor(state, "policy_head.linear_g.weight", (32, 96)),
    )
    add_fp(
        "policy.linear_pass",
        require_tensor(state, "policy_head.linear_pass.weight", (4, 96))[0],
    )
    add_fp(
        "policy.bias2",
        require_tensor(state, "policy_head.bias2.beta", (1, 32, 1, 1)).reshape(32),
    )
    add_fp(
        "policy.conv2p",
        require_tensor(state, "policy_head.conv2p.weight", (4, 32, 1, 1))[
            0
        ].reshape(32),
    )

    add_fp(
        "value.conv1",
        require_tensor(
            state, "value_head.conv1.weight", (32, channels, 1, 1)
        ).reshape(32, channels),
    )
    add_fp(
        "value.bias1",
        require_tensor(state, "value_head.bias1.beta", (1, 32, 1, 1)).reshape(32),
    )
    add_fp(
        "value.linear2",
        require_tensor(state, "value_head.linear2.weight", (64, 96)),
    )
    add_fp(
        "value.bias2",
        require_tensor(state, "value_head.linear2.bias", (64,)),
    )
    add_fp(
        "value.linear_value",
        require_tensor(state, "value_head.linear_valuehead.weight", (3, 64)),
    )
    add_fp(
        "value.bias_value",
        require_tensor(state, "value_head.linear_valuehead.bias", (3,)),
    )
    add_fp(
        "value.linear_misc",
        require_tensor(state, "value_head.linear_miscvaluehead.weight", (10, 64))[:4],
    )
    add_fp(
        "value.bias_misc",
        require_tensor(state, "value_head.linear_miscvaluehead.bias", (10,))[:4],
    )
    add_fp(
        "value.linear_more",
        require_tensor(state, "value_head.linear_moremiscvaluehead.weight", (8, 64))[:2],
    )
    add_fp(
        "value.bias_more",
        require_tensor(state, "value_head.linear_moremiscvaluehead.bias", (8,))[:2],
    )
    if len({name for name, _, _, _ in tensors}) != len(tensors):
        raise AssertionError("duplicate v206 tensor name")
    return tensors


def encode_model(
    model_name: str,
    blocks: int,
    channels: int,
    heads: int,
    ffn: int,
    value_hidden: int,
    tensors: list[Tuple[str, int, np.ndarray, np.ndarray | None]],
) -> bytes:
    name_bytes = model_name.encode("utf-8")
    if not name_bytes or len(name_bytes) > 4096:
        raise ValueError("model name must encode to 1..4096 UTF-8 bytes")
    output = bytearray(MAGIC)
    output += struct.pack(
        "<9I",
        WIRE_REVISION,
        MODEL_VERSION,
        SOURCE_MODEL_VERSION,
        BOARD_LEN,
        blocks,
        channels,
        heads,
        ffn,
        value_hidden,
    )
    output += struct.pack("<I", len(name_bytes)) + name_bytes
    output += struct.pack("<I", len(tensors))
    for name, kind, values, scales in tensors:
        encoded_name = name.encode("ascii")
        if not encoded_name or len(encoded_name) > 65535:
            raise ValueError(f"invalid tensor name {name!r}")
        if values.ndim <= 0 or values.ndim > 4:
            raise ValueError(f"{name}: unsupported rank {values.ndim}")
        output += struct.pack("<HBB", len(encoded_name), kind, values.ndim)
        output += encoded_name
        output += struct.pack("<" + "I" * values.ndim, *values.shape)
        if kind == DTYPE_FP32:
            if scales is not None or values.dtype != np.dtype("<f4"):
                raise AssertionError(f"{name}: malformed FP32 tensor")
            output += values.tobytes(order="C")
        elif kind == DTYPE_S8_PER_OUTPUT:
            if (
                values.dtype != np.int8
                or values.ndim != 2
                or scales is None
                or scales.shape != (values.shape[0],)
            ):
                raise AssertionError(f"{name}: malformed S8 tensor")
            output += struct.pack("<I", 127)
            output += np.ascontiguousarray(scales, dtype="<f4").tobytes(order="C")
            output += values.tobytes(order="C")
        else:
            raise AssertionError(f"{name}: unknown tensor kind")
    return bytes(output)


def write_atomic_gzip(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=destination.name + ".",
            suffix=".partial",
            dir=destination.parent,
            delete=False,
        ) as raw:
            temporary = Path(raw.name)
            # Projection codes are nearly incompressible. Level 1 avoids
            # spending minutes searching for negligible extra compression.
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=1
            ) as zipped:
                zipped.write(payload)
        temporary.replace(destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # The exporter performs many tiny validation reductions. On large-core
    # developer machines, spawning a wide Torch worker team for each one is
    # dramatically slower and needlessly interferes with other work.
    torch.set_num_threads(1)

    checkpoint = args.checkpoint.resolve()
    manifest = args.manifest.resolve()
    destination = args.output.resolve()
    if not checkpoint.is_file() or not manifest.is_file():
        raise ValueError("checkpoint and manifest must be existing files")
    if not destination.name.endswith(".bin.gz"):
        raise ValueError("v206 output must end in .bin.gz")
    if destination.exists() and not args.force:
        raise ValueError(f"refusing to overwrite {destination}; pass --force")

    print("Loading SWA checkpoint...", flush=True)
    config, state = load_swa_state(checkpoint)
    profile, blocks, channels, heads, ffn, value_hidden = validate_profile(config)
    print(f"Validating {profile} S8 manifest...", flush=True)
    manifest_metadata, quantized = load_manifest(manifest, state, blocks)
    print("Collecting and encoding v206 tensors...", flush=True)
    tensors = collect_tensors(state, quantized, blocks, channels, ffn)
    payload = encode_model(
        args.model_name,
        blocks,
        channels,
        heads,
        ffn,
        value_hidden,
        tensors,
    )
    print("Writing deterministic .bin.gz...", flush=True)
    write_atomic_gzip(destination, payload)
    with gzip.open(destination, "rb") as handle:
        round_trip = handle.read()
    if round_trip != payload:
        destination.unlink(missing_ok=True)
        raise RuntimeError("deterministic gzip round-trip verification failed")

    report = {
        "kind": "katago-cpu-ptq-v206-export",
        "wireRevision": WIRE_REVISION,
        "sourceModelVersion": SOURCE_MODEL_VERSION,
        "modelVersion": MODEL_VERSION,
        "profile": profile,
        "boardSize": BOARD_LEN,
        "modelName": args.model_name,
        "checkpoint": str(checkpoint),
        "checkpointSha256": sha256_file(checkpoint),
        "manifest": str(manifest),
        "manifestSha256": sha256_file(manifest),
        "quantization": manifest_metadata,
        "gzipCompressionLevel": 1,
        "tensorCount": len(tensors),
        "uncompressedBytes": len(payload),
        "uncompressedSha256": sha256_bytes(payload),
        "output": str(destination),
        "outputBytes": destination.stat().st_size,
        "outputSha256": sha256_file(destination),
    }
    report_path = destination.with_suffix(destination.suffix + ".json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
